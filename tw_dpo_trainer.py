import os
import shutil

import torch
from torch import autocast
from contextlib import contextmanager, nullcontext
import torch.nn.functional as F
import torch.nn as nn


from accelerate import PartialState
from datasets import Dataset, IterableDataset, load_from_disk

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin,
)

from typing import Dict, List, Union, Tuple, Any, Optional, Literal

from trl import DPOTrainer, DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.utils import DPODataCollatorWithPadding, pad, pad_to_length, flush_left, flush_right, selective_log_softmax
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl.data_utils import maybe_extract_prompt, maybe_apply_chat_template

import glob
import shutil

from huggingface_hub import delete_folder, list_repo_tree

import torch.distributed as dist

import edit_distance

class TokenWeightedDPODataCollator(DataCollatorForPreference):
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        chosen_weights = [torch.tensor(f["chosen_weights"]) for f in examples]
        rejected_weights = [torch.tensor(f["rejected_weights"]) for f in examples]

        batch = super().torch_call(examples)

        batch["chosen_weights"] = pad(chosen_weights, padding_value=0.0)
        batch["rejected_weights"] = pad(rejected_weights, padding_value=0.0)

        assert batch["chosen_weights"].shape == batch["chosen_input_ids"].shape
        assert batch["rejected_weights"].shape == batch["rejected_input_ids"].shape

        return batch

class TokenWeightedDPOTrainer(DPOTrainer):
    @staticmethod
    def tokenize_row(
        features: dict[str, str],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        row = DPOTrainer.tokenize_row(
            features=features,
            processing_class=processing_class,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            add_special_tokens=add_special_tokens,
        )

        for key in ["chosen", "rejected"]:
            input_ids = row[f"{key}_input_ids"]

            weights = features[f"{key}_weights"]
            original_tokens = features["{}_tokens".format(key)]

            if len(weights) != len(original_tokens):
                print("The weights and tokens lengths do not match. This example will be ignored.")
                row[f"{key}_weights"] = [0.0]*len(input_ids)
                return row

            # [Important] Multiplying by response length means the total weights add up to response length, which is the same as DPO. Otherwise, the weights will add up to 1.
            # print("No length normalization")
            weights = [w*len(original_tokens) for w in weights]

            # Special case: When the response is " ", input_ids will not have any token for the response content, but original tokens may have 1 token for the content.
            if len(original_tokens) == 1 and input_ids[0] != original_tokens[0]:
                row[f"{key}_input_ids"] = original_tokens + row[f"{key}_input_ids"]
                input_ids = row[f"{key}_input_ids"]
            
            # input_ids can be longer than original_tokens due to the presence of chat template chat tokens such as <end_of_turn>
            # input_ids can be shorter than origianl_tokens due to truncation
            min_len = min(len(input_ids), len(original_tokens))
            if input_ids[:min_len] != original_tokens[:min_len]:
                new_weights = [0.0] * len(input_ids)
                # print("num input_ids={}; num original_tokens={}".format(len(input_ids), len(original_tokens)))
                sm = edit_distance.SequenceMatcher(a=input_ids, b=original_tokens)
                matches = sm.get_matching_blocks()
                num_matches = 0
                matched_idxes = []
                for a_idx, b_idx, length in matches:
                    if length != 1:
                        print("Matched block has length of {}. This example will be ignored.".format(length))
                        row[f"{key}_weights"] = [0.0]*len(input_ids)
                        return row
                    num_matches = num_matches + 1
                    new_weights[a_idx] = weights[b_idx]
                    matched_idxes.append((a_idx, b_idx))

                print("input_ids do not match original tokens. num_matched_tokens={}; num input_ids={}; num original_tokens={}; prec={}; rec={}".format(num_matches, len(input_ids), len(original_tokens), float(num_matches)/len(input_ids), float(num_matches)/len(original_tokens)))
                if float(num_matches)/len(original_tokens) < 0.5:
                    print("input_ids:\n{}\n\n".format(input_ids))
                    print("original_tokens:\n{}\n\n".format(original_tokens))
                    print("matched_tokens:\n{}\n\n".format([(a_idx, b_idx) for a_idx, b_idx, _ in matches]))
                row[f"{key}_weights"] = new_weights
            else:
                # Align weights: 0.0 for prompt tokens, actual weights for response tokens
                if len(input_ids) > len(original_tokens):
                    aligned_weights = [0.0] * len(input_ids)
                    aligned_weights[:len(original_tokens)] = weights
                else:
                    aligned_weights = weights[:len(input_ids)]

                row[f"{key}_weights"] = aligned_weights
                print("Prefix Matched for response with length {}".format(len(input_ids)))
        
        return row


    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: DPOConfig,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        cache_path = f"./cache/{args.run_name}_{dataset_name}_tokenized"

        with PartialState().main_process_first():

            if PartialState().is_main_process and not os.path.exists(cache_path):
                # shutil.rmtree(cache_path, ignore_errors=True)

                map_kwargs = {}
                if isinstance(dataset, Dataset):
                    map_kwargs["num_proc"] = args.dataset_num_proc
                    map_kwargs["writer_batch_size"] = 10

                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
                dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

                # Apply the chat template if needed
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
                dataset = dataset.map(
                    maybe_apply_chat_template,
                    fn_kwargs={"tokenizer": processing_class, "tools": args.tools},
                    **map_kwargs,
                )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"
                dataset = dataset.map(
                    self.tokenize_row if not self.is_vision_model else self.process_row,
                    remove_columns=["chosen", "rejected"],
                    fn_kwargs={
                        "processing_class": processing_class,
                        "max_prompt_length": args.max_prompt_length,
                        "max_completion_length": args.max_completion_length,
                        "add_special_tokens": False,
                    },
                    **map_kwargs,
                )

                dataset.save_to_disk(cache_path)

        dist.barrier()

        dataset = load_from_disk(cache_path)

        dist.barrier()

        return dataset

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        self._siganture_columns.extend(
            [
                "chosen_weights",
                "rejected_weights",
            ]
        )

    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch, is_ref_model=True)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)
        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"], ref_model_output["weighted_chosen_logps"], ref_model_output["weighted_rejected_logps"]

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        output = DPOTrainer.concatenated_inputs(batch, padding_value)

        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        with torch.no_grad():
            output["completion_weights"] = torch.cat(
                (
                    pad_to_length(batch["chosen_weights"], max_completion_length, pad_value=0.0),
                    pad_to_length(batch["rejected_weights"], max_completion_length, pad_value=0.0)
                )
            )

        return output

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], is_ref_model: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Runs the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.

        Args:
            model:
                Model to run the forward pass on.
            batch:
                Batch of input data.
            is_ref_model:
                Whether this method is being called for the reference model. If `True`, length desensitization is not
                applied.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        completion_weights = concatenated_batch["completion_weights"]

        assert completion_weights.shape == completion_input_ids.shape

        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
            weights = completion_weights
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)

            weights = torch.cat((torch.zeros_like(prompt_input_ids).to(completion_weights.dtype), completion_weights), dim=1)


            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush and truncate
            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    # Flush left to reduce the memory usage
                    # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                    #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                    attention_mask, input_ids, weights, loss_mask = flush_left(attention_mask, input_ids, weights, loss_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    weights = weights[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    # Flush right before truncating left, then flush left
                    # [[0, 0, x, x, x, x],  ->  [[0, 0, x, x],
                    #  [0, x, x, x, 0, 0]]       [0, x, x, x]]
                    attention_mask, input_ids, weights, loss_mask = flush_right(attention_mask, input_ids, weights, loss_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    weights = weights[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    attention_mask, input_ids, weights, loss_mask = flush_left(attention_mask, input_ids, weights, loss_mask)
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            else:
                # Flush left to reduce the memory usage
                # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                attention_mask, input_ids, weights, loss_mask = flush_left(attention_mask, input_ids, weights, loss_mask)

            if self.use_logits_to_keep:
                # Compute logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 for the first label
                model_kwargs["logits_to_keep"] = logits_to_keep

            model_kwargs["output_hidden_states"] = True

            if self.padding_free:
                # Flatten the input_ids, position_ids, and loss_mask
                # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
                #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                weights = weights[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            # Offset the logits by one to align with the labels
            labels = torch.roll(input_ids, shifts=-1, dims=1)

            weights = torch.roll(weights, shifts=-1, dims=1)

            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -logits_to_keep:]

                weights = weights[:, -logits_to_keep:]

                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)


        weighted_per_token_logps = per_token_logps * weights


        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)


        weighted_per_token_logps[~loss_mask] = 0
        weighted_per_token_logps = torch.roll(weighted_per_token_logps, shifts=1, dims=1)


        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

            weighted_per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            weighted_per_token_logps_[attention_mask.bool()] = weighted_per_token_logps
            weighted_per_token_logps = weighted_per_token_logps_


        all_logps = per_token_logps[:, 1:].sum(-1)
        weighted_all_logps = weighted_per_token_logps[:, 1:].sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            # Only use the chosen logits for the RPO loss or SFT loss
            chosen_logits = logits[:num_examples, :-1] if not self.is_encoder_decoder else logits[:num_examples]
            chosen_labels = labels[:num_examples, :-1] if not self.is_encoder_decoder else labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if "ipo" in self.loss_type:
            all_logps = all_logps / loss_mask.sum(-1)
            weighted_all_logps = weighted_all_logps / loss_mask.sum(-1)

        if self.args.ld_alpha is not None and not is_ref_model:
            return NotImplementedError

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        output["weighted_chosen_logps"] = weighted_all_logps[:num_examples]
        output["weighted_rejected_logps"] = weighted_all_logps[num_examples:]

        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        if self.args.use_liger_loss:
            model_output = self._compute_loss_liger(model, batch)
            losses = model_output["loss"]
            chosen_rewards = model_output["chosen_rewards"]
            rejected_rewards = model_output["rejected_rewards"]
            weighted_chosen_rewards = model_output["weighted_chosen_rewards"]
            weighted_rejected_rewards = model_output["weighted_rejected_rewards"]
        else:
            model_output = self.concatenated_forward(model, batch)

            ref_chosen_logps, ref_rejected_logps, weighted_ref_chosen_logps, weighted_ref_rejected_logps = self.compute_ref_log_probs(batch)

            # Initialize combined losses
            losses = 0

            chosen_rewards = 0
            rejected_rewards = 0

            weighted_chosen_rewards = 0
            weighted_rejected_rewards = 0

            device = self.accelerator.device

            chosen_rewards = self.beta * (model_output["chosen_logps"].to(device) - ref_chosen_logps.to(device)).detach()
            rejected_rewards = self.beta * (model_output["rejected_logps"].to(device) - ref_rejected_logps.to(device)).detach()

            # Compute losses for each loss type
            for idx, loss_type in enumerate(self.loss_type):
                # Compute individual loss using standard DPO loss function
                _losses, _weighted_chosen_rewards, _weighted_rejected_rewards = self.dpo_loss(
                    model_output["weighted_chosen_logps"],
                    model_output["weighted_rejected_logps"],
                    weighted_ref_chosen_logps,
                    weighted_ref_rejected_logps,
                    loss_type,
                    model_output,
                )

                # Add weighted contributions
                weight = self.loss_weights[idx] if self.loss_weights else 1.0
                losses = losses + _losses * weight
                weighted_chosen_rewards = weighted_chosen_rewards + _weighted_chosen_rewards * weight
                weighted_rejected_rewards = weighted_rejected_rewards + _weighted_rejected_rewards * weight

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        weighted_reward_accuracies = (weighted_chosen_rewards > weighted_rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""

        metrics[f"{prefix}rewards/weighted_chosen"] = self.accelerator.gather_for_metrics(weighted_chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/weighted_rejected"] = self.accelerator.gather_for_metrics(weighted_rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/weighted_accuracies"] = self.accelerator.gather_for_metrics(weighted_reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/weighted_margins"] = (
            self.accelerator.gather_for_metrics(weighted_chosen_rewards - weighted_rejected_rewards).mean().item()
        )


        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        
        metrics[f"{prefix}logps/weighted_chosen"] = (
            self.accelerator.gather_for_metrics(model_output["weighted_chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/weighted_rejected"] = (
            self.accelerator.gather_for_metrics(model_output["weighted_rejected_logps"]).detach().mean().item()
        )
        
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        return losses.mean(), metrics
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if any([keyword in self.processing_class.name_or_path for keyword in ["gemma-3", "gemma3"]]):
            preprocessor_config = """{
  "do_convert_rgb": null,
  "do_normalize": true,
  "do_pan_and_scan": null,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_processor_type": "Gemma3ImageProcessor",
  "image_seq_length": 256,
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "pan_and_scan_max_num_crops": null,
  "pan_and_scan_min_crop_size": null,
  "pan_and_scan_min_ratio_to_activate": null,
  "processor_class": "Gemma3Processor",
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 896,
    "width": 896
  }
}
"""
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            with open(os.path.join(output_dir, "preprocessor_config.json"), "w") as conf_json:
                conf_json.write(preprocessor_config)

        super().save_model(output_dir=output_dir, _internal_call=_internal_call)


    def _save_checkpoint(self, model, trial):
        if self.accelerator.is_main_process:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)

            last_checkpoint_dir = os.path.join(run_dir, "last-checkpoint")
            if os.path.exists(last_checkpoint_dir):
                print("Removing last_checkpoint_dir found at {}".format(last_checkpoint_dir))
                shutil.rmtree(last_checkpoint_dir)

            global_step_pattern = os.path.join(run_dir, "{}*".format(PREFIX_CHECKPOINT_DIR), "global_step*")
            global_step_paths = glob.glob(global_step_pattern)
            for gsp in global_step_paths:
                if checkpoint_folder not in gsp:
                    print("Removing global step {}".format(gsp))
                    shutil.rmtree(gsp)

                    if self.args.push_to_hub:
                        files = list_repo_tree(repo_id=self.hub_model_id, token=self.args.hub_token, revision=self.args.hub_revision)

                        paths = [f.path for f in files]
                        relative_gsp_path = os.path.relpath(gsp, run_dir)
                        print("Removing global step {} from huggingface".format(relative_gsp_path))
                        if relative_gsp_path in paths:
                            delete_folder(
                                repo_id=self.hub_model_id,
                                path_in_repo=relative_gsp_path,
                                token=self.args.hub_token,
                                revision=self.args.hub_revision
                            )

        self.accelerator.wait_for_everyone()
        
        super()._save_checkpoint(model, trial)
