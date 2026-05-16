#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2024 Zoom Video Communications Inc. All rights reserved.
# Some portions of this code are based on unlicensed code from https://github.com/princeton-nlp/SimPO.

from curses import raw
from imp import load_compiled
import logging
import random
import sys

import torch
import datetime

import transformers
from transformers import AutoModelForCausalLM, set_seed

from config_utils import (
    DataArguments,
    CustomizedDPOConfig,
    H4ArgumentParser,
    ModelArguments,
)

from model_utils import (
    get_checkpoint,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)

from data_utils import maybe_insert_system_message, is_openai_format, get_datasets_new, get_prompt, get_response
from peft import PeftConfig, PeftModel
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, List, Union, Tuple
from trl import DPOTrainer
import torch.nn as nn
import torch.nn.functional as F

import os
import shutil

from self_judging import compute_attention_weights

from tw_dpo_trainer import TokenWeightedDPODataCollator, TokenWeightedDPOTrainer

import glob

from huggingface_hub import delete_folder, delete_file, whoami, list_repo_tree

logger = logging.getLogger(__name__)

# class WPOTrainer(DPOTrainer):
#     @staticmethod
#     def get_batch_logps(
#         logits: torch.FloatTensor,
#         labels: torch.LongTensor,
#         average_log_prob: bool = False,
#         label_pad_token_id: int = -100,
#         is_encoder_decoder: bool = False,
#     ) -> torch.FloatTensor:
#         if logits.shape[:-1] != labels.shape:
#             raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

#         if not is_encoder_decoder:
#             labels = labels[:, 1:].clone()
#             logits = logits[:, :-1, :]
#         loss_mask = labels != label_pad_token_id
#         labels[labels == label_pad_token_id] = 0

#         logprobs = F.log_softmax(logits, dim=-1)
#         per_token_logps = torch.gather(logprobs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
#         if not average_log_prob:
#             logps = (per_token_logps * loss_mask).sum(-1)
#         else:
#             logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)

#         probs = F.softmax(logits, dim=-1)
#         exp_probs = torch.log((probs ** 2).sum(-1))
#         per_token_logps = per_token_logps - exp_probs

#         avg_log_probs = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
#         chosen_probs = avg_log_probs[:avg_log_probs.size(0) // 2]
#         rejected_probs = avg_log_probs[avg_log_probs.size(0) // 2:]
#         weight = torch.clamp(torch.exp(chosen_probs + rejected_probs), max=1)
#         return logps, weight.detach()

#     def dpo_loss(
#         self,
#         policy_chosen_logps: torch.FloatTensor,
#         policy_rejected_logps: torch.FloatTensor,
#         reference_chosen_logps: torch.FloatTensor,
#         reference_rejected_logps: torch.FloatTensor,
#         policy_weights: torch.FloatTensor,
#     ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
#         pi_logratios = policy_chosen_logps - policy_rejected_logps
#         if self.reference_free:
#             ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
#         else:
#             ref_logratios = reference_chosen_logps - reference_rejected_logps

#         pi_logratios = pi_logratios.to(self.accelerator.device)
#         ref_logratios = ref_logratios.to(self.accelerator.device)
#         logits = pi_logratios - ref_logratios

#         losses = (
#             -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) * policy_weights
#             - F.logsigmoid(-self.beta * logits) * self.label_smoothing * policy_weights
#         )

#         chosen_rewards = (
#             self.beta
#             * (
#                 policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
#             ).detach()
#         )
#         rejected_rewards = (
#             self.beta
#             * (
#                 policy_rejected_logps.to(self.accelerator.device)
#                 - reference_rejected_logps.to(self.accelerator.device)
#             ).detach()
#         )

#         return losses, chosen_rewards, rejected_rewards

#     def concatenated_forward(
#         self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
#     ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
#         """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

#         We do this to avoid doing two forward passes, because it's faster for FSDP.
#         """
#         concatenated_batch = self.concatenated_inputs(
#             batch,
#             is_encoder_decoder=self.is_encoder_decoder,
#             label_pad_token_id=self.label_pad_token_id,
#             padding_value=self.padding_value,
#             device=self.accelerator.device,
#         )
#         len_chosen = batch["chosen_labels"].shape[0]

#         model_kwargs = (
#             {
#                 "labels": concatenated_batch["concatenated_labels"],
#                 "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
#             }
#             if self.is_encoder_decoder
#             else {}
#         )
#         all_logits = model(
#             concatenated_batch["concatenated_input_ids"],
#             attention_mask=concatenated_batch["concatenated_attention_mask"],
#             use_cache=False,
#             **model_kwargs,
#         ).logits

#         all_logps, weight = self.get_batch_logps(
#             all_logits,
#             concatenated_batch["concatenated_labels"],
#             average_log_prob=self.loss_type == "ipo",
#             is_encoder_decoder=self.is_encoder_decoder,
#             label_pad_token_id=self.label_pad_token_id,
#         )

#         chosen_logps = all_logps[:len_chosen]
#         rejected_logps = all_logps[len_chosen:]

#         chosen_logits = all_logits[:len_chosen]
#         rejected_logits = all_logits[len_chosen:]

#         return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, weight)

#     def get_batch_loss_metrics(
#         self,
#         model,
#         batch: Dict[str, Union[List, torch.LongTensor]],
#         train_eval: Literal["train", "eval"] = "train",
#     ):
#         """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
#         metrics = {}

#         (
#             policy_chosen_logps,
#             policy_rejected_logps,
#             policy_chosen_logits,
#             policy_rejected_logits,
#             policy_weights,
#         ) = self.concatenated_forward(model, batch)

#         # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
#         if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
#             reference_chosen_logps = batch["reference_chosen_logps"]
#             reference_rejected_logps = batch["reference_rejected_logps"]
#         else:
#             with torch.no_grad():
#                 if self.ref_model is None:
#                     with self.null_ref_context():
#                         (
#                             reference_chosen_logps,
#                             reference_rejected_logps,
#                             _,
#                             _,
#                             _,
#                         ) = self.concatenated_forward(self.model, batch)
#                 else:
#                     (
#                         reference_chosen_logps,
#                         reference_rejected_logps,
#                         _,
#                         _,
#                         _,
#                     ) = self.concatenated_forward(self.ref_model, batch)

#         losses, chosen_rewards, rejected_rewards = self.dpo_loss(
#             policy_chosen_logps,
#             policy_rejected_logps,
#             reference_chosen_logps,
#             reference_rejected_logps,
#             policy_weights,
#         )
#         reward_accuracies = (chosen_rewards > rejected_rewards).float()

#         prefix = "eval_" if train_eval == "eval" else ""
#         metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
#         metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
#         metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
#         metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
#         metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
#         metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
#         metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
#         metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
#         return losses.mean(), metrics


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    auto_insert_empty_system_msg: bool = True,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = get_prompt(prompt_messages=prompt_messages, tokenizer=tokenizer)
            example["text_chosen"] = get_response(prompt=example["text_prompt"], prompt_messages=prompt_messages, response_messages=chosen_messages, tokenizer=tokenizer, is_tw=True)
            example["text_rejected"] = get_response(prompt=example["text_prompt"], prompt_messages=prompt_messages, response_messages=rejected_messages, tokenizer=tokenizer, is_tw=True)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example

def get_weights(
    example,
    attn_source: str = "",
    fix_attn_sink: bool = False,
):
    if fix_attn_sink:
        for key in example:
            if key.startswith("chosen_weights_") or key.startswith("rejected_weights_"):
                if len(example[key]) >= 5:
                    # print("example {}: {}".format(key, example[key]))
                    total_weights = sum(example[key])
                    total_weights_except_first = sum(example[key][1:])

                    example[key][0] = total_weights/len(example[key])
                    for t_idx in range(1, len(example[key])):
                        example[key][t_idx] = example[key][t_idx] * ((total_weights - example[key][0])/total_weights_except_first)

    try:
        # attn_source = int(attn_source)
        example["chosen_weights"] = example["chosen_weights_{}".format(attn_source)]
        example["rejected_weights"] = example["rejected_weights_{}".format(attn_source)]
    except Exception:
        chosen_weights = [0] * len(example["chosen_tokens"])
        rejected_weights = [0] * len(example["rejected_tokens"])
        if attn_source == "mean":
            for key in example:
                if key.startswith("chosen_weights_"):
                    assert len(example[key]) == len(example["chosen_tokens"])
                    chosen_weights = [w1+w2 for w1, w2 in zip(chosen_weights, example[key])]
                elif key.startswith("rejected_weights_"):
                    assert len(example[key]) == len(example["rejected_tokens"])
                    rejected_weights = [w1+w2 for w1, w2 in zip(rejected_weights, example[key])]
        elif attn_source.startswith("mean_"):
            keys = attn_source[len("mean_"):].split("_")
            for key in keys:
                assert len(example["chosen_weights_{}".format(key)]) == len(example["chosen_tokens"])
                assert len(example["rejected_weights_{}".format(key)]) == len(example["rejected_tokens"])
                chosen_weights = [w1+w2 for w1, w2 in zip(chosen_weights, example["chosen_weights_{}".format(key)])]
                rejected_weights = [w1+w2 for w1, w2 in zip(rejected_weights, example["rejected_weights_{}".format(key)])]
        else:
            raise Exception("attn_source {} is not supported".format(attn_source))
        example["chosen_weights"] = chosen_weights
        example["rejected_weights"] = rejected_weights
    return example

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, CustomizedDPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets_new(data_args)
    
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    if not training_args.self_judge:
        raw_datasets = raw_datasets.map(
            get_weights,
            fn_kwargs={
                "attn_source": training_args.attn_source,
                "fix_attn_sink": training_args.fix_attn_sink,
            },
            num_proc=data_args.preprocessing_num_workers,
            desc="Computing the weights using attn_source {}".format(training_args.attn_source)
        )

        column_names = list(raw_datasets["train"].features)
        column_names = [c for c in column_names if c.startswith("chosen_weights_") or c.startswith("rejected_weights_")]
        raw_datasets = raw_datasets.remove_columns(column_names)

        raw_datasets = raw_datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "dpo",
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },
            num_proc=data_args.preprocessing_num_workers,
            desc="Formatting comparisons with prompt template",
        )

        column_names = list(raw_datasets["train"].features)
        column_names = [c for c in column_names if c not in ["text_prompt", "text_chosen", "text_rejected", "chosen_tokens", "rejected_tokens", "chosen_weights", "rejected_weights"]]
        raw_datasets = raw_datasets.remove_columns(column_names)

        # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
        for split in ["train", "test"]:
            raw_datasets[split] = raw_datasets[split].rename_columns(
                {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
            )

        # Log a few random samples from the training set:
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
            logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
            logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        # use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        # use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        attn_implementation="eager",
    )

    model = model_args.model_name_or_path


    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            # use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            # use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
            attn_implementation="eager",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    ref_model = model if training_args.ref_model == "" else training_args.ref_model
    print("ref_model is set to {}".format(ref_model))
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    training_args.model_init_kwargs = model_kwargs
    training_args.ref_model_init_kwargs = ref_model_kwargs

    if training_args.self_judge:
        from datasets import DatasetDict, Dataset, concatenate_datasets, load_dataset

        if isinstance(model, str):
            model_kwargs["device_map"] = "auto"
            model_kwargs["offload_folder"] = "offload/"
            model = AutoModelForCausalLM.from_pretrained(
                model,
                **model_kwargs,
            )

        save_interval = 2000
        assert len(data_args.dataset_splits) == 1
        split = "train"

        save_path = training_args.self_judge_output_path
        completed = Dataset.from_list([])
        if os.path.exists(save_path):
            completed = load_dataset(
                "json",
                data_files={
                    "train": save_path,
                }
            )["train"]
        start_idx = len(completed)

        for chunk_start_idx in range(start_idx, len(raw_datasets[split]), save_interval):
            chunk_end_idx = min(len(raw_datasets[split]), chunk_start_idx + save_interval)
            to_complete = raw_datasets[split].select(range(chunk_start_idx, chunk_end_idx))

            print_attn_score_stat = chunk_end_idx >= len(raw_datasets[split])
            attention_rollout = training_args.attn_source == "attention_rollout"

            print("print_attn_score_stat: {}; attention_rollout: {}".format(print_attn_score_stat, attention_rollout))

            curr_completed = compute_attention_weights(to_complete, model, tokenizer, print_attn_score_stat=print_attn_score_stat, attention_rollout=attention_rollout)
            completed = concatenate_datasets([completed, curr_completed])
            if os.path.exists(save_path):
                os.remove(save_path)
            completed.to_json(save_path, lines=True)

        # os.remove(data_args.dataset_splits[0])
        exit()

    #########################
    # Instantiate DPO trainer
    #########################

    if training_args.token_weighted:
        if training_args.padding_value is not None:
            padding_value = training_args.padding_value
        else:
            if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                padding_value = tokenizer.pad_token_id
            elif hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer.pad_token_id is not None:
                padding_value = tokenizer.tokenizer.pad_token_id
            else:
                raise ValueError(
                    "`padding_value` is not specified in `DPOConfig`, and `pad_token_id` is missing in the "
                    "`processing_class`. Please either set the `padding_value` argument in `DPOConfig`, or set "
                    "`tokenizer.pad_token` (e.g., `tokenizer.pad_token = tokenizer.eos_token`) before instantiating "
                    "the trainer."
                )
        
        collator = TokenWeightedDPODataCollator(pad_token_id=padding_value)

        trainer = TokenWeightedDPOTrainer(
            model=model,
            ref_model=ref_model,
            # model_init_kwargs=model_kwargs,
            args=training_args,
            # beta=training_args.beta,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            processing_class=tokenizer,
            # max_length=training_args.max_length,
            # max_prompt_length=training_args.max_prompt_length,
            peft_config=get_peft_config(model_args),
            data_collator=collator,
        )

    else:
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            # model_init_kwargs=model_kwargs,
            args=training_args,
            # beta=training_args.beta,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            processing_class=tokenizer,
            # max_length=training_args.max_length,
            # max_prompt_length=training_args.max_prompt_length,
            peft_config=get_peft_config(model_args),
        )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    
    last_ckpt_path = os.path.join(training_args.output_dir, "checkpoint-{}".format(trainer.state.global_step))
    if not os.path.exists(last_ckpt_path):
        if trainer.accelerator.is_main_process:
            os.mkdir(last_ckpt_path)
        trainer.save_model(last_ckpt_path)
        logger.info(f"Model saved to {last_ckpt_path}")

        # Save everything else on main process
        # kwargs = {
        #     "finetuned_from": model_args.model_name_or_path,
        #     "dataset": list(data_args.dataset_mixer.keys()),
        #     "dataset_tags": list(data_args.dataset_mixer.keys()),
        #     "tags": ["alignment-handbook"],
        # }
        if trainer.accelerator.is_main_process:
            # trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(last_ckpt_path)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # if training_args.push_to_hub is True:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")

    if trainer.accelerator.is_main_process:
        global_step_pattern = os.path.join(training_args.output_dir, "checkpoint*", "global_step*")
        global_step_paths = glob.glob(global_step_pattern)
        for gsp in global_step_paths:
            print("Removing global step {}".format(gsp))
            shutil.rmtree(gsp)

        root_model_safetensors_pattern = os.path.join(training_args.output_dir, "*.safetensors")
        root_model_safetensors_paths = glob.glob(root_model_safetensors_pattern)
        for rms in root_model_safetensors_paths:
            print("Removing model safetensors {}".format(rms))
            os.remove(rms)

        if training_args.push_to_hub:
            user_name = whoami()["name"]
            if training_args.output_dir.endswith("/"):
                training_args.output_dir = training_args.output_dir[:-1]
            dir_name = os.path.basename(training_args.output_dir)
            
            repo_id = "{}/{}".format(user_name, dir_name)
            print("Deleting model safetensors in the root folder in repo: {}".format(repo_id))

            repo_tree = list_repo_tree(repo_id, recursive=True)

            paths = [x.path for x in repo_tree]

            for path in paths:
                if path.startswith("model") and path.endswith(".safetensors"):
                    delete_file(
                        repo_id=repo_id,
                        path_in_repo=path,
                    )

if __name__ == "__main__":
    main()