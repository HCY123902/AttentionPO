from xml.dom.minidom import Identified
from zipapp import get_interpreter
import torch
from tqdm import tqdm
import numpy as np
from typing import Union, Literal, List

from data_utils import get_prompt, get_response

def avg_weights(w1, w2):
    return [(wt1+wt2)/2 for wt1, wt2 in zip(w1, w2)]

def answer_abbrev(answer):
    if len(answer) <= 40:
        return answer
    return answer[:20] + " ... " + answer[-20:]

def find_start_end(input_ids, tokenizer):
    if tokenizer.name_or_path in ["google/gemma-3-4b-it"]:
        answer_a_start = tokenizer.encode("<|The Start of Answer A|>\n", add_special_tokens=False)
        answer_a_end = tokenizer.encode("\n<|The End of Answer A|>", add_special_tokens=False)
        answer_b_start = tokenizer.encode("<|The Start of Answer B|>\n",  add_special_tokens=False)
        answer_b_end = tokenizer.encode("\n<|The End of Answer B|>",  add_special_tokens=False)
        end_offset = 0
    elif tokenizer.name_or_path in ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "princeton-nlp/Llama-3-Base-8B-SFT"]:
        answer_a_start = tokenizer.encode("<|The Start of Answer A|>\n", add_special_tokens=False)
        answer_a_end = tokenizer.encode("|The End of Answer A|", add_special_tokens=False)
        answer_b_start = tokenizer.encode("<|The Start of Answer B|>\n",  add_special_tokens=False)
        answer_b_end = tokenizer.encode("|The End of Answer B|",  add_special_tokens=False)
        end_offset = 3 # t - 2: at least 2 tokens preceding | for "\n" and "<". Safe to remove more tokens than necessary, but not safe to remove fewer tokens than necessary
    elif tokenizer.name_or_path in ["mistralai/Mistral-7B-v0.1", "alignment-handbook/zephyr-7b-sft-full", "mistralai/Mistral-7B-Instruct-v0.2"]:
        answer_a_start = tokenizer.encode("<|The Start of Answer A|>\n", add_special_tokens=False)[1:]
        answer_a_end = tokenizer.encode("\n<|The End of Answer A|>", add_special_tokens=False)[1:]
        answer_b_start = tokenizer.encode("<|The Start of Answer B|>\n",  add_special_tokens=False)[1:]
        answer_b_end = tokenizer.encode("\n<|The End of Answer B|>",  add_special_tokens=False)[1:]
        end_offset = 2


    res = [-1, -1, -1, -1]

    for t in range(len(input_ids)):
        for i, pattern in [(0, answer_a_start), (2, answer_b_start)]:
            if input_ids[t:t+len(pattern)] == pattern:
                res[i] = t + len(pattern)
        for i, pattern in [(1, answer_a_end), (3, answer_b_end)]:
            if input_ids[t:t+len(pattern)] == pattern:
                res[i] = max(t - end_offset, res[i-1] + 1) 
    assert sorted(res) == res and -1 not in res, "res: {}\n\nprompt: {}".format(res, tokenizer.decode(input_ids, skip_special_tokens=False))
    return res

def truncate_text(text, tokenizer, max_num_tokens, side="left"):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if side == "left":
        tokens = tokens[:max_num_tokens]
    else:
        tokens = tokens[-max_num_tokens:]
    return tokenizer.decode(tokens, skip_special_tokens=False), len(tokens)

def get_input_ids(judge_prompt, question, answer_a, answer_b, tokenizer):
    prompt = judge_prompt.replace("{question}", question).replace("{answer_a}", answer_a).replace("{answer_b}", answer_b)
    prompt_processed = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, return_dict=True)

    input_ids = prompt_processed["input_ids"]
    return input_ids, prompt

def compute_stat(list_of_weights, print_prefix):
    means = []
    stds = []
    max_ws = []
    min_ws = []
    lens = []
    for weights in list_of_weights:
        lens.append(len(weights))
        means.append(np.mean(weights))
        stds.append(np.std(weights))
        max_ws.append(np.max(weights))
        min_ws.append(np.min(weights))
    print(
        "{}:\nmean: {:.4f}\nstd: {:.4f}\nmax: {:.4f}\nmin: {:.4f}\nlen: {:.4f}\n\n".format(
            print_prefix,
            sum(means)/len(means),
            sum(stds)/len(stds),
            sum(max_ws)/len(max_ws),
            sum(min_ws)/len(min_ws),
            sum(lens)/len(lens),
        )
    )


def check_token_matches(question, answer_a, answer_b, tokens_a, tokens_b, tokenizer):
    prompt_msgs = [{"role": "user", "content": question}]
    prompt = get_prompt(prompt_msgs, tokenizer)

    for ans, original_tokens in [(answer_a, tokens_a), (answer_b, tokens_b)]:
        response_msgs = [{"role": "assistant", "content": ans}]
        response = get_response(prompt=prompt, prompt_messages=prompt_msgs, response_messages=response_msgs, tokenizer=tokenizer)
        input_ids = tokenizer(response, add_special_tokens=False)["input_idx"]

        if len(original_tokens) == 1 and input_ids[0] != original_tokens[0]:
            input_ids = original_tokens + input_ids

        min_len = min(len(input_ids), len(original_tokens))

        assert input_ids[:min_len] == original_tokens[:min_len], "input_ids is {}, but originally it is {}".format(input_ids, original_tokens)

def compute_attention_weights(dataset, model, tokenizer, print_attn_score_stat=False, attention_rollout=False):
    weights_map = {
        "cr_unnormalized_chosen_weights": {},
        "cr_unnormalized_rejected_weights": {},
        "rc_unnormalized_chosen_weights": {},
        "rc_unnormalized_rejected_weights": {},
        "cr_chosen_weights": {},
        "cr_rejected_weights": {},
        "rc_chosen_weights": {},
        "rc_rejected_weights": {},
        "chosen_weights": {},
        "rejected_weights": {},
    }

    chosen_tokens_list = []
    rejected_tokens_list = []

    judge_prompt = """Please act as an impartial judge and evaluate the quality of the two responses to the user prompt displayed below. You will be given answer A and answer B. Your job is to evaluate which answer is better.
    
Consider if the answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the answers when needed. Finally, identify any missing important information in the answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output exactly one of the following choices as your final verdict:

1. Answer A is better: A
2. Answer B is better: B

<|The Start of User Prompt|>
{question}
<|The End of User Prompt|>

<|The Start of Answer A|>
{answer_a}
<|The End of Answer A|>

<|The Start of Answer B|>
{answer_b}
<|The End of Answer B|>

Now, output one of the following identifiers and nothing else (no quotes, no spaces, no new lines, no thinking or reasoning, ...): A or B.

Your choice:"""

    empty_count = 0
    cr_last_token_list = []
    rc_last_token_list = []
    last_token_stat = {}

    model.eval()
    with torch.no_grad():
        for row in tqdm(dataset, desc="Extracting Attention Weights"):
            chosen = row["chosen"][-1]["content"].strip()
            rejected = row["rejected"][-1]["content"].strip()

            if chosen == "":
                empty_count = empty_count + 1
                chosen = " "
            if rejected == "":
                empty_count = empty_count + 1
                rejected = " "

            for r in ["cr", "rc"]:
                question = row["prompt"]

                if r == "cr":
                    answer_a = chosen
                    answer_b = rejected
                else:
                    answer_a = rejected
                    answer_b = chosen
                input_ids, _ = get_input_ids(judge_prompt, question, answer_a, answer_b, tokenizer)

                if len(input_ids) >= 2048:
                    answer_a, num_tokens_a = truncate_text(answer_a, tokenizer, 768, side="left")
                    answer_b, num_tokens_b = truncate_text(answer_b, tokenizer, 768, side="left")
                    question, _ = truncate_text(question, tokenizer, 2048 - 300 - num_tokens_a - num_tokens_b, side="right")
                    input_ids, _ = get_input_ids(judge_prompt, question, answer_a, answer_b, tokenizer)
                
                input_tensor = torch.tensor([input_ids]).to(model.device)

                outputs = model(input_tensor, output_attentions=True)

                last_token_id = torch.argmax(outputs.logits[0, -1, :], dim=0).item()
                last_token = tokenizer.decode([last_token_id], skip_special_tokens=False)

                last_token_count = last_token_stat.setdefault(last_token, 0)
                last_token_stat[last_token] = last_token_count + 1

                # Each layer: (batch, num_heads, seq_len, seq_len) -> (num_heads, seq_len, seq_len)
                attentions = [layer_attn[0, :, :, :].mean(dim=0) for layer_attn in outputs.attentions]
                seq_len = outputs.logits.shape[1]
                del outputs

                idxes = find_start_end(input_ids, tokenizer)

                start_a, end_a, start_b, end_b = idxes[0], idxes[1], idxes[2], idxes[3]

                tokens_a = input_ids[start_a:end_a]
                tokens_b = input_ids[start_b:end_b]

                # check_token_matches(question, answer_a, answer_b, tokens_a, tokens_b, tokenizer)

                if r == "cr":
                    cr_last_token_list.append(last_token)
                    chosen_tokens_list.append(tokens_a)
                    rejected_tokens_list.append(tokens_b)
                else:
                    rc_last_token_list.append(last_token)
                
                # Determine attention sources
                if attention_rollout:
                    attn_sources = ["attention_rollout"]
                elif tokenizer.name_or_path in ["google/gemma-3-4b-it"]:
                    # Only use attention sources from the global layers
                    attn_sources = [5, 11, 17, 23, 29]
                    
                elif tokenizer.name_or_path in ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "princeton-nlp/Llama-3-Base-8B-SFT", "mistralai/Mistral-7B-Instruct-v0.2"]:
                    # These models use grouped-query attetnion but no sliding window attention. I think it is fine to take any layer from them
                    assert len(attentions) % 4 == 0
                    interval = len(attentions) // 4
                    attn_sources = list(range(interval, len(attentions)+1, interval))
                    attn_sources = [0] + [s-1 for s in attn_sources]

                elif tokenizer.name_or_path in ["mistralai/Mistral-7B-v0.1", "alignment-handbook/zephyr-7b-sft-full"]:
                    # Uses sliding window attention with window_size = 4096. I think it is fine to take any layer since input_ids are truncated to be within 2k tokens and even the first layer will be able to attend to tokens in both answers.
                    assert len(attentions) % 4 == 0
                    interval = len(attentions) // 4
                    attn_sources = list(range(interval, len(attentions)+1, interval))
                    attn_sources = [0] + [s-1 for s in attn_sources]

                for key in weights_map:
                    if len(weights_map[key]) == 0:
                        weights_map[key] = {attn_source: [] for attn_source in attn_sources}
                
                # Compute Attention Rollout if requested
                rollout_matrix = None
                if attention_rollout:
                    seq_len = input_tensor.shape[1]
                    eye = torch.eye(seq_len).to(model.device)
                    rollout_matrix = eye
                    for avg_layer_attn in attentions:
                        # Add identity to account for residual connections and re-normalize [cite: 3]
                        # A_rollout = 0.5 * A + 0.5 * I
                        layer_attn_res = 0.5 * avg_layer_attn + 0.5 * eye
                        # Recursively multiply matrices [cite: 3, 8]
                        rollout_matrix = torch.matmul(layer_attn_res, rollout_matrix)

                for attn_source in attn_sources:
                    if attn_source == "attention_rollout":
                        mean_attn = rollout_matrix
                    else:
                        mean_attn = attentions[attn_source]

                    # (batch, num_heads, seq_len, seq_len) -> (num_heads, seq_len, seq_len)
                    # attn = outputs.attentions[attn_source][0, :, :, :]
                    # mean_attn = attn.mean(dim=0)

                    assert seq_len == mean_attn.shape[0] and mean_attn.shape[0] == mean_attn.shape[1] and len(mean_attn.shape) == 2

                    last_token_attn = mean_attn[-1, :]

                    attn_a = last_token_attn[start_a:end_a]
                    attn_b = last_token_attn[start_b:end_b]

                    weight_a = (attn_a/attn_a.sum()).cpu().tolist() if attn_a.sum() > 0 else [1.0/attn_a.shape[0]] * attn_a.shape[0]
                    weight_b = (attn_b/attn_b.sum()).cpu().tolist() if attn_b.sum() > 0 else [1.0/attn_b.shape[0]] * attn_b.shape[0]

                    if r == "cr":
                        weights_map["cr_unnormalized_chosen_weights"][attn_source].append(attn_a.cpu().tolist())
                        weights_map["cr_unnormalized_rejected_weights"][attn_source].append(attn_b.cpu().tolist())

                        weights_map["cr_chosen_weights"][attn_source].append(weight_a)
                        weights_map["cr_rejected_weights"][attn_source].append(weight_b)
                    else:
                        weights_map["rc_unnormalized_chosen_weights"][attn_source].append(attn_b.cpu().tolist())
                        weights_map["rc_unnormalized_rejected_weights"][attn_source].append(attn_a.cpu().tolist())

                        weights_map["rc_chosen_weights"][attn_source].append(weight_b)
                        weights_map["rc_rejected_weights"][attn_source].append(weight_a)
                
            for attn_source in weights_map["cr_chosen_weights"]:
                assert len(weights_map["cr_chosen_weights"][attn_source][-1]) == len(chosen_tokens_list[-1]) and len(weights_map["cr_chosen_weights"][attn_source][-1]) == len(weights_map["rc_chosen_weights"][attn_source][-1])
                assert len(weights_map["cr_rejected_weights"][attn_source][-1]) == len(rejected_tokens_list[-1]) and len(weights_map["cr_rejected_weights"][attn_source][-1]) == len(weights_map["rc_rejected_weights"][attn_source][-1])

        for attn_source in weights_map["cr_chosen_weights"]:
            for w1, w2 in zip(weights_map["cr_chosen_weights"][attn_source], weights_map["rc_chosen_weights"][attn_source]):
                weights_map["chosen_weights"][attn_source].append(avg_weights(w1, w2))
            
            for w1, w2 in zip(weights_map["cr_rejected_weights"][attn_source], weights_map["rc_rejected_weights"][attn_source]):
                weights_map["rejected_weights"][attn_source].append(avg_weights(w1, w2))
    
    for key in last_token_stat:
        print("last_token={}: {}".format(key, last_token_stat[key]))

    print("{} responses are empty after stripping. They are changed to a single white space.".format(empty_count))

    for attn_source in weights_map["cr_chosen_weights"]:
        if print_attn_score_stat:
            for key in ["chosen_weights", "rejected_weights", "cr_unnormalized_chosen_weights", "cr_unnormalized_rejected_weights", "rc_unnormalized_chosen_weights", "rc_unnormalized_rejected_weights"]:
                compute_stat(weights_map[key][attn_source], "{}_{}".format(key, attn_source))
        dataset = dataset.add_column("chosen_weights_{}".format(attn_source), weights_map["chosen_weights"][attn_source])
        dataset = dataset.add_column("rejected_weights_{}".format(attn_source), weights_map["rejected_weights"][attn_source])
    dataset = dataset.add_column("chosen_tokens", chosen_tokens_list)
    dataset = dataset.add_column("rejected_tokens", rejected_tokens_list)
    dataset = dataset.add_column("cr_last_token", cr_last_token_list)
    dataset = dataset.add_column("rc_last_token", rc_last_token_list)

    return dataset