export WANDB_PROJECT="attentionpo"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dataset_root="datasets"
config_root="training_configs"

source_name=${MODEL_NAME}

train_source="${dataset_root}/${source_name}/utf_train"
test_source="${dataset_root}/${source_name}/utf_test"



target="${MODEL_NAME}-self-judge"

config_path=${config_root}/${target}

if [ ! -e ${config_path} ]; then
    mkdir ${config_path}
fi

res_path=${dataset_root}/${target}

if [ ! -e ${res_path} ]; then
    mkdir ${res_path}
fi

chat_template=null


if [ "${PART}" = "0" ]; then
    source_split="${train_source}_part_0_10000.json"
    result="train_part_0_10000.jsonl"
elif [ "${PART}" = "1" ]; then
    source_split="${train_source}_part_10000_20000.json"
    result="train_part_10000_20000.jsonl"
elif [ "${PART}" = "2" ]; then
    source_split="${train_source}_part_20000_30000.json"
    result="train_part_20000_30000.jsonl"
elif [ "${PART}" = "3" ]; then
    source_split="${train_source}_part_30000_40000.json"
    result="train_part_30000_40000.jsonl"
elif [ "${PART}" = "4" ]; then
    source_split="${train_source}_part_40000_50000.json"
    result="train_part_40000_50000.jsonl"
elif [ "${PART}" = "5" ]; then
    if [ "${source_name}" = "llama-3-8b" ]; then
        source_split="${train_source}_part_50000_60000.json"
        result="train_part_50000_60000.jsonl"
    elif [ "${source_name}" = "llama-3-8b-inst" ]; then
        source_split="${train_source}_part_50000_59876.json"
        result="train_part_50000_59876.jsonl"
    fi
elif [ "${PART}" = "6" ]; then
    if [ "${source_name}" = "llama-3-8b" ]; then
        source_split="${train_source}_part_60000_61135.json"
        result="train_part_60000_61135.jsonl"
    elif [ "${source_name}" = "llama-3-8b-inst" ]; then
        source_split="${test_source}.json"
        result="test.jsonl"
    fi
elif [ "${PART}" = "7" ]; then
    if [ "${source_name}" = "llama-3-8b" ]; then
        source_split="${test_source}.json"
        result="test.jsonl"
    fi
fi

config_yaml_path=${config_path}/${PART}.yaml

cat > ${config_yaml_path} << EOL
# Model arguments
model_name_or_path: ${MODEL_PATH}
torch_dtype: bfloat16

# Data training arguments
# For definitions, see: src/h4/training/config.py
dataset_mixer:
  json: 1.0
dataset_splits:
- ${source_split}
preprocessing_num_workers: 12
chat_template: ${chat_template}

# DPOTrainer arguments
bf16: true
beta: 0.01
do_eval: true
eval_strategy: steps
eval_steps: 500
gradient_accumulation_steps: 8
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: False
hub_model_id: xxx
learning_rate: 1.0e-6
log_level: info
logging_steps: 10
lr_scheduler_type: cosine
max_length: 8192
max_prompt_length: 6144
num_train_epochs: 1
optim: adamw_torch
output_dir: xxx
per_device_train_batch_size: 1
per_device_eval_batch_size: 2
push_to_hub: false
save_strategy: steps
save_steps: 500
save_total_limit: 4
save_only_model: true
seed: 42
warmup_ratio: 0.1
run_name: xxx
report_to: "wandb"
use_weighting: false
self_judge: true
self_judge_output_path: ${res_path}/${result}
attn_source: ""
EOL

CUDA_VISIBLE_DEVICES=${DEVICES} python run_tw_dpo.py ${config_yaml_path} &> ${config_path}/${PART}.out