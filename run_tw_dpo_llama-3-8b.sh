export WANDB_PROJECT="attentionpo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_name=llama-3-8b

NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES="0,1" ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml run_tw_dpo.py training_configs/${run_name}.yaml &> history_record/${run_name}.out
