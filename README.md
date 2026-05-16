# AttentionPO

## Setup

### Clone the repository.

```
git clone https://github.com/HCY123902/AttentionPO.git
```

### Install Python environment.

```
uv venv attentionpo --python 3.10.16

source attentionpo/bin/activate

cd AttentionPO/

uv pip install -r requirements.txt

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

uv pip install flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

```
huggingface-cli login --token {YOUR_HUGGINGFACE_TOKEN}
export WANDB_API_KEY="YOUR_WANDB_TOKEN"
wandb login
```

## Datasets

### Existing Datasets

We provide the existing datasets here.

| **Model** | **Download** |
| :------------: | :------------: |
| LLaMA-3-8B-Base-SFT | [🤗 HuggingFace](https://huggingface.co/datasets/AttentionPO/llama-3-8b-dataset)   |
| LLaMA-3-8B-Instruct |  [🤗 HuggingFace](https://huggingface.co/datasets/AttentionPO/llama-3-8b-inst-dataset)   |

### Curating on Your Own

```
cd datasets/
python get_init_dataset.py --model llama-3-8b
cd ..
bash run_compute_weights_llama-3-8b.sh
cd datasets/
python merge_chunks.py --model llama-3-8b
python transmit_to_hub.py --model llama-3-8b
```

To curate datasets for LLaMA-3-8B-Instruct, replace `llama-3-8b` with `llama-3-8b-inst`.

## Training

### Trained Checkpoints

| **Model** | **Download** |
| :------------: | :------------: |
| LLaMA-3-8B-Base-SFT | [🤗 HuggingFace](https://huggingface.co/datasets/AttentionPO/llama-3-8b)   |
| LLaMA-3-8B-Instruct |  [🤗 HuggingFace](https://huggingface.co/datasets/AttentionPO/llama-3-8b-inst)   |

### Training Scripts

```
bash run_tw_dpo_llama-3-8b.sh
```

To train LLaMA-3-8B-Instruct, replace `llama-3-8b` with `llama-3-8b-inst`.