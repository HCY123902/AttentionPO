import os
from datasets import load_dataset
from huggingface_hub import whoami


from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--model", type=str)

args = parser.parse_args()

root = "{}-self-judge".format(args.model)

raw_datasets = load_dataset(
    "json",
    data_files = {
        "train": os.path.join(root, "train.jsonl"),
        "test": os.path.join(root, "test.jsonl"),
    },
)

user_name = whoami()["name"]

raw_datasets.push_to_hub(
    "{}/{}-dataset".format(user_name, args.model)
)