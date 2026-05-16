import os
from pathlib import Path
import json

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--model", type=str)

args = parser.parse_args()

root = "{}-self-judge".format(args.model)
root_path = Path(root)

jsonls = sorted(list(root_path.glob("train_part*.jsonl")))

res_path = os.path.join(root, "train.jsonl")

if not os.path.exists(res_path):
    with open(res_path, "w") as res_jsonl:
        for file in jsonls:
            file_path = os.path.join(root, file.name)
            with open(file_path) as src_jsonl:
                for line in src_jsonl:
                    res_jsonl.write(line)
            os.remove(file_path)