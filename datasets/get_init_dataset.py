import os
import json

from datasets import load_dataset

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--model", type=str)

args = parser.parse_args()

model = args.model


if model == "llama-3-8b":
    path = "HuggingFaceH4/ultrafeedback_binarized"
    splits = ["train_prefs", "test_prefs"]
    
elif model == "llama-3-8b-inst":
    path = "princeton-nlp/llama3-ultrafeedback"
    splits = ["train", "test"]

else:
    raise Exception("{} is not supported".format(model))


raw_datasets = load_dataset(
    path
)

chunk_size = 10000

save_root = model
if not os.path.exists(save_root):
    os.mkdir(save_root)

for split in splits:
    res = []
    for s in raw_datasets[split]:
        res.append(
            {
                "prompt": s["prompt"],
                "prompt_id": s["prompt_id"],
                "chosen": s["chosen"],
                "rejected": s["rejected"]
            }
        )
    if "train" in split:
        for i in range(0, len(raw_datasets[split]), chunk_size):
            save_path = os.path.join(save_root, "utf_train_part_{}_{}.json".format(i, min(i+chunk_size, len(raw_datasets[split]))))
            with open(save_path, "w") as res_json:
                json.dump(res[i:i+chunk_size], res_json, indent=4)
    else:
        save_path = os.path.join(save_root, "utf_test.json")
        with open(save_path, "w") as res_json:
            json.dump(res, res_json, indent=4)
    