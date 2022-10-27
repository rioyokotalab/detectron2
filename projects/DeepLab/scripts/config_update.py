import argparse
import os
import re

import wandb


def get_epoch_from_path(model_path, all_epoch):
    epoch_digit = len(str(all_epoch))
    b_name = os.path.basename(model_path)
    cur_epoch_name = os.path.splitext(b_name)[0]
    cur_epoch_list = re.findall(r"\d+ep", cur_epoch_name)
    cur_epoch_tmp = cur_epoch_name
    if len(cur_epoch_list) > 0:
        cur_epoch_tmp = cur_epoch_list[0]
    cur_epoch = re.findall(r"\d+", cur_epoch_tmp)[0]
    cur_epoch_str = str(cur_epoch).zfill(epoch_digit)
    return cur_epoch_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str)
    parser.add_argument("--entity", type=str, default="tomo")
    parser.add_argument("--project", type=str, default="detectron2")
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{args.id}")

    all_epoch = run.config["pretrin"]["epochs"]
    model_path = run.config["args"]["model_path"]
    print(f"{all_epoch=}")
    print(f"{model_path=}")
    epoch_str = get_epoch_from_path(model_path, all_epoch)

    key_name = "cur_epoch"
    run.config[key_name] = epoch_str
    print("update: ", f"{key_name=}, {epoch_str=}")
    run.update()
