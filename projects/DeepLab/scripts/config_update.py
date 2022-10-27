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


def update_one_run(run):
    config = run.config
    pretrain_config = config.get("pretrin", None)
    args_config = config.get("args", None)
    all_epoch = 1000
    if pretrain_config is not None:
        all_epoch = pretrain_config.get("epochs", 1000)

    print(f"{all_epoch=}")

    if args_config is None:
        return
    model_path = args_config.get("model_path", None)
    if model_path is None or model_path == "":
        return
    print(f"{model_path=}")
    epoch_str = get_epoch_from_path(model_path, all_epoch)

    key_name = "cur_epoch"
    run.config[key_name] = epoch_str
    print("update: ", f"{key_name=}, {epoch_str=}")
    run.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--entity", type=str, default="tomo")
    parser.add_argument("--project", type=str, default="detectron2")
    args = parser.parse_args()

    api = wandb.Api()
    if args.id is None:
        runs = api.runs(args.project)
    else:
        run = api.run(f"{args.entity}/{args.project}/{args.id}")
        runs = [run]
    print("run num:", len(runs))

    for run in runs:
        print("start:", run)
        update_one_run(run)
