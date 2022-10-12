import argparse
import os
import sys
import glob
import yaml
import re
import json
import shutil

import wandb


def rename_wandb_name_path(path, remove_str):
    wandb_name = path
    wandb_name = wandb_name.replace(f"{remove_str}", "")
    wandb_name = wandb_name.replace("pretrain", "")
    wandb_name = wandb_name.replace("multi", "")
    wandb_name = wandb_name.replace("/resnet50", "")
    wandb_name = wandb_name.replace("64_2", "")
    wandb_name = wandb_name.replace("epochs_100", "")
    wandb_name = wandb_name.replace("resume", "")
    wandb_name = wandb_name.replace("/", "_")
    wandb_name = wandb_name.replace("pixpro_base_r50_100ep", "")
    wandb_name = wandb_name.replace("convert_d2_models", "")
    wandb_name = re.sub("([_])\\1{%d,}" % 1, "_", wandb_name)
    # wandb_name = wandb_name.replace("__", "_")
    wandb_name = wandb_name.rstrip("_")
    wandb_name = wandb_name.lstrip("_")
    return wandb_name


def get_save_files(root, require_files):
    file_or_dirs = sorted(glob.glob(f"{root}/**", recursive=True))

    save_files = []

    for file_or_dir in file_or_dirs:
        if os.path.isfile(file_or_dir):
            is_require_files = any(w in file_or_dir for w in require_files)
            is_require_files = is_require_files and "wandb/" not in file_or_dir
            if is_require_files:
                save_files.append(file_or_dir)

    save_files = sorted(save_files)
    return save_files


def load_json(metrics_filename, is_dict=True, name="after_log"):
    is_json_format = True
    with open(metrics_filename, "r") as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError:
            is_json_format = False
    if is_json_format:
        if not is_dict:
            metrics = [metrics]
        return metrics

    metrics, key = [], []
    decoder = json.JSONDecoder()
    with open(metrics_filename, "r") as f:
        lines = f.readlines()
    for i, s in enumerate(lines):
        metrics.append(decoder.raw_decode(s))
        key.append(i)

    if name == "":
        if is_dict:
            metrics = dict(zip(key, metrics))
        return metrics

    log_metrics_list, key = [], []
    for i, m in enumerate(metrics):
        for l_m in m[0].keys():
            if "IoU" in l_m:
                tmp_m = {}
                for k, v in m[0].items():
                    tmp_m[f"{name}/{k}"] = v
                # log_metrics_list.append(m[0])
                log_metrics_list.append(tmp_m)
                key.append(i)
                break
    if is_dict:
        log_metrics_list = dict(zip(key, log_metrics_list))
    return log_metrics_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", default="/home/tomo/ssl_pj/detectron2/projects/DeepLab/output"
    )
    parser.add_argument("--project", default="detectron2")
    parser.add_argument("--entity", default="tomo")
    parser.add_argument("--target_path", default="")
    parser.add_argument("--id", default=None)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    root_path = os.path.abspath(args.root_path)
    root = os.path.abspath(args.target_path)

    wandb_name = rename_wandb_name_path(root, root_path)
    wandb_id = args.id
    run = wandb.init(
        entity=args.entity, project=args.project, name=wandb_name, id=wandb_id
    )
    wandb_id = run.id
    print("wandb_id:", wandb_id)
    print("wandb_dir:", root)

    wandb.config.update(args)

    require_files = [".out", ".txt", "metrics.json", "config.yaml"]

    save_files = get_save_files(root, require_files)

    is_config_save = True

    for file_or_dir in save_files:
        l_file_or_dir = file_or_dir
        is_require_files = not ("events." in file_or_dir)
        if is_require_files:
            if args.upload:
                print("save file:", file_or_dir, file=sys.stderr)
                if "config.yaml" in file_or_dir:
                    if is_config_save:
                        with open(file_or_dir, "r") as f:
                            config = yaml.safe_load(f)
                        wandb.config.update(config)
                        is_config_save = False
                elif ".txt" in file_or_dir or ".out" in file_or_dir:
                    base_name = os.path.basename(file_or_dir)
                    ext_log = os.path.splitext(base_name)[1]
                    if ext_log != ".txt":
                        new_f = file_or_dir + ".txt"
                        shutil.copyfile(file_or_dir, new_f)
                        l_file_or_dir = new_f
                wandb.save(l_file_or_dir, base_path=root)
