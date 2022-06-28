import argparse
import os
import sys
import glob
import yaml

import wandb


def rename_wandb_name_path(path, remove_str):
    wandb_name = path
    wandb_name = wandb_name.replace(f"{remove_str}", "")
    wandb_name = wandb_name.replace("/", "_")
    wandb_name = wandb_name.replace("pixpro_base_r50_100ep", "")
    wandb_name = wandb_name.replace("convert_d2_models", "")
    wandb_name = wandb_name.rstrip("_")
    wandb_name = wandb_name.lstrip("_")
    wandb_name = wandb_name.replace("__", "_")
    return wandb_name


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

    file_or_dirs = sorted(glob.glob(f"{root}/**", recursive=True))

    require_files = ["model_final.pth", ".out", ".txt", "metrics.json", "config.yaml"]

    is_config_save = True

    for file_or_dir in file_or_dirs:
        if os.path.isfile(file_or_dir):
            is_require_files = any(w in file_or_dir for w in require_files)
            is_require_files = is_require_files and not ("events." in file_or_dir)
            if is_require_files:
                if args.upload:
                    print("save file:", file_or_dir, file=sys.stderr)
                    if "config.yaml" in file_or_dir:
                        if is_config_save:
                            with open(file_or_dir, "r") as f:
                                config = yaml.safe_load(f)
                            wandb.config.update(config)
                            is_config_save = False
                    wandb.save(file_or_dir, base_path=root)
