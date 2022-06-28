import argparse
import os
import sys
import glob
from collections import defaultdict
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
    parser.add_argument("--target_path", default="")
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    root_path = os.path.abspath(args.root_path)
    root = os.path.abspath(args.target_path)

    wandb_name = rename_wandb_name_path(root, root_path)

    # wandb.init(entity="tomo", project=args.project, name=wandb_name, id=wandb_id)
    # run = wandb.init(entity="tomo", project=args.project, name=wandb_name)
    # wandb_id = run.id

    file_or_dirs = sorted(glob.glob(f"{root}/**", recursive=True))

    require_files = ["model_final.pth", ".out", "config.yaml"]

    save_files = defaultdict(list)

    for file_or_dir in file_or_dirs:
        if os.path.isfile(file_or_dir):
            is_require_files = any(w in file_or_dir for w in require_files)
            is_require_files = is_require_files and not ("events." in file_or_dir)
            if is_require_files:
                dirname = os.path.dirname(file_or_dir)
                save_files[dirname].append(file_or_dir)
                # print("wandb_id:", wandb_id)
                # print("save file:", file_or_dir, file=sys.stderr)
                # wandb.save(file_or_dir, base_path=root)
        # elif os.path.isdir(file_or_dir):
        #     print("dir:", file_or_dir)

    for i, (dirname, save_file_list) in enumerate(save_files.items()):
        wandb_id = None if args.ids is None else args.ids[i]
        local_wandb_name = rename_wandb_name_path(dirname, root_path)
        run = wandb.init(entity="tomo", project=args.project, name=local_wandb_name, id=wandb_id)
        wandb_id = run.id
        # wandb_id = local_wandb_name
        print("wandb_id:", wandb_id)
        print("wandb_dir:", dirname)
        if args.upload:
            wandb.config.update(args)
            for save_file in save_file_list:
                print("save file:", save_file, file=sys.stderr)
                if "config.yaml" in save_file:
                    with open(save_file, "r") as f:
                        config = yaml.safe_load(f)
                    wandb.config.update(config)
                wandb.save(save_file, base_path=dirname)
        run.finish()

    # wandb.save(os.path.join(root, "model_final.pth"), base_path=root)
    # wandb.save(os.path.join(root, "*.out"), base_path=root)
    # wandb.save(os.path.join(root, "config.yaml"), base_path=root)
    # # wandb.save(os.path.join(root, "events.*"))

    # print("wandb_id:", wandb_id)
