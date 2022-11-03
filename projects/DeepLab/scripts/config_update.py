import argparse
import os
import re
import glob

import wandb

import wandb_sync_log


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


def update_one_run(run, conv_d2_name="convert_d2_models",
                   model_root="~/ssl_pj/pixpro_pj/PixPro/output"):
    config = run.config
    pretrain_config_keyname = "pretrin"
    args_config_keyname = "args"
    pretrain_config = config.get(pretrain_config_keyname, None)
    args_config = config.get(args_config_keyname, None)
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

    pretrain_config_file_keyname = "pretrain_config"
    pretrain_config_file = args_config.get(pretrain_config_file_keyname, "")
    if pretrain_config is None or pretrain_config_file == "":
        f_idx = model_path.rfind(conv_d2_name)
        config_list = []
        if f_idx >= 0:
            s_dt_str = os.path.basename(model_path[:f_idx].rstrip("/"))
            config_list = glob.glob(os.path.join(model_root, f"**/{s_dt_str}/config.json"),
                                    recursive=True)
            # print(s_dt_str, config_list, model_path[:f_idx])
            # config_list = [os.path.join(model_path[:f_idx], "config.json")]
            # print(config_list)
        if len(config_list) > 0:
            config_file = config_list[0]
            if not os.path.isfile(config_file):
                print("no exist", config_file, "!!")
                return
            load_pretrain_config = wandb_sync_log.load_json(config_file, True, "")
            all_epoch = load_pretrain_config.get("epochs", 1000)
            # config_wandb_pretrain = {"pretrin": pretrain_config}
            run.config[args_config_keyname][pretrain_config_file_keyname] = config_file
            run.config[pretrain_config_keyname] = load_pretrain_config
            print("update: ", f"{args_config_keyname=}.{pretrain_config_file_keyname=}, {config_file=}")
            print("update: ", f"{pretrain_config_keyname=}, {load_pretrain_config=}")

    tmp_epoch_str = config.get("cur_epoch_str", None)
    epoch_str = get_epoch_from_path(model_path, all_epoch)
    if tmp_epoch_str is not None and epoch_str == tmp_epoch_str:
        if pretrain_config is None or pretrain_config_file == "":
            print("update configs")
            run.update()
        else:
            print("already update")
        return

    key_name_str = "cur_epoch_str"
    key_name = "cur_epoch"
    run.config[key_name_str] = epoch_str
    run.config[key_name] = int(epoch_str)
    print("update: ", f"{key_name_str=}, {epoch_str=}")
    print("update: ", f"{key_name=}, {int(epoch_str)=}")
    run.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--entity", type=str, default="tomo")
    parser.add_argument("--project", type=str, default="detectron2")
    parser.add_argument("--model_root", type=str,
                        default="./ssl_pj/pixpro_pj/PixPro/output")
    parser.add_argument("--conv_d2_name", type=str,
                        default="convert_d2_models")
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
        update_one_run(run, args.conv_d2_name, args.model_root)
