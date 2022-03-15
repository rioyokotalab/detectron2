import argparse
import os
import pickle

import torch


def change_bn_name(key):
    old_name = "bn"
    len_bn = len(old_name)
    new_key = ""
    split_key = key.split(".")
    conv_key, bn_key = "conv", ""
    for local_key in split_key:
        if old_name in local_key:
            bn_key = local_key
            num_bn = int(local_key[len_bn:])
            conv_key += f"{num_bn}"
            break
    if bn_key != "":
        replcace_str = f"{conv_key}.norm"
        new_key = key.replace(bn_key, replcace_str)
    return new_key


def change_layername(key):
    old_name = "layer"
    len_layer = len(old_name)
    split_key = key.split(".")
    res_key = "res"
    for local_key in split_key:
        if old_name in local_key:
            num_layer = int(local_key[len_layer:]) + 1
            res_key += f"{num_layer}"
            return key.replace(local_key, res_key)


def change_downsamplename(key):
    old_name = "downsample"
    split_key = key.split(".")
    shortcur_key = "shortcut"
    for i, local_key in enumerate(split_key):
        if old_name in local_key:
            old_replace_name = old_name
            num_downsample = int(split_key[i + 1])
            old_replace_name += f".{num_downsample}"
            if num_downsample == 1:
                shortcur_key += ".norm"
            return key.replace(old_replace_name, shortcur_key)


def checkpoint2detectron(checkpoint):
    state_dict = {}
    # state_dict["model_state"] = {}
    for k in checkpoint["state_dict"].keys():
        if "model." in k:
            new_k = k.replace("model.", "")
            if "layer" not in new_k and "flow" not in new_k:
                new_k = "stem." + new_k
            new_k = "backbone." + new_k
            if new_k.endswith("num_batches_tracked"):
                continue
            if "downsample" in new_k:
                new_k = change_downsamplename(new_k)
            if "layer" in new_k:
                new_k = change_layername(new_k)
            if "bn" in new_k:
                new_k = change_bn_name(new_k)
            # state_dict["model_state"][new_k] = checkpoint["state_dict"][k]
            state_dict[new_k] = checkpoint["state_dict"][k]
    return state_dict


if __name__ == "__main__":
    ext_torch_list = [".pth", ".pth.tar", ".pyth"]
    ext_pickle_list = [".pkl", ".binaryfile"]
    ext_list = ext_torch_list + ext_pickle_list

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--out_path", default="./output/new_checkpoint")
    parser.add_argument("--ext", default=".pth", choices=ext_list)
    args = parser.parse_args()

    pretrain_encoder_path = args.model_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    checkpoint = torch.load(pretrain_encoder_path, map_location="cpu")
    checkpoint = checkpoint2detectron(checkpoint)

    filename_pth_tar = os.path.basename(pretrain_encoder_path)
    dirname, basename = os.path.split(filename_pth_tar)
    basename_without_ext, ext = basename.split(".", 1)
    path_without_ext = os.path.join(dirname, basename_without_ext)
    print(path_without_ext)
    save_name_ext = args.ext
    save_name = path_without_ext + "{save_name_ext}"
    print("save as:", save_name)
    # new_checkpoint_file_path = os.path.join(out_path, f"{path_without_ext}.pyth")
    new_checkpoint_file_path = os.path.join(out_path, f"{save_name}")
    if save_name_ext in ext_torch_list:
        torch.save(checkpoint, new_checkpoint_file_path)
    elif save_name_ext in ext_pickle_list:
        with open(new_checkpoint_file_path, "wb") as f:
            pickle.dump(checkpoint, f)
