import argparse
import os
import pickle

import torch
import numpy as np


def flatten_dict(d, pre_lst=None, result=None):
    if result is None:
        result = {}
    if pre_lst is None:
        pre_lst = []
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, pre_lst=pre_lst + [k], result=result)
        else:
            new_k = ""
            for lk in pre_lst:
                new_k += lk + "."
            new_k += k
            result[new_k] = v
    return result


def preprocess_dict(checkpoint):
    out = {}
    key = list(checkpoint.keys())[0]
    for k, v in checkpoint[key].items():
        tmp_k = k.replace("~", "")
        tmp_k = tmp_k.replace("//", "/")
        tmp_k = tmp_k.replace("/", ".")
        out[tmp_k] = v
    out = flatten_dict(out)
    return out


def common_change_name(key):
    new_k = key.replace("initial_", "stem.")
    new_k = new_k.replace(".w", ".weight")
    new_k = new_k.replace(".offset", ".bias")
    new_k = new_k.replace(".scale", ".weight")
    new_k = new_k.replace("shortcut_", "shortcut.")
    new_k = new_k.replace("block_", "")
    return new_k


def change_bn_name(key):
    old_name = "batchnorm"
    if old_name not in key:
        return key

    split_key = key.split(".")
    conv_key = "conv"
    for local_key in split_key:
        if old_name in local_key:
            lk_splits = local_key.split("_")
            num_bn = 1
            if len(lk_splits) > 1:
                num_bn = int(lk_splits[1]) + 1
            conv_key += f"{num_bn}"
            replcace_str = f"{conv_key}.norm"
            new_k = key.replace(local_key, replcace_str)
            return new_k
    return key


def change_blockgroupname(key, old_name="block_group"):
    if old_name not in key:
        return key

    no_under_oldname = old_name.replace("_", "")
    split_key = key.split(".")
    res_key = "res"
    for local_key in split_key:
        if old_name in local_key:
            tmp_key = local_key.replace(old_name, no_under_oldname)
            tmp_k_splits = tmp_key.split("_")
            num_bg = 0
            if len(tmp_k_splits) > 1:
                num_bg = int(tmp_k_splits[1]) + 2
            res_key += f"{num_bg}"
            return key.replace(local_key, res_key)
    return key


def change_shortcutname(key):
    shortcut_key = "shortcut"
    if shortcut_key not in key:
        return key

    conv_name = "conv"
    split_keys, new_key = key.split("."), ""
    len_key = len(split_keys) - 1
    for i, local_key in enumerate(split_keys):
        if conv_name in local_key:
            continue
        new_key += local_key
        new_key += "." if i < len_key else ""
    return new_key


def change_convname(key):
    old_name = "conv"
    if old_name not in key:
        return key

    split_key = key.split(".")
    for local_key in split_key:
        if old_name in local_key:
            lk_splits = local_key.split("_")
            is_split = len(lk_splits) > 1
            is_same = old_name == local_key
            if not is_split and not is_same:
                break
            num_conv = int(lk_splits[1]) + 1 if is_split else 1
            replcace_str = old_name + f"{num_conv}"
            return key.replace(local_key, replcace_str)
    return key


def checkpointkey2detectron(checkpoint):
    checkpoint_model_dict = preprocess_dict(checkpoint)
    tmp = [k for k in checkpoint_model_dict if "res" in k]
    print(len(tmp))
    state_dict = {"__author__": "tomo", "matching_heuristics": True}
    state_dict["model"] = {}
    i = 0
    for k in checkpoint_model_dict.keys():
        if "res" in k:
            model_name, rest = k.split(".", 1)
            new_k = rest
            new_k = "backbone." + new_k
            new_k = common_change_name(new_k)
            new_k = change_bn_name(new_k)
            new_k = change_shortcutname(new_k)
            new_k = change_blockgroupname(new_k, "group")
            new_k = change_convname(new_k)
            state_dict["model"][new_k] = checkpoint_model_dict[k]
            i += 1
            # print(i, k, ":", new_k)
            # print(len(state_dict["model"].keys()))
    print(len(state_dict["model"].keys()))
    return state_dict


def haiku_value2torch(model_state_dict):
    state_dict = {}
    for k, v in model_state_dict.items():
        ndim = v.ndim
        old_dim_order = list(range(ndim))
        new_dim_order = old_dim_order[::-1]
        new_v = v.transpose(new_dim_order)
        if "norm" in k:
            old_shape = v.shape
            mul_all = np.prod(old_shape)
            # print(k, v.shape, new_v.shape, mul_all)
            if mul_all in old_shape:
                new_v = new_v.reshape(-1)
            # print(k, v.shape, new_v.shape, mul_all)
        # print(k, v.shape, v.ndim, new_dim_order)
        # print(k, v.shape, new_v.shape)
        state_dict[k] = new_v
    return state_dict


def checkpoint2detectron(checkpoint):
    rename_key_checkpoint = checkpointkey2detectron(checkpoint)
    state_dict = haiku_value2torch(rename_key_checkpoint["model"])
    rename_key_checkpoint["model"] = state_dict
    return rename_key_checkpoint


def main(args):
    pretrain_encoder_path = args.model_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    with open(pretrain_encoder_path, "rb") as f:
        data = pickle.load(f)
    model_state_dict = data[list(data.keys())[0]]
    checkpoint = checkpoint2detectron(model_state_dict)

    filename_pth_tar = os.path.basename(pretrain_encoder_path)
    dirname, basename = os.path.split(filename_pth_tar)
    basename_without_ext, ext = basename.split(".", 1)
    path_without_ext = os.path.join(dirname, basename_without_ext)
    print(path_without_ext)
    save_name_ext = args.ext
    save_name = path_without_ext + f"{save_name_ext}"
    print("save as:", save_name)
    # new_checkpoint_file_path = os.path.join(out_path, f"{path_without_ext}.pyth")
    new_checkpoint_file_path = os.path.join(out_path, f"{save_name}")
    if save_name_ext in ext_torch_list:
        torch.save(checkpoint, new_checkpoint_file_path)
    elif save_name_ext in ext_pickle_list:
        with open(new_checkpoint_file_path, "wb") as f:
            pickle.dump(checkpoint, f)


if __name__ == "__main__":
    ext_torch_list = [".pth", ".pth.tar", ".pyth"]
    ext_pickle_list = [".pkl", ".binaryfile"]
    ext_list = ext_torch_list + ext_pickle_list

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--out_path", default="./output/new_checkpoint")
    parser.add_argument("--ext", default=".pth", choices=ext_list)
    args = parser.parse_args()
    main(args)
