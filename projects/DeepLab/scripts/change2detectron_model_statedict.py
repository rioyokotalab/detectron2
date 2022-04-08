import argparse
import os
import pickle

import torch


def extract_model_state_dict(checkpoint):
    # state_dict = {"__author__": "tomo", "matching_heuristics": True}
    # state_dict["model"] = checkpoint["state_dict"]
    state_dict = checkpoint["state_dict"]
    return state_dict


if __name__ == "__main__":
    ext_torch_list = [".pth", ".pth.tar", ".pyth"]
    ext_pickle_list = [".pkl", ".binaryfile"]
    ext_list = ext_torch_list + ext_pickle_list

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--out_path", default="./output/new_checkpoint")
    args = parser.parse_args()

    pretrain_encoder_path = args.model_path
    out_path = args.out_path

    checkpoint = torch.load(pretrain_encoder_path, map_location="cpu")
    checkpoint = extract_model_state_dict(checkpoint)

    out_basename = os.path.basename(out_path)
    dirname, basename = os.path.split(out_basename)
    basename_without_ext, ext = basename.split(".", 1)
    path_without_ext = os.path.join(dirname, basename_without_ext)
    save_ext = "." + ext
    print(path_without_ext, save_ext)
    print("save as:", out_path)
    if save_ext in ext_torch_list:
        torch.save(checkpoint, out_path)
    elif save_ext in ext_pickle_list:
        with open(out_path, "wb") as f:
            pickle.dump(checkpoint, f)
