import argparse
import os
import glob

import numpy as np
from PIL import Image

PALETTE = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [255, 255, 255],
    ]
).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask_path",
        default="./output/inference/cityscapes_eval_0",
        help="input mask img path",
    )
    parser.add_argument("--color_out_path", default="", help="out path")
    parser.add_argument("--target_dirname", default="inference")
    parser.add_argument("--recursive", "--r", action="store_true")
    parser.add_argument("--img_format", nargs="+", default=["png", "jpg"])
    args = parser.parse_args()

    is_recusive = args.recursive
    t_name = args.target_dirname
    root = os.path.abspath(args.mask_path)

    if is_recusive:
        root = os.path.join(root, "**", t_name, "**")

    # print(root)
    tmp_out_root = args.color_out_path
    is_make_name_out = tmp_out_root == "" or tmp_out_root is None
    if not is_make_name_out:
        out_root = os.path.abspath(tmp_out_root)
        os.makedirs(out_root, exist_ok=True)
        # print(out_root)
    if os.path.isfile(root):
        if is_recusive:
            raise FileNotFoundError("not recursive because your path is file")
        img_paths = [root]
    else:
        assert args.img_format is not None
        img_paths = []
        for img_format in args.img_format:
            ext_str = f"*.{img_format}"
            path_name = os.path.join(root, ext_str)
            # print(path_name)
            img_paths += glob.glob(path_name, recursive=is_recusive)
    img_paths = [img_path for img_path in img_paths if "cityscapes" not in img_path]
    img_paths = [img_path for img_path in img_paths if "_color" not in img_path]
    img_paths = sorted(img_paths)
    # print(img_paths[0], img_paths[-1])

    palette = PALETTE.flatten().copy().reshape(-1, 3)

    for img_path in img_paths:
        dir_name, base_name = os.path.split(img_path)
        img = Image.open(img_path)
        img_copy = img.copy()
        if is_make_name_out:
            out_root = dir_name + "_color"
            if not os.path.exists(out_root):
                # print(out_root)
                os.makedirs(out_root, exist_ok=True)
        out_name = os.path.join(out_root, base_name)
        img_copy.putpalette(palette)
        img_copy.save(out_name)
