#!/bin/bash

set -e
set -u

model_root=$(
    cd "$1" || exit
    pwd
)

input_dir="$model_root/torch_pkls"
output_dir="$model_root/detectron2"

out_ext=".pkl"

git_root=$(git rev-parse --show-toplevel | head -1)

detectron2_scriptdir="$git_root/tools"
myscriptdir="$git_root/projects/DeepLab/scripts"

set -x

mkdir -p "$output_dir"

set +x

models=$(find "$input_dir" -type f -name "*.pth.tar" | sort)

for model_path in ${models};
do
    filename=$(basename "$model_path")
    filename_without_ext=${filename%%.*}
    out_filename="$filename_without_ext""$out_ext"

    set -x

    python "$detectron2_scriptdir/convert-torchvision-to-d2.py" "$model_path" "$output_dir/$out_filename" 

    set +x

done
