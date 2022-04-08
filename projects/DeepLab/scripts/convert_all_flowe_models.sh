#!/bin/bash

set -e

model_root=$(
    cd "$1" || exit
    pwd
)

raft_name="$2"

if [ -z "$raft_name" ]; then
    raft_name="raft-sintel"
fi

set -u

input_dir="$model_root/raw_data/$raft_name"
middle_dir="$model_root/middle_convert/$raft_name"
output_dir="$model_root/detectron2/$raft_name"

# echo "$middle_dir" "$output_dir"

middle_ext=".pth"
out_ext=".pkl"

git_root=$(git rev-parse --show-toplevel | head -1)

detectron2_scriptdir="$git_root/tools"
myscriptdir="$git_root/projects/DeepLab/scripts"

set -x

mkdir -p "$middle_dir"
mkdir -p "$output_dir"

set +x

models=$(find "$input_dir" -type f -name "*.pth.tar" | sort)

for model_path in ${models};
do
    filename=$(basename "$model_path")
    filename_without_ext=${filename%%.*}
    middle_filename="$filename_without_ext""$middle_ext"
    out_filename="$filename_without_ext""$out_ext"
    # echo "$model_path"
    # echo "$filename_without_ext"
    # echo "$middle_filename"
    # echo "$out_filename"
    python "$myscriptdir/change2detectron_model_statedict.py" --model_path "$model_path" --out_path "$middle_dir/$middle_filename"
    python "$detectron2_scriptdir/convert-torchvision-to-d2.py" "$middle_dir/$middle_filename" "$output_dir/$out_filename" 
done
