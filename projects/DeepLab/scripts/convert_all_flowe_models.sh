#!/bin/bash

set -eu

input_dir=$(
    cd "$1" || exit
    pwd
)

if ! echo "$input_dir" | grep "raw_data" --quiet; then
  echo "$input_dir is not raw data dir"
  exit 1
fi

model_root=${input_dir%%/raw_data*}
raft_name=${input_dir##*raw_data/}
middle_dir="$model_root/middle_convert/$raft_name"
output_dir="$model_root/detectron2/$raft_name"

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
