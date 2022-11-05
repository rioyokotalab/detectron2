#!/bin/bash

set -x

git_root=$(git rev-parse --show-toplevel | head -1)
project_root="$git_root/projects/DeepLab"

model_relative_path=${1:-"abci_output/pixpro_base_r50_100ep/bdd100k/images/16/SimCLR/512x1024/20220531_101656/convert_d2_models/ckpt_epoch_100"}

pixpro_root=${2:-"/data/group1/$(whoami)/home_dir/ssl_pj/pixpro_pj/PixPro"}

job_script_name=${3:-"$project_root/jobs/cityscapes/train_nofinetune.sh"}

num=5

gname=$(groups | awk '{ print $1 }')

for i in $(seq 1 ${num});
do
    MODEL_RELATIVE="$model_relative_path" PIXPRO_ROOT="$pixpro_root" pjsub -g $gname "$job_script_name"
    sleep 5
done

