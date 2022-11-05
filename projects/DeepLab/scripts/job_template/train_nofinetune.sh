#!/bin/bash

#------ pjsub option --------#
#PJM -L node=1
#PJM --mpi proc=4
#PJM -L rscgrp=cxgfs-single
#PJM -L elapse=336:00:00
#PJM -N pixpro_cityres50detectron 
#PJM -j
#PJM -X

set -x

echo "start scirpt file cat"
cat "$0"

set +x

echo "end scirpt file cat"

date_str=$(date '+%Y%m%d_%H%M%S')
START_TIMESTAMP=$(date '+%s')
# \#PJM -L rscunit=cx
# \#PJM -L rscgrp=cx-single

detectron2_root="/data/group1/$(whoami)/datasets"
cityscapes_root="$detectron2_root/cityscapes"

# ======== NVMESH ========

nvmesh_root="/beegfs/$(whoami)"
detectron2_root="$nvmesh_root"

# # ======== Copy ========
# 
# local_ssd_path="$PJM_LOCALDIR"
# local_data_root="$local_ssd_path"
# 
# COPY_START_TIMESTAMP=$(date '+%s')
# 
# rsync -avz "$cityscapes_root" "$local_data_root"
# COPY_END_TIMESTAMP=$(date '+%s')
# 
# COPY_E_TIME=$(($COPY_END_TIMESTAMP-$COPY_START_TIMESTAMP))
# echo "copy time: $COPY_E_TIME s"
# 
# detectron2_root="$local_data_root"
# 
# ======== Variables ========

export DETECTRON2_DATASETS="$detectron2_root"

job_id_base=$PJM_JOBID

git_root=$(git rev-parse --show-toplevel | head -1)
project_root="$git_root/projects/DeepLab"
pixpro_root=${PIXPRO_ROOT:-"/data/group1/$(whoami)/home_dir/ssl_pj/pixpro_pj/PixPro"}

config_root="$project_root/configs/Cityscapes-SemanticSegmentation"
config_file="$config_root/deeplab_v3_R_50_myencoder_mg124_poly_40k_bs8.yaml"

# /data/group1/$(whoami)/home_dir/ssl_pj/pixpro_pj/PixPro/output/

model_relative_path=${MODEL_RELATIVE:-"pixpro_base_r50_100ep/bdd100k/of_fix/official_main/official_main_bdd100k/20220809_134206/convert_d2_models/ckpt_epoch_10"}
echo "$model_relative_path"

model_path="$pixpro_root/output/$model_relative_path.pkl"

# date_str=$(date '+%Y%m%d_%H%M%S')
# out_path="$project_root/output/pixpro/cityscapes/R50/flow_output/$model_relative_path/$date_str"
out_path="$project_root/output/pixpro/cityscapes/R50/$model_relative_path/$date_str"

i=1
tmp_out_path="$out_path"
while [ -d "$out_path" ]
do
    out_path="$tmp_out_path""_$i"
    i=$(($i + 1))
done


model_rel_dirname=$(dirname "$model_path")
while_root="$pixpro_root/output"
while true
do
    if [ "$model_rel_dirname" = "$while_root" ];then
        break
    fi
    pretrain_config=$(find "$model_rel_dirname" -name "config.json")
    if [ -n "$pretrain_config" ];then
        break
    fi
    model_rel_dirname=$(dirname "$model_rel_dirname")
done

git_out="$out_path/git_out"
mkdir -p "$git_out"

pwd_path="$(pwd)"
log_file="$PJM_JOBNAME.$job_id_base.out"

# ======== Pyenv ========

GHOME="/data/group1/$(whoami)/home_dir"
export PYENV_ROOT="$GHOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
# pipenv property
export PIPENV_VENV_IN_PROJECT=1
export PIPENV_IGNORE_VIRTUALENVS=1
which python

# ======== Modules ========

module load cuda/10.2.89_440.33.01
module load nccl/2.7.3

module load gcc/8.4.0
module load cmake/3.21.1
module load cudnn/8.2.1
module load openmpi/4.0.4

module list

# ======== MPI ========

nodes=$PJM_NODE
# cpus_pernode=5
gpus_pernode=${PJM_PROC_BY_NODE}

gpus=${PJM_MPI_PROC}
# cpus=$(($nodes * $cpus_pernode))

echo "gpus: $gpus"
echo "gpus per node $gpus_pernode"

MASTER_ADDR=$(cat "$PJM_O_NODEINF" | head -1)
MASTER_PORT=$((10000 + ($job_id_base % 50000)))

# MPI_OPTS="-machinefile $PJM_O_NODEINF"
# MPI_OPTS+=" -np $gpus"
# MPI_OPTS+=" -npernode $gpus_pernode"
# MPI_OPTS+=" -x MASTER_ADDR=$MASTER_ADDR"
# MPI_OPTS+=" -x MASTER_PORT=$MASTER_PORT"
# MPI_OPTS+=" -x NCCL_BUFFSIZE=1048576"
# MPI_OPTS+=" -x NCCL_IB_DISABLE=1"
# MPI_OPTS+=" -x NCCL_IB_TIMEOUT=14"

mpi_backend="nccl"
# mpi_backend="mpi"
# mpi_backend="gloo"

METHOD="tcp://$MASTER_ADDR:$MASTER_PORT"

# ======== Scripts ========


pushd "$project_root"

set -x
git status | tee "$git_out/git_status.txt"
# git log | tee "$git_out/git_log.txt"
git log > "$git_out/git_log.txt"
git diff HEAD | tee "$git_out/git_diff.txt"
git rev-parse HEAD | tee "$git_out/git_head.txt"

python train_net.py \
    --config-file "$config_file" \
    --output "$out_path" \
    --model_path "$model_path" \
    --pretrain_config "$pretrain_config" \
    --no_finetune \
    --dist-url "$METHOD" \
    --num-gpus $gpus \
    --log_name "$pwd_path/$log_file"

time python scripts/cityscapes_color_vis.py \
    --mask_path "$out_path" \
    --r 

popd

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"

mkdir -p "$out_path/script"
script_name=$(basename "$0")
cat "$0" > "$out_path/script/$script_name.sh"

cp "$pwd_path/$log_file" "$out_path"

