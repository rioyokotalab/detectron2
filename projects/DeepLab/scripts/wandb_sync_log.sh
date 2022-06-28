#!/bin/bash
set -ex

git_root=$(git rev-parse --show-toplevel | head -1)
python_path="$git_root/projects/DeepLab/scripts/wandb_sync_log.py"

out_root=${1:-"$git_root/projects/DeepLab/output"}
out_root=$(
    cd "$out_root" || exit
    pwd
)

target_list=$(find "$out_root" -name "convert_d2_models" -type d | sort)
# target_list=$(find "$out_root" -name "convert_d2_models" -type d | sort | head -1)
wandb_project_name=${2:-"detectron2"}

set +x
num_target=$(echo "$target_list" | wc -l)
set -x

echo "$num_target"

wandb_id_list=""

pushd "$out_root"

for t_path in ${target_list};
do
    python "$python_path" --project "$wandb_project_name" --target_path "$t_path" > "wandb_id.log"
    cat "wandb_id.log" >> "wandb_id_all.log"

    set +x
    wandb_ids=$(cat "wandb_id.log" | grep "wandb_id:")
    local_ts=$(cat "wandb_id.log" | grep "wandb_dir:")
    tf_log_num=$(echo "$wandb_ids" | wc -l)
    set -x

    for i in $(seq 1 ${tf_log_num});
    do
        set +x
        local_wandb_id=$(echo "$wandb_ids" | head -"$i")
        local_t=$(echo "$local_ts" | head -"$i")
        set -x
        local_wandb_id=${local_wandb_id##* }
        local_t_path=${local_t##* }
        local_wandb_id=$(echo ${local_wandb_id} | sed -e "s/[\r\n]\+//g")
        # echo "$local_wandb_id $local_t_path"
        tf_log=$(find "$local_t_path" -name "events.*")
        if [ -f "$tf_log" ];then
            wandb sync -p "$wandb_project_name" --id "$local_wandb_id" "$tf_log" 
        fi
        wandb_id_list+="$local_wandb_id "
    done
    python "$python_path" --project "$wandb_project_name" --target_path "$t_path" --upload --ids $wandb_id_list
done

popd