#!/bin/bash
set -ex

git_root=$(git rev-parse --show-toplevel | head -1)
python_path="$git_root/projects/DeepLab/scripts/wandb_sync_log.py"

out_root=${1:-"$git_root/projects/DeepLab/output"}
out_root=$(
    cd "$out_root" || exit
    pwd
)

target_list=$(find "$out_root" -name "model_final.pth" | sort)
# target_list=$(find "$out_root" -name "model_final.pth" | sort | head -1)
wandb_project_name=${2:-"detectron2"}
entity=${3:-"tomo"}

set +x
num_target=$(echo "$target_list" | wc -l)
set -x

echo "$num_target"

pushd "$out_root"

for t_f_path in ${target_list};
do
    t_path="$t_f_path"
    if [ -f "$t_f_path" ];then
        t_path=$(dirname "$t_f_path")
    fi
    wandb_id_list=""
    python "$python_path" --entity "$entity" --project "$wandb_project_name" --target_path "$t_path" > "wandb_id.log"
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
        local_t_path=$(echo ${local_t_path} | sed -e "s/[\r\n]\+//g")
        # echo "$local_wandb_id $local_t_path"
        tmp_tf_log=$(find "$local_t_path" -maxdepth 1 -mindepth 1 -name "events.*")
        tf_num=$(echo "$tmp_tf_log" | wc -l)
        mkdir -p "$local_t_path/tf_logs"
        for tmp_tf in ${tmp_tf_log};
        do
            cp "$tmp_tf" "$local_t_path/tf_logs"
        done
        if [ $tf_num -gt 1 ];then
            tf_log="$local_t_path/tf_logs"
        else
            tf_log="$local_t_path/tf_logs/$(basename "$tmp_tf_log")"
        fi
        if [ -e "$tf_log" ];then
            wandb sync -p "$wandb_project_name" --id "$local_wandb_id" "$tf_log" 
        fi
        wandb_id_list+="$local_wandb_id "
    done
    python "$python_path" --entity "$entity" --project "$wandb_project_name" --target_path "$t_path" --upload --id $wandb_id_list
done

popd
