#!/usr/bin/env bash
T_list=(5 10 20)
lr_list=("3e-3" "1e-3" "3e-4" "1e-4")
for T in ${T_list[@]}; do
    for lr in ${lr_list[@]}; do
        python run_single_job.py --agent_idxes="[2]" --T=$T --lr=$lr
    done
    wait
done
