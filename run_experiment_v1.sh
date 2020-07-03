#!/bin/bash

T_list=(5 10 20)
lr_list=(3e-3 1e-3 3e-4 1e-4)
for T in ${T_list[@]}; do
    for lr in ${lr_list[@]}; do
        sbatch --account=def-whitem --gres=gpu:1 --nodes=1 --mem=4G --time=24:00:00 --mail-user=xli@ualberta.ca --mail-type=END --mail-type=FAIL --job-name=TBPTT_experiment_${T}_${lr} --wrap="python run_single_job.py --agent_idxes='[2]' --T=$T --lr=$lr"
        sleep 1
    done
done
