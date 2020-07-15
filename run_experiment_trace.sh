#!/bin/bash

T_list=(5 10 20)
lr_list=(3e-3 1e-3 3e-4 1e-4)
alpha_list=(0.1 0.2 0.5 0.8 0.9)
for T in ${T_list[@]}; do
    for lr in ${lr_list[@]}; do
        for alpha in ${alpha_list[@]}; do
            sbatch --account=def-whitem --gres=gpu:1 --nodes=1 --mem=4G --time=24:00:00 --mail-user=xli@ualberta.ca --mail-type=END --mail-type=FAIL --job-name=Trace_experiment_${T}_${lr} --wrap="python run_single_job_trace.py --agent_idxes='[11]' --T=$T --lr=$lr --alpha=$alpha"
            sleep 1
        done
    done
done
