#!/bin/bash
agent_list=(0 1 2 3)
T_list=(5 10 20)
lr_list=(3e-3 1e-3 3e-4 1e-4)
for A in ${agent_list[@]}; do
    for T in ${T_list[@]}; do
        for lr in ${lr_list[@]}; do
            sbatch --account=def-whitem --nodes=1 --mem=4G --time=24:00:00 --mail-user=xli@ualberta.ca --mail-type=END --mail-type=FAIL --job-name=TBPTT_experiment_${T}_${lr} --wrap="python run_single_job_rep_shift.py --agent_idx=$A --T=$T --lr=$lr"
            sleep 1
        done
    done
done
