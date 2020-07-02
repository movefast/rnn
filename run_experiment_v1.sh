#!/bin/bash

#SBATCH --account=def-whitem

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00

#SBATCH --mail-user=xli@ualberta.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --job-name=fpp_experiment

T_list=(5 10 20)
lr_list=(3e-3 1e-3 3e-4 1e-4)
for T in ${T_list[@]}; do
    for lr in ${lr_list[@]}; do
        sbatch -p serial_requeue -t 10 --mem=200 --wrap="python run_single_job.py --agent_idxes='[2]' --T=$T --lr=$lr"
        sleep 1
    done
done