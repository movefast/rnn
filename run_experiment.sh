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


echo "[$1]"
python run_long_dep_job.py --agent_idxes="[$1]"
