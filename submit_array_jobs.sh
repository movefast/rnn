#!/bin/bash
#SBATCH --account=def-whitem
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --mail-user=xli@ualberta.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=Easy
./scripts/tasks_${SLURM_ARRAY_TASK_ID}.sh