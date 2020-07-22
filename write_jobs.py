import argparse
import subprocess
from itertools import product

import numpy as np

save_in_folder = "scripts"
env = "DoorWorldWide1"



# def write_run_all_jobs(count, init_count=0):
#     """
#     generate a .sh file to run all experiments
#     """
#     with open('{}run_all.sh'.format(save_in_folder), 'w') as f:
#         f.write('#!/usr/bin/env bash\nchmod +x ./{}tasks_*.sh\n'.format(save_in_folder))
#         for i in range(count-init_count):
#             f.write('./{}tasks_{}.sh &> {}log{}.txt &\n'.format(save_in_folder, init_count+i, save_in_folder, i))
#             if (i+1) % 10 == 0:
#                 f.write('wait\n')


# def write_jobs(all_comb, count, verbose):
#     # init_count = count
#     for agent_idxes, env_name, T, lr, hidden_size, alpha in all_comb:
#         cmd = f"python run_single_job_trace.py --agent_idxes='{agent_idxes}' --env_name={env_name} --T={T} --lr={lr} --hidden_size={hidden_size} --alpha={alpha}\n"
#         with open("{}/tasks_{}.sh".format(save_in_folder, count), 'w') as f:
#             f.write(new_cmd)
#         if verbose == "True":
#             print(count, new_cmd)
#         count += 1
#     return count

template = '''
#!/bin/bash
#SBATCH --account=def-whitem
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

#SBATCH --mail-user=xli@ualberta.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --job-name="{}"
'''

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

def write_single_job(params, count, verbose=True, gpu=False):
    cmd = f"python -m run_single_job_v1 "
    for k, v in params.items():
        cmd += f"--{k}='{v}' " 
    # temp_str = template.format(cmd)
    # if gpu:
    #     temp_str += "#SBATCH --gres=gpu:1\n"
    cmd += "\n"
    with open("{}/tasks_{}.sh".format(save_in_folder, count), 'w+') as f:
        # f.write(temp_str)
        f.write(cmd)
    if verbose == "True":
        print(count, cmd)
    count += 1
    return count

def write_esn_jobs(count=0):
    experiment_settings = {
        "env_name":[env],
        "agent_idx": [8],
        "T":[5, 10, 20],
        "lr":[3e-3, 1e-3, 3e-4, 1e-4],
        "hidden_size": [50, 200, 400, 800, 1600]
    }

    list_of_params = product_dict(**experiment_settings)
    for params in list_of_params:
        count = write_single_job(params, count)
    return count

def write_regular_jobs(count=0):
    experiment_settings = {
        "env_name":[env],
        "agent_idx": [2,7],
        "T":[5, 10, 20],
        "lr":[3e-3, 1e-3, 3e-4, 1e-4],
    }

    list_of_params = product_dict(**experiment_settings)
    for params in list_of_params:
        count = write_single_job(params, count, gpu=True)
    return count

def write_uoro_jobs(count=0):
    experiment_settings = {
        "env_name":[env],
        "agent_idx": [4],
        "lr":[3e-3, 1e-3, 3e-4, 1e-4],
    }

    list_of_params = product_dict(**experiment_settings)
    for params in list_of_params:
        count = write_single_job(params, count)
    return count

def write_random_jobs(count=0):
    experiment_settings = {
        "env_name":[env],
        "agent_idx": [5],
        "lr":[1e-4],
    }

    list_of_params = product_dict(**experiment_settings)
    for params in list_of_params:
        count = write_single_job(params, count)
    return count

def write_dt_jobs(count=0,gpu=True):
    experiment_settings = {
        "env_name":[env],
        "agent_idx": [10],
        "T":[5, 10, 20],
        "lr":[3e-3, 1e-3, 3e-4, 1e-4],
        "alpha_list":[0.1, 0.2, 0.5, 0.8, 0.9]
    }

    list_of_params = product_dict(**experiment_settings)
    for params in list_of_params:
        count = write_single_job(params, count)
    return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default="True", type=str)
    parser.add_argument('--start', default=0, type=int)
    args = parser.parse_args()

    count = args.start

    count = write_regular_jobs(count)
    count = write_dt_jobs(count)
    count = write_uoro_jobs(count)
    count = write_esn_jobs(count)
    count = write_random_jobs(count)

    command = f'chmod +x ./{save_in_folder}/tasks_*.sh'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()
