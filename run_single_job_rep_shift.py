#!/usr/bin/env python
# coding: utf-8
import os

import fire
import numpy as np
import torch
from fastprogress.fastprogress import master_bar, progress_bar
from tqdm import tqdm

import evaluation
from gridworld_with_door import MazeEnvironment
from rep_shift.fpp_agent import RNNAgent as FPPAgent
from rep_shift.rnn_agent import RNNAgent as RNNAgent
from rep_shift.rnn_agent_burn import RNNAgent as RNNAgentBurn
from rep_shift.rnn_agent_zero import RNNAgent as RNNAgentZero


def run_episode(env, agent, state_visits=None, keep_history=False):
    is_terminal = False
    sum_of_rewards = 0
    step_count = 0
    
    obs = env.env_start(keep_history=keep_history)
    action = agent.agent_start(obs)
    
    if state_visits is not None:
        state_visits[obs[0]] += 1

    while not is_terminal:
        reward, obs, is_terminal = env.env_step(action)
        print(agent.steps,end='\r')
        sum_of_rewards -= 1
        step_count += 1
        state = obs
        if step_count == 500:
            agent.agent_end(reward, state, append_buffer=False)
            break
        elif is_terminal:
            agent.agent_end(reward, state, append_buffer=True)
        else:
            action = agent.agent_step(reward, state)

        if state_visits is not None:
            state_visits[state[0]] += 1
    
    if keep_history:
        history = env.history
        env.env_cleanup()
        return sum_of_rewards, history
    else:
        return sum_of_rewards

agents = {
    # 0
    "RNN": RNNAgent,
    # 1
    "Zero": RNNAgentZero,
    # 2
    "Burn": RNNAgentBurn,
    # 3
    "FPP": FPPAgent,
}


def to_list(tups):
    return [list(x) for x in tups]


def get_env_info(n):
    m = (n - 1) // 2
    k = n - m
    return {
        "maze_dim": [n, n], 
        "start_state": [0, n-1], 
        "end_state": [n-1, n-1],
        "obstacles": to_list([*zip([m]*k,range(m,n)), *zip(range(m+1,n), [m]*(k-1))]),
        "doors": {tuple([m-1,m-1]):to_list([*zip([m]*k,range(m+1,n))])},
    }

def get_env_info_for_room(n,a=1):
    m = (n - 1) // 2
    k = n - m
    return {
        "maze_dim": [n, n], 
        "start_state": [0, n-1], 
        "end_state": [n-1, n-1],
        "obstacles": to_list([*zip([m]*k,range(m,n)), *zip(range(m+1,n), [m]*(k-1))]),
        "doors": {tuple([m-a,m-a]):to_list([*zip([m]*k,range(m+1,n))])},
    }


envs = {
    'Grid-World': MazeEnvironment,
}
agent_infos = {
    "RNN": {"step_size": 1e-3},
    "Burn": {"step_size": 1e-3},
    "Zero": {"step_size": 1e-3},
    "FPP": {'beta': 1, 'step_size': 1e-3},
}


env_infos = {
#     "simp": {
#         "maze_dim": [7, 7], 
#         "start_state": [0, 0], 
#         "end_state": [6, 6],
#     }, 
#     'obs': {
#         "maze_dim": [7, 7], 
#         "start_state": [0, 0], 
#         "end_state": [6, 6],
#         "obstacles": [[3, 3], [3, 5], [3, 6], [4, 3], [5, 3], [6, 3]],
#     },
#     'DoorWorldHallway': {
#         "maze_dim": [1, 7], 
#         "start_state": [0, 2], 
#         "end_state": [0, 0],
#         "obstacles":[[0,1]],
#         "doors": {tuple([0,6]):[[0,1]]},
#     },
    'DoorWorldWide1': {
        "maze_dim": [7, 7], 
        "start_state": [0, 6], 
        "end_state": [6, 6],
        "obstacles": [[3, 3], [3, 4], [3, 5], [3, 6], [4, 3], [5, 3], [6, 3]],
        "doors": {tuple([2,2]):[[3, 3], [3, 4], [3, 5], [3, 6]]},
    },
    'DoorWorldWide2': {
        "maze_dim": [7, 7], 
        "start_state": [0, 6], 
        "end_state": [6, 6],
        "obstacles": [[3, 3], [3, 4], [3, 5], [3, 6], [4, 3], [5, 3], [6, 3]],
        "doors": {tuple([1,1]):[[3, 3], [3, 4], [3, 5], [3, 6]]},
    },
    'DoorWorldWide3': {
        "maze_dim": [7, 7], 
        "start_state": [0, 6], 
        "end_state": [6, 6],
        "obstacles": [[3, 3], [3, 4], [3, 5], [3, 6], [4, 3], [5, 3], [6, 3]],
        "doors": {tuple([0,0]):[[3, 3], [3, 4], [3, 5], [3, 6]]},
    },
    'DoorWorldWide3.5': {
        "maze_dim": [7, 7], 
        "start_state": [0, 6], 
        "end_state": [6, 6],
        "obstacles": [[3, 3], [3, 4], [3, 5], [3, 6], [4, 3], [5, 3], [6, 3]],
        "doors": {tuple([6,0]):[[3, 3], [3, 4], [3, 5], [3, 6]]},
    },
    'DoorWorldWide4': {
        "maze_dim": [11, 11], 
        "start_state": [0, 10], 
        "end_state": [10, 10],
        "obstacles": to_list([*zip([5]*6,range(5,11)), *zip(range(6,11), [5]*5)]),
        "doors": {tuple([4,4]):to_list([*zip([5]*6,range(6,11))])},
    },
#     'DoorWorldWide5': get_env_info(13),
#     'DoorWorldWide6': get_env_info(15),
#     'DoorWorldWide7': get_env_info_for_room(13,2),
#     'DoorWorldWide8': get_env_info_for_room(13,4),
#     'DoorWorldWide9': get_env_info_for_room(15,2),
#     'DoorWorldWide10': get_env_info_for_room(15,3),
#     'DoorWorldWide11': get_env_info_for_room(15,4),

}


# ### Train
def train(agent_idx, env_name="DoorWorldWide3", T=10, lr=1e-3, beta=1, hidden_size=50, alpha=0.1):
    all_reward_sums = {} # Contains sum of rewards during episode
    all_state_visits = {} # Contains state visit counts during the last 10 episodes
    all_history = {}


    num_runs = 1
    num_episodes = 1
    Environment = envs['Grid-World']
    env_info = env_infos[env_name]

    if env_name not in all_reward_sums:
        all_reward_sums[env_name] = {}
        all_state_visits[env_name] = {}
        print(env_name)
    algorithm = list(agents.keys())[agent_idx]

    evaluation.init(algorithm)
    print(f"start training for {algorithm}")
    all_reward_sums[env_name][algorithm] = []
    all_state_visits[env_name][algorithm] = []
    for run in tqdm(range(num_runs)):
        agent = agents[algorithm]()
        env = Environment()
        env.env_init(env_info)
        # print(env_info)
        agent_info = {"num_actions": 4, "num_states": env.cols * env.rows, "epsilon": .1, "discount": .9, "T": T, "step_size": lr, "beta": beta, "hidden_size": hidden_size, "alpha": alpha} 
        agent_info["seed"] = run
        np.random.seed(run)
        agent.agent_init(agent_info)
        if agent_idx == 8:
            print(f"T:{agent.T}; learning rate: {agent.step_size}; hidden size: {agent.hidden_size}")
        elif agent_idx == 10:
            print(f"T:{agent.T}; learning rate: {agent.step_size}; alpha: {agent.alpha}")
        else:
            print(f"T:{T}; learning rate: {agent.step_size}")
        reward_sums = []
        state_visits = np.zeros(env.cols * env.rows)
        epsilon = 1
        for episode in range(num_episodes):
            print(f"episode {episode}",end='\r')
            agent.epsilon = epsilon
            if episode < num_episodes - 10:
                sum_of_rewards = run_episode(env, agent) 
            else: 
                # Runs an episode while keeping track of visited states and history
                sum_of_rewards, history = run_episode(env, agent, state_visits, keep_history=True)
                all_history.setdefault(env_name, {}).setdefault(algorithm, []).append(history)
            epsilon *= 0.99
            reward_sums.append(sum_of_rewards)

        all_reward_sums[env_name].setdefault(algorithm, []).append(reward_sums)
        all_state_visits[env_name].setdefault(algorithm, []).append(state_visits)

    file_path = f'metrics/{env_name}/{T}/{agent.step_size}/all_reward_sums_{agent_idx}.torch'
    rep_loss_path = f'metrics/{env_name}/{T}/{agent.step_size}/rep_loss_{agent_idx}.torch'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(all_reward_sums, file_path)
    torch.save(evaluation.metrics.rep_loss, rep_loss_path)


if __name__ == '__main__':
    fire.Fire(train)
