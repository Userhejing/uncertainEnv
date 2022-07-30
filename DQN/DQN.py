import torch
import torch.nn as nn
import gym
import sys, copy, random, pickle, os
import numpy as np
from collections import deque
from gym_custom.Air_conditioner.envs.air_conditioner import AirconditionerEnv
from common.Neural_network import Mlp_Network
from common.utils import Soft_Update, get_device, same_seeds, Categorical_action, Epsilon_greedy_action
from common.Plots import plot_rewards, plot_success
import time

if __name__ == "__main__":
    time_begin = time.time()
    # initialize an asynchronous parallel environment 
    env_name = 'AirconditionerEnv0'
    algorithm_name = 'DQN'
    env_num = 6
    env_seed = 110
    same_seeds(env_seed)
    envs = gym.vector.AsyncVectorEnv([(lambda : AirconditionerEnv())\
                                         for _ in range(env_num)])
    # envs.seed(same_seeds)
    obs_dim = gym.spaces.utils.flatdim(envs.single_observation_space)
    act_dim = gym.spaces.utils.flatdim(envs.single_action_space)
    
    # build network architecture
    device = 'cpu'
    # device = get_device()
    neuron_nums = [obs_dim, 256, 128, act_dim]
    Q1_Network = Mlp_Network(neuron_nums=neuron_nums).to(device)
    Q2_Network = Mlp_Network(neuron_nums=neuron_nums).to(device)
    Q2_Network.requires_grad_(requires_grad=False)  # Q2 doesn't need backpropagation
    Soft_Update(Q1_Network, Q2_Network, 0)  # set tau=0 to make Q2 copy Q1
    # set up optimazation parameters
    learning_rate = 1e-3
    batch_size = 64
    max_episode_num = 1.5e4
    gamma = 0.99
    Train_num = 3
    experience_replay_size = int(1e6)
    experience_replay = deque(maxlen=experience_replay_size)
    replay_start_size = int(1e3)
    Use_Epsilon_Greedy = False
    if Use_Epsilon_Greedy:
        epsilon_max = 1
        epsilon_min = 0.1
        epsilon = epsilon_max
    else:
        Use_Categorical = True
    Loss_fn = nn.MSELoss()
    Q1_optimizer=torch.optim.Adam(Q1_Network.parameters(), lr=learning_rate)

    # model training
    Q1_Network.train()
    episode_index = 0
    reward_log = [0 for _ in range(env_num)]  # attention, don't use tensor here.
    Logging = {}
    Logging['rewards'] = []
    Logging['success'] = []
    Logging['loss'] = []
    Logging['Q'] = []
    observations_last = envs.reset(seed=env_seed)
    observations_last = torch.from_numpy(observations_last).float()
    while episode_index < max_episode_num:
        # get trajectory and save them into experience replay.
        with torch.no_grad():
            observations = observations_last.to(device)
            if len(observations.shape) == 1:
                observations = observations.unsqueeze(dim=0)
            Q1_value = Q1_Network(observations)
            if Use_Epsilon_Greedy:
                actions = Epsilon_greedy_action(Q_value=Q1_value.cpu(), epsilon=epsilon)
            elif Use_Categorical:
                actions = Categorical_action(logits=Q1_value.cpu())
            else:
                raise NotImplementedError
            observations_next, rewards, dones, infos = envs.step(actions)
            observations_next = torch.from_numpy(observations_next).float()
            for i in range(env_num):
                reward_log[i] += rewards[i]
                if dones[i] == False:
                    experience_replay.append((
                        observations_last[i,:],
                        actions[i],
                        rewards[i],
                        observations_next[i,:],  
                        int(dones[i])
                    ))
                else:
                    experience_replay.append((
                        observations_last[i,:],
                        actions[i],
                        rewards[i],
                        # due to the autoreset !
                        torch.from_numpy(infos[i]['terminal_observation']).float(),
                        int(dones[i])
                    ))
                    episode_index += 1
                    Logging['rewards'].append(reward_log[i])
                    Logging['success'].append(infos[i]['success'])
                    reward_log[i] = 0
                    if episode_index % 50 == 0:
                        print(episode_index)
                    if Use_Epsilon_Greedy:
                        epsilon -= (epsilon_max - epsilon_min) / max_episode_num
            observations_last = observations_next
        if len(experience_replay) <= replay_start_size:
            continue

        # DQN
        for _ in range(Train_num):
            exp_replay = random.sample(experience_replay, batch_size)
            obs_t_1 = torch.stack([exp_replay_[0] for exp_replay_ in exp_replay]).to(device)
            a_t_1 = torch.from_numpy(np.array([exp_replay_[1] for exp_replay_ in exp_replay])).long().to(device)
            r_t = torch.from_numpy(np.array([exp_replay_[2] for exp_replay_ in exp_replay])).float().to(device)
            obs_t = torch.stack([exp_replay_[3] for exp_replay_ in exp_replay]).to(device)
            done_t = torch.from_numpy(np.array([exp_replay_[4] for exp_replay_ in exp_replay])).long().to(device)

            Q1_value = Q1_Network(obs_t_1)
            Logging['Q'].append(Q1_value.detach().cpu().numpy())
            with torch.no_grad():
                Q2_value = Q2_Network(obs_t)
                Q2_max_value, _ = torch.max(Q2_value,dim=1)
                Target_Q_value = r_t + gamma * Q2_max_value * (1-done_t)
            Q1_target = Q1_value.gather(dim=1,index=a_t_1.view(-1,1))
            loss = Loss_fn(Q1_target, Target_Q_value.view(-1,1))
            Logging['loss'].append(loss.detach().cpu().item())
            Q1_optimizer.zero_grad()
            loss.backward()
            Q1_optimizer.step()
            Soft_Update(Q1_Network, Q2_Network, 0.99)
            
    envs.close()

    time_over = time.time()
    print(f'Clock time: {time_over-time_begin}')
    # Logging
    log_prefix = f'../log/{env_name}/'
    if not os.path.exists(log_prefix):
        os.mkdir(log_prefix)
    with open(log_prefix + f'Log{env_seed}-{algorithm_name}.txt', 'wb') as fb:
        pickle.dump(Logging, fb)
    torch.save(Q1_Network, log_prefix + 'DQN.pth')
    plot_rewards(Logging['rewards'])
    plot_success(Logging['success'], 50)
