import sys, pickle
from common.Plots import plot_rewards, plot_success

env_name = 'AirconditionerEnv0'
env_seed = 110
algorithm_name = 'DQN'
with open(f'../log/{env_name}/Log{env_seed}-{algorithm_name}.txt', 'rb') as fb:
    Logging_dqn = pickle.load(fb)
plot_rewards(Logging_dqn['rewards'])
plot_success(Logging_dqn['success'], 50)
plot_success(Logging_dqn['loss'], 50, title='loss')
plot_success(Logging_dqn['loss'], 50, title='Q1')
