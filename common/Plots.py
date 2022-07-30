import matplotlib.pyplot as plt
import pickle
import numpy as np
from typing import Sequence

def running_mean(x, N=50): # calculate the mean to make plot more smooth
    x = np.asarray(x)
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

def plot_rewards(
    rewards: Sequence,
    N: int = 50,
    label='DQN',
) -> None:
    
    plt.figure(figsize=(15,10))
    ax = plt.gca()
    ax.plot(running_mean(rewards, N),label=label)
    ax.set_title('reward curve')
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    plt.show()

def plot_success(
    success: Sequence,
    N: int = 1000,
    label='DQN',
    title='success'
) -> None:
    
    plt.figure(figsize=(15,10))
    ax = plt.gca()
    ax.plot(running_mean(success, N), label=label)
    ax.set_title(f'{title} curve')
    ax.set_xlabel('episode')
    ax.set_ylabel(f'{title}')
    plt.show()