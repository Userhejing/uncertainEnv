import sys, time
import torch
import numpy as np
import matplotlib.pyplot as plt
from gym_custom.Air_conditioner.envs.air_conditioner import AirconditionerEnv

env_name = 'AirconditionerEnv0'
seed = 110
Q1_network = torch.load(f'../log/{env_name}/DQN.pth').to('cpu')
env = AirconditionerEnv()
obs = env.reset()

env.Tem_in, env.Tem_air, env.Hum_in, env.Hum_air = 0.2, 0.2, 0.2, 0.2
env.Tem_up, env.Tem_low, env.Hum_up, env.Hum_low=0.7,0.5,0.6,0.5
obs=env._get_obs()

obs_log = [obs]
done = False
while not done:
    env.render()
    time_begin = time.time()
    Q_value = Q1_network(torch.from_numpy(obs).view(1,-1).float())
    act = torch.argmax(Q_value,dim=1).squeeze().numpy()
    time_1step = time.time() - time_begin
    # print(f'time = {time_1step}')
    obs, r, done, info = env.step(int(act))
    obs_log.append(obs)
env.close()
obs_log = np.array(obs_log)

# plot Tem.
plt.figure(figsize=(10,7))
ax = plt.gca()
ax.plot(obs_log[:, 0],label='Tem_in')
ax.plot(obs_log[:, 4],label='Tem_up')
ax.plot(obs_log[:, 5],label='Tem_down')
ax.set_title('Tem curve')
ax.set_xlabel('step')
ax.set_ylabel('Tem')
ax.legend()
plt.show()
# plot Hum.
plt.figure(figsize=(10,7))
ax = plt.gca()
ax.plot(obs_log[:, 2],label='Hum_in')
ax.plot(obs_log[:, 6],label='Hum_up')
ax.plot(obs_log[:, 7],label='Hum_down')
ax.set_title('Hum curve')
ax.set_xlabel('step')
ax.set_ylabel('Hum')
ax.legend()
plt.show()