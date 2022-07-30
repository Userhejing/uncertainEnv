from gym.utils.env_checker import check_env
import gym_custom, gym
from gym_custom.Air_conditioner.envs.air_conditioner import AirconditionerEnv

env = AirconditionerEnv()
obs = env.reset()
print(obs)
act = env.action_space.sample()
obs,r,done,info = env.step(act)
print(act)
print(obs,r,done,info)
act = env.action_space.sample()
obs,r,done,info = env.step(act)
print(act)
print(obs,r,done,info)