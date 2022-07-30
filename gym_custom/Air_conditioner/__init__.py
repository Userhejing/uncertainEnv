from gym.envs.registration import register

register(
    id='gym_custom/Air_conditioner-v0',
    entry_point='Air_conditioner.envs:ElectromobileEnv',
    max_episode_steps=100,
)