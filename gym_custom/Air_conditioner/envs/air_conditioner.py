import gym
from gym import spaces
import numpy as np

class AirconditionerEnv(gym.Env):
    metadata = {"render_modes": [None], "render_fps": 1}

    def __init__(self):
        self.max_episode_steps = 100
        # Observations contains the Tem_in, Tem_air, Hum_in, Hum_air
        #                           Tem_up, Temp_low, Hum_up, Hum_low.
        # which sums to 11 dimensions.
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        # We have 9 actions, corresponding to "hold", "down" and "up" for Tem_air and Hum_air.
        self.action_space = spaces.Discrete(9)
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "hold" for Tem_air and "hold" for Hum_air etc.
        """
        self._action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([0, -0.1]),
            2: np.array([0, 0.1]),
            3: np.array([-0.1, 0]),
            4: np.array([-0.1, -0.1]),
            5: np.array([-0.1, 0.1]),
            6: np.array([0.1, 0]),
            7: np.array([0.1, -0.1]),
            8: np.array([0.1, 0.1])
        }

    def _get_obs(self):
        return np.array([self.Tem_in, self.Tem_air, self.Hum_in,\
            self.Hum_air, self.Tem_up, self.Tem_low, self.Hum_up, self.Hum_low])

    def _get_info(self):
        return {"demo": None}

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        # Choose the agent's location to be the original point.
        self.step_num = 0
        self.success = 0
        Tem_interval = 0.05
        Hum_interval = 0.05
        rand_cfg = np.random.rand()
        if rand_cfg < 0.5:
            # initial data belong to [0, 0.5], target belong to [0.5, 1]
            self.Tem_in = np.random.rand()*0.5
            self.Tem_air = self.Tem_in
            self.Hum_in = np.random.rand()*0.5
            self.Hum_air = self.Hum_in

            self.Tem_target = np.random.rand()*0.5 + 0.5
            self.Tem_up = min([1.0, self.Tem_target+Tem_interval])
            self.Tem_low = max([0.5, self.Tem_target-Tem_interval])
            self.Hum_target = np.random.rand()*0.5 + 0.5
            self.Hum_up = min([1.0, self.Hum_target+Hum_interval])
            self.Hum_low = max([0.5, self.Hum_target-Hum_interval])
        else:
            # initial data belong to [0.5, 1], target belong to [0, 0.5]
            self.Tem_in = np.random.rand()*0.5 + 0.5
            self.Tem_air = self.Tem_in
            self.Hum_in = np.random.rand()*0.5 + 0.5
            self.Hum_air = self.Hum_in

            self.Tem_target = np.random.rand()*0.5
            self.Tem_up = min([0.5, self.Tem_target+Tem_interval])
            self.Tem_low = max([0.0, self.Tem_target-Tem_interval])
            self.Hum_target = np.random.rand()*0.5
            self.Hum_up = min([0.5, self.Hum_target+Hum_interval])
            self.Hum_low = max([0.0, self.Hum_target-Hum_interval])

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        self.step_num += 1
        # change the Tem_in and Hum_in
        Tem_delta = self.Tem_air - self.Tem_in
        Tem_in_step = np.abs(np.power(Tem_delta,2)) / 2.0
        Tem_in_step = np.clip(Tem_in_step, 0.001, 0.2)
        Hum_delta = self.Hum_air - self.Hum_in
        Hum_in_step = np.abs(np.power(Hum_delta,2)) / 2.0
        Hum_in_step = np.clip(Hum_in_step, 0.001, 0.2)
        if Tem_delta > 0:
            self.Tem_in += Tem_in_step
        elif Tem_delta < 0:
            self.Tem_in -= Tem_in_step
            
        if Hum_delta > 0:
            self.Hum_in += Hum_in_step
        elif Hum_delta < 0:
            self.Hum_in -= Hum_in_step
        self.Tem_in = np.clip(self.Tem_in, 0, 1)
        self.Hum_in = np.clip(self.Hum_in, 0, 1)

        # Map the action (element of {0,1,...,8}) to the direction change
        direction = self._action_to_direction[action]
        Tem_air_delta = direction[0]
        Hum_air_delta = direction[1]
        self.Tem_air = np.clip(self.Tem_air + Tem_air_delta, 0, 1)
        self.Hum_air = np.clip(self.Hum_air + Hum_air_delta, 0, 1)

        info = self._get_info()
        done = False
        if self.Tem_in >= self.Tem_low and self.Tem_in <= self.Tem_up and \
            self.Hum_in >= self.Hum_low and self.Hum_in <= self.Hum_up:
            reward = 1.0
            self.success += 1
        else:
            Tem_w, Hum_w = 0.8, 0.2
            reward_Tem, reward_Hum = 0, 0
            if self.Tem_in < self.Tem_low:
                reward_Tem = (self.Tem_in - self.Tem_low)
            elif  self.Tem_in > self.Tem_up:
                reward_Tem = (self.Tem_up - self.Tem_in)

            if self.Hum_in < self.Hum_low:
                reward_Hum = (self.Hum_in - self.Hum_low)
            elif  self.Hum_in > self.Hum_up:
                reward_Hum = (self.Hum_up - self.Hum_in)
            reward = Tem_w*reward_Tem + Hum_w*reward_Hum
        if self.step_num >= self.max_episode_steps:
            done = True
            info['success'] = self.success

        observation = self._get_obs()
        
        return observation, reward, done, info

    def render(self, mode=None):
        pass

    def close(self):
        pass