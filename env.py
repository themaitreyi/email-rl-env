import gym
from gym import spaces
import numpy as np
import random

class EmailEnv(gym.Env):
    def __init__(self):
        super(EmailEnv, self).__init__()

        # Actions: 0 = Important, 1 = Normal, 2 = Spam
        self.action_space = spaces.Discrete(3)

        # State: [is_urgent, is_work, is_spammy]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.state = None
        self.correct_label = None

    def generate_email(self):
        # Random email features
        is_urgent = random.choice([0, 1])
        is_work = random.choice([0, 1])
        is_spammy = random.choice([0, 1])

        state = np.array([is_urgent, is_work, is_spammy], dtype=np.float32)

        # Define correct label
        if is_spammy:
            label = 2  # Spam
        elif is_urgent or is_work:
            label = 0  # Important
        else:
            label = 1  # Normal

        return state, label

    def reset(self):
        self.state, self.correct_label = self.generate_email()
        return self.state

    def step(self, action):
        # Reward logic
        if action == self.correct_label:
            reward = 10
        else:
            reward = -5

        done = True

        return self.state, reward, done, {}