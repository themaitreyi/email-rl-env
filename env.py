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
        difficulty = random.choice(["easy", "medium", "hard"])

        if difficulty == "easy":
            is_spammy = random.choice([0, 1])
            if is_spammy:
                state = np.array([0, 0, 1], dtype=np.float32)
                label = 2
            else:
                state = np.array([1, 1, 0], dtype=np.float32)
                label = 0

        elif difficulty == "medium":
            is_urgent = random.choice([0, 1])
            is_work = random.choice([0, 1])
            is_spammy = random.choice([0, 1])

            state = np.array([is_urgent, is_work, is_spammy], dtype=np.float32)

            if is_spammy:
                label = 2
            elif is_urgent or is_work:
                label = 0
            else:
                label = 1

        else:  # hard
            state = np.array([
                random.choice([0, 1]),
                random.choice([0, 1]),
                random.choice([0, 1])
            ], dtype=np.float32)

            if state[2] == 1 and state[0] == 1:
                label = 1
            elif state[2] == 1:
                label = 2
            elif state[0] == 1 or state[1] == 1:
                label = 0
            else:
                label = 1

        return state, label

    def reset(self):
        self.state, self.correct_label = self.generate_email()
        return self.state

    def step(self, action):
        reward = 0

        # Perfect match
        if action == self.correct_label:
            reward = 1.0

        # Partial correctness
        else:
            if self.correct_label == 2 and action == 1:
                reward = 0.3
            elif self.correct_label == 0 and action == 1:
                reward = 0.5
            else:
                reward = 0.0

        done = True

        return self.state, reward, done, {}