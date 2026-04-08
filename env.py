import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from dataclasses import dataclass


# Typed observation model
@dataclass
class Observation:
    is_urgent: float
    is_work: float
    is_spammy: float


# Typed action model
@dataclass
class Action:
    label: int  # 0, 1, 2


class EmailEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )

        self.state_vec = None
        self.correct_label = None
        self.current_difficulty = None

        self.step_count = 0
        self.max_steps = 3

        random.seed(42)
        np.random.seed(42)

    def generate_email(self):
        difficulty = self.current_difficulty or random.choice(
            ["easy", "medium", "hard"]
        )

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

            state = np.array(
                [is_urgent, is_work, is_spammy], dtype=np.float32
            )

            if is_spammy:
                label = 2
            elif is_urgent or is_work:
                label = 0
            else:
                label = 1

        else:  # hard
            state = np.array(
                [
                    random.choice([0, 1]),
                    random.choice([0, 1]),
                    random.choice([0, 1]),
                ],
                dtype=np.float32,
            )

            if state[2] == 1 and state[0] == 1:
                label = 1
            elif state[2] == 1:
                label = 2
            elif state[0] == 1 or state[1] == 1:
                label = 0
            else:
                label = 1

        return state, label

    def state(self):
        return Observation(
            is_urgent=float(self.state_vec[0]),
            is_work=float(self.state_vec[1]),
            is_spammy=float(self.state_vec[2]),
        )

    # ✅ FIXED RESET (gymnasium compliant)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.state_vec, self.correct_label = self.generate_email()

        return self.state_vec.astype(np.float32), {}

    # ✅ FIXED STEP (gymnasium compliant)
    def step(self, action):
        self.step_count += 1

        # Safe action handling
        try:
            if isinstance(action, Action):
                action = action.label
            action = int(action)
        except:
            action = 1  # fallback safe action

        # Reward logic
        if action == self.correct_label:
            reward = 1.0
        else:
            if self.correct_label == 2 and action == 1:
                reward = 0.3
            elif self.correct_label == 0 and action == 1:
                reward = 0.5
            else:
                reward = 0.0

        terminated = self.step_count >= self.max_steps
        truncated = False

        if not terminated:
            self.state_vec, self.correct_label = self.generate_email()

        info = {
            "difficulty": self.current_difficulty,
            "step": self.step_count,
        }

        return (
            self.state_vec.astype(np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )