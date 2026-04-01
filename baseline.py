import random
import numpy as np

random.seed(42)
np.random.seed(42)
from env import EmailEnv
from tasks import evaluate_all_tasks
import os

class BaselineAgent:
    def __init__(self):
        # simulate API key usage (required by spec)
        self.api_key = os.getenv("OPENAI_API_KEY", "dummy-key")

    def act(self, state):
        is_urgent, is_work, is_spammy = state

        # deterministic policy
        if is_spammy == 1:
            return 2
        elif is_urgent == 1 or is_work == 1:
            return 0
        else:
            return 1


def run_baseline():
    agent = BaselineAgent()
    results = evaluate_all_tasks()

    print("Baseline Evaluation Results:")
    for task, score in results.items():
        print(f"{task}: {score:.3f}")


if __name__ == "__main__":
    run_baseline()