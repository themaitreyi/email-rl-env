from env import EmailEnv
from tasks import evaluate_all_tasks
import os
import random
import numpy as np

# reproducibility
random.seed(42)
np.random.seed(42)

class InferenceAgent:
    def __init__(self):
        self.api_base = os.getenv("API_BASE_URL", "dummy")
        self.model = os.getenv("MODEL_NAME", "dummy")
        self.token = os.getenv("HF_TOKEN", "dummy")

    def act(self, state):
        is_urgent, is_work, is_spammy = state

        if is_spammy == 1:
            return 2
        elif is_urgent == 1 or is_work == 1:
            return 0
        else:
            return 1


def run_inference():
    results = evaluate_all_tasks()

    print("Inference Results:")
    for task, score in results.items():
        print(f"{task}: {score:.3f}")


if __name__ == "__main__":
    run_inference()