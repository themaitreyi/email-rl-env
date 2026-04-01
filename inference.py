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
        # REQUIRED ENV VARIABLES (STRICT)
        self.api_base_url = os.environ.get("API_BASE_URL")
        self.model_name = os.environ.get("MODEL_NAME")
        self.hf_token = os.environ.get("HF_TOKEN")

        # Validation (ensures variables exist but doesn't break execution)
        if self.api_base_url is None:
            raise ValueError("API_BASE_URL not set")
        if self.model_name is None:
            raise ValueError("MODEL_NAME not set")
        if self.hf_token is None:
            raise ValueError("HF_TOKEN not set")

    def act(self, state):
        is_urgent, is_work, is_spammy = state

        # deterministic policy (baseline)
        if is_spammy == 1:
            return 2
        elif is_urgent == 1 or is_work == 1:
            return 0
        else:
            return 1


def run_inference():
    agent = InferenceAgent()

    results = evaluate_all_tasks()

    print("Inference Results:")
    for task, score in results.items():
        print(f"{task}: {score:.3f}")


if __name__ == "__main__":
    run_inference()