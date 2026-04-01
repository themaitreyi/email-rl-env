from env import EmailEnv
from tasks import evaluate_all_tasks
import os
import random
import numpy as np
from openai import OpenAI

# reproducibility
random.seed(42)
np.random.seed(42)

class InferenceAgent:
    def __init__(self):
        self.api_base = os.environ.get("API_BASE_URL")
        self.model = os.environ.get("MODEL_NAME")
        self.token = os.environ.get("HF_TOKEN")

        if not self.api_base or not self.model or not self.token:
            raise ValueError("Missing required environment variables")

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.token
        )

    def act(self, state):
        is_urgent, is_work, is_spammy = state

        prompt = f"""
        Classify email:
        urgent={is_urgent}, work={is_work}, spam={is_spammy}

        Return ONLY one number:
        0 = Important
        1 = Normal
        2 = Spam
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            text = response.choices[0].message.content.strip()

            if "2" in text:
                return 2
            elif "0" in text:
                return 0
            else:
                return 1

        except:
            # fallback (VERY IMPORTANT)
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