import os
from openai import OpenAI
from tasks import evaluate_all_tasks

# MUST use these (provided by validator automatically)
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

class InferenceAgent:
    def act(self, state):
        try:
            prompt = f"""
You are an email classifier.

State:
urgent: {state[0]}
work: {state[1]}
spam: {state[2]}

Return ONLY one number:
0 = important
1 = normal
2 = spam
"""

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            output = response.choices[0].message.content.strip()
            return int(output)

        except:
            return 1  # safe fallback

def run_inference():
    agent = InferenceAgent()
    results = evaluate_all_tasks(agent)

    print("Inference Results:")
    for task, score in results.items():
        print(f"{task}: {score:.3f}")

if __name__ == "__main__":
    run_inference()