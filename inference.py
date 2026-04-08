import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")

def safe_post(endpoint, payload=None):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] {endpoint} failed: {e}")
        return None


def run_inference():
    print("[START] Inference started")

    results = {}
    difficulties = ["easy", "medium", "hard"]

    for difficulty in difficulties:
        total_reward = 0
        steps = 5

        for _ in range(steps):
            reset_data = safe_post("/reset")
            if not reset_data or "state" not in reset_data:
                continue

            state = reset_data["state"]

            # SAFE unpack (no crash)
            try:
                is_urgent, is_work, is_spam = state
            except:
                is_urgent, is_work, is_spam = 0, 0, 0

            # RULE-BASED AGENT
            if is_spam == 1:
                action = 2
            elif is_urgent == 1 or is_work == 1:
                action = 0
            else:
                action = 1

            step_data = safe_post("/step", {"action": action})
            if not step_data:
                continue

            reward = step_data.get("reward", 0)
            total_reward += reward

            print(f"[STEP] {difficulty} reward={reward}")

        # SAFE scoring
        try:
            score = total_reward / (steps * 10)
        except:
            score = 0.0

        score = max(0.0, min(1.0, score))
        results[difficulty] = round(score, 3)

    print("[END] Inference completed")
    print(results)

    return results


if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")