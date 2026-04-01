from env import EmailEnv

# Baseline policy (deterministic)
def baseline_policy(state):
    is_urgent, is_work, is_spammy = state

    if is_spammy == 1:
        return 2
    elif is_urgent == 1 or is_work == 1:
        return 0
    else:
        return 1


class TaskGrader:
    def __init__(self, difficulty, episodes=30):
        self.difficulty = difficulty
        self.episodes = episodes

    def evaluate(self):
        env = EmailEnv()
        env.current_difficulty = self.difficulty

        total_reward = 0

        for _ in range(self.episodes):
            state = env.reset()
            action = baseline_policy(state)
            _, reward, _, _ = env.step(action)
            total_reward += reward

        # Normalize score between 0 and 1
        score = total_reward / self.episodes
        return round(score, 3)


def evaluate_all_tasks():
    results = {}

    for difficulty in ["easy", "medium", "hard"]:
        grader = TaskGrader(difficulty)
        score = grader.evaluate()
        results[difficulty] = score

    return results


if __name__ == "__main__":
    results = evaluate_all_tasks()

    print("Task Evaluation Results:")
    for task, score in results.items():
        print(f"{task}: {score}")