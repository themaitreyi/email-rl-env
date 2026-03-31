from env import EmailEnv

def run_demo():
    env = EmailEnv()

    episodes = 10
    total_reward = 0

    for i in range(episodes):
        state = env.reset()

        is_urgent, is_work, is_spammy = state

        if is_spammy == 1:
            action = 2
        elif is_urgent == 1 or is_work == 1:
            action = 0
        else:
            action = 1

        _, reward, _, _ = env.step(action)
        total_reward += reward

    avg_score = total_reward / episodes

    return f"Average Score over {episodes} episodes: {avg_score}"

if __name__ == "__main__":
    print(run_demo())