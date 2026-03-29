from env import EmailEnv
import numpy as np

env = EmailEnv()

episodes = 20

def smart_policy(state):
    is_urgent, is_work, is_spammy = state

    if is_spammy == 1:
        return 2  # Spam
    elif is_urgent == 1 or is_work == 1:
        return 0  # Important
    else:
        return 1  # Normal

for episode in range(episodes):
    state = env.reset()

    action = smart_policy(state)

    next_state, reward, done, _ = env.step(action)

    print(f"Episode {episode+1}")
    print("State:", state)
    print("Action:", action)
    print("Reward:", reward)
    print("-" * 30)