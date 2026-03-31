from env import EmailEnv
import numpy as np

env = EmailEnv()

def smart_policy(state):
    is_urgent, is_work, is_spammy = state

    if is_spammy == 1:
        return 2
    elif is_urgent == 1 or is_work == 1:
        return 0
    else:
        return 1

episodes = 50
total_reward = 0

for _ in range(episodes):
    state = env.reset()
    action = smart_policy(state)
    _, reward, _, _ = env.step(action)
    total_reward += reward

average_score = total_reward / episodes

print("Episodes:", episodes)
print("Total Reward:", total_reward)
print("Average Score:", average_score)