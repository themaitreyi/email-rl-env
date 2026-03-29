from env import EmailEnv

env = EmailEnv()

state = env.reset()
print("Initial State:", state)

action = env.action_space.sample()
print("Action taken:", action)

next_state, reward, done, _ = env.step(action)

print("Next State:", next_state)
print("Reward:", reward)
print("Done:", done)