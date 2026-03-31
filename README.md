# Smart Email Prioritization using Reinforcement Learning

## Problem Statement

Managing emails efficiently is a real-world problem. Important emails often get lost among spam and low-priority messages.

## Solution

We designed a Reinforcement Learning environment where an agent learns to classify emails into:

* Important
* Normal
* Spam

## Environment Design

### State Space

Each email is represented using 3 features:

* is_urgent
* is_work_related
* is_spammy

### Action Space

* 0 → Important
* 1 → Normal
* 2 → Spam

### Reward Function

* Correct classification → +10
* Incorrect classification → -5

## Goal

Train an agent to maximize correct classification of emails.

## How to Run

```bash
python test_env.py
```

## Demo Output

```bash
Episode 1
State: [1. 0. 0.]
Action: 0
Reward: 10
------------------------------

Episode 2
State: [0. 1. 0.]
Action: 0
Reward: 10
------------------------------
```

## Tech Stack

* Python
* Gym (Reinforcement Learning Environment)
* NumPy

## OpenEnv Integration

This project is designed to be compatible with Meta's OpenEnv framework by defining a custom reinforcement learning environment using Gym interface standards. The environment can be extended and integrated with OpenEnv for scalable agent training.

## Future Improvements

* Use real-world email datasets
* Train using Deep Q-Learning (DQN)
* Deploy using Hugging Face Spaces
