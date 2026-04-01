from fastapi import FastAPI
from env import EmailEnv

app = FastAPI()

env = EmailEnv()


@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state.tolist()}


@app.post("/step")
def step(action: int):
    next_state, reward, done, info = env.step(action)
    return {
        "state": next_state.tolist(),
        "reward": reward,
        "done": done
    }


@app.get("/")
def root():
    return {"message": "Email RL Environment is running"}