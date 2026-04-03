from fastapi import FastAPI
from env import EmailEnv

app = FastAPI()

env = EmailEnv()


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state.tolist()}


@app.post("/step")
def step(action: int = 0):
    next_state, reward, done, info = env.step(action)
    return {
        "state": next_state.tolist(),
        "reward": reward,
        "done": done
    }