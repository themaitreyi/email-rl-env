from fastapi import FastAPI
from env import EmailEnv

app = FastAPI()

env = EmailEnv()


@app.post("/reset")
def reset():
    state, _ = env.reset()
    return {"state": state.tolist()}


@app.post("/step")
def step(action: int):
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return {
        "state": state.tolist(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/")
def root():
    return {"status": "running"}