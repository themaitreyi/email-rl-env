from fastapi import FastAPI
from env import EmailEnv

app = FastAPI()

env = EmailEnv()


@app.api_route("/reset", methods=["GET", "POST"])
def reset():
    state = env.reset()
    return {"state": state.tolist()}


@app.api_route("/step", methods=["GET", "POST"])
def step(action: int = 0):
    next_state, reward, done, info = env.step(action)
    return {
        "state": next_state.tolist(),
        "reward": reward,
        "done": done
    }


@app.get("/")
def root():
    return {"message": "Email RL Environment is running"}