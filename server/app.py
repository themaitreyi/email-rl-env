from fastapi import FastAPI
from env import EmailEnv

app = FastAPI()
env = EmailEnv()

def main():
    return app

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)