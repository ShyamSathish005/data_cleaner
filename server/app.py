from fastapi import FastAPI, HTTPException, Body
from models import Action, Observation, ResetRequest, RewardOut
from env import DataCleaningEnv

app = FastAPI(title='Data Cleaning OpenEnv')
env = DataCleaningEnv()


@app.get('/')
def root():
    return {'status': 'ok', 'service': 'data-cleaning-openenv'}


@app.get('/health')
def health():
    return {'ok': True}

@app.post('/reset')
def reset(payload: ResetRequest = Body(default=ResetRequest(task_id='fix_types'))):
    task_id = payload.task_id or 'fix_types'
    obs = env.reset(task_id)
    return {'observation': obs.model_dump()}

@app.post('/step')
def step(action: Action):
    obs, reward, done, info = env.step(action.model_dump())
    out = RewardOut(reward=reward, raw={'info': info}, done=done, info=info)
    return {'observation': obs.model_dump(), 'reward': out.reward, 'done': out.done, 'info': info}

@app.get('/state')
def state():
    obs = env.state()
    return {'observation': obs.model_dump()}

def main():
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
