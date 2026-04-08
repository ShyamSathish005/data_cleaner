from fastapi import FastAPI, HTTPException
from models import Action, Observation, ResetRequest, RewardOut
from env import DataCleaningEnv

app = FastAPI(title='Data Cleaning OpenEnv')
env = DataCleaningEnv()

@app.post('/reset')
def reset(payload: ResetRequest):
    task_id = payload.task_id
    if not task_id:
        raise HTTPException(status_code=400, detail='task_id required')
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
