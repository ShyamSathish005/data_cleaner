from fastapi import FastAPI, HTTPException
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
def reset(payload: ResetRequest = None):
    task_id = payload.task_id if payload and payload.task_id else 'fix_types'
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
