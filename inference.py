import os
import json
import urllib.request
from typing import Dict, Any
from openai import OpenAI

"""Baseline inference for the Data Cleaning OpenEnv submission."""

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860").strip()

CLIENT = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

def call_env(path: str, payload: Dict[str, Any]):
    url = ENV_BASE_URL.rstrip('/') + path
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode('utf-8')
        return json.loads(body)

def ask_model_for_action(task_id: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the model for one next action; return a safe fallback on parse/errors."""
    prompt = (
        "You are an agent solving a data cleaning task. "
        "Return valid JSON with keys \"name\" and \"params\" only, no markdown, no explanation.\n"
        "Allowed action names: fill_missing, cast_type, drop_duplicates, replace, normalize_dates, clamp_outliers, submit.\n"
        f"Task ID: {task_id}\n"
        f"Observation: {json.dumps(observation)}"
    )
    
    resp = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or '').strip()
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and 'name' in parsed:
            return {'name': parsed.get('name'), 'params': parsed.get('params', {})}
        else:
            print(f"Invalid JSON format for action: {text}")
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e} - Response: {text}")
        
    return {'name': 'submit', 'params': {}}

def run_episode(task_id: str, max_steps=20):
    print(
        f"[START] task_id={task_id} model={MODEL_NAME} llm_api_base={API_BASE_URL} "
        f"env_base={ENV_BASE_URL}"
    )
    res = call_env('/reset', {'task_id': task_id})
    obs = res['observation']
    steps = 0
    final_reward = 0.0
    done = False
    
    for step in range(max_steps):
        steps = step + 1
        action = ask_model_for_action(task_id, obs)
        try:
            step_resp = call_env('/step', action)
        except Exception:
            step_resp = call_env('/step', {'name': 'submit', 'params': {}})
        obs = step_resp['observation']
        final_reward = float(step_resp.get('reward', 0.0))
        done = bool(step_resp.get('done', False))
        score = float(step_resp.get('info', {}).get('accuracy', obs.get('accuracy', 0.0)))
        print(
            f"[STEP] task_id={task_id} step={steps} action={json.dumps(action, separators=(',', ':'))} "
            f"reward={final_reward:.6f} score={score:.6f} done={str(done).lower()}"
        )
        if done:
            break
    final_score = float(obs.get('accuracy', 0.0))
    print(
        f"[END] task_id={task_id} steps={steps} final_score={final_score:.6f} "
        f"final_reward={final_reward:.6f} status={'success' if done else 'max_steps'}"
    )
    return {'task_id': task_id, 'steps': steps, 'score': final_score, 'done': done}

if __name__ == '__main__':
    tasks = ['fix_types', 'normalize_dedupe', 'full_pipeline']
    results = [run_episode(task_id=t, max_steps=20) for t in tasks]
    avg = sum(r['score'] for r in results) / max(len(results), 1)
    print(f"[END] run_summary tasks={len(results)} average_score={avg:.6f}")
