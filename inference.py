import os
import json
import urllib.request
from typing import Dict, Any
from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "missing")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860").strip()

CLIENT = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

def call_env(path: str, payload: Dict[str, Any]):
    url = ENV_BASE_URL.rstrip('/') + path
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode('utf-8')
        return json.loads(body)

def ask_model_for_action(task_id: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "You are a data cleaning agent. Each turn you must return ONLY a JSON object\n"
        "with exactly two keys: 'name' and 'params'. No markdown, no explanation,\n"
        "no code fences. Just raw JSON.\n"
        "Available actions: fill_missing, cast_type, drop_duplicates, replace,\n"
        "normalize_dates, clamp_outliers, submit.\n"
        "When you are done cleaning or have no more useful actions, return:\n"
        "{\"name\": \"submit\", \"params\": {}}"
    )
    # Use exact system prompt
    
    fallback_action = {'name': 'submit', 'params': {}}
    try:
        completion = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Task ID: {task_id}\nObservation: {json.dumps(observation)}"}
            ],
            max_tokens=200,
            temperature=0.0,
        )
        text = completion.choices[0].message.content.strip()
        
        # Handle cases where the model wraps JSON in markdown fences anyway
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

        parsed = json.loads(text)
        if isinstance(parsed, dict) and 'name' in parsed:
            return {'name': parsed.get('name'), 'params': parsed.get('params', {})}
        return fallback_action
    except json.JSONDecodeError:
        return fallback_action
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return fallback_action

def run_episode(task_id: str, max_steps=20):
    print(f"[START] task={task_id} env=data-cleaning-openenv model={MODEL_NAME}", flush=True)
    steps = 0
    done = False
    final_score = 0.0
    rewards = []
    
    try:
        res = call_env('/reset', {'task_id': task_id})
        obs = res.get('observation', {})
        
        for step in range(max_steps):
            steps = step + 1
            action = ask_model_for_action(task_id, obs)
            error_msg = "null"
            step_reward = 0.0
            
            try:
                step_resp = call_env('/step', action)
                obs = step_resp.get('observation', {})
                step_reward = float(step_resp.get('reward', 0.0))
                done = bool(step_resp.get('done', False))
                final_score = float(step_resp.get('info', {}).get('accuracy', obs.get('accuracy', 0.0)))
            except Exception as e:
                error_msg = f"\"{str(e)}\""
                done = True

            rewards.append(step_reward)
            action_json = json.dumps(action, separators=(',', ':'))
            print(f"[STEP] step={steps} action={action_json} reward={step_reward:.2f} done={str(done).lower()} error={error_msg}", flush=True)
            
            if done:
                break
    except Exception as e:
         print(f"[DEBUG] Episode crashed: {e}", flush=True)
    finally:
         rewards_str = ",".join(f"{r:.2f}" for r in rewards)
         print(f"[END] success={str(done).lower()} steps={steps} score={final_score:.3f} rewards={rewards_str}", flush=True)
         return {'task_id': task_id, 'steps': steps, 'score': final_score, 'done': done}

if __name__ == '__main__':
    tasks = ['fix_types', 'normalize_dedupe', 'full_pipeline']
    results = [run_episode(t) for t in tasks]
    avg = sum(r['score'] for r in results) / max(len(results), 1)
    print(f"[SUMMARY] tasks={len(results)} average_score={avg:.3f}", flush=True)
