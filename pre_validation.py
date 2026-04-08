import json
import os
import subprocess
import sys
import time
import urllib.request
from typing import Dict, Any

SPACE_URL = os.environ.get("SPACE_URL", "http://127.0.0.1:7860").rstrip("/")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "data-cleaning-openenv:precheck")


def post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"{SPACE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def check_http():
    ok = True
    try:
        res = post("/reset", {"task_id": "fix_types"})
        ok = ok and "observation" in res
    except Exception as exc:
        print(f"[FAIL] reset endpoint check failed: {exc}")
        return False
    print("[PASS] HTTP reset endpoint responds")
    return ok


def check_openenv_spec_files():
    needed = ["openenv.yaml", "env.py", "models.py", "app.py"]
    missing = [f for f in needed if not os.path.exists(f)]
    if missing:
        print(f"[FAIL] Missing required files: {missing}")
        return False
    print("[PASS] Required OpenEnv files present")
    return True


def check_docker_build():
    try:
        result = run(["docker", "build", "-t", IMAGE_NAME, "."])
    except FileNotFoundError:
        print("[FAIL] Docker CLI not found on this machine")
        return False
    if result.returncode != 0:
        print("[FAIL] Docker build failed")
        print(result.stdout)
        print(result.stderr)
        return False
    print("[PASS] Docker build succeeded")
    return True


def check_inference_exec():
    env = os.environ.copy()
    env.setdefault("ENV_BASE_URL", SPACE_URL)
    env.setdefault("API_BASE_URL", "")
    env.setdefault("MODEL_NAME", "")
    env.setdefault("HF_TOKEN", "")
    result = subprocess.run([sys.executable, "inference.py"], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print("[FAIL] inference.py failed")
        print(result.stdout)
        print(result.stderr)
        return False

    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not any(ln.startswith("[START]") for ln in lines):
        print("[FAIL] inference output missing [START]")
        return False
    if not any(ln.startswith("[STEP]") for ln in lines):
        print("[FAIL] inference output missing [STEP]")
        return False
    if not any(ln.startswith("[END]") for ln in lines):
        print("[FAIL] inference output missing [END]")
        return False

    print("[PASS] inference.py runs and emits structured markers")
    return True


def check_tasks_and_scores():
    task_ids = ["fix_types", "normalize_dedupe", "full_pipeline"]
    for task_id in task_ids:
        reset = post("/reset", {"task_id": task_id})
        assert "observation" in reset
        step = post("/step", {"name": "submit", "params": {}})
        score = float(step.get("info", {}).get("accuracy", step.get("observation", {}).get("accuracy", 0.0)))
        reward = float(step.get("reward", 0.0))
        if not (0.0 <= score <= 1.0):
            print(f"[FAIL] Score out of range for {task_id}: {score}")
            return False
        if not (0.0 <= reward <= 1.0):
            print(f"[FAIL] Reward out of range for {task_id}: {reward}")
            return False

    print("[PASS] 3 tasks verified with scores/rewards in [0.0, 1.0]")
    return True


def main():
    checks = [
        check_http,
        check_openenv_spec_files,
        check_tasks_and_scores,
        check_inference_exec,
    ]

    if os.environ.get("SKIP_DOCKER", "0") != "1":
        checks.append(check_docker_build)

    started = time.time()
    all_ok = True
    for check in checks:
        ok = check()
        all_ok = all_ok and ok

    elapsed = time.time() - started
    print(f"\nValidation finished in {elapsed:.1f}s")
    if not all_ok:
        print("Submission readiness: FAIL")
        sys.exit(1)
    print("Submission readiness: PASS")


if __name__ == "__main__":
    main()
