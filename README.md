# Data Cleaning Pipeline OpenEnv

Submission-ready OpenEnv environment for realistic ETL/Data Ops cleaning tasks. The environment simulates agent-driven repair of messy CSV-style tables and scores each action with dense reward against a hidden golden dataframe.

## Environment Overview

The Data Cleaning Pipeline models common production data quality issues:
- mixed numeric/string typing,
- inconsistent date formatting,
- fuzzy duplicates,
- missing values,
- outliers and logic violations.

This mirrors real-world pre-ingestion cleanup in analytics, data warehousing, and ML feature pipelines.

## OpenEnv Rubric Mapping

- **Interface**: `reset(task_id)`, `step(action) -> (Observation, Reward, done, info)`, `state()`.
- **API**: FastAPI endpoints `/reset`, `/step`, `/state` on port `7860`.
- **Typed models**: Pydantic request/response models in `models.py`.
- **Reward**: Dense delta reward + completion bonus + invalid-op penalty.

## Action Space

Actions use JSON shape:

```json
{"name": "<action_name>", "params": {"...": "..."}}
```

| Action | Purpose | Expected params |
|---|---|---|
| `fill_missing` | Fill null values in a column | `{"column": "age", "strategy": "mean|ffill|<literal>"}` |
| `cast_type` | Cast string/dirty numeric fields to a target type | `{"column": "amount", "target_type": "float|int"}` |
| `drop_duplicates` | Remove duplicate rows (optional subset columns) | `{"subset": ["name", "date"]}` (optional) |
| `replace` | Replace values globally or in one column | `{"column": "name", "old_value": "N/A", "new_value": ""}` |
| `normalize_dates` | Parse and normalize date strings to ISO format | `{"column": "date"}` |
| `clamp_outliers` | Clamp numeric column into bounds | `{"column": "age", "low": 0, "high": 120}` |
| `submit` | End episode and trigger final scoring bonus check | `{}` |

Backward-compatible aliases are supported (`cast`, `dedupe`) for convenience.

## Observation Space

Each step returns:
- `dataframe_preview`: first 5 rows as JSON-safe dict records,
- `markdown_preview`: markdown table preview of first 5 rows,
- `null_counts`: per-column null counts,
- `validation_errors`: runtime/action and logic-validation diagnostics,
- `accuracy`: dense score in `[0, 1]` vs hidden golden dataframe,
- `step_count`: current step number.

All NumPy/Pandas values are cast to Python primitives (`int`, `float`, `str`, `bool`, `None`) before serialization.

## Task Difficulty

- **Easy — `fix_types` (10 rows)**
	- Messiness: numeric values embedded as strings (`"1,200"`, `"$12.50"`).
	- Goal: cast to clean numeric types.

- **Medium — `normalize_dedupe` (50 rows + 10% dupes)**
	- Messiness: mixed date formats (`MM/DD/YY`, ISO), fuzzy-like name variants, injected duplicate rows.
	- Goal: normalize records and remove duplicates.

- **Hard — `full_pipeline` (100 rows)**
	- Messiness: missing values, outliers, and logic violations (e.g., implausible ages).
	- Goal: apply full cleaning sequence toward golden parity.

## Reward Design

- Dense reward: `reward = current_accuracy - previous_accuracy`.
- Completion bonus: `+1.0` on `submit` when `accuracy == 1.0`.
- Invalid operation penalty: `-0.05`.
- Anti-loop control: `done=True` when `step_count >= 15`.

## Setup & Run

### Local (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker (Hugging Face Spaces style)

```bash
docker build -t data-cleaning-openenv .
docker run --rm -p 7860:7860 data-cleaning-openenv
```

## Baseline Inference

The baseline script is `inference.py` at repo root and runs all three tasks for reproducible scoring output.

Required variables (per evaluator instructions):
- `API_BASE_URL` (OpenAI-compatible LLM endpoint)
- `MODEL_NAME`
- `HF_TOKEN`

Environment endpoint variable:
- `ENV_BASE_URL` (OpenEnv server URL, default `http://127.0.0.1:7860`)

Run:

```bash
export ENV_BASE_URL=http://127.0.0.1:7860
export API_BASE_URL=https://your-openai-compatible-endpoint/v1
export MODEL_NAME=gpt-4o
export HF_TOKEN=your_key
python inference.py
```

Inference logs are emitted in strict marker format:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

## Pre-Submission Validation

Run the local validator before submission:

```bash
python pre_validation.py
```

Useful flags:
- `SKIP_DOCKER=1 python pre_validation.py` to skip Docker build during quick local checks.
- `SPACE_URL=https://<your-space>.hf.space python pre_validation.py` to validate deployed Space endpoints.

## GitHub + Hugging Face Push Checklist

1. Commit and push to GitHub.
2. Create/attach a Hugging Face Space (Docker SDK) to the same repository contents.
3. In Space variables/secrets, define:
	- `API_BASE_URL`
	- `MODEL_NAME`
	- `HF_TOKEN`
	- `ENV_BASE_URL` (if your environment URL is not local default)
4. Confirm Space returns `200` on `/reset` and `inference.py` completes.

## Final Smoke Test

Use a basic endpoint ping script to verify `/reset` and `/step` handle NaN-heavy tasks without 500 errors:

```bash
python - <<'PY'
import json, urllib.request

def post(path, payload):
		req = urllib.request.Request(
				f"http://127.0.0.1:7860{path}",
				data=json.dumps(payload).encode("utf-8"),
				headers={"Content-Type": "application/json"},
				method="POST",
		)
		with urllib.request.urlopen(req, timeout=30) as r:
				return json.loads(r.read().decode("utf-8"))

print(post('/reset', {'task_id': 'full_pipeline'}))
print(post('/step', {'name': 'fill_missing', 'params': {'column': 'age', 'strategy': 'mean'}}))
print(post('/step', {'name': 'submit', 'params': {}}))
PY
```

## Directory Structure

```
data_cleaning_openenv/
├── app.py
├── env.py
├── models.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

This directory is ZIP-ready for upload and git push to a Hugging Face Space tagged `openenv`.
