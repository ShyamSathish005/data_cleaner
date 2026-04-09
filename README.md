---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Data Cleaning OpenEnv

A production-grade OpenEnv environment for training and evaluating data-cleaning agents on realistic ETL-style tasks. Agents interact through a structured `reset / step / state` loop, applying discrete repair operations to corrupted dataframes and receiving dense reward signals tied to ground-truth accuracy.

---

## Why This Environment

Data cleaning is one of the highest-cost, highest-frequency tasks in real data engineering. It is also one of the least studied in agent evaluation — most benchmarks focus on code generation or web navigation. This environment fills that gap.

The three tasks model corruptions that appear in actual production pipelines: type coercion failures, inconsistent date and phone formatting, missing values, outlier constraint violations, and duplicate records. An agent that scores well here is solving a problem that data teams face daily.

---

## Task Suite

| ID | Difficulty | Description |
|----|------------|-------------|
| `fix_types` | Easy | All columns are stored as strings. Cast each to its correct type: integer, float, boolean, datetime. |
| `normalize_dedupe` | Medium | Names, dates, and phone numbers are in inconsistent formats. Approximately 15% of rows are duplicates. Normalize and deduplicate. |
| `full_pipeline` | Hard | A heavily corrupted dataset with wrong types, missing values, age and revenue constraint violations, future signup dates, and duplicate rows. Repair everything. |

Each task uses a seeded synthetic dataset so results are fully reproducible across runs.

---

## Observation Space

Every `step` and `reset` call returns an `Observation` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `dataframe_preview` | `list[dict]` | First 5 rows as a list of records |
| `markdown_preview` | `string` | Same rows rendered as a markdown table |
| `null_counts` | `dict[str, int]` | Per-column null value counts |
| `validation_errors` | `list[string]` | Human-readable list of remaining issues |
| `accuracy` | `float` | Current score against ground truth (0.0–1.0) |
| `step_count` | `int` | Number of actions taken so far |

---

## Action Space

All actions follow a single JSON envelope:

```json
{
  "name": "<action_name>",
  "params": { }
}
```

| Action | Key Params | Effect |
|--------|-----------|--------|
| `cast_type` | `column`, `dtype` | Cast a column to `float`, `int`, `str`, or `datetime` |
| `fill_missing` | `column`, `strategy` | Fill nulls via `mean`, `median`, `mode`, `ffill`, `bfill`, or a literal value |
| `drop_duplicates` | `subset` (optional) | Remove duplicate rows, optionally scoped to a column subset |
| `normalize_dates` | `column` | Parse and reformat dates to ISO-8601 (`YYYY-MM-DD`) |
| `replace` | `column`, `old_value`, `new_value` | Replace a specific value in a column |
| `clamp_outliers` | `column`, `low`, `high` | Clip numeric values to a valid range |
| `submit` | — | Finalise the episode and record the score |

---

## Reward Design

The reward function provides dense signal across the full trajectory, not just at episode end.

```
step_reward  = clamp(accuracy_delta, 0.0, 1.0)
invalid_penalty = -0.05   (applied on malformed or no-op actions)
submit_bonus = +1.0       (applied only when accuracy == 1.0 at submit)
```

Agents that improve accuracy at every step receive consistent positive feedback. Agents that repeat ineffective actions are penalised. The submit bonus incentivises committing once the dataframe is genuinely clean rather than running out the step budget.

---

## API Reference

Base URL (local): `http://127.0.0.1:7860`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service metadata and version |
| `GET` | `/health` | Liveness check — returns `{"ok": true}` |
| `POST` | `/reset` | Start a new episode for a given task |
| `POST` | `/step` | Apply one cleaning action |
| `GET` | `/state` | Return the current observation without advancing the episode |

### Reset

```bash
curl -s -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "fix_types"}'
```

### Step

```bash
curl -s -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d '{"name": "cast_type", "params": {"column": "amount", "dtype": "float"}}'
```

### Submit

```bash
curl -s -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d '{"name": "submit", "params": {}}'
```

---

## Setup

### Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t data-cleaning-openenv .
docker run --rm -p 7860:7860 data-cleaning-openenv
```

---

## Baseline Inference

The `inference.py` script runs a deterministic baseline plan against all three tasks, then falls back to an LLM agent for any remaining steps. It uses the OpenAI client and reads credentials from environment variables.

```bash
export ENV_BASE_URL=http://127.0.0.1:7860
export API_BASE_URL=https://your-openai-compatible-endpoint/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_token_here
python inference.py
```

Expected output format:

```
[START] task_id=fix_types model=gpt-4o-mini ...
[STEP]  task_id=fix_types step=1 action={"name":"cast_type",...} reward=0.500000 score=0.500000 done=false
[STEP]  task_id=fix_types step=2 action={"name":"submit",...}    reward=0.000000 score=0.500000 done=true
[END]   task_id=fix_types steps=2 final_score=0.500000 status=max_steps
...
[END]   run_summary tasks=3 average_score=0.xxxxxx
```

---

## Validation

Run the pre-submission validation script to check all endpoints respond correctly before deploying:

```bash
python pre_validation.py
```

To skip Docker checks during local iteration:

```bash
SKIP_DOCKER=1 python pre_validation.py
```

---

## Hugging Face Deployment

This repository is configured as a Docker Space. To deploy:

1. Create a Space at `huggingface.co/spaces` with the **Docker** SDK.
2. Add this repository as the Space remote and push.
3. The container starts on port `7860` automatically — no extra configuration needed.

Set the following Space secrets for inference runs:

| Variable | Purpose |
|----------|---------|
| `API_BASE_URL` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | Model identifier for inference |
| `HF_TOKEN` | API key / Hugging Face token |
| `ENV_BASE_URL` | Override if environment is not on localhost |

---

## Project Structure

```
data-cleaning-openenv/
├── app.py              FastAPI server — reset / step / state endpoints
├── env.py              Core environment logic and action executors
├── models.py           Pydantic models for Observation, Action, Reward
├── inference.py        Baseline inference script (OpenAI client)
├── pre_validation.py   Pre-submission validation checks
├── openenv.yaml        OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Environment Metadata

| Property | Value |
|----------|-------|
| Interface | OpenEnv v1 (`reset / step / state`) |
| Reward | Dense, per-step accuracy delta |
| Episodes | Seeded, reproducible |
| Max steps | 15 / 20 / 30 (easy / medium / hard) |
| Serving port | 7860 |
| Runtime | Python 3.10, FastAPI, pandas, numpy |