# Data Cleaning OpenEnv

Data Cleaning OpenEnv is a FastAPI-based environment for evaluating data-cleaning agents on realistic ETL-style tasks. It provides a structured action interface, dense rewards, and reproducible datasets across multiple difficulty levels.

## Overview

The environment simulates common production data-quality problems:

- Mixed numeric and string types
- Inconsistent date formats
- Duplicate records
- Missing values
- Outliers and rule violations

It is designed for reinforcement-learning style interaction through `reset`, `step`, and `state` operations.

## API Summary

Base URL (local): `http://127.0.0.1:7860`

- `GET /` : Service metadata
- `GET /health` : Health check
- `POST /reset` : Start a task episode
- `POST /step` : Apply one cleaning action
- `GET /state` : Fetch current observation

The service listens on port `7860` to align with Hugging Face Docker Space requirements.

## Action Format

All actions use this JSON structure:

```json
{
  "name": "<action_name>",
  "params": {}
}
```

Supported action names:

- `fill_missing`
- `cast_type`
- `drop_duplicates`
- `replace`
- `normalize_dates`
- `clamp_outliers`
- `submit`

Backward-compatible aliases are accepted for some actions.

## Tasks

- `fix_types`: Focus on type normalization
- `normalize_dedupe`: Date normalization and duplicate handling
- `full_pipeline`: Full cleaning workflow with missing values and outliers

## Reward Design

- Step reward: change in accuracy from previous step
- Completion bonus: awarded on `submit` when accuracy is perfect
- Invalid action penalty: small negative reward
- Episode cap: fixed maximum step count

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Run with Docker

```bash
docker build -t data-cleaning-openenv .
docker run --rm -p 7860:7860 data-cleaning-openenv
```

## Quick API Test

```bash
curl -s http://127.0.0.1:7860/

curl -s -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"fix_types"}'

curl -s -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d '{"name":"submit","params":{}}'
```

## Hugging Face Space Deployment

This repository is configured for a Docker Space.

1. Create or open your Hugging Face Space with Docker SDK.
2. Push this repository to the Space remote.
3. Confirm the app starts and responds on port `7860`.

Suggested environment variables for evaluation workflows:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `ENV_BASE_URL` (optional if not using local default)

## Baseline Inference

Run the baseline script after setting required variables:

```bash
export ENV_BASE_URL=http://127.0.0.1:7860
export API_BASE_URL=https://your-openai-compatible-endpoint/v1
export MODEL_NAME=gpt-4o
export HF_TOKEN=your_token
python inference.py
```

## Validation

```bash
python pre_validation.py
```

To skip Docker during quick local checks:

```bash
SKIP_DOCKER=1 python pre_validation.py
```

## Project Structure

```text
data_cleaning_openenv/
├── app.py
├── env.py
├── models.py
├── inference.py
├── pre_validation.py
├── openenv.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```
