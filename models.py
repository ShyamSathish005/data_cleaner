from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class ResetRequest(BaseModel):
    task_id: str

class Action(BaseModel):
    name: str
    params: Optional[Dict[str, Any]] = {}

class Observation(BaseModel):
    dataframe_preview: List[Dict[str, Any]]
    markdown_preview: str
    null_counts: Dict[str, int]
    validation_errors: List[str]
    accuracy: float
    step_count: int
    steps_remaining: int

class RewardOut(BaseModel):
    reward: float
    raw: Dict[str, Any] = {}
    done: bool = False
    info: Dict[str, Any] = {}
