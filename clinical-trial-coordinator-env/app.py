"""
FastAPI application exposing the OpenEnv standard API:
  POST /reset   → initial observation
  POST /step    → (observation, reward, done, info)
  GET  /state   → current state
  GET  /health  → liveness probe
  GET  /tasks   → list available tasks

Session management: each WebSocket/HTTP session gets its own env instance.
For stateless HTTP we use a simple in-memory session store keyed by session_id.
"""
from __future__ import annotations
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import ClinicalTrialAction
from env import ClinicalTrialCoordinatorEnv

app = FastAPI(
    title="ClinicalTrialCoordinatorEnv",
    description="OpenEnv environment simulating real clinical trial coordination workflows.",
    version="1.0.0",
)

# In-memory session store: session_id → env instance
_sessions: Dict[str, ClinicalTrialCoordinatorEnv] = {}

VALID_TASKS = ["screen_patient", "detect_deviation", "draft_sae_narrative"]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "screen_patient"
    scenario_index: int = 0
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    action_type: str
    payload: Dict[str, Any] = {}
    session_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "ClinicalTrialCoordinatorEnv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "screen_patient",
                "difficulty": "easy",
                "description": "Determine if a patient is eligible for a clinical trial based on inclusion/exclusion criteria.",
                "action_type": "screen_patient",
                "max_steps": 4,
            },
            {
                "name": "detect_deviation",
                "difficulty": "medium",
                "description": "Review a clinical visit record and identify any protocol deviations with correct severity classification.",
                "action_type": "flag_deviation",
                "max_steps": 6,
            },
            {
                "name": "draft_sae_narrative",
                "difficulty": "hard",
                "description": "Draft a complete ICH E2A-compliant Serious Adverse Event narrative including causality assessment and regulatory classification.",
                "action_type": "draft_sae_narrative",
                "max_steps": 8,
            },
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest):
    if req.task_name not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task_name. Choose from: {VALID_TASKS}")

    session_id = req.session_id or str(uuid.uuid4())
    env = ClinicalTrialCoordinatorEnv(
        task_name=req.task_name,
        scenario_index=req.scenario_index,
    )
    _sessions[session_id] = env
    obs = env.reset()

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "task_name": req.task_name,
    }


@app.post("/step")
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if not env:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found. Call /reset first."
        )

    action = ClinicalTrialAction(
        action_type=req.action_type,
        payload=req.payload,
    )
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str):
    env = _sessions.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state()


@app.delete("/session/{session_id}")
def cleanup(session_id: str):
    _sessions.pop(session_id, None)
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# OpenEnv spec: typed model introspection endpoints
# ---------------------------------------------------------------------------

@app.get("/spec/observation")
def observation_schema():
    from models import ClinicalTrialObservation
    return ClinicalTrialObservation.model_json_schema()


@app.get("/spec/action")
def action_schema():
    return ClinicalTrialAction.model_json_schema()


@app.get("/spec/reward")
def reward_schema():
    from models import ClinicalTrialReward
    return ClinicalTrialReward.model_json_schema()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
