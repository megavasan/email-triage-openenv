"""
FastAPI server exposing the Email Triage OpenEnv environment.
Endpoints: /reset, /step, /state, /tasks, /grader, /baseline, /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import sys

# Support both running from root and from server/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import (
    EmailTriageEnv, Action, ActionType, EmailCategory, UrgencyLevel, TASKS
)

app = FastAPI(
    title="Email Triage OpenEnv",
    description="An OpenEnv-compliant environment where AI agents learn to triage emails.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance
env = EmailTriageEnv()


# ── Request/Response Schemas ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"

class StepRequest(BaseModel):
    action_type: str
    category: Optional[str] = None
    urgency: Optional[str] = None
    reply_text: Optional[str] = None
    reason: Optional[str] = None

class GraderRequest(BaseModel):
    task_id: str
    actions: list[dict]


# ── Core OpenEnv Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "email-triage", "version": "1.0.0"}

@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "description": "AI agent learns to triage, classify, and respond to emails",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/health"],
        "tasks": list(TASKS.keys())
    }

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    task_id = (req.task_id if req and req.task_id else None) or "task_easy"
    try:
        obs = env.reset(task_id=task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(req: StepRequest):
    try:
        action = Action(
            action_type=ActionType(req.action_type),
            category=EmailCategory(req.category) if req.category else None,
            urgency=UrgencyLevel(req.urgency) if req.urgency else None,
            reply_text=req.reply_text,
            reason=req.reason
        )
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid action value: {e}")

@app.get("/state")
def state():
    try:
        return env.state().model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "task_id": tid,
                "name": tcfg["name"],
                "description": tcfg["description"],
                "difficulty": tcfg["difficulty"],
                "max_steps": tcfg["max_steps"],
            }
            for tid, tcfg in TASKS.items()
        ],
        "action_schema": {
            "action_type": {
                "type": "string",
                "required": True,
                "values": [a.value for a in ActionType],
            },
            "category": {
                "type": "string",
                "required": False,
                "values": [c.value for c in EmailCategory],
            },
            "urgency": {
                "type": "string",
                "required": False,
                "values": [u.value for u in UrgencyLevel],
            },
            "reply_text": {"type": "string", "required": False},
            "reason": {"type": "string", "required": False}
        }
    }

@app.post("/grader")
def grader(req: GraderRequest):
    from server.environment import grade_action, EMAIL_DATASET

    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")

    task_cfg = TASKS[req.task_id]
    email_map = {e.id: e for e in EMAIL_DATASET}
    email_queue = [email_map[eid] for eid in task_cfg["email_ids"]]

    scores = []
    feedbacks = []

    for i, (email, action_dict) in enumerate(zip(email_queue, req.actions)):
        try:
            action = Action(
                action_type=ActionType(action_dict.get("action_type", "classify")),
                category=EmailCategory(action_dict["category"]) if action_dict.get("category") else None,
                urgency=UrgencyLevel(action_dict["urgency"]) if action_dict.get("urgency") else None,
                reply_text=action_dict.get("reply_text"),
                reason=action_dict.get("reason")
            )
            score, feedback = grade_action(email, action, task_cfg["difficulty"])
            scores.append(round(score, 4))
            feedbacks.append(feedback)
        except Exception as e:
            scores.append(0.0)
            feedbacks.append(f"Error grading action {i}: {e}")

    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {
        "task_id": req.task_id,
        "scores": scores,
        "feedbacks": feedbacks,
        "average_score": avg_score
    }

@app.get("/baseline")
async def baseline():
    import subprocess
    import json

    try:
        result = subprocess.run(
            [sys.executable, "inference.py", "--json"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"output": result.stdout, "error": result.stderr}
        else:
            return {"error": result.stderr or "Failed", "output": result.stdout}
    except subprocess.TimeoutExpired:
        return {"error": "Timed out after 120 seconds"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)
