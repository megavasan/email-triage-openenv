# 📧 Email Triage OpenEnv

An **OpenEnv-compliant reinforcement learning environment** where AI agents learn to manage a realistic email inbox — classifying urgency, categorizing emails, drafting replies, escalating critical issues, and archiving spam.

---

## 🌍 Environment Description & Motivation

Email triage is a task every knowledge worker does daily. It involves:
- Reading unstructured text under time pressure
- Making multi-criteria decisions (urgency × category × action)
- Balancing precision (missing a critical email) vs. recall (over-escalating)
- Producing contextually appropriate replies

This environment trains and evaluates agents on a skill with immediate, real-world value — far beyond toy gridworlds.

---

## 🔍 Observation Space

| Field | Type | Description |
|---|---|---|
| `current_email` | object | Email to process (subject, sender, body, timestamp) |
| `queue_size` | int | Total emails in the episode |
| `processed_count` | int | Emails already handled |
| `current_task` | string | Active task name |
| `task_id` | string | Task identifier |
| `step_number` | int | Current step |
| `feedback` | string | Feedback from last action |

---

## ⚡ Action Space

| Field | Type | Required | Values |
|---|---|---|---|
| `action_type` | enum | ✅ | `classify`, `prioritize`, `draft_reply`, `archive`, `escalate` |
| `category` | enum | ❌ | `billing`, `technical`, `general`, `spam`, `complaint`, `praise`, `urgent_request` |
| `urgency` | enum | ❌ | `low`, `medium`, `high`, `critical` |
| `reply_text` | string | ❌ | Draft reply (for `draft_reply` action) |
| `reason` | string | ❌ | Justification (earns bonus on hard task) |

---

## 🎯 Tasks

### Task 1: Basic Email Classification (Easy)
- **ID:** `task_easy`
- **Emails:** 3 (server outage, spam, overdue invoice)
- **Challenge:** Detect obvious category and urgency signals
- **Max Steps:** 6
- **Expected Score:** 0.70–0.90

### Task 2: Email Triage with Replies (Medium)
- **ID:** `task_medium`
- **Emails:** 3 (API question, customer praise, meeting reschedule)
- **Challenge:** Requires careful reading; must draft a professional reply
- **Max Steps:** 9
- **Expected Score:** 0.50–0.75

### Task 3: Full Inbox Management (Hard)
- **ID:** `task_hard`
- **Emails:** 3 (pricing complaint, partnership inquiry, account-locked emergency)
- **Challenge:** Nuanced judgment, escalation, diplomatic tone, reasoning required
- **Max Steps:** 12
- **Expected Score:** 0.35–0.65

---

## 🏆 Reward Function

Per-step reward is computed as:

| Component | Weight | Notes |
|---|---|---|
| Category accuracy | 35% | Must match ground truth |
| Urgency accuracy | 30% | Partial credit (1 level off = 50%) |
| Action type | 25% | Classify / archive / escalate etc. |
| Reply quality | 10% | Keyword matching for draft_reply |
| Reasoning bonus | +5% | Hard task only, when reason provided |
| Spam escalation penalty | -30% | Penalizes escalating obvious spam |

**Partial progress:** Urgency that's one level off still earns 15% (instead of 30%). This prevents sparse all-or-nothing rewards.

---

## 🚀 Setup & Usage

### Option 1: Run with Docker

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/email-triage-openenv
cd email-triage-openenv
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

Visit: http://localhost:7860

### Option 2: Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run Baseline Agent

```bash
# Get a FREE Groq API key at: https://console.groq.com
export GROQ_API_KEY=your_key_here
python baseline.py
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit an action |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List all tasks + action schema |
| `/grader` | POST | Grade a set of actions for a task |
| `/baseline` | GET | Run baseline agent, return scores |

### Example: Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'
```

### Example: Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "escalate",
    "category": "technical",
    "urgency": "critical",
    "reason": "Production server down, immediate response needed"
  }'
```

### Example: Grader

```bash
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_easy",
    "actions": [
      {"action_type": "escalate", "category": "technical", "urgency": "critical"},
      {"action_type": "archive", "category": "spam", "urgency": "low"},
      {"action_type": "prioritize", "category": "billing", "urgency": "high"}
    ]
  }'
```

---

## 📊 Baseline Scores

Measured with `llama3-8b-8192` via Groq API:

| Task | Difficulty | Score |
|---|---|---|
| Basic Email Classification | Easy | ~0.82 |
| Email Triage with Replies | Medium | ~0.61 |
| Full Inbox Management | Hard | ~0.47 |
| **Overall Average** | — | **~0.63** |

---

## 📁 Project Structure

```
email-triage-openenv/
├── environment.py      # Core env: EmailTriageEnv, models, grader
├── app.py              # FastAPI server with all OpenEnv endpoints
├── baseline.py         # Baseline inference script (Groq/OpenAI)
├── openenv.yaml        # OpenEnv spec metadata
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build
└── README.md           # This file
```

---

## 🏷️ OpenEnv Compliance

- ✅ Typed `Observation`, `Action`, `Reward` Pydantic models
- ✅ `step(action)` → observation, reward, done, info
- ✅ `reset()` → initial observation
- ✅ `state()` → current environment state
- ✅ `openenv.yaml` with full metadata
- ✅ 3 tasks: easy → medium → hard
- ✅ Graders scoring 0.0–1.0 with deterministic criteria
- ✅ Partial-progress reward (non-sparse)
- ✅ Baseline inference script with Groq API
- ✅ Docker deployment
- ✅ Hugging Face Space with `openenv` tag

---

## 📜 License

MIT
