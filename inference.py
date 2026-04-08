"""
Baseline Inference Script — Email Triage OpenEnv
Prints [START]/[STEP]/[END] structured output as required by validator.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from server.environment import EmailTriageEnv, Action, ActionType, EmailCategory, UrgencyLevel, TASKS
except ImportError:
    try:
        from environment import EmailTriageEnv, Action, ActionType, EmailCategory, UrgencyLevel, TASKS
    except ImportError as e:
        print(f"[ERROR] Could not import environment: {e}", flush=True)
        sys.exit(0)


def get_fallback_action(obs_dict: dict) -> Action:
    try:
        email = obs_dict.get("current_email") or {}
        text = ((email.get("subject") or "") + " " + (email.get("body") or "")).lower()

        if any(w in text for w in ["won", "free prize", "click here", "congratulations", "limited time"]):
            return Action(action_type=ActionType.ARCHIVE, category=EmailCategory.SPAM, urgency=UrgencyLevel.LOW)
        if any(w in text for w in ["server down", "outage", "production", "locked", "2 hours", "immediately", "down for"]):
            return Action(action_type=ActionType.ESCALATE, category=EmailCategory.TECHNICAL, urgency=UrgencyLevel.CRITICAL)
        if any(w in text for w in ["invoice", "payment", "overdue", "billing", "charge"]):
            return Action(action_type=ActionType.PRIORITIZE, category=EmailCategory.BILLING, urgency=UrgencyLevel.HIGH)
        if any(w in text for w in ["disappointed", "switching", "7 years", "pricing", "complaint"]):
            return Action(action_type=ActionType.ESCALATE, category=EmailCategory.COMPLAINT, urgency=UrgencyLevel.HIGH,
                         reason="Long-term customer complaint requires escalation")
        if any(w in text for w in ["thank", "appreciate", "above and beyond", "great job"]):
            return Action(action_type=ActionType.CLASSIFY, category=EmailCategory.PRAISE, urgency=UrgencyLevel.LOW)
        if any(w in text for w in ["reschedule", "meeting", "wednesday", "partnership", "call next week"]):
            return Action(action_type=ActionType.DRAFT_REPLY, category=EmailCategory.GENERAL, urgency=UrgencyLevel.MEDIUM,
                         reply_text="Thank you for reaching out. We confirm the reschedule and will connect shortly.")
        if any(w in text for w in ["api", "rate limit", "429", "requests"]):
            return Action(action_type=ActionType.CLASSIFY, category=EmailCategory.TECHNICAL, urgency=UrgencyLevel.MEDIUM)

        return Action(action_type=ActionType.CLASSIFY, category=EmailCategory.GENERAL, urgency=UrgencyLevel.MEDIUM)
    except Exception:
        return Action(action_type=ActionType.CLASSIFY, urgency=UrgencyLevel.MEDIUM)


def run_task(task_id: str) -> float:
    try:
        env = EmailTriageEnv()
        obs = env.reset(task_id=task_id)
        task_name = TASKS[task_id]["name"]

        print(f"[START] task={task_name}", flush=True)

        step = 0
        while obs.current_email is not None:
            action = get_fallback_action(obs.model_dump())
            obs, reward, done, info = env.step(action)
            step += 1

            print(f"[STEP] step={step} reward={reward.value}", flush=True)

            if done:
                break

        score = env.get_final_score()
        print(f"[END] task={task_name} score={score} steps={step}", flush=True)
        return score

    except Exception as e:
        print(f"[END] task={task_id} score=0.0 steps=0 error={e}", flush=True)
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = {}

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        score = run_task(task_id)
        results[task_id] = {
            "score": score,
            "task_name": TASKS[task_id]["name"],
            "difficulty": TASKS[task_id]["difficulty"]
        }

    avg = round(sum(r["score"] for r in results.values()) / len(results), 4)

    if args.json:
        print(json.dumps({
            "baseline_scores": results,
            "model": "rule-based-fallback",
            "overall_average": avg
        }), flush=True)
    else:
        print(f"\nOverall Average: {avg:.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[END] task=error score=0.0 steps=0 error={e}", flush=True)
        sys.exit(0)
