"""
Baseline Inference Script — Email Triage OpenEnv
Uses Groq API (free, OpenAI-compatible) to run an LLM agent against all 3 tasks.

Setup:
    pip install openai
    export GROQ_API_KEY=your_key_here   # free at console.groq.com
    python baseline.py

Outputs:
    Scores for task_easy, task_medium, task_hard
"""

import os
import sys
import json
import argparse
from openai import OpenAI

from environment import EmailTriageEnv, Action, ActionType, EmailCategory, UrgencyLevel, TASKS

# ── Groq client (OpenAI-compatible) ───────────────────────────────────────────
def get_client():
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No API key found. Set GROQ_API_KEY (free at console.groq.com) "
            "or OPENAI_API_KEY in your environment."
        )

    base_url = "https://api.groq.com/openai/v1" if os.environ.get("GROQ_API_KEY") else None
    return OpenAI(api_key=api_key, base_url=base_url)


SYSTEM_PROMPT = """You are an expert email triage assistant. For each email you receive, you must decide:
1. action_type: one of [classify, prioritize, draft_reply, archive, escalate]
2. category: one of [billing, technical, general, spam, complaint, praise, urgent_request]
3. urgency: one of [low, medium, high, critical]
4. reply_text: (if action_type is draft_reply) a short professional reply
5. reason: brief explanation of your decision

Rules:
- SPAM emails → always archive, urgency=low, no reply
- CRITICAL urgency issues → escalate immediately
- Complaints from long-term customers → escalate with reason
- Technical billing issues → prioritize
- Respond ONLY with a JSON object. No markdown, no explanation outside the JSON.

Example response:
{
  "action_type": "classify",
  "category": "technical",
  "urgency": "high",
  "reply_text": null,
  "reason": "API rate limit issue affecting active developer"
}"""


def build_user_prompt(email_data: dict) -> str:
    email = email_data.get("current_email", {})
    if not email:
        return "No email to process."
    return f"""Triage this email:

Subject: {email.get('subject', 'N/A')}
From: {email.get('sender', 'N/A')}
Time: {email.get('timestamp', 'N/A')}
Body:
{email.get('body', 'N/A')}

Respond with a JSON action object."""


def parse_llm_response(response_text: str) -> Action:
    """Parse LLM JSON response into an Action object."""
    try:
        # Strip markdown fences if present
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        data = json.loads(clean)
        return Action(
            action_type=ActionType(data.get("action_type", "classify")),
            category=EmailCategory(data["category"]) if data.get("category") else None,
            urgency=UrgencyLevel(data["urgency"]) if data.get("urgency") else None,
            reply_text=data.get("reply_text"),
            reason=data.get("reason")
        )
    except Exception as e:
        # Fallback action on parse failure
        return Action(action_type=ActionType.CLASSIFY, reason=f"Parse error: {e}")


def run_task(client, task_id: str, model: str = "llama3-8b-8192", verbose: bool = True) -> float:
    env = EmailTriageEnv()
    obs = env.reset(task_id=task_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {obs.current_task} ({task_id})")
        print(f"{'='*60}")

    episode_done = False
    step = 0

    while not episode_done:
        if obs.current_email is None:
            break

        user_prompt = build_user_prompt(obs.model_dump())

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw = response.choices[0].message.content
        except Exception as e:
            if verbose:
                print(f"  [Step {step+1}] API error: {e}")
            raw = '{"action_type": "classify", "category": "general", "urgency": "medium"}'

        action = parse_llm_response(raw)

        obs, reward, episode_done, info = env.step(action)
        step += 1

        if verbose:
            print(f"  [Step {step}] Email: {info['email_id']} | Score: {reward.value:.4f} | {reward.feedback}")

    final_score = env.get_final_score()
    if verbose:
        print(f"\n  Final Score for {task_id}: {final_score:.4f}")

    return final_score


def main():
    parser = argparse.ArgumentParser(description="Baseline inference for Email Triage OpenEnv")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--model", default="llama3-8b-8192", help="Model to use (Groq model name)")
    args = parser.parse_args()

    try:
        client = get_client()
    except EnvironmentError as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"ERROR: {e}")
        sys.exit(1)

    verbose = not args.json
    results = {}

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        score = run_task(client, task_id, model=args.model, verbose=verbose)
        results[task_id] = {
            "score": score,
            "task_name": TASKS[task_id]["name"],
            "difficulty": TASKS[task_id]["difficulty"]
        }

    if args.json:
        print(json.dumps({
            "baseline_scores": results,
            "model": args.model,
            "overall_average": round(sum(r["score"] for r in results.values()) / len(results), 4)
        }))
    else:
        print(f"\n{'='*60}")
        print("BASELINE SUMMARY")
        print(f"{'='*60}")
        for task_id, res in results.items():
            print(f"  {res['task_name']} ({res['difficulty']}): {res['score']:.4f}")
        avg = sum(r["score"] for r in results.values()) / len(results)
        print(f"\n  Overall Average: {avg:.4f}")


if __name__ == "__main__":
    main()
