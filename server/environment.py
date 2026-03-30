"""
Email Triage OpenEnv Environment
An AI agent must triage a queue of emails: classify urgency, assign category, and draft reply starters.
"""

import random
import time
from typing import Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────────

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EmailCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    SPAM = "spam"
    COMPLAINT = "complaint"
    PRAISE = "praise"
    URGENT_REQUEST = "urgent_request"

class ActionType(str, Enum):
    CLASSIFY = "classify"
    PRIORITIZE = "prioritize"
    DRAFT_REPLY = "draft_reply"
    ARCHIVE = "archive"
    ESCALATE = "escalate"


# ── Pydantic Models ────────────────────────────────────────────────────────────

class Email(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    ground_truth_category: EmailCategory
    ground_truth_urgency: UrgencyLevel
    expected_action: ActionType
    reply_keywords: list[str] = Field(default_factory=list)  # words a good reply should contain


class Observation(BaseModel):
    current_email: Optional[Email]
    queue_size: int
    processed_count: int
    current_task: str
    task_id: str
    step_number: int
    feedback: str = ""


class Action(BaseModel):
    action_type: ActionType
    category: Optional[EmailCategory] = None
    urgency: Optional[UrgencyLevel] = None
    reply_text: Optional[str] = None
    reason: Optional[str] = None


class Reward(BaseModel):
    value: float
    breakdown: dict[str, float]
    feedback: str


class EnvState(BaseModel):
    task_id: str
    step_number: int
    queue_size: int
    processed: int
    done: bool
    total_reward: float
    task_name: str


# ── Email Dataset ──────────────────────────────────────────────────────────────

EMAIL_DATASET = [
    # EASY emails — obvious signals
    Email(
        id="e001", subject="URGENT: Server is DOWN!!!",
        sender="ops_team@company.com",
        body="Our production server has been down for 30 minutes. Customers cannot access the platform. We need immediate help. Every minute costs us $5000.",
        timestamp="2024-01-15 09:00:00",
        ground_truth_category=EmailCategory.TECHNICAL,
        ground_truth_urgency=UrgencyLevel.CRITICAL,
        expected_action=ActionType.ESCALATE,
        reply_keywords=["immediate", "team", "escalate", "priority"]
    ),
    Email(
        id="e002", subject="Congratulations on your purchase!",
        sender="noreply@spamstore.biz",
        body="You've won a FREE iPhone! Click here to claim your prize now!!! Limited time offer!!!",
        timestamp="2024-01-15 09:05:00",
        ground_truth_category=EmailCategory.SPAM,
        ground_truth_urgency=UrgencyLevel.LOW,
        expected_action=ActionType.ARCHIVE,
        reply_keywords=[]
    ),
    Email(
        id="e003", subject="Invoice #4521 - Payment Overdue",
        sender="billing@vendor.com",
        body="Your invoice #4521 for $2,340 is now 30 days overdue. Please process payment at your earliest convenience to avoid service interruption.",
        timestamp="2024-01-15 09:10:00",
        ground_truth_category=EmailCategory.BILLING,
        ground_truth_urgency=UrgencyLevel.HIGH,
        expected_action=ActionType.PRIORITIZE,
        reply_keywords=["payment", "invoice", "process", "team"]
    ),
    # MEDIUM emails — require reading carefully
    Email(
        id="e004", subject="Quick question about the API",
        sender="developer@startup.io",
        body="Hi, I've been using your API for a few weeks and it's mostly great. However, I noticed that when I send more than 100 requests per minute, I start getting 429 errors even though my plan says 'unlimited'. Can you clarify? Thanks.",
        timestamp="2024-01-15 09:15:00",
        ground_truth_category=EmailCategory.TECHNICAL,
        ground_truth_urgency=UrgencyLevel.MEDIUM,
        expected_action=ActionType.CLASSIFY,
        reply_keywords=["rate", "limit", "plan", "clarify", "check"]
    ),
    Email(
        id="e005", subject="Feedback on your service",
        sender="customer@gmail.com",
        body="I wanted to say that your support team went above and beyond last week when my account had issues. Sarah was incredibly helpful and patient. Please pass on my appreciation!",
        timestamp="2024-01-15 09:20:00",
        ground_truth_category=EmailCategory.PRAISE,
        ground_truth_urgency=UrgencyLevel.LOW,
        expected_action=ActionType.CLASSIFY,
        reply_keywords=["thank", "appreciate", "Sarah", "forward"]
    ),
    Email(
        id="e006", subject="Re: Re: Re: Meeting rescheduled again",
        sender="partner@business.com",
        body="Hi, I know we've moved this meeting 3 times already, but I need to reschedule once more. Our CEO just called an all-hands for Tuesday. Can we do Wednesday instead? Really sorry about this.",
        timestamp="2024-01-15 09:25:00",
        ground_truth_category=EmailCategory.GENERAL,
        ground_truth_urgency=UrgencyLevel.MEDIUM,
        expected_action=ActionType.DRAFT_REPLY,
        reply_keywords=["Wednesday", "confirm", "calendar", "reschedule"]
    ),
    # HARD emails — nuanced, ambiguous
    Email(
        id="e007", subject="Disappointed with recent changes",
        sender="longtime_customer@email.com",
        body="I've been a customer for 7 years and I'm very disappointed with the new pricing structure. My bill went from $99/month to $149/month without adequate notice. I'm considering switching to your competitor. I'd like to understand what value I'm getting for this increase before I make any decisions.",
        timestamp="2024-01-15 09:30:00",
        ground_truth_category=EmailCategory.COMPLAINT,
        ground_truth_urgency=UrgencyLevel.HIGH,
        expected_action=ActionType.ESCALATE,
        reply_keywords=["apologize", "value", "retention", "manager", "review", "pricing"]
    ),
    Email(
        id="e008", subject="Partnership inquiry",
        sender="bd@techcorp.com",
        body="Hello, I'm the VP of Business Development at TechCorp. We're a Series B startup with 50,000 users and we think there could be a great synergy between our platforms. We'd love to explore a technical integration that could benefit both our user bases. Would you be open to a call next week?",
        timestamp="2024-01-15 09:35:00",
        ground_truth_category=EmailCategory.GENERAL,
        ground_truth_urgency=UrgencyLevel.MEDIUM,
        expected_action=ActionType.DRAFT_REPLY,
        reply_keywords=["interested", "call", "schedule", "team", "explore"]
    ),
    Email(
        id="e009", subject="Account locked - urgent help needed",
        sender="user123@personal.com",
        body="My account got locked and I cannot access any of my data. I have a presentation to a major client in 2 hours that requires files stored in your system. I've tried resetting my password 3 times. Please help ASAP.",
        timestamp="2024-01-15 09:40:00",
        ground_truth_category=EmailCategory.TECHNICAL,
        ground_truth_urgency=UrgencyLevel.CRITICAL,
        expected_action=ActionType.ESCALATE,
        reply_keywords=["immediate", "unlock", "support", "team", "escalate", "2 hours"]
    ),
]


# ── Task Definitions ───────────────────────────────────────────────────────────

TASKS = {
    "task_easy": {
        "name": "Basic Email Classification",
        "description": "Classify the urgency and category of 3 emails with obvious signals.",
        "difficulty": "easy",
        "email_ids": ["e001", "e002", "e003"],
        "required_actions": [ActionType.CLASSIFY, ActionType.PRIORITIZE, ActionType.ARCHIVE, ActionType.ESCALATE],
        "max_steps": 6,
    },
    "task_medium": {
        "name": "Email Triage with Replies",
        "description": "Triage 3 emails requiring careful reading; draft a reply for one.",
        "difficulty": "medium",
        "email_ids": ["e004", "e005", "e006"],
        "required_actions": [ActionType.CLASSIFY, ActionType.DRAFT_REPLY],
        "max_steps": 9,
    },
    "task_hard": {
        "name": "Full Inbox Management",
        "description": "Handle 3 nuanced emails requiring complex decisions, escalations, and diplomatic replies.",
        "difficulty": "hard",
        "email_ids": ["e007", "e008", "e009"],
        "required_actions": [ActionType.ESCALATE, ActionType.DRAFT_REPLY],
        "max_steps": 12,
    },
}


# ── Grader ─────────────────────────────────────────────────────────────────────

def grade_action(email: Email, action: Action, task_difficulty: str) -> tuple[float, str]:
    """Returns (score 0.0-1.0, feedback string)"""
    score = 0.0
    feedback_parts = []

    # Category match
    if action.action_type == ActionType.CLASSIFY and action.category:
        if action.category == email.ground_truth_category:
            score += 0.35
            feedback_parts.append("✓ Category correct")
        else:
            feedback_parts.append(f"✗ Category wrong (expected {email.ground_truth_category.value})")

    # Urgency match
    if action.urgency:
        if action.urgency == email.ground_truth_urgency:
            score += 0.30
            feedback_parts.append("✓ Urgency correct")
        elif abs(list(UrgencyLevel).index(action.urgency) - list(UrgencyLevel).index(email.ground_truth_urgency)) == 1:
            score += 0.15  # partial credit for being one level off
            feedback_parts.append("~ Urgency close (1 level off)")
        else:
            feedback_parts.append(f"✗ Urgency wrong (expected {email.ground_truth_urgency.value})")

    # Action type match
    if action.action_type == email.expected_action:
        score += 0.25
        feedback_parts.append("✓ Action type correct")
    else:
        feedback_parts.append(f"✗ Action type wrong (expected {email.expected_action.value})")

    # Reply quality (for draft_reply actions)
    if action.action_type == ActionType.DRAFT_REPLY and action.reply_text and email.reply_keywords:
        reply_lower = action.reply_text.lower()
        matched = sum(1 for kw in email.reply_keywords if kw.lower() in reply_lower)
        reply_score = (matched / len(email.reply_keywords)) * 0.10
        score += reply_score
        feedback_parts.append(f"Reply keywords matched: {matched}/{len(email.reply_keywords)}")

    # Reason provided bonus (hard task)
    if task_difficulty == "hard" and action.reason:
        score += 0.05
        feedback_parts.append("✓ Reasoning provided")

    # Penalize wrong escalation for spam
    if email.ground_truth_category == EmailCategory.SPAM and action.action_type == ActionType.ESCALATE:
        score = max(0.0, score - 0.3)
        feedback_parts.append("✗ Penalty: escalated spam email")

    return min(1.0, score), " | ".join(feedback_parts)


# ── Main Environment Class ─────────────────────────────────────────────────────

class EmailTriageEnv:
    def __init__(self):
        self._email_map = {e.id: e for e in EMAIL_DATASET}
        self._current_task_id: Optional[str] = None
        self._task_config: Optional[dict] = None
        self._email_queue: list[Email] = []
        self._current_email_index: int = 0
        self._step_number: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._action_scores: list[float] = []
        self._feedback_log: list[str] = []

    def reset(self, task_id: str = "task_easy") -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Choose from {list(TASKS.keys())}")

        self._current_task_id = task_id
        self._task_config = TASKS[task_id]
        self._email_queue = [self._email_map[eid] for eid in self._task_config["email_ids"]]
        self._current_email_index = 0
        self._step_number = 0
        self._total_reward = 0.0
        self._done = False
        self._action_scores = []
        self._feedback_log = []

        return Observation(
            current_email=self._email_queue[0],
            queue_size=len(self._email_queue),
            processed_count=0,
            current_task=self._task_config["name"],
            task_id=task_id,
            step_number=0,
            feedback=f"Task started: {self._task_config['description']}"
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._current_task_id is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_number += 1
        current_email = self._email_queue[self._current_email_index]
        task_difficulty = self._task_config["difficulty"]

        # Grade the action
        score, feedback_str = grade_action(current_email, action, task_difficulty)
        self._action_scores.append(score)
        self._feedback_log.append(feedback_str)
        self._total_reward += score

        # Build reward
        reward = Reward(
            value=round(score, 4),
            breakdown={
                "action_score": round(score, 4),
                "running_average": round(self._total_reward / len(self._action_scores), 4),
            },
            feedback=feedback_str
        )

        # Advance queue
        self._current_email_index += 1
        max_steps = self._task_config["max_steps"]

        if self._current_email_index >= len(self._email_queue) or self._step_number >= max_steps:
            self._done = True
            next_email = None
            obs_feedback = f"Episode complete! Average score: {round(self._total_reward / len(self._action_scores), 4)}"
        else:
            next_email = self._email_queue[self._current_email_index]
            obs_feedback = feedback_str

        obs = Observation(
            current_email=next_email,
            queue_size=len(self._email_queue),
            processed_count=self._current_email_index,
            current_task=self._task_config["name"],
            task_id=self._current_task_id,
            step_number=self._step_number,
            feedback=obs_feedback
        )

        info = {
            "email_id": current_email.id,
            "step": self._step_number,
            "scores_so_far": self._action_scores,
            "done": self._done
        }

        return obs, reward, self._done, info

    def state(self) -> EnvState:
        return EnvState(
            task_id=self._current_task_id or "none",
            step_number=self._step_number,
            queue_size=len(self._email_queue),
            processed=self._current_email_index,
            done=self._done,
            total_reward=round(self._total_reward, 4),
            task_name=self._task_config["name"] if self._task_config else "none"
        )

    def get_final_score(self) -> float:
        if not self._action_scores:
            return 0.0
        return round(sum(self._action_scores) / len(self._action_scores), 4)

