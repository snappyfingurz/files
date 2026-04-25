"""
tasks.py — Task definitions for the Self-Improving Customer Support Agent environment.

Three difficulty levels:
  easy   — simple product complaint with a clear fix.
  medium — customer has multiple overlapping issues.
  hard   — customer demands something that violates company policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Task:
    task_id: str
    difficulty: str                      # "easy" | "medium" | "hard"
    customer_message: str
    # Ground-truth keywords the grader uses for correctness checks
    required_keywords: List[str] = field(default_factory=list)
    # Things that MUST NOT appear (policy violations, wrong offers, etc.)
    forbidden_phrases: List[str] = field(default_factory=list)
    # Short description of the expected resolution path
    ideal_resolution: str = ""
    # Policy constraints the agent must respect
    policy_notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {

    # ------------------------------------------------------------------
    # EASY — single clear issue, standard refund path
    # ------------------------------------------------------------------
    "easy_001": Task(
        task_id="easy_001",
        difficulty="easy",
        customer_message=(
            "I ordered a pair of headphones two weeks ago and they STILL haven't "
            "arrived! This is absolutely unacceptable. I need this resolved TODAY "
            "or I'm disputing the charge with my bank. Order #78342."
        ),
        required_keywords=[
            "apologize",      # Must acknowledge the frustration
            "order",          # Must reference the order
            "track",          # Must offer tracking info or investigation
            "resolve",        # Must commit to a resolution path
        ],
        forbidden_phrases=[
            "not our fault",
            "nothing we can do",
            "read the terms",
        ],
        ideal_resolution=(
            "Sincerely apologise, look up order #78342, share current tracking "
            "status, and offer a replacement shipment or full refund if the "
            "package is confirmed lost."
        ),
        policy_notes=[
            "Refunds are available for orders delayed beyond 10 business days.",
            "Always provide a case or ticket reference number.",
        ],
    ),

    # ------------------------------------------------------------------
    # MEDIUM — customer has three simultaneous problems
    # ------------------------------------------------------------------
    "medium_001": Task(
        task_id="medium_001",
        difficulty="medium",
        customer_message=(
            "I am FURIOUS. Three things went wrong this week:\n"
            "1. My subscription was double-charged on the 15th — $49.99 taken twice.\n"
            "2. Your app crashes every time I try to access my account settings.\n"
            "3. I submitted a support ticket 5 days ago (ticket #5512) and received "
            "zero response.\n"
            "I'm one step away from cancelling and leaving a public review everywhere "
            "I can find. Fix ALL of this NOW."
        ),
        required_keywords=[
            "apologize",
            "charge",          # Address the billing issue
            "refund",          # Commit to refund the duplicate charge
            "bug",             # Acknowledge the technical crash
            "ticket",          # Reference ticket #5512
            "escalate",        # Commit to escalating the ignored ticket
        ],
        forbidden_phrases=[
            "only one issue at a time",
            "contact billing separately",
            "not my department",
        ],
        ideal_resolution=(
            "Apologise for all three issues together. Confirm the duplicate charge "
            "will be refunded within 3–5 business days. Acknowledge the app crash "
            "and provide a workaround or escalate to engineering. Pull up ticket "
            "#5512 and personally ensure a response within 24 hours."
        ),
        policy_notes=[
            "Duplicate charges must be refunded within 5 business days.",
            "Unacknowledged tickets older than 3 days require supervisor escalation.",
            "Always address every issue the customer raised — never triage away items.",
        ],
    ),

    # ------------------------------------------------------------------
    # HARD — customer demands a policy exception that cannot be granted
    # ------------------------------------------------------------------
    "hard_001": Task(
        task_id="hard_001",
        difficulty="hard",
        customer_message=(
            "I want a full refund for my annual plan. I signed up 11 months ago and "
            "used the service almost every day, but I've decided I don't need it "
            "anymore. Your website says '30-day money-back guarantee' — I know that "
            "window passed, but I'm a long-time customer and I DEMAND you honour it "
            "anyway. If you don't, I'll dispute the charge as fraud. Threatening a "
            "chargeback is not a bluff."
        ),
        required_keywords=[
            "apologize",
            "understand",       # Empathise with the frustration
            "policy",           # Cite the actual policy clearly
            "alternative",      # Offer an alternative (partial credit, pause, etc.)
        ],
        forbidden_phrases=[
            "full refund",          # Cannot promise a full refund after 30 days
            "we will refund",       # Any unconditional refund promise is forbidden
            "honour your request",  # Policy exception cannot be granted
            "fraud",                # Do not use the word 'fraud' back at the customer
        ],
        ideal_resolution=(
            "Empathise with the customer's frustration. Clearly and calmly explain "
            "that the 30-day guarantee has expired and a full refund is outside "
            "policy. Offer concrete alternatives: account pause for up to 3 months, "
            "a pro-rated credit toward a future plan, or a discounted renewal. "
            "Do not capitulate to the chargeback threat, but do not be dismissive."
        ),
        policy_notes=[
            "30-day money-back guarantee is a hard limit — no exceptions after expiry.",
            "Agents may offer: account pause (up to 90 days) or 20% renewal credit.",
            "Never threaten legal action in response to a chargeback threat.",
            "Never use the word 'fraud' when speaking to the customer.",
        ],
    ),
}


def get_task(task_id: str) -> Task:
    """Retrieve a task by ID, raising KeyError if not found."""
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: '{task_id}'. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def all_task_ids() -> List[str]:
    return list(TASKS.keys())


def tasks_by_difficulty(difficulty: str) -> List[Task]:
    return [t for t in TASKS.values() if t.difficulty == difficulty]
