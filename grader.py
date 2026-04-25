"""
grader.py — Deterministic grader for agent responses.

Scoring dimensions (each 0–1, averaged for base_score):
  • tone_score        — empathy, no hostile/dismissive language
  • correctness_score — required keywords present, no forbidden phrases
  • resolution_score  — commits to a clear action / resolution path

Final reward formula:
  reward = base_score + improvement_bonus − repeated_mistake_penalty

  improvement_bonus:      +0.10 per mistake from memory that was avoided this turn
  repeated_mistake_penalty: −0.15 per mistake from memory that was repeated this turn
"""

from __future__ import annotations

import re
from typing import List, Set

from models import GraderResult
from tasks import Task


# ---------------------------------------------------------------------------
# Tone vocabulary
# ---------------------------------------------------------------------------

# Phrases that indicate genuine empathy / apology
POSITIVE_TONE_SIGNALS: List[str] = [
    r"\bapologi[sz]e\b",
    r"\bsorry\b",
    r"\bunderstand\b",
    r"\bfrustrat",
    r"\bsincerely\b",
    r"\bvalued?\b",
    r"\bappreciat",
    r"\bthank you for\b",
    r"\bwe hear you\b",
    r"\bwe take this seriously\b",
]

# Phrases that hurt tone (hostile, dismissive, blame-shifting)
NEGATIVE_TONE_SIGNALS: List[str] = [
    r"\bnot our (fault|problem|responsibility)\b",
    r"\byou should have\b",
    r"\bread the (terms|policy|fine print)\b",
    r"\bnothing we can do\b",
    r"\bthat's (impossible|not possible)\b",
    r"\bI (can't|cannot) help\b",
    r"\bwe don't do that\b",
    r"\bcalm down\b",
]


def _matches(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, re.IGNORECASE))


def _count_matches(patterns: List[str], text: str) -> int:
    return sum(1 for p in patterns if _matches(p, text))


# ---------------------------------------------------------------------------
# Mistake tag detection
# ---------------------------------------------------------------------------

MISTAKE_DETECTORS: dict[str, str] = {
    "missing_apology":      r"\b(apologi[sz]e|sorry)\b",          # must MATCH
    "hostile_tone":         r"\b(calm down|not our fault|you should have)\b",  # must NOT match
    "missing_resolution":   r"\b(resolv|fix|address|investigat|refund|replac|escalat|follow.?up)\b",  # must MATCH
    "forbidden_phrase":     "__forbidden__",                        # handled separately
    "missing_empathy":      r"\b(understand|frustrat|hear you|we.*sorry)\b",   # must MATCH
}

# Patterns that indicate a mistake (negated — absence of required things)
MUST_MATCH = {"missing_apology", "missing_resolution", "missing_empathy"}
MUST_NOT_MATCH = {"hostile_tone"}


def detect_mistakes(response: str, task: Task) -> List[str]:
    """Return a list of short mistake-tag strings for the given response."""
    mistakes: List[str] = []

    for tag, pattern in MISTAKE_DETECTORS.items():
        if tag == "forbidden_phrase":
            for phrase in task.forbidden_phrases:
                if phrase.lower() in response.lower():
                    mistakes.append(f"forbidden_phrase:{phrase}")
        elif tag in MUST_MATCH:
            if not _matches(pattern, response):
                mistakes.append(tag)
        elif tag in MUST_NOT_MATCH:
            if _matches(pattern, response):
                mistakes.append(tag)

    # Check required keywords
    for kw in task.required_keywords:
        if kw.lower() not in response.lower():
            mistakes.append(f"missing_keyword:{kw}")

    return mistakes


# ---------------------------------------------------------------------------
# Tone scorer
# ---------------------------------------------------------------------------

def score_tone(response: str) -> float:
    pos = _count_matches(POSITIVE_TONE_SIGNALS, response)
    neg = _count_matches(NEGATIVE_TONE_SIGNALS, response)

    # Base: need at least 2 positive signals for full marks
    pos_ratio = min(pos / 2.0, 1.0)
    # Penalise negatives (each one removes 0.3, floored at 0)
    penalty = neg * 0.3
    return max(0.0, round(pos_ratio - penalty, 4))


# ---------------------------------------------------------------------------
# Correctness scorer
# ---------------------------------------------------------------------------

def score_correctness(response: str, task: Task) -> float:
    total = len(task.required_keywords) + len(task.forbidden_phrases)
    if total == 0:
        return 1.0

    hits = 0
    # Required keywords present → +1 each
    for kw in task.required_keywords:
        if kw.lower() in response.lower():
            hits += 1

    # Forbidden phrases absent → +1 each
    for phrase in task.forbidden_phrases:
        if phrase.lower() not in response.lower():
            hits += 1

    return round(hits / total, 4)


# ---------------------------------------------------------------------------
# Resolution scorer
# ---------------------------------------------------------------------------

RESOLUTION_PATTERNS: List[str] = [
    r"\b(will|will be|going to)\b.{0,40}\b(refund|replac|investigat|escalat|fix|address|resolv)\b",
    r"\b(within|in)\s+\d+\s+(business\s+)?(day|hour|minute)s?\b",
    r"\bcase\s+(number|#|id|reference)\b",
    r"\bticket\b.{0,20}\b(creat|open|rais|submit)\b",
    r"\b(contact|reach).{0,20}\b(you|back)\b",
    r"\b(next steps?|action)\b",
    r"\b(refund|replacement|credit).{0,30}\b(process|initiat|issu)\b",
    r"\bescheat\b|\bescalat\b",
]


def score_resolution(response: str) -> float:
    matched = _count_matches(RESOLUTION_PATTERNS, response)
    # Full marks if 3+ resolution signals found; partial otherwise
    return round(min(matched / 3.0, 1.0), 4)


# ---------------------------------------------------------------------------
# Actionability scorer
# ---------------------------------------------------------------------------

ACTIONABILITY_PATTERNS: List[str] = [
    r"\bi will\b", 
    r"\bwithin \w+\b", 
    r"\bnext step\b", 
    r"\bimmediately\b", 
    r"\bright away\b",
    r"\bi am going to\b",
]

def score_actionability(response: str) -> float:
    matched = _count_matches(ACTIONABILITY_PATTERNS, response)
    if matched >= 2:
        return 1.0
    elif matched == 1:
        return 0.5
    return 0.2


# ---------------------------------------------------------------------------
# Policy Compliance scorer
# ---------------------------------------------------------------------------

def score_policy(response: str, task: Task) -> float:
    # Safely penalize forbidden policy violations without destroying the whole score
    for phrase in task.forbidden_phrases:
        if phrase.lower() in response.lower():
            return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Main grader
# ---------------------------------------------------------------------------

def grade(
    response: str,
    task: Task,
    past_mistakes: List[str],
) -> GraderResult:
    """
    Grade a response and return a GraderResult with the final reward.

    Args:
        response:       The agent's customer-facing reply.
        task:           The task definition (has required/forbidden info).
        past_mistakes:  Mistake tags stored in memory from previous episodes.

    Returns:
        GraderResult with all sub-scores and the final reward.
    """
    tone = score_tone(response)
    correctness = score_correctness(response, task)
    resolution = score_resolution(response)
    actionability = score_actionability(response)
    policy = score_policy(response, task)

    base_score = round(
        (0.30 * tone) +
        (0.25 * correctness) +
        (0.20 * resolution) +
        (0.15 * actionability) +
        (0.10 * policy),
        4
    )

    # Detect mistakes in this response
    current_mistakes: List[str] = detect_mistakes(response, task)
    current_set: Set[str] = set(current_mistakes)
    past_set: Set[str] = set(past_mistakes)

    # Improvement bonus: past mistakes that are NOT in current mistakes
    avoided = past_set - current_set
    improvement_bonus = round(len(avoided) * 0.10, 4)

    # Repeated mistake penalty: past mistakes that ARE in current mistakes
    repeated = past_set & current_set
    repeated_mistake_penalty = round(len(repeated) * 0.15, 4)

    final_reward = round(base_score + improvement_bonus - repeated_mistake_penalty, 4)
    # Clamp to [-0.5, 1.5] to allow slight over/under without being unbounded
    final_reward = max(-0.5, min(1.5, final_reward))

    return GraderResult(
        tone_score=tone,
        correctness_score=correctness,
        resolution_score=resolution,
        actionability_score=actionability,
        policy_compliance_score=policy,
        base_score=base_score,
        improvement_bonus=improvement_bonus,
        repeated_mistake_penalty=repeated_mistake_penalty,
        final_reward=final_reward,
        mistakes_found=current_mistakes,
    )
