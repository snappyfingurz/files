"""
feedback.py — Structured feedback generator for the Self-Improving Customer Support Agent.

Produces human-readable, actionable critique strings that are stored in memory
and surfaced to the agent in the next episode's Observation.

Critique structure:
  [SCORE: x.xx] <summary sentence>
  STRENGTHS: <comma-separated list or "none">
  ISSUES:    <comma-separated list or "none">
  ADVICE:    <one concrete improvement tip>
"""

from __future__ import annotations

from typing import List

from models import GraderResult
from tasks import Task


# ---------------------------------------------------------------------------
# Strength detectors
# ---------------------------------------------------------------------------

def _detect_strengths(response: str, result: GraderResult, task: Task) -> List[str]:
    strengths: List[str] = []

    if result.tone_score >= 0.7:
        strengths.append("empathetic tone")
    if result.correctness_score >= 0.8:
        strengths.append("covered required content")
    if result.resolution_score >= 0.7:
        strengths.append("clear resolution commitment")
    if result.improvement_bonus > 0:
        n = round(result.improvement_bonus / 0.10)
        strengths.append(f"avoided {n} past mistake(s)")
    if len(response.split()) >= 80:
        strengths.append("sufficiently detailed reply")

    return strengths


# ---------------------------------------------------------------------------
# Issue descriptors (human-readable versions of mistake tags)
# ---------------------------------------------------------------------------

_MISTAKE_DESCRIPTIONS: dict[str, str] = {
    "missing_apology":   "no apology or acknowledgement of fault",
    "hostile_tone":      "dismissive or hostile language detected",
    "missing_resolution":"no concrete resolution or follow-up action stated",
    "missing_empathy":   "lacked empathy or validation of customer frustration",
}


def _describe_mistakes(mistakes: List[str]) -> List[str]:
    descriptions: List[str] = []
    for m in mistakes:
        if m in _MISTAKE_DESCRIPTIONS:
            descriptions.append(_MISTAKE_DESCRIPTIONS[m])
        elif m.startswith("forbidden_phrase:"):
            phrase = m.split(":", 1)[1]
            descriptions.append(f"used forbidden phrase: '{phrase}'")
        elif m.startswith("missing_keyword:"):
            kw = m.split(":", 1)[1]
            descriptions.append(f"did not mention '{kw}'")
        else:
            descriptions.append(m.replace("_", " "))
    return descriptions


# ---------------------------------------------------------------------------
# Advice generator
# ---------------------------------------------------------------------------

def _generate_advice(result: GraderResult, task: Task) -> str:
    """Pick the single most impactful piece of advice for this result."""
    # Prioritise the worst dimension
    scores = {
        "tone":        result.tone_score,
        "correctness": result.correctness_score,
        "resolution":  result.resolution_score,
    }
    worst = min(scores, key=scores.get)  # type: ignore[arg-type]

    if worst == "tone":
        return (
            "Open with a genuine apology ('I'm truly sorry…') and mirror the "
            "customer's concern before moving to solutions."
        )
    if worst == "correctness":
        missing_kws = [
            m.split(":", 1)[1]
            for m in result.mistakes_found
            if m.startswith("missing_keyword:")
        ]
        if missing_kws:
            return (
                f"Ensure you explicitly address: {', '.join(missing_kws)}. "
                "These are required for a complete response."
            )
        forbidden = [
            m.split(":", 1)[1]
            for m in result.mistakes_found
            if m.startswith("forbidden_phrase:")
        ]
        if forbidden:
            return (
                f"Remove or rephrase the following: {', '.join(forbidden)}. "
                "These violate company policy."
            )
        return "Review the task's required content areas and cover each one explicitly."

    # resolution is worst
    return (
        "State a concrete next step with a timeframe — e.g. "
        "'I will escalate this to our billing team and you'll hear back within "
        "2 business days.' Vague promises are not enough."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_feedback(
    response: str,
    result: GraderResult,
    task: Task,
) -> str:
    """
    Generate a structured feedback string for storing in memory.

    Args:
        response: The agent's customer-facing reply.
        result:   The GraderResult from grader.grade().
        task:     The task that was being solved.

    Returns:
        A multi-line feedback string ready to be stored and surfaced in the
        next episode's Observation.past_feedback list.
    """
    strengths = _detect_strengths(response, result, task)
    issues = _describe_mistakes(result.mistakes_found)
    advice = _generate_advice(result, task)

    strengths_str = ", ".join(strengths) if strengths else "none identified"
    issues_str = "; ".join(issues) if issues else "none"

    # One-line summary
    if result.base_score >= 0.8:
        summary = f"Good response on task '{task.task_id}' ({task.difficulty})."
    elif result.base_score >= 0.5:
        summary = f"Adequate but improvable response on task '{task.task_id}' ({task.difficulty})."
    else:
        summary = f"Poor response on task '{task.task_id}' ({task.difficulty}) — significant issues."

    feedback = (
        f"[SCORE: {result.final_reward:.2f}] {summary}\n"
        f"STRENGTHS: {strengths_str}\n"
        f"ISSUES:    {issues_str}\n"
        f"ADVICE:    {advice}"
    )
    return feedback
