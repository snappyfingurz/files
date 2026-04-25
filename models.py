"""
models.py — Pydantic data models for the Self-Improving Customer Support Agent environment.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """What the agent submits each step."""

    response: str = Field(
        ...,
        description="The customer-facing reply the agent sends.",
    )
    reflection: str = Field(
        ...,
        description=(
            "The agent's internal reasoning: what it noticed, what it tried to "
            "improve compared with past mistakes, and why it chose this approach."
        ),
    )


# ---------------------------------------------------------------------------
# Observation (returned by reset / step)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Everything the agent can see at the start of a step."""

    task_id: str = Field(
        ...,
        description="The unique identifier for the current task.",
    )
    customer_message: str = Field(
        ...,
        description="The current angry-customer message the agent must address.",
    )
    past_feedback: List[str] = Field(
        default_factory=list,
        description="Structured critique strings from previous episodes.",
    )
    past_mistakes: List[str] = Field(
        default_factory=list,
        description="Short mistake tags remembered from previous episodes.",
    )


# ---------------------------------------------------------------------------
# State (full internal state, returned by state())
# ---------------------------------------------------------------------------

class EpisodeRecord(BaseModel):
    """A single completed episode stored in history."""

    task_id: str
    difficulty: str
    customer_message: str
    agent_response: str
    agent_reflection: str
    score: float
    feedback: str
    mistakes: List[str]


class State(BaseModel):
    """Full environment state (for inspection / serialisation)."""

    history: List[EpisodeRecord] = Field(default_factory=list)
    score: float = Field(default=0.0, description="Cumulative reward so far.")
    step_count: int = Field(default=0)
    current_task_id: Optional[str] = Field(default=None)
    done: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Grader outputs (internal)
# ---------------------------------------------------------------------------

class GraderResult(BaseModel):
    """Detailed breakdown produced by the deterministic grader."""

    tone_score: float = Field(ge=0.0, le=1.0)
    correctness_score: float = Field(ge=0.0, le=1.0)
    resolution_score: float = Field(ge=0.0, le=1.0)
    actionability_score: float = Field(ge=0.0, le=1.0)
    policy_compliance_score: float = Field(ge=0.0, le=1.0)
    conciseness_score: float = Field(ge=0.0, le=1.0)
    clarity_score: float = Field(ge=0.0, le=1.0)
    base_score: float = Field(ge=0.0, le=1.0)
    improvement_bonus: float = Field(ge=0.0)
    repeated_mistake_penalty: float = Field(ge=0.0)
    final_reward: float
    mistakes_found: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step result (returned by step())
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Everything returned after the agent takes an action."""

    observation: Observation
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)
