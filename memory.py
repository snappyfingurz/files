"""
memory.py — Persistent memory system for the Self-Improving Customer Support Agent.

Stores:
  • Mistake tags       (short identifiers, e.g. "missing_apology")
  • Structured feedback strings from the feedback generator
  • Per-task score history for trend analysis

Memory is kept in RAM by default (dict) and can be optionally serialised
to / from a JSON file for multi-process / resumable setups.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional


class AgentMemory:
    """
    Lightweight key-value memory store for mistakes and feedback.

    All lists are de-duplicated on insertion so the context fed to the agent
    remains concise. Ordering is preserved (most-recent first for feedback,
    alphabetical for mistake tags).
    """

    def __init__(self, max_feedback_entries: int = 10, max_mistake_entries: int = 20):
        self._max_feedback = max_feedback_entries
        self._max_mistakes = max_mistake_entries

        # Global across all tasks
        self._mistakes: List[str] = []      # ordered, deduped
        self._feedback: List[str] = []      # most-recent first, capped

        # Per-task score history  { task_id: [score, score, ...] }
        self._score_history: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Mistake management
    # ------------------------------------------------------------------

    def add_mistakes(self, mistakes: List[str]) -> None:
        """Merge new mistake tags into global memory (deduped, capped)."""
        for m in mistakes:
            if m not in self._mistakes:
                self._mistakes.append(m)
        # Keep only the most recent N unique tags
        if len(self._mistakes) > self._max_mistakes:
            self._mistakes = self._mistakes[-self._max_mistakes :]

    def get_mistakes(self) -> List[str]:
        """Return current mistake tags, sorted for determinism."""
        return sorted(self._mistakes)

    def clear_mistakes(self) -> None:
        self._mistakes.clear()

    # ------------------------------------------------------------------
    # Feedback management
    # ------------------------------------------------------------------

    def add_feedback(self, feedback: str) -> None:
        """Prepend feedback (most-recent first) and cap the list."""
        if feedback and feedback not in self._feedback:
            self._feedback.insert(0, feedback)
        if len(self._feedback) > self._max_feedback:
            self._feedback = self._feedback[: self._max_feedback]

    def get_feedback(self) -> List[str]:
        """Return stored feedback strings (most-recent first)."""
        return list(self._feedback)

    def clear_feedback(self) -> None:
        self._feedback.clear()

    # ------------------------------------------------------------------
    # Score history (for trend / improvement bonus meta-analysis)
    # ------------------------------------------------------------------

    def record_score(self, task_id: str, score: float) -> None:
        self._score_history[task_id].append(round(score, 4))

    def get_scores(self, task_id: Optional[str] = None) -> Dict[str, List[float]]:
        if task_id:
            return {task_id: list(self._score_history.get(task_id, []))}
        return {k: list(v) for k, v in self._score_history.items()}

    def average_score(self, task_id: Optional[str] = None) -> float:
        """Return the mean reward across all recorded episodes (or a single task)."""
        if task_id:
            scores = self._score_history.get(task_id, [])
        else:
            scores = [s for lst in self._score_history.values() for s in lst]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    # ------------------------------------------------------------------
    # Full reset
    # ------------------------------------------------------------------

    def reset_all(self) -> None:
        """Wipe everything — use between independent experiment runs."""
        self._mistakes.clear()
        self._feedback.clear()
        self._score_history.clear()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "mistakes": self._mistakes,
            "feedback": self._feedback,
            "score_history": dict(self._score_history),
        }

    def save(self, path: str) -> None:
        """Persist memory to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> "AgentMemory":
        """Restore memory from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Memory file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mem = cls(**kwargs)
        mem._mistakes = data.get("mistakes", [])
        mem._feedback = data.get("feedback", [])
        for tid, scores in data.get("score_history", {}).items():
            mem._score_history[tid] = scores
        return mem

    def __repr__(self) -> str:
        return (
            f"AgentMemory("
            f"mistakes={len(self._mistakes)}, "
            f"feedback={len(self._feedback)}, "
            f"tasks_tracked={len(self._score_history)})"
        )
