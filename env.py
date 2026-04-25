"""
env.py — Self-Improving Customer Support Agent Environment.

Implements the OpenEnv API:
  reset(task_id?)  → Observation
  step(action)     → StepResult
  state()          → State

Episode flow:
  1. reset()  — select a task, clear per-episode state, return Observation
                 (with memory from past episodes)
  2. step()   — agent submits Action; grader scores it; feedback generated;
                 memory updated; StepResult returned (done=True after 1 step
                 per episode by default; set multi_step=True for longer episodes)
  3. state()  — inspect full internal state at any time

Memory persists across episodes so the agent can improve.
"""

from __future__ import annotations

import random
from typing import List, Optional

from feedback import generate_feedback
from grader import grade
from memory import AgentMemory
from models import (
    Action,
    EpisodeRecord,
    Observation,
    State,
    StepResult,
)
from tasks import TASKS, Task, all_task_ids, get_task


class CustomerSupportEnv:
    """
    Self-Improving Customer Support Agent Environment.

    Parameters
    ----------
    task_sequence : list[str] | None
        If provided, tasks are drawn from this list in order (cycling).
        If None, tasks are sampled uniformly at random.
    seed : int | None
        Random seed for reproducible task sampling.
    multi_step : bool
        If True, an episode spans multiple step() calls (up to max_steps).
        If False (default), each episode is exactly 1 step (done=True immediately).
    max_steps : int
        Maximum steps per episode when multi_step=True.
    memory : AgentMemory | None
        Provide an existing memory object to resume from a previous run.
    """

    def __init__(
        self,
        task_sequence: Optional[List[str]] = None,
        seed: Optional[int] = None,
        multi_step: bool = False,
        max_steps: int = 3,
        memory: Optional[AgentMemory] = None,
    ):
        self._task_sequence = task_sequence
        self._sequence_index = 0
        self._rng = random.Random(seed)
        self._multi_step = multi_step
        self._max_steps = max_steps

        # Persistent memory (survives reset)
        self.memory: AgentMemory = memory or AgentMemory()

        # Internal state (reset each episode)
        self._state = State()
        self._current_task: Optional[Task] = None
        self._episode_steps: int = 0
        self._started = False

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Start a new episode.

        Args:
            task_id: Specific task to load. If None, the next task in the
                     configured sequence (or a random task) is selected.

        Returns:
            Observation with the customer message and accumulated memory.
        """
        # Select task
        if task_id is not None:
            self._current_task = get_task(task_id)
        elif self._task_sequence:
            tid = self._task_sequence[self._sequence_index % len(self._task_sequence)]
            self._sequence_index += 1
            self._current_task = get_task(tid)
        else:
            tid = self._rng.choice(all_task_ids())
            self._current_task = get_task(tid)

        # Reset per-episode counters (keep memory and cumulative score)
        self._episode_steps = 0
        self._started = True

        # Update state
        self._state.current_task_id = self._current_task.task_id
        self._state.done = False

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Process one agent action.

        Args:
            action: Action(response=..., reflection=...)

        Returns:
            StepResult(observation, reward, done, info)

        Raises:
            RuntimeError: If called before reset() or after the episode ended.
        """
        if not self._started:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")
        if self._current_task is None:
            raise RuntimeError("No task loaded. Call reset() first.")

        self._episode_steps += 1
        self._state.step_count += 1

        task = self._current_task
        past_mistakes = self.memory.get_mistakes()

        # Grade the response
        result = grade(
            response=action.response,
            task=task,
            past_mistakes=past_mistakes,
        )

        # ── Progressive Punishment System ────────────────────────────
        # Get historical mistake counts from memory
        m_counts = self.memory.get_mistake_counts()
        
        # Avoided mistakes (improvement / avoidance bonus)
        current_set = set(result.mistakes_found)
        past_set = set(past_mistakes)
        avoided = past_set - current_set
        
        # Repeated mistakes (progressive penalty)
        repeated = past_set & current_set
        
        # ── Quality Gate & Reward Shaping ────────────────────────────
        base_score = result.base_score
        
        # 1. Quality Gate: Penalize poor tone or zero resolution effort
        if result.tone_score < 0.3 or result.resolution_score < 0.3:
            base_score *= 0.5

        # 2. Improvement / Avoidance Bonus
        improvement = round(len(avoided) * 0.10, 4)
        improvement_bonus = min(improvement, 0.15) # Cap improvement bonus
        
        # 3. Restrict bonus if base quality is unacceptable
        if base_score < 0.4:
            improvement_bonus *= 0.3
        
        # 4. Progressive repeated mistake penalty
        prog_penalty = 0.0
        for m in repeated:
            count = m_counts.get(m, 0)
            penalty_per = round(min(0.08 * count, 0.5), 4)
            prog_penalty -= penalty_per
        
        # Final Reward Calculation
        # final_reward = base_score + improvement_bonus + avoidance_bonus + progressive_penalty
        # Note: avoidance_bonus is treated as the improvement cap per section 3/4.
        final_reward = round(base_score + improvement_bonus + prog_penalty, 4)
        
        # ── Gibberish Override ──
        if "gibberish_detected" in result.mistakes_found:
            final_reward = -0.20
            improvement_bonus = 0.0
        
        # Clamp to reasonable stable range [-1, 1.5]
        final_reward = max(-1.0, min(1.5, final_reward))

        # Update result with final calculated values for UI / Logging
        result.base_score = base_score
        result.improvement_bonus = improvement_bonus
        result.repeated_mistake_penalty = abs(prog_penalty)
        result.final_reward = final_reward

        # Generate structured feedback
        fb = generate_feedback(
            response=action.response,
            result=result,
            task=task,
        )

        # Update memory
        self.memory.add_mistakes(result.mistakes_found)
        self.memory.add_feedback(fb)
        self.memory.record_score(task.task_id, final_reward)

        # Update cumulative score
        self._state.score = round(self._state.score + final_reward, 4)

        # Record episode in history
        record = EpisodeRecord(
            task_id=task.task_id,
            difficulty=task.difficulty,
            customer_message=task.customer_message,
            agent_response=action.response,
            agent_reflection=action.reflection,
            score=result.final_reward,
            feedback=fb,
            mistakes=result.mistakes_found,
        )
        self._state.history.append(record)

        # Determine if episode is done
        done = (not self._multi_step) or (self._episode_steps >= self._max_steps)
        self._state.done = done

        # Build next observation (or terminal one if done)
        if done:
            obs = self._build_observation()
        else:
            # In multi-step mode, same task continues (agent can retry)
            obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=result.final_reward,
            done=done,
            info={
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "tone_score": result.tone_score,
                "correctness_score": result.correctness_score,
                "resolution_score": result.resolution_score,
                "actionability_score": result.actionability_score,
                "policy_compliance_score": result.policy_compliance_score,
                "conciseness_score": result.conciseness_score,
                "clarity_score": result.clarity_score,
                "base_score": result.base_score,
                "improvement_bonus": result.improvement_bonus,
                "repeated_mistake_penalty": result.repeated_mistake_penalty,
                "mistakes_found": result.mistakes_found,
                "feedback": fb,
                "episode_step": self._episode_steps,
            },
        )

    def state(self) -> State:
        """Return a copy of the full internal state (non-destructive)."""
        return self._state.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return a human-readable summary of the current state."""
        s = self._state
        lines = [
            "=" * 60,
            "Self-Improving Customer Support Agent — Environment State",
            "=" * 60,
            f"  Step count   : {s.step_count}",
            f"  Total reward : {s.score:.4f}",
            f"  Current task : {s.current_task_id or 'none'}",
            f"  Done         : {s.done}",
            f"  Memory       : {self.memory}",
            f"  Episodes     : {len(s.history)}",
        ]
        if s.history:
            last = s.history[-1]
            lines += [
                "",
                "  Last episode:",
                f"    Task       : {last.task_id} ({last.difficulty})",
                f"    Reward     : {last.score:.4f}",
                f"    Mistakes   : {last.mistakes or 'none'}",
                f"    Feedback   :",
            ]
            for line in last.feedback.split("\n"):
                lines.append(f"      {line}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_memory(self, path: str) -> None:
        self.memory.save(path)

    def load_memory(self, path: str) -> None:
        self.memory = AgentMemory.load(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        assert self._current_task is not None
        return Observation(
            customer_message=self._current_task.customer_message,
            past_feedback=self.memory.get_feedback(),
            past_mistakes=self.memory.get_mistakes(),
        )
