"""
train.py — RL loop with a fully LLM-driven policy (no rule-based or hand-templated actions).

OpenEnv flow per episode:
  obs = env.reset()
  while not done:
      action = llm_agent(obs)
      result = env.step(action)
      obs = result.observation
      reward, done = result.reward, result.done

With default single-step episodes (multi_step=False), the inner `while` runs once per `reset`.
"""

from __future__ import annotations

import time

from agent import llm_agent
from env import CustomerSupportEnv

# One environment so memory and cumulative score carry across the curriculum
_ENV = CustomerSupportEnv(seed=42)

# (phase_name, episodes, task_id) — fixed task per phase; memory / score cross phases
CURRICULUM: list[tuple[str, int, str | None]] = [
    ("EASY", 8, "easy_001"),
    ("MEDIUM", 10, "medium_001"),
    ("HARD", 12, "hard_001"),
]


def run_rl_training() -> None:
    print("\n" + "=" * 50)
    print("LLM-driven self-improving support agent (OpenEnv + llm_agent)")
    print("=" * 50)

    total = 0
    env = _ENV

    for phase, n_eps, task_id in CURRICULUM:
        print(f"\n--- Phase: {phase}  ({n_eps} episodes)  task={task_id!r} ---\n")
        for ep in range(n_eps):
            total += 1
            obs = env.reset(task_id=task_id)
            done = False
            while not done:
                action = llm_agent(obs)
                result = env.step(action)
                obs = result.observation
                reward = float(result.reward)
                done = result.done
                rtxt = (action.response or "")[:200].replace("\n", " ")
                print(f"Reward: {reward:.3f}")
                print(f"Response: {rtxt}{'...' if len((action.response or '')) > 200 else ''}")
            time.sleep(0.05)
        st = env.state()
        print(f"  (phase done — cumulative score: {st.score:+.4f}  steps: {st.step_count})")

    st = env.state()
    print("\n" + "=" * 50)
    print(
        f"Done. Episodes: {total}  |  Final cumulative: {st.score:+.4f}  |  total steps: {st.step_count}"
    )
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_rl_training()
