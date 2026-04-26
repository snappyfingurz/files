"""
example_run.py — three episodes (easy / medium / hard) using the LLM agent only.

No hand-authored customer replies. Requires HF_TOKEN for default hf_api backend
(or LLM_BACKEND=local + GPU + transformers).

Run:  python example_run.py
"""

from __future__ import annotations

from agent import llm_agent
from env import CustomerSupportEnv
from models import Action


def run_episode(env: CustomerSupportEnv, task_id: str, label: str) -> None:
    print("\n" + "━" * 60)
    print(f"{label}  —  {task_id}")
    print("━" * 60)
    obs = env.reset(task_id=task_id)
    print(f"Customer (preview): {obs.customer_message[:200]}…")
    print(f"Past mistake tags: {obs.past_mistakes!r}")
    if obs.past_feedback:
        print(f"Most recent feedback: {obs.past_feedback[0][:160]}…")

    done = False
    while not done:
        action: Action = llm_agent(obs)
        result = env.step(action)
        obs = result.observation
        reward = float(result.reward)
        done = result.done
        rtxt = (action.response or "")[:120].replace("\n", " ")
        print(f"Reward: {reward:.3f}")
        print(f"Response: {rtxt}{'...' if len((action.response or '')) > 120 else ''}")
    print(
        f"Done={result.done}  tone={result.info.get('tone_score', 0):.2f}  "
        f"resolution={result.info.get('resolution_score', 0):.2f}  "
        f"mistakes={result.info.get('mistakes_found', [])!r}"
    )
    if result.info.get("feedback"):
        print("Feedback (excerpt):\n", (result.info["feedback"] or "")[:400], "…", sep="")


def main() -> None:
    env = CustomerSupportEnv(seed=7)
    run_episode(env, "easy_001", "EPISODE 1")
    run_episode(env, "medium_001", "EPISODE 2")
    run_episode(env, "hard_001", "EPISODE 3")

    s = env.state()
    print("\n" + "═" * 60)
    print(env.render())
    print(f"\nFinal cumulative reward: {s.score:+.4f}  |  total steps: {s.step_count}")


if __name__ == "__main__":
    main()
