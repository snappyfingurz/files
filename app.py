"""
app.py — Fully automatic LLM-driven RL simulation (no manual customer reply).

Run:  python app.py
"""

from __future__ import annotations

import random
import threading
import time
import traceback

import gradio as gr

from agent import DEFAULT_REFLECTION, ensure_llm_ready, llm_agent
from env import CustomerSupportEnv
from models import Action
from tasks import all_task_ids

_UI = threading.Lock()
_ENV: CustomerSupportEnv = CustomerSupportEnv(
    seed=int(time.time() * 1000) % (2**30) ^ random.randint(1, 999_999)
)

TASK_CHOICES: list[str] = ["Random"] + list(all_task_ids())

DESCRIPTION = """\
**Autonomous self-improving support agent** — the LLM writes every response. **No typing.**

Click **Run simulation** to run: `obs = env.reset()` → `while not done: action = llm_agent(obs); result = env.step(action); obs = result.observation`.

Set `HF_TOKEN` for the Hugging Face API (default), or `LLM_BACKEND=local` for a local `transformers` pipeline. Temperature 0.7–0.8 in `agent.py`. Optional: stop early after **3 consecutive** episodes with reward ≥ 0.9.
"""


def _run_autonomous_loop(
    task: str,
    max_episodes: int,
    early_mastery: bool,
) -> tuple[str, str, str, str, str, str, str, str]:
    try:
        return _run_autonomous_loop_impl(task, max_episodes, early_mastery)
    except Exception as e:
        tb = traceback.format_exc()
        err = f"{type(e).__name__}: {e}"
        sim = f"**Run failed**\n\n{err}\n\n```\n{tb}\n```"
        st = f"Error — {err}"
        return (
            f"**Could not start or finish the run.**\n\n`{err}`\n\n"
            "If the message mentions a missing token, set `HF_TOKEN` (PowerShell: "
            "`$env:HF_TOKEN = 'hf_...'`) or use `LLM_BACKEND=local` with a GPU and transformers.",
            "—",
            "—",
            "—",
            "—",
            "—",
            st,
            sim,
        )


def _run_autonomous_loop_impl(
    task: str,
    max_episodes: int,
    early_mastery: bool,
) -> tuple[str, str, str, str, str, str, str, str]:
    ensure_llm_ready()
    log_lines: list[str] = []
    last_preview = ""
    last_rew = "—"
    last_mist = "—"
    last_cust = "—"
    high_streak = 0
    obs: object | None = None
    episodes_done = 0
    nmax = max(0, int(max_episodes))

    if nmax < 1:
        return (
            "Set **Max episodes** to at least 1.",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
        )

    with _UI:
        for ep in range(1, nmax + 1):
            tid = None if task == "Random" else task
            obs = _ENV.reset(task_id=tid)
            done = False
            while not done:
                action: Action = llm_agent(obs)
                result = _ENV.step(action)
                obs = result.observation
                reward = float(result.reward)
                done = result.done
                prev = (action.response or "")[:120].replace("\n", " ")
                mtags = result.info.get("mistakes_found", []) or []
                mstr = ", ".join(mtags) if mtags else "none"
                cum = _ENV.state().score
                line = (
                    f"Ep {ep} | step reward: {reward:.3f} | running total: {cum:+.3f} | mistakes: {mstr} | "
                    f"response: {prev}{'...' if len((action.response or '')) > 120 else ''}"
                )
                log_lines.append(line)
                print(f"Response: {prev}...")
                print(f"Reward: {reward:.3f}")
                last_preview = (action.response or "")[:2000]
                last_rew = f"{reward:+.4f}"
                last_mist = mstr
                last_cust = (obs.customer_message or "")[:2000]
            episodes_done = ep
            r_ep = float(result.reward)
            if r_ep >= 0.9:
                high_streak += 1
            else:
                high_streak = 0
            if early_mastery and high_streak >= 3:
                log_lines.append(">>> Early stop: reward >= 0.9 for 3 consecutive episodes.")
                break

    full_log = "\n".join(log_lines) if log_lines else "(no steps)"
    st = _ENV.state()
    o = obs
    if o is None:
        return ("—",) * 8
    pm = "\n".join(f"- {m}" for m in o.past_mistakes) or "—"
    pfb = "\n---\n".join(o.past_feedback) if o.past_feedback else "—"
    status = (
        f"Episodes this run: {episodes_done}  |  cumulative: {st.score:+.4f}  |  "
        f"total steps: {st.step_count}"
    )
    return (
        f"**Customer (last observation)**\n\n```\n{last_cust}\n```",
        pm,
        pfb,
        f"**Last model reply**\n\n```\n{last_preview}\n```\n\n**Reflection:** `{DEFAULT_REFLECTION}`",
        f"Last step reward: {last_rew}\nCumulative total: {st.score:+.4f}",
        last_mist,
        status,
        full_log,
    )


with gr.Blocks(title="Autonomous support agent (LLM RL)") as demo:
    gr.Markdown("# Self-Improving Customer Support Agent")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        tsel = gr.Dropdown(choices=TASK_CHOICES, value="Random", label="Task")
        n_ep = gr.Slider(1, 30, value=5, step=1, label="Max episodes per run")
        early = gr.Checkbox(value=True, label="Early stop if reward ≥ 0.9 for 3 episodes in a row")
    run_btn = gr.Button("Run simulation", variant="primary", scale=1)

    cmsg = gr.Markdown()
    m_pm = gr.Textbox(label="Memory: past mistake tags", lines=3, interactive=False)
    m_pfb = gr.Textbox(label="Memory: past feedback (injected into next llm_agent call)", lines=4, interactive=False)
    m_reply = gr.Markdown()
    rbox = gr.Textbox(label="Reward (last step + cumulative)", lines=2, interactive=False)
    mtags = gr.Textbox(label="Mistake tags (last step)", lines=1, interactive=False)
    stline = gr.Textbox(label="Status", lines=1, interactive=False)
    sim_log = gr.Textbox(label="Simulation log (episodes, reward, response preview, mistakes)", lines=22, interactive=False)

    out = [cmsg, m_pm, m_pfb, m_reply, rbox, mtags, stline, sim_log]
    run_btn.click(_run_autonomous_loop, [tsel, n_ep, early], out)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(theme=gr.themes.Soft(), css="")
