"""
app.py — Gradio UI for the Self-Improving Customer Support Agent Environment.

Wraps the existing OpenEnv environment (env.py) with an interactive Gradio
interface.  No modifications to the underlying RL logic, reward system, or
memory.

Usage:
    python app.py

Compatible with Hugging Face Spaces (set `sdk: gradio` in the Space README).
"""

from __future__ import annotations

import gradio as gr

from env import CustomerSupportEnv
from models import Action
from tasks import all_task_ids

# ── Constants ────────────────────────────────────────────────────────────────

TASK_CHOICES = ["Random"] + all_task_ids()

DESCRIPTION = """\
An **OpenEnv** reinforcement-learning environment where **you** play the
support agent.  Respond to angry customers, get graded on tone / correctness /
resolution, and watch the **memory-driven self-improvement** loop in action
across episodes.

1. **Select a task** (or leave "Random") and click **Reset Episode**.
2. Read the customer message, then write your response.
3. Click **Submit** — the deterministic grader scores you instantly.
4. Click **Reset Episode** again to start the next episode.  
   Your past mistakes and feedback carry over in memory!
"""

CUSTOM_CSS = """
.status-bar textarea { font-weight: 600 !important; }
.reward-box textarea {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    text-align: center !important;
}
.score-box textarea { font-family: monospace !important; }
"""

# ── Helper ───────────────────────────────────────────────────────────────────


def _bullets(items: list[str], empty: str = "None") -> str:
    if not items:
        return empty
    return "\n".join(f"• {m}" for m in items)


# ── Core callbacks ───────────────────────────────────────────────────────────


def reset_episode(env_state, task_choice):
    """Reset the environment and return a fresh observation."""
    if env_state is None:
        env_state = CustomerSupportEnv(seed=42)

    task_id = None if task_choice == "Random" else task_choice
    obs = env_state.reset(task_id=task_id)
    st = env_state.state()

    past_mistakes = _bullets(
        obs.past_mistakes, "None yet — this is the first episode."
    )
    past_feedback = (
        "\n\n---\n\n".join(obs.past_feedback)
        if obs.past_feedback
        else "None yet — complete an episode to generate feedback."
    )

    status = (
        f"🟢 Ready  ·  Task: {st.current_task_id}  ·  "
        f"Steps: {st.step_count}  ·  Total Reward: {st.score:.4f}"
    )

    return (
        env_state,              # gr.State
        obs.customer_message,   # customer_msg
        past_mistakes,          # past_mistakes_box
        past_feedback,          # past_feedback_box
        "",                     # response_input  (clear)
        "",                     # reflection_input (clear)
        "—",                    # reward_box
        "—",                    # scores_box
        "—",                    # feedback_box
        "—",                    # mistakes_box
        status,                 # status_bar
    )


def submit_response(env_state, response, reflection):
    """Grade the agent's response via env.step() and return results."""

    # ── guard: env not initialised ───────────────────────────────────
    if env_state is None:
        gr.Warning("Please click 'Reset Episode' first to load a task.")
        return _placeholder_outputs(None, "⚠️ Click Reset to start.")

    st = env_state.state()

    # ── guard: episode already finished ──────────────────────────────
    if st.done:
        gr.Warning(
            "This episode is done. Click 'Reset Episode' to start the next one."
        )
        pm = _bullets(env_state.memory.get_mistakes())
        pf = (
            "\n\n---\n\n".join(env_state.memory.get_feedback())
            if env_state.memory.get_feedback()
            else "—"
        )
        return (
            env_state,
            "✅ Episode finished — click Reset.",
            pm,
            pf,
            response,
            reflection,
            "—",
            "—",
            "—",
            "—",
            f"🔴 Done  ·  Steps: {st.step_count}  ·  Total Reward: {st.score:.4f}",
        )

    # ── guard: empty response ────────────────────────────────────────
    if not response or not response.strip():
        gr.Warning("Please write a response before submitting.")
        msg = (
            env_state._current_task.customer_message
            if env_state._current_task
            else "—"
        )
        pm = _bullets(env_state.memory.get_mistakes())
        pf = (
            "\n\n---\n\n".join(env_state.memory.get_feedback())
            if env_state.memory.get_feedback()
            else "—"
        )
        return (
            env_state,
            msg,
            pm,
            pf,
            response,
            reflection,
            "—",
            "—",
            "—",
            "—",
            f"🟡 Waiting for response  ·  Task: {st.current_task_id}",
        )

    # ── run the step ─────────────────────────────────────────────────
    refl = (
        reflection.strip()
        if (reflection and reflection.strip())
        else "No reflection provided."
    )
    action = Action(response=response.strip(), reflection=refl)

    try:
        result = env_state.step(action)
    except RuntimeError as exc:
        gr.Warning(str(exc))
        return _placeholder_outputs(env_state, f"⚠️ {exc}")

    st = env_state.state()

    # ── format outputs ───────────────────────────────────────────────
    reward_str = f"{result.reward:+.4f}"

    scores_str = (
        f"Tone:          {result.info.get('tone_score', 0):.4f}\n"
        f"Correctness:   {result.info.get('correctness_score', 0):.4f}\n"
        f"Resolution:    {result.info.get('resolution_score', 0):.4f}\n"
        f"Actionability: {result.info.get('actionability_score', 0):.4f}\n"
        f"Policy:        {result.info.get('policy_compliance_score', 0):.4f}\n"
        f"───────────────────────\n"
        f"Base Score:    {result.info.get('base_score', 0):.4f}\n"
        f"+ Improvement: {result.info.get('improvement_bonus', 0):.4f}\n"
        f"- Penalty:     {result.info.get('repeated_mistake_penalty', 0):.4f}"
    )

    feedback_str = result.info.get("feedback", "—")

    mistakes = result.info.get("mistakes_found", [])
    mistakes_str = (
        _bullets(mistakes)
        if mistakes
        else "✅ No mistakes found — excellent response!"
    )

    pm = _bullets(
        result.observation.past_mistakes,
        "No accumulated mistakes.",
    )
    pf = (
        "\n\n---\n\n".join(result.observation.past_feedback)
        if result.observation.past_feedback
        else "—"
    )

    done_label = (
        "DONE — click Reset for next episode"
        if result.done
        else "Awaiting next step"
    )
    status = (
        f"{'🔴' if result.done else '🟢'} {done_label}  ·  "
        f"Steps: {st.step_count}  ·  Total Reward: {st.score:.4f}"
    )

    return (
        env_state,
        result.observation.customer_message,
        pm,
        pf,
        "",  # clear response input
        "",  # clear reflection input
        reward_str,
        scores_str,
        feedback_str,
        mistakes_str,
        status,
    )


def _placeholder_outputs(env_state, message: str):
    """Return a full tuple of placeholder values for error / guard states."""
    return (
        env_state,
        message,
        "—",
        "—",
        "",
        "",
        "—",
        "—",
        "—",
        "—",
        "🔴 " + message,
    )


# ── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks() as demo:

    env_state = gr.State(value=None)

    # ── Header ────────────────────────────────────────────────────────
    gr.Markdown("# 🎯 Self-Improving Customer Support Agent")
    gr.Markdown(DESCRIPTION)

    # ── Controls row ──────────────────────────────────────────────────
    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=TASK_CHOICES,
            value="Random",
            label="Task Selection",
            scale=2,
        )
        reset_btn = gr.Button("🔄 Reset Episode", variant="primary", scale=1)

    status_bar = gr.Textbox(
        label="Status",
        value="Click 'Reset Episode' to begin.",
        interactive=False,
        elem_classes=["status-bar"],
    )

    # ── Main two-column layout ────────────────────────────────────────
    with gr.Row(equal_height=True):

        # Left column — environment info
        with gr.Column(scale=1):
            customer_msg = gr.Textbox(
                label="📨 Customer Message",
                lines=8,
                interactive=False,
                placeholder="Click Reset to load a task…",
            )
            with gr.Accordion("🧠 Memory — Past Mistakes", open=False):
                past_mistakes_box = gr.Textbox(
                    lines=5,
                    interactive=False,
                    show_label=False,
                    value="—",
                )
            with gr.Accordion("🧠 Memory — Past Feedback", open=False):
                past_feedback_box = gr.Textbox(
                    lines=8,
                    interactive=False,
                    show_label=False,
                    value="—",
                )

        # Right column — agent input
        with gr.Column(scale=1):
            response_input = gr.Textbox(
                label="✍️ Your Response",
                lines=8,
                placeholder="Type your customer support response here…",
            )
            reflection_input = gr.Textbox(
                label="💭 Your Reflection (optional)",
                lines=2,
                placeholder=(
                    "Why did you choose this approach? "
                    "What did you learn from past feedback?"
                ),
            )
            submit_btn = gr.Button(
                "📤 Submit Response",
                variant="primary",
            )

    # ── Results section ───────────────────────────────────────────────
    gr.Markdown("### 📊 Grading Results")

    with gr.Row():
        reward_box = gr.Textbox(
            label="🏆 Reward",
            value="—",
            interactive=False,
            scale=1,
            elem_classes=["reward-box"],
        )
        scores_box = gr.Textbox(
            label="Score Breakdown",
            value="—",
            lines=7,
            interactive=False,
            scale=2,
            elem_classes=["score-box"],
        )

    with gr.Row():
        feedback_box = gr.Textbox(
            label="📝 Grader Feedback",
            value="—",
            lines=5,
            interactive=False,
            scale=2,
        )
        mistakes_box = gr.Textbox(
            label="⚠️ Mistakes Found This Episode",
            value="—",
            lines=5,
            interactive=False,
            scale=1,
        )

    # ── Wire events ───────────────────────────────────────────────────
    all_outputs = [
        env_state,
        customer_msg,
        past_mistakes_box,
        past_feedback_box,
        response_input,
        reflection_input,
        reward_box,
        scores_box,
        feedback_box,
        mistakes_box,
        status_bar,
    ]

    reset_btn.click(
        fn=reset_episode,
        inputs=[env_state, task_dropdown],
        outputs=all_outputs,
    )

    submit_btn.click(
        fn=submit_response,
        inputs=[env_state, response_input, reflection_input],
        outputs=all_outputs,
    )


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
