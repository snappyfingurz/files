# Self-Improving Customer Support Agent — OpenEnv Environment

An RL training environment in which an agent handles angry customer messages, receives deterministic grading, accumulates structured feedback in memory, and **improves across episodes** by avoiding past mistakes.

---

## ⚡ Quick Start

### 1. Interactive Gradio UI (Recommended)
You can test the environment manually, play as the agent, and receive live grading.
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Gradio App
python app.py
# The UI will be available at http://localhost:7860
```

### 2. Local Script Demo
Run the minimal example script which plays 3 episodes (easy, medium, hard).
```bash
python example_run.py
```

### 3. Docker Demo
```bash
docker build -t support-agent-env .
docker run --rm support-agent-env
```

---

## 🏗️ Project Structure

```text
self_improving_agent/
├── models.py       — Pydantic data models (Action, Observation, State, …)
├── tasks.py        — 3 task definitions (easy / medium / hard)
├── grader.py       — Deterministic grader (tone, correctness, resolution)
├── memory.py       — Persistent mistake & feedback memory
├── feedback.py     — Structured critique generator
├── env.py          — Main environment (OpenEnv API)
├── app.py          — Interactive Gradio UI wrapper
├── train.py        — RL training pipeline (Baseline vs. Adaptive)
├── agent.py        — Baseline and Adaptive agent implementations
├── example_run.py  — Runnable core code demo
├── openenv.yaml    — Environment specification
├── Dockerfile      — Docker containerization
├── .gitignore      — Git ignore definitions
└── requirements.txt
```

---

## 📈 Training Pipeline

The repository includes a comparative training pipeline via `train.py` which demonstrates memory-driven learning. It evaluates two agent architectures against the environment constraints over 60 episodes.

### Agents
1. **Baseline Agent**: Uses fixed response heuristics and entirely ignores environment memory.
2. **Adaptive Agent**: Utilizes the `AgentMemory` to retrieve stored `past_mistakes` and `past_feedback` to iteratively enhance its output. 

### Running the Pipeline
```bash
python train.py
```
This routine tests both paradigms sequentially and yields before/after metrics demonstrating the profound performance uplifts of the Adaptive Agent. The resulting JSON dumps all telemetry required for plotting.

---

## 🛠️ OpenEnv API Integration

```python
from env import CustomerSupportEnv
from models import Action

env = CustomerSupportEnv(seed=42)

# Start an episode (random task by default)
obs = env.reset()
# or explicitly set: obs = env.reset(task_id="hard_001")

print(obs.customer_message)   # What the customer said
print(obs.past_mistakes)      # Tags from previous episodes
print(obs.past_feedback)      # Structured critique from previous episodes

# Submit an action
action = Action(
    response="I'm truly sorry to hear this…",
    reflection="I noticed past mistake 'missing_apology' so I led with a genuine apology.",
)
result = env.step(action)

print(result.reward)          # float — this episode's reward
print(result.done)            # True (single-step mode) or False (multi-step)
print(result.info)            # Detailed grader breakdown
print(result.observation)     # Updated observation for next step

# Inspect full state
state = env.state()
print(state.score)            # Cumulative reward
print(state.history)          # All episode records
```

---

## 📝 Tasks

| ID          | Difficulty | Scenario                                                       |
|-------------|------------|----------------------------------------------------------------|
| `easy_001`  | Easy       | Delayed shipment — customer wants resolution today             |
| `medium_001`| Medium     | Three simultaneous issues: billing, app crash, ignored ticket  |
| `hard_001`  | Hard       | Refund demand outside 30-day policy window + chargeback threat |

---

## 🏆 Reward Formula

```text
reward = base_score
       + improvement_bonus        (+0.10 per past mistake avoided this turn)
       − repeated_mistake_penalty (−0.15 per past mistake repeated this turn)

base_score = (tone_score + correctness_score + resolution_score) / 3
```

All three sub-scores are in `[0, 1]`. Final reward is clamped to `[−0.5, 1.5]`.

### Score Definitions
- **Tone Score**: (+ empathy phrases: *"I'm sorry"*, *"I understand"*) vs (− hostile phrases: *"calm down"*, *"read the terms"*).
- **Correctness Score**: (+ task-specific required keywords present) vs (+ forbidden phrases absent).
- **Resolution Score**: (+ concrete next steps with timeframes) vs (+ ticket / case references).

---

## 🧠 Memory System

`AgentMemory` stores and curates:

| Store          | Content                              | Cap  |
|----------------|--------------------------------------|------|
| `past_mistakes`| Short mistake tags (e.g. `missing_apology`) | 20  |
| `past_feedback`| Full structured critique strings     | 10   |
| `score_history`| Per-task reward lists                | ∞    |

Memory **persists across episodes** (survives `reset()`). It is surfaced to the agent via `Observation.past_mistakes` and `Observation.past_feedback`.

### Serialise / restore memory:
```python
env.save_memory("memory.json")
env.load_memory("memory.json")
```

---

## 🚫 Mistake Tags

| Tag                        | Meaning                                      |
|----------------------------|----------------------------------------------|
| `missing_apology`          | No "sorry" / "apologize" in response         |
| `hostile_tone`             | Dismissive language detected                 |
| `missing_resolution`       | No concrete action / follow-up stated        |
| `missing_empathy`          | No validation of customer's frustration      |
| `forbidden_phrase:<phrase>`| A policy-violating phrase was used           |
| `missing_keyword:<kw>`     | A required content keyword was absent        |

---

## 📋 Requirements

- Python ≥ 3.10
- pydantic ≥ 2.0, < 3.0
- gradio ≥ 4.0

---

## 📄 License
MIT
