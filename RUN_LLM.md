# LLM self-improving support agent — runbook

## What this is

- **OpenEnv** in `env.py` (unchanged API): `reset` → `step(Action)` → `Observation` + reward.
- **Grader** in `grader.py` (unchanged): scores the **LLM** reply.
- **Memory** in `memory.py` (unchanged): past mistake tags + feedback strings, fed into `Observation`.
- **Policy** in `agent.py`: `llm_agent(obs)` → `Action` (no hand-written customer replies, no template cache).

## Environment

| Variable | Meaning |
|----------|--------|
| `HF_TOKEN` | **Required** for the default `hf_api` path: create a token at [HF settings](https://huggingface.co/settings/tokens). Without it, you will see the same “trouble generating a full reply” fail-safe on every step. The client uses `provider=hf-inference` (set `HF_INFERENCE_PROVIDER` to override). |
| `LLM_BACKEND` | `hf_api` (default) or `local` (local `transformers` pipeline; needs GPU/VRAM for 7B). |
| `HF_LLM_MODEL` | API model id (default `Qwen/Qwen2.5-7B-Instruct`). |
| `HF_LLM_LOCAL_MODEL` | Local pipeline model (default `mistralai/Mistral-7B-Instruct-v0.2`). |
| `LLM_TEMPERATURE` | Default `0.7` (controlled variability). |
| `LOAD_IN_4BIT` | If set, try 4-bit load for local pipeline (requires `bitsandbytes`, etc.). |

## Local (Windows / macOS / Linux)

```bash
cd /path/to/files
python -m venv .venv
. .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
set HF_TOKEN=hf_xxx         # or export HF_TOKEN=... on bash
python train.py             # curriculum loop + logging
# or
python example_run.py       # three LLM episodes
# or
python app.py               # Gradio at http://127.0.0.1:7860 (or next free port)
```

**Local 7B model (optional, heavy):**

```bash
pip install "transformers[torch]" accelerate
set LLM_BACKEND=local
set HF_LLM_LOCAL_MODEL=mistralai/Mistral-7B-Instruct-v0.2
python example_run.py
```

## Google Colab

1. Upload the project or clone the repo.
2. In a cell:

```python
!pip install -q pydantic gradio "huggingface_hub<1" "transformers[torch]" accelerate
%env HF_TOKEN=hf_xxx
%env LLM_BACKEND=hf_api
```

3. `!python train.py` or `!python example_run.py`
4. For **local GPU** in Colab (e.g. T4): set `LLM_BACKEND=local` and install `bitsandbytes` if using 4-bit.

## API vs local

- **hf_api**: works on CPU; needs network; `HF_TOKEN` recommended.
- **local**: no API cost; needs a large GPU for 7B; first run downloads weights.

## Training log

`train.py` prints each step:

```text
Reward: 0.xxx
Response: ...
```

This matches the project spec for verifying non-degenerate outputs and reward movement.
