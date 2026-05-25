# Conductor Reproduction Workspace

This repository contains the training and evaluation code for reproducing
**Learning to Orchestrate Agents in Natural Language with the Conductor**.
The Conductor is a router/planner LLM trained with reinforcement learning to
compose natural-language workflows over a pool of worker LLMs.

For a fuller technical map of the codebase, training flow, evaluation flow, and
paper setup, see [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md).

## Layout

```text
conductor/          Training code, Hydra configs, datasets, trainer, rewards
conductor_ood_eval/ OpenAI-compatible model server and OOD evaluation scripts
paper.pdf           ICLR 2026 paper
docs/               Maintained project documentation
```

## Environment

This workspace is set up with `uv` and Python 3.11.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -r conductor/requirements.txt
```

Secrets are expected in `.env`, which is git-ignored. Typical keys:

```bash
WANDB_API_KEY=...
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://...
OPENAI_MODEL_NAME=...
```

## Train

Smoke training with Qwen3-4B and an OpenAI-compatible worker:

```bash
set -a
source .env
set +a

cd conductor
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python train.py \
  run_cfg@_global_=run_conductor_smoke_openai
```

Four-GPU MMLU run:

```bash
set -a
source .env
set +a

cd conductor
OPENAI_TIMEOUT=60 CUDA_VISIBLE_DEVICES=0,1,2,3 ../.venv/bin/accelerate launch \
  --num_processes 4 \
  --main_process_port 12987 \
  train.py run_cfg@_global_=run_conductor_mmlu_openai_full
```

## Evaluate

For checkpoint evaluation through the training stack, use `evaluate_only` or set
`eval_after_training: true` in the run config.

The OOD evaluation path is under `conductor_ood_eval/`:

```bash
cd conductor_ood_eval
../.venv/bin/python main.py --config configs/conductor_v1_7c22.json
```

Benchmark launch examples live in `conductor_ood_eval/examples/`.

## Paper Snapshot

The paper trains a 7B Qwen2.5-based Conductor with GRPO over workers including
Gemini 2.5 Pro, Claude Sonnet 4, GPT-5, DeepSeek-R1-Distill-Qwen-32B,
Gemma-3-27B-it, and Qwen3-32B variants. It trains on MATH, MMLU, RLPR, and
LiveCodeBench V1, then evaluates on MATH500, MMLU, RLPR, LiveCodeBench V6,
AIME25, GPQA-Diamond, and BigCodeBench.
