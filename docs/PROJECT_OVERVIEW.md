# Project Overview

This repository contains code for reproducing and serving **The Conductor**, an
LLM trained to orchestrate other LLM agents. The paper frames the Conductor as a
router/planner model: given a user task and a set of worker models, it emits a
natural-language workflow with subtasks, assigned worker ids, and an access list
that controls which previous worker responses each step can see.

## Repository Layout

```text
.
├── paper.pdf                         # ICLR 2026 paper
├── Conductor_revision_supplement.pdf # supplementary material
├── conductor/                        # training code
│   ├── cfgs/                         # Hydra configs for model/data/trainer/runs
│   ├── custom_data/                  # dataset wrappers and routing prompts
│   ├── trainers/                     # GRPO trainer and Conductor reward
│   ├── evaluation/                   # task/evaluation helpers
│   ├── scripts/                      # install and launch helpers
│   ├── train.py                      # Hydra training/evaluation entry point
│   └── llm_clients.py                # worker model API clients
└── conductor_ood_eval/               # OpenAI-compatible serving/eval pipeline
    ├── configs/                      # server engine configs
    ├── examples/                     # benchmark launch scripts
    ├── models/                       # single, router, and conductor engines
    ├── server/                       # OpenAI-compatible HTTP API
    ├── main.py                       # server entry point
    └── query_oai_responses.py        # OpenAI-style batch query helper
```

## Training Flow

Training is launched from `conductor/train.py` with Hydra. A run config under
`conductor/cfgs/run_cfg/` composes three major config groups:

- `model_cfg`: base model and tokenizer, for example `qwen7bi.yaml` or the local
  smoke config `qwen3_4bi.yaml`.
- `data_cfg`: dataset builder. `guf_dataset.yaml` calls
  `custom_data.dataset.make_dataset`, which wraps task modules such as
  `guf.tasks.mmlu.MMLUTask`.
- `trainer_cfg`: trainer and reward. `conductor_grpo.yaml` instantiates
  `trainers.conductor_engine.gufGRPOTrainer` and `ConductorReward`.

At each GRPO step, the local Conductor model generates workflow completions. The
reward code parses three lists from each completion:

- `model_id`: which worker model handles each workflow step.
- `subtasks`: natural-language instructions for those workers.
- `access_list`: which previous worker outputs are visible to later steps.

Malformed workflow outputs receive zero reward. Valid workflows are executed by
calling worker models through `trainers/conductor_utils.py` and
`llm_clients.py`; correctness is scored by the task implementation. W&B logging
is enabled through `report_to: wandb` and uses `WANDB_API_KEY` from `.env`.

Example local smoke run:

```bash
set -a
source .env
set +a

cd conductor
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python train.py \
  run_cfg@_global_=run_conductor_smoke_openai
```

Example 4-GPU MMLU run using the local Qwen3-4B config:

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

The local `run_conductor_mmlu_openai_full.yaml` config is a practical
reproduction/smoke variant, not the full paper recipe: it uses `Qwen/Qwen3-4B`
with PEFT/4-bit loading, trains on MMLU, uses a single OpenAI-compatible worker
from `OPENAI_MODEL_NAME`, and enables `eval_after_training`.

## Evaluation Flow

There are two evaluation paths:

1. **In-training evaluation** through `conductor/train.py`.
   Set `evaluate_only` to a checkpoint path, or set `eval_after_training: true`
   in a run config. The trainer instantiates the same dataset/reward stack and
   calls `trainer.evaluate()`.

2. **OOD/server-based evaluation** through `conductor_ood_eval/`.
   This subproject starts an OpenAI-compatible server that wraps a local
   Conductor, router, or single-model engine. Benchmark scripts in
   `conductor_ood_eval/examples/` then evaluate the served endpoint with
   frameworks such as lighteval, FastChat/MT-Bench, BigCodeBench, and
   task-specific scripts.

Example server launch:

```bash
cd conductor_ood_eval
../.venv/bin/python main.py --config configs/conductor_v1_7c22.json
```

Example smoke server config for an external OpenAI-compatible endpoint:

```bash
cd conductor_ood_eval
../.venv/bin/python smoke_eval.py --config configs/smoke_openai_env.json
```

## What The Paper Trains

The main paper trains a **7B Conductor** initialized from a Qwen2.5 checkpoint.
The Conductor is trained with GRPO to produce workflows of up to five steps over
seven workers:

- Gemini 2.5 Pro
- Claude Sonnet 4
- GPT-5
- DeepSeek-R1-Distill-Qwen-32B
- Gemma-3-27B-it
- Qwen3-32B direct mode
- Qwen3-32B reasoning mode

The reported main training setup is:

- 960 training problems from MATH, MMLU, RLPR, and LiveCodeBench V1.
- 200 GRPO iterations.
- 4 questions per iteration and 64 rollouts per question, for batch size 256.
- AdamW with learning rate `1e-6`, cosine schedule, warmup ratio `0.03`.
- No KL regularization and no reference model synchronization.
- Conductor max completion length 1024.
- Worker max completion length 4096, worker temperature 0.2.
- Closed-source reasoning budgets set to low-cost minima during training.
- Main Conductor trained on 2 NVIDIA H100 80GB GPUs.

The paper also reports two finetuning extensions:

- **Adaptive worker selection**: finetune with randomized worker subsets so the
  Conductor can adapt to arbitrary open/closed worker pools.
- **Recursive Conductor**: finetune the trained Conductor for 20 iterations on a
  350-sample subset, allowing the Conductor to call itself as a worker for
  test-time recursive scaling.

## Paper Benchmarks

The paper evaluates both in-domain and out-of-domain generalization.

In-domain tasks:

- MATH500
- MMLU
- RLPR
- LiveCodeBench V6

Out-of-domain tasks:

- AIME25
- GPQA-Diamond
- BigCodeBench hard/complete subset

Paper-reported headline Table 1 scores for Conductor:

| Task | Score |
| --- | ---: |
| MATH500 | 99.4 |
| MMLU | 94.1 |
| RLPR | 44.75 |
| LiveCodeBench | 83.93 |
| AIME25 | 93.3 |
| BigCodeBench | 37.86 |
| GPQA-Diamond | 87.5 |
| Average | 77.27 |

The controlled multi-agent comparison in Table 7 reports Conductor at:

| Task | Score |
| --- | ---: |
| MATH500 | 89.33 |
| MMLU | 93.14 |
| RLPR | 42.63 |
| LiveCodeBench | 64.29 |
| Average | 72.35 |

## Local Reproduction Notes

The local environment has been configured with `uv` and Python 3.11. A smoke
OpenAI-compatible API call succeeded with `OPENAI_MODEL_NAME=gpt-4o-mini`.
The MMLU full-data training config was started on four GPUs and intentionally
interrupted around step 72/200, so it should be rerun to obtain final train/eval
metrics.
