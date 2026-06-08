# NVfp4 GEMV Autoresearch

An advisor-worker agent pair that iteratively optimizes a CUDA kernel for batched NVfp4 matrix-vector multiplication on NVIDIA B200. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements it, evaluates on a B200 via Modal, and logs the result.

## Task

Implement a batched GEMV kernel for NVfp4 (e2m1) inputs with fp8 (e4m3fn) block scale factors, producing fp16 output. Ranked by geometric mean latency across three shapes.

`custom_kernel` receives a 7-element tuple — `a, b, sfa, sfb, sfa_permuted, sfb_permuted, c = data`:

| Tensor | Shape | Dtype |
|---|---|---|
| `a` | `M × K//2 × L` | `float4_e2m1fn_x2` |
| `b` | `128 × K//2 × L` | `float4_e2m1fn_x2` (row 0 only) |
| `sfa` | `M × K//16 × L` | `float8_e4m3fn` |
| `sfb` | `128 × K//16 × L` | `float8_e4m3fn` (row 0 only) |
| `sfa_permuted` | tcgen05 MMA layout | `float8_e4m3fn` |
| `sfb_permuted` | tcgen05 MMA layout | `float8_e4m3fn` |
| `c` | `M × 1 × L` | `float16` (output buffer) |

**Benchmark shapes and speed-of-light targets (B200 @ 1.5 GHz):**

| M | K | L | SOL (µs) |
|---|---|---|---|
| 7168 | 16384 | 1 | 8.622 |
| 4096 | 7168 | 8 | 17.275 |
| 7168 | 2048 | 4 | 4.317 |

## Setup

```bash
uv sync
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-sonnet-4-6   # optional, this is the default
```

## Running the agent

```bash
uv run nvfp4_gemv/agent.py --iterations 20
```

Start from a specific baseline file:

```bash
uv run nvfp4_gemv/agent.py --baseline nvfp4_gemv/baseline_v2.py --iterations 20
```

Use different models for advisor and worker:

```bash
uv run nvfp4_gemv/agent.py --advisor-model claude-opus-4-8 --worker-model claude-sonnet-4-6 --iterations 20
```

In tmux (recommended for long runs):

```bash
tmux new-session -d -s agent "set -a && source .env && set +a && uv run nvfp4_gemv/agent.py --baseline nvfp4_gemv/baseline_v2.py --iterations 25 2>&1 | tee nvfp4_gemv/agent_run.log"
tmux attach -t agent
```

Evaluate a baseline file without running the agent:

```bash
uv run nvfp4_gemv/run_eval.py nvfp4_gemv/baseline_v2.py -o results.json
```

## Structure

```
nvfp4_gemv/
├── agent.py            — advisor-worker agentic loop
├── advisor_prompt.md   — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md    — worker system prompt: mandatory sequence, rules
├── program.md          — original single-agent system prompt (kept for reference)
├── submission.py       — the kernel file the worker edits each iteration
├── run_eval.py         — submits submission.py to the Modal B200 evaluator
├── tools.py            — log_experiment and get_experiment_history tools
├── baseline_v2.py      — custom CUDA kernel baseline (21.3 µs geomean)
├── baseline43.py       — torch._scaled_mm baseline (202.1 µs geomean)
└── runs/               — one directory per run: history, TSV log, plots, best submission
```

Each run directory contains:
- `experiment_history.md` — full log of every attempt with code and result
- `results.tsv` — tab-separated summary for plotting
- `progress.png` — latency plot updated each iteration
- `best_submission.py` — snapshot of the fastest kernel found
- `proposals.md` — advisor proposals for every iteration
