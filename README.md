# Mechanistic Interpretability of a Toy LLM

Reproducible experiments for grokking on modular addition, with a focus on
mechanistic interpretability (baseline grokking, Fourier structure, noise
experiments, and head ablations).

## Project Layout

```text
configs/
notebooks/
results/
scripts/
src/
```

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run experiments from the project root as modules/scripts (to be added in
later commits).

Smoke run:

```bash
WANDB_MODE=offline python3 scripts/run_baseline.py --config configs/smoke.yaml
```

Baseline run:

```bash
WANDB_MODE=offline python3 scripts/run_baseline.py --config configs/default.yaml
```
