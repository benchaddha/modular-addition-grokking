#!/bin/zsh
# Strategy 3 trajectory sweep driver: one background job per seed-cell.
set -u
cd "$(dirname "$0")/.."
PY=modular-addition-env/bin/python
OUTDIR=results/strategy3
mkdir -p "$OUTDIR" "$OUTDIR/logs"

run_job() {
  local label=$1; shift
  local ckpts=("$@")
  OMP_NUM_THREADS=2 $PY scripts/run_strategy3_trajectory.py \
    --checkpoints "${ckpts[@]}" \
    --out "$OUTDIR/trajectory_${label}.jsonl" \
    --skip-existing \
    > "$OUTDIR/logs/${label}.log" 2>&1
  echo "JOB_DONE ${label} exit=$?"
}

pids=()
for cell in p97_1layer p113_2layer; do
  for seed in 100 101 102 103 104; do
    ckpts=(results/strategy2/${cell}/seed_${seed}/artifacts/*.pt)
    ckpts=(${ckpts:#*_best.pt})
    run_job "${cell}_seed${seed}" "${ckpts[@]}" &
    pids+=($!)
  done
done
for seed in 52 53 54; do
  ckpts=(results/physics_checkpoint_early_window/seed_${seed}/t0/artifacts/*.pt)
  ckpts=(${ckpts:#*_best.pt})
  run_job "original_seed${seed}" "${ckpts[@]}" &
  pids+=($!)
done
wait
echo "ALL_JOBS_DONE"
