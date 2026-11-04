#!/bin/bash
# Kuyil's size ladder on one GPU box: 3 sizes x 3 seeds, all at once (each run needs well under 2 GB).
#   PYTHON=.venv/bin/python ./run_ladder.sh indictts.pt    -> runs/<size>-s<seed>/{log.jsonl,best.pt,last.pt}, logs runs/*.log
# Totals with the 816,597-parameter vocoder: 1.4m = 1,401,562 (PiVo's sizes), 5m = 4,892,394, 20m = 22,651,402.
set -euo pipefail
PACK=${1:?usage: run_ladder.sh PACK}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}  # 9 runs x torch's default (one thread per core) swamps the CPUs
declare -A SIZE=(
  [1.4m]=""
  [5m]="--width 192 --ff-width 512 --text-layers 8 --decoder-layers 4 --duration-width 128"
  [20m]="--width 384 --ff-width 1024 --heads 8 --kv-heads 4 --text-layers 10 --decoder-layers 6 --duration-width 256"
)
mkdir -p runs
for size in 1.4m 5m 20m; do
  for seed in 42 43 44; do
    # shellcheck disable=SC2086
    nohup nice -n 10 "${PYTHON:-python}" train.py "$PACK" "runs/$size-s$seed" --seed "$seed" ${SIZE[$size]} > "runs/$size-s$seed.log" 2>&1 &
  done
done
wait
grep -h '"epoch"' runs/*.log | tail -n 9
# Keep the results before the pod goes: every run's checkpoints and logs to R2, then verify by checksum.
DEST=${R2_DEST:-r2:zhagaram-data/kuyil/v0/runs}
rclone copy runs "$DEST" && rclone check runs "$DEST" --one-way && echo "BACKED UP to $DEST" || echo "BACKUP FAILED: rerun: rclone copy runs $DEST"
