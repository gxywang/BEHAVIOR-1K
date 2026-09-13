#!/bin/bash
# Work through a queue of benchmark jobs, reading each one against baselines.json as it finishes.
#
#   scripts/run_queue.sh JOBS.txt GPU PORT [OUT]
#
# JOBS.txt: one job per line, "task instances..." (blank lines and # comments ignored), e.g.
#   putting_away_toys 0 1
#   assembling_gift_baskets 0 1 2
# OUT (default runs/queue_report.txt): every job's read_run.py summary appended as it completes, each headed by
# "=== JOB <task> <instances> -> <run dir>" and closed by "=== DONE". Watch that file to follow a queue without
# watching the runs: it is what turns "wait for the run and look at it" into something that reports itself.
#
# Views: head + wrist cameras by default, or head pitch views for a task whose room stops the wrist swings
# (VIEWS env var). Everything else matches the challenge-style bench invocation in README.md.
set -u
JOBS="${1:?usage: run_queue.sh JOBS.txt GPU PORT [OUT]}"
GPU="${2:?}"
PORT="${3:?}"
OUT="${4:-runs/queue_report.txt}"
VIEWS="${VIEWS:-head left_wrist right_wrist}"
KNOWLEDGE="${KNOWLEDGE:-oracle}"
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."   # repo root
export OMNIGIBSON_HEADLESS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$(dirname "$OUT")" runs/queue_logs
while read -r task instances; do
  [ -z "${task:-}" ] && continue
  case "$task" in \#*) continue ;; esac
  stamp="$(date +%m%d_%H%M%S)"
  dir="runs/queue_${task}_${stamp}"
  log="runs/queue_logs/${task}_${stamp}.log"
  echo "=== JOB $task [$instances] -> $dir" >> "$OUT"
  ./b1k/bin/python -m omnigibson.tiptop.bench --task-name "$task" --instances $instances \
    --knowledge "$KNOWLEDGE" --grasping-mode sticky --torso 1.2 -1.7 -0.9 0.0 \
    --views $VIEWS --host localhost --port "$PORT" --out-dir "$dir" > "$log" 2>&1
  code=$?
  python3 OmniGibson/omnigibson/tiptop/scripts/read_run.py "$dir" "$log" >> "$OUT" 2>&1 \
    || echo "  (read_run failed; bench exited $code, log $log)" >> "$OUT"
  echo "=== DONE $task exit $code" >> "$OUT"
done < "$JOBS"
echo "=== QUEUE EMPTY" >> "$OUT"
