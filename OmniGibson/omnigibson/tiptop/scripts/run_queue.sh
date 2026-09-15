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
# torch takes one CPU thread per core by default -- 128 on this box -- and every concurrent run does the same. Six
# runs is then some 800 threads over 256 cores, and the whole machine slows down together: measured 1.37 simulator
# steps per second with the box quiet against 0.47 with six runs going, so each run took three times as long and
# the throughput gain from running them at once was almost nothing. OmniGibson's own evaluator has the same knob
# (eval/evaluator.py TORCH_NUM_THREADS) and leaves it off. Override by exporting OMP_NUM_THREADS.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$(dirname "$OUT")" runs/queue_logs
# The pipeline needs three services, not one: the planner this queue talks to, and the M2T2 grasp server the
# planner itself calls for grasp proposals (perception.m2t2.url in the planner's config, 127.0.0.1:8123). A run
# without M2T2 does not fail loudly -- every round comes back "TiptopPlanningError: Cannot connect to host
# localhost:8123" and the instance scores 0.0, which reads exactly like a task the pipeline cannot do. That cost
# three runs on 2026-09-13. Check both before spending an hour on a job.
for service in "$PORT:the planner" "8123:the M2T2 grasp server"; do
  port="${service%%:*}"
  if ! python3 -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if not s.connect_ex(('127.0.0.1', int('$port'))) else 1)"; then
    echo "=== QUEUE REFUSED: nothing is listening on 127.0.0.1:$port (${service#*:})" >> "$OUT"
    exit 2
  fi
done

while read -r task instances; do
  [ -z "${task:-}" ] && continue
  case "$task" in \#*) continue ;; esac
  stamp="$(date +%m%d_%H%M%S)"
  dir="runs/queue_${task}_${stamp}"
  log="runs/queue_logs/${task}_${stamp}.log"
  # Which code and which planner produced this number. A queue picks each job up when the one before it finishes,
  # so a long queue spans several commits and the jobs at the end are not running what the jobs at the start ran.
  # Without this line a result cannot be attributed to a change, which is most of what a result is for. The
  # PYTHONPATH matters just as much: unset, python imports the MAIN tree's omnigibson no matter which worktree
  # the command was typed in.
  code="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  tree="${PYTHONPATH:-<main tree: no PYTHONPATH set>}"
  echo "=== JOB $task [$instances] -> $dir" >> "$OUT"
  echo "    code $code, planner :$PORT on gpu $GPU, imports $tree" >> "$OUT"
  # A task whose goal is toggled_on(...) and whose description says press: hold needs a SECOND planner for the
  # other arm -- one hand holds the thing, the other presses it. Without PRESS_PORT the instance crashes on
  # "press 'hold' needs the right-arm planner", which is how turning_on_radio, a task that scored 0.9, came back
  # as a crash on 2026-09-14. Pass PRESS_PORT=<port of an r1pro_right server> for those tasks.
  press_args=""
  [ -n "${PRESS_PORT:-}" ] && press_args="--press-port ${PRESS_PORT}"
  ./b1k/bin/python -m omnigibson.tiptop.bench --task-name "$task" --instances $instances \
    --knowledge "$KNOWLEDGE" --grasping-mode sticky --torso 1.2 -1.7 -0.9 0.0 \
    --views $VIEWS --host localhost --port "$PORT" $press_args --out-dir "$dir" > "$log" 2>&1
  code=$?
  python3 OmniGibson/omnigibson/tiptop/scripts/read_run.py "$dir" "$log" >> "$OUT" 2>&1 \
    || echo "  (read_run failed; bench exited $code, log $log)" >> "$OUT"
  echo "=== DONE $task exit $code" >> "$OUT"
done < "$JOBS"
echo "=== QUEUE EMPTY" >> "$OUT"
