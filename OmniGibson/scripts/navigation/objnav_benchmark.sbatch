#!/bin/bash
#SBATCH --job-name=objnav_bench
#SBATCH -p shenlong2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH -o /scratch/gxwang2/b1k/objnav_eval/logs/bench_%A_%a.out
#
# Object-navigation benchmark, one array task per (scene, task) pair so Kit starts
# once per task and not once per target. Runs unattended: results are one JSON per
# (scene, task, target, condition), written as each call finishes, and every run
# skips what already exists -- a killed task loses at most one call.
#
#   RUN=/scratch/gxwang2/b1k/objnav_eval/run1
#   mkdir -p $RUN/logs
#   <build $RUN/pairs.txt: one "<scene> <task>" per line>
#   sbatch --array=1-$(wc -l < $RUN/pairs.txt)%4 --export=ALL,RUN=$RUN objnav_benchmark.sbatch
set -o pipefail

BK=/projects/illinois/eng/cs/shenlong/personals/gloria/projects/BEHAVIOR-1K
SUB=/scratch/gxwang2/b1k/b1k-submission/third-party
DET=/scratch/gxwang2/b1k/objnav2/detbench
RUN="${RUN:-/scratch/gxwang2/b1k/objnav_eval/run1}"
PAIRS="${PAIRS:-$RUN/pairs.txt}"
RESULTS="$RUN/results"
CONDITIONS="${CONDITIONS:-gt,detector}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ATTEMPTS="${ATTEMPTS:-3}"
WATCH="$RUN/det_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"

# Omniverse leaves a ~7 GB shader/texture cache behind; an uncleaned one is the
# likely cause of the drained shenlong node. Clean it on the way out -- but only
# once we are the last of our jobs on this node, because concurrent array tasks
# share the cache and wiping it under them costs every one of them a rebuild.
cleanup() {
  touch "$WATCH/STOP" 2>/dev/null
  local mine
  mine=$(squeue -h -u "$USER" -w "$(hostname -s)" -t R -o "%i" 2>/dev/null | wc -l)
  if [ "${mine:-0}" -le 1 ]; then
    echo "cleanup: last job of $USER on $(hostname -s), removing caches"
    rm -rf "/tmp/og_appdata_$USER" "/tmp/torchinductor_$USER"
  else
    echo "cleanup: $mine jobs of $USER still on $(hostname -s), keeping the shared cache"
  fi
}
trap cleanup EXIT

mkdir -p "$RESULTS" "$WATCH" "$RUN/logs"
# CUDA silently fails without this (stub libs shadow the driver) while the job still exits 0.
source $BK/cluster_env.sh
export PYTHONUNBUFFERED=1
export YOLO_CONFIG_DIR=$DET/yolo_cfg
export HF_HOME=$DET/hf_home

read -r SCENE TASK <<<"$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PAIRS")"
if [ -z "$SCENE" ] || [ -z "$TASK" ]; then echo "no pair at line ${SLURM_ARRAY_TASK_ID} of $PAIRS"; exit 2; fi
echo "=== $(date -Is) $(hostname -s) array=${SLURM_ARRAY_TASK_ID} $SCENE / $TASK conditions=$CONDITIONS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv
echo "3dmap $(git -C $SUB/3dmap rev-parse --short HEAD)  nav2py $(git -C $SUB/nav2py rev-parse --short HEAD)  B1K $(git -C $BK rev-parse --short HEAD)"

# Detector server, supervised: a crash must cost one move_base_to call, not the run.
DETLOG="$RUN/logs/detector_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
if [[ "$CONDITIONS" == *detector* ]]; then
  (
    cd $DET || exit 1
    while [ ! -f "$WATCH/STOP" ]; do
      PYTHONPATH=$SUB/3dmap/src $DET/venv/bin/python -m mapping_lab.objnav.detect_server \
        --watch "$WATCH" --detector yoloe --weights $DET/yoloe-11l-seg.pt --imgsz 1088 --conf 0.4 >>"$DETLOG" 2>&1
      [ -f "$WATCH/STOP" ] && break
      echo "[supervisor] detect_server exited $?, restarting" >>"$DETLOG"
      sleep 5
    done
  ) &
  sleep 30
fi

count_results() { find "$RESULTS/$SCENE/$TASK" -name "[0-9][0-9][0-9]_*.json" 2>/dev/null | wc -l; }

cd $BK/OmniGibson
STATUS=1
for attempt in $(seq 1 "$ATTEMPTS"); do
  before=$(count_results)
  echo "--- attempt $attempt/$ATTEMPTS ($before results so far)"
  $BK/b1k/bin/python scripts/navigation/run_objnav_benchmark.py \
    --scene "$SCENE" --task "$TASK" --conditions "$CONDITIONS" \
    --output-dir "$RESULTS" --nav2py-root $SUB/nav2py --mapping-lab-root $SUB/3dmap/src \
    --detector-dir "$WATCH" $EXTRA_ARGS
  STATUS=$?
  after=$(count_results)
  echo "--- attempt $attempt exit $STATUS, results $before -> $after"
  # Kit segfaults on teardown (exit 139) after the results are already on disk, so
  # progress, not the exit status, decides whether another attempt is worth it.
  if [ "$STATUS" -eq 0 ]; then break; fi
  if [ "$after" -le "$before" ] && [ "$attempt" -gt 1 ]; then
    echo "--- no progress in two attempts, giving up on $SCENE/$TASK"; break
  fi
done

touch "$WATCH/STOP"
wait
echo "=== $(date -Is) done $SCENE / $TASK results=$(count_results) last_exit=$STATUS"
