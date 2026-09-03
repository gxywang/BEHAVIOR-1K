#!/bin/bash
# TiPToP planning server for the OmniGibson Franka Panda (ws://host:8765). Needs M2T2 running (start_m2t2.sh).
# Env: TIPTOP_DIR (default: this repo's tiptop/ submodule), TIPTOP_CONFIG (default tiptop/config/tiptop_sim_panda.yml;
#      tiptop/config/tiptop_sim_r1pro.yml for the R1Pro), TIPTOP_HOST/TIPTOP_PORT, TIPTOP_PARTICLES,
#      TIPTOP_MAX_PLANNING_TIME, TIPTOP_RERUN_MODE (disabled|stream|connect|save), TIPTOP_RERUN_URL (connect mode).
#      Extra arguments are passed through to tiptop-server. See ../DEPLOYMENT.md.
set -e
export PATH="$HOME/.pixi/bin:$PATH"
unset LD_LIBRARY_PATH  # the pixi env ships its own CUDA libraries; host CUDA on the path breaks it
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${TIPTOP_DIR:-$HERE/../../../../tiptop}"
exec pixi run tiptop-server --config "${TIPTOP_CONFIG:-tiptop/config/tiptop_sim_panda.yml}" \
  --host "${TIPTOP_HOST:-127.0.0.1}" --port "${TIPTOP_PORT:-8765}" --num-particles "${TIPTOP_PARTICLES:-128}" \
  --max-planning-time "${TIPTOP_MAX_PLANNING_TIME:-30}" --rerun-mode "${TIPTOP_RERUN_MODE:-disabled}" \
  --rerun-url "${TIPTOP_RERUN_URL:-rerun+http://127.0.0.1:9876/proxy}" "$@"
