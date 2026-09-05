#!/bin/bash
# TiPToP planning server (ws://host:8765). Needs M2T2 at the config's perception.m2t2.url (start_m2t2.sh, :8123).
# Env: TIPTOP_DIR (default: this repo's tiptop/ submodule), TIPTOP_CONFIG (default tiptop/config/tiptop_sim_panda.yml;
#      tiptop/config/tiptop_sim_r1pro.yml for the R1Pro), TIPTOP_HOST/TIPTOP_PORT (127.0.0.1:8765),
#      TIPTOP_PARTICLES (128 here; tiptop-server's own default is 256), TIPTOP_MAX_PLANNING_TIME (30 s; server 60),
#      TIPTOP_GPU (index or UUID; pins CUDA_VISIBLE_DEVICES on shared multi-GPU boxes).
#      TIPTOP_RERUN_MODE (default serve: this process hosts the Rerun web viewer on 127.0.0.1, TIPTOP_RERUN_WEB_PORT
#      9090 + TIPTOP_RERUN_GRPC_PORT 9876, one recording for its lifetime; connect|stream|save|disabled as in
#      tiptop_websocket_server.py), TIPTOP_RERUN_URL (connect mode). Extra arguments go to tiptop-server.
# See ../DEPLOYMENT.md and ../README.md "What Rerun shows".
set -e
export PATH="$HOME/.pixi/bin:$PATH"
unset LD_LIBRARY_PATH  # the pixi env ships its own CUDA libraries; host CUDA on the path breaks it
# Shared multi-GPU boxes: TIPTOP_GPU=<index|uuid> pins this service to one card. Unset (laptop) = unchanged.
if [ -n "${TIPTOP_GPU:-}" ]; then export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$TIPTOP_GPU"; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${TIPTOP_DIR:-$HERE/../../../../tiptop}"
exec pixi run tiptop-server --config "${TIPTOP_CONFIG:-tiptop/config/tiptop_sim_panda.yml}" \
  --host "${TIPTOP_HOST:-127.0.0.1}" --port "${TIPTOP_PORT:-8765}" --num-particles "${TIPTOP_PARTICLES:-128}" \
  --max-planning-time "${TIPTOP_MAX_PLANNING_TIME:-30}" --rerun-mode "${TIPTOP_RERUN_MODE:-serve}" \
  --rerun-url "${TIPTOP_RERUN_URL:-rerun+http://127.0.0.1:9876/proxy}" \
  --rerun-web-port "${TIPTOP_RERUN_WEB_PORT:-9090}" --rerun-grpc-port "${TIPTOP_RERUN_GRPC_PORT:-9876}" "$@"
