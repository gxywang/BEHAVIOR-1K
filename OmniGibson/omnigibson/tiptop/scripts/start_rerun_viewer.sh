#!/bin/bash
# Headless Rerun web viewer for the TiPToP planner, on a server with no display.
#
# Runs the rerun CLI pinned in the tiptop pixi env (0.27.3, must match the SDK the planner logs with) as a gRPC
# server plus a web viewer, so nothing needs installing on the laptop -- a browser over an ssh -L tunnel is enough.
# Start this BEFORE the planner, then run the planner with TIPTOP_RERUN_MODE=connect.
#
# Env: TIPTOP_DIR (default: this repo's tiptop/ submodule), RERUN_GRPC_PORT (9876), RERUN_WEB_PORT (9090),
#      RERUN_MEMORY_LIMIT (4GB). Extra arguments are passed through (e.g. a .rrd path to replay one instead).
# See ../README.md "Rerun from the laptop".
set -e
unset LD_LIBRARY_PATH  # the pixi env ships its own libraries
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${TIPTOP_DIR:-$HERE/../../../../tiptop}"

RERUN="$PWD/.pixi/envs/default/lib/python3.12/site-packages/rerun_sdk/rerun_cli/rerun"
[ -x "$RERUN" ] || { echo "rerun CLI not found at $RERUN -- is the tiptop pixi env installed?" >&2; exit 1; }

GRPC="${RERUN_GRPC_PORT:-9876}"
WEB="${RERUN_WEB_PORT:-9090}"
cat >&2 <<MSG
Rerun $("$RERUN" --version | head -1 | cut -d' ' -f2) serving:
  gRPC (planner connects here) : rerun+http://127.0.0.1:${GRPC}/proxy
  web viewer (browser)         : http://127.0.0.1:${WEB}

On the laptop:
  ssh -N -L ${WEB}:127.0.0.1:${WEB} -L ${GRPC}:127.0.0.1:${GRPC} shenlong-gpu-01
  then open  http://127.0.0.1:${WEB}/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A${GRPC}%2Fproxy
Start the planner with TIPTOP_RERUN_MODE=connect (TIPTOP_RERUN_URL defaults to this address).
MSG

exec "$RERUN" --serve-web --bind 127.0.0.1 --port "$GRPC" --web-viewer-port "$WEB" \
  --server-memory-limit "${RERUN_MEMORY_LIMIT:-4GB}" "$@"
