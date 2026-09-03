#!/bin/bash
# M2T2 grasp server for TiPToP (http://host:8123). Env: M2T2_DIR (default ~/tiptop-services/M2T2), M2T2_HOST, M2T2_PORT.
# See ../DEPLOYMENT.md for how the M2T2 clone is set up.
set -e
export PATH="$HOME/.pixi/bin:$PATH"
unset LD_LIBRARY_PATH
cd "${M2T2_DIR:-$HOME/tiptop-services/M2T2}"
exec pixi run server --host "${M2T2_HOST:-127.0.0.1}" --port "${M2T2_PORT:-8123}" "$@"
