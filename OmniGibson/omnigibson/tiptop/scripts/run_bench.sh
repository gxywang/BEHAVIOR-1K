#!/usr/bin/env bash
# Run this checkout's bridge, even when the simulator environment has another b1k editable install.
set -euo pipefail
TIPTOP_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$TIPTOP_REPO_ROOT/tiptop:$TIPTOP_REPO_ROOT/OmniGibson${PYTHONPATH:+:$PYTHONPATH}"
export OMNIGIBSON_HEADLESS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "${SIM_PYTHON:-$TIPTOP_REPO_ROOT/b1k/bin/python}" -m omnigibson.tiptop.bench "$@"
