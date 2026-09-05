#!/bin/bash
# M2T2 grasp server for TiPToP. Env: M2T2_DIR (default ~/tiptop-services/M2T2), M2T2_HOST / M2T2_PORT (default
# 127.0.0.1:8123 -- must match perception.m2t2.url in the planner's config, tiptop/config/*.yml, which has no flag
# for it), M2T2_GPU (index or UUID; pins CUDA_VISIBLE_DEVICES on shared multi-GPU boxes).
# See ../DEPLOYMENT.md for how the M2T2 clone is set up.
set -e
export PATH="$HOME/.pixi/bin:$PATH"
unset LD_LIBRARY_PATH
# Shared multi-GPU boxes: M2T2_GPU=<index|uuid> pins this service to one card. Unset (laptop) = unchanged.
if [ -n "${M2T2_GPU:-}" ]; then export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$M2T2_GPU"; fi
cd "${M2T2_DIR:-$HOME/tiptop-services/M2T2}"
exec pixi run server --host "${M2T2_HOST:-127.0.0.1}" --port "${M2T2_PORT:-8123}" "$@"
