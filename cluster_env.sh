#!/bin/bash
# Source this before running OmniGibson/Isaac Sim on the Illinois Campus Cluster
# (shenlong / shenlong2 partitions). See DEBUG.md for why each line is needed.
#
#   source cluster_env.sh
#   python -m omnigibson.examples.environments.behavior_env_demo

# 1. CRITICAL: the cuda/* module prepends .../lib64/stubs to LD_LIBRARY_PATH.
#    Those stub libs carry the real SONAMEs (libcuda.so.1, libnvidia-ml.so.1) and
#    shadow the actual driver in /usr/lib64, so NVML returns 9 (DRIVER_NOT_LOADED)
#    and cuInit returns 34 (STUB_LIBRARY). Kit then reports "No CUDA devices found".
#    Only the UNVERSIONED sonames (libcuda.so, libnvidia-ml.so) live in the
#    stub dir, so only code that dlopens those is affected -- which Isaac Sim's
#    carb.cudainterop does. Guard the empty case so we never export a
#    set-but-empty LD_LIBRARY_PATH.
if [ -n "$LD_LIBRARY_PATH" ]; then
  _og_clean="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/stubs$' | paste -sd:)"
  if [ -n "$_og_clean" ]; then export LD_LIBRARY_PATH="$_og_clean"; else unset LD_LIBRARY_PATH; fi
  unset _og_clean
fi

# 2. Keep the ~7 GB Omniverse shader/texture cache off Lustre (/projects).
#    On Lustre, Kit startup takes ~5 min; node-local disk cuts it substantially.
export OMNIGIBSON_APPDATA_PATH="${OMNIGIBSON_APPDATA_PATH:-/tmp/og_appdata_${USER}}"
mkdir -p "$OMNIGIBSON_APPDATA_PATH"

# 5. torch.compile cache on node-local disk (BehaviorTask sampling triggers inductor).
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_${USER}}"

# 6. WebRTC streaming endpoint.
#    OmniGibson hardcodes PUBLIC_IP to 172.22.224.37 (shenlong-gpu-01), which
#    is wrong on the campus cluster. The streaming client must be told an
#    address IT can reach, so prefer this node's campus-routable address
#    (hsn0.1840, 141.142.x.x) over the cluster-internal one (hsn0, 172.29.x.x).
#
#    NB: `hostname -I` lists a useless link-local 169.254.x address FIRST on
#    these nodes (that is the BMC USB NIC) -- always select explicitly.
#
#    Override cases:
#      - reaching the node via an SSH tunnel  -> OMNIGIBSON_PUBLIC_IP=127.0.0.1
#      - client running inside the cluster    -> use the 172.29.x address
if [ -z "$OMNIGIBSON_PUBLIC_IP" ]; then
  # campus-routable (what an off-cluster streaming client needs)
  OMNIGIBSON_PUBLIC_IP="$(hostname -I | tr ' ' '\n' | grep -E '^141\.142\.' | head -1)"
  # fall back to cluster-internal, then to the default-route address
  [ -z "$OMNIGIBSON_PUBLIC_IP" ] && OMNIGIBSON_PUBLIC_IP="$(hostname -I | tr ' ' '\n' | grep -E '^172\.29\.' | head -1)"
  [ -z "$OMNIGIBSON_PUBLIC_IP" ] && OMNIGIBSON_PUBLIC_IP="$(ip -4 route get 8.8.8.8 2>/dev/null | grep -oP 'src \K\S+')"
  export OMNIGIBSON_PUBLIC_IP
fi

echo "cluster_env: host=$(hostname -s) appdata=$OMNIGIBSON_APPDATA_PATH"
echo "cluster_env: streaming client should connect to -> $OMNIGIBSON_PUBLIC_IP  (ports 8211 HTTP / 49100 WebRTC)"
echo "cluster_env: nvml check ->"
python3 -c "import ctypes;print('  nvmlInit rc =',ctypes.CDLL('libnvidia-ml.so.1').nvmlInit_v2(),'(0 = OK)')" 2>/dev/null \
  || echo "  (python3 unavailable for check)"
