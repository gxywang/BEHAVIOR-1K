# Usage docs for lab
## Basics
- `module load cuda-toolkit/13.0`
- on server: `export OMNIGIBSON_HEADLESS=1` and `export OMNIGIBSON_REMOTE_STREAMING=webrtc`
  - if using `CUDA_VISIBLE_DEVICES=`, make sure to add this to `.bashrc` to make GPU ordering consistent with `nvidia_smi`: `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- to run `uv` install script: `bash setup_uv.sh   --new-env b1k   --omnigibson   --bddl   --dataset  --joylo  --eval --accept-nvidia-eula   --accept-dataset-tos`
- on local machine, download [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
- on local streaming client, connect to server  
  - for `shenlong-gpu-01`: `172.22.224.37`
  - for `shenlong-gpu-02`: `172.22.224.85`
- on server, export the corresponding IP to `export OMNIGIBSON_PUBLIC_IP=`

## Debug
- If using `campus-cluster`, and running into CUDA issues, try `source cluster_env.sh`

## Dataset info
- robot: R1Pro
- action_frequency: 30
- physics_frequency: 120
- rendering_frequency: 30