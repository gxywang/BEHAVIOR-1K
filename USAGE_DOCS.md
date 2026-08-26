# Usage docs for lab
## Basics
- on server: `export OMNIGIBSON_HEADLESS=1` and `export OMNIGIBSON_REMOTE_STREAMING=webrtc`
  - if using `CUDA_VISIBLE_DEVICES=`, make sure to add this to `.bashrc` to make GPU ordering consistent with `nvidia_smi`: `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- to run `uv` install script: `bash setup_uv.sh   --new-env b1k   --omnigibson   --bddl   --dataset  --joylo  --eval --accept-nvidia-eula   --accept-dataset-tos   --cuda-version 13.0`
- download [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
- on streaming client, connect to server `172.22.224.37`
