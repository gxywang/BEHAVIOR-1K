# Usage docs for lab
## Basics
- `module load cuda-toolkit/13.0`
- on server: `export OMNIGIBSON_REMOTE_STREAMING=webrtc` to stream, `export OMNIGIBSON_HEADLESS=1` to run without a viewport
  - set only one: streaming already forces the app headless, and `OMNIGIBSON_HEADLESS=1` additionally skips aiming the viewport camera, so you stream a default pose (see the runbook below)
  - if using `CUDA_VISIBLE_DEVICES=`, make sure to add this to `.bashrc` to make GPU ordering consistent with `nvidia_smi`: `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- to run `uv` install script: `bash setup_uv.sh   --new-env b1k   --omnigibson   --bddl   --dataset  --joylo  --eval --accept-nvidia-eula   --accept-dataset-tos`
- download [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
- on streaming client, connect to server `172.22.224.37` for `shenlong-gpu-01`
  - for `campus-cluster`'s `shenlong` partition: `172.29.128.5`
  - for `shenlong-gpu-02`: `172.22.224.85`

## shenlong-gpu-01 runbook (set up 2026-09-04)

Everything is installed and verified on `shenlong-gpu-01` (8x RTX PRO 6000 Blackwell, sm_120, driver 580/CUDA 13,
1.5 TB RAM, no sudo, no conda). Three isolated envs as `OmniGibson/omnigibson/tiptop/DEPLOYMENT.md` prescribes:

| Component | Where | Contents |
|---|---|---|
| sim + bridge | uv venv `b1k` (repo root) | python 3.11, torch 2.7.0+cu128 (sm_120), Isaac Sim 5.1, OmniGibson editable |
| TiPToP planner | `tiptop/.pixi` (16 GB) | python 3.12, torch 2.7.1/cu129, cuRobo `b5fad1d` + cuTAMP 0.0.6, all 5 CUDA kernels sm_120 |
| M2T2 grasp server | `~/tiptop-services/M2T2/.pixi` (13 GB) | python 3.11, torch 2.8.0/cu129, `pointnet2_ops` sm_120 |

M2T2's committed lock could not target Blackwell (DEPLOYMENT issue 2); its `pixi.toml` is patched locally to
`[system-requirements] cuda = "12.8"`, `cuda-toolkit = "12.9.*"`, `pytorch-gpu = ">=2.7,<2.9"` (`pixi.toml.orig`
keeps the original). That patch is NOT in git — re-apply it on any fresh M2T2 clone.

### GPU pinning (shared box — other users hold most cards)

Always pin, in every launch shell. `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes indices match `nvidia-smi`:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2   # check nvidia-smi first; 2 was free on 2026-09-04
```

- **Leave `OMNIGIBSON_GPU_ID` unset.** It maps to `--/renderer/activeGpu=N --/physics/cudaDevice=N`
  (`simulator.py:172`), and under the mask CUDA ordinal 2 does not exist — PhysX would target a missing device.
- `CUDA_VISIBLE_DEVICES` **does** pin Isaac Sim's Vulkan renderer, not just compute: Kit drops every Vulkan device
  whose CUDA context fails. Verify in the Kit log's `[gpu.foundation]` table
  (`OmniGibson/appdata/local/logs/Kit/OmniGibson/3.9/kit_*.log`): the `Active | Yes: 0` row's **Bus-ID** must be the
  target card's (`0x75` = GPU 2). The 14 `Skipping NVIDIA GPU due CUDA being in bad state` warnings are the 7 masked
  cards and are expected.
- The service launchers now take `TIPTOP_GPU` / `M2T2_GPU` (index or UUID); unset = unchanged laptop behaviour.

### Streaming the viewport to a laptop

```bash
export OMNIGIBSON_REMOTE_STREAMING=webrtc      # and do NOT set OMNIGIBSON_HEADLESS
```

- **Unset `OMNIGIBSON_HEADLESS`.** Streaming already forces the app headless (`simulator.py:173`), but
  `tiptop/scene.py:166` only aims the viewport camera when `gm.HEADLESS` is false — with it set you stream a
  default camera pose.
- Client: the standalone **Isaac Sim WebRTC Streaming Client** (5.1). Type the bare IP `172.22.224.37`, no port.
- Ports: **TCP 49100 only** — and nothing else. OmniGibson sets `/app/livestream/proto = "websocket"`
  (`simulator.py:277`), so the video rides the same WebSocket as the signalling; no UDP media port is ever bound.
  Verified 2026-09-04 with a client on the campus network: `ss -uanp` shows no UDP media socket, the Kit log reports
  `NVST_CCE_CONNECTED` / `All Streams connected`, and the picture arrives. 49100 is hardcoded in the client and binds
  `0.0.0.0`. `ufw` is disabled on the host. This also means an SSH tunnel works as a fallback:
  `ssh -N -L 49100:127.0.0.1:49100 shenlong-gpu-01`, then point the client at `127.0.0.1`.
- The stream shows **only the main viewport**. Until 2026-09-04 every robot/external camera also docked its own
  ViewportWindow into the streamed frame (`gm.HEADLESS` is false while streaming), which cluttered the picture and
  cost an extra render per camera per frame; `vision_sensor.py` now suppresses those under `REMOTE_STREAMING`.
  Check with `grep -oE "ViewportTexture_[0-9]+" <kit log> | sort -u` — only `_0` should appear, while
  `Replicator`, `Replicator_01`, `Replicator_02` must all still be created (the sensors still render for capture).
- Streaming problems now leave a trail: `/app/livestream/logLevel=debug` and `webrtc/logQosStatus` are set, so the
  Kit log carries signalling, peer state and reason codes instead of just CONNECTED/DISCONNECTED.
- `gm.HTTP_PORT` / the `:8211` browser client is **dead** on Isaac Sim 5.1 — `omni.services.streamclient.webrtc`
  is not shipped and nothing binds the port. The startup line now prints the address for the desktop client.
- `gm.PUBLIC_IP` is hardcoded to gpu-01's `172.22.224.37`; on gpu-02 or campus-cluster export `OMNIGIBSON_PUBLIC_IP`.

### Running the stack

```bash
cd ~/projects/BEHAVIOR-1K
M2T2_GPU=2 OmniGibson/omnigibson/tiptop/scripts/start_m2t2.sh &                  # http://127.0.0.1:8123
TIPTOP_GPU=2 TIPTOP_CONFIG=tiptop/config/tiptop_sim_r1pro.yml TIPTOP_PARTICLES=256 \
  TIPTOP_MAX_PLANNING_TIME=40 TIPTOP_RERUN_MODE=save \
  OmniGibson/omnigibson/tiptop/scripts/start_tiptop_server.sh &                  # ws://127.0.0.1:8765
curl -s localhost:8123/health; curl -s localhost:8765/health                     # planner answers after cuRobo warm-up

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 OMNIGIBSON_HEADLESS=1 \
  b1k/bin/python -m omnigibson.tiptop.run live --embodiment r1pro \
    --activity assembling_gift_baskets --place wicker_basket.n.01_2:table.n.02_1:0.28,0.16 --sequential \
    --goal "inside(butter_cookie.n.01_1,wicker_basket.n.01_2);inside(candle.n.01_2,wicker_basket.n.01_2)" \
    --task "put the item in the wicker basket" --grasping-mode sticky --no-gt \
    --host localhost --port 8765 --out-dir runs/server_live
```

Use `--rerun-mode save` (DEPLOYMENT issue 9): `stream` needs a display. `runs/stream_scene.py` loads the same
scene and idles, for streaming without running an episode.

Timings measured here: scene + task load 160 s, capture 10 s, plan 4-8 s, execute 25 s, goal scoring 10 s.
VRAM with all three services up: ~21 GB of 96 GB (sim 14 GB, planner 5.6 GB, M2T2 1.2 GB).

Notes that differ from DEPLOYMENT.md's predictions:
- pixi is 0.79; `pixi install --locked` did **not** rewrite `pixi.lock` (issue 4 was about 0.78).
- cuRobo's unpinned `origin/main` happened to be the known-good `b5fad1d`; a later `setup-planners` may move it.
- Issue 17 (RAM) is moot at 1.5 TB. `data.pyg.org` resolves again; irrelevant to the pixi envs either way.
- The private submodule uses a repo-scoped deploy key at `~/.ssh/id_ed25519` plus a global
  `url."git@github.com:WenzhouDing/".insteadOf` rewrite. Other GitHub SSH clones from this account will fail
  ("Repository not found") because `~/.ssh/config` sets `IdentitiesOnly yes` — use HTTPS for those.

## TiPToP submodule (branch `dev/tiptop`)
- `tiptop/` is a git submodule: BEHAVIOR-1K only pins a tiptop commit (URL/branch in `.gitmodules`; `git submodule status` prefix `-` = not checked out, `+` = checkout differs from the pin)
  - `.gitmodules` points at the private repo `WenzhouDing/tiptop` (read access needed); upstream `tiptop-robot/tiptop` is only reachable through the `upstream` remote inside `tiptop/`
- fresh clone: `git clone --recurse-submodules -b dev/tiptop https://github.com/gxywang/BEHAVIOR-1K.git`
- running TiPToP against the sim (three isolated envs, launchers) and deploying it on a server incl. known issues (Blackwell, cuRobo/M2T2 rebuilds): `OmniGibson/omnigibson/tiptop/README.md` and `OmniGibson/omnigibson/tiptop/DEPLOYMENT.md`
  - existing clone / empty `tiptop/`: `git submodule update --init`
  - once the submodule points at the private repo you need read access to it for these to succeed; the rest of the clone works without it
- after every `git pull`: `git submodule update --init` (`git pull` alone leaves `tiptop/` at the old commit; `git status` then shows `modified: tiptop (new commits)`)
  - consumers only, once: `git config submodule.recurse true` makes `git pull`/`git checkout` update `tiptop/` automatically (skip if you develop in `tiptop/`: every checkout or pull that moves the pin detaches your `tiptop/` branch)
- install: `pip install -e tiptop` after `git submodule update --init` (version comes from tiptop's git tags; `setup.sh` does not install it)
- develop in `tiptop/` (fresh clones and `git submodule update` leave it on a detached HEAD; commits made there are hidden by the next `git submodule update`):
  - `git -C tiptop checkout main && git -C tiptop pull --ff-only`
  - edit, `git -C tiptop commit`, `git -C tiptop push` (push tiptop FIRST)
  - `git add tiptop && git commit -m "tiptop: bump to $(git -C tiptop rev-parse --short HEAD)" && git push`
  - once per clone: `git config push.recurseSubmodules check` (refuses to push a pin nobody can fetch), `git config diff.submodule log`, `git config status.submoduleSummary true`
  - hooks: `cd tiptop && pre-commit install` (BEHAVIOR-1K hooks never run on submodule commits)
  - never `git commit -a` / `git add -A` in BEHAVIOR-1K unless you mean to move the pin; undo a stray move: unstaged: `git submodule update`; after `git add -A`: `git restore --staged tiptop && git submodule update`; after `git commit -a`: `git reset --soft HEAD~1 && git restore --staged tiptop && git submodule update`
  - on branches without `.gitmodules` (e.g. `main`) `tiptop/` shows as untracked (with `submodule.recurse true` its working tree is removed and restored on checkout); leave it alone and never `git clean -ffd` / `-ffdx` in BEHAVIOR-1K (`-fd` skips nested repos, the second `-f` deletes them)
- move the pin to latest `main` without developing: `git -C tiptop checkout main && git -C tiptop pull --ff-only && git add tiptop && git commit -m "tiptop: bump"` (`git submodule update --remote tiptop` does the same fetch but leaves a detached HEAD)
- `upload-pack: not our ref <sha>` / `Fetched in submodule path 'tiptop', but it did not contain <sha>`: the pinned commit is not on the URL your `tiptop/` uses; either the author forgot `git -C tiptop push`, or your URL is stale (next bullet)
- re-point the submodule from upstream to the private repo (done 2026-09-02 for `WenzhouDing/tiptop`; kept for reference):
  - `git submodule set-url tiptop https://github.com/<user>/tiptop.git && git -C tiptop push -u origin main --tags` (`set-url` also re-points `origin` inside `tiptop/`; `--tags` keeps the setuptools_scm versions)
  - `git add .gitmodules && git commit -m "tiptop: track private repo"`, and push this BEFORE the first pin that only exists in the private repo
  - everyone else, once that commit is on the remote: `git pull --no-recurse-submodules && git submodule sync tiptop && git submodule update --init` (a plain `git pull` aborts with `not our ref` because it tries to fetch the new pin from the old URL, and `sync` only sees the new URL after the `.gitmodules` change is merged, hence `--no-recurse-submodules` first, then `sync`, then `update`)
  - use ssh while `.gitmodules` stays https (fetch and push, survives `git submodule sync`): `git -C tiptop config url."git@github.com:<user>/".insteadOf "https://github.com/<user>/"`
  - pull upstream tiptop-robot changes: `git -C tiptop remote add upstream https://github.com/tiptop-robot/tiptop.git` (once per clone; skip if `git -C tiptop remote -v` already lists `upstream`), then `git -C tiptop fetch upstream && git -C tiptop merge upstream/main`
  - CI: `.github/workflows/tests.yml` and `profiling.yml` check out with `submodules: true`; with a private submodule those checkouts fail unless the checkout step is given a token or ssh key with access
