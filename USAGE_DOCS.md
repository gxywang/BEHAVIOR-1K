# Usage docs for lab
## Basics
- `module load cuda-toolkit/13.0`
- on server: run `OMNIGIBSON_HEADLESS=1` and watch in Rerun (see [Bring-up](#bring-up)); `OMNIGIBSON_REMOTE_STREAMING=webrtc` is the unreliable alternative — never set both
- always `export CUDA_DEVICE_ORDER=PCI_BUS_ID` alongside `CUDA_VISIBLE_DEVICES=`, so indices match `nvidia-smi`
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

### Bring-up

Every shell that launches anything needs the GPU pin. Check `nvidia-smi` first — the box is shared.

```bash
cd ~/projects/BEHAVIOR-1K
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
```

**1. Services** (needed for `live` / `task`; not for `capture` or `stream_scene`). The planner answers `/health`
only after cuRobo warms up, ~40 s, and it hosts the Rerun viewer itself (web 9090, gRPC 9876, on 127.0.0.1): one
recording per planner process, gone when the planner exits. One terminal or tmux window each, in the foreground,
so Ctrl-C stops them.

```bash
M2T2_GPU=2 OmniGibson/omnigibson/tiptop/scripts/start_m2t2.sh
TIPTOP_GPU=2 TIPTOP_CONFIG=tiptop/config/tiptop_sim_r1pro.yml TIPTOP_PARTICLES=256 TIPTOP_MAX_PLANNING_TIME=40 \
  OmniGibson/omnigibson/tiptop/scripts/start_tiptop_server.sh

curl -s localhost:8123/health; curl -s localhost:8765/health    # {"status":"healthy"...} and OK
```

**2. Rerun** (the view of a run: planner's perception + plan, the simulator's robot, objects and cameras; see
`OmniGibson/omnigibson/tiptop/README.md` "Rerun" for what each entity is). On the laptop:

```bash
ssh -N -L 9090:127.0.0.1:9090 -L 9876:127.0.0.1:9876 shenlong-gpu-01
# then open http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy  and keep the time panel on Following
```

Reload the tab after restarting the planner. If `ssh -L` says a port is in use, a `rerun` on the laptop holds it:
close it, or map other local ports (`-L 9091:127.0.0.1:9090 -L 9877:127.0.0.1:9876` and the URL with 9091/9877).

**3. Run one of these.** (a) and (b) are headless and the picture is in Rerun (the robot's head camera and a
third-person overview are streamed there by the simulator, so no WebRTC client is needed); (c) is the WebRTC
path and must run without `OMNIGIBSON_HEADLESS`.

```bash
# a) the demo: load the scene, one basket onto the coffee table, stand once, fill the basket round by round
OMNIGIBSON_HEADLESS=1 ./b1k/bin/python -m omnigibson.tiptop.run live \
    --embodiment r1pro --activity assembling_gift_baskets \
    --place wicker_basket.n.01_2:table.n.02_1:0.28,0.16 \
    --stand-for butter_cookie.n.01_1,wicker_basket.n.01_2 --sequential \
    --goal "inside(butter_cookie.n.01_1,wicker_basket.n.01_2)" \
    --task "put the item in the wicker basket" --grasping-mode sticky --no-gt \
    --host localhost --port 8765 --out-dir runs/demo
    # more items from the same spot: list them all in --stand-for and --goal (README "Challenge tasks")

# b) capture one frame only, no planner
OMNIGIBSON_HEADLESS=1 ./b1k/bin/python -m omnigibson.tiptop.run capture \
    --embodiment r1pro --activity assembling_gift_baskets --out-dir runs/cap

# c) look at a scene over WebRTC (unreliable on this GPU, see below) -- no services needed
OMNIGIBSON_REMOTE_STREAMING=webrtc ./b1k/bin/python \
    OmniGibson/omnigibson/tiptop/scripts/stream_scene.py --activity assembling_gift_baskets \
    --place wicker_basket.n.01_2:table.n.02_1:0.28,0.16 \
    --stand-for butter_cookie.n.01_1,wicker_basket.n.01_2
```

The scene takes ~160 s to load; the robot appears in Rerun as soon as it is placed. Per round the client logs how
each perceived object pairs with a simulated one (`perceived 'candle' (goal, 85 grasps) = simulated candle_4 (2.9 cm
off)`); a goal object with no partner within 8 cm is a false detection. Stop everything with Ctrl-C in each
window, or `pkill -u $USER -f "tiptop-server|m2t2_server|omnigibson.tiptop.run"` for anything detached.

### Streaming the viewport to a laptop

- **Set `OMNIGIBSON_REMOTE_STREAMING=webrtc` and leave `OMNIGIBSON_HEADLESS` unset.** Streaming already runs the
  app windowless (`simulator.py:173`), but `tiptop/scene.py:184` only aims the viewport camera while `gm.HEADLESS`
  is false — set both and you stream a default pose.
- Client: the standalone **Isaac Sim WebRTC Streaming Client** (5.1). Type the bare IP, no port.
- **TCP 49100 is the only port that must reach the server.** OmniGibson selects
  `/app/livestream/proto = "websocket"` (`simulator.py:277`), so video rides the signalling socket and no UDP media
  port is bound; `ufw` is disabled on the host. An SSH tunnel therefore works as a fallback:
  `ssh -N -L 49100:127.0.0.1:49100 shenlong-gpu-01`, then point the client at `127.0.0.1`.
- **WebRTC streaming is unreliable on this GPU -- use Rerun instead.** The client connects and gets a few frames,
  then the encoder stops producing and the client drops, over and over: one session logged 4 `FIRST_FRAME_SENT`
  against 115 `Could not get encoded frame` and 24 `CLIENT_DISCONNECT_UNINTENDED`, which is the
  scene / black / scene cycle you see. The bundled StreamSDK (`omni.kit.streamsdk.plugins-7.6.3`) was built before
  this card shipped. `UseRefactoredVideoEncoder=1` (a StreamSDK regkey, read from the environment) looked like a
  fix on one short connection and is not one -- a longer session with it set still threw 115 errors. Do not rely on
  it. Two dead ends recorded so nobody repeats them: the `GPU ... is not white-listed` warning gates nothing
  (that branch falls through to the same success path), and NVENC on the card is healthy (h264/hevc/av1 all encode
  at 200+ fps outside Isaac Sim). There is no drop-in fix for 5.1 (checked 2026-09-05): the newer
  `omni.kit.streamsdk.plugins` 7.7.2 in the Kit 107 registry ships a byte-identical streaming server library, the
  official 5.1.0 container carries the same 7.6.3, and the StreamSDK builds that work on this card only come with
  Isaac Sim 6.0 (Kit 110, python 3.12).
- **What to use instead.** [Rerun](#bring-up): the planner's perception and plan, the simulator's robot and
  objects, and the robot's head camera plus a third-person overview streamed from the simulator at 5 Hz -- none
  of it needs an encoder -- and the per-round `live.mp4` the bridge writes. `stream_scene.py` and
  `OMNIGIBSON_REMOTE_STREAMING` still work when the stream happens to hold, but treat a working picture as luck
  rather than a guarantee.
- The stream shows only the main viewport: `vision_sensor.py` suppresses auxiliary camera windows under
  `REMOTE_STREAMING`. To confirm, `grep -oE "ViewportTexture_[0-9]+" <kit log> | sort -u` should print only `_0`,
  while `Replicator`, `Replicator_01` and `Replicator_02` are all still created (the sensors still render).
- When a session misbehaves, read the Kit log: `/app/livestream/logLevel=debug` and the WebRTC QoS callback are on,
  so `carb.livestream-rtc.plugin` lines carry signalling, peer state and disconnect reasons.
- `gm.PUBLIC_IP` defaults to gpu-01's `172.22.224.37`; on gpu-02 or campus-cluster export `OMNIGIBSON_PUBLIC_IP`.
- `OMNIGIBSON_HTTP_PORT` / port 8211 does nothing on Isaac Sim 5.x — no browser client is shipped.

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
