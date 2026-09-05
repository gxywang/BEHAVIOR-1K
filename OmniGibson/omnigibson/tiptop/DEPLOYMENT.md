# Deploying the TiPToP bridge on a lab GPU server

Written 2026-09-02 from a working laptop install (RTX 4090 Laptop, sm_89, driver 580 = CUDA 13.0). **Nothing here has
been exercised on a Blackwell server yet**; the "Known issues" list is what an audit of the laptop install predicts
will bite on an RTX PRO 6000 (Blackwell, sm_120) or similar, with a CUDA 12.8 or 13 driver. Read this before the
first `pixi install` on a new machine. Companion docs: [README.md](README.md) (architecture, conventions, runbook),
[USAGE_DOCS.md](../../../USAGE_DOCS.md) (lab servers, sim install with uv, submodule workflow).

## Layout: three isolated environments, nothing shared

| Component | Environment | Location | Talks to |
|---|---|---|---|
| OmniGibson sim + this bridge | uv venv `b1k` from `setup_uv.sh` (python 3.11, torch 2.7.0+cu128, Isaac Sim 5.1) | repo root | planner `ws://host:8765` |
| TiPToP planner | pixi env `tiptop/.pixi` (python 3.12, conda-forge torch 2.7.1 cu129, cuda-toolkit 12.9, nvcc 12.9.86) | `tiptop/` submodule | M2T2 `http://host:8123` |
| M2T2 grasp server | pixi env `~/tiptop-services/M2T2/.pixi` (python 3.11, torch 2.4.1 cu120 as locked today) | separate clone | nothing |

Only websocket/HTTP crosses the boundaries. Never `pip install` tiptop into the sim env: numpy 2 vs 1.26, two
different cuRobo forks with the same import name, python 3.12 vs 3.11. `rerun-sdk` cannot go into the sim env either
(0.27 needs numpy 2), which is why the sim mirrors its state into the server's Rerun over the websocket instead.

## Bring-up order on a fresh server

```bash
# 0. pixi is not on PATH by default; keep host CUDA libraries out of the pixi envs
curl -fsSL https://pixi.sh/install.sh | sh; export PATH="$HOME/.pixi/bin:$PATH"
unset LD_LIBRARY_PATH        # also after `module load cuda-toolkit/13.0`; the envs ship their own CUDA 12.x runtime

# 1. simulator: follow USAGE_DOCS.md (setup_uv.sh). The bridge needs the `eval` extra (msgpack) and websockets>=15.

# 2. planner (~16 GB, 10-25 min). setup-planners clones cuRobo + cuTAMP and compiles cuRobo for the GPU it finds.
cd <repo>/tiptop && pixi install --locked && pixi run setup-planners

# 3. grasp server (~13 GB). On Blackwell read issue 2 BEFORE `pixi install`.
git clone https://github.com/williamshen-nz/M2T2.git ~/tiptop-services/M2T2 && cd ~/tiptop-services/M2T2
pixi install && pixi run setup && pixi run download-weights

# 4. services (launchers in this directory; TIPTOP_HOST=0.0.0.0 to serve other machines)
<repo>/OmniGibson/omnigibson/tiptop/scripts/start_m2t2.sh
<repo>/OmniGibson/omnigibson/tiptop/scripts/start_tiptop_server.sh      # hosts the Rerun viewer too (item 9)

# 5. smoke test from the sim env (headless). Expect: scene ready ~30 s, plan 5-8 s, success check on(mug, bowl) true.
cd <repo>/OmniGibson && OMNIGIBSON_HEADLESS=1 python -m omnigibson.tiptop.run live --host localhost --port 8765 \
    --out-dir runs/smoke --grasping-mode sticky --no-video
```

## Verify the GPU builds before trusting a run

```bash
cd <repo>/tiptop
pixi run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
#   Blackwell needs sm_100 or sm_120 in that list (the locked torch has both)
pixi run cuobjdump --list-elf curobo/src/curobo/curobolib/geom_cu.cpython-312-x86_64-linux-gnu.so
#   must list the server's own sm_XX (sm_120 for RTX PRO 6000); sm_89-only means the laptop build was copied
cd ~/tiptop-services/M2T2
pixi run python -c "import torch, pointnet2_ops._ext as e; print(torch.__version__, torch.cuda.get_arch_list(), e.__file__)"
curl -s localhost:8123/health; curl -s localhost:8765/health   # planner answers only after cuRobo warm-up (~10 s)
```

## Known issues, in the order you will hit them

1. **cuRobo is compiled for one GPU and tracks an unpinned branch.** `tiptop/curobo/` is a gitignored clone that
   `install/install-curobo.sh` fast-forwards to `origin/main` of `williamshen-nz/curobo` on every run, then
   `pip install -e .` compiles five CUDA extensions for the GPU it detects (no `TORCH_CUDA_ARCH_LIST`, no PTX). The
   fork's build-skip fingerprint covers torch/CUDA versions but not the GPU arch, so a copied `tiptop/curobo/` with
   `.so` files from another machine is silently kept and fails at runtime with "no kernel image is available".
   Always build from a fresh clone on the target. Known-good commit with torch 2.7.1/cu129: `b5fad1d` (2026-03-21);
   if a newer main breaks, pin it in `install/install-curobo.sh` (replace the fetch/checkout/pull block with
   `git checkout b5fad1d`). Optional: `TORCH_CUDA_ARCH_LIST="12.0" pixi run install-curobo` to force the arch.
2. **M2T2's committed lock cannot target Blackwell.** It resolves torch 2.4.1 (conda-forge cuda120 build, arch list
   ends at sm_90) and nvcc 12.1 (`--list-gpu-arch` ends at compute_90); its PointNet++ op `pointnet2_ops._ext`
   is compiled by `pixi run setup` for the local GPU only. Fix (untested): in `~/tiptop-services/M2T2/pixi.toml` set
   `[system-requirements] cuda = "12.8"`, `cuda-toolkit = "12.8.*"` (or `12.9.*`), `pytorch-gpu = ">=2.7"`,
   keep `python = "3.11.*"` and `numpy = "<2"`; then `pixi install` (re-locks), `pixi run setup`,
   `pixi run download-weights`, and check the arch list. `m2t2_server.py` calls `torch.load` without
   `weights_only`; the checkpoint loads fine with the torch>=2.6 default. Weights: git-lfs clone of
   `huggingface.co/wentao-yuan/m2t2`, md5 checked by the task.
3. **Driver, toolkit, glibc.** Blackwell needs driver >= 570 (CUDA 12.8). The tiptop env's CUDA 12.9 runtime runs on
   a 12.8 driver through CUDA minor-version compatibility; pixi only requires the `__cuda` virtual package >= 12
   (it reports the driver's version, e.g. `__cuda=13.0` on the laptop). Host toolkits are never used. glibc must
   be >= 2.31 (the open3d wheel). If `pixi install` says the platform does not satisfy `cuda`, the driver is missing
   or too old, not the toolkit.
4. **pixi quirks.** pixi 0.78 warns that `[system-requirements]` is deprecated (harmless). `pixi lock --check` is
   NOT read-only in 0.78: it rewrites `pixi.lock` to format v7. Use `pixi install --locked`, and
   `git -C tiptop checkout -- pixi.lock` if the lock got rewritten; do not commit a v7 lock by accident.
5. **`python -m tiptop...` from inside `tiptop/` breaks cuTAMP's import** (the gitignored `tiptop/cutamp/` clone
   shadows the installed package: "required 0.0.6, found <0.0.2"). Use the console scripts, `pixi run tiptop-server`
   / `pixi run tiptop-h5`, as the launchers do.
6. **Network and credentials.** github.com: the private submodule `WenzhouDing/tiptop` (deploy key or token; the
   laptop uses an ssh `insteadOf` rewrite), `williamshen-nz/curobo`, `tiptop-robot/cuTAMP` (tag v0.0.6),
   `facebookresearch/segment-anything-2` at the locked commit; conda-forge and PyPI; huggingface.co via git-lfs
   (M2T2 weights, ~230 MB); dl.fbaipublicfiles.com (SAM-2 checkpoint, 900 MB, fetched into `tiptop/tiptop/.cache/`
   on first use, only needed without ground-truth masks); huggingface.co for the Grounding DINO weights
   (`IDEA-Research/grounding-dino-base`, ~900 MB into `~/.cache/huggingface`) that the server fetches and loads at
   start-up when `perception.detector: grounding_dino` (both sim configs; `transformers` is in `pixi.lock`).
   `GOOGLE_API_KEY` is only needed with `perception.detector: gemini`.
7. **cuTAMP patch.** `tiptop/install/install-cutamp.sh` applies `tiptop/install/patches/cutamp-*.patch` on top of
   the pinned cuTAMP release (a `get_world_cfg` list-aliasing fix that otherwise crashes every plan skeleton tried
   after the first motion-planning attempt with `KeyError: 'table'`). A cuTAMP checkout made without the script
   needs `git apply` of the same patch; upstreaming it to tiptop-robot/cuTAMP is the real fix.
8. **The first request is slow.** warp JIT-compiles the cuRobo/cuTAMP kernels per GPU into `~/.cache/warp` on the
   first plan; cuRobo warms up MotionGen at server start (`/health` is 200 only afterwards).
9. **Rerun.** The planner hosts the viewer itself (`--rerun-mode serve`, the launcher's default): gRPC 9876 and the
   web viewer 9090, both on 127.0.0.1, one recording for the planner's lifetime, viewer killed with the planner.
   The laptop needs only a browser and `ssh -N -L 9090:127.0.0.1:9090 -L 9876:127.0.0.1:9876 <server>`; see
   README.md "Rerun" for the URL and what the view holds. A taken 9876/9090 is a startup error, not a fallback
   (`pkill -u $USER -f 'rerun --serve-web'`, or `--rerun-grpc-port` / `--rerun-web-port`). `--rerun-mode stream`
   spawns a native window and is laptop-only; `save` writes one `tiptop.rrd` per request under
   `tiptop/tiptop_server_outputs/<timestamp>/`, replayable with `pixi run rerun --serve-web --bind 127.0.0.1 <file>`.
   The simulator's mirror (joints, its own object meshes, camera images) works in every mode. Viewer and SDK must
   be the *same* version (0.27.3 here); a `rerun` of another version on `PATH` (`~/.local/bin/rerun` was 0.30.2
   on the laptop) is why the planner runs the SDK's own binary and never a `rerun` from `PATH`.
10. **Ports.** Launchers bind 127.0.0.1. Same machine: nothing to do. Sim on the laptop, planner on the server:
   `ssh -N -L 8765:127.0.0.1:8765 server` and `--host localhost`, or `TIPTOP_HOST=0.0.0.0`. The client refuses to
   run unless the server metadata says `robot_type: panda`, `dof: 7` (the launcher's `--config` guarantees it).
11. **Sharing one GPU.** Laptop numbers: Isaac Sim 4-5 GB (GUI adds 1-2), planner 4 GB idle and 6.3 GB peak at
    128 particles, M2T2 1.3 GB. On a 96 GB card raise `TIPTOP_PARTICLES=256` and `TIPTOP_MAX_PLANNING_TIME=60`.
12. **Planner variance.** The same observation can fail once with "Motion planning failed for 32/59 satisfying
    particle(s)" and succeed on the next request (M2T2 grasp sampling differs per call). Retry before debugging.
13. **Grasp physics.** Physical grasps of the thin YCB mug slip; demos and smoke tests use `--grasping-mode sticky`.
14. **WebRTC streaming is unreliable on RTX PRO 6000 Blackwell; use Rerun.** The client connects, a few frames
    arrive, the encoder stops producing and the client drops, repeatedly -- one session: 4 `FIRST_FRAME_SENT`
    against 115 `VideoEncoder: Could not get encoded frame` and 24 `CLIENT_DISCONNECT_UNINTENDED`. The bundled
    StreamSDK 7.6.3 predates the card (device `0x2BB5`). `UseRefactoredVideoEncoder=1`, a StreamSDK regkey read
    from the environment, appeared to fix it on one short connection and does NOT hold up: a longer session with
    it set still threw 115 errors. Two dead ends: the `GPU ... is not white-listed` warning gates nothing
    (disassembly shows that branch falling through to the same success path), and NVENC is healthy on the card
    (h264/hevc/av1 encode at 200+ fps outside Isaac Sim). No drop-in fix exists for Isaac Sim 5.1 (checked
    2026-09-05): the Kit 107 registry's newer `omni.kit.streamsdk.plugins` 7.7.2 ships a byte-identical
    `libNvStreamServer.so` (StreamSDK 04.72, sm_52 kernels only), the official 5.1.0 container carries the same
    7.6.3, and the StreamSDK generations that work on this card (04.84/04.86) come only with
    `omni.kit.livestream.webrtc` 10.x for Kit 110 / Isaac Sim 6.0 (python 3.12). Use Rerun (item 9), which carries
    the robot's head camera and a third-person view of the workspace from the simulator, plus the per-round
    `live.mp4`; none of them needs an encoder.
15. **Remote sim streaming.** With `OMNIGIBSON_REMOTE_STREAMING=webrtc` the Kit app is launched windowless, but
    `gm.HEADLESS` stays false, so the viewport-camera code in `scene.py` still runs — that is what aims the streamed
    view, and it is why you must NOT also set `OMNIGIBSON_HEADLESS=1`. Auxiliary sensor cameras are kept out of the
    streamed frame (`vision_sensor.py` checks `REMOTE_STREAMING`); everything else is identical.
16. **CI on `dev/tiptop`.** `tests.yml`/`profiling.yml` check out submodules; the private tiptop repo needs a token
    there (see USAGE_DOCS.md).
17. **Launcher paths.** `scripts/start_tiptop_server.sh` defaults to this repo's `tiptop/`, `scripts/start_m2t2.sh`
    to `~/tiptop-services/M2T2`; override with `TIPTOP_DIR` / `M2T2_DIR`. The laptop also has copies in
    `~/tiptop-services/bin/` that are not in git.
18. **System RAM, not just VRAM.** A whole-task run in a BEHAVIOR house scene was OOM-killed on the 30 GB laptop
    (2 GB swap) after 16 transfers: Isaac client 14 GB RSS, planner 4.4 GB, Rerun viewer 2.8 GB, M2T2 1.1 GB, plus
    the desktop. The kernel killed the client, which shared VSCode's cgroup, and took the editor session with it.
    Mitigations: the planner caps its viewer at `--rerun-memory-limit 2GB` (Rerun's own default is 75% of RAM; the
    oldest non-static data is dropped past it, and `--no-state-stream` on the client keeps per-step state and camera
    images out of it entirely); run long clients in their own
    cgroup with a cap and a high OOM score, e.g. `systemd-run --user --scope -p MemoryMax=18G choom -n 800 -- python
    -m omnigibson.tiptop.run task ...`; give the box real swap (16 GB) before a long run; watch `rss_gb` in the
    per-transfer log lines for growth.

## R1Pro specifics

- Planner config: `TIPTOP_CONFIG=tiptop/config/tiptop_sim_r1pro.yml`; use `TIPTOP_PARTICLES=256 TIPTOP_MAX_PLANNING_TIME=40`
  (128 particles left too few feasible pick+place samples for the 11-joint chain).
- The embodiment assets under `tiptop/tiptop/embodiments/assets/r1pro/` are committed; only the Rerun meshes are not:
  run `pixi run python scripts/make_r1pro_embodiment.py --copy-meshes` in `tiptop/` on a machine that has the
  OmniGibson robot assets (any machine with the sim datasets), or accept a mesh-less robot in Rerun.
- Regenerating the embodiment needs the OmniGibson robot assets; re-validate with `scripts/check_r1pro_embodiment.py`
  against a fresh simulator probe (scratch script `probe_r1pro.py` in the session notes; prints joint order, eef and
  camera poses at sampled configurations).
- VRAM on the laptop during an Rs_int episode: about 9.5 GB total with both services idle (Isaac + scene ~5 GB).
- Never attach `seg_instance` to a robot-mounted camera in this Isaac build (segfault after ~35 steps); the bridge
  captures through an external shadow camera. Keep the robot camera rgb-only.

## Files

- Bridge (this directory): `protocol.py`, `client.py` (planning client + the Rerun mirror), `scene.py`, `executor.py`,
  `run.py`, `r1pro.py`, `scripts/` (service launchers, `stream_scene.py`, `probe_r1pro.py`), tests in
  `OmniGibson/tests/test_tiptop_protocol.py`.
- Planner side (`tiptop/` submodule): `tiptop/tiptop_websocket_server.py`, `tiptop/config/tiptop_sim_{panda,r1pro}.yml`,
  `tiptop/embodiments/` (R1Pro), `scripts/make_r1pro_embodiment.py`, `install/install-curobo.sh`,
  `install/install-cutamp.sh`, `docs/simulation.md`, `pixi.toml` + `pixi.lock`.
