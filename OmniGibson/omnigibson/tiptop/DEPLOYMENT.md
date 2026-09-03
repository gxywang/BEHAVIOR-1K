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
TIPTOP_RERUN_MODE=save <repo>/OmniGibson/omnigibson/tiptop/scripts/start_tiptop_server.sh

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
   on first use, only needed without ground-truth masks). `GOOGLE_API_KEY` is needed only for `--no-gt`.
7. **The first request is slow.** warp JIT-compiles the cuRobo/cuTAMP kernels per GPU into `~/.cache/warp` on the
   first plan; cuRobo warms up MotionGen at server start (`/health` is 200 only afterwards).
8. **Rerun on a headless box.** `--rerun-mode stream` spawns a viewer window and needs a display. Use `save`
   (one `tiptop.rrd` per request under `tiptop/tiptop_server_outputs/<timestamp>/`, open later with `rerun <file>`)
   or `connect` with `--rerun-url rerun+http://127.0.0.1:9876/proxy` plus a reverse tunnel to a viewer on your
   laptop (`ssh -R 9876:127.0.0.1:9876 server`, `rerun` running locally). The client's sim-state mirror works in
   all modes.
9. **Ports.** Launchers bind 127.0.0.1. Same machine: nothing to do. Sim on the laptop, planner on the server:
   `ssh -N -L 8765:127.0.0.1:8765 server` and `--host localhost`, or `TIPTOP_HOST=0.0.0.0`. The client refuses to
   run unless the server metadata says `robot_type: panda`, `dof: 7` (the launcher's `--config` guarantees it).
10. **Sharing one GPU.** Laptop numbers: Isaac Sim 4-5 GB (GUI adds 1-2), planner 4 GB idle and 6.3 GB peak at
    128 particles, M2T2 1.3 GB. On a 96 GB card raise `TIPTOP_PARTICLES=256` and `TIPTOP_MAX_PLANNING_TIME=60`.
11. **Planner variance.** The same observation can fail once with "Motion planning failed for 32/59 satisfying
    particle(s)" and succeed on the next request (M2T2 grasp sampling differs per call). Retry before debugging.
12. **Grasp physics.** Physical grasps of the thin YCB mug slip; demos and smoke tests use `--grasping-mode sticky`.
13. **Remote sim streaming.** With `OMNIGIBSON_REMOTE_STREAMING=webrtc` the sim is forced headless; the GUI
    viewport code in `scene.py` is skipped, everything else is identical.
14. **CI on `dev/tiptop`.** `tests.yml`/`profiling.yml` check out submodules; the private tiptop repo needs a token
    there (see USAGE_DOCS.md).
15. **Launcher paths.** `scripts/start_tiptop_server.sh` defaults to this repo's `tiptop/`, `scripts/start_m2t2.sh`
    to `~/tiptop-services/M2T2`; override with `TIPTOP_DIR` / `M2T2_DIR`. The laptop also has copies in
    `~/tiptop-services/bin/` that are not in git.

## Files

- Bridge (this directory): `protocol.py`, `client.py`, `scene.py`, `executor.py`, `run.py`, `scripts/`,
  tests in `OmniGibson/tests/test_tiptop_protocol.py`.
- Planner side (`tiptop/` submodule): `tiptop/tiptop_websocket_server.py`, `tiptop/config/tiptop_sim_panda.yml`,
  `install/install-curobo.sh`, `install/install-cutamp.sh`, `docs/simulation.md`, `pixi.toml` + `pixi.lock`.
