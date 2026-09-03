# TiPToP ↔ OmniGibson bridge

This package connects [TiPToP](https://github.com/tiptop-robot/tiptop) (task-and-motion planning for
pick-and-place from one RGB-D image and a language instruction) to the BEHAVIOR-1K simulator. It is the
OmniGibson counterpart of TiPToP's own IsaacLab client (`droid-sim-evals`) and speaks TiPToP's existing
contracts only; **nothing here imports `tiptop`, and TiPToP never imports OmniGibson.**

## Architecture

```
 laptop / any machine (conda env `behavior`)          GPU server or laptop (tiptop pixi env)         (M2T2 pixi env)
 ┌──────────────────────────────────────────┐        ┌──────────────────────────────────────┐        ┌────────────────┐
 │ OmniGibson / Isaac Sim                   │  ws    │ tiptop-server  :8765                 │  http  │ M2T2 :8123     │
 │  omnigibson.tiptop.scene   (Franka Panda)│ ─────▶ │  perception: Gemini+SAM2 or GT masks │ ─────▶ │ grasp proposals│
 │  omnigibson.tiptop.client  (1 request)   │ ◀───── │  cuTAMP + cuRobo  →  joint-space plan│        └────────────────┘
 │  omnigibson.tiptop.executor(open loop)   │  JSON  └──────────────────────────────────────┘
 └──────────────────────────────────────────┘        offline variant: obs.h5 ──▶ tiptop-h5 ──▶ tiptop_plan.json ──▶ replay
```

One request per episode: the simulator sends `{rgb, depth, intrinsics, world_from_cam, task, q_init}` (msgpack with
numpy arrays), the server answers a JSON plan `{q_init, steps: [trajectory{positions (N,7), dt} | gripper{open|close}]}`,
and the simulator executes it open loop. The same data can go through files (`obs.h5` → `tiptop-h5` → `tiptop_plan.json`).

### Where development goes

| Concern | Repo | Why |
|---|---|---|
| Planner, perception, embodiments (cuTAMP/cuRobo configs, `tool_from_ee`, gripper spheres), server protocol | **tiptop** (git submodule `tiptop/`, your private repo) | manipulation logic; runs in its own pixi env, deployable to a server unchanged |
| Scenes, tasks, robot/controller configs, observation capture, plan execution, success metrics, videos | **BEHAVIOR-1K** (`OmniGibson/omnigibson/tiptop/`) | needs Isaac Sim; sim-only |
| The wire contract | owned by tiptop; the submodule pin records which server version this client was validated against | bump the pin when the protocol changes |

Sim-only shortcuts (e.g. ground-truth masks) are implemented **inside tiptop behind optional request fields**, so the
real-robot path stays untouched and the simulator never needs tiptop's dependencies.

### Environment policy

Three isolated environments, by design (they cannot be merged: Python 3.12 vs 3.11, numpy 2 vs `<2`, two different
`curobo` forks with the same import name, torch 2.7.1/cu129 vs Isaac Sim's 2.7.0/cu128):

| Env | Where | Contents |
|---|---|---|
| `behavior` (conda) | laptop | OmniGibson + Isaac Sim + this client (needs only `websockets`, `msgpack`, `h5py`, `numpy`, already present) |
| tiptop (pixi, `tiptop/.pixi`) | laptop or GPU server | tiptop, cuTAMP 0.0.6, cuRobo fork, SAM-2 (`tiptop/tiptop/.cache/sam2.1_hiera_large.pt`) |
| M2T2 (pixi) | `~/tiptop-services/M2T2` | grasp server + weights |

Never `pip install -e tiptop` into `behavior`.

## Conventions that matter

- **World frame = robot base frame** (cuRobo `base_link` = `panda_link0`). The camera pose is re-expressed in the base
  frame with `T.relative_pose_transform`, so the robot may sit anywhere in the OmniGibson world (here on a 0.75 m table).
- **Camera axes**: OmniGibson/USD cameras look down −z with +y up; TiPToP expects OpenCV (+z forward, +y down).
  Conversion: `q_cv = quat_multiply(q_usd, [1, 0, 0, 0])` (180° about the camera x axis).
- **Depth**: `depth_linear` (distance to image plane) in metres, invalid pixels set to 0. Not `depth` (ray length).
- **Embodiment**: OmniGibson `franka` with the default Franka hand ↔ TiPToP/cuTAMP `panda`. The kinematics, joint order
  (`panda_joint1..7`), limits and tool frame (OmniGibson `eef_link` = `panda_hand` + 0.105 m) match exactly, so no new
  cuTAMP embodiment is needed. `tiptop/config/tiptop_sim_panda.yml` selects it.
- **Controllers**: absolute joint targets (`JointController`, position mode, `use_delta_commands: false`,
  `action_normalize: false`, limits `null`) and a binary gripper (`+1` open, `−1` close; the DROID client uses the
  opposite polarity). Plans are resampled from their `dt` (0.02 s) to the env step (1/30 s); gripper events hold the arm
  for `--gripper-hold-steps`.
- **M2T2 crop box** (`apply_bounds`): x ∈ [0, 1], |y| ≤ 0.3, z ∈ [−0.2, 0.5] m in the base frame. Keep objects there.
  The robot base column and the hand at `DROID_Q_INIT` are inside that box and in view too; this is harmless because
  grasps are only kept within 1 cm of object-mask points and the planner's world is built from the table plane and the
  object meshes, but it does spend some of M2T2's point budget on robot geometry.
- `capture` writes a numeric self-check (`validation` in `capture.json`): table top at base z≈0, object mask
  centroids within a few cm of their true positions, camera looking down. Wrong conventions fail loudly here.

## Running

Services (any machine with a GPU; `~/.pixi/bin` on PATH, `LD_LIBRARY_PATH` unset):

```bash
cd ~/tiptop-services/M2T2 && pixi run server --host 127.0.0.1 --port 8123          # grasp server
cd ~/Desktop/BEHAVIOR-1K/tiptop && pixi run tiptop-server --config tiptop/config/tiptop_sim_panda.yml \
    --num-particles 128 --max-planning-time 30 --rerun-mode disabled                 # planner, ws://…:8765
# remote server: ssh -N -L 8765:127.0.0.1:8765 user@shenlong-gpu-01, then --host localhost below
```

Simulator (conda env `behavior`; set `OMNIGIBSON_HEADLESS=1` or unset it for the GUI):

```bash
python -m omnigibson.tiptop.run capture --out-dir runs/scene1                       # obs.h5, capture.json, rgb/depth/masks png
cd tiptop && pixi run tiptop-h5 --config tiptop/config/tiptop_sim_panda.yml \
    --h5-path runs/scene1/obs.h5 --task-instruction "put the mug in the bowl" --no-rr-spawn   # offline planning
python -m omnigibson.tiptop.run replay --plan <run>/tiptop_plan.json --scene runs/scene1/capture.json --out-dir runs/replay
python -m omnigibson.tiptop.run live --host localhost --port 8765 --out-dir runs/live   # end-to-end over the websocket
```

Demo with both visualizations on one machine: start the planner with `--rerun-mode stream`
(`TIPTOP_RERUN_MODE=stream ~/tiptop-services/bin/start_tiptop_server.sh`), leave `OMNIGIBSON_HEADLESS` unset and run
`live`. The Rerun viewer pops up on the first request (RGB, masks, grasps, robot, plan) and is re-spawned if you close
it; the Isaac Sim window shows the executed trajectory. `--rerun-mode save` writes `tiptop.rrd` into each server run
directory instead (`rerun <file>` to view later); `--rerun-mode connect --rerun-url rerun+http://host:9876/proxy`
streams to a viewer you started yourself (`rerun` in the tiptop env), e.g. when the planner runs on another machine.

Options: `--objects mug,bowl,apple,banana`, `--task`, `--goal "on(mug,bowl)"` (drives the ground-truth atoms and the
success check), `--grasping-mode physical|assisted|sticky`, `--no-video`, `--no-gt` (live only: use Gemini + SAM2,
needs `GOOGLE_API_KEY` on the server).

Outputs per run: `obs.h5`, `capture.json` (poses, intrinsics, validation), `*_result.json` (tracking errors, gripper
events, success), `*.mp4` from the external camera.

## Ground truth vs. Gemini

Without a `GOOGLE_API_KEY`, TiPToP cannot detect objects or parse the instruction (Gemini does both). The simulator
therefore sends `gt_labels` / `gt_masks` (from OmniGibson instance segmentation) and `gt_atoms` (from `--goal`);
the tiptop fork skips Gemini and SAM2 when these keys are present and runs everything else unchanged (M2T2 grasps,
table RANSAC, convex hulls, cuTAMP, cuRobo). Drop the keys (`--no-gt`) to exercise the full open-vocabulary pipeline.

## Tests

`pytest OmniGibson/tests/test_tiptop_protocol.py` covers the msgpack-numpy wire format, request validation, plan
parsing/resampling and the H5 layout without Isaac Sim.
