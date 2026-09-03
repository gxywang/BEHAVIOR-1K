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
numpy arrays), the server answers a JSON plan `{q_init, steps: [trajectory{positions (N, dof), dt} | gripper{open|close}]}`,
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

Deploying the services on a lab GPU server (Blackwell, CUDA 12.8/13 drivers) and the known issues you will hit:
[DEPLOYMENT.md](DEPLOYMENT.md). Launchers for both services are in [scripts/](scripts/).

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
During execution the client also streams the simulator's state back to the server (`sim_state` messages on the same
websocket), so the planner's robot model in Rerun follows the simulated joints in real time and each object gets a
green copy that follows its simulated pose next to the grey perceived one; `--no-state-stream` turns this off, and
`replay --state-stream host:port` mirrors an offline replay into a running server's view. In the viewer, `log_time` is
wall-clock and `sim_time` is simulated seconds since the capture (0 = the planner's input, robot at the capture
posture). If the 2D views are black and the robot is a heap at the origin, the time cursor is before the data: the
viewer keeps the time panel state (paused, loop selection, speed) across runs, so press the go-to-end button or
choose *Following*, and check the selected recording. Start a viewer yourself only with the tiptop env's `rerun`
(`pixi run rerun`); a stray `rerun` of another version on `PATH` that already listens on port 9876 gets reused.

Options: `--objects mug,bowl,apple,banana`, `--task`, `--goal "on(mug,bowl)"` (drives the ground-truth atoms and the
success check), `--grasping-mode physical|assisted|sticky`, `--no-video`, `--no-gt` (live only: send just the object
names and goal atoms, the server runs its detector + SAM2 on the image; see "Ground truth vs. detector").

Outputs per run: `obs.h5`, `capture.json` (poses, intrinsics, validation), `*_result.json` (tracking errors, gripper
events, success), `*.mp4` from the external camera.

## R1Pro in a BEHAVIOR scene

`--embodiment r1pro` runs the challenge robot inside a BEHAVIOR scene with its own head camera. Navigation is out of
scope: `--near <furniture>` parks the robot flush with a piece of furniture on a free side (floor and room checked),
`--spawn preset:furniture:dx,dy` drops preset objects onto it, `--scene-objects` adds existing scene objects.

```bash
TIPTOP_CONFIG=tiptop/config/tiptop_sim_r1pro.yml TIPTOP_PARTICLES=256 TIPTOP_MAX_PLANNING_TIME=40 \
    OmniGibson/omnigibson/tiptop/scripts/start_tiptop_server.sh                       # planner for the R1Pro
OMNIGIBSON_HEADLESS=1 python -m omnigibson.tiptop.run live --embodiment r1pro --scene-model Rs_int \
    --not-load ceilings,straight_chair --near breakfast_table_skczfi_0 --side=-x --standoff 0.0 \
    --spawn mug:breakfast_table_skczfi_0:-0.17,0.35 --spawn bowl:breakfast_table_skczfi_0:-0.20,0.15 \
    --task "put the mug in the bowl" --goal "on(mug,bowl)" --grasping-mode sticky --out-dir runs/r1pro
```

What the R1Pro path does differently (all in `r1pro.py`):

- **Planner model.** TiPToP's `r1pro_left` embodiment (tiptop submodule, `tiptop/embodiments/`) is generated from the
  same URDF and cuRobo collision spheres OmniGibson ships for the robot and validated against the simulator (URDF FK
  within 0.03 deg / 0.0 mm over 25 random full-body configurations). It plans torso (4) + left arm (7); the right
  arm and both grippers are locked. The server advertises joint names, locked values and the home pose in its
  metadata and the simulator applies them before capturing, so both sides agree by construction.
- **Frames.** World frame = `base_link` (`robot.get_position_orientation()`, floor level). Joint vectors are always
  gathered by joint name: OmniGibson interleaves the two arms in its joint order.
- **Cameras.** The head camera (`zed_link`, 720x720, aperture 40 mm as in the challenge) or the left wrist camera.
  Instance segmentation attached to a robot-mounted camera leaks GPU memory and segfaults Isaac's synthetic-data graph
  after ~35 steps, so the robot camera renders rgb only and an external "shadow" VisionSensor with identical
  intrinsics is moved onto its pose for the capture frame. The robot's own pixels are removed from the depth.
- **Body.** The base gets the evaluator's 250 kg mass; without it the leaning challenge torso posture tips the robot
  over. The planned joint vector (torso + arm) drives the trunk and arm_left controllers, everything else holds.
- **Reach.** With the torso locked at the challenge posture the arm only reaches 0.4-0.6 m ahead on its left, which
  is why the torso is planned; stand with the target front-left and within ~0.6 m of the base.

First verified episode (2026-09-02, Rs_int, `breakfast_table_skczfi_0`, mug into bowl, sticky grasping):

| Stage | Result |
|---|---|
| Model validation | URDF FK vs simulator 0.0 mm / 0.03 deg over 25 configs; cuRobo FK == URDF FK |
| Capture | head camera 720x720, robot pixels masked, object centroid check passed |
| Planning | 25 of 256 particles feasible, cuTAMP 1.3 s, 4.9 s round trip, 7 trajectories / 748 waypoints |
| Execution | tracking lag <= 0.005 rad on torso + arm, grasp detected, OnTop(bowl) true |

Outputs: `runs/r1pro_10/{live.mp4, live_result.json, obs.h5, capture.json}` and the server's
`tiptop_server_outputs/<timestamp>/` (with `tiptop.rrd` in `--rerun-mode save`).

Tooling: `scripts/probe_r1pro.py` (runs Isaac Sim) writes the joint order and link/camera poses at sampled
configurations that `tiptop/scripts/check_r1pro_embodiment.py` validates the planner model against; `replay --plan`
reads the embodiment provenance that `live` stores in `tiptop_plan.json`, and `--scene capture.json` reuses settled
object poses for both embodiments.

## Challenge tasks (`--activity`)

`--activity <name>` loads a 2026 BEHAVIOR Challenge task instead of spawning objects: the evaluator's scene and room
list (from `datasets/2026-challenge-task-instances/metadata/`), the pre-sampled task instance through OmniGibson's
`BehaviorTask` (`--activity-instance`, 0 = the template), the task objects tracked under per-instance names
(`candle_1` for `candle.n.01_1`), and the goal scored the way the challenge does (`TaskMetric`: 1 on full success,
else the newly satisfied fraction of the best goal option; `forpairs` goals ground into thousands of options, so
every grounded predicate is evaluated once and memoized). `--goal` uses BDDL names and predicates
(`inside(candle.n.01_2,wicker_basket.n.01_2)`; `inside`/`ontop` become TiPToP's `on`), `--no-gt` sends only the
categories (`candle`, `wicker basket`) so the detector finds every instance and the goal takes the largest one.
Instance segmentation crashes Isaac on the first render in these scenes, so captures render rgb + depth only
(also all the challenge allows) and ground-truth masks come from geometry instead of the annotator
([gt_masks.py](gt_masks.py): the depth pixels within 8 mm of an object's mesh at its current pose; `--seg-instance`
opts back into the annotator where it works, e.g. Rs_int).

`task` runs the whole `inside(item, container)` goal: per container, stage it on `--stage-support` at the free spot
near the item that a base pose can reach together with it, stand there, capture, plan, execute, score with the task's
own predicate; a type is skipped after failing for two containers in a row, and any unplaced item of the type that
ends up inside counts. Each transfer's directory holds the capture, plan, video and result; `task_summary.json` has
the per-transfer records (with the client's RSS, see DEPLOYMENT.md item 17) and the final goal status.

```bash
python -m omnigibson.tiptop.run task --embodiment r1pro --activity assembling_gift_baskets \
    --stage-support table.n.02_1 --task "put the item in the wicker basket" --grasping-mode sticky \
    --attempts-per-item 2 --host localhost --port 8765 --out-dir runs/gift_task
```

What replaces navigation for now: `--stand-for ITEM,TARGET` searches a base pose from which both objects are ahead,
on the left and within reach of the left arm; `--place OBJ:SUPPORT:DX,DY` teleports an object (used to bring a
basket from the floor onto the coffee table, standing in for a carry); `--sequential` runs one
capture/plan/execute round per goal atom, re-standing before each. The rules forbid teleporting during evaluation,
so these are test scaffolding until the base moves under its own controller.

Target task: `assembling_gift_baskets` (task 26, house_double_floor_lower living room, 4 wicker baskets on the
floor, 4 candles + 4 butter cookies + 4 Swiss cheeses + 4 bows on a 0.41 m coffee table, goal one of each inside
every basket; the longest task at 869 s mean human time, no 2025 submission ever completed it).

```bash
python -m omnigibson.tiptop.run live --embodiment r1pro --activity assembling_gift_baskets \
    --place wicker_basket.n.01_2:table.n.02_1:0.28,0.16 --sequential \
    --goal "inside(butter_cookie.n.01_1,wicker_basket.n.01_2);inside(candle.n.01_2,wicker_basket.n.01_2)" \
    --task "put the item in the wicker basket" --grasping-mode sticky --no-gt \
    --host localhost --port 8765 --out-dir runs/gift_5
```

Result of that command (2026-09-03, headless, detector + SAM2, no ground truth): both transfers succeeded, the
task's partial-credit score went 0 -> 0.0625 -> 0.125 (2 of the 16 `inside` predicates; the best 2025 submission
reached 0.31 on this task, none completed it). Per round: stand 1 s, capture 10 s (the renderer needs ~25 frames
to settle after the base moves), detect + segment 2 s, cuTAMP 1-2 s, execute 25 s, goal scoring 7 s.

| Item | Detector | M2T2 grasps | Outcome |
|---|---|---|---|
| butter cookie (8 cm, 2 cm thick) | 0.68 with the "round cookie" prompt | tens | grasped (fingers 4.7 / 3.1 cm), inside the basket |
| pillar candle (10 x 11 cm) | 0.4-0.5 | hundreds | grasped, inside the basket |
| Swiss cheese (12 x 10 x 2 cm slab) | 0.46-0.52 | none to few | plan found once, gripper closed on nothing |
| bow (8 cm ribbon) | 0.4-0.56 with "gift bow" | few | not attempted |

What stands between this and the full task: items farther than ~0.9 m from the basket need the base to move while
holding (TiPToP plans pick and place from one base pose; the basket must be re-placed along the table or carried),
the four baskets sit on the floor (a floor-level pick or a carry that the pipeline does not model), and the flat
cheese and bows get almost no grasps from M2T2. Fixed on the way: a cuTAMP list-aliasing bug that appended the
surfaces to the movables once motion planning ran (`tiptop/install/patches/`, applied by `install-cutamp.sh`),
duplicate and nested detector boxes, the capture camera's temporal ghosting after teleports.

## Ground truth vs. detector

Three perception modes, chosen by what the request carries:

- **Ground truth** (default): `gt_labels` / `gt_masks` and `gt_atoms` from `--goal` (or the task). Masks come from
  geometry, [gt_masks.py](gt_masks.py) (rendered depth against the objects' meshes; disjoint, IoU 0.90-0.96 against
  the annotator on the Panda demo, the rest a 1-4 px table halo), or from OmniGibson's instance segmentation with
  `--seg-instance`. Objects out of view are dropped from the request; a goal object out of view is an error. The
  server skips detection and SAM2 and runs everything else unchanged (M2T2 grasps, table RANSAC, convex hulls,
  cuTAMP, cuRobo). Fast and exact; for development only.
- **Competition style** (`--no-gt`): only `gt_labels` and `gt_atoms`, i.e. the object names and the goal an agent
  knows from the task definition, no masks. The server's detector (`perception.detector`, set to `grounding_dino`
  in both sim configs: Grounding DINO base from Hugging Face, loaded together with SAM2 when the server starts) finds
  one box per name in the head-camera RGB and SAM2 segments it. No `GOOGLE_API_KEY`. The capture also carries
  `robot_mask`, the robot's own pixels (a self-filter; here from the simulator's segmentation, so only with
  `--seg-instance`; on a real robot rendered from the kinematics): SAM2 gets them as negative prompts and they are removed from every mask, otherwise
  a gripper in front of an object becomes "the mug" (mask IoU 0.33 without, 0.95 with; bowl 0.96). 0.3 s per frame
  once warm, 2.7 GB VRAM for both models.
- **Gemini** (`perception.detector: gemini`, the upstream default): Gemini detects the objects and translates the
  task; needs `GOOGLE_API_KEY`; atoms sent with the request take precedence over its translation.

The R1Pro looks before it reaches: the capture is taken with the left shoulder abducted (`LOOK_ARM` in
[r1pro.py](r1pro.py), arm out of the head camera's view; mug 3881 px instead of 2005 behind the gripper), then
the arm returns to the ready posture and the request's `q_init` is that posture, so the plan starts where the
robot is. `--no-look` captures in the ready posture instead.

Still privileged with `--no-gt`: the self-filter comes from the simulator's segmentation, and the names are scene
object names (`mug`, `bowl`) that a task layer would map from the BDDL categories.

## Tests

`pytest OmniGibson/tests/test_tiptop_protocol.py` covers the msgpack-numpy wire format, request validation, plan
parsing/resampling and the H5 layout without Isaac Sim.
