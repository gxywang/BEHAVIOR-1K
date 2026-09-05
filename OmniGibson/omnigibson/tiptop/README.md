# TiPToP ↔ OmniGibson bridge

This package runs [TiPToP](https://github.com/tiptop-robot/tiptop) -- task-and-motion planning for pick-and-place
from one RGB-D image and a goal -- against the BEHAVIOR-1K simulator, with the challenge robot (R1Pro) in the
challenge scenes. It is the OmniGibson counterpart of TiPToP's own IsaacLab client and speaks only TiPToP's
existing contracts: **nothing here imports `tiptop`, and TiPToP never imports OmniGibson.** They are separate
processes in separate Python environments and talk over a websocket.

The runbook for the lab server (GPU pinning, tunnels, the exact shell lines) is [USAGE_DOCS.md](../../../USAGE_DOCS.md);
installing the planner and grasp server on a new machine and the problems you will meet is [DEPLOYMENT.md](DEPLOYMENT.md).

## Architecture

```
 simulator process (uv venv b1k: python 3.11, Isaac Sim 5.1)     planner process (tiptop/.pixi: python 3.12)        grasp server
 ┌──────────────────────────────────────────────────┐            ┌───────────────────────────────────────────┐    ┌────────────────┐
 │ omnigibson.tiptop.run   CLI: rounds, scoring      │  ws :8765  │ tiptop-server                              │    │ M2T2           │
 │ scene.py / r1pro.py     scene, capture, posture   │ ─────────▶ │  masks: GT from the request, or            │ ─▶ │ :8123 (http)   │
 │ client.TiptopClient     one request per round     │ ◀───────── │         Grounding DINO + SAM2 on the image │    │ grasp proposals│
 │ executor.PlanExecutor   open-loop joint tracking  │  JSON plan │  M2T2 grasps → table plane + convex hulls  │    └────────────────┘
 │ client.SimStateStream   mirror: meshes once, then ├──────────▶ │  cuTAMP + cuRobo → joint-space plan        │
 │                         poses + JPEGs, 2nd ws     │            │  Rerun: one recording per process          │
 └──────────────────────────────────────────────────┘            │   └─ child process: rerun --serve-web       │ ◀── browser
                                                                  └───────────────────────────────────────────┘     (ssh -L 9090, 9876)
```

Three isolated environments, by design (they cannot be merged: python 3.11 vs 3.12, numpy 1 vs 2, two cuRobo forks
with the same import name, Isaac Sim's torch 2.7.0/cu128 vs the planner's 2.7.1/cu129): the sim env (`b1k` from
`setup_uv.sh` on the server, a conda env on a laptop), the planner's pixi env in the `tiptop/` submodule, and
M2T2's pixi env in a separate clone. Only websocket/HTTP crosses the boundaries. Simulator-only shortcuts
(ground-truth masks, goal hints, the mirror) live in tiptop behind optional request fields, so the real-robot
path stays untouched.

| Concern | Repo |
|---|---|
| Planner, perception, embodiments (cuRobo/cuTAMP configs, tool frame, gripper spheres), the wire protocol | **tiptop** (submodule `tiptop/`, private fork; runs in its own pixi env, deployable unchanged) |
| Scenes, tasks, robot and controller configs, capture, plan execution, scoring, video, the Rerun mirror's sim side | **BEHAVIOR-1K** (this directory; needs Isaac Sim) |

## One round, step by step

1. **Stand.** The base is teleported once per episode (`--stand-for`, `--near`, `--robot-pose`); the planner never
   moves it. `--stand-for ITEM[,ITEM...],TARGET` searches a pose from which every named object is ahead, on the
   left, within the arm's reach and inside the head camera's view (see "R1Pro specifics").
2. **Capture** (`R1ProSim.capture`). The left arm swings out of the head camera's view (`LOOK_ARM`), an external
   "shadow" camera with the head camera's intrinsics is moved onto its pose and renders rgb + `depth_linear` until two
   consecutive frames agree (the renderer accumulates over time after a teleport), then the arm returns to the ready
   posture, which becomes the plan's `q_init`. Masks: ground truth from geometry (`gt_masks.py`: depth pixels within
   8 mm of an object's mesh; Isaac's instance-segmentation annotator crashes in the house scenes), or none with
   `--no-gt`. `validate_capture` warns when a goal object is cut by the image border (its hull would run past the
   real object -- the silent failure mode of the pipeline).
3. **Request** (`protocol.build_request`, msgpack with numpy arrays, one websocket connection per request):
   `rgb, depth, intrinsics, world_from_cam` (OpenCV camera in the robot base frame), `task, q_init`, plus
   `gt_labels, gt_atoms` and either `gt_masks` (oracle) or `goal_hints` + `robot_mask` (detector).
4. **Plan** (`tiptop-server`, `_run_pipeline`). Masks → point cloud in the base frame → M2T2 grasps (associated to
   objects by contact point) → table plane by RANSAC + one convex hull per object → cuTAMP samples pick/place
   skeletons over 256 particles, cuRobo refines the motions → `{q_init, steps: [trajectory{positions, dt} |
   gripper{open|close}]}`. The response also carries `objects: {label: {position, movable, grasps}}`, what
   perception made of the frame, and `save_dir`, the run directory with the planner's own logs and images.
5. **Execute** (`executor.PlanExecutor`). Trajectories are resampled from the plan's `dt` to the env step (1/30 s)
   and tracked with absolute joint targets; gripper events hold the arm for `--gripper-hold-steps`; grasps are
   `sticky` for the demos (physical grasps of thin objects slip). Tracking lag and gripper state go to the result.
6. **Score.** With `--activity` the task's own goal predicates are evaluated the way the challenge does (`TaskMetric`:
   1 on full success, else the newly satisfied fraction of the best goal option); `forpairs` goals ground into
   hundreds of thousands of options, so each grounded predicate is evaluated once and memoized.

Throughout, the simulator mirrors itself into the planner's Rerun (step 0 of "What Rerun shows").

## Bring-up

```bash
OmniGibson/omnigibson/tiptop/scripts/start_m2t2.sh                                  # http://127.0.0.1:8123
TIPTOP_CONFIG=tiptop/config/tiptop_sim_r1pro.yml TIPTOP_PARTICLES=256 TIPTOP_MAX_PLANNING_TIME=40 \
    OmniGibson/omnigibson/tiptop/scripts/start_tiptop_server.sh                     # ws://127.0.0.1:8765 + Rerun
curl -s localhost:8123/health; curl -s localhost:8765/health                         # the planner answers after ~40 s
```

One terminal (or tmux window) per service, in the foreground, so Ctrl-C stops it. Launcher knobs: `TIPTOP_GPU` /
`M2T2_GPU` (pin a card on a shared box), `TIPTOP_HOST=0.0.0.0` to serve other machines, `TIPTOP_RERUN_MODE`
(`serve` by default), `TIPTOP_DIR` / `M2T2_DIR`. The Rerun view is at
`http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy`; from a laptop, tunnel both ports
(`ssh -N -L 9090:127.0.0.1:9090 -L 9876:127.0.0.1:9876 <server>`, plus 8765 if the simulator runs on the laptop).

## The demo: one basket, four items, one base pose

The test that exercises the whole pipeline: the `assembling_gift_baskets` challenge scene loads, one wicker basket
is put on the coffee table with one candle, one cheese, one cookie and one bow next to it, the robot is placed once,
flush with the table's long side, and fills the basket round by round with oracle masks -- four captures, four
plans, four executions, the robot never moves.

```bash
OMNIGIBSON_HEADLESS=1 python -m omnigibson.tiptop.run live --embodiment r1pro --activity assembling_gift_baskets \
    --place wicker_basket.n.01_2:table.n.02_1:0.20,0.50 --place candle.n.01_4:table.n.02_1:0.05,0.12 \
    --place butter_cookie.n.01_1:table.n.02_1:0.25,0.12 --place bow.n.08_3:table.n.02_1:0.32,-0.30 \
    --torso 1.2 -1.7 -0.9 0.0 \
    --stand-for candle.n.01_4,swiss_cheese.n.01_1,butter_cookie.n.01_1,bow.n.08_3,wicker_basket.n.01_2 --sequential \
    --goal "inside(candle.n.01_4,wicker_basket.n.01_2);inside(swiss_cheese.n.01_1,wicker_basket.n.01_2);inside(butter_cookie.n.01_1,wicker_basket.n.01_2);inside(bow.n.08_3,wicker_basket.n.01_2)" \
    --task "prepare a gift basket: put the candle, the cheese, the cookie and the bow in the wicker basket" \
    --grasping-mode sticky --host localhost --port 8765 --out-dir runs/demo
```

What happens, and how long it takes (shenlong-gpu-01, 2026-09-05):

| Phase | What to see | Time |
|---|---|---|
| scene + task load | the log; nothing in Rerun yet but the planner's robot at its home pose | 3-5 min |
| stage + stand | `placed ... on table`, `head camera at z 1.25 m sees a surface at z 0.42 from 0.40 m ahead`, `standing for ...: (4.29, 5.68) yaw -165 deg, distances [0.77, 0.76, 0.58, 0.82, 0.58]`; the robot, its 21 green objects and three camera views appear in Rerun | 10 s |
| per round: capture | `ground-truth masks from geometry: pixels per label {...}` | 8 s |
| per round: plan | `server planned in 3.3s`, then one `perceived 'candle_4' (goal, 173 grasps) = simulated candle_4 (2.4 cm off)` line per object; hulls and the goal object's grasps replace the previous round's in the 3D view | 3-4 s |
| per round: execute | `[2] gripper close ... is_grasping=1`, the arm in the 3D view and the cameras; `live.mp4` written | 25-30 s |
| per round: score | `round N ...: {'q_score': 0.0625 * (N+1), ...}` | 20 s |

Result of that command: all four items ended up inside the basket, task score 0 → 0.25 (4 of its 16 `inside`
predicates; the best 2025 submission reached 0.31 on this task, none completed it); a repeat the same night
placed three (M2T2 returned no grasps for the bow, see "Known limits"). Outputs: `runs/demo/round_0N/`
with `capture.json` (poses, intrinsics, validation), `rgb.png`, `depth.png`, `gt_masks.png`, `obs.h5`,
`tiptop_plan.json`, `server_response.json`, `live.mp4`, `live_result.json` (tracking errors, gripper events, the
goal status, final object poses, and `perception`: the pairing above), and `sequential_summary.json`; on the
planner side `tiptop/tiptop_server_outputs/<timestamp>/` per request (its log, `masks_viz.png`, the cuTAMP
environment, grasps, `metadata.json`).

Why it is set up this way (all measured in the scene):

- No single base pose reaches four items where the task leaves them: the coffee table is 0.82 x 1.67 m, the arm
  reaches ~0.9 m. So the candle, cookie and bow are teleported next to the basket at the table's +x edge, where
  the cheese already is (`--place OBJ:SUPPORT:DX,DY`, offsets from the support's centre). Teleporting is test
  scaffolding: the rules forbid it during evaluation, and the base does not move under its own controller yet.
- `--torso 1.2 -1.7 -0.9 0` starts the torso a little lower than the challenge posture (head camera at 1.25 m
  instead of 1.40) and pitched forward. The pitch is what lets the robot stand close: for the posture
  `apply_posture` established, the base-pose search measures where the camera's bottom image edge meets each
  object's support (`camera_floor_distance`: 0.55 m ahead in the challenge posture, 0.40 m tilted) and keeps the
  object beyond that plus 8 cm. Crouching the hips further (joint1 1.3, joint2
  -1.9) puts cuRobo's sphere model of the robot in self-collision at every tilt, so every plan fails with
  `INVALID_START_STATE_SELF_COLLISION`; deeper still (1.5, -2.2) the simulator cannot hold the locked right arm.
- Oracle masks make the run about planning and execution; `--no-gt` runs the same set-up on the detector + SAM2
  (competition style), where the instance the goal means is passed as a `goal_hint` and the pairing lines tell
  you what the detector actually found.
- `--sequential` gives one capture/plan/execute round per goal atom from where the robot stands; `--restand`
  teleports the base to a fresh pose before each round instead. `--stand-for` fails loudly, with its rejection
  counts, when no single pose reaches everything named.

## What Rerun shows

The planner hosts the viewer (`--rerun-mode serve`): one recording per planner process, the viewer is the
planner's child process and dies with it, so a fresh tab never shows an earlier session, and the SDK's own viewer
binary is used (versions match by construction; a `rerun` of another version on `PATH` is the classic mismatch).
The planner refuses to start while 9876 or 9090 is taken: `pkill -u $USER -f 'rerun --serve-web'`, or pass
`--rerun-grpc-port` / `--rerun-web-port`. The layout is sent with the recording: the 3D world on the left, the
simulator's cameras and the last request's masks on the right.

| entity | what | from |
|---|---|---|
| `r1pro_left/...` (`panda/...`) | the planner's robot model at the simulator's current joints | planner (URDF); joints from the simulator |
| `world/sim/<task name>` | green: the simulator's own meshes of the task objects at their simulated poses; grey-blue: furniture named on the command line (`--place` supports, `--near`, `--stage-support`) | simulator, meshes once, poses every 2 env steps |
| `world/objects/<label>`, `grasps/<label>/...`, `world/table`, `pcd`, `cam` | grey: what perception reconstructed for the *last* request -- hulls, the top 30 grasps of the goal's objects, table plane, cloud, camera -- cleared when the next request arrives | planner |
| `sim/head_cam`, `sim/wrist_cam` (R1Pro) or `sim/cam` (Panda), `sim/overview` | the head camera, the left wrist camera, a third-person view over the robot's left shoulder | simulator, every 6 env steps |
| `masks` | the last request's image with its masks and boxes (`rgb`, `bboxes`, `obj_pcd/*` are logged too but hidden: the same content) | planner |

Names: `world/sim/*` uses the simulator's task names (`candle_4` is `candle.n.01_4`); `world/objects/*` uses
perception's, which with ground-truth masks are the same names and with the detector number instances by box
size (the largest keeps the plain label, then `_2`, `_3`, ...), so `candle_2` is usually a *different* candle in
the two trees. Nothing is matched by name;
the pairing is by position (`protocol.match_objects`, 8 cm), logged after every plan and saved with the result. A
perceived object with no simulated partner is a false detection or a hull that landed somewhere else, and the
goal object is the first line to read.

Everything sits on the `log_time` timeline (wall clock). Keep the viewer on *Following* (time panel, bottom);
the simulator runs slower than real time, so the view is live but not real-time-scaled. Reload the tab after a
planner restart. The simulator side is `client.SimStateStream`, a second websocket connection to the planner's
port, open for the whole session: meshes once (decimated to 4000 triangles), then joints and poses every 2 env
steps and 480 px JPEGs in every third message; a failure only switches the mirror off (`--no-state-stream` does so
up front; `replay --state-stream host:port` mirrors an offline replay). Other planner modes: `save` writes one
`tiptop.rrd` per request under `tiptop/tiptop_server_outputs/<ts>/` (`cd tiptop && pixi run rerun --serve-web
--bind 127.0.0.1 <file>` replays it), `connect` streams to a viewer you started, `stream` spawns the native
viewer (needs a display). The R1Pro renders as a mesh-less set of frames unless its visual meshes have been
generated next to the URDF (gitignored, ~50 MB): `cd tiptop && pixi run python scripts/make_r1pro_embodiment.py
--copy-meshes`, then `git -C tiptop checkout -- tiptop/embodiments/assets/r1pro/r1pro_left_meta.yml`.

## Perception modes

Chosen by what the request carries:

- **Ground truth** (default): `gt_labels` (every tracked task object, `candle_4` style), `gt_masks` from geometry
  (or from Isaac's annotator with `--seg-instance`, where it works: Rs_int, not the house scenes) and `gt_atoms`
  per instance. Objects out of view are dropped from the request; a goal object out of view is an error. The
  server skips detection and SAM2 and runs everything else unchanged. Fast and exact; for development.
- **Competition style** (`--no-gt`): only `gt_labels` (categories: `candle`), `gt_atoms` and `goal_hints` (where the
  instance the goal means is, so the closest detected instance gets the plain label), i.e. what an agent knows from
  the task definition. Grounding DINO (prompts per category in `tiptop_sim_r1pro.yml`, e.g. "round cookie") finds
  boxes in the head-camera image, SAM2 segments them; `robot_mask`, the robot's own pixels, keeps SAM2 off an
  occluding gripper (only available with `--seg-instance`). Instances are numbered by box size, largest first.
- **Gemini** (`perception.detector: gemini`, tiptop's upstream default): Gemini detects the objects and translates
  the task; needs `GOOGLE_API_KEY`; atoms sent with the request take precedence.

## CLI

`python -m omnigibson.tiptop.run <subcommand>`, inside the sim env, with `OMNIGIBSON_HEADLESS=1` (or unset for the
Isaac GUI):

| subcommand | does | own flags |
|---|---|---|
| `capture` | build the scene, write `obs.h5` + `capture.json` (offline input for `tiptop-h5`) | |
| `live` | capture, plan on a running `tiptop-server`, execute, score | `--host --port --plan-timeout --no-state-stream --sequential --restand` |
| `replay` | build the scene, execute a `tiptop_plan.json` | `--plan --state-stream HOST:PORT` |
| `task` | work through a challenge task's whole `inside(item, container)` goal: containers staged on `--stage-support` one at a time, a fresh base pose per transfer, items verified with the task's own predicate | `--host --port --plan-timeout --no-state-stream --stage-support --attempts-per-item` |

Flags shared by all: `--embodiment franka|r1pro`, `--activity NAME` (+ `--activity-instance`, `--rooms`), scene
set-up `--place OBJ:SUPPORT[:DX,DY]`, `--spawn PRESET:SUPPORT[:DX,DY]`, `--scene-objects`; the base
`--stand-for ITEM[,ITEM...],TARGET` | `--near FURNITURE [--side] [--standoff]` | `--robot-pose X Y YAW`; the posture
`--torso J1 J2 J3 J4`, `--no-look`; the capture `--camera head|wrist`, `--head-aperture`, `--seg-instance`,
`--no-gt`; the goal `--goal "pred(a,b);..."` (BDDL names with `--activity`), `--task`; execution
`--grasping-mode physical|assisted|sticky`, `--gripper-hold-steps`, `--finger-max-effort`, `--settle-steps`,
`--no-video`; `--scene capture.json` reuses an earlier capture's settled object poses; `--not-load` drops object
categories from the scene. `--help` on a subcommand lists them with defaults.

Panda tabletop (no BEHAVIOR scene, `TIPTOP_CONFIG=tiptop/config/tiptop_sim_panda.yml` on the planner):

```bash
python -m omnigibson.tiptop.run live --host localhost --port 8765 --out-dir runs/live      # mug into bowl
python -m omnigibson.tiptop.run capture --out-dir runs/scene1
cd tiptop && pixi run tiptop-h5 --config tiptop/config/tiptop_sim_panda.yml \
    --h5-path ../runs/scene1/obs.h5 --task-instruction "put the mug in the bowl" --no-rr-spawn     # offline planning
python -m omnigibson.tiptop.run replay --plan <run>/tiptop_plan.json --scene runs/scene1/capture.json --out-dir runs/replay
```

## Conventions that matter

- **World frame = robot base frame** (cuRobo `base_link`: `panda_link0` on the Panda, the floor-level `base_link` on
  the R1Pro). The camera pose is re-expressed there, so the robot may stand anywhere in the world.
- **Camera axes**: OmniGibson/USD cameras look down −z with +y up; TiPToP expects OpenCV (+z forward, +y down):
  `q_cv = quat_multiply(q_usd, [1, 0, 0, 0])` (180° about the camera x axis).
- **Depth**: `depth_linear` (distance to the image plane) in metres, invalid pixels 0. Not `depth` (ray length).
- **Quaternions**: OmniGibson (x, y, z, w); the droid H5 layout (w, x, y, z).
- **Controllers**: absolute joint targets (`JointController`, position mode, no deltas, no normalization, no limits)
  on the trunk and arm groups, joints gathered by name (OmniGibson interleaves the two arms in its joint order);
  binary grippers (`+1` open, `−1` close; the DROID client uses the opposite polarity); the R1Pro base sits on a
  holonomic controller that is fed zeros, so it holds still.
- **Embodiments**: OmniGibson `franka` ↔ tiptop `panda` (identical kinematics, joint order and tool frame, no new
  embodiment needed); OmniGibson `r1pro` ↔ tiptop `r1pro_left` (generated from the same URDF and collision spheres).
  The server advertises the embodiment (joint names, locked joints, home pose) in its metadata and the simulator
  applies it before capturing, so both sides agree by construction; the client refuses a server that plans for
  another robot.
- **M2T2 crop**: the Panda config keeps M2T2's built-in box (x 0..1, |y| ≤ 0.3, z −0.2..0.5 m); the R1Pro config
  crops the cloud to `perception.m2t2.crop_bounds` (the workspace ahead of a floor-level base) instead.

## R1Pro specifics

- **Planner model.** `r1pro_left` plans torso (4) + left arm (7); the right arm and both grippers are locked. FK
  agrees with the simulator to 0.03° / 0.0 mm over 25 random configurations (`scripts/probe_r1pro.py` +
  `tiptop/scripts/check_r1pro_embodiment.py`). With the torso locked the arm reaches only 0.4-0.6 m ahead on its
  left, which is why the torso is planned.
- **Posture.** `apply_posture` holds the locked joints and drives the planned ones to the embodiment's `q_home`
  (or `--torso` for the torso entries) and checks the simulator holds it (0.03 rad); it runs before the base pose
  is chosen because the head camera's reach (`camera_floor_distance`: where the bottom image edge meets a support)
  follows from it. The base gets the evaluator's 250 kg mass; without it the leaning posture tips the robot over.
- **Cameras.** Head (`zed_link`, 720x720, 40 mm aperture = 99° HFOV as in the challenge) for the capture, left wrist
  (`left_realsense_link`, 480x480) and an external overview camera for the mirror. Instance segmentation attached to
  a robot-mounted camera leaks GPU memory and segfaults after ~35 steps in this Isaac build, so the robot cameras
  render rgb only and an external shadow camera is moved onto the head camera's pose for the capture frame.
- **Base pose** (`best_base_pose`): candidates on rings 0.25-0.9 m around the named objects' centroid, facing it,
  yaw ±60° in 15° steps; rejected when an object is behind (< 0.15 m ahead), well to the right (> 0.3 m), beyond
  reach (0.9 m), nearer than the camera's reach for its own support height, hidden behind the container, outside
  ±45° of forward, or
  the footprint (0.36 m half extent) is not on a floor inside a room and free of other objects. Score: the farthest
  object's distance, a penalty for objects not on the left, for turning, and for object edges falling outside the
  camera frame (a basket cut by the border reconstructs 8 cm too long and the item is released beside it -- seen
  2026-09-04). `--stand-for` and `task` use it; `choose_stage_spot` (task) picks the container's spot on the table
  with it.

## Known limits

- One base pose per round: items farther than ~0.9 m from the container need the base to move while holding, which
  the pipeline does not model; `task` mode re-stands and re-stages the container per transfer instead, and the
  four baskets of the gift task sit on the floor (a floor-level pick or a carry).
- Placement goes onto the top face of the container's convex hull with a 1 cm surface shrink, which is less than a
  wicker rim: an item can be set down on the rim (2026-09-05, from a stretched 0.8 m reach) and topple the basket.
- Flat objects (cheese slabs, bows) get few M2T2 grasps; the planner succeeds on them from close, orthogonal
  viewpoints and fails from others.
- Planner variance: the same capture can fail once with "Motion planning failed for 32/74 satisfying particles"
  and succeed next time (grasp sampling differs per call). Retry before debugging.
- Teleports (`--place`, `--stand-for`, `--torso`) are scaffolding the rules forbid during evaluation.

## Tests

`pytest OmniGibson/tests/test_tiptop_protocol.py OmniGibson/tests/test_tiptop_gt_masks.py` (no Isaac Sim): the
msgpack-numpy wire format, request validation, plan parsing and resampling, the H5 layout, the name helpers, the
position-based perception pairing, and the geometry masks.

## History

- 2026-09-02: Panda, mug into bowl over the websocket (25 of 256 particles feasible, 1.3 s cuTAMP); R1Pro in Rs_int,
  mug into bowl with the head camera and sticky grasps.
- 2026-09-03: gift-basket task, detector + SAM2, cookie then candle into a staged basket with a fresh base pose per
  round, task score 0 → 0.125.
- 2026-09-04: the bridge and the planner on shenlong-gpu-01 (Blackwell); frame-coverage check; Rerun served by the
  planner with the simulator mirrored into it; perception paired with the simulator by position (the name-based
  mirror had moved the wrong object).
- 2026-09-05: the demo above with oracle masks, four items from one base pose, task score 0.25; wrist camera and
  layout in Rerun; camera reach measured per posture.
