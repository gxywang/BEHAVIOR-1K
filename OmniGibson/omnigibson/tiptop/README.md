# TiPToP ↔ OmniGibson bridge

This package runs [TiPToP](https://github.com/tiptop-robot/tiptop) -- task-and-motion planning for pick-and-place
from one RGB-D image and a goal -- against the BEHAVIOR-1K simulator, with the challenge robot (R1Pro) in the
challenge scenes. It is the OmniGibson counterpart of TiPToP's own IsaacLab client and speaks only TiPToP's
existing contracts: **nothing here imports `tiptop`, and TiPToP never imports OmniGibson.** They are separate
processes in separate Python environments and talk over a websocket.

The runbook for the lab server (GPU pinning, tunnels, the exact shell lines) is [USAGE_DOCS.md](../../../USAGE_DOCS.md);
installing the planner and grasp server on a new machine and the problems you will meet is [DEPLOYMENT.md](DEPLOYMENT.md).

## Results

2026-09-09, the challenge's public test instances 0-9, `python -m omnigibson.tiptop.bench` (the challenge's metric,
timeout and instance loading; see "Benchmark"). Oracle knowledge (masks and button poses from the simulator) and a
teleported base, so these bound the manipulation part and are not challenge scores. Each run is
`runs/bench_<task>_pass<N>/`: `json/` per instance, `summary.json` with a `what_failed` line per instance,
`videos/` with one video per instance that ends on its verdict. Labels: *no base pose* = no free floor pose put the
object within reach (1.1 m at most); *no plan* = the planner found no satisfying plan for the round; *not visible*
= the goal object had no pixels in the capture; *ran, not inside* = the place executed but the predicate stayed
false (the item landed outside the rim or fell); *released* = the last-resort open of a hand no plan could empty.

- **turning_on_radio**, five passes (pick with the left hand, press with the right while holding): mean q_score
  **0.7 / 0.7 / 0.6 / 0.7 / 0.7**, 34 of 50 instances. Pass 5 is the reference: the current code (the one retry
  policy without the put-down and re-pick, the press face inscribed in the button's radius, the plan's gripper
  start state), each video ending on the flip with the button's marker turned green. Pass 4's videos end the same
  way; the tails of passes 1-3 are a frozen copy of their last frame, which predates the marker's colour change.
  - pass 1 (`runs/bench_radio_pass1`, 7/10): 302, 303 no base pose within 0.9 m (the search now widens to 1.1 m);
    309 press: no plan x2 after the pick.
  - pass 2 (`runs/bench_radio_pass2`, 7/10): 302 press: no plan x2 (two picks); 303 no base pose; 309 press: no
    plan x2.
  - pass 3 (`runs/bench_radio_pass3`, 6/10, final code): 301 press: no plan x4, even after a put-down and a re-pick;
    302 two pick rounds ran, the radio was never grasped (0.95 m reach), so no press; 307 the press ran twice
    without toggling, then the re-pick's press had no plan x2; 309 press: no plan x4 (three picks, one put-down).
  - pass 4 (`runs/bench_radio_pass4`, 7/10, current code): 302 press: no plan x4 after three picks and a put-down;
    308 press: no plan x4 after one pick and a put-down; 309 the press ran twice without toggling, then the re-pick's
    press had no plan x2.
  - pass 5 (`runs/bench_radio_pass5`, 7/10, current code, `--rounds 2`, no re-pick): 302 press: no plan x2 (the
    switch 0.57 m ahead, tilted 7 deg down; the first pick closed on nothing and the hand was opened before the
    second); 303 press: no plan x2 (0.63 m ahead); 309 press: no plan x2 (0.61 m ahead). Every press that planned
    toggled the switch (7 of 7; pass 4 had two executed presses that missed by 2 mm), and the median instance took
    722 env steps against 839 in pass 4.
  - What fails is the right hand's press plan, which depends on the grasp the left hand chose (a switch left
    0.65 m ahead: no plan 3/3; 0.59 m: plans 3/3), not the pick (45 of 50 instances ended with the radio in hand).
- **assembling_gift_baskets** (four baskets on the floor, one candle, cheese, cookie and bow each, 16 transfers):
  - pass 1 (`runs/bench_baskets_pass1`, 2 instances, stopped): 301 9/16 (seven rounds with the item not visible
    from the capture pose, no plan x2, bow_2 no base pose); 302 1/16 (not visible x19, no plan x11: a failed
    put-down left the item in the hand and every later request named it; fixed for pass 2).
  - pass 2 (`runs/bench_baskets_pass2`): mean **0.6375**, 303 and 310 complete. 301 11/16 bows 2 and 4 no base
    pose (six attempts), no plan x2; 302 15/16 bow_4 no plan x2; 304 11/16 basket_2's items no plan x9; 305 0/16,
    306 3/16, 309 3/16 the same loop: a failed place left the item in the hand and no put-down plan emptied it
    (48 / 45 / 43 failed rounds, the table no base pose from where it stood); 307 14/16 bow_4 no base pose,
    candle_1 not visible; 308 13/16 bow_1 no base pose x4, no plan x1.
  - pass 3 (`runs/bench_baskets_pass3`, the last-resort release added): mean **0.875**, 304 and 310 complete.
    301 11/16 bow_4 no base pose x4, no plan x3 (cheese_3, cookie_1); 302 14/16 bows 3 and 4 ran, not inside;
    303 13/16 basket_4's cheese_2, cookie_1, bow_2 no plan x4, one released; 305 10/16 basket_4 in a room corner
    (no base pose x5), bow_3 and candle_2 no base pose, no plan x3; 306 15/16 candle_2 no plan x4 and not visible
    x2, one released; 307 15/16 bow_4 no base pose x2; 308 15/16 bow_1 no base pose x3, one released; 309 15/16
    cheese_2 ran, not inside.
  - What remains costs one or two items per instance: a bow at the far edge of the table that no base pose
    reaches, a basket standing in a room corner, and places with no satisfying plan; 27-33 min of wall time and
    about 16k of the 39k allowed env steps per instance.

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
M2T2's pixi env in a separate clone. Only websocket/HTTP crosses the boundaries. What a simulator client can add
to a request (oracle masks, button poses, what the hands hold, the mirror) lives in tiptop behind optional request
fields, so the real-robot path stays untouched.

| Concern | Repo |
|---|---|
| Planner, perception, embodiments (cuRobo/cuTAMP configs, tool frame, gripper spheres), the wire protocol | **tiptop** (submodule `tiptop/`, private fork; runs in its own pixi env, deployable unchanged) |
| Scenes, tasks, robot and controller configs, capture, what the planner is told (`knowledge.py`), plan execution, scoring, video, the Rerun mirror's sim side, task strategies and the benchmark | **BEHAVIOR-1K** (this directory; needs Isaac Sim) |

Modules in this directory: `protocol.py` (wire and file formats, no OmniGibson imports), `client.py` (websocket
client and the Rerun mirror), `scene.py` (the simulator: stepping, capture, episode accounting), `r1pro.py` (the
R1Pro in a BEHAVIOR scene: posture, cameras, task scope, base-pose search), `knowledge.py` (what the client tells
the planner beyond the image: an oracle source and an onboard source), `executor.py` (plan execution, video),
`strategies.py` (how a task is split into rounds), `bench.py` (the challenge-style benchmark), `replay.py` (re-plan
a saved round), `run.py` (the CLI).

## One round, step by step

1. **Stand.** The base is teleported once per episode (`--stand-for`, `--near`, `--robot-pose`); the planner never
   moves it. `--stand-for [ITEM,...,]TARGET` (one name for a one-object task) searches a pose from which every named object is ahead, on the
   left, within the arm's reach and inside the head camera's view (see "R1Pro specifics").
2. **Capture** (`R1ProSim.capture`). The left arm swings out of the head camera's view (`LOOK_ARM`), an external
   "shadow" camera with the head camera's intrinsics is moved onto its pose and renders rgb + `depth_linear` until two
   consecutive frames agree (the renderer accumulates over time after a teleport), then the arm returns to the ready
   posture, which becomes the plan's `q_init`. `validate_capture` warns when a goal object is cut by the image
   border (its hull would run past the real object -- the silent failure mode of the pipeline).
3. **Request** (`protocol.build_request`, msgpack with numpy arrays, one websocket connection per request):
   `rgb, depth, intrinsics, world_from_cam` (OpenCV camera in the robot base frame), `task, q_init`, plus what the
   knowledge source knows (`knowledge.py`, `protocol.attach_knowledge`; see "What the planner is told"):
   `gt_labels, gt_atoms` always, `gt_masks` from the oracle source, `gt_buttons` from the oracle source (true poses)
   or the onboard source (detections carried from earlier rounds), `held_labels` / `in_hand` for what the hands
   hold, `workspace_bounds` when the round works at the floor.
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
| per round: capture | `oracle masks: pixels per label {...}` | 8 s |
| per round: plan | `server planned in 3.3s`, then one `perceived 'candle_4' (goal, 173 grasps) = simulated candle_4 (2.4 cm off)` line per object; hulls and the goal object's grasps replace the previous round's in the 3D view | 3-4 s |
| per round: execute | `[2] gripper close ... is_grasping=1`, the arm in the 3D view and the cameras; `live.mp4` written | 25-30 s |
| per round: score | `round N ...: {'q_score': 0.0625 * (N+1), ...}` | 20 s |

Result of that command: all four items ended up inside the basket, task score 0 → 0.25 (4 of its 16 `inside`
predicates; the best 2025 submission reached 0.31 on this task, none completed it); a repeat the same night
placed three (M2T2 returned no grasps for the bow, see "Known limits"). Outputs (now under `runs/archive/`): `runs/archive/demo/round_0N/`
with `capture.json` (poses, intrinsics, validation), `rgb.png`, `depth.png`, `gt_masks.png`, `obs.h5`,
`tiptop_plan.json`, `server_response.json`, `live.mp4`, `live_result.json` (tracking errors, gripper events, the
goal status, final object poses, and `perception`: the pairing above), and `sequential_summary.json` plus
`full.mp4`, the whole run in one video (every round with its captures, holds and arm switches; captioned with the
round's goal). Videos show the capture camera on the left and, down the right, the overview and the left wrist
camera; they are fed from every second simulator step at 15 fps, so they play in simulated time; on the
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
- Oracle masks make the run about planning and execution; `--knowledge onboard` runs the same set-up on the
  detector + SAM2 (competition style), and the pairing lines tell you what the detector actually found.
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
| `world/sim/<task name>` | green: the simulator's own meshes of the task objects at their simulated poses; grey-blue: furniture named on the command line (`--place` supports, `--near`) | simulator, meshes once, poses every 2 env steps |
| `world/objects/<label>`, `grasps/<label>/...`, `world/table`, `pcd`, `cam` | grey: what perception reconstructed for the *last* request -- hulls, the top 30 grasps of the goal's objects, table plane, cloud, camera -- cleared when the next request arrives | planner |
| `sim/head_cam`, `sim/wrist_cam` (R1Pro) or `sim/cam` (Panda), `sim/overview` | the head camera, the left wrist camera, a third-person view over the robot's left shoulder (ahead and to the right with `--overview front`) | simulator, every 6 env steps |
| `masks` | the last request's image with its masks and boxes (`rgb`, `bboxes`, `obj_pcd/*` are logged too but hidden: the same content) | planner |

Names: `world/sim/*` uses the simulator's task names (`candle_4` is `candle.n.01_4`); `world/objects/*` uses
perception's, which with ground-truth masks are the same names and with the detector number instances by box
size (the largest keeps the plain label, then `_2`, `_3`, ...), so `candle_2` is usually a *different* candle in
the two trees. Nothing is matched by name;
the pairing is by position (`protocol.match_objects`, within half the object's extent, since a hull seen from one
side is centred above the object's centre), logged after every plan and saved with the result. A
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

## What the planner is told

The planner works from the image and from what the request says about the scene. Where that comes from is one
choice, `--knowledge`, made in `knowledge.py` and nowhere else; the rest of the pipeline never asks which source it
got. Every run records the source in its results, and the oracle one logs a PRIVILEGED warning at start.

- **`oracle`** (the default, for development): the simulator's truth. Labels per instance (`candle_4`), `gt_masks`
  from geometry (`gt_masks.py`: depth pixels within 8 mm of an object's mesh; or Isaac's annotator with
  `--seg-instance`, where it works: Rs_int, not the house scenes) and the true pose of every toggle button the task
  presses (`gt_buttons`, sent in every round so the pick round can choose a grasp that presents it). Objects out of
  view are dropped from the request; a goal object out of view is an error (`GoalNotVisible`). The server skips
  detection and SAM2 and runs everything else unchanged. The challenge forbids all of this at evaluation time.
- **`onboard`** (competition style): what an agent knows. Category names (`candle`), the goal atoms, the gripper
  state (`held_labels`, `in_hand`), and for a toggle button its label (`<object>_button`) so the detector looks
  for it; a button detected in an earlier round is carried through a grasp by the arm's kinematics and sent as a
  prior (`ButtonTracker`). Grounding DINO (prompts per category in `tiptop_sim_r1pro.yml`, e.g. "round cookie")
  finds boxes in the head-camera image, SAM2 segments them; `robot_mask`, the robot's own pixels, keeps SAM2 off
  an occluding gripper (only available with `--seg-instance`). Instances are numbered by box size, largest first,
  so a category-level goal acts on the largest (closest) instance.
- **Gemini** (`perception.detector: gemini`, tiptop's upstream default): Gemini detects the objects and translates
  the task; needs `GOOGLE_API_KEY`; atoms sent with the request take precedence.

Both sources report what the hands hold, read from the robot's own grasp assist (sticky / assisted grasping):
`in_hand` names objects the planned arm holds, so a plan can start holding one (a carry: pick at the table, move,
place into a basket on the floor), and `held_labels` names what the other hand holds, which stays an obstacle.
The base pose search (`--stand-for`, the strategies) reads object poses from the simulator too: navigation is a
teleport stand-in and is counted as one in the benchmark's results.

Toggle buttons (`toggled_on(obj)` goals, e.g. `turning_on_radio`): the button is a 2 cm marker on the object and
the atom goes out as `pressed(<label>_button)`. The oracle source sends its pose (`button_hints` -> `gt_buttons`:
base-frame position, the outward normal of the face it sits on from the object's own mesh, and the radius within
which OmniGibson counts a finger). With the onboard source the planner finds it: the label goes into
`gt_labels`, Grounding DINO looks for it (`phrases` in the planner config, "small red button" for the radio) in a
zoomed view of the detected object's box, SAM2 masks it in a zoomed crop, and the mask's depth points give the
position, a plane through the surrounding depth points the normal (0.7 cm and 4 deg off the true button on the
held radio). The button label is asked for in every round, so the button is first seen on the table before the
pick (in the hand it often sits at the image border); the planner reports what it detected (`buttons` in the
response) and `ButtonTracker` keeps it: once the object is grasped, the pose moves with the gripper that closed
on it (p_now = T_eef_now inv(T_eef_at_close) p_then, the arm's own kinematics; 1.6 cm off after a 43 cm lift with
a 137 deg turn) and is sent as `gt_buttons`, a prior that a fresh detection in the press round overrides when it
lands within 5 cm. So that the press round can see the button, the planner chooses the grasp with it: a grasp
fixes where the object ends up after the lift, so the pick round keeps only the M2T2 grasps that leave the button
facing the head camera (`tiptop/presenting.py`; the best few when none do), and the left arm's home pose holds
the object in the camera's view. The request also names what the other hand holds (`held_labels`), which the
planner keeps as an obstacle rather than something to pick up. Either way the planner's `Push` plans hover,
press and back-off along the normal with the gripper closed. With the oracle source the executor stops the press segment as soon as the simulator's `ToggledOn` flips (the onboard source has no such signal: the press runs to its planned depth; a finger on the object
inside that radius for 5 steps) and lets the back-off run; meanwhile the bridge keeps OmniGibson's sticky or
assisted grasp off the pressing arm (`block_grasping`), since a closed gripper touching an object for 0.3 s would
otherwise attach it. The round is scored by the task's own `toggled_on`.

Two hands (`--press-port`): with `--sequential` and a goal like `holding(radio);toggled_on(radio)`, the first round
picks and holds with the left arm on the usual planner, then `adopt_embodiment` switches to planning the right arm
(`r1pro_right`, a second `tiptop-server` on that port; nothing moves, the left gripper keeps its close command and
the left joints are held where they are) and the press round captures in place, with the held object in the head
camera's view, and presses with the closed right gripper (its press point is the midpoint of the fingertips). The
Rerun mirror keeps reporting the left
embodiment's joints. `--overview front` puts the third-person camera ahead and to the right of the robot, looking
back at both hands (the default stands over the left shoulder, where the pressing hand is hidden by the torso);
`full.mp4` in the output directory is the whole run. Bring-up for it:

```bash
M2T2_GPU=4 OmniGibson/omnigibson/tiptop/scripts/start_m2t2.sh
TIPTOP_GPU=4 TIPTOP_CONFIG=tiptop/config/tiptop_sim_r1pro.yml TIPTOP_PARTICLES=256 TIPTOP_MAX_PLANNING_TIME=40 \
  OmniGibson/omnigibson/tiptop/scripts/start_tiptop_server.sh
TIPTOP_GPU=4 TIPTOP_CONFIG=tiptop/config/tiptop_sim_r1pro_right.yml TIPTOP_PORT=8766 TIPTOP_RERUN_MODE=connect \
  TIPTOP_RERUN_URL=rerun+http://127.0.0.1:9876/proxy TIPTOP_PARTICLES=256 TIPTOP_MAX_PLANNING_TIME=40 \
  OmniGibson/omnigibson/tiptop/scripts/start_tiptop_server.sh
OMNIGIBSON_HEADLESS=1 ./b1k/bin/python -m omnigibson.tiptop.run live --embodiment r1pro --activity turning_on_radio \
  --stand-for radio_receiver.n.01_1 --goal "holding(radio_receiver.n.01_1);toggled_on(radio_receiver.n.01_1)" \
  --sequential --press-port 8766 --grasping-mode sticky --task "pick up the radio and press its button" \
  --host localhost --port 8765 --overview front --out-dir runs/radio_bimanual
    # add --knowledge onboard for Grounding DINO + SAM2 on the radio and its button instead of oracle masks and the button's pose
```

## Benchmark

`python -m omnigibson.tiptop.bench` runs a whole challenge task the way the challenge evaluates a policy
(`omnigibson.eval.eval`): the same public test instances (indices 0-9 are the reported ones; the evaluator's own
`load_task_instance` loads them), the same timeout (1.5x the mean human demonstration length, in env steps; every
hold, capture and plan counts), the same metrics (`TaskMetric`, `AgentMetric`), the same result JSON per rollout
under `<out>/json/`, plus `summary.json` with the mean q_score. Two things are stand-ins, and every result says so
(`bench.knowledge`, `bench.teleports`): the base is teleported to the pose `best_base_pose` picks for each round
instead of navigating, and with `--knowledge oracle` the planner is told the simulator's masks and button poses. A
number from this benchmark bounds the manipulation part of the pipeline; it is not a challenge score.

Outputs per instance: `videos/<task>_<instance>_0.mp4` is the whole episode in one video, every env step from the
first teleport to the end (the capture camera left, the overview and the wrist camera right), each frame stamped
with what the robot is doing (`teleport: stand for ...`, `round N: holding(...) [left arm]`, `release`) and the
step count over the timeout; after the episode the final state stays on screen for 3 s under the verdict
(`RESULT: SUCCESS q_score 1 1/1 satisfied`, or `RESULT: FAILED (...) q_score 0.688 11/16 satisfied` with the first
unsatisfied atoms). The file is a fragmented MP4, so it plays while the run is still going and after a killed run.
`<task>_<instance>_0/rNN_<arm>_<predicate>/` is one directory per planning round with the planner's exact request
(`obs.h5`, `capture.json`, the `rgb.png`/`depth.png`/`gt_masks.png` it saw), its answer (`server_response.json`,
`tiptop_plan.json`) and the execution's outcome (`live_result.json`); a failed round has the request and the error
only. They are for replaying a round against a planner, not videos: the bench writes no per-round clip. The result
JSON's `bench.video` names the video and `bench.rounds` lists the rounds with their env step. To re-plan a saved
round without the simulator (does a planner failure reproduce?):

```bash
./b1k/bin/python -m omnigibson.tiptop.replay runs/bench_radio_pass3/turning_on_radio_301_0/r02_right_toggled_on --port 8766 --repeats 3
```

The task strategies (`strategies.py`) order the rounds; the retry is the episode's and the same for every task
(`--rounds`, default 2: a goal gets two planning rounds, a pick two base poses at least 15 cm apart, a put-down is
done when the hand is empty; nothing else recovers, see "Kept out of the pipeline"). `turning_on_radio` picks the
radio up with the left hand and presses the switch with the right (the second planner on `--press-port` is
required: a held radio cannot slide away under the press, a free-standing one did, 20 cm across the glass table,
without toggling); `assembling_gift_baskets` does 16 transfers, each a pick at the table, a teleport to the basket with
the item in the gripper (OmniGibson moves a grasp-assisted object with the robot) and a place round that starts
holding it (`in_hand` in the request; the planner's `MoveHolding` -> `Place`). Baskets nearest the table come
first; within a kind, the items nearest the table's edge are tried first, `--attempts-per-item` of them per basket.

Results, 2026-09-09, public test instances 0-9, oracle knowledge, teleported base, sticky grasps (`runs/bench_radio_pass1`,
`runs/bench_radio_pass2`, `runs/bench_radio_pass3`, `runs/bench_radio_pass4`, `runs/bench_radio_pass5`; 0.7, 0.7, 0.6,
0.7 and 0.7 in five passes, so read the number as about 0.68 with the press plan as the source of variance):

| task | pass | mean q_score | successes | median env steps (of the timeout) | failure causes |
|---|---|---|---|---|---|
| turning_on_radio | 1 | 0.7 | 7/10 | 690 / 3224 | 2x no standing pose within 0.9 m, 1x press: no plan |
| turning_on_radio | 2 | 0.7 | 7/10 | 784 / 3224 | 1x no standing pose within 1.0 m, 2x press: no plan (one after a planner CUDA fault) |
| turning_on_radio | 3 | 0.6 | 6/10 | 861 / 3224 | 2x press: no plan even after a re-pick, 1x pick from 0.95 m never grasped, 1x press executed without toggling then re-pick planning failed |
| turning_on_radio | 4 | 0.7 | 7/10 | 747 / 3224 | 2x press: no plan x4 even after a put-down and re-pick, 1x press executed twice without toggling then no plan for the re-pick; videos end on the flip (marker green) |
| turning_on_radio | 5 | 0.7 | 7/10 | 751 / 3224 | 3x press: no plan x2 (the switch 0.57-0.63 m ahead); no executed press missed; the one retry policy, no put-down and re-pick |
| assembling_gift_baskets | 1 (2 instances, stopped) | 0.31 | 0/2 | 11200 / 39090 | a failed put-down left the item in the hand and blocked every later pick; items knocked to the floor were re-picked |
| assembling_gift_baskets | 2 | 0.6375 | 2/10 (16/16 twice; 15, 14, 13, 11, 11, 3, 3, 0 of 16) | 13600 / 39090 | 3 instances lost to one item no put-down plan could set down after a failed place (the hand stayed full); 45 stand attempts found no pose even at 1.1 m (a basket in a corner) |
| assembling_gift_baskets | 3 | 0.875 | 2/10 (16/16 twice; 15, 15, 15, 15, 14, 13, 11, 10 of 16) | 16000 / 39090 | the last-resort release ended the stuck-item loops (4 releases, 27 failed of 374 rounds); what remains is one or two items per instance: bows no base pose reaches (3 instances), a basket in a corner (1), places with no satisfying plan (2) |

The gift-basket run took 25-40 min of wall time per instance (250 rounds of capture, plan and execution over the
10 instances, 0 planner faults) for 110-600 s of simulated time; the challenge timeout was never reached. The pick
and the carry work: 147 pick rounds executed, 103 place rounds executed, 88 items placed. Pass 2 lost most of its
points to an object the hand could not put down again after a failed place; pass 3's last-resort release (open the
hand where it is) ended those loops and raised the mean to 0.875 (`runs/bench_baskets_pass3`, 27-33 min per instance).
What remains costs one or two items per instance: a bow at the far edge of the table that no base pose reaches
even at 1.1 m, a basket standing in a room corner, and places whose plan has no satisfying particles.

What fails is not the pick (26 of 30 instances ended with the radio in the hand) but the press with the grasp the
pick chose: the right arm has a plan when the switch ends up about 0.59 m ahead of the base facing right and none
when it is 6 cm further (a replayed request fails 3 of 3 times at 0.65 m, plans 3 of 3 at 0.59 m). The left hold
pose and the press-pose sampling are where the next gain is. Pass 5 (the press face inscribed in the button's
radius) confirms it: every press that planned toggled the switch, and the three failures had no press plan at
0.57-0.63 m. The planners had 0 CUDA faults in pass 3 (42 requests) and pass 5 against 4 in the ~60 requests
before the collision caches were sized (DEPLOYMENT item 11).

```bash
OMNIGIBSON_HEADLESS=1 ./b1k/bin/python -m omnigibson.tiptop.bench --task-name turning_on_radio \
    --instances 0 1 2 3 4 5 6 7 8 9 --knowledge oracle --grasping-mode sticky --host localhost --port 8765 \
    --press-port 8766 --overview front --out-dir runs/bench_radio_pass4
OMNIGIBSON_HEADLESS=1 ./b1k/bin/python -m omnigibson.tiptop.bench --task-name assembling_gift_baskets \
    --instances 0 1 2 3 4 5 6 7 8 9 --knowledge oracle --grasping-mode sticky --torso 1.2 -1.7 -0.9 0.0 \
    --host localhost --port 8765 --out-dir runs/bench_baskets_pass4
```

## CLI

`python -m omnigibson.tiptop.run <subcommand>`, inside the sim env, with `OMNIGIBSON_HEADLESS=1` (or unset for the
Isaac GUI):

| subcommand | does | own flags |
|---|---|---|
| `capture` | build the scene, write `obs.h5` + `capture.json` (offline input for `tiptop-h5`) | |
| `live` | capture, plan on a running `tiptop-server`, execute, score | `--host --port --press-host --press-port --plan-timeout --no-state-stream --sequential` |
| `replay` | build the scene, execute a `tiptop_plan.json` | `--plan --state-stream HOST:PORT` |

A whole challenge task on its test instances is `python -m omnigibson.tiptop.bench` (see "Benchmark").

Flags shared by all: `--embodiment franka|r1pro`, `--activity NAME` (+ `--activity-instance`, `--rooms`), scene
set-up `--place OBJ:SUPPORT[:DX,DY]`, `--spawn PRESET:SUPPORT[:DX,DY]`, `--scene-objects`; the base
`--stand-for [ITEM,...,]TARGET` | `--near FURNITURE [--side] [--standoff]` | `--robot-pose X Y YAW`; the posture
`--torso J1 J2 J3 J4`, `--no-look`; the capture `--camera head|wrist`, `--head-aperture`, `--seg-instance`;
what the planner is told `--knowledge oracle|onboard`; the goal `--goal "pred(a,b);..."` (BDDL names with
`--activity`), `--task`; execution
`--grasping-mode physical|assisted|sticky`, `--gripper-hold-steps`, `--finger-max-effort`, `--settle-steps`,
`--no-video`, `--overview shoulder|front` (where the third-person camera stands); `--scene capture.json` reuses an
earlier capture's settled object poses; `--not-load` drops object
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
  2026-09-04). `--stand-for` and the benchmark's strategies use it.

## Kept out of the pipeline

Nothing task-specific is written into the pipeline or a strategy: a strategy orders a task's goals and skips what
it cannot do, and every goal of every task gets the same retry (`--rounds`). What follows was tried or proposed
for one task and is kept here, with what it did, in case a task needs it later.

- **Put the radio down and pick it up again (turning_on_radio passes 3-4; removed 2026-09-09).** When the press
  found no plan twice, the strategy planned `ontop(radio, table)`, a fresh pick, and two more presses. It rescued
  one instance in 20 (pass 4, 303: the third grasp pressed the switch) at the price of a put-down, a pick and two
  press rounds per use, and the put-down itself misbehaved: the carried hull has no underside, so the planner
  lowered the radio into the table (308 ended on its back at the table's far edge), and in 3 of the 7 put-down
  rounds the fingers were already fully open before the plan's open event (the arm-switch bug fixed the same day,
  see History). To bring it back as a pipeline feature: a generic "re-grasp when the goal has no plan with this
  grasp" step in `Episode`, for every task, not a radio rule. The strategy's old form is in git (`ab3655ff2`).
- **A present point in the right hand's workspace (proposed, not applied).** `experimental.present_point`
  (`tiptop_sim_r1pro_right.yml`, `[0, -0.15, 1.2]`) is where a grasp should leave the switch facing. Every
  first-pick grasp in passes 1-4 presented it the same way: facing the robot's right (normal within 12 deg of -y)
  at 0.51-0.69 m ahead of the base. The presses with no plan were the ones at 0.62 m and beyond, the ones that
  planned at 0.60 m and nearer, so the reach edge is the distance the left arm's lift leaves the radio at, which
  the present point does not set (the left embodiment's plan-end pose, `q_home`, does). The rounds that faced the
  switch up or toward the torso were the removed cycle's re-picks.
- **Back off from the actual press pose (proposed, not applied).** When the switch flips early the executor stops
  the push and runs the planned back-off from where the plan is, not the arm; the jump is at most the press depth
  (4.5 cm) and it happens on success only.
- **A put-down height from the object's known extent (proposed, not applied).** The carried hull is the pick-time
  view, without an underside, so a place sets the object's seen part on the surface and its unseen part through
  it. A generic fix belongs in the planner's carry model (`tiptop/in_hand.py`: extend the hull down to the support
  plane the object rested on when it was picked); it would serve every place round, the baskets' included.
- **A pass with assisted grasping (not run).** The challenge's default `grasping_mode` is `assisted`; the passes
  used `sticky` (allowed). Same code, one flag.

## Known limits

- **Finding the radio's switch without oracle information does not work yet (2026-09-08).** The `turning_on_radio`
  radio has two red controls: the power switch on its black round speaker panel, and a knob on its top edge. In the
  task's placement the speaker panel faces away from the robot, so on the table only the knob is visible and every
  detector phrase tried scores it like the switch. The detector now accepts a button only inside a dark
  "black circle" context 2-6 times its size (`perception.grounding_dino.contexts`), which rejects the knob and keeps
  the switch when its panel is in view; and when no button is seen before the pick the planner presents the object's
  far side. But the radio hangs upright from a handle grasp and the two grasp families differ by a half turn, so the
  speaker face can be turned toward either arm, never up toward the head camera: in the hand it is edge-on
  (face cosine to the camera 0.04), the panel is a sliver, and the context test fails. The fix would be a look
  from the right wrist camera before the press (the face points at the right arm anyway), a precomputed right-arm
  look pose, a second shadow camera with the wrist optics, and a table fallback for close-up views. The demo therefore
  uses the oracle button pose (`button_hints` -> `gt_buttons`); `--knowledge onboard` remains an experiment.
- The base moves by teleport (`place_robot`), between rounds only: a carry is a pick round, a teleport with the
  object in the gripper (OmniGibson moves a grasp-assisted object with the robot) and a place round that starts
  holding it (`in_hand`). Nothing plans the base's path. While a hand holds something the capture keeps the arm
  where it is and the gripper closed (the swing out of view used to open it and drop the object, 2026-09-09).
- A planner can hit a CUDA fault (an illegal memory access in cuRobo, a few times in ~40 requests on the shared
  server) and is useless afterwards; it exits and its launcher relaunches it, and the benchmark waits for
  `/health` before each request (DEPLOYMENT item 11). Reachability: some instances put an object where no base
  pose within 0.9 m is free (a radio at the far side of the table with a sofa behind it); the benchmark widens
  the search to 1.1 m once (the torso leans) and then skips the object.
- Placement goes onto the top face of the container's convex hull with a 1 cm surface shrink, which is less than a
  wicker rim: an item can be set down on the rim (2026-09-05, from a stretched 0.8 m reach) and topple the basket.
- Flat objects (cheese slabs, bows) get few M2T2 grasps; the planner succeeds on them from close, orthogonal
  viewpoints and fails from others.
- Planner variance: the same capture can fail once with "Motion planning failed for 32/74 satisfying particles"
  and succeed next time (grasp sampling differs per call). Retry before debugging.
- Teleports (`--place`, `--stand-for`, `--torso`) are scaffolding the rules forbid during evaluation.
- Pressing needs an empty hand (`Push` requires `HandEmpty`), so "hold the radio and press its button" is two
  plans for two arms (`--press-port`), not one. The held object is wherever the grasp left it: nothing yet turns
  the button toward a camera or the free hand, and the right arm's model locks the left arm at its ready pose, so
  the left hand has to be home (the holding plan ends there) when the switch happens.
- Perception's table used to be whatever plane most objects' contact points touched; on the radio (its underside
  hidden, 8 cm above the glass) that was a tilted plane through its own face, which cut its hull to the top slab.
  The planner now takes only near-horizontal planes and allows contact points up to 10 cm above the table.

## Tests

`pytest OmniGibson/tests/test_tiptop_protocol.py OmniGibson/tests/test_tiptop_gt_masks.py
OmniGibson/tests/test_tiptop_knowledge.py` (no Isaac Sim): the msgpack-numpy wire format, request validation, plan
parsing and resampling, the H5 layout, the name helpers, the position-based perception pairing, the geometry
masks, the knowledge sources (what each attaches to a request, the button tracker), the task strategies against
a scripted episode, and the benchmark's summary.

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
- 2026-09-08: toggled_on goals (the radio): closed-gripper press, the two-hands hold-and-press with a second planner,
  grasps chosen to present the button, detector experiments on the button (documented limit).
- 2026-09-09: what the planner is told became one swappable source (`knowledge.py`); the carry (`in_hand`) and the
  challenge-style benchmark with task strategies; turning_on_radio 0.7 / 0.7 / 0.6 and assembling_gift_baskets
  0.6375 on the 10 public instances (oracle knowledge, teleported base); cuRobo collision caches sized at build
  time after four CUDA faults.
- 2026-09-09 (later): the benchmark video ended at the instant the goal predicate flipped, so a press looked
  unfinished and a failure looked like a success; now the final state stays on screen for 3 s under the verdict,
  every frame carries the step count, and the file is a fragmented MP4 that plays during and after a killed run. The
  bench writes no per-round clips (the episode video covers them). assembling_gift_baskets pass 3 (the last-resort
  release): 0.875 on the 10 public instances. turning_on_radio pass 4, recorded with this code: 0.7; its videos end
  on the flip with the marker green (OmniGibson's ToggledOn flips after 5 steps of fingertip overlap with the
  button's marker, before any visible push, and the task ends that step).
- 2026-09-09 (hygiene pass, after an audit): the press stop signal comes from the knowledge source (the oracle knows
  when a switch flips; the onboard source runs the press to its planned depth); the grasp-assist read handles
  physical grasping; the task's floor is looked up, not assumed; per-episode state (``arm``, ``teleports``, the
  grasp block) is declared and reset by ``begin_episode``; one ``recording`` block replaces three recorder
  lifecycles; ``bddl_category`` replaces four name parsers; the base-pose search and footprint thresholds are named
  constants; ``R1ProSim`` takes ``overview_view`` / ``look_arm`` as arguments; ``--restand``, ``predicate_holds``
  and the planner's ``goal_hints`` are gone; ``obs.h5`` holds the whole request and ``replay.py`` re-plans a round.
- 2026-09-09 (no task-specific recovery): the radio's put-down and re-pick cycle left the strategies; `Episode.pick`
  / `achieve` / `put_down` are the one retry policy (`--rounds`), the same for both tasks. Two generic fixes from
  the pass 4 dig: the planner's press face is the square inscribed in the button's radius (a corner of the old face
  was 35 mm off a 22 mm switch), and a plan carries `gripper_init`, so a hand that closed on nothing is opened
  before its next grasp; and the arm switch now hands each arm back its own gripper command (a left hand holding
  the radio was commanded open by the next left plan: in 3 of 7 put-down rounds the radio was out of the fingers
  before the plan's open event). turning_on_radio pass 5 with this code: 0.7 (7/10); every planned press toggled
  the switch, the three failures had no press plan at the reach edge (0.57-0.63 m); median 722 env steps per
  instance against 839 in pass 4.
