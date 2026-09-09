"""CLI for the TiPToP <-> OmniGibson bridge.

Run inside the sim env with OMNIGIBSON_HEADLESS=1 (or unset it for the GUI):

  python -m omnigibson.tiptop.run capture --out-dir runs/scene1
  python -m omnigibson.tiptop.run replay  --plan <tiptop_plan.json> --scene runs/scene1/capture.json --out-dir runs/replay
  python -m omnigibson.tiptop.run live    --host localhost --port 8765 --out-dir runs/live

`capture` writes obs.h5 (droid-sim-evals layout + the knowledge source's masks) for `tiptop-h5`; `live` talks to
`tiptop-server` and, unless --no-state-stream, mirrors the simulator into the server's Rerun view for the whole
session. A whole challenge task on its test instances is `python -m omnigibson.tiptop.bench`.

What the planner is told beyond the image is chosen with --knowledge (see knowledge.py): `oracle` reads the
simulator (masks, button poses; privileged, for development), `onboard` sends only what an agent knows.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("omnigibson.tiptop")

DEFAULT_TASK = "put the mug in the bowl"
DEFAULT_GOAL = "on(mug,bowl)"


def parse_goal(goal: str) -> list[dict]:
    """'on(mug,bowl);holding(apple)' -> [{'predicate': 'on', 'args': ['mug', 'bowl']}, ...]"""
    atoms = []
    for part in [p.strip() for p in goal.split(";") if p.strip()]:
        pred, rest = part.split("(", 1)
        atoms.append({"predicate": pred.strip(), "args": [a.strip() for a in rest.rstrip(")").split(",") if a.strip()]})
    return atoms


def atom_text(atoms: list[dict]) -> str:
    return "; ".join(f"{a['predicate']}({', '.join(a['args'])})" for a in atoms)


EXPECTED_ROBOT_TYPE = {"franka": "panda", "r1pro": "r1pro_left"}
EXPECTED_DOF = {"franka": 7, "r1pro": None}  # r1pro: set from the server's embodiment metadata in check_embodiment()
PRESS_ROBOT_TYPE = "r1pro_right"  # the second planner of a two-hands run


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", required=True, help="directory for outputs (created)")
    p.add_argument(
        "--embodiment",
        default="franka",
        choices=sorted(EXPECTED_ROBOT_TYPE),
        help="franka: tabletop Panda; r1pro: BEHAVIOR scene",
    )
    p.add_argument(
        "--objects", default="mug,bowl", help="comma-separated object presets (franka only; r1pro uses --spawn)"
    )
    r1 = p.add_argument_group("r1pro", "BEHAVIOR-1K R1Pro in a BEHAVIOR scene (navigation assumed done)")
    r1.add_argument("--scene-model", default="Rs_int")
    r1.add_argument(
        "--rooms",
        default=None,
        help="comma-separated rooms to load: load_room_types without --activity (default: the whole scene), room "
        "instances with it (default: the evaluator's list for the task; the house scene with all of them costs the "
        "client ~16 GB of RAM)",
    )
    r1.add_argument("--near", default=None, help="furniture name to stand next to, e.g. breakfast_table_skczfi_0")
    r1.add_argument(
        "--side",
        default="auto",
        choices=["auto", "-x", "+x", "-y", "+y"],
        help="which side of --near to stand on; write --side=-x",
    )
    r1.add_argument("--standoff", type=float, default=0.30, help="gap between robot footprint and the furniture (m)")
    r1.add_argument(
        "--robot-pose", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW"), help="explicit base pose"
    )
    r1.add_argument(
        "--spawn",
        action="append",
        default=[],
        metavar="PRESET:SUPPORT[:DX,DY]",
        help="drop a preset object onto furniture",
    )
    r1.add_argument("--scene-objects", default="", help="comma-separated names of existing scene objects to manipulate")
    r1.add_argument("--camera", default="head", choices=["head", "wrist"])
    r1.add_argument(
        "--activity",
        default=None,
        help="load a BEHAVIOR challenge task instead of spawning objects (e.g. assembling_gift_baskets); "
        "--goal then uses BDDL names, e.g. 'inside(candle.n.01_1,wicker_basket.n.01_1)'",
    )
    r1.add_argument("--activity-instance", type=int, default=0, help="training instance id (0 = the template)")
    r1.add_argument(
        "--place",
        action="append",
        default=[],
        metavar="OBJ:SUPPORT[:DX,DY]",
        help="teleport a scene/task object onto a support before the episode (test stand-in for a carry), "
        "e.g. wicker_basket.n.01_2:table.n.02_1:0.13,0.31",
    )
    r1.add_argument(
        "--stand-for",
        default=None,
        metavar="[ITEM,...,]TARGET",
        help="choose the base pose once so every ITEM and the TARGET are in the left arm's reach; a single name "
        "for a one-object task (navigation stand-in; alternative to --near / --robot-pose)",
    )
    p.add_argument(
        "--knowledge",
        default="oracle",
        choices=["oracle", "onboard"],
        help="what the planner is told beyond the image: 'oracle' reads the simulator (instance masks, button poses; "
        "privileged, for development), 'onboard' sends only the task's object names, the goal and the gripper "
        "state, and the planner's detector does the rest",
    )
    r1.add_argument(
        "--seg-instance",
        action="store_true",
        help="render Isaac instance segmentation on the capture camera (oracle masks then come from it instead of "
        "the objects' geometry); the annotator segfaults in large BEHAVIOR scenes",
    )
    r1.add_argument(
        "--no-look", action="store_true", help="capture in the ready posture instead of swinging the arm out of view"
    )
    r1.add_argument(
        "--not-load",
        default="ceilings",
        help="comma-separated object categories left out of the scene (e.g. straight_chair)",
    )
    r1.add_argument(
        "--head-aperture",
        type=float,
        default=None,
        help="head camera horizontal aperture (mm); default: the challenge evaluator's setting (40)",
    )
    r1.add_argument(
        "--torso",
        type=float,
        nargs=4,
        default=None,
        metavar=("J1", "J2", "J3", "J4"),
        help="start the torso here instead of the embodiment's q_home (rad; lower and more forward puts the head "
        "camera closer to the table, so the robot can stand nearer)",
    )
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument(
        "--goal", default=DEFAULT_GOAL, help="goal atoms, e.g. 'on(mug,bowl)'; used for gt_atoms and success checks"
    )
    p.add_argument("--grasping-mode", default="physical", choices=["physical", "assisted", "sticky"])
    p.add_argument("--settle-steps", type=int, default=90, help="env steps to let objects settle after reset")
    p.add_argument("--no-video", action="store_true")
    p.add_argument(
        "--overview",
        choices=["shoulder", "front"],
        default="shoulder",
        help="third-person camera in the video and the mirror: over the left shoulder at the workspace, or ahead and "
        "to the right looking back at both hands (the two-hands demo)",
    )
    p.add_argument("--gripper-hold-steps", type=int, default=25)
    p.add_argument("--scene", default=None, help="capture.json of an earlier capture: reuse its settled object poses")
    p.add_argument(
        "--finger-max-effort",
        type=float,
        default=None,
        help="finger drive force in N (USD default 20; real Franka hand 70)",
    )


def add_planner_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--plan-timeout", type=float, default=900.0)
    p.add_argument(
        "--no-state-stream", action="store_true", help="do not mirror the simulator into the server's Rerun view"
    )
    p.add_argument("--press-host", default=None, help="host of the planner for the other arm (default --host)")
    p.add_argument(
        "--press-port",
        type=int,
        default=None,
        help="a second tiptop-server on this port plans the other arm (r1pro_right) for rounds whose goals are all "
        "toggled_on(...): the first arm keeps holding what it picked up",
    )


def parse_spawns(specs, flag: str = "--spawn") -> list[tuple[str, str, float, float]]:
    """'NAME:SUPPORT[:DX,DY]' -> (name, support, dx, dy) for --spawn and --place; validated before Isaac Sim starts."""
    spawns = []
    for spec in specs:
        parts = spec.split(":")
        try:
            if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
                raise ValueError
            dx, dy = (float(v) for v in parts[2].split(",")) if len(parts) == 3 else (0.0, 0.0)
        except ValueError:
            raise SystemExit(
                f"{flag} {spec!r}: expected NAME:SUPPORT or NAME:SUPPORT:DX,DY (e.g. mug:table_x_0:-0.2,0.15)"
            )
        spawns.append((parts[0], parts[1], dx, dy))
    return spawns


def build_r1pro_sim(args, embodiment: dict | None, max_steps: int = 10**8):
    """The R1Pro in the scene the arguments describe, in the planner's posture, standing where they say."""
    from omnigibson.tiptop.r1pro import HEAD_APERTURE_MM, R1ProSim, make_r1pro_env_config

    spawns, places = parse_spawns(args.spawn), parse_spawns(args.place, flag="--place")
    scene_model, room_instances = args.scene_model, None
    if args.activity:
        from omnigibson.tiptop.r1pro import challenge_task_info

        scene_model, room_instances = challenge_task_info(args.activity)
        if args.rooms:  # fewer rooms than the evaluator loads: a smaller scene for a 30 GB machine
            room_instances = [r for r in args.rooms.split(",") if r]
        log.info(f"challenge task {args.activity}: scene {scene_model}, rooms {room_instances}")
    cfg = make_r1pro_env_config(
        scene_model=scene_model,
        load_room_types=None if args.activity else [r for r in (args.rooms or "").split(",") if r],
        spawn_presets=[sp[0] for sp in spawns],
        grasping_mode=args.grasping_mode,
        camera=args.camera,
        head_aperture_mm=HEAD_APERTURE_MM if args.head_aperture is None else args.head_aperture,
        not_load_object_categories=[c for c in args.not_load.split(",") if c],
        activity=args.activity,
        activity_instance_id=args.activity_instance,
        load_room_instances=room_instances,
        segmentation=args.seg_instance,  # the annotator is opt-in; oracle masks come from geometry
        max_steps=max_steps,
    )
    sim = R1ProSim(cfg, camera=args.camera)
    sim.overview_view = args.overview
    if args.activity:
        sim.track_task_objects()
    # furniture the run names is drawn in the Rerun mirror, so the view has a table under the objects
    sim.track_context(*{support for _, support, _, _ in places + spawns}, args.near)
    for name, support, dx, dy in places:
        sim.place_on(name, support, dx, dy)  # settles during the holds below
    if args.no_look:
        sim.look_arm = None
    apply_embodiment_posture(sim, args, embodiment)
    if args.robot_pose:
        sim.place_robot(*args.robot_pose)
    elif args.stand_for:
        sim.place_robot_for(*[n for n in args.stand_for.split(",") if n], ignore_names=[sp[0] for sp in spawns])
    elif args.near:
        sim.place_robot_near(args.near, side=args.side, standoff=args.standoff, ignore_names=[sp[0] for sp in spawns])
    else:
        log.info("no --stand-for / --near / --robot-pose: the robot stays where the scene put it")
    for preset, support, dx, dy in spawns:
        sim.place_on(preset, support, dx, dy)
    sim.track(*[n for n in args.scene_objects.split(",") if n])
    if args.scene:
        with open(args.scene) as f:
            poses = json.load(f)["extras"]["object_poses_world"]
        sim.apply_object_poses({k: v for k, v in poses.items() if k in sim.objects})
        log.info(f"applied object poses from {args.scene}")
    if args.finger_max_effort is not None:
        sim.set_finger_max_effort(args.finger_max_effort)
    sim.hold(args.settle_steps, sim.OPEN)
    if args.activity:
        sim.mark_goal_initial()
        log.info(f"task goal at start: {sim.goal_status()}")
    return sim


def apply_embodiment_posture(sim, args, embodiment: dict | None) -> dict:
    """Hold the planner's locked joints and go to its home pose (``--torso`` overrides the torso entries); the
    embodiment comes from the server metadata or a plan's provenance, else from the submodule's meta file."""
    from omnigibson.tiptop.r1pro import ROBOT_TYPE, load_embodiment_meta

    if embodiment is None:
        embodiment = load_embodiment_meta()
        log.info(f"posture from {embodiment['robot_type']} meta file (no server metadata / plan provenance)")
    else:
        if embodiment.get("robot_type") != ROBOT_TYPE:
            raise ValueError(f"embodiment {embodiment.get('robot_type')!r} is not {ROBOT_TYPE!r}")
        try:  # best effort: warn when the submodule's generated meta drifted from what the server/plan carries
            local = load_embodiment_meta(ROBOT_TYPE)
            if (
                local["locked_joints"] != embodiment["locked_joints"]
                or local["joint_names"] != embodiment["joint_names"]
            ):
                log.warning(
                    "server/plan embodiment differs from the local tiptop submodule's meta file; using the former"
                )
        except FileNotFoundError:
            pass
    q_home = [float(v) for v in embodiment["q_home"]]
    if args.torso:  # the planner moves the torso anyway; this only changes where the episode (and the capture) starts
        for joint, value in zip(embodiment["torso_joints"], args.torso):
            q_home[embodiment["joint_names"].index(joint)] = value
    # the posture decides how close the head camera can see, so it comes before the base pose is chosen
    sim.apply_posture(
        embodiment["locked_joints"], q_home, settle_steps=args.settle_steps, joint_names=embodiment["joint_names"]
    )
    return embodiment


def build_sim(args, embodiment: dict | None = None):
    if args.embodiment == "r1pro":
        return build_r1pro_sim(args, embodiment)
    from omnigibson.tiptop.scene import TiptopSim, make_env_config

    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    cfg = make_env_config(objects=objects, grasping_mode=args.grasping_mode)
    sim = TiptopSim(cfg)
    if args.finger_max_effort is not None:
        sim.set_finger_max_effort(args.finger_max_effort)
    if args.scene:
        with open(args.scene) as f:
            poses = json.load(f)["extras"]["object_poses_world"]
        sim.apply_object_poses(poses)
        log.info(f"applied object poses from {args.scene}")
    sim.hold(args.settle_steps, sim.OPEN)
    return sim


def do_capture(sim, args, out_dir: Path, atoms: list[dict], knowledge, floor: bool = False) -> tuple[dict, dict]:
    """Render, attach what the knowledge source knows about ``atoms``, validate, and save the observation."""
    import imageio

    from omnigibson.tiptop.knowledge import GoalNotVisible
    from omnigibson.tiptop.protocol import save_observation_h5

    request, extras = sim.capture(args.task)
    try:
        known = knowledge.describe(atoms, request, extras, floor=floor)
    except GoalNotVisible:
        imageio.imwrite(out_dir / "rgb_failed.png", sim.last_capture_rgb)  # what the camera saw
        raise
    known.attach(request)
    report = sim.validate_capture(request, extras)
    for problem in report["problems"]:
        log.warning(f"capture validation: {problem}")
    save_observation_h5(out_dir / "obs.h5", request, extras["cam_pos_base"], extras["cam_quat_wxyz_ros"])
    imageio.imwrite(out_dir / "rgb.png", request["rgb"])
    depth_vis = np.clip(request["depth"] / 2.0, 0, 1)
    imageio.imwrite(out_dir / "depth.png", (depth_vis * 255).astype(np.uint8))
    if known.masks is not None:
        seg_vis = np.zeros_like(request["rgb"])
        colors = [(255, 80, 80), (80, 200, 255), (120, 255, 120), (255, 220, 80)]
        for i, mask in enumerate(known.masks):
            seg_vis[mask.astype(bool)] = colors[i % len(colors)]
        imageio.imwrite(out_dir / "gt_masks.png", seg_vis)
    meta = {
        "task": args.task,
        "goal_atoms": known.atoms,
        "knowledge": {**knowledge.report(), **known.summary()},
        "intrinsics": request["intrinsics"].tolist(),
        "world_from_cam": request["world_from_cam"].tolist(),
        "q_init": request["q_init"].tolist(),
        "validation": report,
        "extras": {k: v for k, v in extras.items() if k not in ("seg_instance",)},
    }
    with open(out_dir / "capture.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"capture saved to {out_dir} (validation problems: {report['problems'] or 'none'})")
    return request, extras


def setup_logging() -> None:
    """Give the bridge's logger its own INFO stream: OmniGibson's handler on the "omnigibson" logger hides INFO."""
    tiptop_log = logging.getLogger("omnigibson.tiptop")
    tiptop_log.setLevel(logging.INFO)
    tiptop_log.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    tiptop_log.addHandler(handler)


def open_state_stream(hostport: str | None, sim):
    """Mirror the simulator into the server's Rerun view for the rest of the session; None if disabled."""
    if not hostport:
        return None
    from omnigibson.tiptop.client import SimStateStream

    host, _, port = hostport.rpartition(":")
    stream = SimStateStream(host or "localhost", int(port) if port else 8765)
    stream.attach(sim)  # keeps retrying on its own when the server is not there yet
    return stream


def connect_planners(args):
    """The planning server(s) of a live run, checked against the embodiments this client executes on, before Isaac
    Sim starts: (client, its metadata, press client or None, its metadata or None)."""
    from omnigibson.tiptop.client import TiptopClient

    client = TiptopClient(
        args.host,
        args.port,
        expected_robot_type=EXPECTED_ROBOT_TYPE[args.embodiment],
        expected_dof=EXPECTED_DOF[args.embodiment],
    )
    client.wait_for_server()
    metadata = client.fetch_metadata()
    client.check_embodiment()  # fail here, before Isaac Sim starts, if the server plans for another robot
    press_client = press_meta = None
    if getattr(args, "press_port", None):
        press_client = TiptopClient(
            args.press_host or args.host, args.press_port, expected_robot_type=PRESS_ROBOT_TYPE, expected_dof=None
        )
        press_client.wait_for_server()
        press_meta = press_client.fetch_metadata()
        press_client.check_embodiment()
    return client, metadata, press_client, press_meta


def perception_report(request: dict, extras: dict, response: dict) -> dict:
    """Pair what the server perceived with the simulator's objects by position, and say what the goal acts on.

    Perception numbers instances by box size, the simulator by task instance, so the names agree only by
    chance (2026-09-04: the server's "candle_2" was the simulator's candle_1, one candle over). Logged per round and
    saved with the result; a goal object without a simulated partner is the first thing to look at.
    """
    from omnigibson.tiptop.protocol import MATCH_MAX_DIST, match_objects

    perceived = response.get("objects") or {}
    simulated = {name: pose["aabb_center"] for name, pose in extras["object_poses_base"].items()}
    # a hull seen from one side is centred above the object's centre, by up to half its size
    tolerance = {
        name: max(MATCH_MAX_DIST, 0.5 * float(np.ptp(np.asarray(pose["aabb_corners"], dtype=np.float64), axis=0).max()))
        for name, pose in extras["object_poses_base"].items()
    }
    match = match_objects({label: info["position"] for label, info in perceived.items()}, simulated, tolerance)
    goal_args = {a for atom in request.get("gt_atoms") or [] for a in atom["args"]}
    for label in sorted(perceived, key=lambda name: (name not in goal_args, name)):
        info, m = perceived[label], match[label]
        role = (
            "goal"
            if label in goal_args
            else (
                "in hand"
                if info.get("in_hand")
                else "held"
                if info.get("held")
                else "movable"
                if info["movable"]
                else "surface"
            )
        )
        head = f"perceived {label!r} ({role}, {info['grasps']} grasps)"
        if m["sim"] is None:
            nearest = f"{100 * m['dist']:.0f} cm" if m["dist"] is not None else "nothing tracked"
            (log.warning if role == "goal" else log.info)(
                f"{head}: no simulated object within its allowance (nearest {nearest}): false detection or misplaced hull"
            )
        else:
            log.info(f"{head} = simulated {m['sim']} ({100 * m['dist']:.1f} cm off)")
    return {label: dict(perceived[label], **match[label]) for label in perceived}


def live_round(
    sim,
    args,
    client,
    out_dir: Path,
    atoms: list[dict],
    knowledge,
    floor: bool = False,
    score: bool = True,
    record: bool = True,
) -> dict:
    """Capture, ask the server for a plan for these atoms, save it and execute it (``score``: evaluate the task's
    goal afterwards; a benchmark scores once at the end instead, the whole goal costs 46 s on the gift-basket task;
    ``record``: write this round's own clip, ``<out_dir>/live.mp4``; a driver recording the whole run passes False)."""
    from omnigibson.tiptop.client import TiptopPlanningError

    request, extras = do_capture(sim, args, out_dir, atoms, knowledge, floor=floor)
    try:
        response = client.plan(request, timeout_s=args.plan_timeout)
    except TiptopPlanningError:
        if client.last_response and client.last_response.get("objects"):  # what perception made of the frame
            perception_report(request, extras, client.last_response)
        raise
    with open(out_dir / "server_response.json", "w") as f:
        json.dump({k: v for k, v in response.items() if k != "plan"}, f, indent=2)
    with open(out_dir / "tiptop_plan.json", "w") as f:
        json.dump(
            {
                "version": response["plan"]["version"],
                "embodiment": response.get("embodiment") or client.metadata.get("embodiment"),
                "q_init": response["plan"]["q_init"].tolist(),
                "steps": [
                    dict(
                        s,
                        positions=s["positions"].tolist(),
                        velocities=None if s["velocities"] is None else s["velocities"].tolist(),
                    )
                    if s["type"] == "trajectory"
                    else s
                    for s in response["plan"]["steps"]
                ],
            },
            f,
        )
    log.info(
        f"server planned in {response.get('server_timing', {}).get('infer_ms', 0) / 1000:.1f}s (round trip {response['client_roundtrip_s']:.1f}s), save_dir={response.get('save_dir')}"
    )
    match = perception_report(request, extras, response)
    knowledge.learned(response)
    return do_execute(
        sim,
        args,
        out_dir,
        response["plan"],
        tag="live",
        atoms=atoms,
        knowledge=knowledge,
        extra={"perception": match},
        score=score,
        record=record,
    )


def do_execute(
    sim,
    args,
    out_dir: Path,
    plan: dict,
    tag: str,
    atoms: list[dict] | None = None,
    knowledge=None,
    extra: dict | None = None,
    score: bool = True,
    record: bool = True,
) -> dict:
    """Execute a plan and check the goal: the task's own with --activity (``atoms``, default --goal, then names the
    objects whose AABBs are logged), else every --goal atom; ``score=False`` skips the task's goal evaluation and
    reports the goal objects' poses only. ``record`` writes the execution as ``<out_dir>/<tag>.mp4`` (unless
    --no-video). ``extra`` is saved with the result."""
    from omnigibson.tiptop.executor import PlanExecutor, VideoRecorder, check_success
    from omnigibson.tiptop.protocol import plan_summary

    atoms = parse_goal(args.goal) if atoms is None else list(atoms)
    log.info(f"executing plan: {plan_summary(plan)}")
    video = VideoRecorder(out_dir / f"{tag}.mp4") if record and not args.no_video else None
    # a press ends as soon as the simulator's toggle flips (the plan pushes a little past the surface)
    press_targets = [atom["args"][0] for atom in atoms if atom["predicate"] == "toggled_on"] if args.activity else []
    press_done = (lambda: all(sim.toggled(name) for name in press_targets)) if press_targets else None
    executor = PlanExecutor(sim, gripper_hold_steps=args.gripper_hold_steps, press_done=press_done)
    if video is not None:
        sim.recorders.append(video)
    if press_targets and args.grasping_mode != "physical":
        sim.block_grasping(getattr(sim, "arm", None))  # the press closes the gripper; it must not grasp the object
    try:
        stats = executor.execute(plan)
        note_hands(sim, atoms, executor, knowledge)
    finally:
        if press_targets and args.grasping_mode != "physical":
            sim.unblock_grasping()
        if video is not None:
            sim.recorders.remove(video)
            video.close()
    if press_targets:
        stats["buttons"] = {name: sim.press_state(name) for name in press_targets}
        log.info(f"buttons after the plan: {stats['buttons']}")
    if args.activity:
        success = sim.goal_status() if score else {"success": None, "scored": False}
        success["all"] = success["success"]
        # where the goal objects ended up relative to their targets (BDDL names)
        success["poses"] = {}
        for atom in atoms:
            for name in atom["args"]:
                try:
                    obj = sim.scene_object(name)
                    lo, hi = [v.cpu().numpy().round(3).tolist() for v in obj.aabb]
                    success["poses"][name] = {"aabb": [lo, hi]}
                except Exception:  # noqa: BLE001 - diagnostics only
                    pass
        log.info(f"goal object AABBs: {success['poses']}")
    else:
        success = check_success(sim, parse_goal(args.goal))
    result = {
        "plan_summary": plan_summary(plan),
        "execution": stats,
        "success": success,
        "held": dict(sim.held_objects),
        "final_object_poses_world": sim.object_poses_world(),
        **(extra or {}),
    }
    with open(out_dir / f"{tag}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"success check: {json.dumps(success)}")
    return result


def note_hands(sim, atoms: list[dict], executor, knowledge) -> None:
    """Update what the hands hold after a plan: from the robot's grasp assist when it has one (sticky / assisted
    grasping), else from the plan's own goals (a holding goal took the object, a placement let it go). A newly
    taken object is reported to the knowledge source (a button on it moves with the gripper from now on)."""
    arm = getattr(sim, "arm", None) or sim.robot.default_arm
    before = dict(sim.held_objects)
    grasped = sim.grasped_labels()
    if grasped is not None:
        sim.held_objects = grasped
    else:
        for atom in atoms:
            if atom["predicate"] == "holding" and executor.close_eef is not None:
                sim.held_objects[sim.tracked_label(atom["args"][0])] = arm
            elif atom["predicate"] in ("on", "inside", "ontop", "nextto") and len(atom["args"]) == 2:
                sim.held_objects.pop(sim.tracked_label(atom["args"][0]), None)
    for label, holder in sim.held_objects.items():
        if label not in before and knowledge is not None and executor.close_eef is not None:
            knowledge.picked(label, holder, executor.close_eef)
    if sim.held_objects != before:
        log.info(f"hands now hold {sim.held_objects or 'nothing'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_cap = sub.add_parser("capture", help="build the scene and write obs.h5 + capture.json (offline tiptop-h5 input)")
    add_common(p_cap)
    p_rep = sub.add_parser("replay", help="build the scene and execute a tiptop_plan.json")
    add_common(p_rep)
    p_rep.add_argument("--plan", required=True)
    p_rep.add_argument(
        "--state-stream", default=None, metavar="HOST:PORT", help="mirror the sim into a tiptop-server's Rerun view"
    )
    p_live = sub.add_parser("live", help="capture, ask a running tiptop-server for a plan, execute it")
    add_common(p_live)
    add_planner_args(p_live)
    p_live.add_argument(
        "--sequential",
        action="store_true",
        help="one capture/plan/execute round per goal atom from where the robot stands, instead of one plan for the "
        "whole goal",
    )
    p_live.add_argument(
        "--restand",
        action="store_true",
        help="with --sequential and --activity: teleport the base to a reachable pose before every round",
    )
    args = parser.parse_args(argv)
    setup_logging()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import omnigibson as og
    from omnigibson.tiptop.knowledge import make_knowledge
    from omnigibson.tiptop.protocol import load_plan_json

    exit_code = 0
    stream = None
    try:
        t0 = time.time()
        client = press_client = press_meta = None
        if args.cmd == "live":
            # the server's embodiment metadata (locked posture, home pose) shapes the scene, so fetch it first
            client, metadata, press_client, press_meta = connect_planners(args)
            sim = build_sim(args, embodiment=metadata.get("embodiment"))
            if not args.no_state_stream:
                stream = open_state_stream(f"{args.host}:{args.port}", sim)
        elif args.cmd == "replay":
            with open(args.plan) as f:
                plan_json = json.load(f)
            sim = build_sim(args, embodiment=plan_json.get("embodiment"))  # provenance saved by `live`
            stream = open_state_stream(args.state_stream, sim)
        else:
            sim = build_sim(args)
        log.info(f"scene ready in {time.time() - t0:.1f}s (sim dt {sim.dt:.4f}s)")
        atoms_all = parse_goal(args.goal)
        knowledge = make_knowledge(args.knowledge, sim, atoms_all)
        if args.cmd == "capture":
            do_capture(sim, args, out_dir, atoms_all, knowledge)
        elif args.cmd == "replay":
            plan = load_plan_json(args.plan)
            if args.embodiment == "r1pro" and not plan_json.get("embodiment"):
                log.warning("plan has no embodiment provenance; assuming it was made for the local tiptop embodiment")
            do_execute(sim, args, out_dir, plan, tag="replay", knowledge=knowledge)
        elif args.cmd == "live":
            rounds = [[atom] for atom in atoms_all] if args.sequential else [atoms_all]
            outcomes = []
            full = None
            if args.sequential and not args.no_video:  # the whole run, rounds and the holds between them
                from omnigibson.tiptop.executor import VideoRecorder

                full = VideoRecorder(out_dir / "full.mp4")
                sim.recorders.append(full)
            for i, atoms in enumerate(rounds):
                round_dir = out_dir / f"round_{i:02d}" if args.sequential else out_dir
                round_dir.mkdir(parents=True, exist_ok=True)
                sim.video_caption = f"round {i}: {atom_text(atoms)}"
                try:
                    if args.sequential and args.restand and args.activity and len(atoms[0]["args"]) == 2:
                        sim.place_robot_for(*atoms[0]["args"])  # navigation stand-in for this transfer
                        sim.hold(args.settle_steps, sim.OPEN)
                    round_client = client
                    if press_client is not None and all(atom["predicate"] == "toggled_on" for atom in atoms):
                        sim.adopt_embodiment(press_meta["embodiment"])  # the other arm presses; this one keeps holding
                        round_client = press_client
                        sim.video_caption += f"  [{sim.arm} arm presses, {sim.other_arm} holds]"
                    outcomes.append(live_round(sim, args, round_client, round_dir, atoms, knowledge))
                except Exception as e:
                    if not args.sequential:
                        raise
                    log.exception(f"round {i} {atoms} failed")
                    outcomes.append({"error": f"{type(e).__name__}: {e}"})
                if args.sequential:
                    log.info(f"round {i} {atoms}: {outcomes[-1].get('success', outcomes[-1].get('error'))}")
            if full is not None:
                sim.video_caption = "done: " + str(sim.goal_status().get("success")) if args.activity else "done"
                sim.hold(30, sim.last_gripper)  # a second of the final state closes the video
                sim.recorders.remove(full)
                full.close()
            if args.sequential:
                summary = {"rounds": outcomes, "knowledge": knowledge.report()}
                if args.activity:
                    summary["task_goal"] = sim.goal_status()
                    log.info(f"task goal after {len(rounds)} rounds: {summary['task_goal']}")
                with open(out_dir / "sequential_summary.json", "w") as f:
                    json.dump(summary, f, indent=2, default=str)
    except Exception:
        log.exception("run failed")
        exit_code = 1
    finally:
        if stream is not None:
            stream.close()
        if og.app is not None:  # og.shutdown() exits with status 0 when Isaac Sim was never launched
            og.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
