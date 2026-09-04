"""CLI for the TiPToP <-> OmniGibson bridge.

Run inside the `behavior` conda env with OMNIGIBSON_HEADLESS=1 (or unset it for the GUI):

  python -m omnigibson.tiptop.run capture --out-dir runs/scene1
  python -m omnigibson.tiptop.run replay  --plan <tiptop_plan.json> --scene runs/scene1/capture.json --out-dir runs/replay
  python -m omnigibson.tiptop.run live    --host localhost --port 8765 --out-dir runs/live

`capture` writes obs.h5 (droid-sim-evals layout + ground-truth masks) for `tiptop-h5`; `live` talks to `tiptop-server`.
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


EXPECTED_ROBOT_TYPE = {"franka": "panda", "r1pro": "r1pro_left"}
EXPECTED_DOF = {"franka": 7, "r1pro": None}  # r1pro: set from the server's embodiment metadata in check_embodiment()


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
    r1.add_argument("--activity-instance", type=int, default=0, help="task instance id (0 = the template)")
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
        metavar="ITEM,TARGET",
        help="choose the base pose automatically so ITEM and TARGET are both in the left arm's reach "
        "(navigation stand-in; alternative to --near / --robot-pose)",
    )
    p.add_argument(
        "--no-gt",
        action="store_true",
        help="competition-style perception: send just the task's object names and goal atoms, no ground-truth masks "
        "(the server runs its detector + SAM2 on the image)",
    )
    r1.add_argument(
        "--seg-instance",
        action="store_true",
        help="render Isaac instance segmentation on the capture camera; ground-truth masks otherwise come from object "
        "geometry; the annotator segfaults in large BEHAVIOR scenes",
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
        "--head-aperture", type=float, default=40.0, help="head camera horizontal aperture (mm); eval uses 40"
    )
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument(
        "--goal", default=DEFAULT_GOAL, help="goal atoms, e.g. 'on(mug,bowl)'; used for gt_atoms and success checks"
    )
    p.add_argument("--grasping-mode", default="physical", choices=["physical", "assisted", "sticky"])
    p.add_argument("--settle-steps", type=int, default=90, help="env steps to let objects settle after reset")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--gripper-hold-steps", type=int, default=25)
    p.add_argument("--scene", default=None, help="capture.json of an earlier capture: reuse its settled object poses")
    p.add_argument(
        "--finger-max-effort",
        type=float,
        default=None,
        help="finger drive force in N (USD default 20; real Franka hand 70)",
    )


def parse_spawns(specs) -> list[tuple[str, str, float, float]]:
    """'PRESET:SUPPORT[:DX,DY]' -> (preset, support, dx, dy); validated before Isaac Sim starts."""
    spawns = []
    for spec in specs:
        parts = spec.split(":")
        try:
            if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
                raise ValueError
            dx, dy = (float(v) for v in parts[2].split(",")) if len(parts) == 3 else (0.0, 0.0)
        except ValueError:
            raise SystemExit(
                f"--spawn {spec!r}: expected PRESET:SUPPORT or PRESET:SUPPORT:DX,DY (e.g. mug:table_x_0:-0.2,0.15)"
            )
        spawns.append((parts[0], parts[1], dx, dy))
    return spawns


def build_r1pro_sim(args, embodiment: dict | None):
    from omnigibson.tiptop.r1pro import ROBOT_TYPE, R1ProSim, load_embodiment_meta, make_r1pro_env_config

    spawns = parse_spawns(args.spawn)
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
        head_aperture_mm=args.head_aperture,
        not_load_object_categories=[c for c in args.not_load.split(",") if c],
        activity=args.activity,
        activity_instance_id=args.activity_instance,
        load_room_instances=room_instances,
        segmentation=args.seg_instance,  # the annotator is opt-in; masks come from geometry
    )
    sim = R1ProSim(cfg, camera=args.camera)
    if args.activity:
        sim.track_task_objects()
    for spec in args.place:
        parts = spec.split(":")
        dx, dy = (float(v) for v in parts[2].split(",")) if len(parts) > 2 else (0.0, 0.0)
        sim.place_on(parts[0], parts[1], dx, dy)  # settles during the posture hold below
    if args.no_look:
        sim.look_arm = None
    if args.robot_pose:
        sim.place_robot(*args.robot_pose)
    elif args.stand_for:
        item, target = args.stand_for.split(",")
        sim.place_robot_for(item, target, ignore_names=[sp[0] for sp in spawns])
    elif args.near:
        sim.place_robot_near(args.near, side=args.side, standoff=args.standoff, ignore_names=[sp[0] for sp in spawns])
    else:
        log.warning("no --near / --robot-pose: the robot stays where the scene put it")
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
    sim.apply_posture(
        embodiment["locked_joints"],
        embodiment["q_home"],
        settle_steps=args.settle_steps,
        joint_names=embodiment["joint_names"],
    )
    sim.hold(args.settle_steps, sim.OPEN)
    if args.activity:
        sim.mark_goal_initial()
        log.info(f"task goal at start: {sim.goal_status()}")
    return sim


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


def do_capture(
    sim, args, out_dir: Path, atoms: list[dict] | None = None, hints: dict | None = None
) -> tuple[dict, dict]:
    import imageio

    from omnigibson.tiptop.protocol import save_observation_h5

    atoms = parse_goal(args.goal) if atoms is None else list(atoms)
    no_gt = args.no_gt
    if args.activity:
        # BDDL names -> request labels; --no-gt asks the detector for categories (any candle will do)
        labels, atoms = sim.tiptop_goal(atoms, category_level=no_gt)
    else:
        labels = sorted({a for atom in atoms for a in atom["args"] if a in sim.objects})
    if no_gt:  # no segmentation rendered: send the names and the goal, no masks
        request, extras = sim.capture(args.task)
        request["gt_labels"], request["gt_atoms"] = list(labels), list(atoms)
        if hints:
            request["goal_hints"] = {k: [float(v) for v in vals] for k, vals in hints.items()}
    else:
        try:
            request, extras = sim.capture(args.task, gt_labels=labels, gt_atoms=atoms)
        except ValueError:
            if sim.last_capture_rgb is not None:  # what the camera saw when a goal object was missing
                imageio.imwrite(out_dir / "rgb_failed.png", sim.last_capture_rgb)
            raise
    report = sim.validate_capture(request, extras)
    for problem in report["problems"]:
        log.warning(f"capture validation: {problem}")
    save_observation_h5(out_dir / "obs.h5", request, extras["cam_pos_base"], extras["cam_quat_wxyz_ros"])
    imageio.imwrite(out_dir / "rgb.png", request["rgb"])
    depth_vis = np.clip(request["depth"] / 2.0, 0, 1)
    imageio.imwrite(out_dir / "depth.png", (depth_vis * 255).astype(np.uint8))
    seg_vis = np.zeros_like(request["rgb"])
    if "gt_masks" in request:
        colors = [(255, 80, 80), (80, 200, 255), (120, 255, 120), (255, 220, 80)]
        for i, mask in enumerate(request["gt_masks"]):
            seg_vis[mask.astype(bool)] = colors[i % len(colors)]
        imageio.imwrite(out_dir / "gt_masks.png", seg_vis)
    meta = {
        "task": args.task,
        "goal_atoms": atoms,
        "gt_labels": labels,
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


def open_state_stream(hostport: str | None):
    """Optional live mirror of the sim into the server's Rerun view; None if disabled or unavailable."""
    if not hostport:
        return None
    from omnigibson.tiptop.client import SimStateStream

    host, _, port = hostport.rpartition(":")
    stream = SimStateStream(host or "localhost", int(port) if port else 8765)
    return stream if stream.open() else None


def live_round(sim, args, client, out_dir: Path, atoms: list[dict], hints: dict | None = None) -> dict:
    """Capture, ask the server for a plan for these atoms, save it and execute it."""
    request, extras = do_capture(sim, args, out_dir, atoms=atoms, hints=hints)
    response = client.plan(request, timeout_s=args.plan_timeout)
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
    stream = None if args.no_state_stream else open_state_stream(f"{args.host}:{args.port}")
    return do_execute(sim, args, out_dir, response["plan"], tag="live", state_stream=stream, atoms=atoms)


def choose_stage_spot(
    sim, item: str, container: str, support: str, radius: float = 1.2, spacing: float = 0.09, max_spots: int = 8
) -> tuple[float, float]:
    """(dx, dy) on ``support`` for the container: the free spot near ``item`` a base pose reaches together with it.

    Of the free spots within ``radius`` of the item's AABB center (a crowded table has none within 0.7 m; two
    points 1.2 m apart can still both be within reach of a base between them), the closest ``max_spots`` that lie
    ``spacing`` apart (adjacent grid cells are near-duplicates) are searched, closest first, and the one whose best base
    pose scores lowest wins; the container's origin goes to the spot, as place_on does (its AABB center follows within its
    origin-to-center offset), and the container and its contents, about to move, do not count as obstacles. With no
    reachable spot the closest one is used, as free_spot_on(near=) did, so place_robot_for still reports the
    rejections. One scene_aabbs() snapshot serves every search, so staging costs seconds, not one AABB query per
    candidate pose.
    """
    t0 = time.time()
    near = sim.scene_object(item).aabb_center.cpu().numpy()[:2].astype(np.float64)
    spots = [(float(np.hypot(x - near[0], y - near[1])), x, y) for x, y, _ in sim.free_spots_on(container, support)]
    spots = sorted(s for s in spots if s[0] <= radius)
    tried = []
    for s in spots:
        if all(np.hypot(s[1] - t[1], s[2] - t[2]) >= spacing for t in tried):
            tried.append(s)
            if len(tried) >= max_spots:
                break
    ignore = [sim.scene_object(n) for n in (container, *sim.contents_of(container))]
    aabbs = sim.scene_aabbs()
    half_widths = (sim.xy_radius(item), sim.xy_radius(container))  # keep both edges in frame, not just their centres
    best = None
    for d, x, y in tried:
        pose, _ = sim.best_base_pose(near, (x, y), ignore=ignore, aabbs=aabbs, half_widths=half_widths)
        if pose is not None and (best is None or pose[0] < best[0]):
            best = (pose[0], d, x, y)
    if best is None:
        log.info(
            f"staging {container} for {item}: no feasible spot ({len(tried)} of {len(spots)} spots within {radius} m "
            f"searched, {time.time() - t0:.1f}s); falling back to the closest one"
        )
        return sim.free_spot_on(container, support, near=near)
    score, d, x, y = best
    lo, hi = [v.cpu().numpy() for v in sim.scene_object(support).aabb]
    cx, cy = ((lo + hi) / 2)[:2].tolist()
    log.info(
        f"staging {container} for {item}: spot ({x:.2f}, {y:.2f}), {d:.2f} m from the item, base-pose score "
        f"{score:.2f} (best of {len(tried)} of {len(spots)} spots within {radius} m, {time.time() - t0:.1f}s)"
    )
    return x - cx, y - cy


def run_task(sim, args, client, out_dir: Path) -> dict:
    """Work through a challenge task's `inside` goal: every container gets one item of each type.

    Navigation stand-ins: containers are teleported onto --stage-support one at a time, the base is teleported to a
    reachable pose per transfer. Each transfer is one capture/plan/execute round; the item is verified with the
    task's own `inside` predicate and, on failure, the next unplaced item of that type is tried.
    """
    import psutil  # ships with isaacsim-kernel; not an OmniGibson dependency, so imported here

    from omnigibson.tiptop.r1pro import bddl_category

    task = sim.env.task
    pairs = []
    for head in task.ground_goal_state_options[0]:
        terms = list(getattr(head, "terms", []))
        if terms and terms[0] == "inside" and len(terms) == 3:
            pairs.append((terms[1], terms[2]))
    if not pairs:
        raise ValueError("task goal has no inside(item, container) predicates; nothing this driver can do")
    containers = list(dict.fromkeys(c for _, c in pairs))
    items_by_type = {}
    for item, _ in pairs:
        items_by_type.setdefault(bddl_category(item), []).append(item)
    for cat in items_by_type:
        items_by_type[cat] = list(dict.fromkeys(items_by_type[cat]))
    log.info(f"task: {len(containers)} containers x {list(items_by_type)} ({len(pairs)} predicates)")

    placed, transfers, n = set(), [], 0
    type_failures = dict.fromkeys(items_by_type, 0)  # containers in a row a type failed for; skipped after 2
    unreachable = set()  # items no base pose could reach together with a container: tried last from then on
    for basket in containers:
        home = sim.scene_object(basket).get_position_orientation()
        for cat, items in items_by_type.items():
            if type_failures[cat] >= 2:
                log.info(f"skipping {cat}: failed for the last {type_failures[cat]} containers")
                continue
            candidates = [i for i in items if i not in placed]
            if args.stage_support:  # items near the table's edge are the ones a base pose can reach
                lo, hi = [v.cpu().numpy() for v in sim.scene_object(args.stage_support).aabb]

                def edge_gap(name):
                    c = sim.scene_object(name).aabb_center.cpu().numpy()
                    return min(c[0] - lo[0], hi[0] - c[0], c[1] - lo[1], hi[1] - c[1])

                candidates.sort(key=edge_gap)
            candidates.sort(key=lambda name: name in unreachable)  # stable: keeps the edge order within each group
            ok = False
            for item in candidates[: args.attempts_per_item]:
                n += 1
                round_dir = out_dir / f"t{n:02d}_{bddl_category(item).replace(' ', '_')}_{basket.split('_')[-1]}"
                round_dir.mkdir(parents=True, exist_ok=True)
                record = {"item": item, "container": basket, "dir": str(round_dir)}
                t0 = time.time()
                try:
                    if args.stage_support:  # bring the container (with what it holds) next to this item
                        dx, dy = choose_stage_spot(sim, item, basket, args.stage_support)
                        sim.place_on_with_contents(basket, args.stage_support, dx, dy)
                        sim.hold(args.settle_steps, sim.OPEN)
                    sim.place_robot_for(item, basket)
                    sim.hold(args.settle_steps, sim.OPEN)
                    atoms = [{"predicate": "inside", "args": [item, basket]}]
                    _, tiptop_atoms = sim.tiptop_goal(atoms, category_level=bool(args.no_gt))
                    hints = {lab: sim.base_hint(b) for lab, b in zip(tiptop_atoms[0]["args"], (item, basket))}
                    live_round(sim, args, client, round_dir, atoms=atoms, hints=hints)
                    # the goal names the category, so any item of this type that ended up inside counts
                    now_inside = [i for i in items if i not in placed and sim.predicate_holds("inside", i, basket)]
                    record["inside"], record["placed"] = bool(now_inside), now_inside
                except Exception as e:  # noqa: BLE001 - one failed transfer must not end the task
                    log.exception(f"transfer {n} {item} -> {basket} failed")
                    record["inside"], record["error"] = False, f"{type(e).__name__}: {e}"
                    if isinstance(e, RuntimeError) and str(e).startswith("no base pose"):
                        unreachable.add(item)  # mid-table; the next container's candidates start with the others
                record["seconds"] = round(time.time() - t0, 1)
                record["rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)  # the last run died at 14 GB
                transfers.append(record)
                log.info(
                    f"transfer {n}: {item} -> {basket}: "
                    f"{'OK ' + str(record.get('placed')) if record['inside'] else 'failed'} "
                    f"({record['seconds']}s, rss {record['rss_gb']} GB)"
                )
                if record["inside"]:
                    placed.update(record["placed"])
                    ok = True
                    break
            type_failures[cat] = 0 if ok else type_failures[cat] + 1
        if args.stage_support:  # done with this container: back to the floor with its contents, freeing the table
            sim.move_with_contents(basket, home[0], home[1])
            sim.hold(args.settle_steps, sim.OPEN)
    summary = {"transfers": transfers, "placed": sorted(placed), "task_goal": sim.goal_status()}
    with open(out_dir / "task_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"TASK RESULT: {len(placed)}/{len(pairs)} items placed, {summary['task_goal']}")
    return summary


def do_execute(
    sim, args, out_dir: Path, plan: dict, tag: str, state_stream=None, atoms: list[dict] | None = None
) -> dict:
    """Execute a plan and check the goal: the task's own with --activity (``atoms``, default --goal, then names the
    objects whose AABBs are logged), else every --goal atom."""
    from omnigibson.tiptop.executor import PlanExecutor, VideoRecorder, check_success
    from omnigibson.tiptop.protocol import plan_summary

    atoms = parse_goal(args.goal) if atoms is None else list(atoms)
    log.info(f"executing plan: {plan_summary(plan)}")
    video = None if args.no_video else VideoRecorder(out_dir / f"{tag}.mp4", fps=15, every=2)
    executor = PlanExecutor(sim, gripper_hold_steps=args.gripper_hold_steps, video=video, state_stream=state_stream)
    try:
        stats = executor.execute(plan)
    finally:
        if state_stream is not None:
            state_stream.close()
    if args.activity:
        success = sim.goal_status()
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
    if video is not None:
        video.close()
    result = {
        "plan_summary": plan_summary(plan),
        "execution": stats,
        "success": success,
        "final_object_poses_world": sim.object_poses_world(),
    }
    with open(out_dir / f"{tag}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"success check: {json.dumps(success)}")
    return result


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
    p_live.add_argument("--host", default="localhost")
    p_live.add_argument("--port", type=int, default=8765)
    p_live.add_argument("--plan-timeout", type=float, default=900.0)
    p_live.add_argument(
        "--sequential",
        action="store_true",
        help="one capture/plan/execute round per goal atom (re-standing for each with --activity), instead of one "
        "plan for the whole goal",
    )
    p_live.add_argument(
        "--no-state-stream", action="store_true", help="do not mirror the executed sim state into the server's Rerun"
    )
    p_task = sub.add_parser("task", help="work through a challenge task's whole inside(item, container) goal")
    add_common(p_task)
    p_task.add_argument("--host", default="localhost")
    p_task.add_argument("--port", type=int, default=8765)
    p_task.add_argument("--plan-timeout", type=float, default=900.0)
    p_task.add_argument("--no-state-stream", action="store_true")
    p_task.add_argument(
        "--stage-support",
        default=None,
        help="BDDL name of the furniture each container is brought onto before it is filled (e.g. table.n.02_1)",
    )
    p_task.add_argument("--attempts-per-item", type=int, default=2, help="candidate items tried per type per container")
    args = parser.parse_args(argv)

    # OmniGibson installs its own handler on the "omnigibson" logger; give this sub-logger its own INFO stream
    tiptop_log = logging.getLogger("omnigibson.tiptop")
    tiptop_log.setLevel(logging.INFO)
    tiptop_log.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    tiptop_log.addHandler(handler)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import omnigibson as og
    from omnigibson.tiptop.protocol import load_plan_json

    exit_code = 0
    try:
        t0 = time.time()
        client = None
        if args.cmd in ("live", "task"):
            from omnigibson.tiptop.client import TiptopClient

            # the server's embodiment metadata (locked posture, home pose) shapes the scene, so fetch it first
            client = TiptopClient(
                args.host,
                args.port,
                expected_robot_type=EXPECTED_ROBOT_TYPE[args.embodiment],
                expected_dof=EXPECTED_DOF[args.embodiment],
            )
            client.wait_for_server()
            metadata = client.fetch_metadata()
            client.check_embodiment()  # fail here, before Isaac Sim starts, if the server plans for another robot
            sim = build_sim(args, embodiment=metadata.get("embodiment"))
        elif args.cmd == "replay":
            with open(args.plan) as f:
                plan_json = json.load(f)
            sim = build_sim(args, embodiment=plan_json.get("embodiment"))  # provenance saved by `live`
        else:
            sim = build_sim(args)
        log.info(f"scene ready in {time.time() - t0:.1f}s (sim dt {sim.dt:.4f}s)")
        if args.cmd == "capture":
            do_capture(sim, args, out_dir)
        elif args.cmd == "task":
            if not args.activity:
                raise ValueError("task needs --activity")
            run_task(sim, args, client, out_dir)
        elif args.cmd == "replay":
            plan = load_plan_json(args.plan)
            if args.embodiment == "r1pro" and not plan_json.get("embodiment"):
                log.warning("plan has no embodiment provenance; assuming it was made for the local tiptop embodiment")
            do_execute(sim, args, out_dir, plan, tag="replay", state_stream=open_state_stream(args.state_stream))
        elif args.cmd == "live":
            atoms_all = parse_goal(args.goal)
            rounds = [[atom] for atom in atoms_all] if args.sequential else [atoms_all]
            outcomes = []
            for i, atoms in enumerate(rounds):
                round_dir = out_dir / f"round_{i:02d}" if args.sequential else out_dir
                round_dir.mkdir(parents=True, exist_ok=True)
                try:
                    if args.sequential and args.activity and len(atoms[0]["args"]) == 2:
                        sim.place_robot_for(*atoms[0]["args"])  # navigation stand-in for this transfer
                        sim.hold(args.settle_steps, sim.OPEN)
                    outcomes.append(live_round(sim, args, client, round_dir, atoms))
                except Exception as e:
                    if not args.sequential:
                        raise
                    log.exception(f"round {i} {atoms} failed")
                    outcomes.append({"error": f"{type(e).__name__}: {e}"})
                if args.sequential:
                    log.info(f"round {i} {atoms}: {outcomes[-1].get('success', outcomes[-1].get('error'))}")
            if args.sequential:
                summary = {"rounds": outcomes}
                if args.activity:
                    summary["task_goal"] = sim.goal_status()
                    log.info(f"task goal after {len(rounds)} rounds: {summary['task_goal']}")
                with open(out_dir / "sequential_summary.json", "w") as f:
                    json.dump(summary, f, indent=2, default=str)
    except Exception:
        log.exception("run failed")
        exit_code = 1
    finally:
        if og.app is not None:  # og.shutdown() exits with status 0 when Isaac Sim was never launched
            og.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
