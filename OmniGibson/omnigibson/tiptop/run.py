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
    r1.add_argument("--rooms", default=None, help="comma-separated load_room_types (default: whole scene)")
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
    cfg = make_r1pro_env_config(
        scene_model=args.scene_model,
        load_room_types=[r for r in (args.rooms or "").split(",") if r],
        spawn_presets=[sp[0] for sp in spawns],
        grasping_mode=args.grasping_mode,
        camera=args.camera,
        head_aperture_mm=args.head_aperture,
        not_load_object_categories=[c for c in args.not_load.split(",") if c],
    )
    sim = R1ProSim(cfg, camera=args.camera)
    if args.robot_pose:
        sim.place_robot(*args.robot_pose)
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


def do_capture(sim, args, out_dir: Path) -> tuple[dict, dict]:
    from omnigibson.tiptop.protocol import save_observation_h5

    atoms = parse_goal(args.goal)
    labels = sorted({a for atom in atoms for a in atom["args"] if a in sim.objects})
    request, extras = sim.capture(args.task, gt_labels=labels, gt_atoms=atoms)
    report = sim.validate_capture(request, extras)
    for problem in report["problems"]:
        log.warning(f"capture validation: {problem}")
    save_observation_h5(out_dir / "obs.h5", request, extras["cam_pos_base"], extras["cam_quat_wxyz_ros"])
    import imageio

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


def do_execute(sim, args, out_dir: Path, plan: dict, tag: str, state_stream=None) -> dict:
    from omnigibson.tiptop.executor import PlanExecutor, VideoRecorder, check_success
    from omnigibson.tiptop.protocol import plan_summary

    log.info(f"executing plan: {plan_summary(plan)}")
    video = None if args.no_video else VideoRecorder(out_dir / f"{tag}.mp4", fps=15, every=2)
    executor = PlanExecutor(sim, gripper_hold_steps=args.gripper_hold_steps, video=video, state_stream=state_stream)
    try:
        stats = executor.execute(plan)
    finally:
        if state_stream is not None:
            state_stream.close()
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
    p_live.add_argument(
        "--no-gt", action="store_true", help="do not send ground-truth masks (server then needs Gemini + SAM2)"
    )
    p_live.add_argument("--plan-timeout", type=float, default=900.0)
    p_live.add_argument(
        "--no-state-stream", action="store_true", help="do not mirror the executed sim state into the server's Rerun"
    )
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
        if args.cmd == "live":
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
        elif args.cmd == "replay":
            plan = load_plan_json(args.plan)
            if args.embodiment == "r1pro" and not plan_json.get("embodiment"):
                log.warning("plan has no embodiment provenance; assuming it was made for the local tiptop embodiment")
            do_execute(sim, args, out_dir, plan, tag="replay", state_stream=open_state_stream(args.state_stream))
        elif args.cmd == "live":
            request, extras = do_capture(sim, args, out_dir)
            if args.no_gt:
                request = {k: v for k, v in request.items() if not k.startswith("gt_")}
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
            do_execute(sim, args, out_dir, response["plan"], tag="live", state_stream=stream)
    except Exception:
        log.exception("run failed")
        exit_code = 1
    finally:
        if og.app is not None:  # og.shutdown() exits with status 0 when Isaac Sim was never launched
            og.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
