"""Closed-loop move_base_to(object) in OmniGibson: spin, detect, search, approach, drive.

Thin harness around the policy code that lives outside the simulator:
``nav2py.objnav.ObjectSearchPolicy`` decides (spin / goto / approach / done),
``mapping_lab.objnav`` turns head-camera frames into hypotheses, and the
detector runs in its own environment behind ``mapping_lab.objnav.detect_server``
(file protocol, ``--detector-dir``). Driving reuses ``run_nav2py_benchmark``'s
costmap / profile / navigator plumbing with ``--costmap-source b1k-gt``.

Success is graded against simulator ground truth only after the fact: the
policy never reads object poses.

    python scripts/navigation/run_object_search.py --scene Rs_int --task composting_waste \
        --target trash_can --nav2py-root ../nav2py --mapping-lab-root ../3dmap/src \
        --detector-dir /tmp/det --output result.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from generate_nav_benchmark import build_env_config, controller_no_op_action, load_robot_config, seed_everything
from generate_object_nav_benchmark import find_template, template_robot_pose
from omnigibson.eval.utils.eval_utils import set_sensor_modalities
from omnigibson.macros import gm

import run_nav2py_benchmark as rnb
from capture_object_search_frames import head_sensor, patched_scene_file, scene_ground_truth, to_list

DEFAULT_TASK_INSTANCES_ROOT = str(Path(__file__).resolve().parents[3] / "datasets" / "2026-challenge-task-instances")
DEFAULT_B1K_MAP_ROOT = "/scratch/gxwang2/b1k/b1k_gt_out/final_v4"
NAV_ARGS = [
    "--costmap-source", "b1k-gt", "--disable-path-smoothing", "--desired-linear-velocity", "0.3",
    "--profile-max-linear-velocity", "0.75", "--profile-max-angular-velocity", "1.75",
    "--command-max-linear-velocity", "0.75", "--command-max-angular-velocity", "1.0",
    "--runtime-extra-clearance", "0.2", "--success-distance", "0.1",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target", required=True, help="OmniGibson category (search) or scene object name (inventory)")
    parser.add_argument("--task-instance-id", type=int, default=0)
    parser.add_argument("--task-instances-root", default=DEFAULT_TASK_INSTANCES_ROOT)
    parser.add_argument("--b1k-map-root", default=DEFAULT_B1K_MAP_ROOT)
    parser.add_argument("--nav2py-root", default=None)
    parser.add_argument("--mapping-lab-root", required=True, help="3dmap/src, for mapping_lab.objnav")
    parser.add_argument("--detector-dir", required=True, help="watch dir of a running mapping_lab.objnav.detect_server")
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot-config", default=str(Path(__file__).resolve().parents[2] / "omnigibson" / "eval" / "r1pro.yaml"))
    parser.add_argument("--robot-pose-key", default="robot")
    parser.add_argument("--max-viewpoints", type=int, default=10)
    parser.add_argument("--frames-per-spin", type=int, default=8)
    parser.add_argument("--spin-angular-velocity", type=float, default=1.0)
    parser.add_argument("--max-nav-steps", type=int, default=1800)
    parser.add_argument("--success-distance", type=float, default=0.1)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-frames", default=None, help="directory for spin_XX.npz (rgb, depth, K, camera poses) per spin")
    parser.add_argument("--verify-radius", type=float, default=0.75, help="a verification detection must land within this of the hypothesis")
    return parser.parse_args(argv)


def frame_arrays(robot, sensor_name, sensor):
    obs, _ = robot.get_obs()
    frame = obs[sensor_name]
    rgb = frame["rgb"]
    rgb = rgb.detach().cpu().numpy() if isinstance(rgb, th.Tensor) else np.asarray(rgb)
    depth = frame["depth_linear"]
    depth = depth.detach().cpu().numpy() if isinstance(depth, th.Tensor) else np.asarray(depth)
    pos, quat = sensor.get_position_orientation()
    return rgb[..., :3].astype(np.uint8), depth.astype(np.float32), to_list(pos), to_list(quat)


def robot_pose(robot):
    position, orientation = robot.get_position_orientation()
    yaw = float(T.quat2euler(orientation)[2].item())
    return float(position[0].item()), float(position[1].item()), yaw


def spin_and_capture(env, robot, sensor_name, sensor, args):
    """Rotate in place at ``--spin-angular-velocity`` and grab a frame every 360/N degrees."""
    x0, y0, yaw0 = robot_pose(robot)
    targets = [(yaw0 + 2.0 * math.pi * k / args.frames_per_spin) for k in range(args.frames_per_spin)]
    frames = []
    turned = 0.0
    previous_yaw = yaw0
    next_index = 0
    dt = 1.0 / 30.0
    max_steps = int((2.0 * math.pi / args.spin_angular_velocity) / dt * 2.5)
    spin_command = th.zeros(3)
    spin_command[2] = args.spin_angular_velocity
    base_group_key, _ = robot.controllers["base"]
    from omnigibson.controllers import ControllerView

    for _ in range(max_steps):
        if next_index >= args.frames_per_spin:
            break
        target = (targets[next_index] - yaw0) % (2.0 * math.pi)
        if next_index == 0 or turned + 1e-3 >= target:
            # Stop briefly so the frame is not motion-blurred by physics, then capture.
            for _ in range(3):
                env.step({robot.name: controller_no_op_action(robot)})
            rgb, depth, cam_pos, cam_quat = frame_arrays(robot, sensor_name, sensor)
            bx, by, byaw = robot_pose(robot)
            frames.append({"rgb": rgb, "depth": depth, "cam_pos": cam_pos, "cam_quat": cam_quat, "base": [bx, by, byaw]})
            next_index += 1
            continue
        action = controller_no_op_action(robot)
        action[robot.base_action_idx] = ControllerView.reverse_preprocess_command(base_group_key, spin_command)
        env.step({robot.name: action})
        _, _, yaw = robot_pose(robot)
        delta = (yaw - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
        turned += delta
        previous_yaw = yaw
    for _ in range(3):
        env.step({robot.name: controller_no_op_action(robot)})
    return frames


def navigate_to(env, robot, goal, costmap, profile, navigation_config, command_limits, nav_api, args, nav_args, label):
    """Drive with nav2py to ``goal`` (x, y[, yaw]); returns a diagnostic dict with ``reached``."""
    navigator = rnb.make_navigator(profile, costmap, True, nav_api, navigation_config, disable_dynamic_safety=True)
    navigator.submit(
        nav_api["NavigationTask"](
            label,
            goal_pose=nav_api["Pose2D"](float(goal[0]), float(goal[1]), float(goal[2]) if len(goal) > 2 else 0.0),
            goal_semantics=nav_api["GoalSemantics"].POSITION_ONLY,
        )
    )
    dt = profile.control_period
    reached = False
    step = 0
    for step in range(args.max_nav_steps):
        now = step * dt
        state = rnb.robot_state_estimate(robot, now, nav_api, nav_args)
        command = navigator.tick(state, now)
        executed = rnb.cap_command_to_controller_limits(command, command_limits)
        env.step({robot.name: rnb.action_from_nav2py_command(robot, executed)})
        x, y, _ = robot_pose(robot)
        distance = math.hypot(x - goal[0], y - goal[1])
        if distance <= args.success_distance:
            reached = True
            break
        if navigator.status().state.value in {"succeeded", "failed", "blocked", "canceled"}:
            reached = distance <= args.success_distance
            break
    for _ in range(5):
        env.step({robot.name: controller_no_op_action(robot)})
    x, y, yaw = robot_pose(robot)
    status = navigator.status()
    return {
        "label": label,
        "goal": [float(v) for v in goal],
        "reached": bool(reached),
        "final_distance_m": math.hypot(x - goal[0], y - goal[1]),
        "final_pose": [x, y, yaw],
        "steps": step + 1,
        "sim_seconds": (step + 1) * dt,
        "nav2py_state": status.state.value,
        "nav2py_reason": status.reason,
    }


def rotate_to_yaw(env, robot, yaw_goal, max_steps=240, gain=2.0, max_w=1.0):
    from omnigibson.controllers import ControllerView

    base_group_key, _ = robot.controllers["base"]
    for _ in range(max_steps):
        _, _, yaw = robot_pose(robot)
        error = (yaw_goal - yaw + math.pi) % (2.0 * math.pi) - math.pi
        if abs(error) < 0.05:
            break
        command = th.zeros(3)
        command[2] = max(-max_w, min(max_w, gain * error))
        action = controller_no_op_action(robot)
        action[robot.base_action_idx] = ControllerView.reverse_preprocess_command(base_group_key, command)
        env.step({robot.name: action})
    for _ in range(3):
        env.step({robot.name: controller_no_op_action(robot)})


def save_spin(path, frames, K, floor_z):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rgb=np.stack([f["rgb"] for f in frames]),
        depth=np.stack([f["depth"] for f in frames]),
        cam_pos=np.asarray([f["cam_pos"] for f in frames], dtype=np.float64),
        cam_quat_xyzw=np.asarray([f["cam_quat"] for f in frames], dtype=np.float64),
        base=np.asarray([f["base"] for f in frames], dtype=np.float64),
        K=np.asarray(K, dtype=np.float64),
        floor_z=np.asarray(floor_z, dtype=np.float64),
    )


def structure_cells_mask(obstacle, inventory, spec, oriented_box_footprint):
    """Static obstacle cells that no inventory object explains (walls) plus mirror footprints.

    A reflection backprojects onto the mirror plane, i.e. into these cells; the
    tracker rejects hypotheses whose footprint mostly lies inside them.
    """
    explained = np.zeros(spec.shape, dtype=bool)
    mirrors = np.zeros(spec.shape, dtype=bool)
    for inst in inventory:
        footprint = oriented_box_footprint(spec, inst.translation, inst.rotation_xyzw, inst.size)
        if inst.category == "mirror":
            mirrors |= footprint
        else:
            explained |= footprint
    return (np.asarray(obstacle, dtype=bool) & ~explained) | mirrors


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    rnb.add_nav2py_to_path(args.nav2py_root)
    sys.path.insert(0, str(Path(args.mapping_lab_root).expanduser().resolve()))
    nav_api = rnb.load_nav2py()
    from benchmarks.costmaps import navigation_2d_layers, planning_costmap_from_navigation_2d
    from mapping_lab.objnav import HypothesisTracker, Observation, detection_points
    from mapping_lab.objnav.detect_server import DetectorClient, DetectorDown
    from mapping_lab.objnav.oracle import camera_geometry
    from nav2py.objnav import GridSpec, Hypothesis, ObjectSearchPolicy, costmap_with_obstacle, inventory_instances, oriented_box_footprint, resolve_target
    from nav2py.objnav.footprint import cells_footprint

    nav_args = rnb.parse_args(NAV_ARGS + ["--benchmark", "unused", "--output", "unused"])
    nav_args.success_distance = args.success_distance
    navigation_config = rnb.make_navigation_config(nav_api, nav_args)

    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False

    template_path = find_template(Path(args.task_instances_root), args.scene, args.task, args.task_instance_id)
    map_dir = rnb.b1k_map_directory(args.b1k_map_root, args.scene)
    with open(Path(map_dir).parent / "semantic_occupancy" / "scene_inventory.json", encoding="utf-8") as f:
        inventory = inventory_instances(json.load(f))
    inventory_names = {inst.source_name for inst in inventory}

    robot_cfg = load_robot_config(args.robot_config)
    robot_cfg["obs_modalities"] = ["rgb"]
    command_limits = rnb.resolve_command_limits(robot_cfg, nav_args)
    cfg = build_env_config(scene_model=args.scene, robot_cfg=robot_cfg, scene_instance=template_path.stem, load_room_instances=None)
    cfg["scene"]["scene_file"] = patched_scene_file(template_path)
    started = time.time()
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    if robot.model in ("r1", "r1pro"):
        og.sim.stop()
        robot.base_footprint_link.mass = 250.0
        og.sim.play()
    env.reset(get_obs=False)
    print(f"environment ready in {time.time() - started:.0f}s", flush=True)

    og_resolution = float(env.scene.trav_map.map_resolution)
    robot_erosion_m = rnb.og_robot_erosion_meters(robot, og_resolution)
    costmap = planning_costmap_from_navigation_2d(map_dir, robot_erosion_m, nav_args.runtime_extra_clearance)
    traversable, obstacle, unknown = navigation_2d_layers(map_dir)
    spec = GridSpec.from_costmap(costmap)
    profile = rnb.make_robot_profile(robot, nav_api, nav_args, clearance_is_in_costmap=True)

    start_position, start_quat = template_robot_pose(template_path, args.robot_pose_key)
    rnb.place_robot(robot, {"start_position": start_position, "start_quat": start_quat})
    for _ in range(args.settle_steps):
        env.step({robot.name: controller_no_op_action(robot)})
    rnb.zero_robot_velocities(robot)
    sensor_name, sensor = head_sensor(robot)
    set_sensor_modalities(sensor, {"rgb", "depth_linear"})
    og.sim.update_handles()
    env.load_observation_space()
    K = sensor.intrinsic_matrix.detach().cpu().numpy().astype(np.float64)
    floor_z = float(start_position[2])
    _, cam_quat0 = sensor.get_position_orientation()
    camera = camera_geometry(K, (int(sensor.image_height), int(sensor.image_width)), to_list(cam_quat0))
    x0, y0, yaw0 = robot_pose(robot)
    print(f"head camera K diag=({K[0,0]:.1f},{K[1,1]:.1f}) c=({K[0,2]:.1f},{K[1,2]:.1f}) hfov={camera['hfov_deg']:.1f} deg "
          f"pitch={camera['pitch_deg']:.1f} deg; settled start ({x0:.3f},{y0:.3f},{yaw0:.2f}) vs template "
          f"({start_position[0]:.3f},{start_position[1]:.3f})", flush=True)
    structure_cells = structure_cells_mask(obstacle, inventory, spec, oriented_box_footprint)

    ground_truth = scene_ground_truth(env, robot, inventory_names)
    gt_targets = [o for o in ground_truth if (o["category"] == args.target or o["name"] == args.target) and "aabb" in o]
    prompts = [args.target] if not resolve_target(inventory, args.target) else []

    client = DetectorClient(Path(args.detector_dir))
    tracker = HypothesisTracker(max_centroid_distance_m=0.30, confirm_min_score=0.4)
    policy = ObjectSearchPolicy(
        costmap, obstacle | unknown, traversable | obstacle, inventory, args.target,
        max_viewpoints=args.max_viewpoints, confirm_with_planner=True,
    )
    events = []
    counters = {"frame": 0, "spin": 0}
    frames_dir = Path(args.save_frames) / args.scene / f"{args.task}__{args.target}" if args.save_frames else None

    def frame_observations(frame, dets):
        observations = []
        for det in dets:
            points = detection_points(frame["depth"], det.mask, K, frame["cam_pos"], frame["cam_quat"],
                                      erode_px=2, depth_band_m=0.15, max_depth_m=6.0, floor_z=floor_z)
            if len(points) >= 20:
                observations.append(Observation(det.category, float(det.score), points, int(det.mask.sum())))
        return observations

    def integrate(frames, detections):
        n_obs = 0
        for frame, dets in zip(frames, detections):
            observations = frame_observations(frame, dets)
            tracker.add_frame(counters["frame"], observations)
            counters["frame"] += 1
            n_obs += len(observations)
        hyps = tracker.export(grid=(spec.resolution, spec.origin, spec.shape), floor_z=floor_z, structure_cells=structure_cells)
        policy_hyps = [
            Hypothesis(h["category"], h["instance_id"], tuple(h["centroid"]), np.asarray(h["footprint_cells"]),
                       h["n_frames"], h["score"], h["confirmed"])
            for h in hyps
        ]
        return n_obs, hyps, policy_hyps

    x, y, _ = robot_pose(robot)
    action = policy.begin((x, y))
    wall_started = time.time()
    error = None
    try:
        while action.kind != "done":
            x, y, yaw = robot_pose(robot)
            if action.kind == "spin":
                t0 = time.time()
                frames = spin_and_capture(env, robot, sensor_name, sensor, args)
                t_spin = time.time() - t0
                if frames_dir is not None:
                    save_spin(frames_dir / f"spin_{counters['spin']:02d}.npz", frames, K, floor_z)
                counters["spin"] += 1
                detections = client.detect(np.stack([f["rgb"] for f in frames]), prompts)
                t_detect = time.time() - t0 - t_spin
                n_obs, hyps, policy_hyps = integrate(frames, detections)
                x, y, yaw = robot_pose(robot)
                action = policy.report_spin((x, y), policy_hyps)
                events.append({
                    "event": "spin", "robot_pose": [x, y, yaw], "frames": len(frames), "observations": n_obs,
                    "detections": int(sum(len(d) for d in detections)),
                    "hypotheses": [{k: v for k, v in h.items() if k != "footprint_cells"} for h in hyps],
                    "ipc": client.timings[-1], "seconds": time.time() - t0, "spin_seconds": t_spin,
                    "detect_seconds": t_detect, "next": action.kind,
                })
                print(f"spin at ({x:.2f},{y:.2f}): {sum(len(d) for d in detections)} detections, {n_obs} observations, "
                      f"{len(hyps)} hypotheses ({t_spin:.1f}s spin, {t_detect:.2f}s detect, {time.time() - t0:.1f}s total) -> {action.kind}", flush=True)
            elif action.kind in ("goto", "approach"):
                drive_costmap = costmap
                if action.kind == "approach" and policy.pending is not None:
                    # Drive on the costmap the pose was chosen on: the hypothesis footprint is not
                    # in the static map, so without it the planner could route through the object.
                    drive_costmap = costmap_with_obstacle(
                        costmap, cells_footprint(spec, policy.pending.footprint_cells), policy.robot_erosion_m, policy.extra_clearance_m
                    )
                nav = navigate_to(env, robot, action.pose, drive_costmap, profile, navigation_config, command_limits, nav_api, args, nav_args, action.kind)
                if action.kind == "approach" and nav["reached"]:
                    rotate_to_yaw(env, robot, action.pose[2])
                x, y, yaw = robot_pose(robot)
                nav["final_pose"] = [x, y, yaw]
                events.append({"event": action.kind, **nav})
                print(f"{action.kind} -> {action.pose}: reached={nav['reached']} final_distance={nav['final_distance_m']:.3f} state={nav['nav2py_state']}", flush=True)
                if action.kind == "goto":
                    action = policy.report_arrival((x, y), nav["reached"])
                elif not nav["reached"]:
                    action = policy.report_approach((x, y), False)
                else:
                    # One verification look facing the hypothesis: re-detect the target within
                    # --verify-radius of its centroid, else reject it and keep searching.
                    pending = policy.pending
                    rgb, depth, cam_pos, cam_quat = frame_arrays(robot, sensor_name, sensor)
                    frame = {"rgb": rgb, "depth": depth, "cam_pos": cam_pos, "cam_quat": cam_quat, "base": [x, y, yaw]}
                    if frames_dir is not None:
                        save_spin(frames_dir / f"verify_{counters['spin']:02d}.npz", [frame], K, floor_z)
                    detections = client.detect(rgb[None], prompts)
                    observations = frame_observations(frame, detections[0])
                    seen = []
                    for obs in observations:
                        if obs.category != args.target:
                            continue
                        centroid = np.median(obs.points, axis=0)
                        seen.append(float(np.linalg.norm(centroid[:2] - np.asarray(pending.centroid[:2]))))
                    verified = bool(seen) and min(seen) <= args.verify_radius
                    events.append({"event": "verify", "robot_pose": [x, y, yaw], "hypothesis": pending.instance_id,
                                   "target_detections": len(seen), "nearest_m": min(seen) if seen else None, "verified": verified})
                    print(f"verify hypothesis {pending.instance_id}: {len(seen)} target detections, nearest {min(seen) if seen else None} -> {'ok' if verified else 'REJECT'}", flush=True)
                    if verified:
                        action = policy.report_approach((x, y), True)
                    else:
                        policy.reject_pending()
                        action = policy.report_arrival((x, y), True)  # spin again from here
            else:
                break
    except (DetectorDown, TimeoutError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"DETECTOR_DOWN {error}", flush=True)
        action = type(action)("done", result={"status": "DETECTOR_DOWN", "error": error,
                                              "viewpoints_visited": policy.viewpoints_visited})

    result = action.result
    x, y, yaw = robot_pose(robot)
    grading = {}
    if gt_targets:
        centres = [((np.asarray(o["aabb"][0]) + np.asarray(o["aabb"][1])) / 2.0) for o in gt_targets]
        grading["gt_targets"] = [o["name"] for o in gt_targets]
        grading["final_base_to_nearest_gt_centre_m"] = float(min(np.linalg.norm(np.array([x, y]) - c[:2]) for c in centres))
        if result.get("target_estimate") and result["target_estimate"].get("centroid"):
            est = np.asarray(result["target_estimate"]["centroid"][:2])
            grading["estimate_error_xy_m"] = float(min(np.linalg.norm(est - c[:2]) for c in centres))
        if result.get("approach_pose"):
            pose = np.asarray(result["approach_pose"][:2])
            grading["approach_pose_to_nearest_gt_centre_m"] = float(min(np.linalg.norm(pose - c[:2]) for c in centres))
            # Standoff to the GT footprint boundary (AABB), the quantity the policy aims at.
            def aabb_distance(p, o):
                lo, hi = np.asarray(o["aabb"][0])[:2], np.asarray(o["aabb"][1])[:2]
                d = np.maximum(np.maximum(lo - p, p - hi), 0.0)
                return float(np.linalg.norm(d))
            grading["approach_pose_to_gt_aabb_m"] = min(aabb_distance(pose, o) for o in gt_targets)
            grading["final_base_to_gt_aabb_m"] = min(aabb_distance(np.array([x, y]), o) for o in gt_targets)
    output = {
        "scene": args.scene,
        "task": args.task,
        "target": args.target,
        "template": str(template_path),
        "prompts": prompts,
        "start_position": start_position,
        "final_pose": [x, y, yaw],
        "result": result,
        "grading": grading,
        "events": events,
        "policy_log": policy.log,
        "ipc_timings": client.timings,
        "wall_seconds": time.time() - wall_started,
        "settled_start": [x0, y0, yaw0],
        "ground_truth_targets": gt_targets,
        "camera": camera,
        "error": error,
        "frames_dir": str(frames_dir) if frames_dir else None,
        "costmap": {"map_dir": str(map_dir), "robot_erosion_m": robot_erosion_m, "extra_clearance_m": nav_args.runtime_extra_clearance,
                    "plannable_cells": int((costmap.data < 253).sum())},
        "nav": {"success_distance_m": args.success_distance, "navigation_config": rnb.navigation_config_diagnostics(navigation_config),
                "command_limits": command_limits},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1)
    print(f"RESULT {result['status']} grading={grading} wall={output['wall_seconds']:.0f}s", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
