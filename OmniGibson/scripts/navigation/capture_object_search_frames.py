"""Capture head-camera spins along the object-search viewpoint sequence of a task template.

Thin simulator harness for the object-search workstream. The search policy
(``nav2py.objnav.ObjectSearchPolicy``) chooses the viewpoints on the b1k
ground-truth planning costmap exactly as it would online, but with no
detections fed back, so the sequence is the open-loop exploration path. At
every viewpoint the robot is teleported, spun through ``--frames-per-spin``
headings, and the head camera's rgb / depth_linear / seg_instance frames are
written with the camera intrinsics and poses. Ground-truth poses of the task
objects that are absent from the static map are saved next to the frames so
the detector + backprojection pipeline (3dmap ``mapping_lab.objnav``) can be
evaluated offline against them.

Example:
    python scripts/navigation/capture_object_search_frames.py \
        --scene Rs_int --task composting_waste:trash_can --task store_honey:jar_of_honey \
        --nav2py-root ../nav2py --output-dir /scratch/.../capture
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from generate_nav_benchmark import build_env_config, controller_no_op_action, load_robot_config, place_robot, seed_everything
from generate_object_nav_benchmark import find_template, template_robot_pose
from omnigibson.eval.utils.eval_utils import set_sensor_modalities
from omnigibson.macros import gm
from run_nav2py_benchmark import add_nav2py_to_path, b1k_map_directory, og_robot_erosion_meters

DEFAULT_TASK_INSTANCES_ROOT = str(Path(__file__).resolve().parents[3] / "datasets" / "2026-challenge-task-instances")
DEFAULT_B1K_MAP_ROOT = "/scratch/gxwang2/b1k/b1k_gt_out/final_v4"
HEAD_SENSOR_LINK = "zed_link"
STRUCTURE_CATEGORIES = {"floors", "walls", "ceilings"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        help="task_name[:target_category]. The target only steers the viewpoint prior; repeat for several tasks.",
    )
    parser.add_argument("--task-instance-id", type=int, default=0)
    parser.add_argument("--task-instances-root", default=DEFAULT_TASK_INSTANCES_ROOT)
    parser.add_argument("--b1k-map-root", default=DEFAULT_B1K_MAP_ROOT)
    parser.add_argument("--nav2py-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--robot-config", default=str(Path(__file__).resolve().parents[2] / "omnigibson" / "eval" / "r1pro.yaml"))
    parser.add_argument("--robot-pose-key", default="robot")
    parser.add_argument("--max-viewpoints", type=int, default=10)
    parser.add_argument("--view-radius", type=float, default=4.0)
    parser.add_argument("--frames-per-spin", type=int, default=8)
    parser.add_argument("--settle-steps", type=int, default=6)
    parser.add_argument("--runtime-extra-clearance", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-seg-instance", dest="seg_instance", action="store_false", help="capture rgb + depth only")
    return parser.parse_args(argv)


def load_nav2py_objnav():
    from benchmarks.costmaps import navigation_2d_layers, planning_costmap_from_navigation_2d
    from nav2py.objnav import ObjectSearchPolicy, inventory_instances, resolve_target

    return {
        "ObjectSearchPolicy": ObjectSearchPolicy,
        "inventory_instances": inventory_instances,
        "resolve_target": resolve_target,
        "navigation_2d_layers": navigation_2d_layers,
        "planning_costmap_from_navigation_2d": planning_costmap_from_navigation_2d,
    }


def to_list(value):
    if isinstance(value, th.Tensor):
        value = value.detach().cpu()
    return np.asarray(value, dtype=np.float64).tolist()


def yaw_quat(yaw):
    return th.tensor([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], dtype=th.float32)


def patched_scene_file(template_path, image_size=1080):
    """The task template with its robot given a head camera.

    Task templates carry their own robot (``class_name: Robot``, ``obs_modalities: []``).
    ``include_robots: False`` does not filter it (``REGISTERED_ROBOTS`` holds model
    names, not ``Robot``) and ``Environment._load_robots`` skips the ``robots`` config
    once a robot exists, so sensors must be requested on the template's robot itself.
    Passing the patched dict as ``scene_file`` overrides ``scene_instance``.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    robots = []
    for name, info in scene["objects_info"]["init_info"].items():
        if info.get("class_module", "").startswith("omnigibson.robots"):
            info["args"]["obs_modalities"] = ["rgb"]
            info["args"]["sensor_config"] = {
                "VisionSensor": {"sensor_kwargs": {"image_height": int(image_size), "image_width": int(image_size)}}
            }
            robots.append(name)
    if not robots:
        raise KeyError(f"No robot in {template_path}")
    return scene


def head_sensor(robot):
    for name, sensor in robot.sensors.items():
        if HEAD_SENSOR_LINK in name:
            return name, sensor
    raise KeyError(f"No {HEAD_SENSOR_LINK} camera among {list(robot.sensors)}")


def scene_ground_truth(env, robot, inventory_names):
    objects = []
    for obj in env.scene.objects:
        if obj.name == robot.name:
            continue
        category = getattr(obj, "category", None)
        record = {
            "name": obj.name,
            "category": category,
            "in_static_map": obj.name in inventory_names,
            "is_structure": category in STRUCTURE_CATEGORIES,
        }
        try:
            position, orientation = obj.get_position_orientation()
            record["position"] = to_list(position)
            record["quat_xyzw"] = to_list(orientation)
            lo, hi = obj.aabb
            record["aabb"] = [to_list(lo), to_list(hi)]
        except Exception as error:  # cloth / particle / unloaded objects
            record["error"] = f"{type(error).__name__}: {error}"
        objects.append(record)
    return objects


def capture_spin(env, robot, sensor_name, sensor, base_position, base_quat, frames_per_spin, settle_steps):
    """Teleport to ``base_position`` and record ``frames_per_spin`` head-camera frames around 360 degrees."""
    rgb, depth, seg, cam_pos, cam_quat, base_pos_out, base_quat_out, labels = [], [], [], [], [], [], [], []
    base_yaw0 = float(T.quat2euler(base_quat)[2].item())
    for k in range(frames_per_spin):
        yaw = base_yaw0 + 2.0 * math.pi * k / frames_per_spin
        place_robot(robot, base_position, yaw_quat(yaw))
        for _ in range(settle_steps):
            env.step({robot.name: controller_no_op_action(robot)})
        obs, info = robot.get_obs()
        frame = obs[sensor_name]
        frame_info = info.get(sensor_name, {})
        rgb.append(np.asarray(frame["rgb"].detach().cpu().numpy() if isinstance(frame["rgb"], th.Tensor) else frame["rgb"])[..., :3].astype(np.uint8))
        d = frame["depth_linear"]
        depth.append(np.asarray(d.detach().cpu().numpy() if isinstance(d, th.Tensor) else d, dtype=np.float32))
        s = frame.get("seg_instance")
        if s is not None:
            seg.append(np.asarray(s.detach().cpu().numpy() if isinstance(s, th.Tensor) else s).astype(np.int32))
        labels.append({str(key): str(value) for key, value in frame_info.get("seg_instance", {}).items()})
        pos, quat = sensor.get_position_orientation()
        cam_pos.append(to_list(pos))
        cam_quat.append(to_list(quat))
        bpos, bquat = robot.get_position_orientation()
        base_pos_out.append(to_list(bpos))
        base_quat_out.append(to_list(bquat))
    return {
        "rgb": np.stack(rgb),
        "depth": np.stack(depth),
        "seg_instance": np.stack(seg) if seg else np.zeros((0,), dtype=np.int32),
        "cam_pos": np.asarray(cam_pos, dtype=np.float64),
        "cam_quat_xyzw": np.asarray(cam_quat, dtype=np.float64),
        "base_pos": np.asarray(base_pos_out, dtype=np.float64),
        "base_quat_xyzw": np.asarray(base_quat_out, dtype=np.float64),
        "seg_labels": labels,
    }


def run_task(args, api, scene, task_spec, output_root):
    task, _, target = task_spec.partition(":")
    template_path = find_template(Path(args.task_instances_root), scene, task, args.task_instance_id)
    map_dir = b1k_map_directory(args.b1k_map_root, scene)
    with open(Path(map_dir).parent / "semantic_occupancy" / "scene_inventory.json", encoding="utf-8") as f:
        inventory = api["inventory_instances"](json.load(f))
    inventory_names = {inst.source_name for inst in inventory}
    if not target:
        target = "__search_target__"
    if api["resolve_target"](inventory, target):
        print(f"  target {target!r} resolves from the inventory; using a search-only sentinel for viewpoint selection")
        target = "__search_target__"

    robot_cfg = load_robot_config(args.robot_config)
    # Same path as the eval RGBDFullResWrapper: create the cameras with rgb, then widen the head camera.
    robot_cfg["obs_modalities"] = ["rgb"]
    cfg = build_env_config(scene_model=scene, robot_cfg=robot_cfg, scene_instance=template_path.stem, load_room_instances=None)
    cfg["scene"]["scene_file"] = patched_scene_file(template_path)
    started = time.time()
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    if robot.model in ("r1", "r1pro"):
        og.sim.stop()
        robot.base_footprint_link.mass = 250.0
        og.sim.play()
    env.reset(get_obs=False)
    print(f"  environment ready in {time.time() - started:.0f}s")

    og_resolution = float(env.scene.trav_map.map_resolution)
    robot_erosion_m = og_robot_erosion_meters(robot, og_resolution)
    costmap = api["planning_costmap_from_navigation_2d"](map_dir, robot_erosion_m, args.runtime_extra_clearance)
    traversable, obstacle, unknown = api["navigation_2d_layers"](map_dir)
    # Parity check against the cv2 construction the benchmark harness uses.
    from run_nav2py_benchmark import load_nav2py, make_b1k_costmap

    reference = make_b1k_costmap(map_dir, robot, load_nav2py(), og_resolution, erode_for_robot=True, extra_clearance=args.runtime_extra_clearance)
    costmap_parity = bool(np.array_equal(reference.data, costmap.data))
    print(f"  planning costmap parity with make_b1k_costmap: {costmap_parity} (plannable cells {(costmap.data < 253).sum()})")

    start_position, start_quat = template_robot_pose(template_path, args.robot_pose_key)
    start_position = th.tensor(start_position, dtype=th.float32)
    start_quat = th.tensor(start_quat, dtype=th.float32)
    policy = api["ObjectSearchPolicy"](
        costmap,
        obstacle | unknown,
        traversable | obstacle,
        inventory,
        target,
        max_viewpoints=args.max_viewpoints,
        view_radius_m=args.view_radius,
        confirm_with_planner=False,
    )

    out_dir = output_root / scene / task
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  robot sensors: {list(robot.sensors)}", flush=True)
    if not robot.sensors:
        # Diagnose why no camera was created (obs modalities / sensor config / link children).
        print(f"  robot._obs_modalities={getattr(robot, '_obs_modalities', None)} sensor_config={getattr(robot, '_sensor_config', None)}")
        for link_name, link in robot.links.items():
            children = [(str(p.GetPrimPath()), p.GetPrimTypeInfo().GetTypeName()) for p in link.prim.GetChildren()]
            if any(t == "Camera" for _, t in children) or "zed" in link_name or "realsense" in link_name:
                print(f"  link {link_name}: {children}")
        print(f"  robot cfg obs_modalities passed: {robot_cfg.get('obs_modalities')}", flush=True)
    sensor_name, sensor = head_sensor(robot)
    set_sensor_modalities(sensor, {"rgb", "depth_linear"} | ({"seg_instance"} if args.seg_instance else set()))
    og.sim.update_handles()
    env.load_observation_space()
    # A freshly attached annotator returns an empty tensor until it has rendered, and
    # ``_remap_instance_segmentation`` crashes on that (job 10567584); render first.
    for _ in range(3):
        og.sim.render()
    for attempt in range(5):
        try:
            robot.get_obs()
            break
        except RuntimeError as error:
            print(f"  warm-up get_obs attempt {attempt}: {error}", flush=True)
            og.sim.render()
    print(f"  head sensor {sensor_name}: modalities {sorted(sensor.modalities)} {sensor.image_height}x{sensor.image_width}", flush=True)
    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scene": scene,
                "task": task,
                "template": str(template_path),
                "robot_start_position": to_list(start_position),
                "robot_start_quat_xyzw": to_list(start_quat),
                "objects": scene_ground_truth(env, robot, inventory_names),
                "head_sensor": sensor_name,
                "image_size": [int(sensor.image_height), int(sensor.image_width)],
            },
            f,
            indent=1,
        )

    action = policy.begin((float(start_position[0]), float(start_position[1])))
    robot_xy = (float(start_position[0]), float(start_position[1]))
    position = start_position.clone()
    viewpoints = []
    index = 0
    while action.kind != "done":
        if action.kind == "spin":
            spin_started = time.time()
            frames = capture_spin(env, robot, sensor_name, sensor, position, start_quat, args.frames_per_spin, args.settle_steps)
            K = sensor.intrinsic_matrix.detach().cpu().numpy().astype(np.float64)
            labels = frames.pop("seg_labels")
            np.savez_compressed(out_dir / f"viewpoint_{index:02d}.npz", K=K, **frames)
            viewpoints.append(
                {
                    "index": index,
                    "commanded_position": to_list(position),
                    "frames": args.frames_per_spin,
                    "seg_labels": labels,
                    "capture_seconds": time.time() - spin_started,
                }
            )
            print(f"  viewpoint {index} at ({position[0]:.2f}, {position[1]:.2f}) captured in {time.time() - spin_started:.1f}s")
            index += 1
            robot_xy = (float(frames["base_pos"][-1][0]), float(frames["base_pos"][-1][1]))
            action = policy.report_spin(robot_xy, [])
        elif action.kind == "goto":
            x, y, _ = action.pose
            position = th.tensor([x, y, float(start_position[2])], dtype=th.float32)
            action = policy.report_arrival((x, y), reached=True)
        else:  # approach never happens without hypotheses
            break

    with open(out_dir / "viewpoints.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scene": scene,
                "task": task,
                "target_for_prior": target,
                "costmap": {
                    "map_dir": str(map_dir),
                    "robot_erosion_m": robot_erosion_m,
                    "extra_clearance_m": args.runtime_extra_clearance,
                    "parity_with_make_b1k_costmap": costmap_parity,
                    "plannable_cells": int((costmap.data < 253).sum()),
                },
                "viewpoints": viewpoints,
                "policy_log": policy.log,
                "final": action.result,
            },
            f,
            indent=1,
        )
    og.clear()


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    add_nav2py_to_path(args.nav2py_root)
    api = load_nav2py_objnav()
    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False
    output_root = Path(args.output_dir)
    try:
        for task_spec in args.task:
            print(f"\n### {args.scene} / {task_spec}", flush=True)
            try:
                run_task(args, api, args.scene, task_spec, output_root)
            except Exception:
                # og.shutdown() exits the process with status 0, so print the traceback first.
                traceback.print_exc()
                sys.stdout.flush()
                raise
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
