#!/usr/bin/env python3
"""Generate one object-reference navigation episode from a task instance."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch as th

import omnigibson as og
from generate_nav_benchmark import (
    CHALLENGE_SCENES,
    build_env_config,
    episode_points_are_valid,
    eroded_floor_map,
    load_robot_config,
    point_is_free,
    seed_everything,
    to_float_list,
)
from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_ROOMS
from omnigibson.macros import gm
from omnigibson.object_states import Open
from omnigibson.utils.motion_planning_utils import astar


def parse_radii(value: str) -> list[float]:
    radii = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not radii or any(radius < 0 for radius in radii):
        raise ValueError("--approach-radii must contain non-negative numbers")
    return radii


def quat_xyzw_to_yaw(quat: list[float]) -> float:
    x, y, z, w = [float(value) for value in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def point_in_loaded_rooms(env: og.Environment, point: th.Tensor) -> bool:
    rooms = env.scene.load_room_instances
    return rooms is None or env.scene.seg_map.get_room_instance_by_point(point[:2]) in rooms


def clearance_floor_map(env: og.Environment, floor_trav_map: th.Tensor, extra_clearance: float) -> th.Tensor:
    if extra_clearance < 0:
        raise ValueError("--extra-clearance must be non-negative")
    if extra_clearance == 0:
        return floor_trav_map
    resolution = float(env.scene.trav_map.map_resolution)
    kernel_size = int(math.ceil(extra_clearance / resolution))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return th.tensor(cv2.erode(floor_trav_map.cpu().numpy(), kernel))


def clearance_path_distance(
    env: og.Environment, trav_map: th.Tensor, start: th.Tensor, goal: th.Tensor
) -> float | None:
    if not point_is_free(env.scene, trav_map, start) or not point_is_free(env.scene, trav_map, goal):
        return None
    start_cell = tuple(env.scene.trav_map.world_to_map(start[:2]).tolist())
    goal_cell = tuple(env.scene.trav_map.world_to_map(goal[:2]).tolist())
    path = astar(trav_map, start_cell, goal_cell)
    if path is None:
        return None
    path_world = env.scene.trav_map.map_to_world(path)
    return float(th.sum(th.norm(path_world[1:] - path_world[:-1], dim=1)).item())


def project_goal_near_object(
    env: og.Environment,
    floor_trav_map: th.Tensor,
    clearance_trav_map: th.Tensor,
    start_position: list[float],
    object_pos: list[float],
    radii: list[float],
    angles_per_radius: int,
    min_distance: float,
    max_distance: float,
) -> dict[str, Any] | None:
    start = th.tensor(start_position, dtype=th.float32)
    object_xy = np.asarray(object_pos[:2], dtype=np.float64)
    candidates = []
    for radius in radii:
        count = 1 if radius == 0 else angles_per_radius
        for index in range(count):
            angle = 0.0 if radius == 0 else 2.0 * math.pi * index / angles_per_radius
            xy = object_xy + radius * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            point = th.tensor([xy[0], xy[1], start_position[2]], dtype=th.float32)
            if not point_is_free(env.scene, floor_trav_map, point) or not point_in_loaded_rooms(env, point):
                continue
            distance = clearance_path_distance(env, clearance_trav_map, start, point)
            if distance is None or distance < min_distance or distance > max_distance:
                continue
            candidates.append(
                {
                    "goal_position": to_float_list(point),
                    "geodesic_distance": distance,
                    "goal_projection_radius": float(radius),
                    "goal_projection_angle": float(angle),
                }
            )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item["goal_projection_radius"], item["geodesic_distance"]))


def find_templates(root: Path, scene: str, task: str) -> dict[int, Path]:
    prefix = f"{scene}_task_{task}_0_"
    suffix = "_template.json"
    matches = {}
    for path in sorted(root.rglob(f"{prefix}*{suffix}")):
        instance_text = path.name[len(prefix) : -len(suffix)]
        if instance_text.isdigit():
            matches.setdefault(int(instance_text), path)
    if not matches:
        raise FileNotFoundError(f"No task templates for {scene}/{task} under {root}")
    return matches


def find_template(root: Path, scene: str, task: str, instance_id: int = 0) -> Path:
    templates = find_templates(root, scene, task)
    if instance_id not in templates:
        raise FileNotFoundError(f"No task-instance {instance_id} for {scene}/{task} under {root}")
    return templates[instance_id]


def template_robot_pose(template_path: Path, robot_pose_key: str) -> tuple[list[float], list[float]]:
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)
    poses = template.get("metadata", {}).get("task", {}).get("robot_poses", {}).get(robot_pose_key)
    if not poses:
        raise KeyError(f"No {robot_pose_key} pose in {template_path}")
    pose = poses[0]
    return [float(value) for value in pose["position"]], [float(value) for value in pose["orientation"]]


@dataclass
class PairContext:
    env: og.Environment
    task: str
    scene: str
    floor: int
    task_instance_id: int
    template_path: Path
    floor_trav_map: th.Tensor
    clearance_trav_map: th.Tensor
    initial_position: list[float]
    initial_quat: list[float]
    radii: list[float]
    angles_per_radius: int
    min_distance: float
    max_distance: float
    settle_steps: int
    state_overrides: list[dict[str, str]]


def apply_state_overrides(env: og.Environment, overrides: list[dict[str, str]]) -> None:
    for override in overrides:
        obj = env.scene.object_registry("name", override["object"], None)
        if obj is None:
            raise KeyError(f"State-override object not found: {override['object']}")
        if override["state"] != "open" or Open not in obj.states:
            raise ValueError(f"Only Open-capable objects support state overrides: {obj.name}")
        if not obj.states[Open].set_value(True, fully=True):
            raise RuntimeError(f"Could not open {obj.name}")
    if overrides:
        og.sim.step()


def create_context(args: argparse.Namespace, template_path: Path, task: str, task_instance_id: int) -> PairContext:
    room_instances = TASK_NAMES_TO_ROOMS.get(task)
    robot_cfg = load_robot_config(args.robot_config)
    cfg = build_env_config(
        scene_model=args.scene,
        robot_cfg=copy.deepcopy(robot_cfg),
        scene_instance=template_path.stem,
        load_room_instances=room_instances,
    )
    cache_dir = Path(gm.APPDATA_PATH) / "global" / "cache" / "texturecache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = og.Environment(configs=cfg)
    if env.robots[0].model in ("r1", "r1pro"):
        og.sim.stop()
        env.robots[0].base_footprint_link.mass = 250.0
        og.sim.play()

    env.reset(get_obs=False)
    apply_state_overrides(env, args.state_overrides)
    floor_trav_map = eroded_floor_map(env.scene, args.floor, env.robots[0])
    initial_position, initial_quat = template_robot_pose(template_path, args.robot_pose_key)
    return PairContext(
        env=env,
        task=task,
        scene=args.scene,
        floor=args.floor,
        task_instance_id=task_instance_id,
        template_path=template_path,
        floor_trav_map=floor_trav_map,
        clearance_trav_map=clearance_floor_map(env, floor_trav_map, args.extra_clearance),
        initial_position=initial_position,
        initial_quat=initial_quat,
        radii=args.radii,
        angles_per_radius=args.angles_per_radius,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        settle_steps=args.settle_steps,
        state_overrides=args.state_overrides,
    )


def close_context(context: PairContext) -> None:
    og.clear()


def object_position(context: PairContext, reference: str) -> list[float]:
    obj = context.env.scene.object_registry("name", reference, None)
    if obj is None:
        raise KeyError(f"Object reference not found: {reference}")
    position, _ = obj.get_position_orientation()
    return to_float_list(position)


def approach_candidates(context: PairContext, object_pos: list[float], z: float) -> list[list[float]]:
    candidates = []
    object_xy = np.asarray(object_pos[:2], dtype=np.float64)
    for radius in context.radii:
        count = 1 if radius == 0 else context.angles_per_radius
        for index in range(count):
            angle = 0.0 if radius == 0 else 2.0 * math.pi * index / context.angles_per_radius
            xy = object_xy + radius * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            point = th.tensor([xy[0], xy[1], z], dtype=th.float32)
            if point_is_free(context.env.scene, context.floor_trav_map, point) and point_in_loaded_rooms(
                context.env, point
            ):
                candidates.append(to_float_list(point))
    return candidates


def resolve_start_candidates(
    context: PairContext, start_reference: str, start_position: list[float] | None
) -> tuple[list[list[float]], list[float]]:
    if start_position is not None:
        return [start_position], context.initial_quat
    if start_reference == "robot_initial":
        return [context.initial_position], context.initial_quat
    return (
        approach_candidates(context, object_position(context, start_reference), context.initial_position[2]),
        context.initial_quat,
    )


def generate_pair_episode(
    context: PairContext,
    start_reference: str,
    goal_reference: str,
    episode_id: str,
    start_position: list[float] | None = None,
) -> dict[str, Any]:
    target_position = object_position(context, goal_reference)
    start_candidates, start_quat = resolve_start_candidates(context, start_reference, start_position)
    best = None
    for candidate in start_candidates:
        projection = project_goal_near_object(
            env=context.env,
            floor_trav_map=context.floor_trav_map,
            clearance_trav_map=context.clearance_trav_map,
            start_position=candidate,
            object_pos=target_position,
            radii=context.radii,
            angles_per_radius=context.angles_per_radius,
            min_distance=context.min_distance,
            max_distance=context.max_distance,
        )
        if projection is None:
            continue
        candidate_key = (projection["goal_projection_radius"], projection["geodesic_distance"])
        if best is None or candidate_key < best[0]:
            best = (candidate_key, candidate, projection)
    if best is None:
        raise RuntimeError(f"No valid path for {start_reference} -> {goal_reference}")

    _, resolved_start, projection = best
    context.env.reset(get_obs=False)
    apply_state_overrides(context.env, context.state_overrides)
    start_tensor = th.tensor(resolved_start, dtype=th.float32)
    goal_tensor = th.tensor(projection["goal_position"], dtype=th.float32)
    quat_tensor = th.tensor(start_quat, dtype=th.float32)
    if not episode_points_are_valid(
        context.env, context.floor_trav_map, start_tensor, goal_tensor, quat_tensor, context.settle_steps
    ):
        raise RuntimeError(f"Start pose is invalid after settling for {start_reference} -> {goal_reference}")
    settled_position, _ = context.env.robots[0].get_position_orientation()
    if not point_is_free(context.env.scene, context.clearance_trav_map, settled_position[:2]):
        raise RuntimeError(f"Start pose lacks extra clearance for {start_reference} -> {goal_reference}")

    return {
        "episode_id": episode_id,
        "scene_model": context.scene,
        "floor": context.floor,
        "start_position": resolved_start,
        "start_yaw": quat_xyzw_to_yaw(start_quat),
        "start_quat": start_quat,
        "goal_position": projection["goal_position"],
        "geodesic_distance": projection["geodesic_distance"],
        "task_name": context.task,
        "task_instance_id": context.task_instance_id,
        "template_path": str(context.template_path),
        "scene_instance": context.template_path.stem,
        "load_room_instances": TASK_NAMES_TO_ROOMS.get(context.task),
        "start_reference": start_reference,
        "goal_reference": goal_reference,
        "goal_reference_position": target_position,
        "goal_source": "nearest_traversable_point_near_object_reference",
        "state_overrides": context.state_overrides,
        **projection,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-instances-root", default="datasets")
    parser.add_argument("--scene", choices=CHALLENGE_SCENES, default="house_double_floor_lower")
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-instance-id", type=int, default=0)
    parser.add_argument("--start", required=True, help="robot_initial or one task-instance object name")
    parser.add_argument("--goal", required=True, help="One task-instance object or region name")
    parser.add_argument("--open-object", action="append", default=[], help="Object to initialize fully open.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--floor", type=int, default=0)
    parser.add_argument("--min-distance", type=float, default=1.0)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--approach-radii", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--angles-per-radius", type=int, default=36)
    parser.add_argument("--extra-clearance", type=float, default=0.2)
    parser.add_argument("--robot-pose-key", default="R1Pro")
    parser.add_argument(
        "--robot-config",
        default=str(Path(__file__).resolve().parents[2] / "omnigibson" / "eval" / "r1pro.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_distance > args.max_distance:
        raise ValueError("--min-distance must be <= --max-distance")
    args.task_instances_root = Path(args.task_instances_root).expanduser()
    args.radii = parse_radii(args.approach_radii)
    args.state_overrides = [{"object": name, "state": "open"} for name in args.open_object]
    seed_everything(args.seed)
    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False

    template_path = find_template(args.task_instances_root, args.scene, args.task, args.task_instance_id)
    context = create_context(args, template_path, args.task, args.task_instance_id)
    try:
        episode = generate_pair_episode(context, args.start, args.goal, f"{args.scene}_{args.task}_000")
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"episodes": [episode]}, f, indent=2)
            f.write("\n")
    finally:
        close_context(context)
        og.shutdown()


if __name__ == "__main__":
    main()
