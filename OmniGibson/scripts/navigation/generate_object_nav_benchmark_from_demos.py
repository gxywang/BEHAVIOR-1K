#!/usr/bin/env python3
"""Generate object-goal navigation benchmark episodes from BEHAVIOR demos.

This keeps the existing random point-navigation generator untouched. It reads
LeRobot demo annotations to find navigation skills, maps annotated scene object
names through the task template's ``metadata.task.inst_to_name``, reads the
actual object and robot poses from the matching ``*-tro_state.json``, then
projects each object target to a nearby traversable point.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th

import omnigibson as og
from generate_nav_benchmark import (
    CHALLENGE_SCENES,
    build_env_config,
    controller_no_op_action,
    eroded_floor_map,
    episode_points_are_valid,
    load_robot_config,
    point_is_free,
    seed_everything,
    to_float_list,
)
from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_ROOMS
from omnigibson.macros import gm


DEFAULT_OUTPUT = "outputs/navigation/object_nav_benchmark.json"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-root", default="datasets/demonstrations", help="LeRobot demo dataset root.")
    parser.add_argument(
        "--task-instances-root",
        default="datasets",
        help="Root containing task template JSONs and matching *-tro_state.json files.",
    )
    parser.add_argument("--scene", choices=CHALLENGE_SCENES, default="house_double_floor_lower")
    parser.add_argument("--task", action="append", default=[], help="Task name to include. Repeat as needed.")
    parser.add_argument("--num-episodes-per-task", type=int, default=20)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--floor", type=int, default=0)
    parser.add_argument("--min-distance", type=float, default=1.0)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--approach-radii",
        default="0.5,0.75,1.0,1.25,1.5,2.0",
        help="Comma-separated candidate radii in meters around the target object.",
    )
    parser.add_argument("--angles-per-radius", type=int, default=36)
    parser.add_argument(
        "--start-source",
        choices=("tro-initial", "reconstructed-state", "reconstructed-action"),
        default="tro-initial",
        help=(
            "tro-initial uses exact TRO robot_poses and only accepts navigation skills starting at frame 0. "
            "reconstructed-state/action integrate LeRobot base velocities to approximate later skill starts."
        ),
    )
    parser.add_argument("--robot-pose-key", default="R1Pro", help="Preferred key inside tro_state robot_poses.")
    parser.add_argument(
        "--robot-config",
        default=str(Path(__file__).resolve().parents[2] / "omnigibson" / "eval" / "r1pro.yaml"),
    )
    return parser.parse_args()


def parse_radii(value: str) -> list[float]:
    radii = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not radii or any(radius < 0 for radius in radii):
        raise ValueError("--approach-radii must contain non-negative numbers")
    return radii


def normalize_task_name(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected one task name, got {value}")
        return str(value[0])
    return str(value).strip().strip("[]'\"")


def quat_xyzw_to_yaw(quat: list[float]) -> float:
    x, y, z, w = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat_xyzw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]


def flatten_object_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        names = []
        for item in value:
            names.extend(flatten_object_ids(item))
        return names
    return []


def load_episode_rows(demo_root: Path) -> pd.DataFrame:
    paths = sorted((demo_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet files found under {demo_root / 'meta' / 'episodes'}")
    rows = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    rows["task_name"] = rows["tasks"].map(normalize_task_name)
    return rows.sort_values(["task_name", "episode_index"]).reset_index(drop=True)


def load_annotation(demo_root: Path, row: pd.Series) -> dict[str, Any]:
    path = demo_root / str(row["annotation_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def navigation_skills(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    skills = []
    for skill in annotation.get("skill_annotation", []):
        skill_type = skill.get("skill_type", [])
        is_navigation = skill_type == "navigation" or (
            isinstance(skill_type, list) and "navigation" in skill_type
        )
        if is_navigation:
            skills.append(skill)
    return skills


def index_templates(root: Path, scene: str) -> dict[str, Path]:
    prefix = f"{scene}_task_"
    suffix = "_0_0_template.json"
    templates = {}
    for path in sorted(root.rglob(f"{prefix}*{suffix}")):
        task_name = path.name[len(prefix) : -len(suffix)]
        templates.setdefault(task_name, path)
    return templates


def index_tro_states(root: Path, scene: str) -> dict[tuple[str, int], Path]:
    prefix = f"{scene}_task_"
    suffix = "_template-tro_state.json"
    states = {}
    for path in sorted(root.rglob(f"{prefix}*{suffix}")):
        rest = path.name[len(prefix) : -len(suffix)]
        if "_0_" not in rest:
            continue
        task_name, instance_text = rest.rsplit("_0_", 1)
        if instance_text.isdigit():
            states.setdefault((task_name, int(instance_text)), path)
    return states


def load_template_mapping(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        template = json.load(f)
    mapping = template.get("metadata", {}).get("task", {}).get("inst_to_name")
    if not isinstance(mapping, dict):
        raise ValueError(f"Missing metadata.task.inst_to_name in {path}")
    return mapping


def load_tro_state(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_robot_pose(tro_state: dict[str, Any], preferred_key: str) -> tuple[list[float], list[float], str]:
    robot_poses = tro_state.get("robot_poses", {})
    for key in (preferred_key, "robot", preferred_key.lower()):
        poses = robot_poses.get(key)
        if poses:
            pose = poses[0]
            return [float(v) for v in pose["position"]], [float(v) for v in pose["orientation"]], key
    raise KeyError(f"No robot pose found for {preferred_key} or robot")


def object_position(tro_state: dict[str, Any], bddl_name: str) -> list[float]:
    state = tro_state.get(bddl_name)
    if state is None:
        raise KeyError(f"{bddl_name} not found in TRO state")
    pos = state.get("root_link", {}).get("pos")
    if pos is None:
        raise KeyError(f"{bddl_name} has no root_link.pos in TRO state")
    return [float(v) for v in pos]


def resolve_data_path(demo_root: Path, row: pd.Series) -> Path:
    chunk_index = int(row["data/chunk_index"])
    file_index = int(row["data/file_index"])
    candidates = [
        demo_root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet",
        demo_root / "data" / f"file-{file_index:03d}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No data parquet found. Tried: {candidates}")


def load_episode_frames(demo_root: Path, row: pd.Series) -> pd.DataFrame:
    path = resolve_data_path(demo_root, row)
    df = pd.read_parquet(path, columns=["episode_index", "frame_index", "timestamp", "action", "observation.state"])
    df = df[df["episode_index"] == int(row["episode_index"])].copy()
    if df.empty:
        raise ValueError(f"No frame rows found for episode_index={int(row['episode_index'])} in {path}")
    return df.sort_values("frame_index").reset_index(drop=True)


def matrix_from_column(values: pd.Series) -> np.ndarray:
    return np.stack(values.to_numpy()).astype(np.float64, copy=False)


def reconstruct_base_trajectory(
    frames: pd.DataFrame,
    initial_position: list[float],
    initial_quat: list[float],
    source: str,
) -> np.ndarray:
    column = "observation.state" if source == "reconstructed-state" else "action"
    local_vel = matrix_from_column(frames[column])[:, :3]
    timestamps = frames["timestamp"].to_numpy(dtype=np.float64)
    frame_indices = frames["frame_index"].to_numpy(dtype=np.int64)
    dt = np.diff(timestamps, prepend=timestamps[0])
    dt[0] = float(np.median(dt[1:])) if len(dt) > 1 else 1.0 / 30.0

    trajectory = np.zeros((len(frames), 5), dtype=np.float64)
    x, y, z = [float(v) for v in initial_position]
    yaw = quat_xyzw_to_yaw(initial_quat)
    trajectory[0] = [frame_indices[0], x, y, z, yaw]
    for i in range(1, len(frames)):
        vx, vy, wz = local_vel[i - 1]
        x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * dt[i]
        y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * dt[i]
        yaw += wz * dt[i]
        trajectory[i] = [frame_indices[i], x, y, z, yaw]
    return trajectory


def skill_start_pose(
    demo_root: Path,
    row: pd.Series,
    skill: dict[str, Any],
    tro_position: list[float],
    tro_quat: list[float],
    start_source: str,
    trajectory_cache: dict[int, np.ndarray],
) -> tuple[list[float], list[float], str] | None:
    frame_start = int(skill.get("frame_duration", [0, 0])[0])
    if frame_start == 0:
        return tro_position, tro_quat, "tro-initial"
    if start_source == "tro-initial":
        return None

    episode_index = int(row["episode_index"])
    if episode_index not in trajectory_cache:
        frames = load_episode_frames(demo_root, row)
        trajectory_cache[episode_index] = reconstruct_base_trajectory(frames, tro_position, tro_quat, start_source)
    trajectory = trajectory_cache[episode_index]
    frame_indices = trajectory[:, 0].astype(np.int64)
    idx = int(np.searchsorted(frame_indices, frame_start, side="left"))
    if idx >= len(trajectory):
        return None
    pose = trajectory[idx]
    return [float(pose[1]), float(pose[2]), float(pose[3])], yaw_to_quat_xyzw(float(pose[4])), start_source


def point_in_loaded_rooms(env: og.Environment, point: th.Tensor) -> bool:
    rooms = env.scene.load_room_instances
    if rooms is None:
        return True
    return env.scene.seg_map.get_room_instance_by_point(point[:2]) in rooms


def shortest_path_distance(env: og.Environment, floor: int, start: th.Tensor, goal: th.Tensor) -> float | None:
    _, distance = env.scene.get_shortest_path(floor, start[:2], goal[:2], entire_path=False, robot=env.robots[0])
    if distance is None:
        return None
    return float(distance.item() if hasattr(distance, "item") else distance)


def project_goal_near_object(
    env: og.Environment,
    floor_trav_map: th.Tensor,
    floor: int,
    start_position: list[float],
    object_pos: list[float],
    radii: list[float],
    angles_per_radius: int,
    min_distance: float,
    max_distance: float,
) -> dict[str, Any] | None:
    start = th.tensor(start_position, dtype=th.float32)
    object_xy = np.array(object_pos[:2], dtype=np.float64)
    candidates = []

    for radius in radii:
        angle_count = 1 if radius == 0 else angles_per_radius
        for angle_idx in range(angle_count):
            angle = 0.0 if radius == 0 else (2.0 * math.pi * angle_idx / angles_per_radius)
            xy = object_xy + radius * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            point = th.tensor([xy[0], xy[1], start_position[2]], dtype=th.float32)
            if not point_is_free(env.scene, floor_trav_map, point):
                continue
            if not point_in_loaded_rooms(env, point):
                continue
            distance = shortest_path_distance(env, floor, start, point)
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


def build_episode(
    env: og.Environment,
    floor_trav_map: th.Tensor,
    row: pd.Series,
    skill: dict[str, Any],
    target_name: str,
    bddl_name: str,
    object_pos: list[float],
    start_position: list[float],
    start_quat: list[float],
    start_source: str,
    args: argparse.Namespace,
    episode_idx: int,
) -> dict[str, Any] | None:
    start_tensor = th.tensor(start_position, dtype=th.float32)
    if not point_in_loaded_rooms(env, start_tensor):
        return None

    projection = project_goal_near_object(
        env=env,
        floor_trav_map=floor_trav_map,
        floor=args.floor,
        start_position=start_position,
        object_pos=object_pos,
        radii=args.radii,
        angles_per_radius=args.angles_per_radius,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
    )
    if projection is None:
        return None

    env.reset(get_obs=False)
    goal_tensor = th.tensor(projection["goal_position"], dtype=th.float32)
    quat_tensor = th.tensor(start_quat, dtype=th.float32)
    if not episode_points_are_valid(env, floor_trav_map, start_tensor, goal_tensor, quat_tensor, args.settle_steps):
        return None

    frame_duration = skill.get("frame_duration", [])
    return {
        "episode_id": f"{args.scene}_{row['task_name']}_{episode_idx:03d}",
        "scene_model": args.scene,
        "floor": int(args.floor),
        "start_position": [float(v) for v in start_position],
        "start_yaw": quat_xyzw_to_yaw(start_quat),
        "start_quat": [float(v) for v in start_quat],
        "goal_position": projection["goal_position"],
        "geodesic_distance": projection["geodesic_distance"],
        "task_name": row["task_name"],
        "scene_instance": args.scene_instance,
        "load_room_instances": args.load_room_instances,
        "demo_episode_index": int(row["episode_index"]),
        "demo_raw_episode_id": int(row["raw_episode_id"]),
        "task_instance_id": int(row["task_instance_id"]),
        "annotation_path": str(row["annotation_path"]),
        "navigation_skill_idx": int(skill.get("skill_idx", -1)),
        "navigation_skill_frame_duration": [int(v) for v in frame_duration],
        "start_source": start_source,
        "target_object_name": target_name,
        "target_bddl_instance": bddl_name,
        "target_object_position": [float(v) for v in object_pos],
        "goal_source": "nearest_traversable_point_near_tro_object_pose",
        **projection,
    }


def generate_task_episodes(
    task_name: str,
    rows: pd.DataFrame,
    template_path: Path,
    tro_index: dict[tuple[str, int], Path],
    robot_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    args.scene_instance = template_path.stem
    args.load_room_instances = TASK_NAMES_TO_ROOMS.get(task_name)
    inst_to_name = load_template_mapping(template_path)
    name_to_inst = {name: inst for inst, name in inst_to_name.items()}

    cfg = build_env_config(
        scene_model=args.scene,
        robot_cfg=copy.deepcopy(robot_cfg),
        scene_instance=args.scene_instance,
        load_room_instances=args.load_room_instances,
    )

    appdata_cache = Path(gm.APPDATA_PATH) / "global" / "cache" / "texturecache"
    appdata_cache.mkdir(parents=True, exist_ok=True)

    log(f"\nLoaded template: {args.scene_instance}")
    log(f"Task: {task_name}")
    log(f"Demo rows available for task: {len(rows)}")
    env = og.Environment(configs=cfg)
    if env.robots[0].model in ("r1", "r1pro"):
        og.sim.stop()
        env.robots[0].base_footprint_link.mass = 250.0
        og.sim.play()

    episodes = []
    skipped = {}
    trajectory_cache = {}
    floor_trav_map = eroded_floor_map(env.scene, args.floor, env.robots[0])

    for _, row in rows.iterrows():
        if len(episodes) >= args.num_episodes_per_task:
            break
        tro_path = tro_index.get((task_name, int(row["task_instance_id"])))
        if tro_path is None:
            skipped["missing_tro"] = skipped.get("missing_tro", 0) + 1
            continue

        annotation = load_annotation(args.demo_root, row)
        skills = navigation_skills(annotation)
        if not skills:
            skipped["no_navigation_skill"] = skipped.get("no_navigation_skill", 0) + 1
            continue

        tro_state = load_tro_state(tro_path)
        tro_position, tro_quat, robot_pose_key = get_robot_pose(tro_state, args.robot_pose_key)
        for skill in skills:
            if len(episodes) >= args.num_episodes_per_task:
                break
            start_pose = skill_start_pose(
                args.demo_root,
                row,
                skill,
                tro_position,
                tro_quat,
                args.start_source,
                trajectory_cache,
            )
            if start_pose is None:
                skipped["nonzero_skill_start_without_reconstruction"] = (
                    skipped.get("nonzero_skill_start_without_reconstruction", 0) + 1
                )
                continue
            start_position, start_quat, used_start_source = start_pose
            target_names = flatten_object_ids(skill.get("object_id", []))
            for target_name in dict.fromkeys(target_names):
                if len(episodes) >= args.num_episodes_per_task:
                    break
                bddl_name = name_to_inst.get(target_name)
                if bddl_name is None:
                    skipped["unmapped_target"] = skipped.get("unmapped_target", 0) + 1
                    continue
                try:
                    obj_pos = object_position(tro_state, bddl_name)
                except KeyError:
                    skipped["missing_object_pose"] = skipped.get("missing_object_pose", 0) + 1
                    continue
                episode = build_episode(
                    env=env,
                    floor_trav_map=floor_trav_map,
                    row=row,
                    skill=skill,
                    target_name=target_name,
                    bddl_name=bddl_name,
                    object_pos=obj_pos,
                    start_position=start_position,
                    start_quat=start_quat,
                    start_source=used_start_source,
                    args=args,
                    episode_idx=len(episodes),
                )
                if episode is None:
                    skipped["invalid_or_unreachable"] = skipped.get("invalid_or_unreachable", 0) + 1
                    continue
                episode["template_path"] = str(template_path)
                episode["tro_state_path"] = str(tro_path)
                episode["robot_pose_key"] = robot_pose_key
                episodes.append(episode)
                log(
                    f"  [{len(episodes):03d}] {episode['episode_id']}: "
                    f"{target_name} -> {episode['goal_position']} "
                    f"({episode['geodesic_distance']:.3f} m)"
                )

    og.clear()
    log(f"Generated {len(episodes)} episodes for {task_name}. Skipped: {skipped}")
    return episodes


def write_benchmark(path: Path, args: argparse.Namespace, robot_cfg: dict[str, Any], episodes: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "robot": robot_cfg["model"],
                "floor": args.floor,
                "min_distance": args.min_distance,
                "max_distance": args.max_distance,
                "settle_steps": args.settle_steps,
                "goal_source": "nearest_traversable_point_near_tro_object_pose",
                "pose_source": "template inst_to_name + matching tro_state poses",
                "start_source": args.start_source,
                "approach_radii": args.radii,
                "angles_per_radius": args.angles_per_radius,
                "episodes": episodes,
            },
            f,
            indent=2,
        )
        f.write("\n")


def main() -> None:
    args = parse_args()
    if args.num_episodes_per_task < 1:
        raise ValueError("--num-episodes-per-task must be at least 1")
    if args.min_distance > args.max_distance:
        raise ValueError("--min-distance must be <= --max-distance")
    if args.angles_per_radius < 1:
        raise ValueError("--angles-per-radius must be at least 1")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative")

    args.demo_root = Path(args.demo_root).expanduser()
    args.task_instances_root = Path(args.task_instances_root).expanduser()
    args.radii = parse_radii(args.approach_radii)

    seed_everything(args.seed)
    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False

    rows = load_episode_rows(args.demo_root)
    if args.task:
        rows = rows[rows["task_name"].isin(args.task)]

    templates = index_templates(args.task_instances_root, args.scene)
    tro_states = index_tro_states(args.task_instances_root, args.scene)
    tasks = sorted(set(rows["task_name"]) & set(templates))
    log(f"Demo root: {args.demo_root}")
    log(f"Task-instances root: {args.task_instances_root}")
    log(f"Scene: {args.scene}")
    log(f"Demo rows after task filter: {len(rows)}")
    log(f"Template tasks found for scene: {len(templates)}")
    log(f"TRO states found for scene: {len(tro_states)}")
    log(f"Tasks selected: {tasks}")
    if args.task:
        missing = sorted(set(args.task) - set(tasks))
        if missing:
            raise ValueError(f"Requested tasks have no matching demo rows and template: {missing}")
    if not tasks:
        raise ValueError(f"No demo tasks with templates found for scene {args.scene}")

    robot_cfg = load_robot_config(args.robot_config)
    out_path = Path(args.output).expanduser()
    try:
        for task_name in tasks:
            task_rows = rows[rows["task_name"] == task_name]
            task_path = out_path.parent / f"{out_path.stem}_{args.scene}_{task_name}.json"
            if args.skip_existing and task_path.is_file():
                print(f"Skipping existing object-goal benchmark: {task_path}")
                continue
            episodes = generate_task_episodes(
                task_name=task_name,
                rows=task_rows,
                template_path=templates[task_name],
                tro_index=tro_states,
                robot_cfg=robot_cfg,
                args=args,
            )
            if not episodes:
                raise RuntimeError(f"No valid object-goal episodes were generated for {task_name}")
            tmp_path = task_path.with_suffix(".json.tmp")
            write_benchmark(tmp_path, args, robot_cfg, episodes)
            tmp_path.replace(task_path)
            log(f"Wrote object-goal benchmark: {task_path}")
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
