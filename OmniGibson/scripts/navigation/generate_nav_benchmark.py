import argparse
import copy
import heapq
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch as th
import yaml

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.controllers import ControllerView
from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_ROOMS
from omnigibson.macros import gm
from omnigibson.tasks.behavior_task import BehaviorTask


CHALLENGE_SCENES = (
    "house_double_floor_lower",
    "house_double_floor_upper",
    "house_single_floor",
    "office_cubicles_right",
    "restaurant_diner",
    "hotel_suite_large",
    "Rs_int",
)

DEFAULT_OUTPUT = "outputs/navigation/nav_benchmark_test.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Sample R1Pro point-navigation benchmark episodes.")
    parser.add_argument("--scene", choices=CHALLENGE_SCENES, default="house_single_floor")
    parser.add_argument("--task", action="append", default=[], help="Task name to include. Repeat as needed.")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes per task template")
    parser.add_argument("--all-scenes", action="store_true", help="Generate episodes for all challenge scenes")
    parser.add_argument(
        "--num-episodes-per-scene",
        type=int,
        default=None,
        help="Override episode count per task template in each selected scene",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip task benchmarks whose final output file already exists"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-distance", type=float, default=1.0)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--max-trials", type=int, default=500)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--extra-clearance",
        type=float,
        default=0.2,
        help="Additional obstacle clearance in meters beyond OmniGibson's robot-base erosion.",
    )
    parser.add_argument(
        "--safety-clearance",
        type=float,
        default=0.1,
        help=(
            "Additional generation-only clearance margin in meters. This avoids boundary cells that satisfy "
            "--extra-clearance statically but can be rejected by nav2py's dynamic safety envelope at startup."
        ),
    )
    parser.add_argument(
        "--robot-config",
        default=str(Path(__file__).resolve().parents[2] / "omnigibson" / "eval" / "r1pro.yaml"),
    )
    return parser.parse_args()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def to_float_list(value):
    if isinstance(value, th.Tensor):
        value = value.detach().cpu().tolist()
    return [float(x) for x in value]


def load_robot_config(path):
    with open(path, "r", encoding="utf-8") as f:
        robot_cfg = yaml.safe_load(f)

    robot_cfg = dict(robot_cfg)
    robot_cfg.pop("eval", None)
    robot_cfg["obs_modalities"] = []
    robot_cfg["position"] = [-50.0, -50.0, 0.0]
    return robot_cfg


def build_env_config(scene_model, robot_cfg, scene_instance, load_room_instances):
    return {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
            "automatic_reset": False,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "scene_instance": scene_instance,
            "trav_map_resolution": 0.1,
            "default_erosion_radius": 0.0,
            "trav_map_with_objects": True,
            "num_waypoints": 1,
            "waypoint_resolution": 0.2,
            "load_room_types": None,
            "load_room_instances": load_room_instances,
            "include_robots": False,
        },
        "robots": [robot_cfg],
        "objects": [],
        "task": {
            "type": "DummyTask",
            "include_obs": False,
        },
    }


def controller_no_op_action(robot):
    action = []
    for group_key, controller_idx in robot.controllers.values():
        action.append(ControllerView.compute_no_op_action(group_key, controller_idx).float())
    return th.cat(action) if action else th.empty(0, dtype=th.float32)


def place_robot(robot, position, orientation):
    robot.set_joint_positions(robot.reset_joint_pos, drive=False)
    robot.set_position_orientation(position=position, orientation=orientation)
    robot.set_linear_velocity(th.zeros(3))
    robot.set_angular_velocity(th.zeros(3))
    robot.set_joint_velocities(th.zeros(robot.n_dof), drive=False)


def eroded_floor_map(scene, floor, robot):
    trav_map = th.clone(scene.trav_map.floor_map[floor])
    return scene.trav_map._erode_trav_map(trav_map, robot=robot)


def disk_kernel(radius_m, resolution):
    radius_cells = int(math.ceil(radius_m / resolution))
    offsets = np.arange(-radius_cells, radius_cells + 1)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    return ((xx * resolution) ** 2 + (yy * resolution) ** 2 <= radius_m**2).astype(np.uint8)


def clearance_floor_map(env, floor_trav_map, extra_clearance):
    if extra_clearance < 0:
        raise ValueError("--extra-clearance must be non-negative")
    if extra_clearance == 0:
        return floor_trav_map

    resolution = float(env.scene.trav_map.map_resolution)
    kernel = disk_kernel(extra_clearance, resolution)
    return th.tensor(cv2.erode(floor_trav_map.cpu().numpy(), kernel))


def point_to_map_cell(scene, point):
    map_size = scene.trav_map.map_size
    resolution = float(scene.trav_map.map_resolution)
    origin = -0.5 * map_size * resolution
    col = math.floor((float(point[0]) - origin) / resolution)
    row = math.floor((float(point[1]) - origin) / resolution)
    return int(row), int(col)


def point_is_free(scene, trav_map, point):
    row, col = point_to_map_cell(scene, point)
    if row < 0 or col < 0 or row >= trav_map.shape[0] or col >= trav_map.shape[1]:
        return False
    return int(trav_map[row, col]) == 255


def cell_is_free(trav_map, cell):
    row, col = cell
    return 0 <= row < trav_map.shape[0] and 0 <= col < trav_map.shape[1] and int(trav_map[row, col]) == 255


def clearance_astar(trav_map, start_cell, goal_cell):
    if not cell_is_free(trav_map, start_cell) or not cell_is_free(trav_map, goal_cell):
        return None

    neighbors = [
        (0, 1, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (-1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
    ]
    frontier = [(0.0, start_cell)]
    came_from = {}
    g_score = {start_cell: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return th.tensor(list(reversed(path)))

        current_g = g_score[current]
        for dr, dc, move_cost in neighbors:
            neighbor = (current[0] + dr, current[1] + dc)
            if not cell_is_free(trav_map, neighbor):
                continue
            if dr != 0 and dc != 0 and (
                not cell_is_free(trav_map, (current[0] + dr, current[1]))
                or not cell_is_free(trav_map, (current[0], current[1] + dc))
            ):
                continue
            tentative_g = current_g + move_cost
            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            heuristic = math.hypot(goal_cell[0] - neighbor[0], goal_cell[1] - neighbor[1])
            heapq.heappush(frontier, (tentative_g + heuristic, neighbor))

    return None


def episode_points_are_valid(env, floor_trav_map, start, goal, start_quat, settle_steps):
    robot = env.robots[0]
    if not point_is_free(env.scene, floor_trav_map, start):
        return False
    if not point_is_free(env.scene, floor_trav_map, goal):
        return False

    place_robot(robot, start, start_quat)
    for _ in range(settle_steps):
        env.step({robot.name: controller_no_op_action(robot)})

    settled_position, _ = robot.get_position_orientation()
    return point_is_free(env.scene, floor_trav_map, settled_position[:2])


def path_initial_yaw(path_world, start, goal):
    start_xy = th.as_tensor(start[:2], dtype=path_world.dtype, device=path_world.device)
    for waypoint in path_world[1:]:
        delta = waypoint - start_xy
        if float(th.norm(delta).item()) > 1e-4:
            return math.atan2(float(delta[1].item()), float(delta[0].item()))

    goal_xy = th.as_tensor(goal[:2], dtype=path_world.dtype, device=path_world.device)
    delta = goal_xy - start_xy
    return math.atan2(float(delta[1].item()), float(delta[0].item()))


def clearance_path_info(env, trav_map, start, goal):
    if not point_is_free(env.scene, trav_map, start) or not point_is_free(env.scene, trav_map, goal):
        return None
    start_cell = tuple(env.scene.trav_map.world_to_map(start[:2]).tolist())
    goal_cell = tuple(env.scene.trav_map.world_to_map(goal[:2]).tolist())
    path = clearance_astar(trav_map, start_cell, goal_cell)
    if path is None:
        return None
    path_world = env.scene.trav_map.map_to_world(path)
    return {
        "distance": float(th.sum(th.norm(path_world[1:] - path_world[:-1], dim=1)).item()),
        "initial_yaw": path_initial_yaw(path_world, start, goal),
    }


def clearance_path_distance(env, trav_map, start, goal):
    path_info = clearance_path_info(env, trav_map, start, goal)
    return None if path_info is None else path_info["distance"]


def sample_episode(
    env,
    scene_model,
    episode_idx,
    floor,
    floor_trav_map,
    clearance_trav_map,
    min_distance,
    max_distance,
    max_trials,
    settle_steps,
    extra_clearance,
    safety_clearance,
    validation_clearance,
):
    robot = env.robots[0]

    for trial in range(1, max_trials + 1):
        env.reset(get_obs=False)
        _, start = env.scene.get_random_point(floor=floor, robot=robot)
        _, goal = env.scene.get_random_point(floor=floor, reference_point=start, robot=robot)
        rooms = env.scene.load_room_instances
        if rooms is not None and any(
            env.scene.seg_map.get_room_instance_by_point(point[:2]) not in rooms for point in (start, goal)
        ):
            continue
        path_info = clearance_path_info(env, clearance_trav_map, start, goal)
        if path_info is None:
            continue

        distance = path_info["distance"]
        if distance < min_distance or distance > max_distance:
            continue

        start_yaw = path_info["initial_yaw"]
        start_quat = T.euler2quat(th.tensor([0.0, 0.0, start_yaw]))
        if not episode_points_are_valid(env, floor_trav_map, start, goal, start_quat, settle_steps):
            continue
        settled_position, _ = robot.get_position_orientation()
        if not point_is_free(env.scene, clearance_trav_map, settled_position[:2]):
            continue

        return {
            "episode_id": f"{scene_model}_{episode_idx:03d}",
            "scene_model": scene_model,
            "floor": int(floor),
            "start_position": to_float_list(start),
            "start_yaw": start_yaw,
            "start_quat": to_float_list(start_quat),
            "goal_position": to_float_list(goal),
            "geodesic_distance": distance,
            "sampling_trial": trial,
            "extra_clearance": extra_clearance,
            "safety_clearance": safety_clearance,
            "validation_clearance": validation_clearance,
            "start_yaw_source": "clearance_path_initial_heading",
        }

    raise RuntimeError(
        f"Failed to sample {scene_model} episode {episode_idx} after {max_trials} trials "
        f"with distance range [{min_distance}, {max_distance}]."
    )


def verify_episode(env, clearance_trav_map, episode):
    start = th.tensor(episode["start_position"], dtype=th.float32)
    goal = th.tensor(episode["goal_position"], dtype=th.float32)
    distance = clearance_path_distance(env, clearance_trav_map, start, goal)
    if distance is None:
        raise RuntimeError(f"Stored episode lacks a clearance-valid path on replay check: {episode['episode_id']}")


def sample_scene(scene_model, task_name, scene_instance, load_room_instances, robot_cfg, args, num_episodes=None):
    cfg = build_env_config(
        scene_model=scene_model,
        robot_cfg=copy.deepcopy(robot_cfg),
        scene_instance=scene_instance,
        load_room_instances=load_room_instances,
    )

    print(f"Loaded scene: {scene_model}")
    print(f"Task template: {scene_instance}")
    print(f"Robot: {robot_cfg['model']}")

    # Ensure OmniGibson appdata cache directory exists and is writable to avoid texture cache write errors
    appdata_cache = Path(gm.APPDATA_PATH) / "global" / "cache" / "texturecache"
    appdata_cache.mkdir(parents=True, exist_ok=True)

    env = og.Environment(configs=cfg)
    if env.robots[0].model in ("r1", "r1pro"):
        og.sim.stop()
        env.robots[0].base_footprint_link.mass = 250.0
        og.sim.play()
    episodes = []
    count = num_episodes if num_episodes is not None else args.num_episodes
    floor_trav_map = eroded_floor_map(env.scene, 0, env.robots[0])
    validation_clearance = args.extra_clearance + args.safety_clearance
    clearance_trav_map = clearance_floor_map(env, floor_trav_map, validation_clearance)
    print(
        f"Validation clearance: robot erosion + {args.extra_clearance:.3f}m extra "
        f"+ {args.safety_clearance:.3f}m safety margin"
    )
    for local_idx in range(count):
        episode = sample_episode(
            env=env,
            scene_model=scene_model,
            episode_idx=local_idx,
            floor=0,
            floor_trav_map=floor_trav_map,
            clearance_trav_map=clearance_trav_map,
            min_distance=args.min_distance,
            max_distance=args.max_distance,
            max_trials=args.max_trials,
            settle_steps=args.settle_steps,
            extra_clearance=args.extra_clearance,
            safety_clearance=args.safety_clearance,
            validation_clearance=validation_clearance,
        )
        verify_episode(env, clearance_trav_map, episode)
        episode["episode_id"] = f"{scene_model}_{task_name}_{local_idx:03d}"
        episode["task_name"] = task_name
        episode["scene_instance"] = scene_instance
        episode["load_room_instances"] = load_room_instances
        episodes.append(episode)
        print(f"\nSampled episode {episode['episode_id']}:")
        print(f"  start = {episode['start_position']}")
        print(f"  goal = {episode['goal_position']}")
        print(f"  shortest path = {episode['geodesic_distance']:.3f} m")

    # Clear simulation state before returning so next scene can be loaded cleanly
    og.clear()
    return episodes


def write_benchmark(path, args, robot_cfg, episodes):
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "robot": robot_cfg["model"],
                "floor": 0,
                "min_distance": args.min_distance,
                "max_distance": args.max_distance,
                "settle_steps": args.settle_steps,
                "extra_clearance": args.extra_clearance,
                "safety_clearance": args.safety_clearance,
                "episodes": episodes,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"\nSaved {len(episodes)} episodes to:")
    print(f"  {output}")


def main():
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be at least 1")
    if args.num_episodes_per_scene is not None and args.num_episodes_per_scene < 1:
        raise ValueError("--num-episodes-per-scene must be at least 1")
    if args.min_distance > args.max_distance:
        raise ValueError("--min-distance must be <= --max-distance")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative")
    if args.extra_clearance < 0:
        raise ValueError("--extra-clearance must be non-negative")
    if args.safety_clearance < 0:
        raise ValueError("--safety-clearance must be non-negative")

    seed_everything(args.seed)

    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False

    robot_cfg = load_robot_config(args.robot_config)
    task_metadata = Path(gm.DATA_PATH) / "2026-challenge-task-instances" / "metadata" / "available_tasks.yaml"
    with open(task_metadata, "r", encoding="utf-8") as f:
        available_tasks = yaml.safe_load(f)
    unknown_tasks = sorted(set(args.task) - set(available_tasks))
    if unknown_tasks:
        raise ValueError(f"Unknown competition task(s): {', '.join(unknown_tasks)}")
    try:
        # determine which scenes to generate
        if args.all_scenes:
            scenes = list(CHALLENGE_SCENES)
        else:
            scenes = [args.scene]

        per_scene = args.num_episodes_per_scene if args.num_episodes_per_scene is not None else args.num_episodes

        # Write one file per task template (atomic write).
        out_path = Path(args.output)
        out_parent = out_path.parent
        out_parent.mkdir(parents=True, exist_ok=True)

        for scene in scenes:
            scene_task_names = [
                name
                for name, configs in available_tasks.items()
                if configs[0]["scene_model"] == scene
            ]
            task_names = [name for name in scene_task_names if not args.task or name in args.task]
            if not task_names:
                selected = f" selected task(s) {', '.join(args.task)}" if args.task else ""
                raise ValueError(f"No competition tasks found for scene {scene}{selected}")
            if args.task:
                print(
                    f"Scene {scene}: {len(scene_task_names)} task(s) available; "
                    f"generating {len(task_names)} selected task(s)."
                )
            else:
                print(f"Scene {scene}: generating benchmarks for {len(task_names)} task(s).")
            for task_name in task_names:
                task_path = out_parent / f"{out_path.stem}_{scene}_{task_name}.json"
                if args.skip_existing and task_path.is_file():
                    print(f"Skipping existing task benchmark: {task_path}")
                    continue
                scene_instance = BehaviorTask.get_cached_activity_scene_filename(scene, task_name, 0, 0)
                eps = sample_scene(
                    scene_model=scene,
                    task_name=task_name,
                    scene_instance=scene_instance,
                    load_room_instances=TASK_NAMES_TO_ROOMS[task_name],
                    robot_cfg=robot_cfg,
                    args=args,
                    num_episodes=per_scene,
                )
                tmp_path = task_path.with_suffix(".json.tmp")
                write_benchmark(path=tmp_path, args=args, robot_cfg=robot_cfg, episodes=eps)
                tmp_path.replace(task_path)
                print(f"Wrote task benchmark: {task_path}")
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
