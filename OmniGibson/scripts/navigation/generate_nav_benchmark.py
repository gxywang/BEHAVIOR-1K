import argparse
import copy
import json
import math
import os
import random
from pathlib import Path

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


def sample_episode(
    env,
    scene_model,
    episode_idx,
    floor,
    floor_trav_map,
    min_distance,
    max_distance,
    max_trials,
    settle_steps,
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
        _, distance = env.scene.get_shortest_path(floor, start[:2], goal[:2], entire_path=False, robot=robot)

        if distance is None:
            continue

        distance = float(distance.item() if hasattr(distance, "item") else distance)
        if distance < min_distance or distance > max_distance:
            continue

        start_yaw = float(th.rand(1).item() * 2.0 * math.pi)
        start_quat = T.euler2quat(th.tensor([0.0, 0.0, start_yaw]))
        if not episode_points_are_valid(env, floor_trav_map, start, goal, start_quat, settle_steps):
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
        }

    raise RuntimeError(
        f"Failed to sample {scene_model} episode {episode_idx} after {max_trials} trials "
        f"with distance range [{min_distance}, {max_distance}]."
    )


def verify_episode(env, episode):
    robot = env.robots[0]
    start = th.tensor(episode["start_position"], dtype=th.float32)
    goal = th.tensor(episode["goal_position"], dtype=th.float32)
    _, distance = env.scene.get_shortest_path(episode["floor"], start[:2], goal[:2], entire_path=False, robot=robot)
    if distance is None:
        raise RuntimeError(f"Stored episode is unreachable on replay check: {episode['episode_id']}")


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
    for local_idx in range(count):
        episode = sample_episode(
            env=env,
            scene_model=scene_model,
            episode_idx=local_idx,
            floor=0,
            floor_trav_map=floor_trav_map,
            min_distance=args.min_distance,
            max_distance=args.max_distance,
            max_trials=args.max_trials,
            settle_steps=args.settle_steps,
        )
        verify_episode(env, episode)
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

    seed_everything(args.seed)

    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False

    robot_cfg = load_robot_config(args.robot_config)
    task_metadata = Path(gm.DATA_PATH) / "2026-challenge-task-instances" / "metadata" / "available_tasks.yaml"
    with open(task_metadata, "r", encoding="utf-8") as f:
        available_tasks = yaml.safe_load(f)
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
            task_names = [name for name, configs in available_tasks.items() if configs[0]["scene_model"] == scene]
            if not task_names:
                raise ValueError(f"No competition tasks found for scene {scene}")
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
