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
from omnigibson.macros import gm


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
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--floor", type=int, default=0)
    parser.add_argument("--min-distance", type=float, default=1.0)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--max-trials", type=int, default=500)
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


def build_env_config(scene_model, robot_cfg):
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
            "trav_map_resolution": 0.1,
            "default_erosion_radius": 0.0,
            "trav_map_with_objects": True,
            "num_waypoints": 1,
            "waypoint_resolution": 0.2,
            "load_room_types": None,
            "load_room_instances": None,
            "include_robots": False,
        },
        "robots": [robot_cfg],
        "objects": [],
        "task": {
            "type": "DummyTask",
            "include_obs": False,
        },
    }


def sample_episode(env, scene_model, episode_idx, floor, min_distance, max_distance, max_trials):
    robot = env.robots[0]

    for trial in range(1, max_trials + 1):
        _, start = env.scene.get_random_point(floor=floor, robot=robot)
        _, goal = env.scene.get_random_point(floor=floor, reference_point=start, robot=robot)
        _, distance = env.scene.get_shortest_path(floor, start[:2], goal[:2], entire_path=False, robot=robot)

        if distance is None:
            continue

        distance = float(distance.item() if hasattr(distance, "item") else distance)
        if distance < min_distance or distance > max_distance:
            continue

        start_yaw = float(th.rand(1).item() * 2.0 * math.pi)
        start_quat = T.euler2quat(th.tensor([0.0, 0.0, start_yaw]))
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


def sample_scene(scene_model, robot_cfg, args):
    cfg = build_env_config(scene_model=scene_model, robot_cfg=copy.deepcopy(robot_cfg))

    print(f"Loaded scene: {scene_model}")
    print(f"Robot: {robot_cfg['model']}")

    env = og.Environment(configs=cfg)
    episodes = []
    for local_idx in range(args.num_episodes):
        episode = sample_episode(
            env=env,
            scene_model=scene_model,
            episode_idx=local_idx,
            floor=args.floor,
            min_distance=args.min_distance,
            max_distance=args.max_distance,
            max_trials=args.max_trials,
        )
        verify_episode(env, episode)
        episodes.append(episode)
        print(f"\nSampled episode {episode['episode_id']}:")
        print(f"  start = {episode['start_position']}")
        print(f"  goal = {episode['goal_position']}")
        print(f"  shortest path = {episode['geodesic_distance']:.3f} m")

    return episodes


def main():
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be at least 1")
    if args.min_distance > args.max_distance:
        raise ValueError("--min-distance must be <= --max-distance")

    seed_everything(args.seed)

    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False

    robot_cfg = load_robot_config(args.robot_config)
    try:
        episodes = sample_scene(scene_model=args.scene, robot_cfg=robot_cfg, args=args)
    finally:
        og.shutdown()

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "robot": robot_cfg["model"],
                "floor": args.floor,
                "min_distance": args.min_distance,
                "max_distance": args.max_distance,
                "episodes": episodes,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"\nSaved {len(episodes)} episodes to:")
    print(f"  {output}")


if __name__ == "__main__":
    main()
