import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch as th

import omnigibson as og
from omnigibson.controllers import ControllerView
from omnigibson.macros import gm


DEFAULT_INPUT = "outputs/navigation/nav_benchmark_test.json"
DEFAULT_ROBOT_CONFIG = str(
    Path(__file__).resolve().parents[2] / "omnigibson" / "eval" / "r1pro.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Check saved nav benchmark episodes for validity.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to benchmark JSON file.")
    parser.add_argument(
        "--robot-config",
        default=DEFAULT_ROBOT_CONFIG,
        help="Robot yaml used when generating episodes.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--distance-tol",
        type=float,
        default=0.25,
        help="Absolute tolerance (m) for geodesic distance match.",
    )
    parser.add_argument("--max-trials", type=int, default=500)
    parser.add_argument("--settle-steps", type=int, default=10)
    return parser.parse_args()


def seed_everything(seed):
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def load_benchmark(path):
    p = Path(path).expanduser()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes") if isinstance(data, dict) else data
    if episodes is None:
        raise RuntimeError(f"No 'episodes' found in {p}")
    return data


def load_robot_config(path):
    # If the generator saved a robot model name only, we don't need to parse YAML here.
    if path is None:
        return None
    import yaml

    p = Path(path).expanduser()
    with open(p, "r", encoding="utf-8") as f:
        robot_cfg = yaml.safe_load(f)

    robot_cfg = dict(robot_cfg)
    robot_cfg.pop("eval", None)
    robot_cfg["obs_modalities"] = []
    robot_cfg["position"] = [-50.0, -50.0, 0.0]
    return robot_cfg


def build_env_config(scene_model, robot_cfg):
    cfg = {
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
        "robots": [robot_cfg] if robot_cfg is not None else [],
        "objects": [],
        "task": {"type": "DummyTask", "include_obs": False},
    }
    return cfg


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
        return False, (row, col), None
    cost = int(trav_map[row, col])
    return cost == 255, (row, col), cost


def validate_episode_points(env, floor_trav_map, episode, settle_steps):
    robot = env.robots[0]
    start = th.tensor(episode["start_position"], dtype=th.float32)
    goal = th.tensor(episode["goal_position"], dtype=th.float32)
    start_quat = th.tensor(episode["start_quat"], dtype=th.float32)

    start_free, start_cell, start_cost = point_is_free(env.scene, floor_trav_map, start)
    if not start_free:
        return f"start occupied cell={start_cell} value={start_cost}"

    goal_free, goal_cell, goal_cost = point_is_free(env.scene, floor_trav_map, goal)
    if not goal_free:
        return f"goal occupied cell={goal_cell} value={goal_cost}"

    place_robot(robot, start, start_quat)
    for _ in range(settle_steps):
        env.step({robot.name: controller_no_op_action(robot)})

    settled_position, _ = robot.get_position_orientation()
    settled_free, settled_cell, settled_cost = point_is_free(env.scene, floor_trav_map, settled_position[:2])
    if not settled_free:
        return f"settled start occupied cell={settled_cell} value={settled_cost}"

    return None


def check_episodes(data, args):
    episodes = data.get("episodes")
    if not episodes:
        print("No episodes to check.")
        return 0

    # Group episodes by template and room selection for efficient env reuse.
    groups = {}
    for ep in episodes:
        key = (ep["scene_model"], ep.get("scene_instance"), tuple(ep.get("load_room_instances") or ()))
        groups.setdefault(key, []).append(ep)

    failures = []
    for (scene_model, scene_instance, _), eps in groups.items():
        print(f"\nChecking scene: {scene_model} ({len(eps)} episodes)")
        cfg = build_env_config(
            scene_model=scene_model,
            robot_cfg=data.get("robot_cfg") if data.get("robot_cfg") else None,
        )
        cfg["scene"]["scene_instance"] = scene_instance
        cfg["scene"]["load_room_instances"] = eps[0].get("load_room_instances")
        env = og.Environment(configs=cfg)
        floor_trav_maps = {}

        for ep in eps:
            floor = int(ep.get("floor", 0))
            start = ep["start_position"]
            goal = ep["goal_position"]
            stored_dist = float(ep.get("geodesic_distance", -1.0))
            episode_failures = []

            if env.robots:
                if floor not in floor_trav_maps:
                    floor_trav_maps[floor] = eroded_floor_map(env.scene, floor, env.robots[0])
                invalid_reason = validate_episode_points(env, floor_trav_maps[floor], ep, args.settle_steps)
                if invalid_reason is not None:
                    episode_failures.append(invalid_reason)

            _, distance = env.scene.get_shortest_path(
                floor,
                start[:2],
                goal[:2],
                entire_path=False,
                robot=env.robots[0] if env.robots else None,
            )
            if distance is None:
                episode_failures.append("unreachable")
            else:
                distance = float(distance.item() if hasattr(distance, "item") else distance)
                diff = abs(distance - stored_dist)
                if diff > args.distance_tol:
                    episode_failures.append(
                        f"distance_mismatch stored={stored_dist:.3f} now={distance:.3f} diff={diff:.3f}"
                    )

            if episode_failures:
                failures.extend((ep["episode_id"], failure) for failure in episode_failures)
                print(f"  [FAIL] {ep['episode_id']}: {'; '.join(episode_failures)}")
                continue

            print(f"  [OK]   {ep['episode_id']}: {distance:.3f} m (matches stored)")

        og.clear()

    print(f"\nChecked {len(episodes)} episodes: {len(failures)} failures")
    if failures:
        for f in failures:
            print(f" - {f[0]}: {f[1]}")
        return 1
    return 0


def main():
    args = parse_args()
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative")
    if args.seed is not None:
        seed_everything(args.seed)

    data = load_benchmark(args.input)

    # If the saved benchmark includes the robot name only, allow loading a YAML if provided
    if args.robot_config is not None:
        robot_cfg = load_robot_config(args.robot_config)
        data["robot_cfg"] = robot_cfg

    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False

    try:
        rc = check_episodes(data, args)
    finally:
        og.shutdown()

    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
