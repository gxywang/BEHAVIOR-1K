import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch as th

import omnigibson as og
from omnigibson.macros import gm


DEFAULT_INPUT = "outputs/navigation/nav_benchmark_test.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Check saved nav benchmark episodes for validity.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to benchmark JSON file.")
    parser.add_argument("--robot-config", default=None, help="Optional robot yaml used when generating episodes.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--distance-tol", type=float, default=0.25, help="Absolute tolerance (m) for geodesic distance match.")
    parser.add_argument("--max-trials", type=int, default=500)
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


def check_episodes(data, args):
    episodes = data.get("episodes")
    if not episodes:
        print("No episodes to check.")
        return 0

    # group episodes by scene for efficient env reuse
    groups = {}
    for ep in episodes:
        groups.setdefault(ep["scene_model"], []).append(ep)

    failures = []
    for scene_model, eps in groups.items():
        print(f"\nChecking scene: {scene_model} ({len(eps)} episodes)")
        cfg = build_env_config(scene_model=scene_model, robot_cfg=data.get("robot_cfg") if data.get("robot_cfg") else None)
        env = og.Environment(configs=cfg)

        for ep in eps:
            floor = int(ep.get("floor", 0))
            start = ep["start_position"]
            goal = ep["goal_position"]
            stored_dist = float(ep.get("geodesic_distance", -1.0))

            _, distance = env.scene.get_shortest_path(floor, start[:2], goal[:2], entire_path=False, robot=env.robots[0] if env.robots else None)
            if distance is None:
                failures.append((ep["episode_id"], "unreachable"))
                print(f"  [FAIL] {ep['episode_id']}: unreachable")
                continue
            distance = float(distance.item() if hasattr(distance, "item") else distance)
            diff = abs(distance - stored_dist)
            if diff > args.distance_tol:
                failures.append((ep["episode_id"], f"distance_mismatch stored={stored_dist:.3f} now={distance:.3f} diff={diff:.3f}"))
                print(f"  [FAIL] {ep['episode_id']}: stored {stored_dist:.3f} m vs now {distance:.3f} m (diff {diff:.3f} m)")
            else:
                print(f"  [OK]   {ep['episode_id']}: {distance:.3f} m (matches stored)")

        og.shutdown()

    print(f"\nChecked {len(episodes)} episodes: {len(failures)} failures")
    if failures:
        for f in failures:
            print(f" - {f[0]}: {f[1]}")
        return 1
    return 0


def main():
    args = parse_args()
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
