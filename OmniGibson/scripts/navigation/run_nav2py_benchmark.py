import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from generate_nav_benchmark import (
    build_env_config,
    load_robot_config,
    seed_everything,
    to_float_list,
)
from omnigibson.controllers import ControllerView
from omnigibson.macros import gm


DEFAULT_BENCHMARK = "outputs/navigation/nav_benchmark_test.json"
DEFAULT_OUTPUT = "outputs/navigation/nav2py_results.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run nav2py on saved BEHAVIOR navigation benchmark episodes.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Path to benchmark JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write result JSON.")
    parser.add_argument("--nav2py-root", default=None, help="Path to a local nav2py checkout if it is not installed.")
    parser.add_argument(
        "--robot-config",
        default=str(Path(__file__).resolve().parents[2] / "omnigibson" / "eval" / "r1pro.yaml"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--success-distance", type=float, default=0.5)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--costmap-source",
        choices=("nav2py-inflated", "og-eroded"),
        default="nav2py-inflated",
        help=(
            "nav2py-inflated uses the raw OmniGibson traversability map and lets nav2py inflate it. "
            "og-eroded uses OmniGibson's robot-eroded traversability map and treats that map as already inflated."
        ),
    )
    return parser.parse_args()


def add_nav2py_to_path(nav2py_root):
    if nav2py_root is not None:
        sys.path.insert(0, str(Path(nav2py_root).expanduser().resolve()))
        return

    sibling_checkout = Path(__file__).resolve().parents[4] / "nav2py"
    if sibling_checkout.exists():
        sys.path.insert(0, str(sibling_checkout))


def load_nav2py():
    try:
        from nav2py import (
            CircleFootprint,
            GoalSemantics,
            KinematicType,
            NavigationConfig,
            NavigationTask,
            Navigator,
            Pose2D,
            RecoveryManeuver,
            RobotProfile,
            StateEstimate,
        )
        from nav2py.maps import Costmap2D
    except ImportError as exc:
        raise RuntimeError(
            "Could not import nav2py. Install it in the active environment or pass --nav2py-root."
        ) from exc

    return {
        "CircleFootprint": CircleFootprint,
        "Costmap2D": Costmap2D,
        "GoalSemantics": GoalSemantics,
        "KinematicType": KinematicType,
        "NavigationConfig": NavigationConfig,
        "NavigationTask": NavigationTask,
        "Navigator": Navigator,
        "Pose2D": Pose2D,
        "RecoveryManeuver": RecoveryManeuver,
        "RobotProfile": RobotProfile,
        "StateEstimate": StateEstimate,
    }


def load_benchmark(path):
    p = Path(path).expanduser()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes") if isinstance(data, dict) else data
    if not episodes:
        raise RuntimeError(f"No benchmark episodes found in {p}")
    return data, episodes


def group_episodes_by_scene(episodes):
    groups = {}
    for episode in episodes:
        groups.setdefault(episode["scene_model"], []).append(episode)
    return groups


def make_costmap(scene, floor, robot, nav2py_api, erode_for_robot=False):
    trav_map = th.clone(scene.trav_map.floor_map[floor])
    if erode_for_robot:
        trav_map = scene.trav_map._erode_trav_map(trav_map, robot=robot)
    trav_map = trav_map.detach().cpu().numpy()
    occupancy = np.full(trav_map.shape, 100, dtype=np.int16)
    occupancy[trav_map == 255] = 0

    map_size = scene.trav_map.map_size
    resolution = float(scene.trav_map.map_resolution)
    origin = (-0.5 * map_size * resolution, -0.5 * map_size * resolution)
    return nav2py_api["Costmap2D"].from_occupancy(
        occupancy,
        resolution=resolution,
        origin=origin,
        occupied_threshold=65,
        frame_id="map",
    )


def make_costmap_bundle(scene, floor, robot, nav2py_api):
    return {
        "raw": make_costmap(scene, floor, robot, nav2py_api, erode_for_robot=False),
        "og_eroded": make_costmap(scene, floor, robot, nav2py_api, erode_for_robot=True),
    }


def select_costmap(costmap_bundle, costmap_source):
    if costmap_source == "nav2py-inflated":
        return costmap_bundle["raw"], False
    return costmap_bundle["og_eroded"], True


def make_robot_profile(robot, nav2py_api, clearance_is_in_costmap=False):
    robot_radius = float(th.norm(robot.reset_joint_pos_aabb_extent[:2]).item() / 2.0)
    radius = 0.01 if clearance_is_in_costmap else robot_radius
    footprint_padding = 0.0 if clearance_is_in_costmap else 0.2
    inflation_radius = 0.0 if clearance_is_in_costmap else robot_radius + 0.2
    recovery_maneuvers = nav2py_api["RecoveryManeuver"]
    return nav2py_api["RobotProfile"](
        name=robot.model,
        kinematic_type=nav2py_api["KinematicType"].HOLONOMIC,
        footprint=nav2py_api["CircleFootprint"](radius),
        max_forward_velocity=0.75,
        max_reverse_velocity=0.75,
        max_lateral_velocity=0.75,
        max_angular_velocity=1.0,
        max_linear_acceleration=1.0,
        max_linear_deceleration=1.0,
        max_linear_jerk=20.0,
        max_braking_deceleration=1.5,
        max_angular_acceleration=2.0,
        max_angular_deceleration=2.0,
        max_angular_jerk=40.0,
        can_rotate_in_place=True,
        control_period=1.0 / 30.0,
        command_latency=0.0,
        footprint_padding=footprint_padding,
        inflation_radius=inflation_radius,
        allowed_recovery_maneuvers=frozenset(
            {
                recovery_maneuvers.STOP,
                recovery_maneuvers.ROTATE,
                recovery_maneuvers.REVERSE,
                recovery_maneuvers.LATERAL_ESCAPE,
            }
        ),
    )


def robot_profile_diagnostics(profile):
    return {
        "footprint_radius": float(profile.footprint.bounding_radius),
        "footprint_padding": float(profile.footprint_padding),
        "inflation_radius": float(profile.inflation_radius),
        "padded_footprint_radius": float(profile.padded_footprint.bounding_radius),
    }


def make_navigator(profile, costmap, costmap_is_profile_inflated, nav2py_api):
    navigator = nav2py_api["Navigator"](profile, costmap, nav2py_api["NavigationConfig"]())
    if costmap_is_profile_inflated:
        navigator.update_costmap(costmap, profile_inflated=True, base_costmap=costmap)
    return navigator


def point_cost_diagnostic(costmap, point):
    cell = costmap.world_to_map(float(point[0]), float(point[1]))
    if cell is None:
        return {
            "cell": None,
            "cost": None,
            "start_rejected": True,
            "goal_rejected": True,
            "free_neighbor_cells_r1": 0,
        }

    row, col = cell
    cost = int(costmap.data[row, col])
    free_neighbors = 0
    for r in range(row - 1, row + 2):
        for c in range(col - 1, col + 2):
            if 0 <= r < costmap.data.shape[0] and 0 <= c < costmap.data.shape[1]:
                free_neighbors += int(not costmap.is_lethal(r, c, lethal_cost=253))

    return {
        "cell": [int(row), int(col)],
        "cost": cost,
        "start_rejected": cost >= 254,
        "goal_rejected": cost >= 253,
        "free_neighbor_cells_r1": free_neighbors,
    }


def command_diagnostic(command):
    if command is None:
        return None
    return {
        "timestamp": float(command.timestamp),
        "valid_until": float(command.valid_until),
        "is_stop": bool(command.is_stop),
        "velocity": {
            "vx": float(command.velocity.vx),
            "vy": float(command.velocity.vy),
            "wz": float(command.velocity.wz),
        },
    }


def safety_decision_diagnostic(decision):
    if decision is None:
        return None
    return {
        "action": decision.action.value,
        "reason": decision.reason,
        "command": command_diagnostic(decision.command),
    }


def state_estimate_diagnostic(state, costmap):
    return {
        "timestamp": float(state.timestamp),
        "pose": {
            "x": float(state.pose.x),
            "y": float(state.pose.y),
            "yaw": float(state.pose.yaw),
        },
        "linear_velocity_body": [float(state.linear_velocity[0]), float(state.linear_velocity[1])],
        "angular_velocity_body": float(state.angular_velocity),
        "linear_speed_body": math.hypot(float(state.linear_velocity[0]), float(state.linear_velocity[1])),
        "map": point_cost_diagnostic(costmap, [state.pose.x, state.pose.y]),
    }


def robot_state_diagnostic(robot, timestamp, costmap):
    position, orientation = robot.get_position_orientation()
    yaw = float(T.quat2euler(orientation)[2].item())
    linear_velocity_world = robot.get_linear_velocity()
    angular_velocity_world = robot.get_angular_velocity()
    rotation_world_to_body = T.quat2mat(orientation).T
    linear_velocity_body = rotation_world_to_body @ linear_velocity_world
    angular_velocity_body = rotation_world_to_body @ angular_velocity_world

    return {
        "timestamp": float(timestamp),
        "position": to_float_list(position),
        "yaw": yaw,
        "linear_velocity_world": to_float_list(linear_velocity_world),
        "angular_velocity_world": to_float_list(angular_velocity_world),
        "linear_velocity_body": to_float_list(linear_velocity_body),
        "angular_velocity_body": to_float_list(angular_velocity_body),
        "linear_speed_body": math.hypot(float(linear_velocity_body[0]), float(linear_velocity_body[1])),
        "map": point_cost_diagnostic(costmap, position[:2]),
    }


def episode_costmap_diagnostics(costmap_bundle, active_costmap, episode):
    diagnostics = {}
    for name, costmap in {
        "raw": costmap_bundle["raw"],
        "og_eroded": costmap_bundle["og_eroded"],
        "active": active_costmap,
    }.items():
        diagnostics[name] = {
            "start": point_cost_diagnostic(costmap, episode["start_position"]),
            "goal": point_cost_diagnostic(costmap, episode["goal_position"]),
        }
    return diagnostics


def controller_no_op_action(robot):
    action = []
    for group_key, controller_idx in robot.controllers.values():
        action.append(ControllerView.compute_no_op_action(group_key, controller_idx).float())
    return th.cat(action) if action else th.empty(0, dtype=th.float32)


def action_from_nav2py_command(robot, command):
    action = controller_no_op_action(robot)
    if command is None:
        return action

    base_command = th.tensor(
        [command.velocity.vx, command.velocity.vy, command.velocity.wz],
        dtype=th.float32,
    )
    base_group_key, _ = robot.controllers["base"]
    action[robot.base_action_idx] = ControllerView.reverse_preprocess_command(base_group_key, base_command)
    return action


def place_robot(robot, episode):
    position = th.tensor(episode["start_position"], dtype=th.float32)
    orientation = th.tensor(episode["start_quat"], dtype=th.float32)
    robot.set_joint_positions(robot.reset_joint_pos, drive=False)
    robot.set_position_orientation(position=position, orientation=orientation)
    robot.set_linear_velocity(th.zeros(3))
    robot.set_angular_velocity(th.zeros(3))
    robot.set_joint_velocities(th.zeros(robot.n_dof), drive=False)


def robot_state_estimate(robot, timestamp, nav2py_api):
    position, orientation = robot.get_position_orientation()
    yaw = float(T.quat2euler(orientation)[2].item())
    rotation_world_to_body = T.quat2mat(orientation).T
    linear_velocity = rotation_world_to_body @ robot.get_linear_velocity()
    angular_velocity = rotation_world_to_body @ robot.get_angular_velocity()

    return nav2py_api["StateEstimate"](
        timestamp=timestamp,
        frame_id="map",
        pose=nav2py_api["Pose2D"](float(position[0].item()), float(position[1].item()), yaw),
        linear_velocity=(float(linear_velocity[0].item()), float(linear_velocity[1].item())),
        angular_velocity=float(angular_velocity[2].item()),
        velocity_available=True,
    )


def xy_distance(position, goal):
    return math.hypot(float(position[0]) - float(goal[0]), float(position[1]) - float(goal[1]))


def run_episode(env, robot, episode, costmap_bundle, profile, nav2py_api, args):
    place_robot(robot, episode)

    for _ in range(args.settle_steps):
        env.step({robot.name: controller_no_op_action(robot)})

    costmap, costmap_is_profile_inflated = select_costmap(costmap_bundle, args.costmap_source)
    navigator = make_navigator(profile, costmap, costmap_is_profile_inflated, nav2py_api)
    costmap_diagnostics = episode_costmap_diagnostics(costmap_bundle, navigator.costmap, episode)
    goal = episode["goal_position"]
    navigator.submit(
        nav2py_api["NavigationTask"](
            episode["episode_id"],
            goal_pose=nav2py_api["Pose2D"](float(goal[0]), float(goal[1]), 0.0),
            goal_semantics=nav2py_api["GoalSemantics"].POSITION_ONLY,
        )
    )

    dt = profile.control_period
    commanded_steps = 0
    success = False
    last_state = None
    last_command = None
    last_safety_decision = None
    for step in range(args.max_steps):
        now = step * dt
        state = robot_state_estimate(robot, now, nav2py_api)
        command = navigator.tick(state, now)
        last_state = state
        last_command = command
        last_safety_decision = navigator.last_safety_decision
        if command is not None and not command.is_stop:
            commanded_steps += 1

        env.step({robot.name: action_from_nav2py_command(robot, command)})
        position, _ = robot.get_position_orientation()
        final_distance = xy_distance(position[:2], goal[:2])
        success = final_distance <= args.success_distance
        nav_status = navigator.status()
        if success or nav_status.state.value in {"succeeded", "failed", "blocked", "canceled"}:
            break

    status = navigator.status()
    position, orientation = robot.get_position_orientation()
    final_yaw = float(T.quat2euler(orientation)[2].item())
    return {
        "episode_id": episode["episode_id"],
        "scene_model": episode["scene_model"],
        "floor": int(episode.get("floor", 0)),
        "success": success,
        "costmap_source": args.costmap_source,
        "robot_profile": robot_profile_diagnostics(profile),
        "costmap_diagnostics": costmap_diagnostics,
        "nav2py_state": status.state.value,
        "nav2py_reason": status.reason,
        "steps": step + 1,
        "commanded_steps": commanded_steps,
        "sim_time": (step + 1) * dt,
        "start_position": episode["start_position"],
        "goal_position": episode["goal_position"],
        "final_position": to_float_list(position),
        "final_yaw": final_yaw,
        "final_distance": xy_distance(position[:2], goal[:2]),
        "geodesic_distance": float(episode["geodesic_distance"]),
        "remaining_distance": status.remaining_distance,
        "progress": status.progress,
        "last_tick_state": state_estimate_diagnostic(last_state, navigator.costmap),
        "final_robot_state": robot_state_diagnostic(robot, (step + 1) * dt, navigator.costmap),
        "last_command": command_diagnostic(last_command),
        "last_safety_decision": safety_decision_diagnostic(last_safety_decision),
    }


def summarize_results(results):
    successes = sum(1 for result in results if result["success"])
    final_distances = [result["final_distance"] for result in results]
    return {
        "total": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "mean_final_distance": float(np.mean(final_distances)) if final_distances else None,
        "max_final_distance": float(np.max(final_distances)) if final_distances else None,
    }


def write_results(path, benchmark_path, nav2py_root, args, results):
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": str(Path(benchmark_path).expanduser()),
        "nav2py_root": None if nav2py_root is None else str(Path(nav2py_root).expanduser()),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "success_distance": args.success_distance,
        "costmap_source": args.costmap_source,
        "summary": summarize_results(results),
        "episodes": results,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return output


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.success_distance <= 0.0:
        raise ValueError("--success-distance must be positive")

    seed_everything(args.seed)
    add_nav2py_to_path(args.nav2py_root)
    nav2py_api = load_nav2py()
    _, episodes = load_benchmark(args.benchmark)

    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False

    robot_cfg = load_robot_config(args.robot_config)
    results = []
    try:
        for scene_model, scene_episodes in group_episodes_by_scene(episodes).items():
            print(f"\nRunning scene: {scene_model} ({len(scene_episodes)} episodes)")
            cfg = build_env_config(scene_model=scene_model, robot_cfg=robot_cfg)
            env = og.Environment(configs=cfg)
            robot = env.robots[0]
            profile = make_robot_profile(
                robot,
                nav2py_api,
                clearance_is_in_costmap=args.costmap_source == "og-eroded",
            )

            costmap_bundles = {}
            for episode in scene_episodes:
                floor = int(episode.get("floor", 0))
                costmap_bundles.setdefault(floor, make_costmap_bundle(env.scene, floor, robot, nav2py_api))
                result = run_episode(env, robot, episode, costmap_bundles[floor], profile, nav2py_api, args)
                results.append(result)
                active = result["costmap_diagnostics"]["active"]
                print(
                    f"  {result['episode_id']}: "
                    f"{'SUCCESS' if result['success'] else 'FAIL'} "
                    f"final_distance={result['final_distance']:.3f}m "
                    f"state={result['nav2py_state']}"
                )
                if not result["success"]:
                    print(
                        f"    active_costs: start={active['start']['cost']} "
                        f"goal={active['goal']['cost']} "
                        f"reason={result['nav2py_reason']}"
                    )

            og.clear()

        output = write_results(args.output, args.benchmark, args.nav2py_root, args, results)
        summary = summarize_results(results)
        print(f"\nSaved results to: {output}")
        print(f"Success rate: {summary['successes']}/{summary['total']} ({summary['success_rate']:.1%})")
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
