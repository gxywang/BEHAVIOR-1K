import argparse
import json
import math
import sys
import time
from dataclasses import replace
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
PREINFLATED_COSTMAP_FOOTPRINT_RADIUS = 1e-6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run nav2py on saved BEHAVIOR navigation benchmark episodes.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Path to benchmark JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write result JSON.")
    parser.add_argument("--nav2py-root", default=None, help="Path to a local nav2py checkout if it is not installed.")
    parser.add_argument(
        "--episode-ids",
        nargs="+",
        default=None,
        help="Run only these full benchmark episode IDs.",
    )
    parser.add_argument(
        "--robot-config",
        default=str(Path(__file__).resolve().parents[2] / "omnigibson" / "eval" / "r1pro.yaml"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--success-distance", type=float, default=0.1)
    parser.add_argument(
        "--success-criterion",
        choices=("center-distance", "footprint-overlap"),
        default="center-distance",
        help=(
            "center-distance succeeds when base-center XY distance is <= --success-distance. "
            "footprint-overlap succeeds when the goal point lies inside the robot's final XY footprint."
        ),
    )
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--costmap-source",
        choices=("nav2py-inflated", "og-eroded", "og-eroded-soft"),
        default="nav2py-inflated",
        help=(
            "nav2py-inflated uses the raw OmniGibson traversability map and lets nav2py inflate it. "
            "og-eroded uses OmniGibson's robot-eroded traversability map and treats that map as already inflated. "
            "og-eroded-soft adds non-lethal costs near OG-eroded obstacles to prefer higher-clearance paths."
        ),
    )
    parser.add_argument(
        "--disable-dynamic-safety",
        action="store_true",
        help=(
            "Bypass nav2py's dynamic collision safety veto while still logging the delegated safety decision. "
            "This is for diagnosis only."
        ),
    )
    parser.add_argument(
        "--trace-failures",
        action="store_true",
        help="Store compact per-step trace data for failed episodes.",
    )
    parser.add_argument(
        "--disable-path-smoothing",
        action="store_true",
        help="Disable A* line-of-sight path smoothing.",
    )
    parser.add_argument("--desired-linear-velocity", type=float, default=None)
    parser.add_argument("--min-lookahead-distance", type=float, default=None)
    parser.add_argument("--max-lookahead-distance", type=float, default=None)
    parser.add_argument("--lookahead-time", type=float, default=None)
    parser.add_argument("--collision-horizon", type=float, default=None)
    parser.add_argument("--profile-max-linear-velocity", type=float, default=None)
    parser.add_argument("--profile-max-angular-velocity", type=float, default=None)
    parser.add_argument(
        "--state-linear-velocity-deadband",
        type=float,
        default=0.01,
        help=(
            "Treat smaller measured base linear velocities as simulator settling noise before "
            "passing state to nav2py."
        ),
    )
    parser.add_argument(
        "--state-angular-velocity-deadband",
        type=float,
        default=0.01,
        help=(
            "Treat smaller measured base angular velocities as simulator settling noise before "
            "passing state to nav2py."
        ),
    )
    parser.add_argument(
        "--command-max-linear-velocity",
        type=float,
        default=None,
        help=(
            "Maximum absolute vx/vy command sent to OmniGibson. Defaults to the base controller "
            "command_output_limits from --robot-config."
        ),
    )
    parser.add_argument(
        "--command-max-angular-velocity",
        type=float,
        default=None,
        help=(
            "Maximum absolute wz command sent to OmniGibson. Defaults to the base controller "
            "command_output_limits from --robot-config."
        ),
    )
    parser.add_argument(
        "--safety-slowdown-scales",
        default="0.75,0.5,0.25,0.125,0.0625",
        help=(
            "Comma-separated uniform slowdown scales for nav2py dynamic collision safety. "
            "Use 'nav2py-default' to keep nav2py's built-in ladder."
        ),
    )
    parser.add_argument("--soft-cost-radius", type=float, default=0.75)
    parser.add_argument("--soft-cost-scaling-factor", type=float, default=3.0)
    parser.add_argument("--planner-cost-penalty", type=float, default=None)
    parser.add_argument(
        "--visual-step-sleep",
        type=float,
        default=0.0,
        help="Sleep this many seconds after each simulator step so non-headless runs are visible.",
    )
    parser.add_argument(
        "--keep-open-on-complete",
        action="store_true",
        help="Wait before shutdown after the run finishes, useful for inspecting the non-headless viewer.",
    )
    parser.add_argument(
        "--keep-open-seconds",
        type=float,
        default=None,
        help="Seconds to keep the viewer open. If omitted with --keep-open-on-complete, wait for Enter.",
    )
    parser.add_argument(
        "--viewer-camera-mode",
        choices=("none", "follow"),
        default="none",
        help="Viewer camera behavior for non-headless visualization.",
    )
    parser.add_argument("--viewer-camera-distance", type=float, default=3.0)
    parser.add_argument("--viewer-camera-height", type=float, default=2.0)
    parser.add_argument("--viewer-camera-target-height", type=float, default=0.7)
    return parser.parse_args(argv)


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
            SafetyAction,
            StateEstimate,
        )
        from nav2py.components import FootprintCollisionModel, SafetyDecision
        from nav2py.maps import Costmap2D
    except ImportError as exc:
        raise RuntimeError(
            "Could not import nav2py. Install it in the active environment or pass --nav2py-root."
        ) from exc

    return {
        "CircleFootprint": CircleFootprint,
        "Costmap2D": Costmap2D,
        "FootprintCollisionModel": FootprintCollisionModel,
        "GoalSemantics": GoalSemantics,
        "KinematicType": KinematicType,
        "NavigationConfig": NavigationConfig,
        "NavigationTask": NavigationTask,
        "Navigator": Navigator,
        "Pose2D": Pose2D,
        "RecoveryManeuver": RecoveryManeuver,
        "RobotProfile": RobotProfile,
        "SafetyAction": SafetyAction,
        "SafetyDecision": SafetyDecision,
        "StateEstimate": StateEstimate,
    }


def load_benchmark(path):
    p = Path(path).expanduser()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes") if isinstance(data, dict) else data
    if not episodes:
        raise RuntimeError(f"No benchmark episodes found in {p}")
    if any(
        not episode.get("task_name") or not episode.get("scene_instance") or "load_room_instances" not in episode
        for episode in episodes
    ):
        raise ValueError("Benchmark lacks task-template metadata. Regenerate it with generate_nav_benchmark.py.")
    return data, episodes


def filter_episodes(episodes, episode_ids):
    if episode_ids is None:
        return episodes

    selected = set(episode_ids)
    filtered = [episode for episode in episodes if episode["episode_id"] in selected]
    missing = sorted(selected - {episode["episode_id"] for episode in episodes})
    if missing:
        raise ValueError(f"Requested episode IDs not found in benchmark: {missing}")
    if not filtered:
        raise ValueError("No episodes remain after applying --episode-ids")
    return filtered


def group_episodes_by_scene(episodes):
    groups = {}
    for episode in episodes:
        key = (episode["scene_model"], episode["scene_instance"], tuple(episode["load_room_instances"] or ()))
        groups.setdefault(key, []).append(episode)
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


def make_soft_costmap(costmap, radius, cost_scaling_factor):
    soft_costmap = costmap.copy()
    hard_obstacles = costmap.data >= 254
    soft_costmap.inflate(radius, cost_scaling_factor, inscribed_radius=0.0)
    soft_costmap.data[hard_obstacles] = costmap.data[hard_obstacles]
    return soft_costmap


def make_costmap_bundle(scene, floor, robot, nav2py_api, args):
    bundle = {
        "raw": make_costmap(scene, floor, robot, nav2py_api, erode_for_robot=False),
        "og_eroded": make_costmap(scene, floor, robot, nav2py_api, erode_for_robot=True),
    }
    if args.costmap_source == "og-eroded-soft":
        bundle["og_eroded_soft"] = make_soft_costmap(
            bundle["og_eroded"],
            args.soft_cost_radius,
            args.soft_cost_scaling_factor,
        )
    return bundle


def select_costmap(costmap_bundle, costmap_source):
    if costmap_source == "nav2py-inflated":
        return costmap_bundle["raw"], False
    if costmap_source == "og-eroded-soft":
        return costmap_bundle["og_eroded_soft"], True
    return costmap_bundle["og_eroded"], True


def make_robot_profile(robot, nav2py_api, args, clearance_is_in_costmap=False):
    robot_radius = float(th.norm(robot.reset_joint_pos_aabb_extent[:2]).item() / 2.0)
    # OmniGibson-eroded maps already include robot clearance; nav2py only requires a positive radius.
    radius = PREINFLATED_COSTMAP_FOOTPRINT_RADIUS if clearance_is_in_costmap else robot_radius
    footprint_padding = 0.0 if clearance_is_in_costmap else 0.2
    inflation_radius = 0.0 if clearance_is_in_costmap else robot_radius + 0.2
    max_linear_velocity = 0.75 if args.profile_max_linear_velocity is None else args.profile_max_linear_velocity
    max_angular_velocity = 1.0 if args.profile_max_angular_velocity is None else args.profile_max_angular_velocity
    recovery_maneuvers = nav2py_api["RecoveryManeuver"]
    return nav2py_api["RobotProfile"](
        name=robot.model,
        kinematic_type=nav2py_api["KinematicType"].HOLONOMIC,
        footprint=nav2py_api["CircleFootprint"](radius),
        max_forward_velocity=max_linear_velocity,
        max_reverse_velocity=max_linear_velocity,
        max_lateral_velocity=max_linear_velocity,
        max_angular_velocity=max_angular_velocity,
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
        "max_forward_velocity": float(profile.max_forward_velocity),
        "max_reverse_velocity": float(profile.max_reverse_velocity),
        "max_lateral_velocity": float(profile.max_lateral_velocity),
        "max_angular_velocity": float(profile.max_angular_velocity),
    }


def robot_footprint_diagnostics(robot):
    extent_xy = robot.reset_joint_pos_aabb_extent[:2]
    return {
        "extent_xy": to_float_list(extent_xy),
        "bounding_radius": float(th.norm(extent_xy).item() / 2.0),
    }


def _symmetric_limit(lower, upper, index):
    return min(abs(float(lower[index])), abs(float(upper[index])))


def resolve_command_limits(robot_cfg, args):
    linear = args.command_max_linear_velocity
    angular = args.command_max_angular_velocity
    output_limits = robot_cfg.get("controller_config", {}).get("base", {}).get("command_output_limits")
    if output_limits is not None:
        lower, upper = output_limits
        if linear is None:
            linear = min(_symmetric_limit(lower, upper, 0), _symmetric_limit(lower, upper, 1))
        if angular is None:
            angular = _symmetric_limit(lower, upper, 2)

    return {
        "max_linear_velocity": None if linear is None else float(linear),
        "max_angular_velocity": None if angular is None else float(angular),
    }


def command_limits_diagnostics(command_limits):
    return {
        "max_linear_velocity": command_limits["max_linear_velocity"],
        "max_angular_velocity": command_limits["max_angular_velocity"],
    }


def parse_safety_slowdown_scales(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        scales = tuple(float(scale) for scale in value)
    elif value == "nav2py-default":
        return None
    else:
        scales = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not scales:
        raise ValueError("--safety-slowdown-scales must contain at least one value")
    if any(scale <= 0.0 or scale >= 1.0 for scale in scales):
        raise ValueError("--safety-slowdown-scales values must be in (0, 1)")
    return scales


class DiagnosticCollisionModel:
    def __init__(self, delegate, nav2py_api, disabled=False, slowdown_scales=None):
        self.delegate = delegate
        if slowdown_scales is not None:
            self.delegate._slowdown_scales = slowdown_scales
        self.disabled = disabled
        self.compatibility = delegate.compatibility
        self._safety_decision_cls = nav2py_api["SafetyDecision"]
        self._continue_action = nav2py_api["SafetyAction"].CONTINUE
        self.last_requested_command = None
        self.last_delegate_decision = None

    def evaluate(self, context):
        self.last_requested_command = context.command
        self.last_delegate_decision = self.delegate.evaluate(context)
        if self.disabled:
            return self._safety_decision_cls(context.command, self._continue_action)
        return self.last_delegate_decision

    def filter(self, command, pose, profile, costmap, *, lethal_cost, horizon):
        self.last_requested_command = command
        if self.disabled:
            return command
        return self.delegate.filter(
            command,
            pose,
            profile,
            costmap,
            lethal_cost=lethal_cost,
            horizon=horizon,
        )


def make_navigation_config(nav2py_api, args):
    config = nav2py_api["NavigationConfig"]()
    controller_updates = {}
    for arg_name, field_name in (
        ("desired_linear_velocity", "desired_linear_velocity"),
        ("min_lookahead_distance", "min_lookahead_distance"),
        ("max_lookahead_distance", "max_lookahead_distance"),
        ("lookahead_time", "lookahead_time"),
        ("collision_horizon", "collision_horizon"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            controller_updates[field_name] = value

    if controller_updates:
        config = replace(config, controller=replace(config.controller, **controller_updates))
    if args.planner_cost_penalty is not None:
        config = replace(config, planner=replace(config.planner, cost_penalty=args.planner_cost_penalty))
    if args.disable_path_smoothing:
        config = replace(config, planner=replace(config.planner, smooth_path=False))
    return config


def navigation_config_diagnostics(config):
    return {
        "planner": {
            "allow_diagonal": bool(config.planner.allow_diagonal),
            "cost_penalty": float(config.planner.cost_penalty),
            "lethal_cost": int(config.planner.lethal_cost),
            "max_planning_time": float(config.planner.max_planning_time),
            "smooth_path": bool(config.planner.smooth_path),
        },
        "controller": {
            "desired_linear_velocity": float(config.controller.desired_linear_velocity),
            "min_lookahead_distance": float(config.controller.min_lookahead_distance),
            "max_lookahead_distance": float(config.controller.max_lookahead_distance),
            "lookahead_time": float(config.controller.lookahead_time),
            "collision_horizon": float(config.controller.collision_horizon),
        },
    }


def make_navigator(
    profile,
    costmap,
    costmap_is_profile_inflated,
    nav2py_api,
    navigation_config,
    disable_dynamic_safety=False,
    safety_slowdown_scales=None,
):
    collision_model = DiagnosticCollisionModel(
        nav2py_api["FootprintCollisionModel"](),
        nav2py_api,
        disabled=disable_dynamic_safety,
        slowdown_scales=safety_slowdown_scales,
    )
    navigator = nav2py_api["Navigator"](
        profile,
        costmap,
        navigation_config,
        collision_model=collision_model,
    )
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


def step_trace_entry(
    step,
    now,
    state,
    costmap,
    final_position,
    final_distance,
    nav_status,
    command,
    executed_command,
    command_was_capped,
    requested_command,
    safety_decision,
    safety_decision_without_override,
):
    return {
        "step": int(step),
        "time": float(now),
        "pose": {
            "x": float(state.pose.x),
            "y": float(state.pose.y),
            "yaw": float(state.pose.yaw),
        },
        "linear_velocity_body": [float(state.linear_velocity[0]), float(state.linear_velocity[1])],
        "angular_velocity_body": float(state.angular_velocity),
        "map": point_cost_diagnostic(costmap, final_position[:2]),
        "final_distance": float(final_distance),
        "nav2py_state": nav_status.state.value,
        "nav2py_reason": nav_status.reason,
        "requested_command": command_diagnostic(requested_command),
        "command": command_diagnostic(command),
        "executed_command": command_diagnostic(executed_command),
        "command_was_capped": bool(command_was_capped),
        "safety_decision": safety_decision_diagnostic(safety_decision),
        "safety_decision_without_override": safety_decision_diagnostic(safety_decision_without_override),
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


def clamp_abs(value, limit):
    if limit is None:
        return value
    return max(-limit, min(limit, value))


def cap_command_to_controller_limits(command, command_limits):
    if command is None or command.is_stop:
        return command

    velocity = command.velocity
    capped_velocity = type(velocity)(
        vx=clamp_abs(velocity.vx, command_limits["max_linear_velocity"]),
        vy=clamp_abs(velocity.vy, command_limits["max_linear_velocity"]),
        wz=clamp_abs(velocity.wz, command_limits["max_angular_velocity"]),
    )
    if capped_velocity == velocity:
        return command
    return replace(command, velocity=capped_velocity)


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


def zero_robot_velocities(robot):
    robot.set_linear_velocity(th.zeros(3))
    robot.set_angular_velocity(th.zeros(3))
    robot.set_joint_velocities(th.zeros(robot.n_dof), drive=False)


def place_robot(robot, episode):
    position = th.tensor(episode["start_position"], dtype=th.float32)
    orientation = th.tensor(episode["start_quat"], dtype=th.float32)
    robot.set_joint_positions(robot.reset_joint_pos, drive=False)
    robot.set_position_orientation(position=position, orientation=orientation)
    zero_robot_velocities(robot)


def look_at_orientation(camera_position, target_position):
    forward = target_position - camera_position
    forward = forward / th.norm(forward)
    world_up = th.tensor([0.0, 0.0, 1.0], dtype=th.float32, device=forward.device)
    if th.norm(th.linalg.cross(forward, world_up)) < 1e-6:
        world_up = th.tensor([0.0, 1.0, 0.0], dtype=th.float32, device=forward.device)

    right = th.linalg.cross(forward, world_up)
    right = right / th.norm(right)
    up = th.linalg.cross(-forward, right)
    up = up / th.norm(up)

    # USD cameras look along local -Z, so the local Z axis points away from the target.
    rotation = th.stack([right, up, -forward], dim=1)
    return T.mat2quat(rotation)


def update_viewer_camera(robot, args):
    if args.viewer_camera_mode != "follow" or gm.HEADLESS:
        return

    position, orientation = robot.get_position_orientation()
    yaw = float(T.quat2euler(orientation)[2].item())
    offset = th.tensor(
        [
            -math.cos(yaw) * args.viewer_camera_distance,
            -math.sin(yaw) * args.viewer_camera_distance,
            args.viewer_camera_height,
        ],
        dtype=th.float32,
    )
    camera_position = position + offset
    target_position = position + th.tensor([0.0, 0.0, args.viewer_camera_target_height], dtype=th.float32)
    camera_orientation = look_at_orientation(camera_position, target_position)
    og.sim.viewer_camera.set_position_orientation(position=camera_position, orientation=camera_orientation)


def apply_deadband(value, deadband):
    value = float(value)
    return 0.0 if abs(value) < deadband else value


def robot_state_estimate(robot, timestamp, nav2py_api, args):
    position, orientation = robot.get_position_orientation()
    yaw = float(T.quat2euler(orientation)[2].item())
    rotation_world_to_body = T.quat2mat(orientation).T
    linear_velocity = rotation_world_to_body @ robot.get_linear_velocity()
    angular_velocity = rotation_world_to_body @ robot.get_angular_velocity()
    vx = apply_deadband(linear_velocity[0].item(), args.state_linear_velocity_deadband)
    vy = apply_deadband(linear_velocity[1].item(), args.state_linear_velocity_deadband)
    wz = apply_deadband(angular_velocity[2].item(), args.state_angular_velocity_deadband)

    return nav2py_api["StateEstimate"](
        timestamp=timestamp,
        frame_id="map",
        pose=nav2py_api["Pose2D"](float(position[0].item()), float(position[1].item()), yaw),
        linear_velocity=(vx, vy),
        angular_velocity=wz,
        velocity_available=True,
    )


def xy_distance(position, goal):
    return math.hypot(float(position[0]) - float(goal[0]), float(position[1]) - float(goal[1]))


def point_inside_robot_footprint(robot, point):
    position, orientation = robot.get_position_orientation()
    yaw = float(T.quat2euler(orientation)[2].item())
    dx = float(point[0]) - float(position[0])
    dy = float(point[1]) - float(position[1])
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    half_extent = robot.reset_joint_pos_aabb_extent[:2] / 2.0
    return abs(local_x) <= float(half_extent[0]) and abs(local_y) <= float(half_extent[1])


def run_episode(env, robot, episode, costmap_bundle, profile, navigation_config, command_limits, nav2py_api, args):
    env.reset(get_obs=False)
    place_robot(robot, episode)
    update_viewer_camera(robot, args)

    for _ in range(args.settle_steps):
        env.step({robot.name: controller_no_op_action(robot)})
        update_viewer_camera(robot, args)

    zero_robot_velocities(robot)
    update_viewer_camera(robot, args)

    costmap, costmap_is_profile_inflated = select_costmap(costmap_bundle, args.costmap_source)
    navigator = make_navigator(
        profile,
        costmap,
        costmap_is_profile_inflated,
        nav2py_api,
        navigation_config,
        disable_dynamic_safety=args.disable_dynamic_safety,
        safety_slowdown_scales=args.safety_slowdown_scales,
    )
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
    last_executed_command = None
    last_requested_command = None
    last_safety_decision = None
    last_safety_decision_without_override = None
    step_trace = []
    first_lethal_cell_event = None
    capped_command_steps = 0
    for step in range(args.max_steps):
        now = step * dt
        state = robot_state_estimate(robot, now, nav2py_api, args)
        command = navigator.tick(state, now)
        executed_command = cap_command_to_controller_limits(command, command_limits)
        command_was_capped = executed_command != command
        if command_was_capped:
            capped_command_steps += 1
        last_state = state
        last_command = command
        last_executed_command = executed_command
        last_requested_command = navigator.collision_model.last_requested_command
        last_safety_decision = navigator.last_safety_decision
        last_safety_decision_without_override = navigator.collision_model.last_delegate_decision
        if executed_command is not None and not executed_command.is_stop:
            commanded_steps += 1

        env.step({robot.name: action_from_nav2py_command(robot, executed_command)})
        update_viewer_camera(robot, args)
        if args.visual_step_sleep > 0.0:
            time.sleep(args.visual_step_sleep)
        position, _ = robot.get_position_orientation()
        final_distance = xy_distance(position[:2], goal[:2])
        goal_inside_footprint = point_inside_robot_footprint(robot, goal)
        success = (
            goal_inside_footprint
            if args.success_criterion == "footprint-overlap"
            else final_distance <= args.success_distance
        )
        nav_status = navigator.status()
        map_diagnostic = point_cost_diagnostic(navigator.costmap, position[:2])
        if first_lethal_cell_event is None and map_diagnostic["cost"] is not None and map_diagnostic["cost"] >= 253:
            first_lethal_cell_event = {
                "step": int(step),
                "time": float(now),
                "position": to_float_list(position),
                "map": map_diagnostic,
                "final_distance": float(final_distance),
            }
        if args.trace_failures:
            step_trace.append(
                step_trace_entry(
                    step,
                    now,
                    state,
                    navigator.costmap,
                    position,
                    final_distance,
                    nav_status,
                    command,
                    executed_command,
                    command_was_capped,
                    last_requested_command,
                    last_safety_decision,
                    last_safety_decision_without_override,
                )
            )
        if success or nav_status.state.value in {"succeeded", "failed", "blocked", "canceled"}:
            break

    status = navigator.status()
    position, orientation = robot.get_position_orientation()
    final_yaw = float(T.quat2euler(orientation)[2].item())
    result = {
        "episode_id": episode["episode_id"],
        "scene_model": episode["scene_model"],
        "task_name": episode["task_name"],
        "scene_instance": episode["scene_instance"],
        "load_room_instances": episode["load_room_instances"],
        "floor": int(episode.get("floor", 0)),
        "success": success,
        "success_criterion": args.success_criterion,
        "costmap_source": args.costmap_source,
        "robot_profile": robot_profile_diagnostics(profile),
        "robot_footprint": robot_footprint_diagnostics(robot),
        "controller_command_limits": command_limits_diagnostics(command_limits),
        "costmap_diagnostics": costmap_diagnostics,
        "nav2py_state": status.state.value,
        "nav2py_reason": status.reason,
        "steps": step + 1,
        "commanded_steps": commanded_steps,
        "capped_command_steps": capped_command_steps,
        "sim_time": (step + 1) * dt,
        "start_position": episode["start_position"],
        "goal_position": episode["goal_position"],
        "final_position": to_float_list(position),
        "final_yaw": final_yaw,
        "final_distance": xy_distance(position[:2], goal[:2]),
        "goal_inside_robot_footprint": point_inside_robot_footprint(robot, goal),
        "geodesic_distance": float(episode["geodesic_distance"]),
        "remaining_distance": status.remaining_distance,
        "progress": status.progress,
        "last_tick_state": state_estimate_diagnostic(last_state, navigator.costmap),
        "final_robot_state": robot_state_diagnostic(robot, (step + 1) * dt, navigator.costmap),
        "last_command": command_diagnostic(last_command),
        "last_executed_command": command_diagnostic(last_executed_command),
        "last_requested_command": command_diagnostic(last_requested_command),
        "last_safety_decision": safety_decision_diagnostic(last_safety_decision),
        "last_safety_decision_without_override": safety_decision_diagnostic(last_safety_decision_without_override),
        "first_lethal_cell_event": first_lethal_cell_event,
    }
    if args.trace_failures and not success:
        result["step_trace"] = step_trace
    return result


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


def write_results(path, benchmark_path, nav2py_root, navigation_config, command_limits, args, results):
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": str(Path(benchmark_path).expanduser()),
        "nav2py_root": None if nav2py_root is None else str(Path(nav2py_root).expanduser()),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "success_distance": args.success_distance,
        "success_criterion": args.success_criterion,
        "costmap_source": args.costmap_source,
        "dynamic_safety_disabled": args.disable_dynamic_safety,
        "trace_failures": args.trace_failures,
        "soft_cost_radius": args.soft_cost_radius,
        "soft_cost_scaling_factor": args.soft_cost_scaling_factor,
        "safety_slowdown_scales": args.safety_slowdown_scales,
        "state_linear_velocity_deadband": args.state_linear_velocity_deadband,
        "state_angular_velocity_deadband": args.state_angular_velocity_deadband,
        "navigation_config": navigation_config_diagnostics(navigation_config),
        "controller_command_limits": command_limits_diagnostics(command_limits),
        "summary": summarize_results(results),
        "episodes": results,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return output


def main(args=None, shutdown=True):
    args = parse_args() if args is None else args
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.success_distance <= 0.0:
        raise ValueError("--success-distance must be positive")
    for arg_name in (
        "desired_linear_velocity",
        "min_lookahead_distance",
        "max_lookahead_distance",
        "lookahead_time",
        "collision_horizon",
        "profile_max_linear_velocity",
        "profile_max_angular_velocity",
        "state_linear_velocity_deadband",
        "state_angular_velocity_deadband",
        "command_max_linear_velocity",
        "command_max_angular_velocity",
        "soft_cost_radius",
        "soft_cost_scaling_factor",
        "planner_cost_penalty",
        "visual_step_sleep",
        "keep_open_seconds",
        "viewer_camera_distance",
        "viewer_camera_height",
        "viewer_camera_target_height",
    ):
        value = getattr(args, arg_name)
        if value is not None and value < 0.0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be positive")
        if arg_name not in {"visual_step_sleep", "keep_open_seconds"} and value is not None and value == 0.0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be positive")

    args.safety_slowdown_scales = parse_safety_slowdown_scales(args.safety_slowdown_scales)

    seed_everything(args.seed)
    add_nav2py_to_path(args.nav2py_root)
    nav2py_api = load_nav2py()
    navigation_config = make_navigation_config(nav2py_api, args)
    if navigation_config.controller.min_lookahead_distance > navigation_config.controller.max_lookahead_distance:
        raise ValueError("--min-lookahead-distance must be less than or equal to --max-lookahead-distance")
    _, episodes = load_benchmark(args.benchmark)
    episodes = filter_episodes(episodes, args.episode_ids)

    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False

    robot_cfg = load_robot_config(args.robot_config)
    command_limits = resolve_command_limits(robot_cfg, args)
    results = []
    try:
        scene_groups = list(group_episodes_by_scene(episodes).items())
        for scene_index, ((scene_model, scene_instance, _), scene_episodes) in enumerate(scene_groups):
            print(f"\nRunning template: {scene_instance} ({len(scene_episodes)} episodes)")
            cfg = build_env_config(
                scene_model=scene_model,
                robot_cfg=robot_cfg,
                scene_instance=scene_instance,
                load_room_instances=scene_episodes[0]["load_room_instances"],
            )
            env = og.Environment(configs=cfg)
            robot = env.robots[0]
            if robot.model in ("r1", "r1pro"):
                og.sim.stop()
                robot.base_footprint_link.mass = 250.0
                og.sim.play()
            profile = make_robot_profile(
                robot,
                nav2py_api,
                args,
                clearance_is_in_costmap=args.costmap_source in {"og-eroded", "og-eroded-soft"},
            )

            costmap_bundles = {}
            for episode in scene_episodes:
                floor = int(episode.get("floor", 0))
                if floor not in costmap_bundles:
                    costmap_bundles[floor] = make_costmap_bundle(env.scene, floor, robot, nav2py_api, args)
                result = run_episode(
                    env,
                    robot,
                    episode,
                    costmap_bundles[floor],
                    profile,
                    navigation_config,
                    command_limits,
                    nav2py_api,
                    args,
                )
                results.append(result)
                active = result["costmap_diagnostics"]["active"]
                print(
                    f"  {result['episode_id']}: "
                    f"{'SUCCESS' if result['success'] else 'FAIL'} "
                    f"final_distance={result['final_distance']:.3f}m "
                    f"criterion={result['success_criterion']} "
                    f"state={result['nav2py_state']}"
                )
                if not result["success"]:
                    print(
                        f"    active_costs: start={active['start']['cost']} "
                        f"goal={active['goal']['cost']} "
                        f"reason={result['nav2py_reason']}"
                    )

            if not (args.keep_open_on_complete and scene_index == len(scene_groups) - 1):
                og.clear()

        output = write_results(
            args.output,
            args.benchmark,
            args.nav2py_root,
            navigation_config,
            command_limits,
            args,
            results,
        )
        summary = summarize_results(results)
        print(f"\nSaved results to: {output}")
        print(
            f"Success rate: {summary['successes']}/{summary['total']} "
            f"({summary['success_rate']:.1%}) using {args.success_criterion}"
        )
        if shutdown and args.keep_open_on_complete:
            keep_viewer_open(args.keep_open_seconds)
    except Exception:
        if not shutdown:
            og.clear()
        raise
    finally:
        if shutdown:
            og.shutdown()


def keep_viewer_open(seconds):
    if seconds is not None:
        print(f"\nKeeping viewer open for {seconds:.1f}s...")
        time.sleep(seconds)
        return

    try:
        input("\nRun complete. Press Enter to close OmniGibson...")
    except EOFError:
        print("\nRun complete. Press Ctrl+C to close OmniGibson.")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
