"""Interactively step through one saved navigation benchmark episode in the OmniGibson viewer."""

import argparse
import sys
import time
from pathlib import Path

import omnigibson as og
import omnigibson.lazy as lazy
import torch as th

import run_nav2py_benchmark as runner
from generate_nav_benchmark import build_env_config, load_robot_config, seed_everything
from omnigibson.macros import gm
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.utils.ui_utils import KeyboardEventHandler


DEFAULT_OUTPUT = "outputs/navigation/interactive_nav2py_result.json"
GOAL_MARKER_HEIGHT = 0.06


def option_was_supplied(option, argv):
    return option in argv or any(value.startswith(f"{option}=") for value in argv)


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description="Open one benchmark episode and wait for Enter before every nav2py control step."
    )
    parser.add_argument("--episode-id", required=True, help="The one full benchmark episode ID to visualize.")
    interactive_args, runner_argv = parser.parse_known_args(argv)

    if "--episode-ids" in runner_argv:
        parser.error("Use --episode-id; this visualizer supports exactly one episode.")

    args = runner.parse_args(runner_argv)
    args.episode_ids = [interactive_args.episode_id]
    args.keep_open_on_complete = True
    if not option_was_supplied("--output", runner_argv):
        args.output = DEFAULT_OUTPUT
    if not option_was_supplied("--viewer-camera-mode", runner_argv):
        args.viewer_camera_mode = "follow"
    if not option_was_supplied("--viewer-camera-distance", runner_argv):
        args.viewer_camera_distance = 1.5
    if not option_was_supplied("--viewer-camera-height", runner_argv):
        args.viewer_camera_height = 1.7
    if not option_was_supplied("--viewer-camera-target-height", runner_argv):
        args.viewer_camera_target_height = 0.55
    args.viewer_camera_lateral_offset = 1.1
    args.viewer_camera_target_forward_offset = 0.7
    return args


def command_text(command):
    if command is None:
        return "none"
    if command.is_stop:
        return "stop"
    velocity = command.velocity
    return f"vx={velocity.vx:.3f} vy={velocity.vy:.3f} wz={velocity.wz:.3f}"


class VisualizationClosed(Exception):
    pass


class VisualStepGate:
    """Keeps Kit rendering while the user inspects the current navigation state."""

    def __init__(self):
        self.advance_requested = False
        self.closed = False

    def advance(self):
        self.advance_requested = True

    def close(self):
        self.closed = True

    def wait(self, message):
        self.advance_requested = False
        print(f"\n{message}\nClick the OmniGibson viewport, then press N for the next step. Press Esc to close.")
        while not self.advance_requested and not self.closed:
            og.sim.render()
            time.sleep(0.01)
        if self.closed:
            raise VisualizationClosed

    def wait_until_ready(self):
        self.wait("Start pose is loaded and settled.")

    def wait_for_control_step(self, step, now, state, command, executed_command):
        del state, command
        self.wait(f"Step {step:04d}  t={now:.2f}s  command: {command_text(executed_command)}")

    def wait_until_closed(self):
        print("\nRun complete. Inspect the viewer; press Esc in the viewport to close.")
        while not self.closed:
            og.sim.render()
            time.sleep(0.01)


def add_goal_marker(env, episode):
    marker = PrimitiveObject(
        relative_prim_path="/interactive_nav_goal_marker",
        name="interactive_nav_goal_marker",
        primitive_type="Cylinder",
        radius=0.15,
        height=GOAL_MARKER_HEIGHT,
        visual_only=True,
        rgba=th.tensor([0.0, 0.7, 1.0, 0.85]),
    )
    env.scene.add_object(marker)
    position = th.tensor(episode["goal_position"], dtype=th.float32)
    position[2] = env.scene.get_floor_height(int(episode.get("floor", 0))) + GOAL_MARKER_HEIGHT / 2.0
    return marker, position


def main(argv=None):
    args = parse_args(argv)
    if gm.HEADLESS or gm.REMOTE_STREAMING:
        raise RuntimeError(
            "This script needs a desktop OmniGibson session. Unset OMNIGIBSON_HEADLESS "
            "and OMNIGIBSON_REMOTE_STREAMING first."
        )
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.success_distance <= 0.0:
        raise ValueError("--success-distance must be positive")

    args.safety_slowdown_scales = runner.parse_safety_slowdown_scales(args.safety_slowdown_scales)
    seed_everything(args.seed)
    runner.add_nav2py_to_path(args.nav2py_root)
    nav2py_api = runner.load_nav2py()
    navigation_config = runner.make_navigation_config(nav2py_api, args)
    if navigation_config.controller.min_lookahead_distance > navigation_config.controller.max_lookahead_distance:
        raise ValueError("--min-lookahead-distance must be less than or equal to --max-lookahead-distance")

    _, episodes = runner.load_benchmark(args.benchmark)
    episode = runner.filter_episodes(episodes, args.episode_ids)[0]
    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False

    robot_cfg = load_robot_config(args.robot_config)
    command_limits = runner.resolve_command_limits(robot_cfg, args)
    cfg = build_env_config(
        scene_model=episode["scene_model"],
        robot_cfg=robot_cfg,
        scene_instance=episode["scene_instance"],
        load_room_instances=episode["load_room_instances"],
    )

    try:
        env = og.Environment(configs=cfg)
        gate = VisualStepGate()
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.N, gate.advance)
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.ESCAPE, gate.close)
        camera_mover = og.sim.enable_viewer_camera_teleoperation()
        camera_mover.set_delta(0.5)
        robot = env.robots[0]
        goal_marker, goal_marker_position = add_goal_marker(env, episode)
        if robot.model in ("r1", "r1pro"):
            og.sim.stop()
            robot.base_footprint_link.mass = 250.0
            og.sim.play()
        profile = runner.make_robot_profile(
            robot,
            nav2py_api,
            args,
            clearance_is_in_costmap=args.costmap_source in {"og-eroded", "og-eroded-soft"},
        )
        costmap_bundle = runner.make_costmap_bundle(env.scene, int(episode.get("floor", 0)), robot, nav2py_api, args)

        print(f"\nLoaded {episode['episode_id']}.")
        print(f"Success criterion: {runner.format_success_criterion(args, robot)}")

        def after_reset():
            goal_marker.set_position_orientation(position=goal_marker_position)
            gate.wait_until_ready()

        result = runner.run_episode(
            env,
            robot,
            episode,
            costmap_bundle,
            profile,
            navigation_config,
            command_limits,
            nav2py_api,
            args,
            after_reset=after_reset,
            before_control_step=gate.wait_for_control_step,
        )
        output = runner.write_results(
            args.output,
            args.benchmark,
            args.nav2py_root,
            navigation_config,
            command_limits,
            args,
            [result],
        )
        print(
            f"\n{'SUCCESS' if result['success'] else 'FAIL'}: {result['episode_id']} "
            f"final_distance={result['final_distance']:.3f}m state={result['nav2py_state']}"
        )
        print(f"Saved result to: {Path(output)}")
        gate.wait_until_closed()
    except VisualizationClosed:
        print("\nVisualization closed.")
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
