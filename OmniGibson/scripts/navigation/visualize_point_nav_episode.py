"""Interactively step through one random point-navigation benchmark episode."""

import argparse
import sys
from pathlib import Path

import omnigibson as og
import omnigibson.lazy as lazy

import run_nav2py_benchmark as runner
from generate_nav_benchmark import build_env_config, load_robot_config, seed_everything
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardEventHandler
from visualize_nav2py_episode import VisualStepGate, VisualizationClosed, add_goal_marker, option_was_supplied


DEFAULT_OUTPUT = "outputs/navigation/interactive_point_nav_result.json"


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
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
        args.viewer_camera_distance = 0.9
    if not option_was_supplied("--viewer-camera-height", runner_argv):
        args.viewer_camera_height = 1.45
    if not option_was_supplied("--viewer-camera-target-height", runner_argv):
        args.viewer_camera_target_height = 0.55
    args.viewer_camera_lateral_offset = 0.65
    args.viewer_camera_target_forward_offset = 0.4
    return args


def main(argv=None):
    args = parse_args(argv)
    if gm.HEADLESS and not gm.REMOTE_STREAMING:
        raise RuntimeError(
            "This script needs either a desktop viewer or OMNIGIBSON_REMOTE_STREAMING=native."
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
    runtime_extra_clearance = runner.effective_runtime_extra_clearance([episode], args.runtime_extra_clearance)
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
        if gm.REMOTE_STREAMING:
            print(f"Remote streaming enabled: {gm.REMOTE_STREAMING}")
        gate = VisualStepGate()
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.N, gate.advance)
        KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.ESCAPE, gate.close)
        camera_mover = og.sim.enable_viewer_camera_teleoperation()
        camera_mover.set_delta(0.5)
        robot = env.robots[0]
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
        costmap_bundle = runner.make_costmap_bundle(
            env.scene,
            int(episode.get("floor", 0)),
            robot,
            nav2py_api,
            args,
            runtime_extra_clearance,
        )

        print(f"\nLoaded {episode['episode_id']}.")
        print(f"Costmap: {args.costmap_source}; runtime extra clearance={runtime_extra_clearance:.3f}m")
        if runtime_extra_clearance > args.runtime_extra_clearance:
            print(
                f"  Using benchmark validation clearance {runtime_extra_clearance:.3f}m "
                f"instead of requested {args.runtime_extra_clearance:.3f}m."
            )
        print(f"Success criterion: {runner.format_success_criterion(args, robot)}")

        def after_reset():
            goal_marker, goal_marker_position = add_goal_marker(env, episode)
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
    except Exception as exc:
        print(f"\nVisualizer failed during setup or stepping: {type(exc).__name__}: {exc}")
        raise
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
