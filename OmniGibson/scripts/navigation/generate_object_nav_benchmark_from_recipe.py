#!/usr/bin/env python3
"""Generate an ordered navigation benchmark from a task recipe."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import omnigibson as og
from generate_nav_benchmark import CHALLENGE_SCENES, seed_everything
from generate_object_nav_benchmark import create_context, find_templates, generate_pair_episode, parse_radii
from omnigibson.macros import gm


def load_recipe(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        recipe = json.load(f)
    if not isinstance(recipe, dict) or not isinstance(recipe.get("task"), str):
        raise ValueError("Recipe must contain a task string")
    chain = recipe.get("navigation_chain")
    if not isinstance(chain, list) or not chain:
        raise ValueError("Recipe must contain a non-empty navigation_chain")
    for index, leg in enumerate(chain):
        if not isinstance(leg, dict) or not isinstance(leg.get("start"), str) or not isinstance(leg.get("end"), str):
            raise ValueError(f"navigation_chain[{index}] must contain start and end strings")
        if index == 0 and leg["start"] != "robot_initial":
            raise ValueError("The first recipe leg must start at robot_initial")
        if index and leg["start"] != chain[index - 1]["end"]:
            raise ValueError(f"navigation_chain[{index}] must start at the preceding leg's end")
    overrides = recipe.get("state_overrides", [])
    if not isinstance(overrides, list):
        raise ValueError("state_overrides must be a list")
    for override in overrides:
        if (
            not isinstance(override, dict)
            or not isinstance(override.get("object"), str)
            or override.get("state") != "open"
        ):
            raise ValueError("Each state override must contain object and state: open")
    return recipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--task-instances-root", default="datasets")
    parser.add_argument("--scene", choices=CHALLENGE_SCENES, default="house_double_floor_lower")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-instances", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--floor", type=int, default=0)
    parser.add_argument("--min-distance", type=float, default=1.0)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--approach-radii", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--angles-per-radius", type=int, default=36)
    parser.add_argument("--extra-clearance", type=float, default=0.2)
    parser.add_argument("--robot-pose-key", default="R1Pro")
    parser.add_argument(
        "--robot-config",
        default=str(Path(__file__).resolve().parents[2] / "omnigibson" / "eval" / "r1pro.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_distance > args.max_distance:
        raise ValueError("--min-distance must be <= --max-distance")
    if args.num_instances < 1:
        raise ValueError("--num-instances must be at least 1")
    args.recipe = Path(args.recipe).expanduser()
    args.task_instances_root = Path(args.task_instances_root).expanduser()
    args.radii = parse_radii(args.approach_radii)
    recipe = load_recipe(args.recipe)
    if recipe.get("scene") not in (None, args.scene):
        raise ValueError(f"Recipe scene {recipe['scene']!r} does not match --scene {args.scene!r}")
    args.state_overrides = recipe.get("state_overrides", [])

    seed_everything(args.seed)
    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False

    templates = find_templates(args.task_instances_root, args.scene, recipe["task"])
    if args.num_instances > len(templates):
        raise ValueError(
            f"Requested {args.num_instances} task instances, but only {len(templates)} are available for "
            f"{args.scene}/{recipe['task']}"
        )
    instance_ids = random.Random(args.seed).sample(sorted(templates), args.num_instances)
    try:
        episodes = []
        for instance_id in instance_ids:
            context = create_context(args, templates[instance_id], recipe["task"], instance_id)
            try:
                prior_goal = None
                for index, leg in enumerate(recipe["navigation_chain"]):
                    episode = generate_pair_episode(
                        context=context,
                        start_reference=leg["start"],
                        goal_reference=leg["end"],
                        episode_id=f"{args.scene}_{recipe['task']}_{instance_id:03d}_{index:02d}",
                        start_position=prior_goal,
                    )
                    episodes.append(episode)
                    prior_goal = episode["goal_position"]
                    print(
                        f"[{instance_id:03d}:{index + 1:02d}] {leg['start']} -> {leg['end']}: "
                        f"{episode['geodesic_distance']:.3f} m",
                        flush=True,
                    )
            finally:
                og.clear()

        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scene": args.scene,
                    "task": recipe["task"],
                    "recipe_path": str(args.recipe),
                    "num_instances": args.num_instances,
                    "sampled_task_instance_ids": instance_ids,
                    "state_overrides": args.state_overrides,
                    "episodes": episodes,
                },
                f,
                indent=2,
            )
            f.write("\n")
    finally:
        og.clear()
        og.shutdown()


if __name__ == "__main__":
    main()
