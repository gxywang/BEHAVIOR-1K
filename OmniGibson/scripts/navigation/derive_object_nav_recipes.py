#!/usr/bin/env python3
"""Derive object-navigation recipes from the teleoperation demonstrations.

The demonstrated ``navigation: move to`` skills are the source of truth: their
``object_id`` values are resolved against the task's ``0_0`` template with
:mod:`list_object_nav_references`, chained in demonstrated order, and each leg is
then verified for REACHABILITY on the ground-truth ``navigation_2d`` map of the
scene -- a cell that is traversable in the raw map is not necessarily plannable
once the map is eroded for the robot footprint and the runtime clearance.

Verification replicates, cell for cell, the planning map
``run_nav2py_benchmark.make_b1k_costmap`` builds for ``--costmap-source b1k-gt``
(OmniGibson's own robot erosion in metres, then a disk of extra clearance) and
the approach-pose choice ``generate_object_nav_benchmark.project_goal_near_object``
makes (rings around the goal object, ``min`` by projection radius then geodesic
distance). Nothing here imports omnigibson, so it runs on a login node.

Each emitted recipe is valid input to generate_object_nav_benchmark_from_recipe.py.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from list_object_nav_references import (
    DEFAULT_DEMO_ROOT,
    flatten_object_ids,
    join_strings,
    resolve_reference,
)

DEFAULT_TASK_INSTANCES_ROOT = "datasets/2026-challenge-task-instances"
DEFAULT_MAP_ROOT = "/scratch/gxwang2/b1k/b1k_gt_out/final_v4"
DEFAULT_NAV2PY_ROOT = "/scratch/gxwang2/b1k/b1k-submission/third-party/nav2py"
DEFAULT_OUTPUT_ROOT = "/scratch/gxwang2/b1k/objnav/recipes"

# R1Pro: norm(reset_joint_pos_aabb_extent[:2]) / 2, the radius run_nav2py_benchmark
# uses for the robot profile and for OmniGibson's own traversability erosion.
R1PRO_BOUNDING_RADIUS = 0.4996
# OmniGibson builds its traversability map at this resolution; the metres of
# clearance its square erosion removes depend on it (see og_robot_erosion_meters).
OG_MAP_RESOLUTION = 0.1

# Structural categories are resolvable but are not object-navigation goals.
STRUCTURAL_CATEGORIES = frozenset({"floors", "walls", "ceilings", "driveway", "lawn"})

DROP_REASONS = (
    "unresolved_reference",
    "ambiguous_reference",
    "structural_goal",
    "repeat_of_previous_goal",
    "goal_object_missing_pose",
    "no_traversable_approach_pose",
    "no_clearance_at_approach_poses",
    "no_reachable_approach_pose",
    "shorter_than_min_distance",
    "longer_than_max_distance",
)


def og_robot_erosion_meters(robot_radius: float, og_resolution: float) -> float:
    """Metres of clearance OmniGibson's ``_erode_trav_map`` actually removes.

    Copied from run_nav2py_benchmark so the recipe check and the runtime plan on
    the same free space: a ``radius_pixel`` square anchored at its centre removes
    ``radius_pixel // 2`` cells, i.e. 0.30 m for the R1Pro at a 0.1 m map.
    """
    radius_pixel = int(math.ceil((robot_radius + 0.2) / og_resolution))
    return (radius_pixel // 2) * og_resolution


def disk_kernel(radius_m: float, resolution: float) -> np.ndarray:
    radius_cells = int(math.ceil(radius_m / resolution))
    offsets = np.arange(-radius_cells, radius_cells + 1)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    return ((xx * resolution) ** 2 + (yy * resolution) ** 2 <= radius_m**2).astype(np.uint8)


class SceneMap:
    """Ground-truth navigation_2d map of one scene, eroded to plannable free space."""

    def __init__(self, map_root: Path, scene: str, robot_radius: float, extra_clearance: float,
                 og_resolution: float) -> None:
        import cv2
        from scipy import ndimage

        from benchmarks.costmaps import load_navigation_2d

        directory = Path(map_root).expanduser() / scene / "navigation_2d"
        grid, resolution, origin, metadata = load_navigation_2d(directory)
        self.scene = scene
        self.directory = directory
        self.resolution = resolution
        self.origin = origin
        self.traversable = grid == metadata["cell_values"]["traversable"]

        free = np.where(self.traversable, 255, 0).astype(np.uint8)
        self.robot_erosion_m = og_robot_erosion_meters(robot_radius, og_resolution)
        half_cells = int(round(self.robot_erosion_m / resolution))
        side = 2 * half_cells + 1
        free = cv2.erode(free, np.ones((side, side), dtype=np.uint8))
        if extra_clearance > 0.0:
            free = cv2.erode(free, disk_kernel(extra_clearance, resolution))
        self.free = free == 255
        self.labels, self.num_components = ndimage.label(self.free, structure=np.ones((3, 3), dtype=int))

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = int(math.floor((y - self.origin[1]) / self.resolution))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (self.origin[0] + (col + 0.5) * self.resolution,
                self.origin[1] + (row + 0.5) * self.resolution)

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.free.shape[0] and 0 <= col < self.free.shape[1]

    def is_free(self, cell: tuple[int, int]) -> bool:
        return self.in_bounds(cell) and bool(self.free[cell])

    def is_traversable(self, cell: tuple[int, int]) -> bool:
        return self.in_bounds(cell) and bool(self.traversable[cell])

    def component(self, cell: tuple[int, int]) -> int:
        return int(self.labels[cell]) if self.in_bounds(cell) else 0

    def snap_to_free(self, cell: tuple[int, int], radius_m: float) -> tuple[tuple[int, int], float] | None:
        """Nearest plannable cell within ``radius_m``; used only for the robot's own start pose."""
        if self.is_free(cell):
            return cell, 0.0
        radius_cells = int(math.ceil(radius_m / self.resolution))
        row, col = cell
        rows = slice(max(row - radius_cells, 0), min(row + radius_cells + 1, self.free.shape[0]))
        cols = slice(max(col - radius_cells, 0), min(col + radius_cells + 1, self.free.shape[1]))
        window = self.free[rows, cols]
        if not window.any():
            return None
        candidate_rows, candidate_cols = np.nonzero(window)
        candidate_rows = candidate_rows + rows.start
        candidate_cols = candidate_cols + cols.start
        distances = np.hypot(candidate_rows - row, candidate_cols - col) * self.resolution
        best = int(np.argmin(distances))
        if distances[best] > radius_m:
            return None
        return (int(candidate_rows[best]), int(candidate_cols[best])), float(distances[best])

    def geodesic_distances(self, start_cell, targets, max_distance):
        """Geodesic metres from ``start_cell`` to each target cell, over plannable cells only.

        The search is cropped to a window of ``max_distance`` around the start -- a path no
        longer than that cannot leave it -- and stopped once the frontier passes that cost, so
        distances at or below ``max_distance`` are exact and longer ones are reported as inf.
        """
        from skimage.graph import MCP_Geometric

        if not targets:
            return {}
        half = int(math.ceil(max_distance / self.resolution)) + 2
        rows, cols = self.free.shape
        row0, row1 = max(start_cell[0] - half, 0), min(start_cell[0] + half + 1, rows)
        col0, col1 = max(start_cell[1] - half, 0), min(start_cell[1] + half + 1, cols)
        window = self.free[row0:row1, col0:col1]
        costs = np.where(window, 1.0, np.inf)
        local = {}
        for cell in targets:
            if row0 <= cell[0] < row1 and col0 <= cell[1] < col1:
                local[cell] = (cell[0] - row0, cell[1] - col0)
        if not local:
            return {}
        mcp = MCP_Geometric(costs, fully_connected=True)
        cumulative, _ = mcp.find_costs(
            [[start_cell[0] - row0, start_cell[1] - col0]],
            ends=list(local.values()),
            find_all_ends=True,
        )
        cumulative = np.asarray(cumulative, dtype=np.float64)
        return {cell: float(cumulative[index]) * self.resolution for cell, index in local.items()}


def parse_radii(value: str) -> list[float]:
    radii = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not radii or any(radius < 0 for radius in radii):
        raise ValueError("--approach-radii must contain non-negative numbers")
    return radii


def scene_of_each_task(task_instances_root: Path) -> tuple[dict[str, tuple[str, Path]], int]:
    """Map task name -> (scene, 0_0 template path) from the challenge task instances.

    The root ships each 0_0 template twice (``scenes/`` and ``scene_test/public/``). The two
    copies hold the same objects at the same poses but different robot start poses, so which one
    wins matters; the first sorted path wins here exactly as generate_object_nav_benchmark's
    find_templates picks it, so a recipe is verified against the template the generator loads.
    """
    mapping: dict[str, tuple[str, Path]] = {}
    duplicates = 0
    for path in sorted(task_instances_root.rglob("*_task_*_0_0_template.json")):
        scene = path.parent.parent.name
        prefix, suffix = f"{scene}_task_", "_0_0_template.json"
        if not path.name.startswith(prefix) or not path.name.endswith(suffix):
            continue
        task = path.name[len(prefix) : -len(suffix)]
        if task in mapping:
            duplicates += 1
            continue
        mapping[task] = (scene, path)
    return mapping, duplicates


def load_template(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        template = json.load(f)
    objects = template["objects_info"]["init_info"]
    categories = {name: entry.get("args", {}).get("category", "unknown") for name, entry in objects.items()}
    rooms = {name: ", ".join(entry.get("args", {}).get("in_rooms", [])) for name, entry in objects.items()}
    by_category: dict[str, list[str]] = defaultdict(list)
    for name, category in categories.items():
        by_category[category].append(name)
    registry = template.get("state", {}).get("registry", {}).get("object_registry", {})
    positions = {
        name: [float(value) for value in entry["root_link"]["pos"]]
        for name, entry in registry.items()
        if isinstance(entry, dict) and isinstance(entry.get("root_link"), dict)
    }
    robot_poses = template.get("metadata", {}).get("task", {}).get("robot_poses", {})
    task_scope = {
        name for name in template.get("metadata", {}).get("task", {}).get("inst_to_name", {}).values()
        if name in objects
    }
    return {
        "task_scope": task_scope,
        "instances": set(objects),
        "categories": categories,
        "rooms": rooms,
        "by_category": by_category,
        "positions": positions,
        "robot_poses": robot_poses,
    }


def robot_initial_pose(robot_poses: dict[str, Any], preferred_key: str) -> tuple[str, list[float]] | None:
    """The generator's --robot-pose-key pose, falling back to the generic 'robot' pose."""
    for key in (preferred_key, "robot"):
        poses = robot_poses.get(key)
        if poses:
            return key, [float(value) for value in poses[0]["position"]]
    return None


def demo_task_index(demo_root: Path) -> dict[str, int]:
    index: dict[str, int] = {}
    with open(demo_root / "meta" / "tasks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index[row["task_name"]] = int(row["task_index"])
    return index


def demo_move_to_chains(demo_root: Path, task_idx: int) -> list[tuple[str, list[str]]]:
    """Per demo episode, the object_id of each ``move to`` navigation skill, in order."""
    chains = []
    directory = demo_root / "annotations" / f"task-{task_idx:04d}"
    if not directory.is_dir():
        return chains
    for path in sorted(directory.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            annotation = json.load(f)
        skills = annotation.get("skill_annotation", [])
        references = []
        for skill in sorted(skills, key=lambda item: item.get("skill_idx", 0)):
            description = join_strings(skill.get("skill_description", []))
            skill_type = join_strings(skill.get("skill_type", []))
            if "move to" not in description and "navigation" not in skill_type:
                continue
            names = flatten_object_ids(skill.get("object_id", []))
            if names:
                # "move to" cites the object driven to first; later entries are context.
                references.append(names[0].strip())
        chains.append((path.stem, references))
    return chains


class LegVerifier:
    """Replicates the generator's approach-pose choice, but only over REACHABLE poses."""

    def __init__(self, scene_map: SceneMap, args: argparse.Namespace, min_distance: float) -> None:
        self.map = scene_map
        self.radii = args.radii
        self.angles = args.angles_per_radius
        self.min_distance = min_distance
        self.max_distance = args.max_distance
        self.cache: dict[tuple[tuple[int, int], str], dict[str, Any]] = {}

    def verify(self, start_cell: tuple[int, int], goal_name: str, goal_xy: list[float]) -> dict[str, Any]:
        key = (start_cell, goal_name)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self._verify(start_cell, goal_xy)
        self.cache[key] = result
        return result

    def _verify(self, start_cell, goal_xy):
        component = self.map.component(start_cell)
        traversable_radius = None
        free_radius = None
        candidates = []
        for radius in self.radii:
            count = 1 if radius == 0 else self.angles
            for index in range(count):
                angle = 0.0 if radius == 0 else 2.0 * math.pi * index / self.angles
                x = goal_xy[0] + radius * math.cos(angle)
                y = goal_xy[1] + radius * math.sin(angle)
                cell = self.map.world_to_cell(x, y)
                if self.map.is_traversable(cell) and traversable_radius is None:
                    traversable_radius = radius
                if not self.map.is_free(cell):
                    continue
                if free_radius is None:
                    free_radius = radius
                if self.map.component(cell) == component:
                    candidates.append((radius, cell))
        if not candidates:
            if free_radius is not None:
                return {"ok": False, "reason": "no_reachable_approach_pose"}
            reason = "no_clearance_at_approach_poses" if traversable_radius is not None else "no_traversable_approach_pose"
            return {"ok": False, "reason": reason}
        distances = self.map.geodesic_distances(start_cell, [cell for _, cell in candidates], self.max_distance)
        scored = [
            (radius, distances[cell], cell)
            for radius, cell in candidates
            if np.isfinite(distances.get(cell, np.inf))
        ]
        in_range = [item for item in scored if self.min_distance <= item[1] <= self.max_distance]
        if not in_range:
            if not scored:
                # Every candidate is in the start's component, so a missing or infinite cost means
                # only that no path within --max-distance reaches it: the search window is cropped
                # to that distance and a shorter path cannot leave it.
                return {"ok": False, "reason": "longer_than_max_distance"}
            shortest = min(item[1] for item in scored)
            reason = "shorter_than_min_distance" if shortest < self.min_distance else "longer_than_max_distance"
            return {"ok": False, "reason": reason, "nearest_distance": shortest}
        radius, distance, cell = min(in_range, key=lambda item: (item[0], item[1]))
        x, y = self.map.cell_to_world(*cell)
        return {
            "ok": True,
            "approach_pose": [x, y],
            "approach_cell": list(cell),
            "approach_radius": radius,
            "geodesic_distance": distance,
            # The generator takes the smallest projection ring holding a usable pose. When the
            # closest ring that clears the robot footprint is cut off from the start by inflation,
            # we fall through to a farther ring that is actually reachable instead of dropping the
            # leg; repicked_ring records that this happened.
            "repicked_ring": free_radius is not None and radius > free_radius,
            "closest_clearance_radius": free_radius,
            "closest_traversable_radius": traversable_radius,
        }


def build_recipe_chain(references: list[str], template: dict[str, Any], verifier: LegVerifier,
                       start_cell: tuple[int, int], task_scope_disambiguation: bool = False) -> dict[str, Any]:
    """Resolve, verify and chain one demo episode's move-to references.

    An ambiguous reference (a category with several instances in the scene) is skipped, not
    guessed. With ``task_scope_disambiguation`` one narrowing is allowed, because it comes from
    the task definition rather than from us: if exactly one of those instances is in the task's
    own BDDL object scope, that instance is the reference. Every leg records which rule matched.
    """
    legs: list[dict[str, str]] = []
    verified: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    previous = "robot_initial"
    cell = start_cell
    for reference in references:
        status, matches, _ = resolve_reference(reference, template["instances"], template["by_category"])
        if status == "unresolved":
            dropped.append({"reference": reference, "reason": "unresolved_reference"})
            continue
        resolution = "template_instance" if matches[:1] == [reference] else "unique_category"
        if status == "ambiguous":
            scoped = sorted(set(matches) & template["task_scope"]) if task_scope_disambiguation else []
            if len(scoped) != 1:
                dropped.append({"reference": reference, "reason": "ambiguous_reference",
                                "candidates": ", ".join(matches)})
                continue
            matches, resolution = scoped, "bddl_task_scope"
        goal = matches[0]
        if template["categories"].get(goal) in STRUCTURAL_CATEGORIES:
            dropped.append({"reference": reference, "reason": "structural_goal"})
            continue
        if goal == previous:
            dropped.append({"reference": reference, "reason": "repeat_of_previous_goal"})
            continue
        position = template["positions"].get(goal)
        if position is None:
            dropped.append({"reference": reference, "reason": "goal_object_missing_pose"})
            continue
        outcome = verifier.verify(cell, goal, position)
        if not outcome["ok"]:
            dropped.append({"reference": reference, "reason": outcome["reason"]})
            continue
        legs.append({"start": previous, "end": goal})
        verified.append({
            "start": previous,
            "end": goal,
            "start_pose": list(verifier.map.cell_to_world(*cell)),
            "start_cell": list(cell),
            "resolution": resolution,
            "goal_category": template["categories"].get(goal, "unknown"),
            "goal_rooms": template["rooms"].get(goal, ""),
            "goal_object_position": position,
            "approach_pose": outcome["approach_pose"],
            "approach_cell": outcome["approach_cell"],
            "approach_radius": outcome["approach_radius"],
            "geodesic_distance": outcome["geodesic_distance"],
            "repicked_ring": outcome["repicked_ring"],
            "closest_clearance_radius": outcome["closest_clearance_radius"],
            "closest_traversable_radius": outcome["closest_traversable_radius"],
        })
        previous = goal
        cell = tuple(outcome["approach_cell"])
    return {"legs": legs, "verified": verified, "dropped": dropped}


def derive_task(task: str, scene: str, template_path: Path, args: argparse.Namespace,
                scene_map: SceneMap, demo_index: dict[str, int]) -> dict[str, Any]:
    template = load_template(template_path)
    report: dict[str, Any] = {
        "task": task,
        "scene": scene,
        "template": str(template_path),
        "recipes": [],
        "drop_reasons": Counter(),
        "notes": [],
    }
    pose = robot_initial_pose(template["robot_poses"], args.robot_pose_key)
    if pose is None:
        report["notes"].append("template has no robot pose")
        return report
    report["robot_pose_key"], initial_position = pose
    start = scene_map.world_to_cell(initial_position[0], initial_position[1])
    snapped = scene_map.snap_to_free(start, args.start_snap)
    if snapped is None:
        report["notes"].append(
            f"robot start pose {initial_position[:2]} has no plannable cell within {args.start_snap} m "
            "after robot+clearance erosion"
        )
        return report
    start_cell, snap_distance = snapped
    report["start_snap_m"] = snap_distance

    task_idx = demo_index.get(task)
    if task_idx is None:
        report["notes"].append("task is not in the demo corpus")
        return report
    report["demo_task_index"] = task_idx

    chains = demo_move_to_chains(args.demo_root, task_idx)
    report["demo_episodes"] = len(chains)
    raw: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for episode, references in chains:
        if references:
            raw[tuple(references)].append(episode)
    report["distinct_demo_chains"] = len(raw)
    if not raw:
        report["notes"].append("no 'move to' navigation skills in this task's demonstrations")
        return report

    ordered = sorted(raw.items(), key=lambda item: (-len(item[1]), -len(item[0]), item[0]))
    # Every demonstrated chain is tried at the primary floor first; a task still short of
    # --recipes-per-task is then topped up at the fallback floor from the chains the primary floor
    # rejected. Each recipe records the floor it was derived at and the summary reports each tier's
    # travel distribution separately, so a short recipe is never passed off as a long one.
    floors = [args.min_distance]
    if args.fallback_min_distance and args.fallback_min_distance < args.min_distance:
        floors.append(args.fallback_min_distance)
    seen: set[tuple[str, ...]] = set()
    used_chains: set[tuple[str, ...]] = set()
    # Drops are kept per demonstrated chain and overwritten when that chain is retried at the
    # fallback floor, so a retried chain is counted once, at the floor that decided it.
    drops_by_chain: dict[tuple[str, ...], list[str]] = {}
    for floor in floors:
        if len(report["recipes"]) >= args.recipes_per_task:
            break
        before = len(report["recipes"])
        verifier = LegVerifier(scene_map, args, floor)
        for references, episodes in ordered:
            if len(report["recipes"]) >= args.recipes_per_task:
                break
            if references in used_chains:
                continue
            built = build_recipe_chain(list(references), template, verifier, start_cell,
                                       args.task_scope_disambiguation)
            drops_by_chain[references] = [drop["reason"] for drop in built["dropped"]]
            if not built["legs"]:
                continue
            legs = built["legs"][: args.max_legs]
            verified = built["verified"][: args.max_legs]
            signature = tuple(leg["end"] for leg in legs)
            if signature in seen:
                continue
            seen.add(signature)
            used_chains.add(references)
            report["recipes"].append({
                "legs": legs,
                "verified": verified,
                "dropped": built["dropped"],
                "truncated": len(built["legs"]) > args.max_legs,
                "demo_episodes": episodes,
                "demo_episode_count": len(episodes),
                "demo_references": list(references),
                "min_distance": floor,
            })
        added = len(report["recipes"]) - before
        if added and floor < args.min_distance:
            report["notes"].append(
                f"{added} recipe(s) needed the {floor} m fallback distance floor; their demonstrated "
                f"chains hold no leg that drives {args.min_distance} m"
            )
    for reasons in drops_by_chain.values():
        report["drop_reasons"].update(reasons)
    if len(report["recipes"]) < args.recipes_per_task:
        shortfall = (
            f"only {len(report['recipes'])} distinct verified chains from {len(raw)} distinct demonstrated "
            f"chains over {len(chains)} episodes"
        )
        if len(raw) < args.recipes_per_task:
            shortfall += (
                f"; the demonstrations themselves hold only {len(raw)} distinct move-to chains, fewer than "
                f"the {args.recipes_per_task} requested"
            )
        elif report["drop_reasons"]:
            reason, count = report["drop_reasons"].most_common(1)[0]
            shortfall += f"; dominant leg-drop reason {reason} ({count})"
        report["notes"].append(shortfall)
        report["shortfall"] = shortfall
    return report


def write_recipes(report: dict[str, Any], args: argparse.Namespace, scene_map: SceneMap) -> list[Path]:
    directory = args.output_root / report["scene"]
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, recipe in enumerate(report["recipes"], start=1):
        path = directory / f"{report['task']}_{number}.json"
        document = {
            "scene": report["scene"],
            "task": report["task"],
            "navigation_chain": recipe["legs"],
            "state_overrides": [],
            "derivation": {
                "source": "behavior1k-20k demonstration annotations; skill_type navigation / 'move to'",
                "demo_task_index": report["demo_task_index"],
                "demo_episodes_with_this_chain": recipe["demo_episode_count"],
                "demo_episode_examples": recipe["demo_episodes"][:5],
                "demo_references": recipe["demo_references"],
                "template": report["template"],
                "robot_pose_key": report["robot_pose_key"],
                "dropped_legs": recipe["dropped"],
                "chain_truncated_to_max_legs": recipe["truncated"],
                "distance_floor_m": recipe["min_distance"],
            },
            "reachability_check": {
                "map": str(scene_map.directory),
                "robot_bounding_radius_m": args.robot_radius,
                "omnigibson_robot_erosion_m": scene_map.robot_erosion_m,
                "runtime_extra_clearance_m": args.extra_clearance,
                "robot_start_snap_m": report["start_snap_m"],
                "success_distance_m": args.success_distance,
                "legs": recipe["verified"],
            },
            "generator_invocation": {
                "script": "generate_object_nav_benchmark_from_recipe.py",
                "--scene": report["scene"],
                "--num-instances": 1,
                "--min-distance": recipe["min_distance"],
                "--max-distance": args.max_distance,
                "--approach-radii": args.approach_radii,
                "--angles-per-radius": args.angles_per_radius,
                "--extra-clearance": args.extra_clearance,
                "--robot-pose-key": report["robot_pose_key"],
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)
            f.write("\n")
        paths.append(path)
    return paths


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": at(0.10),
        "p25": at(0.25),
        "median": statistics.median(ordered),
        "p75": at(0.75),
        "p90": at(0.90),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-instances-root", default=DEFAULT_TASK_INSTANCES_ROOT)
    parser.add_argument("--demo-root", default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--map-root", default=DEFAULT_MAP_ROOT,
                        help="Root holding <scene>/navigation_2d ground-truth maps.")
    parser.add_argument("--nav2py-root", default=DEFAULT_NAV2PY_ROOT,
                        help="Repository providing benchmarks.costmaps.load_navigation_2d.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", default=None, help="Write the full derivation report here as JSON.")
    parser.add_argument("--task", action="append", default=None, help="Restrict to these tasks (repeatable).")
    parser.add_argument("--recipes-per-task", type=int, default=5)
    parser.add_argument("--max-legs", type=int, default=6)
    parser.add_argument("--min-distance", type=float, default=2.0,
                        help="Geodesic distance floor for a leg. The default is deliberately above the "
                             "generator's own 1.0 m so that no episode is trivial next to the 0.5 m "
                             "success radius; it is passed through to the generator in each recipe.")
    parser.add_argument("--fallback-min-distance", type=float, default=1.0,
                        help="Floor retried for a task that yields nothing at --min-distance, so a task "
                             "whose demonstrations only ever moved locally is still covered. Those "
                             "recipes are labelled with the floor they needed. 0 disables the fallback.")
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--approach-radii", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--angles-per-radius", type=int, default=36)
    parser.add_argument("--extra-clearance", type=float, default=0.2)
    parser.add_argument("--robot-radius", type=float, default=R1PRO_BOUNDING_RADIUS)
    parser.add_argument("--og-map-resolution", type=float, default=OG_MAP_RESOLUTION)
    parser.add_argument("--robot-pose-key", default="R1Pro")
    parser.add_argument("--task-scope-disambiguation", action="store_true",
                        help="Allow an ambiguous category reference to be narrowed when exactly one "
                             "instance of that category is in the task's own BDDL object scope.")
    parser.add_argument("--start-snap", type=float, default=0.3,
                        help="Largest distance the robot's own start pose may be moved onto plannable free space.")
    parser.add_argument("--success-distance", type=float, default=0.5,
                        help="Runner success radius; net travel is reported as geodesic distance minus this.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.nav2py_root).expanduser()))
    args.task_instances_root = Path(args.task_instances_root).expanduser()
    args.demo_root = Path(args.demo_root).expanduser()
    args.output_root = Path(args.output_root).expanduser()
    args.radii = parse_radii(args.approach_radii)

    task_scene, duplicate_templates = scene_of_each_task(args.task_instances_root)
    demo_index = demo_task_index(args.demo_root)
    demo_tasks = sorted(demo_index)
    without_template = [task for task in demo_tasks if task not in task_scene]
    selected = [task for task in demo_tasks if task in task_scene]
    if args.task:
        selected = [task for task in selected if task in set(args.task)]

    print(f"Demo tasks: {len(demo_tasks)}; with a challenge-scene template: {len(selected)}; "
          f"without: {len(without_template)} {without_template}", flush=True)
    if duplicate_templates:
        print(f"  {duplicate_templates} further copies of those templates exist under "
              f"{args.task_instances_root}; the first sorted path wins, as in the generator", flush=True)

    maps: dict[str, SceneMap] = {}
    reports = []
    for position, task in enumerate(selected, start=1):
        scene, template_path = task_scene[task]
        if scene not in maps:
            maps[scene] = SceneMap(args.map_root, scene, args.robot_radius, args.extra_clearance,
                                   args.og_map_resolution)
            scene_map = maps[scene]
            print(f"  map {scene}: {int(scene_map.traversable.sum())} traversable cells -> "
                  f"{int(scene_map.free.sum())} plannable after {scene_map.robot_erosion_m:.2f} m robot erosion "
                  f"+ {args.extra_clearance} m clearance, {scene_map.num_components} components", flush=True)
        scene_map = maps[scene]
        report = derive_task(task, scene, template_path, args, scene_map, demo_index)
        report["files"] = [str(path) for path in write_recipes(report, args, scene_map)]
        reports.append(report)
        print(f"[{position:3d}/{len(selected)}] {scene}/{task}: {len(report['recipes'])} recipes, "
              f"{sum(len(recipe['legs']) for recipe in report['recipes'])} legs"
              + (f" | {'; '.join(report['notes'])}" if report["notes"] else ""), flush=True)

    summarize(reports, without_template, args)
    if args.report:
        for report in reports:
            report["drop_reasons"] = dict(report["drop_reasons"])
        with open(Path(args.report).expanduser(), "w", encoding="utf-8") as f:
            json.dump({"args": {key: str(value) for key, value in vars(args).items()}, "tasks": reports}, f, indent=2)
            f.write("\n")


def summarize(reports: list[dict[str, Any]], without_template: list[str], args: argparse.Namespace) -> None:
    recipes = [(report, recipe) for report in reports for recipe in report["recipes"]]
    legs = [(report, leg) for report, recipe in recipes for leg in recipe["verified"]]
    travel = [leg["geodesic_distance"] - args.success_distance for _, leg in legs]
    object_pairs = {(report["scene"], report["task"], leg["start"], leg["end"]) for report, leg in legs}
    geometric_pairs = {
        (report["scene"], tuple(leg["start_cell"]), tuple(leg["approach_cell"]))
        for report, leg in legs
    }
    counts = Counter(len(report["recipes"]) for report in reports)
    drops: Counter = Counter()
    for report in reports:
        drops.update(report["drop_reasons"])
    covered = [report for report in reports if report["recipes"]]

    print("\n=== coverage ===")
    print(f"demo tasks without a challenge-scene template: {len(without_template)}")
    print(f"tasks with >=1 recipe: {len(covered)}/{len(reports)}")
    print(f"recipes: {len(recipes)}; legs (= episodes at --num-instances 1): {len(legs)}")
    print("recipes per task: " + ", ".join(f"{number}:{count}" for number, count in sorted(counts.items())))
    print(f"distinct (start, goal) object pairs: {len(object_pairs)}")
    print(f"distinct (start pose, goal pose) problems: {len(geometric_pairs)}")
    resolutions = Counter(leg["resolution"] for _, leg in legs)
    print("leg goal resolution: " + ", ".join(f"{key}:{value}" for key, value in sorted(resolutions.items())))
    print(f"repicked approach rings (closest ring unusable after inflation): "
          f"{sum(1 for _, leg in legs if leg['repicked_ring'])}/{len(legs)}")
    print("\n=== dropped legs ===")
    for reason in DROP_REASONS:
        print(f"  {reason:32s} {drops.get(reason, 0)}")
    if travel:
        print("\n=== required net travel per leg (geodesic - success radius), metres ===")
        for key, value in percentiles(travel).items():
            print(f"  {key:8s} {value:.2f}" if key != "count" else f"  {key:8s} {value}")
        for floor in sorted({recipe["min_distance"] for _, recipe in recipes}):
            tier = [
                leg["geodesic_distance"] - args.success_distance
                for _, recipe in recipes for leg in recipe["verified"]
                if recipe["min_distance"] == floor
            ]
            tier_tasks = {report["task"] for report, recipe in recipes if recipe["min_distance"] == floor}
            print(f"  -- legs derived at the {floor} m distance floor ({len(tier_tasks)} tasks): "
                  + ", ".join(f"{key} {value:.2f}" if key != "count" else f"{key} {value}"
                              for key, value in percentiles(tier).items()))
        buckets = Counter(min(int(value), 9) for value in travel)
        print("  histogram (net metres): " + ", ".join(
            f"[{index},{index + 1}):{buckets[index]}" if index < 9 else f"[9,inf):{buckets[9]}"
            for index in range(10) if buckets[index]))


if __name__ == "__main__":
    main()
