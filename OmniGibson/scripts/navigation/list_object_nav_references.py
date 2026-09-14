#!/usr/bin/env python3
"""List object references available for an object-navigation task recipe."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def find_template(root: Path, scene: str, task: str) -> Path:
    name = f"{scene}_task_{task}_0_0_template.json"
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"No task template named {name} under {root}")
    return matches[0]


def flatten_object_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    names = []
    for item in value:
        names.extend(flatten_object_ids(item))
    return names


def task_index(demo_root: Path, task: str) -> int | None:
    path = demo_root / "meta" / "tasks.jsonl"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("task_name") == task:
                return int(row["task_index"])
    return None


def task_description(demo_root: Path, task: str) -> str | None:
    path = demo_root / "meta" / "tasks.jsonl"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("task_name") == task:
                return row.get("task")
    return None


def bddl_goal(bddl_root: Path, task: str) -> str | None:
    path = bddl_root / "bddl" / "activity_definitions" / task / "problem0.bddl"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    start = text.find("(:goal")
    if start < 0:
        return None
    depth = 0
    for index, char in enumerate(text[start:], start):
        depth += char == "("
        depth -= char == ")"
        if depth == 0:
            return text[start : index + 1]
    return None


def translated_goal_requirements(goal: str, task_mapping: dict[str, str], categories: dict[str, str], rooms: dict[str, str]):
    variable_types = {}
    for variable, object_type in re.findall(r"\?([^\s]+)\s*-\s*([^\s\)]+)", goal):
        variable_types[variable] = object_type

    def format_reference(reference: str) -> str:
        if reference.startswith("?"):
            concrete_name = reference[1:]
            if concrete_name in task_mapping:
                return format_reference(concrete_name)
            names = [
                scene_name
                for bddl_name, scene_name in task_mapping.items()
                if bddl_name.rsplit("_", 1)[0] == variable_types.get(reference[1:])
            ]
            return "all {" + ", ".join(sorted(names)) + "}" if names else reference
        scene_name = task_mapping.get(reference, reference)
        if scene_name not in categories:
            return scene_name
        return f"{scene_name} [{categories[scene_name]}; {rooms.get(scene_name, '')}]"

    requirements = []
    for predicate, arguments in re.findall(r"\(([A-Za-z_][^\s\(\)]*)\s+([^\(\)]+)\)", goal):
        if predicate in {"and", "or", "not"}:
            continue
        requirements.append(f"{predicate}({', '.join(format_reference(arg) for arg in arguments.split())})")
    return requirements


def annotation_references(demo_root: Path, task: str) -> dict[str, set[str]]:
    index = task_index(demo_root, task)
    if index is None:
        return {}
    references: dict[str, set[str]] = defaultdict(set)
    for path in sorted((demo_root / "annotations" / f"task-{index:04d}").glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            annotation = json.load(f)
        for skill in annotation.get("skill_annotation", []):
            description = ", ".join(skill.get("skill_description", []))
            skill_type = ", ".join(skill.get("skill_type", []))
            for name in flatten_object_ids(skill.get("object_id", [])):
                references[name].add(f"{skill_type}: {description}")
    return references


def print_section(title: str, rows: list[tuple[str, str, str]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (none)")
        return
    width = max(len(row[0]) for row in rows)
    for name, category, detail in rows:
        print(f"  {name:<{width}}  {category:<16}  {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-instances-root", default="datasets/2026-challenge-task-instances")
    parser.add_argument("--demo-root", default="datasets", help="Demo dataset root; annotations are optional.")
    parser.add_argument(
        "--bddl-root",
        default=str(Path(__file__).resolve().parents[3] / "bddl3"),
        help="Path to the bddl3 repository directory.",
    )
    parser.add_argument("--scene", default="house_double_floor_lower")
    parser.add_argument("--task", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_path = find_template(Path(args.task_instances_root).expanduser(), args.scene, args.task)
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    objects = template["objects_info"]["init_info"]
    categories = {name: entry.get("args", {}).get("category", "unknown") for name, entry in objects.items()}
    rooms = {name: ", ".join(entry.get("args", {}).get("in_rooms", [])) for name, entry in objects.items()}
    task_mapping = template.get("metadata", {}).get("task", {}).get("inst_to_name", {})
    demo_root = Path(args.demo_root).expanduser()
    referenced = annotation_references(demo_root, args.task)
    description = task_description(demo_root, args.task)
    goal = bddl_goal(Path(args.bddl_root).expanduser(), args.task)

    print(f"Task: {args.task}")
    print(f"Scene: {args.scene}")
    print(f"Template: {template_path}")
    if description:
        print(f"Task instruction: {description}")
    if goal:
        print_section(
            "Goal requirements (translated from BDDL)",
            [(requirement, "", "") for requirement in translated_goal_requirements(goal, task_mapping, categories, rooms)],
        )
    print_section(
        "Task objects (use the scene-object name in recipes)",
        [
            (scene_name, categories.get(scene_name, "unknown"), f"BDDL: {bddl_name}; rooms: {rooms.get(scene_name, '')}")
            for bddl_name, scene_name in sorted(task_mapping.items())
            if scene_name != "robot"
        ],
    )
    print_section(
        "Objects referenced by demonstration annotations",
        [
            (name, categories.get(name, "not in template"), "; ".join(sorted(skills)))
            for name, skills in sorted(referenced.items())
        ],
    )
    print_section(
        "Doors in this task instance",
        [
            (name, category, f"rooms: {rooms.get(name, '')}")
            for name, category in sorted(categories.items())
            if category == "door"
        ],
    )


if __name__ == "__main__":
    main()
