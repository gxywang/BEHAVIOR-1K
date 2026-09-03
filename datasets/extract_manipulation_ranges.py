#!/usr/bin/env python3
"""Extract per-episode manipulation intervals from BEHAVIOR-1K annotations.

The output is JSONL so it can be streamed and indexed without loading the
20,000-episode dataset into memory. Each line contains one episode and a list
of manipulation skill intervals.

Example:
    python datasets/extract_manipulation_ranges.py \
        --dataset-root /shared/perception/datasets/behavior1k-20k \
        --output datasets/behavior1k_manipulation_ranges.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = Path("/shared/perception/datasets/behavior1k-20k")
DEFAULT_FPS = 30.0
MANIPULATION_SKILL_TYPES = frozenset(("coordinated", "uncoordinated"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract manipulation skill time ranges from BEHAVIOR-1K annotation JSON files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"LeRobot dataset root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path; use '-' to write to stdout.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate used for seconds fields (default: read meta/info.json, then 30).",
    )
    parser.add_argument(
        "--include-skill-types",
        nargs="+",
        default=sorted(MANIPULATION_SKILL_TYPES),
        help="Skill types to retain (default: coordinated uncoordinated).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if an annotation file is missing or malformed; otherwise warn and continue.",
    )
    return parser.parse_args()


def read_fps(dataset_root: Path, override: float | None) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--fps must be positive")
        return override

    info_path = dataset_root / "meta" / "info.json"
    try:
        fps = float(json.loads(info_path.read_text())["fps"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        fps = DEFAULT_FPS
    if fps <= 0:
        raise ValueError(f"Invalid FPS in {info_path}: {fps}")
    return fps


def episode_rows(dataset_root: Path) -> Iterator[dict[str, Any]]:
    episode_dir = dataset_root / "meta" / "episodes"
    for shard_path in sorted(episode_dir.glob("chunk-*/file-*.parquet")):
        table = pq.read_table(shard_path)
        for row in table.to_pylist():
            yield row


def annotation_path(dataset_root: Path, row: dict[str, Any]) -> Path:
    relative_path = row.get("annotation_path")
    if not relative_path:
        raise ValueError("episode metadata has no annotation_path")
    return dataset_root / str(relative_path)


def extract_ranges(
    annotation: dict[str, Any],
    fps: float,
    included_skill_types: set[str],
) -> list[dict[str, Any]]:
    ranges = []
    for skill in annotation.get("skill_annotation", []):
        skill_types = [str(value) for value in skill.get("skill_type", [])]
        if not included_skill_types.intersection(skill_types):
            continue

        frame_duration = skill.get("frame_duration")
        if not isinstance(frame_duration, list) or len(frame_duration) != 2:
            raise ValueError(f"invalid frame_duration: {frame_duration!r}")
        start_frame, end_frame = (int(frame_duration[0]), int(frame_duration[1]))
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(f"invalid frame range: {frame_duration!r}")

        # Annotation frame_duration is inclusive, matching frame indices in
        # the LeRobot data. Keep that convention explicit in the output.
        ranges.append(
            {
                "skill_idx": int(skill["skill_idx"]),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": start_frame / fps,
                "end_time_s": (end_frame + 1) / fps,
                "skill_id": skill.get("skill_id", []),
                "skill_description": skill.get("skill_description", []),
                "skill_type": skill_types,
                "object_id": skill.get("object_id", []),
                "manipulating_object_id": skill.get("manipulating_object_id", []),
            }
        )
    return ranges


def output_record(
    row: dict[str, Any], annotation: dict[str, Any], ranges: list[dict[str, Any]], fps: float
) -> dict[str, Any]:
    tasks = row.get("tasks", [])
    return {
        "episode_index": int(row["episode_index"]),
        "raw_episode_id": int(row["raw_episode_id"]),
        "task_index": int(row["task_index"]),
        "task": tasks[0] if tasks else None,
        "num_frames": int(row["length"]),
        "fps": fps,
        "annotation_path": row.get("annotation_path"),
        "task_name": annotation.get("task_name"),
        "manipulation_ranges": ranges,
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    fps = read_fps(dataset_root, args.fps)
    included_skill_types = set(args.include_skill_types)
    if not included_skill_types:
        raise ValueError("at least one skill type must be included")

    output = None if str(args.output) == "-" else args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output.open("w")
    else:
        output_handle = sys.stdout

    total = 0
    written = 0
    missing = 0
    try:
        for row in episode_rows(dataset_root):
            total += 1
            path = annotation_path(dataset_root, row)
            try:
                annotation = json.loads(path.read_text())
                ranges = extract_ranges(annotation, fps, included_skill_types)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                if args.strict:
                    raise RuntimeError(f"Could not process episode {row.get('episode_index')}: {path}") from error
                missing += 1
                print(f"warning: skipping {path}: {error}", file=sys.stderr)
                continue

            output_handle.write(json.dumps(output_record(row, annotation, ranges, fps)) + "\n")
            written += 1
    finally:
        if output is not None:
            output_handle.close()

    print(f"wrote {written}/{total} episodes to {args.output} ({missing} skipped)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
