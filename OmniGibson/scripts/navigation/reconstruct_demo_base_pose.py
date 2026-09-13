#!/usr/bin/env python3
"""Reconstruct an R1Pro base trajectory for one LeRobot demo episode.

This is a diagnostic helper for demo-derived navigation benchmark generation. It
uses a task-instance TRO file for the initial R1Pro pose, then integrates the
demo's robot-local base velocity from the LeRobot parquet frames. If a matching
raw HDF5 file is provided, it also compares the reconstructed trajectory against
the exact simulator base qpos stored in raw replay state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROBOT_TYPE = "R1Pro"
BASE_QPOS_INDICES = np.asarray([0, 1, 5], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-root", default="datasets/demonstrations", help="LeRobot demo dataset root.")
    parser.add_argument(
        "--task-instances-root",
        default="datasets/2026-challenge-task-instances",
        help="2026 challenge task-instances dataset root used to locate TRO state files.",
    )
    parser.add_argument("--tro-state", default=None, help="Explicit *-tro_state.json path. Overrides auto lookup.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode-index", type=int, help="LeRobot episode_index to reconstruct.")
    group.add_argument("--raw-episode-id", type=int, help="Raw episode id, e.g. 950 for episode_00000950.")
    parser.add_argument(
        "--velocity-source",
        choices=("state", "action"),
        default="state",
        help="Use observation.state[:3] measured base_qvel or action[:3] commanded base velocity.",
    )
    parser.add_argument(
        "--raw-hdf5",
        default=None,
        help="Optional exact raw HDF5 path for comparison, e.g. task-0000/episode_00000950.hdf5.",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Optional rawdata root. Used if --raw-hdf5 is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/navigation/demo_pose_reconstruction",
        help="Directory for CSV trajectory and JSON summary.",
    )
    return parser.parse_args()


def normalize_task_name(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected one task name, got {value}")
        return str(value[0])
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text.strip("[]").strip("'\"")
    return text


def quat_xyzw_to_yaw(quat: list[float]) -> float:
    x, y, z, w = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat_xyzw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]


def quat_xyzw_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def euler_xyz_to_quat_xyzw(euler: np.ndarray) -> np.ndarray:
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]
    cr = np.cos(0.5 * roll)
    sr = np.sin(0.5 * roll)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    return np.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        axis=-1,
    )


def quat_xyzw_array_to_yaw(quat: np.ndarray) -> np.ndarray:
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def load_episode_metadata(demo_root: Path, episode_index: int | None, raw_episode_id: int | None) -> pd.Series:
    rows = []
    for path in sorted((demo_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(path)
        if episode_index is not None:
            df = df[df["episode_index"] == episode_index]
        else:
            df = df[df["raw_episode_id"] == raw_episode_id]
        if not df.empty:
            rows.append(df)

    if not rows:
        ident = f"episode_index={episode_index}" if episode_index is not None else f"raw_episode_id={raw_episode_id}"
        raise FileNotFoundError(f"No episode metadata found for {ident}")

    matches = pd.concat(rows, ignore_index=True)
    if len(matches) != 1:
        raise ValueError(f"Expected one episode metadata row, got {len(matches)}")
    return matches.iloc[0]


def resolve_data_path(demo_root: Path, row: pd.Series) -> Path:
    chunk_index = int(row["data/chunk_index"])
    file_index = int(row["data/file_index"])
    candidates = [
        demo_root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet",
        demo_root / "data" / f"file-{file_index:03d}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No data parquet found. Tried: {candidates}")


def load_episode_frames(demo_root: Path, row: pd.Series) -> pd.DataFrame:
    path = resolve_data_path(demo_root, row)
    df = pd.read_parquet(
        path,
        columns=["episode_index", "frame_index", "timestamp", "action", "observation.state"],
    )
    df = df[df["episode_index"] == int(row["episode_index"])].copy()
    if df.empty:
        raise ValueError(f"No frame rows found for episode_index={int(row['episode_index'])} in {path}")
    df.sort_values("frame_index", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_annotation(demo_root: Path, row: pd.Series) -> dict[str, Any]:
    path = demo_root / str(row["annotation_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def navigation_skills(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    skills = []
    for skill in annotation.get("skill_annotation", []):
        skill_type = skill.get("skill_type", [])
        if isinstance(skill_type, str):
            is_navigation = skill_type == "navigation"
        else:
            is_navigation = "navigation" in skill_type
        if is_navigation:
            skills.append(skill)
    return skills


def find_tro_state(task_instances_root: Path, task_name: str, task_instance_id: int) -> Path:
    pattern = f"scenes/*/json/*_task_{task_name}_instances/*_0_{task_instance_id}_template-tro_state.json"
    matches = sorted(task_instances_root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one TRO match for pattern {pattern}, got {len(matches)}")
    return matches[0]


def load_initial_pose(path: Path) -> tuple[list[float], list[float]]:
    with open(path, "r", encoding="utf-8") as f:
        tro = json.load(f)
    robot_poses = tro.get("robot_poses", {})
    for key in (ROBOT_TYPE, "robot", ROBOT_TYPE.lower()):
        poses = robot_poses.get(key)
        if poses:
            pose = poses[0]
            return [float(v) for v in pose["position"]], [float(v) for v in pose["orientation"]]
    raise KeyError(f"No {ROBOT_TYPE} robot pose found in {path}")


def matrix_from_column(values: pd.Series) -> np.ndarray:
    return np.stack(values.to_numpy()).astype(np.float64, copy=False)


def reconstruct_trajectory(
    frames: pd.DataFrame,
    initial_position: list[float],
    initial_quat: list[float],
    velocity_source: str,
) -> np.ndarray:
    values = matrix_from_column(frames["observation.state" if velocity_source == "state" else "action"])
    local_vel = values[:, :3]
    timestamps = frames["timestamp"].to_numpy(dtype=np.float64)
    frame_indices = frames["frame_index"].to_numpy(dtype=np.int64)
    dt = np.diff(timestamps, prepend=timestamps[0])
    if len(dt) > 1:
        dt[0] = float(np.median(dt[1:]))
    else:
        dt[0] = 1.0 / 30.0

    trajectory = np.zeros((len(frames), 7), dtype=np.float64)
    x, y, z = [float(v) for v in initial_position]
    yaw = quat_xyzw_to_yaw(initial_quat)

    trajectory[0] = [frame_indices[0], timestamps[0], x, y, z, yaw, 0.0]
    for i in range(1, len(frames)):
        vx, vy, wz = local_vel[i - 1]
        prev_yaw = yaw
        x += (math.cos(prev_yaw) * vx - math.sin(prev_yaw) * vy) * dt[i]
        y += (math.sin(prev_yaw) * vx + math.cos(prev_yaw) * vy) * dt[i]
        yaw += wz * dt[i]
        trajectory[i] = [frame_indices[i], timestamps[i], x, y, z, yaw, dt[i]]
    return trajectory


def get_uuid(name: str) -> int:
    return int(np.float32(int(hashlib.md5(name.encode()).hexdigest(), 16) % (10**8)).item())


def find_robot_name(scene: dict[str, Any]) -> str:
    init_info = scene.get("objects_info", {}).get("init_info", {})
    candidates = [
        name
        for name, info in init_info.items()
        if isinstance(info, dict)
        and ("robot" in info.get("class_module", "") or info.get("class_name", "").lower().startswith("r1"))
    ]
    if not candidates:
        candidates = [name for name in scene["state"]["registry"].get("object_registry", {}) if name.startswith("robot")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one robot candidate, got {candidates}")
    return candidates[0]


def find_robot_offsets(states: np.ndarray, uuid: np.float32, n_joints: int) -> np.ndarray:
    rows, cols = np.where(states == uuid)
    if len(rows) == 0:
        raise ValueError(f"Robot uuid not found in raw state frames={states.shape[0]}")

    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    cols = cols[order]
    offsets = np.empty(states.shape[0], dtype=np.int64)
    out_row = 0
    i = 0
    min_width = 15 + 2 * n_joints

    while i < len(rows):
        row = int(rows[i])
        if row != out_row:
            raise ValueError(f"Robot uuid missing at state row {out_row}")
        j = i + 1
        while j < len(rows) and rows[j] == row:
            j += 1

        valid_cols = []
        for col in cols[i:j]:
            col = int(col)
            if col + min_width > states.shape[1]:
                continue
            root_ori = states[row, col + 5 : col + 9]
            if np.all(np.isfinite(root_ori)) and 0.95 <= float(np.linalg.norm(root_ori)) <= 1.05:
                valid_cols.append(col)

        if len(valid_cols) != 1:
            raise ValueError(f"Robot uuid row={row} candidates={cols[i:j].tolist()} valid={valid_cols}")
        offsets[row] = valid_cols[0]
        out_row += 1
        i = j

    if out_row != states.shape[0]:
        raise ValueError(f"Robot uuid missing after row {out_row - 1}; state_frames={states.shape[0]}")
    return offsets


def natural_demo_key(name: str) -> tuple[int, str]:
    if name.startswith("demo_") and name[5:].isdigit():
        return int(name[5:]), name
    return -1, name


def load_raw_global_base_pose(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        scene = json.loads(f["data"].attrs["scene_file"])
        demos = sorted((name for name in f["data"].keys() if name.startswith("demo_")), key=natural_demo_key)
        if not demos:
            raise ValueError("Raw HDF5 has no data/demo_* groups")
        states = f["data"][demos[-1]]["state"][:]

    robot_name = find_robot_name(scene)
    robot_state = scene["state"]["registry"]["object_registry"][robot_name]
    n_joints = len(robot_state["joint_pos"])
    offsets = find_robot_offsets(states, np.float32(get_uuid(robot_name)), n_joints)
    row_idx = np.arange(states.shape[0])[:, None]
    root_pos = states[:, offsets[:, None] + np.asarray([2, 3, 4], dtype=np.int64)].astype(np.float64)
    root_quat = states[:, offsets[:, None] + np.asarray([5, 6, 7, 8], dtype=np.int64)].astype(np.float64)
    base_qpos = states[row_idx, offsets[:, None] + 15 + np.arange(n_joints, dtype=np.int64)].astype(np.float64)

    base_position = root_pos + base_qpos[:, :3]
    base_quat = quat_xyzw_multiply(root_quat, euler_xyz_to_quat_xyzw(base_qpos[:, 3:6]))
    yaw = quat_xyzw_array_to_yaw(base_quat)
    return np.column_stack([base_position[:, 0], base_position[:, 1], yaw])


def resolve_raw_hdf5(args: argparse.Namespace, row: pd.Series) -> Path | None:
    if args.raw_hdf5:
        path = Path(args.raw_hdf5).expanduser()
        return path if path.exists() else None
    if not args.raw_root:
        return None
    path = (
        Path(args.raw_root).expanduser()
        / f"task-{int(row['task_index']):04d}"
        / f"episode_{int(row['raw_episode_id']):08d}.hdf5"
    )
    return path if path.exists() else None


def trajectory_pose_at(trajectory: np.ndarray, frame: int) -> dict[str, Any]:
    matches = np.where(trajectory[:, 0].astype(np.int64) == int(frame))[0]
    if len(matches) == 0:
        raise ValueError(f"Frame {frame} not found in reconstructed trajectory")
    row = trajectory[int(matches[0])]
    yaw = float(row[5])
    return {
        "frame": int(row[0]),
        "timestamp": float(row[1]),
        "position": [float(row[2]), float(row[3]), float(row[4])],
        "yaw": yaw,
        "quat": yaw_to_quat_xyzw(yaw),
    }


def compare_to_raw(trajectory: np.ndarray, raw_qpos: np.ndarray) -> dict[str, Any]:
    n = min(len(trajectory), len(raw_qpos))
    recon = trajectory[:n, 2:6][:, [0, 1, 3]]
    exact = raw_qpos[:n]
    error = recon - exact
    error[:, 2] = wrap_angle(error[:, 2])
    xy_error = np.linalg.norm(error[:, :2], axis=1)
    return {
        "frames_compared": int(n),
        "xy_rmse_m": float(np.sqrt(np.mean(xy_error**2))),
        "xy_mean_m": float(np.mean(xy_error)),
        "xy_max_m": float(np.max(xy_error)),
        "yaw_rmse_rad": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "yaw_mean_abs_rad": float(np.mean(np.abs(error[:, 2]))),
        "yaw_max_abs_rad": float(np.max(np.abs(error[:, 2]))),
        "first_error": error[0].tolist(),
        "last_error": error[n - 1].tolist(),
    }


def write_csv(path: Path, trajectory: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "timestamp", "x", "y", "z", "yaw", "dt"])
        writer.writerows(trajectory.tolist())


def main() -> None:
    args = parse_args()
    demo_root = Path(args.demo_root).expanduser()
    row = load_episode_metadata(demo_root, args.episode_index, args.raw_episode_id)
    task_name = normalize_task_name(row["tasks"])
    frames = load_episode_frames(demo_root, row)
    annotation = load_annotation(demo_root, row)
    skills = navigation_skills(annotation)

    tro_path = Path(args.tro_state).expanduser() if args.tro_state else find_tro_state(
        Path(args.task_instances_root).expanduser(),
        task_name,
        int(row["task_instance_id"]),
    )
    initial_position, initial_quat = load_initial_pose(tro_path)
    trajectory = reconstruct_trajectory(frames, initial_position, initial_quat, args.velocity_source)

    output_dir = Path(args.output_dir).expanduser()
    stem = f"task-{int(row['task_index']):04d}_episode-{int(row['episode_index']):06d}_raw-{int(row['raw_episode_id']):08d}_{args.velocity_source}"
    csv_path = output_dir / f"{stem}_trajectory.csv"
    json_path = output_dir / f"{stem}_summary.json"
    write_csv(csv_path, trajectory)

    nav_skill_summaries = []
    for skill in skills:
        start_frame, goal_frame = [int(v) for v in skill["frame_duration"]]
        nav_skill_summaries.append(
            {
                "skill_idx": int(skill["skill_idx"]),
                "skill_description": skill.get("skill_description"),
                "skill_type": skill.get("skill_type"),
                "object_id": skill.get("object_id"),
                "frame_duration": [start_frame, goal_frame],
                "start_pose": trajectory_pose_at(trajectory, start_frame),
                "goal_pose": trajectory_pose_at(trajectory, goal_frame),
            }
        )

    summary = {
        "episode_index": int(row["episode_index"]),
        "raw_episode_id": int(row["raw_episode_id"]),
        "task_index": int(row["task_index"]),
        "task_name": task_name,
        "task_instance_id": int(row["task_instance_id"]),
        "annotation_path": str(row["annotation_path"]),
        "tro_state_path": str(tro_path),
        "velocity_source": args.velocity_source,
        "pose_source": f"tro_initial_pose_plus_integrated_{args.velocity_source}",
        "frames": int(len(frames)),
        "initial_pose": {"position": initial_position, "quat": initial_quat, "yaw": quat_xyzw_to_yaw(initial_quat)},
        "navigation_skills": nav_skill_summaries,
        "trajectory_csv": str(csv_path),
    }

    raw_hdf5 = resolve_raw_hdf5(args, row)
    if raw_hdf5 is not None:
        raw_qpos = load_raw_global_base_pose(raw_hdf5)
        summary["raw_hdf5_path"] = str(raw_hdf5)
        summary["raw_comparison"] = compare_to_raw(trajectory, raw_qpos)
    elif args.raw_hdf5 or args.raw_root:
        summary["raw_hdf5_path"] = None
        summary["raw_comparison"] = None

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved trajectory CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")
    for skill in nav_skill_summaries:
        print(
            f"Navigation skill {skill['skill_idx']} frames {skill['frame_duration']}: "
            f"start={skill['start_pose']['position']} goal={skill['goal_pose']['position']}"
        )
    if summary.get("raw_comparison"):
        print("Raw HDF5 comparison:")
        print(json.dumps(summary["raw_comparison"], indent=2))


if __name__ == "__main__":
    main()
