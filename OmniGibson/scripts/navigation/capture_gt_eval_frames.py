"""Capture head-camera frames WITH OmniGibson ground truth for detector accuracy scoring.

Unlike ``capture_object_search_frames.py`` (which follows the open-loop search
policy and so mostly photographs empty rooms), this harness deliberately aims the
head camera at the task-scale objects: for every eval target it samples base poses
on the b1k planning costmap in an annulus around the object and faces it. Each
frame stores rgb + depth_linear + seg_instance together with the
per-frame ``info`` dicts, because the instance registry is process-global and is
not stable across runs.

A handful of 360-degree spins at plain traversable cells are also captured so the
benchmark contains frames where the small targets are absent (precision needs
those).

Example:
    python scripts/navigation/capture_gt_eval_frames.py --scene Rs_int \
        --task composting_waste --output-dir /scratch/.../gtbench/frames
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from generate_nav_benchmark import build_env_config, controller_no_op_action, load_robot_config, place_robot, seed_everything
from generate_object_nav_benchmark import find_template, template_robot_pose
from omnigibson.eval.utils.eval_utils import set_sensor_modalities
from omnigibson.macros import gm
from run_nav2py_benchmark import add_nav2py_to_path, b1k_map_directory, og_robot_erosion_meters

DEFAULT_TASK_INSTANCES_ROOT = str(Path(__file__).resolve().parents[3] / "datasets" / "2026-challenge-task-instances")
DEFAULT_B1K_MAP_ROOT = "/scratch/gxwang2/b1k/b1k_gt_out/final_v4"
HEAD_SENSOR_LINK = "zed_link"
STRUCTURE_CATEGORIES = {"floors", "walls", "ceilings"}
# Never worth aiming a viewpoint at: fixtures with no search meaning.
SKIP_CATEGORIES = {"downlight", "electric_switch", "picture", "mirror", "fixed_window", "openable_window", "ceilings", "floors", "walls", "carpet", "door"}
# Only the modality set that is known to survive a long capture. omni.syntheticdata's
# _post_process_graph_tick segfaults Kit inside env.step() when extra annotators are
# attached: first with bbox_2d_tight (jobs 10572586-89, SIGSEGV after 10 frames), then
# still with seg_semantic (job 10572795, SIGSEGV before frame 5). This is exactly the
# set capture_object_search_frames.py runs for a full capture without crashing.
# Nothing downstream loses anything: the scorer matches on per-instance seg_instance
# masks, and an instance's CATEGORY comes from the scene object record (objects[].category
# in frames.json), not from seg_semantic. bbox_2d_tight is additionally the weaker ground
# truth -- keyed by semantic id, so four water glasses on one table share one box.
MODALITIES = {"rgb", "depth_linear", "seg_instance"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--task-instance-id", type=int, default=0)
    p.add_argument("--task-instances-root", default=DEFAULT_TASK_INSTANCES_ROOT)
    p.add_argument("--b1k-map-root", default=DEFAULT_B1K_MAP_ROOT)
    p.add_argument("--nav2py-root", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--robot-config", default=str(Path(__file__).resolve().parents[2] / "omnigibson" / "eval" / "r1pro.yaml"))
    p.add_argument("--robot-pose-key", default="robot")
    p.add_argument("--image-size", type=int, default=1080)
    p.add_argument("--max-targets", type=int, default=14)
    p.add_argument("--poses-per-target", type=int, default=3)
    p.add_argument("--min-range", type=float, default=1.2)
    p.add_argument("--max-range", type=float, default=3.6)
    p.add_argument("--spins", type=int, default=2)
    p.add_argument("--frames-per-spin", type=int, default=8)
    p.add_argument("--settle-steps", type=int, default=4)
    p.add_argument("--runtime-extra-clearance", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def to_list(v):
    if isinstance(v, th.Tensor):
        v = v.detach().cpu()
    return np.asarray(v, dtype=np.float64).tolist()


def numpy_of(v):
    return v.detach().cpu().numpy() if isinstance(v, th.Tensor) else np.asarray(v)


def yaw_quat(yaw):
    return th.tensor([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], dtype=th.float32)


def patched_scene_file(template_path, image_size):
    with open(template_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    found = False
    for _, info in scene["objects_info"]["init_info"].items():
        if info.get("class_module", "").startswith("omnigibson.robots"):
            info["args"]["obs_modalities"] = ["rgb"]
            info["args"]["sensor_config"] = {
                "VisionSensor": {"sensor_kwargs": {"image_height": int(image_size), "image_width": int(image_size)}}
            }
            found = True
    if not found:
        raise KeyError(f"No robot in {template_path}")
    return scene


def head_sensor(robot):
    for name, sensor in robot.sensors.items():
        if HEAD_SENSOR_LINK in name:
            return name, sensor
    raise KeyError(f"No {HEAD_SENSOR_LINK} camera among {list(robot.sensors)}")


def object_records(env, robot, inventory_names):
    records = []
    for obj in env.scene.objects:
        if obj.name == robot.name:
            continue
        category = getattr(obj, "category", None)
        rec = {"name": obj.name, "category": category, "in_static_map": obj.name in inventory_names,
               "is_structure": category in STRUCTURE_CATEGORIES}
        try:
            pos, quat = obj.get_position_orientation()
            lo, hi = obj.aabb
            rec["position"] = to_list(pos)
            rec["quat_xyzw"] = to_list(quat)
            rec["aabb"] = [to_list(lo), to_list(hi)]
            ext = np.asarray(rec["aabb"][1]) - np.asarray(rec["aabb"][0])
            rec["extent"] = [float(v) for v in ext]
            rec["max_extent"] = float(ext.max())
            rec["diag"] = float(np.linalg.norm(ext))
        except Exception as error:
            rec["error"] = f"{type(error).__name__}: {error}"
        records.append(rec)
    return records


def choose_targets(records, max_targets):
    """Task objects first (absent from the static map), then the smallest mapped objects."""
    usable = [r for r in records if "position" in r and not r["is_structure"] and r["category"] not in SKIP_CATEGORIES]
    task_objects = [r for r in usable if not r["in_static_map"]]
    rest = sorted([r for r in usable if r["in_static_map"]], key=lambda r: r["max_extent"])
    small = [r for r in rest if r["max_extent"] <= 0.5]
    large = [r for r in rest if r["max_extent"] > 0.5]
    picked, seen_cat = [], {}
    for pool, cap in ((task_objects, 99), (small, 4), (large, 6)):
        for rec in pool:
            if len(picked) >= max_targets:
                break
            # at most two instances of a mapped category, so viewpoints stay spread out
            if rec["in_static_map"] and seen_cat.get(rec["category"], 0) >= 2:
                continue
            if sum(1 for p in picked if p["in_static_map"] and (p["max_extent"] > 0.5) == (rec["max_extent"] > 0.5)) >= cap:
                continue
            seen_cat[rec["category"]] = seen_cat.get(rec["category"], 0) + 1
            picked.append(rec)
    return picked[:max_targets]


def free_cells(costmap):
    rows, cols = np.nonzero(costmap.data < 253)
    xs = costmap.origin_x + (cols + 0.5) * costmap.resolution
    ys = costmap.origin_y + (rows + 0.5) * costmap.resolution
    return np.stack([xs, ys], axis=1)


def sample_poses(cells, target_xy, n, rng, lo, hi, min_sep=0.6):
    """Free cells that face ``target_xy`` from roughly the middle of [lo, hi] metres.

    The post-erosion plannable area is small (Rs_int: 2.6k cells = ~1 m2 at 2 cm),
    so the annulus is widened twice before giving up; a frame from a bad range still
    contains the object, and GT tells us afterwards whether it was visible.
    """
    delta = cells - np.asarray(target_xy, dtype=np.float64)[None, :]
    dist = np.linalg.norm(delta, axis=1)
    for low, high in ((lo, hi), (0.6, 6.0), (0.0, float("inf"))):
        ok = np.nonzero((dist >= low) & (dist <= high))[0]
        if ok.size:
            break
    else:
        return []
    if ok.size == 0:
        return []
    # prefer the middle of the annulus, break ties randomly so the poses spread in angle
    mid = 0.5 * (lo + hi)
    order = ok[np.argsort(np.abs(dist[ok] - mid) + rng.uniform(0.0, 0.35, ok.size))]
    picked = []
    for idx in order:
        xy = cells[idx]
        if any(np.linalg.norm(xy - p[0]) < min_sep for p in picked):
            continue
        yaw = math.atan2(target_xy[1] - xy[1], target_xy[0] - xy[0])
        picked.append((xy, yaw, float(dist[idx])))
        if len(picked) >= n:
            break
    return picked


def grab_frame(robot, sensor_name):
    obs, info = robot.get_obs()
    frame, finfo = obs[sensor_name], info.get(sensor_name, {})
    rgb = numpy_of(frame["rgb"])[..., :3].astype(np.uint8)
    seg_i = numpy_of(frame["seg_instance"]).astype(np.int32)
    depth = numpy_of(frame["depth_linear"]).astype(np.float32)
    meta = {
        # The instance registry is a process-global counter, so id -> name is not stable
        # across runs and has to travel with the frames.
        "seg_instance_labels": {str(k): str(v) for k, v in finfo.get("seg_instance", {}).items()},
    }
    return rgb, seg_i, depth, meta


def run(args):
    template_path = find_template(Path(args.task_instances_root), args.scene, args.task, args.task_instance_id)
    map_dir = b1k_map_directory(args.b1k_map_root, args.scene)
    from benchmarks.costmaps import planning_costmap_from_navigation_2d
    from nav2py.objnav import inventory_instances

    with open(Path(map_dir).parent / "semantic_occupancy" / "scene_inventory.json", encoding="utf-8") as f:
        inventory_names = {inst.source_name for inst in inventory_instances(json.load(f))}

    robot_cfg = load_robot_config(args.robot_config)
    robot_cfg["obs_modalities"] = ["rgb"]
    cfg = build_env_config(scene_model=args.scene, robot_cfg=robot_cfg, scene_instance=template_path.stem, load_room_instances=None)
    cfg["scene"]["scene_file"] = patched_scene_file(template_path, args.image_size)
    started = time.time()
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    if robot.model in ("r1", "r1pro"):
        og.sim.stop()
        robot.base_footprint_link.mass = 250.0
        og.sim.play()
    env.reset(get_obs=False)
    print(f"  environment ready in {time.time() - started:.0f}s", flush=True)

    sensor_name, sensor = head_sensor(robot)
    set_sensor_modalities(sensor, set(MODALITIES))
    og.sim.update_handles()
    env.load_observation_space()
    for _ in range(3):
        og.sim.render()
    for attempt in range(6):
        try:
            robot.get_obs()
            break
        except RuntimeError as error:
            print(f"  warm-up get_obs {attempt}: {error}", flush=True)
            og.sim.render()
    print(f"  head sensor {sensor_name}: {sorted(sensor.modalities)} {sensor.image_height}x{sensor.image_width}", flush=True)

    og_resolution = float(env.scene.trav_map.map_resolution)
    costmap = planning_costmap_from_navigation_2d(
        map_dir, og_robot_erosion_meters(robot, og_resolution), args.runtime_extra_clearance
    )
    cells = free_cells(costmap)
    print(f"  plannable cells {len(cells)}", flush=True)

    records = object_records(env, robot, inventory_names)
    targets = choose_targets(records, args.max_targets)
    print("  targets: " + ", ".join(f"{t['name']}({t['max_extent']:.2f}m{'*' if not t['in_static_map'] else ''})" for t in targets), flush=True)

    start_position, start_quat = template_robot_pose(template_path, args.robot_pose_key)
    z = float(start_position[2])
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_dir) / args.scene / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for target in targets:
        for xy, yaw, dist in sample_poses(cells, target["position"][:2], args.poses_per_target, rng, args.min_range, args.max_range):
            plan.append({"kind": "aimed", "target": target["name"], "target_category": target["category"],
                         "xy": [float(xy[0]), float(xy[1])], "yaw": float(yaw), "range_m": dist})
    if len(cells):
        for k in range(args.spins):
            xy = cells[rng.integers(0, len(cells))]
            for j in range(args.frames_per_spin):
                plan.append({"kind": "spin", "target": None, "target_category": None,
                             "xy": [float(xy[0]), float(xy[1])], "yaw": 2 * math.pi * j / args.frames_per_spin, "spin": k})
    print(f"  {len(plan)} frames planned", flush=True)

    rgbs, segi, deps, frames_meta = [], [], [], []
    K = sensor.intrinsic_matrix.detach().cpu().numpy().astype(np.float64)

    def flush_frames():
        """Checkpoint to disk. Kit can segfault inside the render graph mid-capture
        (job 10572586), so a partial benchmark must survive rather than be lost."""
        if not frames_meta:
            return
        np.savez_compressed(out_dir / "frames.npz", rgb=np.stack(rgbs), seg_instance=np.stack(segi),
                            depth=np.stack(deps), K=K)
        with open(out_dir / "frames.json", "w", encoding="utf-8") as f:
            json.dump({"scene": args.scene, "task": args.task, "template": str(template_path),
                       "head_sensor": sensor_name, "image_size": [int(sensor.image_height), int(sensor.image_width)],
                       "modalities": sorted(MODALITIES), "objects": records,
                       "targets": [t["name"] for t in targets], "frames": frames_meta}, f, indent=1)

    t_capture = time.time()
    for i, item in enumerate(plan):
        place_robot(robot, th.tensor([item["xy"][0], item["xy"][1], z], dtype=th.float32), yaw_quat(item["yaw"]))
        for _ in range(args.settle_steps):
            env.step({robot.name: controller_no_op_action(robot)})
        rgb, si, dp, meta = grab_frame(robot, sensor_name)
        cam_pos, cam_quat = sensor.get_position_orientation()
        base_pos, base_quat = robot.get_position_orientation()
        rgbs.append(rgb); segi.append(si.astype(np.uint16)); deps.append(dp.astype(np.float16))
        meta.update(item)
        meta["frame"] = i
        meta["cam_pos"] = to_list(cam_pos)
        meta["cam_quat_xyzw"] = to_list(cam_quat)
        meta["base_pos"] = to_list(base_pos)
        meta["base_quat_xyzw"] = to_list(base_quat)
        ids, counts = np.unique(si, return_counts=True)
        meta["instance_pixels"] = {str(int(a)): int(b) for a, b in zip(ids, counts)}
        frames_meta.append(meta)
        if (i + 1) % 2 == 0:
            flush_frames()
            print(f"  {i + 1}/{len(plan)} frames ({time.time() - t_capture:.0f}s, checkpointed)", flush=True)

    flush_frames()
    print(f"  wrote {len(frames_meta)} frames to {out_dir} in {time.time() - t_capture:.0f}s", flush=True)
    og.clear()


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    add_nav2py_to_path(args.nav2py_root)
    with gm.unlocked():
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
