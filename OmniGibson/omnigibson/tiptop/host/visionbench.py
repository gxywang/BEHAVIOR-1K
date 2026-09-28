"""Export fixed human-demo situations for a small offline manipulation-vision benchmark.

Run this file directly with --validate for CPU-only manifest checks. Capture one task per process, with
OMNIGIBSON_HEADLESS=1. Model workers receive input.json/input.npz only; labels are separate oracle evaluation data.
Existing skill snapshots contain ten hold steps after replay to the named frame; this offset is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
VIEWS = ("head", "left_wrist", "right_wrist")
SCHEMA = "visionbench/1"
log = logging.getLogger(__name__)


class _RenderNotes(logging.Handler):
    """Retain the existing capture API's convergence warnings as evaluation provenance."""

    def __init__(self):
        super().__init__()
        self.views = {}

    def emit(self, record):
        match = re.match(r"(head|left_wrist|right_wrist): capture (converged|did not converge) after (\d+) renders",
                         record.getMessage())
        if match:
            name, status, renders = match.groups()
            self.views[name] = {"converged": status == "converged", "renders": int(renders)}


def read_cases(path: Path, task: str | None = None, ids: list[str] | None = None) -> list[dict]:
    """Validate hashes and selection before importing OmniGibson or starting a GPU process."""
    doc = json.loads(path.read_text())
    if doc.get("schema") != "visionbench-cases/1":
        raise ValueError("unsupported case manifest schema")
    cases = doc["cases"]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    if ids and set(ids) - {c["id"] for c in cases}:
        raise ValueError(f"unknown case IDs: {set(ids) - {c['id'] for c in cases}}")
    selected = [c for c in cases if (task is None or c["task"] == task) and (not ids or c["id"] in ids)]
    if not selected:
        raise ValueError("no cases selected")
    for case in selected:
        if Path(case["id"]).name != case["id"]:
            raise ValueError("case ID must be a directory basename")
        snapshot = ROOT / case["snapshot"]
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != case["snapshot_sha256"]:
            raise ValueError(f"snapshot hash mismatch: {snapshot}")
        state = json.loads(snapshot.read_text())
        if "robot_r1" not in state or not case["category_prompts"]:
            raise ValueError(f"incomplete snapshot or vocabulary: {case['id']}")
        if case["mode"] != "train" or case["frame"] < 0:
            raise ValueError(f"invalid demonstration coordinates: {case['id']}")
    return selected


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _stamp(og, sim) -> dict:
    return {
        "physics_step": int(og.sim.current_time_step_index),
        "sim_time_s": float(og.sim.current_time),
        "bridge_steps": int(sim.n_steps),
        "robot_pose": [_array(v).tolist() for v in sim.robot.get_position_orientation()],
        "robot_qpos": _array(sim.robot.get_joint_positions()).tolist(),
        "object_poses": {label: [_array(v).tolist() for v in obj.get_position_orientation()]
                         for label, obj in sorted(sim.objects.items())},
        "object_qpos": {label: _array(obj.get_joint_positions()).tolist()
                        for label, obj in sorted(sim.objects.items()) if obj.n_joints},
    }


def _assert_frozen(before: dict, after: dict) -> None:
    if before != after:
        raise RuntimeError("physics, robot, or object state changed during render-only capture")


def _label_objects(sim, case: dict) -> list[dict]:
    """All task objects and same-category scene distractors, independently of visibility."""
    from b1k.bridge.protocol import bddl_category

    prompts = case["category_prompts"]
    sim.objects = {}
    sim.track_task_objects(skip_categories=())
    scope = sim.task_scope()
    by_name, asset_prompts, objects = {}, {}, []
    for bddl, obj in sorted(scope.items()):
        category = bddl_category(bddl)
        if category not in prompts:
            raise ValueError(f"task category {category} absent from fixed manifest prompts")
        phrase = prompts[category]
        by_name[obj.name] = bddl
        asset_prompts.setdefault(obj.category, set()).add(phrase)
        objects.append({"id": bddl, "category": phrase, "target": bddl in case["target_ids"],
                        "label": sim.tracked_label(bddl), "task_object": True})
    for obj in sorted(sim.env.scene.objects, key=lambda o: o.name):
        if obj.name in by_name or obj is sim.robot:
            continue
        candidates = asset_prompts.get(obj.category, set())
        if not candidates and obj.category in prompts:
            candidates = {prompts[obj.category]}
        if len(candidates) != 1:
            continue
        label = f"scene_{obj.name}"
        sim.objects[label] = obj
        objects.append({"id": f"scene:{obj.name}", "category": next(iter(candidates)), "target": False,
                        "label": label, "task_object": False})
    missing = set(case["target_ids"]) - {o["id"] for o in objects}
    # Floors are outside the object benchmark; preserve this exclusion explicitly in labels.json.
    unsupported = {name for name in missing if not name.startswith(("floor.", "lawn."))}
    if unsupported:
        raise ValueError(f"target objects missing from loaded task: {sorted(unsupported)}")
    return objects


def _aabb_base(sim, obj) -> list:
    import torch

    lo, hi = [_array(v) for v in obj.aabb]
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    points = np.stack([_array(sim.to_base(torch.tensor(p, dtype=torch.float32), identity)[0]) for p in corners])
    return [points.min(axis=0).tolist(), points.max(axis=0).tolist()]


def validate_arrays(inputs: dict, labels: dict, count: int) -> None:
    for name in VIEWS:
        rgb, depth = inputs[f"{name}_rgb"], inputs[f"{name}_depth"]
        k, pose = inputs[f"{name}_K"], inputs[f"{name}_base_from_cam"]
        masks = labels[f"{name}_masks"]
        if rgb.dtype != np.uint8 or rgb.shape != (*depth.shape, 3):
            raise ValueError(f"invalid RGB/depth shape: {name}")
        if depth.dtype != np.float32 or not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError(f"invalid depth: {name}")
        if k.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
            raise ValueError(f"invalid calibration: {name}")
        if not np.allclose(pose[3], [0, 0, 0, 1]) or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError(f"invalid camera transform/intrinsics: {name}")
        if masks.dtype != np.bool_ or masks.shape != (count, *depth.shape):
            raise ValueError(f"invalid oracle masks: {name}")


def capture_case(og, sim, case: dict, out: Path, labeler=None, restore_snapshot=True) -> dict:
    from PIL import Image

    from omnigibson.tiptop.host.demo_cases import held, restore
    from omnigibson.tiptop.host.visionbench_coverage import precision_coverage
    from omnigibson.tiptop.oracle.segmenter import OracleSegmenter

    directory = out / case["id"]
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = ROOT / case["snapshot"]
    t0 = time.monotonic()
    if restore_snapshot:
        restore(sim.env, json.loads(snapshot.read_text()), case["instance"], case["mode"])
    objects = (labeler or _label_objects)(sim, case)
    # Link meshes are cached in object-local coordinates. Task instances may change scaling.
    sim._link_meshes.clear()
    before = _stamp(og, sim)
    frames = {}
    view_stamps = {}
    render_notes = _RenderNotes()
    capture_log = logging.getLogger("omnigibson.tiptop.r1pro")
    capture_log.addHandler(render_notes)
    try:
        for name in VIEWS:
            frames[name] = sim.view_frame(name)
            view_stamps[name] = _stamp(og, sim)
            _assert_frozen(before, view_stamps[name])
    finally:
        capture_log.removeHandler(render_notes)
    primary, extras = frames["head"]
    request = dict(primary, view_name="head", views=[dict(frames[n][0], name=n) for n in VIEWS[1:]])
    extras = dict(extras, views={n: frames[n][1] for n in VIEWS[1:]})
    masks = OracleSegmenter(sim).masks([o["label"] for o in objects], request, extras).value
    after = _stamp(og, sim)
    _assert_frozen(before, after)
    inputs, labels = {}, {}
    for name, (view, _) in frames.items():
        for source, suffix in (("rgb", "rgb"), ("depth", "depth"), ("intrinsics", "K"),
                               ("world_from_cam", "base_from_cam")):
            inputs[f"{name}_{suffix}"] = view[source]
        labels[f"{name}_masks"] = np.stack([masks[name][o["label"]] for o in objects]).astype(bool)
    validate_arrays(inputs, labels, len(objects))
    for obj in objects:
        obj["aabb_base"] = _aabb_base(sim, sim.objects[obj["label"]])
        obj["visible_pixels"] = {name: int(masks[name][obj["label"]].sum()) for name in VIEWS}
        del obj["label"]
    metadata = {
        "schema": SCHEMA, "id": case["id"], "task": case["task"], "episode_index": case["episode_index"],
        "frame": case["frame"], "source_snapshot_sha256": case["snapshot_sha256"], "views": list(VIEWS),
        "categories": sorted(set(case["category_prompts"].values())),
        "prompts": sorted(set(case["category_prompts"].values())),
        "depth_source": "simulator_linear_depth", "depth_self_filter": "robot_link_geometry",
        "camera_convention": "OpenCV +x right +y down +z forward; base_from_cam maps camera points to robot base",
        "source_materialization_hold_steps": case["source_materialization_hold_steps"],
        "capture_extra_hold_steps": 0, "physics_steps_during_capture": 0, "dataset_fps": 30,
    }
    truth = {
        "schema": SCHEMA, "id": case["id"], "objects": objects, "label_source": "oracle_geometry",
        "geometry_tolerance_m": sim.gt_mask_tol, "render_convergence": render_notes.views,
        "label_scope": "task objects plus same-category scene distractors matched by task asset category",
        "aabb_source": "world AABB corners transformed into robot base (conservative full-object bounds)",
        "excluded_floor_target_ids": [i for i in case["target_ids"] if i.startswith(("floor.", "lawn."))],
        "original_replay_fidelity": case["fidelity"], "restore_stamp": before,
        "view_stamps": view_stamps, "capture_stamp": after, "held_objects": held(sim.robot),
    }
    truth.update(precision_coverage(case))
    np.savez_compressed(directory / "input.npz", **inputs)
    np.savez_compressed(directory / "labels.npz", **labels)
    (directory / "input.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (directory / "labels.json").write_text(json.dumps(truth, indent=2) + "\n")
    for name in VIEWS:
        Image.fromarray(inputs[f"{name}_rgb"]).save(directory / f"{name}.png")
        overlay = inputs[f"{name}_rgb"].copy()
        for i, mask in enumerate(labels[f"{name}_masks"]):
            color = np.array([(67 * i + 71) % 255, (131 * i + 139) % 255, (193 * i + 211) % 255])
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)
        Image.fromarray(overlay).save(directory / f"{name}_oracle.png")
    return {"id": case["id"], "ok": True, "wall_s": round(time.monotonic() - t0, 2),
            "objects": len(objects), "visible_objects": {n: int(labels[f"{n}_masks"].any(axis=(1, 2)).sum())
                                                        for n in VIEWS}}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("visionbench_cases.json"))
    parser.add_argument("--out", type=Path, default=ROOT / "runs/vision/captures")
    parser.add_argument("--task", help="capture exactly one task per process")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--validate", action="store_true", help="check metadata and snapshot hashes, no sim imports")
    args = parser.parse_args(argv)
    cases = read_cases(args.manifest, args.task, args.ids)
    if args.validate:
        print(json.dumps({"ok": True, "cases": len(cases), "tasks": sorted({c['task'] for c in cases})}))
        return
    if len({c["task"] for c in cases}) != 1:
        parser.error("capture one task per process: supply --task or --ids")
    args.out.mkdir(parents=True, exist_ok=True)

    import omnigibson as og
    from omnigibson.eval.evaluator import DISABLED_TRANSITION_RULES
    from omnigibson.tiptop.r1pro import R1ProSim, challenge_task_info, make_r1pro_env_config
    from omnigibson.tiptop.run import setup_logging

    setup_logging()
    for rule in DISABLED_TRANSITION_RULES:
        rule.ENABLED = False
    task = cases[0]["task"]
    scene, rooms = challenge_task_info(task)
    cfg = make_r1pro_env_config(scene_model=scene, load_room_instances=rooms, activity=task,
                               grasping_mode="assisted", camera="head", views=VIEWS[1:], segmentation=False)
    failed = False
    try:
        try:
            sim = R1ProSim(cfg, camera="head", views=VIEWS[1:], look_arm=None)
        except Exception as error:
            for case in cases:
                directory = args.out / case["id"]
                directory.mkdir(parents=True, exist_ok=True)
                result = {"id": case["id"], "ok": False, "phase": "initialization", "error": str(error),
                          "traceback": traceback.format_exc()}
                (directory / "failure.json").write_text(json.dumps(result, indent=2) + "\n")
                with (args.out / "capture_results.jsonl").open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
            raise
        for case in cases:
            directory = args.out / case["id"]
            if (directory / "input.json").exists() or (directory / "failure.json").exists():
                raise FileExistsError(f"capture already exists; use a fresh output directory: {directory}")
            try:
                result = capture_case(og, sim, case, args.out)
            except Exception as error:
                failed = True
                result = {"id": case["id"], "ok": False, "error": str(error), "traceback": traceback.format_exc()}
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "failure.json").write_text(json.dumps(result, indent=2) + "\n")
                log.exception("capture failed for %s", case["id"])
            with (args.out / "capture_results.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        if og.app is not None:
            og.shutdown()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
