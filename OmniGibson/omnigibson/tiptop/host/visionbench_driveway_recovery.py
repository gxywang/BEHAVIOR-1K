"""Additive recovery for seven predeclared driveway scope failures; never replay or replace cases.

The original capture implementation and all successful captures remain immutable. A local
R1Pro subclass exposes only the missing driveway BDDL binding, then the established capture
and relabel functions run against the original saved snapshot in a fresh output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
SELECTION_SHA = "d504a3cda4338df2b3cd65d11eb62d1e9eea0890742b8e060b29f50ede9d3c88"
ALLOWED_IDS = (
    "comprehensive.chopping_wood.e8884.f429",
    "comprehensive.chopping_wood.e8832.f1804",
    "comprehensive.stacking_wood.e11721.f360",
    "comprehensive.stacking_wood.e11631.f1500",
    "comprehensive.stacking_wood.e11611.f644",
    "comprehensive.clean_your_rusty_garden_tools.e14210.f2610",
    "comprehensive.clean_your_rusty_garden_tools.e14309.f6180",
)
ERROR = "task targets missing from complete scene labels: ['driveway.n.01_1']"
NATIVE = (
    "input.json",
    "input.npz",
    "labels.json",
    "labels.npz",
    "labels_strict4mm.npz",
    "head.png",
    "left_wrist.png",
    "right_wrist.png",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def verify_bindings(rows):
    for row in rows:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("Frozen recovery input changed: " + row["path"])


def selection_cases(selection_path):
    if sha(selection_path) != SELECTION_SHA:
        raise ValueError("Recovery is limited to the original frozen all-task selection.")
    selection = read(selection_path)
    eligible = {c["id"] for c in selection["cases"] if any(t.startswith("driveway.") for t in c["target_ids"])}
    if eligible != set(ALLOWED_IDS):
        raise ValueError("Metadata-derived driveway case set changed.")
    verify_bindings([{"path": p, "sha256": s} for p, s in selection["source_receipts"].items()])
    return {c["id"]: c for c in selection["cases"]}


def declare(selection_path, recovery):
    cases = selection_cases(selection_path)
    if recovery.exists():
        raise FileExistsError(recovery)
    host = Path(__file__).parent
    sources = [
        *read(selection_path)["source_receipts"],
        str(Path(__file__).resolve()),
        str(host / "visionbench_driveway_recovery_verify.py"),
        str(host / "visionbench_driveway_recovery_continue.py"),
        str(ROOT / "OmniGibson/omnigibson/tiptop/r1pro.py"),
        str(ROOT / "tiptop/b1k/bridge/protocol.py"),
    ]
    value = {
        "schema": "driveway-capture-recovery-amendment/1",
        "created_at": time.time(),
        "selection": binding(selection_path),
        "allowed_case_ids": list(ALLOWED_IDS),
        "case_metadata": [cases[c] for c in ALLOWED_IDS],
        "sources": [binding(p) for p in sources],
        "cause": "R1Pro.task_scope filters driveway with floors; frozen benchmark excludes only floor/lawn.",
        "action": "Expose only selected driveway bindings from raw task.object_scope; restore saved snapshot.",
        "required_failure": ERROR,
        "new_replay_frames": 0,
        "settle_steps": 0,
        "case_selection_changed": False,
        "primary_geometry_algorithm_changed": False,
        "successful_captures_changed": False,
        "model_or_prompt_or_threshold_changed": False,
        "predictions_read": False,
        "no_replacements": True,
        "eligibility": "Actual matching failure, failed task receipt, original materialization and snapshot, no success receipt.",
    }
    write_new(recovery / "amendment.json", value)
    return value


def validate_failure(case, captures, selection_sha=SELECTION_SHA):
    identifier = case["id"]
    if identifier not in ALLOWED_IDS:
        raise ValueError("Case not in fixed metadata allowlist.")
    if (captures / "case_receipts" / (identifier + ".json")).exists() or (captures / identifier).exists():
        raise ValueError("Never recover an existing success or overwrite a capture.")
    failure_path = captures / "_attempt_failures" / identifier / "failure.json"
    failure = read(failure_path)
    if (
        failure.get("id") != identifier
        or failure.get("primary_id") != identifier
        or failure.get("ok") is not False
        or failure.get("error") != ERROR
        or failure.get("selection_sha256") != selection_sha
    ):
        raise ValueError("Only the exact observed driveway-binding failure is recoverable.")
    task_path = captures / "task_receipts" / (case["task"] + ".json")
    task = read(task_path)
    if task.get("ok") is not False or task.get("selection_sha256") != selection_sha:
        raise ValueError("Original failed task receipt required.")
    matching = [r for r in task["cases"] if r.get("primary_id") == identifier]
    if len(matching) != 1 or matching[0].get("ok") is not False or failure not in matching[0]["attempts"]:
        raise ValueError("Failure does not match original task receipt.")
    materialization_path = captures / "materialization" / (identifier + ".json")
    materialization = read(materialization_path)
    actual = materialization["case"]
    if materialization["selection_sha256"] != selection_sha:
        raise ValueError("Materialization belongs to a different selection.")
    expected = dict(case)
    expected["snapshot_sha256"] = actual.get("snapshot_sha256")
    expected["fidelity"] = actual.get("fidelity")
    if expected != actual or sha(case["snapshot"]) != actual["snapshot_sha256"]:
        raise ValueError("Materialization metadata or immutable snapshot differs.")
    return actual, [binding(p) for p in (failure_path, task_path, materialization_path, case["snapshot"])]


def extend_scope(filtered, raw, scene_objects, names):
    """Pure binding adapter: no taxonomy, selection, poses, physics or object creation."""
    if not names or any(not n.startswith("driveway.") for n in names):
        raise ValueError("Only declared driveway bindings may be exposed.")
    result = dict(filtered)
    for name in names:
        obj = raw.get(name)
        if (
            obj is None
            or getattr(obj, "category", None) != "driveway"
            or not getattr(obj, "fixed_base", False)
            or not any(obj is x for x in scene_objects)
        ):
            raise ValueError("Driveway binding must resolve to the existing loaded driveway object.")
        if name in result and result[name] is not obj:
            raise ValueError("Conflicting pre-existing task binding.")
        result[name] = obj
    return result


def restore_state_audit(sim, snapshot, tolerance=1e-5):
    """Compare every saved rigid pose/joint with restored state, with the existing 1e-5 pose tolerance."""
    import numpy as np

    errors = {}
    for name, expected in snapshot.items():
        obj = sim.env.scene.object_registry("name", name)
        if obj is None:
            raise ValueError("Saved object absent after restore: " + name)
        actual = obj.dump_state(serialized=False)
        for key in ("pos", "ori"):
            a = np.asarray(expected["root_link"][key], dtype=float)
            value = actual["root_link"][key]
            b = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=float)
            error = min(np.max(np.abs(a - b)), np.max(np.abs(a + b))) if key == "ori" else np.max(np.abs(a - b))
            errors[name + "/root_link/" + key] = float(error)
        if "joint_pos" in expected:
            a = np.asarray(expected["joint_pos"], dtype=float)
            value = actual["joint_pos"]
            b = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=float)
            if a.shape != b.shape:
                raise ValueError("Saved joint dimensions changed: " + name)
            errors[name + "/joint_pos"] = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not errors or any(not np.isfinite(x) or x > tolerance for x in errors.values()):
        raise ValueError("Saved snapshot pose/joint restoration failed: " + repr(errors))
    return {
        "tolerance": tolerance,
        "maximum_absolute_error": max(errors.values()),
        "objects_checked": len(snapshot),
        "components": errors,
    }


def prepare(selection_path, captures, recovery):
    cases = selection_cases(selection_path)
    amendment = read(recovery / "amendment.json")
    verify_bindings(amendment["sources"] + [amendment["selection"]])
    execution = read(captures / "execution.json")
    if execution.get("status") != "incomplete":
        raise ValueError("Wait for the original supervisor to finish with preserved failures.")
    failures = set()
    for identifier, case in cases.items():
        receipt = captures / "case_receipts" / (identifier + ".json")
        if not receipt.exists():
            validate_failure(case, captures)
            failures.add(identifier)
    if not failures or failures - set(ALLOWED_IDS):
        raise ValueError("Only declared, observed driveway failures may remain.")
    protected = []
    for identifier, case in cases.items():
        paths = [Path(case["snapshot"]), captures / "materialization" / (identifier + ".json")]
        if identifier not in failures:
            paths.extend(p for p in sorted((captures / identifier).iterdir()) if p.is_file())
            paths.append(captures / "case_receipts" / (identifier + ".json"))
        else:
            paths.append(captures / "_attempt_failures" / identifier / "failure.json")
        protected.extend(binding(p) for p in paths)
    original = []
    for path in [captures / "execution.json", *sorted((captures / "task_receipts").glob("*.json"))]:
        destination = recovery / "original_receipts" / path.relative_to(captures)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(path.read_bytes())
        original.append({"original": binding(path), "preserved": binding(destination)})
    result = {
        "schema": "driveway-capture-recovery-preparation/1",
        "amendment": binding(recovery / "amendment.json"),
        "captures": str(captures.resolve()),
        "recovery": str(recovery.resolve()),
        "eligible_case_ids": [c for c in ALLOWED_IDS if c in failures],
        "protected_files": protected,
        "original_receipts": original,
        "successful_cases_before": len(cases) - len(failures),
        "cases_expected": len(cases),
    }
    write_new(recovery / "preparation.json", result)
    return result


def worker(selection_path, captures, recovery, task):
    cases = selection_cases(selection_path)
    amendment = read(recovery / "amendment.json")
    verify_bindings(amendment["sources"] + [amendment["selection"]])
    prepared = read(recovery / "preparation.json")
    rows = [cases[c] for c in prepared["eligible_case_ids"] if cases[c]["task"] == task]
    if not rows:
        raise ValueError("No declared recovery cases for this task.")
    materialized = [validate_failure(c, captures) for c in rows]
    import omnigibson as og
    from omegaconf import OmegaConf
    from omnigibson.eval.evaluator import DISABLED_TRANSITION_RULES, DEFAULT_ROBOT_CONFIG_PATH, EVAL_BASE_LINK_MASS
    from omnigibson.eval.utils.eval_utils import generate_basic_environment_config
    from omnigibson.tiptop.host import demo_cases
    from omnigibson.tiptop.r1pro import R1ProSim, challenge_task_info, make_r1pro_env_config
    from omnigibson.tiptop.run import setup_logging
    from visionbench import VIEWS, _stamp, _assert_frozen, capture_case
    from visionbench_expand import configure_demo_replay
    from visionbench_v2 import expanded_objects, relabel_case

    setup_logging()
    for rule in DISABLED_TRANSITION_RULES:
        rule.ENABLED = False
    scene, rooms = challenge_task_info(task)
    config = make_r1pro_env_config(
        scene_model=scene,
        load_room_instances=rooms,
        activity=task,
        grasping_mode="assisted",
        camera="head",
        views=VIEWS[1:],
        segmentation=False,
    )
    evaluator_robot = OmegaConf.to_container(OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH))
    metadata = demo_cases.available_tasks()[task][0]
    configure_demo_replay(config, evaluator_robot, metadata)
    evaluator = generate_basic_environment_config(task, metadata)
    evaluator["scene"]["load_room_instances"] = rooms
    config["scene"], config["task"] = evaluator["scene"], evaluator["task"]
    config["env"] = evaluator["env"] | {"external_sensors": config["env"]["external_sensors"]}

    class RecoverySim(R1ProSim):
        recovery_names = ("driveway.n.01_1",)

        def task_scope(self):
            return extend_scope(
                super().task_scope(), self.env.task.object_scope, self.env.scene.objects, self.recovery_names
            )

    source, output = recovery / "_source_captures", recovery / "captures"
    try:
        sim = RecoverySim(config, camera="head", views=VIEWS[1:], look_arm=None)
        og.sim.stop()
        sim.robot.base_footprint_link.mass = EVAL_BASE_LINK_MASS
        og.sim.play()
        for case, inputs in materialized:
            identifier = case["id"]
            if (source / identifier).exists() or (output / identifier).exists():
                raise FileExistsError("Recovery output already exists; preserve it for investigation.")
            start = time.monotonic()
            snapshot = read(case["snapshot"])
            demo_cases.restore(sim.env, snapshot, case["instance"], case["mode"])
            restored = restore_state_audit(sim, snapshot)
            sim.objects = {}
            sim.track_task_objects(skip_categories=())
            before = _stamp(og, sim)
            binding_evidence = [
                {
                    "task_id": n,
                    "scene_name": sim.env.task.object_scope[n].name,
                    "asset_category": sim.env.task.object_scope[n].category,
                    "fixed_base": bool(sim.env.task.object_scope[n].fixed_base),
                }
                for n in sim.recovery_names
            ]
            capture_case(
                og,
                sim,
                case,
                source,
                labeler=lambda host, item: expanded_objects(host, item)[0],
                restore_snapshot=False,
            )
            result = relabel_case(og, sim, case, source, output, restore_snapshot=False)
            # Expanded scene labels add distractors to sim.objects, so compare the persistent time/robot stamp.
            after = _stamp(og, sim)
            _assert_frozen(
                {k: v for k, v in before.items() if k not in ("object_poses", "object_qpos")},
                {k: v for k, v in after.items() if k not in ("object_poses", "object_qpos")},
            )
            restored_after = restore_state_audit(sim, snapshot)
            labels_path = output / identifier / "labels.json"
            labels = read(labels_path)
            for row in labels["objects"]:
                obj = sim.env.scene.object_registry("name", row["scene_name"])
                model = getattr(obj, "model", None)
                row.update(
                    asset_model=str(model) if model is not None else None,
                    asset_id=f"{obj.category}.{model}" if model is not None else None,
                    asset_identity_source="Runtime scene category/model, evaluation labels only.",
                )
            evidence_path = recovery / "case_provenance" / (identifier + ".json")
            evidence = {
                "schema": "driveway-capture-recovery-case/1",
                "case_id": identifier,
                "amendment": binding(recovery / "amendment.json"),
                "inputs": inputs,
                "snapshot_restore_before_capture": restored,
                "snapshot_restore_after_capture": restored_after,
                "binding_evidence": binding_evidence,
                "new_replay_frames": 0,
                "settle_steps": 0,
                "target_scope_note": "Original annotation referents include supports; driveway remains a declared target, including the support-only case.",
                "physics_steps_after_restore": after["physics_step"] - before["physics_step"],
                "sim_time_after_restore_s": after["sim_time_s"] - before["sim_time_s"],
                "source_outputs": [binding(p) for p in sorted((source / identifier).iterdir()) if p.is_file()],
            }
            write_new(evidence_path, evidence)
            recovery_link = binding(evidence_path)
            labels.update(
                cohort=case["cohort"],
                split=case["split"],
                human_skill=case["human_skill"],
                stratum=case["stratum"],
                stage=case["stage"],
                selection_sha256=SELECTION_SHA,
                materialization_receipt=f"materialization/{identifier}.json",
                asset_identity_is_evaluation_only=True,
                target_resolution=case["target_resolution"],
                all_initial_task_object_ids=case["all_initial_task_object_ids"],
                mask_provenance={
                    "method": "Geometry proximity to captured linear depth",
                    "main_tolerance_m": 0.008,
                    "sensitivity_tolerance_m": 0.004,
                    "renderer_instance_ground_truth": False,
                    "limitations": "Approximate geometry labels; contact halos/depth-mesh errors possible.",
                },
                capture_recovery=recovery_link,
            )
            labels_path.write_text(json.dumps(labels, indent=2) + "\n")
            result.update(
                selection_sha256=SELECTION_SHA,
                primary_id=identifier,
                effective_id=identifier,
                attempts=[],
                capture_recovery=recovery_link,
                replay_frames=0,
                original_replay_frames=case["frame"],
                wall_s=round(time.monotonic() - start, 2),
            )
            verify_bindings(inputs)
            write_new(output / "case_receipts" / (identifier + ".json"), result)
            print(json.dumps({"phase": "recovered", "case": identifier, "receipt": result}), flush=True)
    finally:
        if og.app is not None:
            og.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "prepare", "worker"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.command == "declare":
        result = declare(args.selection.resolve(), args.recovery.resolve())
    elif args.command == "prepare":
        result = prepare(args.selection.resolve(), args.captures.resolve(), args.recovery.resolve())
    else:
        result = worker(args.selection.resolve(), args.captures.resolve(), args.recovery.resolve(), args.task)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
