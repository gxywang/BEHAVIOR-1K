"""CPU-only independent provenance verification and publication for driveway recovery."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ALLOWED = {
    "comprehensive.chopping_wood.e8884.f429",
    "comprehensive.chopping_wood.e8832.f1804",
    "comprehensive.stacking_wood.e11721.f360",
    "comprehensive.stacking_wood.e11631.f1500",
    "comprehensive.stacking_wood.e11611.f644",
    "comprehensive.clean_your_rusty_garden_tools.e14210.f2610",
    "comprehensive.clean_your_rusty_garden_tools.e14309.f6180",
}
SELECTION_SHA = "d504a3cda4338df2b3cd65d11eb62d1e9eea0890742b8e060b29f50ede9d3c88"
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
ERROR = "task targets missing from complete scene labels: ['driveway.n.01_1']"
VIEWS = ("head", "left_wrist", "right_wrist")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def verify(recovery, captures, published=False, mapping=None):
    """All original paths can be resolved exclusively through a portable manifest map."""
    mapping = dict(mapping or {})
    strict_mapping = bool(mapping)

    def resolve(path):
        key = str(path)
        if key in mapping:
            return Path(mapping[key])
        if strict_mapping:
            raise ValueError("Unmapped original path in portable recovery audit: " + key)
        return Path(path)

    def checked(record):
        path = resolve(record["path"])
        if not path.is_file() or sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Recovery provenance checksum mismatch: " + record["path"])
        return path

    amendment = read(recovery / "amendment.json")
    preparation = read(recovery / "preparation.json")
    if amendment["schema"] != "driveway-capture-recovery-amendment/1":
        raise ValueError("Unsupported recovery amendment.")
    if amendment["selection"]["sha256"] != SELECTION_SHA or set(amendment["allowed_case_ids"]) != ALLOWED:
        raise ValueError("Wrong fixed selection or recovery allowlist.")
    if (
        amendment["new_replay_frames"]
        or amendment["settle_steps"]
        or amendment["predictions_read"]
        or amendment["case_selection_changed"]
        or amendment["primary_geometry_algorithm_changed"]
        or amendment["successful_captures_changed"]
        or amendment["model_or_prompt_or_threshold_changed"]
        or not amendment["no_replacements"]
    ):
        raise ValueError("Recovery exceeds the allowed unchanged protocol.")
    checked(preparation["amendment"])
    selection = read(checked(amendment["selection"]))
    cases = {c["id"]: c for c in selection["cases"]}
    ids = preparation["eligible_case_ids"]
    if not ids or len(ids) != len(set(ids)) or set(ids) - ALLOWED:
        raise ValueError("Invalid actual recovery IDs.")
    for record in amendment["sources"]:
        checked(record)
    # After reconciliation, task/execution originals resolve to their preserved byte-for-byte copies.
    original_execution = None
    for row in preparation["original_receipts"]:
        preserved = checked(row["preserved"])
        if sha(preserved) != row["original"]["sha256"]:
            raise ValueError("Original receipt was not preserved exactly.")
        if published:
            mapping[row["original"]["path"]] = preserved
        else:
            checked(row["original"])
        if Path(row["original"]["path"]).name == "execution.json":
            original_execution = read(preserved)
    if original_execution is None or original_execution["status"] != "incomplete":
        raise ValueError("Expected original incomplete supervisor execution.")
    for record in preparation["protected_files"]:
        checked(record)
    if published and (recovery / "verification_published.json").exists():
        sealed = read(recovery / "verification_published.json")
        if (
            sealed["amendment_sha256"] != sha(recovery / "amendment.json")
            or sealed["preparation_sha256"] != sha(recovery / "preparation.json")
            or sealed["recovered_case_ids"] != ids
            or sealed["verifier_source_sha256"] != sha(__file__)
        ):
            raise ValueError("Published recovery verification source or amendment changed.")
        for record in sealed["artifacts"]:
            checked(record)
    artifacts, object_counts = [], {}
    for identifier in ids:
        case = cases[identifier]
        evidence_path = recovery / "case_provenance" / (identifier + ".json")
        evidence = read(evidence_path)
        if (
            evidence["schema"] != "driveway-capture-recovery-case/1"
            or evidence["case_id"] != identifier
            or evidence["new_replay_frames"]
            or evidence["settle_steps"]
            or evidence["physics_steps_after_restore"]
            or evidence["sim_time_after_restore_s"]
        ):
            raise ValueError("Invalid recovery capture physics/provenance.")
        checked(evidence["amendment"])
        original_inputs = [checked(row) for row in evidence["inputs"]]
        failure = read(next(p for p in original_inputs if p.name == "failure.json"))
        if (
            failure.get("id") != identifier
            or failure.get("primary_id") != identifier
            or failure.get("ok") is not False
            or failure.get("selection_sha256") != SELECTION_SHA
            or failure.get("error") != ERROR
        ):
            raise ValueError("Missing exact original capture failure.")
        original_task = read(next(p for p in original_inputs if p.name == case["task"] + ".json"))
        task_rows = [r for r in original_task["cases"] if r.get("primary_id") == identifier]
        if original_task["ok"] is not False or len(task_rows) != 1 or failure not in task_rows[0]["attempts"]:
            raise ValueError("Preserved failed task does not contain original failure.")
        materialized = read(next(p for p in original_inputs if p.parent.name == "materialization"))
        actual = materialized["case"]
        expected = dict(case, snapshot_sha256=actual["snapshot_sha256"], fidelity=actual["fidelity"])
        if actual != expected or materialized["selection_sha256"] != SELECTION_SHA:
            raise ValueError("Selected frame, task, episode, targets or prompts changed.")
        snapshot = checked(next(r for r in evidence["inputs"] if r["path"] == case["snapshot"]))
        if sha(snapshot) != actual["snapshot_sha256"]:
            raise ValueError("Snapshot differs from original materialization.")
        state = read(snapshot)
        for field in ("snapshot_restore_before_capture", "snapshot_restore_after_capture"):
            audit = evidence[field]
            expected_components = {n + "/root_link/" + k for n in state for k in ("pos", "ori")}
            expected_components |= {n + "/joint_pos" for n, obj in state.items() if "joint_pos" in obj}
            if (
                audit["tolerance"] != 1e-5
                or audit["objects_checked"] != len(state)
                or set(audit["components"]) != expected_components
                or any(not np.isfinite(v) or v < 0 or v > 1e-5 for v in audit["components"].values())
                or audit["maximum_absolute_error"] != max(audit["components"].values())
            ):
                raise ValueError("Snapshot restoration audit is incomplete or outside tolerance.")
        for record in evidence["source_outputs"]:
            checked(record)
        folder = captures / identifier if published else recovery / "captures" / identifier
        receipt_path = (
            resolve(Path(preparation["captures"]) / "case_receipts" / (identifier + ".json"))
            if published and strict_mapping
            else captures / "case_receipts" / (identifier + ".json")
            if published
            else recovery / "captures/case_receipts" / (identifier + ".json")
        )
        receipt = read(receipt_path)
        if (
            receipt.get("ok") is not True
            or receipt["effective_id"] != identifier
            or receipt["primary_id"] != identifier
            or receipt["attempts"]
            or receipt["replay_frames"]
            or receipt["original_replay_frames"] != case["frame"]
            or receipt["selection_sha256"] != SELECTION_SHA
        ):
            raise ValueError("Invalid recovered success receipt.")
        checked(receipt["capture_recovery"])
        if receipt["capture_recovery"]["sha256"] != sha(evidence_path):
            raise ValueError("Recovered receipt has wrong evidence link.")
        metadata, labels = read(folder / "input.json"), read(folder / "labels.json")
        if labels["capture_recovery"] != receipt["capture_recovery"]:
            raise ValueError("Label recovery provenance differs from receipt.")
        if (
            metadata["id"] != identifier
            or metadata["task"] != case["task"]
            or metadata["episode_index"] != case["episode_index"]
            or metadata["frame"] != case["frame"]
            or metadata["source_snapshot_sha256"] != actual["snapshot_sha256"]
            or metadata["source_materialization_hold_steps"]
            or metadata["capture_extra_hold_steps"]
            or metadata["physics_steps_during_capture"]
            or labels["physics_steps_during_relabel"]
            or labels["renders_during_relabel"]
            or labels["cohort"] != case["cohort"]
            or labels["split"] != case["split"]
            or set(metadata["categories"]) != set(case["category_prompts"].values())
            or set(metadata["prompts"]) != set(case["category_prompts"].values())
            or labels["geometry_tolerance_m"] != 0.008
            or labels["strict_geometry_tolerance_m"] != 0.004
        ):
            raise ValueError("Recovered inputs violate frozen capture protocol.")
        if {"asset_id", "asset_model", "target_ids", "objects", "expected_target_models"} & set(metadata):
            raise ValueError("Evaluation identities leaked into model input.")
        if (
            labels["asset_identity_is_evaluation_only"] is not True
            or labels["mask_provenance"]["renderer_instance_ground_truth"] is not False
        ):
            raise ValueError("Missing approximate geometry provenance.")
        source_by_name = {Path(r["path"]).name: r for r in evidence["source_outputs"]}
        for name in ("input.json", "input.npz", "head.png", "left_wrist.png", "right_wrist.png"):
            if sha(folder / name) != source_by_name[name]["sha256"]:
                raise ValueError("Source capture bytes changed during relabel or publication.")
        if labels["source_file_sha256"] != {
            n: source_by_name[n]["sha256"] for n in ("input.json", "input.npz", "labels.json", "labels.npz")
        }:
            raise ValueError("Relabel source hashes differ.")
        targets = {n for obj in labels["objects"] if obj["target"] for n in obj["task_ids"]}
        if not set(case["target_ids"]) <= targets:
            raise ValueError("A declared target is missing after recovery.")
        driveway = [o for o in labels["objects"] if "driveway.n.01_1" in o["task_ids"]]
        if len(driveway) != 1 or not driveway[0]["target"] or not driveway[0]["task_object"]:
            raise ValueError("Driveway target binding still absent.")
        if evidence["binding_evidence"] != [
            {
                "task_id": "driveway.n.01_1",
                "scene_name": driveway[0]["scene_name"],
                "asset_category": "driveway",
                "fixed_base": True,
            }
        ]:
            raise ValueError("Driveway binding evidence does not match actual labels.")
        if driveway[0]["scene_name"] not in state:
            raise ValueError("Recovered driveway was absent from the immutable snapshot.")
        with (
            np.load(folder / "input.npz", allow_pickle=False) as images,
            np.load(folder / "labels.npz", allow_pickle=False) as masks,
            np.load(folder / "labels_strict4mm.npz", allow_pickle=False) as strict,
        ):
            for view in VIEWS:
                shape = images[view + "_rgb"].shape[:2]
                for archive in (masks, strict):
                    value = archive[view + "_masks"]
                    if (
                        value.dtype != bool
                        or value.shape != (len(labels["objects"]), *shape)
                        or np.any(value.sum(0) > 1)
                    ):
                        raise ValueError("Recovered masks fail shape, boolean or disjointness checks.")
                for index, obj in enumerate(labels["objects"]):
                    if int(masks[view + "_masks"][index].sum()) != obj["visible_pixels"][view]:
                        raise ValueError("Recovered mask count does not match labels.")
        for path in [evidence_path, receipt_path, *(folder / name for name in NATIVE)]:
            artifacts.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size})
        object_counts[identifier] = len(labels["objects"])
    return {
        "schema": "driveway-capture-recovery-verification/1",
        "ok": True,
        "selection_sha256": SELECTION_SHA,
        "amendment_sha256": sha(recovery / "amendment.json"),
        "preparation_sha256": sha(recovery / "preparation.json"),
        "recovered_case_ids": ids,
        "protected_files_verified": len(preparation["protected_files"]),
        "artifacts": artifacts,
        "objects_per_case": object_counts,
        "published": published,
        "predictions_read": False,
        "verifier_source_sha256": sha(__file__),
    }


def promote(recovery, captures):
    before = verify(recovery, captures)
    write_new(recovery / "verification_before_publish.json", before)
    preparation = read(recovery / "preparation.json")
    ids = preparation["eligible_case_ids"]
    for identifier in ids:
        if (captures / identifier).exists() or (captures / "case_receipts" / (identifier + ".json")).exists():
            raise FileExistsError("Refuse to overwrite an existing case.")
    for identifier in ids:
        shutil.copytree(recovery / "captures" / identifier, captures / identifier)
        source = recovery / "captures/case_receipts" / (identifier + ".json")
        with (captures / "case_receipts" / (identifier + ".json")).open("xb") as stream:
            stream.write(source.read_bytes())
    # Verification finishes before any task or global status can become complete and trigger sealing.
    after = verify(recovery, captures, published=True)
    write_new(recovery / "verification_published.json", after)
    original = read(recovery / "original_receipts/execution.json")
    selection = read(read(recovery / "amendment.json")["selection"]["path"])
    affected = {c["task"] for c in selection["cases"] if c["id"] in ids}

    def replace(path, value):
        tmp = path.with_suffix(".recovery.tmp")
        write_new(tmp, value)
        tmp.replace(path)

    for task in affected:
        rows = [read(captures / "case_receipts" / (c["id"] + ".json")) for c in selection["cases"] if c["task"] == task]
        if not all(r["ok"] for r in rows):
            raise ValueError("Task is not completely recovered.")
        replace(
            captures / "task_receipts" / (task + ".json"),
            {
                "task": task,
                "ok": True,
                "cases": rows,
                "selection_sha256": SELECTION_SHA,
                "recovery_verification_sha256": sha(recovery / "verification_published.json"),
                "original_task_receipt": str(recovery / "original_receipts/task_receipts" / (task + ".json")),
            },
        )
    task_status = {
        t["task"]: read(captures / "task_receipts" / (t["task"] + ".json"))["ok"] for t in selection["tasks"]
    }
    if len(task_status) != 100 or not all(task_status.values()):
        raise ValueError("Recovery did not complete every predeclared task.")
    execution = dict(
        original,
        status="complete",
        task_status=task_status,
        recovery={
            "schema": "driveway-capture-recovery-reconciliation/1",
            "verification_sha256": sha(recovery / "verification_published.json"),
            "verification": str(recovery / "verification_published.json"),
            "recovered_case_ids": ids,
            "original_execution": str(recovery / "original_receipts/execution.json"),
            "original_execution_sha256": sha(recovery / "original_receipts/execution.json"),
            "original_task_attempts_preserved": True,
        },
    )
    replace(captures / "execution.json", execution)
    write_new(
        recovery / "publication.json",
        {
            "schema": "driveway-capture-recovery-publication/1",
            "ok": True,
            "verification_sha256": sha(recovery / "verification_published.json"),
            "execution_sha256": sha(captures / "execution.json"),
            "recovered_case_ids": ids,
            "task_receipts": {t: sha(captures / "task_receipts" / (t + ".json")) for t in sorted(affected)},
        },
    )
    return after


def stage_provenance(selection, recovery, stage, sim_data):
    import visionbench_all_tasks_package_v3 as package

    verify(recovery, selection.parent / "captures", published=True)
    package.stage_metadata(selection, stage, sim_data)
    manifest = read(stage / "STAGING_MANIFEST.json")
    files = manifest["files"]
    # Include the complete small recovery chain, raw seven-case source captures, and failed original receipts.
    for path in sorted(recovery.rglob("*")):
        if (
            path.is_file()
            and not path.is_symlink()
            and (path.suffix != ".log")
            and path.name not in ("status.json", "continuation.log")
        ):
            package.add_file(
                stage, files, path, "driveway_recovery/" + str(path.relative_to(recovery)), "driveway_recovery_evidence"
            )
    for name in (
        "visionbench_driveway_recovery.py",
        "visionbench_driveway_recovery_verify.py",
        "visionbench_driveway_recovery_continue.py",
    ):
        package.add_file(
            stage, files, Path(__file__).parent / name, "source_recovery/" + name, "driveway_recovery_source"
        )
    existing_sources = {r["original_path"] for r in files}
    amendment = read(recovery / "amendment.json")
    for index, row in enumerate(amendment["sources"]):
        if row["path"] not in existing_sources:
            package.add_file(
                stage,
                files,
                row["path"],
                "source_recovery/dependency_" + str(index) + "_" + Path(row["path"]).name,
                "recovery_dependency_source",
            )
    prepared = read(recovery / "preparation.json")
    for identifier in prepared["eligible_case_ids"]:
        original = Path(prepared["captures"]) / "_attempt_failures" / identifier / "failure.json"
        package.add_file(
            stage,
            files,
            original,
            "driveway_failure_originals/" + identifier + "/failure.json",
            "original_driveway_failure",
        )
    diagnosis_root = selection.parent / "fidelity_diagnostics/saved_query_outliers_163cases"
    for path in sorted(diagnosis_root.iterdir()):
        if path.is_file() and path.suffix in (".json", ".png"):
            package.add_file(
                stage, files, path, "saved_query_outlier_diagnosis/" + path.name, "preserved_replay_fidelity_diagnosis"
            )
    for name in ("robots/robot.py", "prims/entity_prim.py"):
        source = Path(__file__).resolve().parents[2] / name
        package.add_file(
            stage,
            files,
            source,
            "saved_query_outlier_diagnosis/source/" + Path(name).name,
            "holonomic_pose_representation_source",
        )
    manifest.update(
        driveway_recovery={
            "verification_sha256": sha(recovery / "verification_published.json"),
            "recovery_root": str(recovery),
            "portable_root": "driveway_recovery",
            "verifier_source_sha256": sha(__file__),
        },
        files=files,
    )
    (stage / "STAGING_MANIFEST.json").rename(stage / "STAGING_BEFORE_RECOVERY_MANIFEST.json")
    write_new(stage / "STAGING_MANIFEST.json", manifest)
    return package.verify_files(stage, manifest)


def verify_portable(provenance, main_captures):
    manifest = read(provenance / "ARTIFACT_MANIFEST.json")
    import visionbench_all_tasks_package_v3 as package

    package.verify_files(provenance, manifest)
    mapping = {}
    for row in manifest["files"]:
        if row["original_path"] in mapping and sha(mapping[row["original_path"]]) != row["sha256"]:
            raise ValueError("Conflicting original source mapping.")
        mapping[row["original_path"]] = provenance / row["path"]
    recovery = provenance / manifest["driveway_recovery"]["portable_root"]
    prepared = read(recovery / "preparation.json")
    old_captures = Path(prepared["captures"])
    for row in prepared["protected_files"]:
        path = Path(row["path"])
        try:
            relative = path.relative_to(old_captures)
        except ValueError:
            continue
        if len(relative.parts) == 2 and relative.parts[0].startswith("comprehensive."):
            mapping[row["path"]] = main_captures / relative
    for identifier in prepared["eligible_case_ids"]:
        for name in NATIVE:
            mapping[str(old_captures / identifier / name)] = main_captures / identifier / name
    # Original successful receipts/materializations/snapshots are already in companion manifest.
    return verify(recovery, main_captures, published=True, mapping=mapping)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("verify", "promote", "stage", "verify-portable"))
    p.add_argument("--recovery", type=Path)
    p.add_argument("--captures", type=Path)
    p.add_argument("--published", action="store_true")
    p.add_argument("--selection", type=Path)
    p.add_argument("--stage", type=Path)
    p.add_argument("--sim-data", type=Path)
    p.add_argument("--provenance", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.command == "verify":
        result = verify(a.recovery, a.captures, a.published)
    elif a.command == "promote":
        result = promote(a.recovery, a.captures)
    elif a.command == "stage":
        result = stage_provenance(a.selection, a.recovery, a.stage, a.sim_data)
    else:
        result = verify_portable(a.provenance, a.captures)
    if a.output:
        write_new(a.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
