"""CPU failure-chain, scope, pose and publication checks; no simulator initialization."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent))

import visionbench_driveway_recovery as recovery
import visionbench_driveway_recovery_verify as verifier


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) + "\n")


def test_scope_preserves_all_existing_bindings_and_adds_only_fixed_driveway():
    obj = SimpleNamespace(category="driveway", fixed_base=True)
    table, floor = object(), object()
    old = {"table.n.02_1": table}
    raw = dict(old, **{"driveway.n.01_1": obj, "floor.n.01_1": floor})
    result = recovery.extend_scope(old, raw, [table, obj, floor], ["driveway.n.01_1"])
    assert result == {"table.n.02_1": table, "driveway.n.01_1": obj}
    assert old == {"table.n.02_1": table}


@pytest.mark.parametrize(
    "category,fixed,loaded,names",
    [
        ("floor", True, True, ["driveway.n.01_1"]),
        ("driveway", False, True, ["driveway.n.01_1"]),
        ("driveway", True, False, ["driveway.n.01_1"]),
        ("driveway", True, True, ["floor.n.01_1"]),
        ("driveway", True, True, []),
    ],
)
def test_scope_rejects_unrelated_dynamic_or_unloaded_objects(category, fixed, loaded, names):
    obj = SimpleNamespace(category=category, fixed_base=fixed)
    with pytest.raises(ValueError):
        recovery.extend_scope({}, {"driveway.n.01_1": obj}, [obj] if loaded else [], names)


def test_restore_checks_every_saved_pose_and_joint_and_quaternion_sign():
    snapshot = {
        "robot_r1": {"root_link": {"pos": [1, 2, 3], "ori": [0, 0, 0, 1]}, "joint_pos": [0.3]},
        "driveway": {"root_link": {"pos": [0, 0, 0], "ori": [0, 0, 0, 1]}},
    }
    states = copy.deepcopy(snapshot)
    states["driveway"]["root_link"]["ori"] = [0, 0, 0, -1]
    objects = {name: SimpleNamespace(dump_state=lambda serialized=False, s=s: s) for name, s in states.items()}
    sim = SimpleNamespace(
        env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda key, name: objects.get(name)))
    )
    audit = recovery.restore_state_audit(sim, snapshot)
    assert audit["objects_checked"] == 2 and len(audit["components"]) == 5 and audit["maximum_absolute_error"] == 0
    states["robot_r1"]["joint_pos"] = [0.31]
    with pytest.raises(ValueError, match="restoration failed"):
        recovery.restore_state_audit(sim, snapshot)


@pytest.fixture
def failure_case(tmp_path):
    captures = tmp_path / "captures"
    identifier = recovery.ALLOWED_IDS[0]
    case = {
        "id": identifier,
        "task": "chopping_wood",
        "snapshot": str(tmp_path / "snapshot.json"),
        "target_ids": ["driveway.n.01_1"],
        "frame": 429,
        "episode_index": 8884,
        "category_prompts": {"driveway": "driveway"},
    }
    write(Path(case["snapshot"]), {"robot_r1": {}})
    actual = dict(case, snapshot_sha256=recovery.sha(case["snapshot"]), fidelity={"frozen": "unchanged"})
    failure = {
        "id": identifier,
        "primary_id": identifier,
        "ok": False,
        "error": recovery.ERROR,
        "selection_sha256": recovery.SELECTION_SHA,
    }
    write(captures / "_attempt_failures" / identifier / "failure.json", failure)
    write(
        captures / "task_receipts/chopping_wood.json",
        {
            "ok": False,
            "selection_sha256": recovery.SELECTION_SHA,
            "cases": [{"primary_id": identifier, "ok": False, "attempts": [failure]}],
        },
    )
    write(
        captures / "materialization" / (identifier + ".json"),
        {"selection_sha256": recovery.SELECTION_SHA, "case": actual},
    )
    return case, captures


def test_exact_failure_chain_is_eligible(failure_case):
    case, captures = failure_case
    actual, proof = recovery.validate_failure(case, captures)
    assert actual["frame"] == 429 and len(proof) == 4


@pytest.mark.parametrize(
    "change", ["frame", "snapshot", "different_error", "task_success", "existing_success", "other_id"]
)
def test_failure_chain_rejects_substitution_and_existing_success(failure_case, change):
    case, captures = failure_case
    identifier = case["id"]
    if change == "frame":
        case["frame"] = 430
    elif change == "snapshot":
        Path(case["snapshot"]).write_text("{}")
    elif change == "different_error":
        p = captures / "_attempt_failures" / identifier / "failure.json"
        v = recovery.read(p)
        v["error"] = "CUDA out of memory"
        write(p, v)
    elif change == "task_success":
        p = captures / "task_receipts/chopping_wood.json"
        v = recovery.read(p)
        v["ok"] = True
        write(p, v)
    elif change == "existing_success":
        write(captures / "case_receipts" / (identifier + ".json"), {"ok": True})
    else:
        case["id"] = "not-a-declared-case"
    with pytest.raises(ValueError):
        recovery.validate_failure(case, captures)


def test_publication_never_sets_complete_before_postcopy_verification(tmp_path, monkeypatch):
    captures = tmp_path / "captures"
    root = tmp_path / "recovery"
    identifier = recovery.ALLOWED_IDS[0]
    write(root / "preparation.json", {"eligible_case_ids": [identifier]})
    write(root / "captures" / identifier / "input.json", {"saved": True})
    write(root / "captures/case_receipts" / (identifier + ".json"), {"ok": True})
    write(captures / "execution.json", {"status": "incomplete"})
    (captures / "case_receipts").mkdir()
    calls = []

    def check(root, cap, published=False):
        calls.append(published)
        assert verifier.read(captures / "execution.json")["status"] == "incomplete"
        if published:
            raise ValueError("deliberately fail independent validation")
        return {"ok": True}

    monkeypatch.setattr(verifier, "verify", check)
    with pytest.raises(ValueError, match="independent validation"):
        verifier.promote(root, captures)
    assert calls == [False, True]
    assert verifier.read(captures / "execution.json")["status"] == "incomplete"
    assert not (root / "publication.json").exists()
    assert (captures / identifier / "input.json").is_file()


def test_publication_refuses_existing_target(tmp_path, monkeypatch):
    root = tmp_path / "recovery"
    captures = tmp_path / "captures"
    identifier = recovery.ALLOWED_IDS[0]
    write(root / "preparation.json", {"eligible_case_ids": [identifier]})
    (captures / identifier).mkdir(parents=True)
    monkeypatch.setattr(verifier, "verify", lambda *a, **k: {"ok": True})
    with pytest.raises(FileExistsError):
        verifier.promote(root, captures)


@pytest.fixture
def verified_capture(tmp_path, monkeypatch):
    import shutil

    base = tmp_path / "original"
    root = base / "recovery"
    captures = base / "captures"
    identifier = recovery.ALLOWED_IDS[0]
    native = root / "captures" / identifier
    source = root / "_source_captures" / identifier
    snapshot = base / "snapshots" / (identifier + ".json")
    state = {
        "robot_r1": {"root_link": {"pos": [0, 0, 0], "ori": [0, 0, 0, 1]}},
        "driveway_asset": {"root_link": {"pos": [0, 0, 0], "ori": [0, 0, 0, 1]}},
    }
    write(snapshot, state)
    case = {
        "id": identifier,
        "task": "chopping_wood",
        "episode_index": 8884,
        "frame": 429,
        "snapshot": str(snapshot),
        "snapshot_sha256": None,
        "target_ids": ["driveway.n.01_1"],
        "category_prompts": {"driveway": "driveway"},
        "cohort": "comprehensive_test",
        "split": "test",
    }
    selection = base / "selection.json"
    write(selection, {"cases": [case]})
    selection_sha = recovery.sha(selection)
    monkeypatch.setattr(verifier, "SELECTION_SHA", selection_sha)
    materialized = dict(case, snapshot_sha256=recovery.sha(snapshot), fidelity={"unchanged": True})
    materialization = captures / "materialization" / (identifier + ".json")
    write(materialization, {"selection_sha256": selection_sha, "case": materialized})
    failure = {
        "id": identifier,
        "primary_id": identifier,
        "ok": False,
        "error": verifier.ERROR,
        "selection_sha256": selection_sha,
    }
    failure_path = captures / "_attempt_failures" / identifier / "failure.json"
    write(failure_path, failure)
    task_path = captures / "task_receipts/chopping_wood.json"
    write(task_path, {"ok": False, "cases": [{"primary_id": identifier, "attempts": [failure]}]})
    write(captures / "execution.json", {"status": "incomplete"})
    original = []
    for path in (task_path, captures / "execution.json"):
        destination = root / "original_receipts" / path.relative_to(captures)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        original.append({"original": recovery.binding(path), "preserved": recovery.binding(destination)})
    amendment = {
        "schema": "driveway-capture-recovery-amendment/1",
        "selection": recovery.binding(selection),
        "allowed_case_ids": list(verifier.ALLOWED),
        "sources": [],
        "new_replay_frames": 0,
        "settle_steps": 0,
        "predictions_read": False,
        "case_selection_changed": False,
        "primary_geometry_algorithm_changed": False,
        "successful_captures_changed": False,
        "model_or_prompt_or_threshold_changed": False,
        "no_replacements": True,
    }
    write(root / "amendment.json", amendment)
    protected = base / "successful_original.txt"
    protected.write_text("unchanged successful case")
    write(
        root / "preparation.json",
        {
            "amendment": recovery.binding(root / "amendment.json"),
            "eligible_case_ids": [identifier],
            "protected_files": [recovery.binding(protected)],
            "original_receipts": original,
            "captures": str(captures),
        },
    )
    metadata = {
        "id": identifier,
        "task": case["task"],
        "episode_index": 8884,
        "frame": 429,
        "source_snapshot_sha256": materialized["snapshot_sha256"],
        "source_materialization_hold_steps": 0,
        "capture_extra_hold_steps": 0,
        "physics_steps_during_capture": 0,
        "categories": ["driveway"],
        "prompts": ["driveway"],
    }
    write(source / "input.json", metadata)
    write(source / "labels.json", {"source_only": True})
    arrays = {v + "_rgb": np.zeros((2, 2, 3), dtype=np.uint8) for v in verifier.VIEWS}
    masks = {v + "_masks": np.ones((1, 2, 2), dtype=bool) for v in verifier.VIEWS}
    np.savez(source / "input.npz", **arrays)
    np.savez(source / "labels.npz", **masks)
    for view in verifier.VIEWS:
        (source / (view + ".png")).write_bytes(b"fixed source image")
    errors = {n + "/root_link/" + k: 0.0 for n in state for k in ("pos", "ori")}
    restoration = {"tolerance": 1e-5, "objects_checked": 2, "components": errors, "maximum_absolute_error": 0.0}
    evidence = {
        "schema": "driveway-capture-recovery-case/1",
        "case_id": identifier,
        "amendment": recovery.binding(root / "amendment.json"),
        "inputs": [recovery.binding(p) for p in (failure_path, task_path, materialization, snapshot)],
        "new_replay_frames": 0,
        "settle_steps": 0,
        "physics_steps_after_restore": 0,
        "sim_time_after_restore_s": 0,
        "snapshot_restore_before_capture": restoration,
        "snapshot_restore_after_capture": restoration,
        "binding_evidence": [
            {
                "task_id": "driveway.n.01_1",
                "scene_name": "driveway_asset",
                "asset_category": "driveway",
                "fixed_base": True,
            }
        ],
        "source_outputs": [recovery.binding(p) for p in source.iterdir()],
    }
    evidence_path = root / "case_provenance" / (identifier + ".json")
    write(evidence_path, evidence)
    native.mkdir(parents=True)
    for name in ("input.json", "input.npz", "head.png", "left_wrist.png", "right_wrist.png"):
        shutil.copyfile(source / name, native / name)
    for name in ("labels.npz", "labels_strict4mm.npz"):
        np.savez(native / name, **masks)
    labels = {
        "capture_recovery": recovery.binding(evidence_path),
        "physics_steps_during_relabel": 0,
        "renders_during_relabel": 0,
        "cohort": case["cohort"],
        "split": "test",
        "geometry_tolerance_m": 0.008,
        "strict_geometry_tolerance_m": 0.004,
        "asset_identity_is_evaluation_only": True,
        "mask_provenance": {"renderer_instance_ground_truth": False},
        "source_file_sha256": {
            n: recovery.sha(source / n) for n in ("input.json", "input.npz", "labels.json", "labels.npz")
        },
        "objects": [
            {
                "task_ids": ["driveway.n.01_1"],
                "target": True,
                "task_object": True,
                "scene_name": "driveway_asset",
                "visible_pixels": dict.fromkeys(verifier.VIEWS, 4),
            }
        ],
    }
    write(native / "labels.json", labels)
    write(
        root / "captures/case_receipts" / (identifier + ".json"),
        {
            "ok": True,
            "effective_id": identifier,
            "primary_id": identifier,
            "attempts": [],
            "replay_frames": 0,
            "original_replay_frames": 429,
            "selection_sha256": selection_sha,
            "capture_recovery": recovery.binding(evidence_path),
        },
    )
    return base, root, captures, identifier


def test_independent_verifier_accepts_complete_chain(verified_capture):
    _, root, captures, identifier = verified_capture
    result = verifier.verify(root, captures)
    assert result["ok"] and result["recovered_case_ids"] == [identifier] and result["protected_files_verified"] == 1


@pytest.mark.parametrize(
    "mutation", ["mask_count", "missing_target", "new_physics", "different_prompt", "protected_bytes"]
)
def test_independent_verifier_rejects_corruption(verified_capture, mutation):
    base, root, captures, identifier = verified_capture
    native = root / "captures" / identifier
    if mutation in ("mask_count", "missing_target"):
        p = native / "labels.json"
        obj = verifier.read(p)
        if mutation == "mask_count":
            obj["objects"][0]["visible_pixels"]["head"] = 3
        else:
            obj["objects"][0]["target"] = False
        write(p, obj)
    elif mutation == "different_prompt":
        p = native / "input.json"
        obj = verifier.read(p)
        obj["categories"] = ["floor"]
        write(p, obj)
    elif mutation == "new_physics":
        p = root / "case_provenance" / (identifier + ".json")
        obj = verifier.read(p)
        obj["physics_steps_after_restore"] = 1
        write(p, obj)
    else:
        (base / "successful_original.txt").write_text("changed")
    with pytest.raises(ValueError):
        verifier.verify(root, captures)


def publish_fixture(root, captures, identifier):
    import shutil

    shutil.copytree(root / "captures" / identifier, captures / identifier)
    (captures / "case_receipts").mkdir()
    shutil.copyfile(
        root / "captures/case_receipts" / (identifier + ".json"), captures / "case_receipts" / (identifier + ".json")
    )
    report = verifier.verify(root, captures, published=True)
    write(root / "verification_published.json", report)


def test_published_inventory_cannot_be_weakened(verified_capture):
    _, root, captures, identifier = verified_capture
    publish_fixture(root, captures, identifier)
    p = root / "preparation.json"
    obj = verifier.read(p)
    obj["protected_files"] = []
    write(p, obj)
    with pytest.raises(ValueError, match="source or amendment changed"):
        verifier.verify(root, captures, published=True)


def test_strict_relocation_has_no_original_filesystem_dependency(verified_capture, tmp_path):
    import shutil

    base, root, captures, identifier = verified_capture
    publish_fixture(root, captures, identifier)
    moved = tmp_path / "relocated"
    shutil.copytree(base, moved)
    mapping = {str(p): moved / p.relative_to(base) for p in base.rglob("*") if p.is_file()}
    shutil.rmtree(base)
    result = verifier.verify(moved / "recovery", moved / "captures", published=True, mapping=mapping)
    assert result["ok"] and result["recovered_case_ids"] == [identifier]


def test_continuation_handles_supervisor_exit_race(monkeypatch):
    import visionbench_driveway_recovery_continue as continuation

    def disappeared(path):
        raise FileNotFoundError("process exited")

    monkeypatch.setattr(Path, "read_text", disappeared)
    assert continuation.process_present(617317) is False


def test_gpu_readiness_rejects_any_compute_owner_or_low_memory(monkeypatch):
    import visionbench_driveway_recovery_continue as continuation

    state = {"compute": "", "free": 45000}

    def output(command, text):
        if "--query-gpu=index,uuid,memory.free" in command:
            return f"1, uuid1, {state['free']}\n3, uuid3, 45000\n"
        return state["compute"]

    monkeypatch.setattr(continuation.subprocess, "check_output", output)
    assert continuation.gpu_readiness([1, 3], 40000)[0]
    state["compute"] = "uuid3, 123456\n"
    assert not continuation.gpu_readiness([1, 3], 40000)[0]
    state["compute"] = ""
    state["free"] = 39000
    assert not continuation.gpu_readiness([1, 3], 40000)[0]
