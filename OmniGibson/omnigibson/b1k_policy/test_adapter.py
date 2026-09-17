"""The adapter and the self-mask, exercised against recorded captures -- no Isaac Sim, no omnigibson import.

Two things are measured here, both against ground truth that came out of the simulator:

* the **proprioception layout** and the **camera order and convention**, against ``omnigibson/eval/r1pro.yaml``
  and against a recorded capture's ``extras`` (which stores the camera pose in both conventions, so the USD ->
  OpenCV turn is not a matter of opinion);
* the **self-mask**, against the privileged ``robot_mask`` the bridge wrote into ``obs.h5`` from the simulator's
  own link poses. That is exactly the thing ``selfmask.py`` exists to replace, so it is exactly the ground truth.

A recorded round is a ``(obs.h5, capture.json)`` pair. It carries no proprioception -- that is the whole reason
this adapter exists -- but it does carry the 11 joints the planner drives (``q_init``: torso 1-4, left arm 1-7)
and the left gripper's two fingers, and the bridge locks the right arm at zero and the right gripper open
(``r1pro_left_meta.yml``). That is a full 22-joint configuration, so a proprioception vector can be SYNTHESISED
from a recording and the self-mask measured on it.

It is only the render-time configuration if nothing moved between the plan request and the render, which is not
always true (the bridge swings the arm out of the head camera's way for some captures, and the locked right arm
sags under gravity as an episode runs). ``_render_time_config`` keeps only the rounds where the synthesised
configuration reproduces all three recorded camera poses to 1 mm and 0.05 degrees -- a proprioception-free check
that the joints really are the rendered ones. Roughly one round in fifteen passes it; there are enough.

Point ``B1K_RUNS`` at a directory of recorded runs (default ``~/projects/BEHAVIOR-1K/runs``) and
``B1K_RECORDED_ROUND`` at a single round, as ``b1k/test_observation.py`` does.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(os.environ.get("B1K_POLICY", Path.home() / "projects/b1k-submission/third-party/tiptop"))))
# Both modules are imported by path, never as ``omnigibson.b1k_policy.*``: importing the omnigibson package pulls
# in the simulator, and the whole point of this pair is that neither of them needs it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b1k.observation import (  # noqa: E402
    CAM_REL_POSE_ORDER,
    CAMERA_INTRINSICS,
    PROPRIO_DIM,
    PROPRIO_SLICES,
    CameraView,
    SensorObservation,
    load_oracle_observation,
    pose_to_matrix,
)

from adapter import OPENCV_FROM_USD, ObservationAdapter, camera_sensor_names, check_proprio_layout  # noqa: E402
from b1k.selfmask import ARM_LINKS, R1ProSelfMask, default_urdf_path  # noqa: E402

RUNS = Path(os.environ.get("B1K_RUNS", Path.home() / "projects/BEHAVIOR-1K/runs"))
RECORDED_ROUND = Path(
    os.environ.get(
        "B1K_RECORDED_ROUND",
        RUNS / "queue_putting_dirty_dishes_in_sink_0916_211033/putting_dirty_dishes_in_sink_304_0/r01_left_holding",
    )
)

# The bridge's planned joints (r1pro_left_meta.yml: joint_names) and the posture it locks the rest at.
PLANNED_JOINTS = ["torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"] + [
    f"left_arm_joint{i}" for i in range(1, 8)
]
LOCKED = {f"right_arm_joint{i}": 0.0 for i in range(1, 8)} | {
    "right_gripper_finger_joint1": 0.05,
    "right_gripper_finger_joint2": 0.05,
}
# The camera sensor sits at its wrist link's origin, turned 90 degrees about the link's z; the head camera sits
# 6 cm off zed_link. Constants of the robot, measured once from a recording whose configuration is exact and
# re-checked by test_fk_reproduces_the_recorded_camera_poses on every round the gate accepts.
CAMERA_LINK = {"head": "zed_link", "left_wrist": "left_realsense_link", "right_wrist": "right_realsense_link"}

MASK_LINKS = {
    "head": ARM_LINKS["left"] + ARM_LINKS["right"],
    "left_wrist": ARM_LINKS["left"],
    "right_wrist": ARM_LINKS["right"],
}


# ------------------------------------------------------------------ recorded-capture helpers


def view_extras(capture, name):
    return capture["extras"] if name == "head" else capture["extras"]["views"][name]


def synthetic_proprio(capture) -> np.ndarray:
    """A 61-vector holding the joints a recorded capture pins down; every other field is zero.

    Only the fields ``selfmask`` reads are meaningful: trunk_qpos, arm_left_qpos, arm_right_qpos and both
    gripper_*_qpos. Velocities and eef poses are zero, which is honest -- a recording does not have them.
    """
    q = dict(zip(PLANNED_JOINTS, capture["q_init"])) | LOCKED
    fingers = capture["extras"].get("q_fingers") or [0.05, 0.05]
    q["left_gripper_finger_joint1"], q["left_gripper_finger_joint2"] = fingers
    proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
    proprio[PROPRIO_SLICES["trunk_qpos"]] = [q[f"torso_joint{i}"] for i in range(1, 5)]
    for side in ("left", "right"):
        proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = [q[f"{side}_arm_joint{i}"] for i in range(1, 8)]
        proprio[PROPRIO_SLICES[f"gripper_{side}_qpos"]] = [q[f"{side}_gripper_finger_joint{i}"] for i in (1, 2)]
    return proprio


def camera_offsets(mask: R1ProSelfMask, capture) -> dict:
    """``{view: (4, 4) link_from_sensor}`` measured on one round whose configuration is known to be exact."""
    poses = mask.link_poses(synthetic_proprio(capture))
    out = {}
    for view, link in CAMERA_LINK.items():
        extras = view_extras(capture, view)
        w, x, y, z = extras["cam_quat_wxyz_ros"]
        out[view] = np.linalg.inv(poses[link]) @ pose_to_matrix(extras["cam_pos_base"], [x, y, z, w]).astype(np.float64)
    return out


def camera_pose_errors(mask: R1ProSelfMask, capture, offsets) -> dict:
    """``{view: (metres, radians)}`` between the recorded camera pose and the one FK predicts."""
    poses = mask.link_poses(synthetic_proprio(capture))
    errors = {}
    for view, link in CAMERA_LINK.items():
        extras = capture["extras"] if view == "head" else capture["extras"].get("views", {}).get(view)
        if extras is None:
            continue
        w, x, y, z = extras["cam_quat_wxyz_ros"]
        recorded = pose_to_matrix(extras["cam_pos_base"], [x, y, z, w]).astype(np.float64)
        predicted = poses[link] @ offsets[view]
        turn = predicted[:3, :3].T @ recorded[:3, :3]
        angle = np.arccos(np.clip((np.trace(turn) - 1) / 2, -1, 1))
        errors[view] = (float(np.linalg.norm(predicted[:3, 3] - recorded[:3, 3])), float(angle))
    return errors


def recorded_rounds(limit=None):
    for queue in sorted(p for p in RUNS.glob("queue_*") if p.is_dir()):
        for episode in sorted(p for p in queue.iterdir() if p.is_dir()):
            for round_dir in sorted(p for p in episode.iterdir() if p.is_dir()):
                if (round_dir / "obs.h5").exists() and (round_dir / "capture.json").exists():
                    yield round_dir
                    if limit is not None:
                        limit -= 1
                        if limit <= 0:
                            return


@pytest.fixture(scope="module")
def mask():
    pytest.importorskip("trimesh")
    try:
        default_urdf_path()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return R1ProSelfMask(dilate=0)


@pytest.fixture(scope="module")
def round_dir():
    if not (RECORDED_ROUND / "obs.h5").exists():
        pytest.skip(f"no recorded capture at {RECORDED_ROUND}; set B1K_RECORDED_ROUND")
    pytest.importorskip("h5py")
    return RECORDED_ROUND


@pytest.fixture(scope="module")
def capture(round_dir):
    with open(round_dir / "capture.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def offsets(mask, capture):
    return camera_offsets(mask, capture)


@pytest.fixture(scope="module")
def gated_rounds(mask, offsets):
    """Recorded rounds whose synthesised configuration is the one that was rendered, and their proprio."""
    kept = []
    for round_dir in recorded_rounds():
        with open(round_dir / "capture.json") as f:
            recorded = json.load(f)
        if len(recorded.get("q_init", [])) != len(PLANNED_JOINTS):
            continue
        errors = camera_pose_errors(mask, recorded, offsets)
        if len(errors) == 3 and all(p < 1e-3 and a < np.deg2rad(0.05) for p, a in errors.values()):
            kept.append((round_dir, synthetic_proprio(recorded)))
        if len(kept) >= 40:
            break
    if not kept:
        pytest.skip(f"no recorded round under {RUNS} whose configuration is the rendered one")
    return kept


# ------------------------------------------------------------------ what the evaluator hands over


def test_proprio_layout_comes_out_of_the_challenge_config():
    """r1pro.yaml's proprio_obs list, in its own order, must tile the 61 numbers exactly as PROPRIO_SLICES says."""
    layout = check_proprio_layout()
    assert layout == dict(PROPRIO_SLICES)
    assert max(s.stop for s in layout.values()) == PROPRIO_DIM == 61


def test_camera_order_is_the_config_order_not_alphabetical():
    names = camera_sensor_names()
    assert tuple(names) == CAM_REL_POSE_ORDER == ("left_wrist", "right_wrist", "head")
    assert names["head"].endswith("zed_link:Camera:0")


def test_cam_rel_poses_are_usd_axes_and_are_turned_into_opencv(capture):
    """The evaluator reads the sensor pose straight off the prim, so cam_rel_poses is USD (-z forward). A recorded
    capture stores the same camera in BOTH conventions, which settles it: they differ by 180 degrees about x."""
    for view in CAM_REL_POSE_ORDER:
        extras = view_extras(capture, view)
        usd_world = pose_to_matrix(extras["cam_pos_world"], extras["cam_quat_xyzw_world_usd"])
        cv_world = pose_to_matrix(extras["cam_pos_world"], extras["cam_quat_xyzw_world_cv"])
        np.testing.assert_allclose(usd_world @ OPENCV_FROM_USD, cv_world, atol=1e-5)


def test_adapter_rebuilds_a_recorded_capture_from_an_evaluator_shaped_observation(round_dir, capture):
    """Feed the adapter exactly what the evaluator would emit for this recorded instant and get the recording back:
    same rgb, same depth, same base_from_cam per view, and the challenge's intrinsics."""
    recorded = load_oracle_observation(round_dir)
    flat = {"task_id": np.array([17], dtype=np.int64), "robot_r1::proprio": synthetic_proprio(capture)}
    poses = []
    for view in CAM_REL_POSE_ORDER:
        extras = view_extras(capture, view)
        w, x, y, z = extras["cam_quat_wxyz_ros"]
        usd = pose_to_matrix(extras["cam_pos_base"], [x, y, z, w]) @ OPENCV_FROM_USD  # undo the turn the adapter does
        poses.append(np.concatenate([usd[:3, 3], _matrix_to_quat_xyzw(usd[:3, :3])]))
    flat["robot_r1::cam_rel_poses"] = np.concatenate(poses)
    sensors = camera_sensor_names()
    for view, sensor in sensors.items():
        rgba = np.dstack([recorded.views[view].rgb, np.full(recorded.views[view].resolution + (1,), 255, np.uint8)])
        flat[f"robot_r1::{sensor}::rgb"] = rgba
        flat[f"robot_r1::{sensor}::depth_linear"] = recorded.views[view].depth

    obs = ObservationAdapter()(flat)
    assert type(obs) is SensorObservation and obs.task_id == 17 and obs.report()["privileged"] is False
    assert set(obs.views) == set(CAM_REL_POSE_ORDER)
    for view, built in obs.views.items():
        np.testing.assert_array_equal(built.rgb, recorded.views[view].rgb)  # alpha dropped
        np.testing.assert_array_equal(built.depth, recorded.views[view].depth)
        np.testing.assert_allclose(built.base_from_cam, recorded.views[view].base_from_cam, atol=1e-5)
        np.testing.assert_allclose(built.intrinsics, CAMERA_INTRINSICS[view], atol=1e-3)


def test_ray_length_depth_is_refused_by_name(capture):
    """The modality called 'depth' is ray length. Taking it for depth_linear is the silent frame bug, so it is an
    error that names the modality rather than a fallback."""
    flat = {"task_id": np.array([0]), "robot_r1::proprio": synthetic_proprio(capture)}
    flat["robot_r1::cam_rel_poses"] = np.zeros(21, dtype=np.float32)
    flat["robot_r1::cam_rel_poses"][3::7] = 1.0  # unit quaternions
    sensor = camera_sensor_names()["head"]
    flat[f"robot_r1::{sensor}::rgb"] = np.zeros((8, 8, 4), np.uint8)
    flat[f"robot_r1::{sensor}::depth"] = np.ones((8, 8), np.float32)
    with pytest.raises(KeyError, match="RAY LENGTH"):
        ObservationAdapter()(flat)


def test_wrong_resolution_is_refused_because_the_intrinsics_are_for_one(capture):
    """CAMERA_INTRINSICS is 306 px at 720 px wide and 388.66 at 480. Under the low-res eval wrapper the same
    numbers would be 3x wrong and nothing downstream would notice, so the adapter checks the resolution."""
    flat = {"task_id": np.array([0]), "robot_r1::proprio": synthetic_proprio(capture)}
    flat["robot_r1::cam_rel_poses"] = np.zeros(21, dtype=np.float32)
    flat["robot_r1::cam_rel_poses"][3::7] = 1.0
    sensor = camera_sensor_names()["head"]
    flat[f"robot_r1::{sensor}::rgb"] = np.zeros((224, 224, 4), np.uint8)
    flat[f"robot_r1::{sensor}::depth_linear"] = np.ones((224, 224), np.float32)
    with pytest.raises(ValueError, match="focal"):
        ObservationAdapter()(flat)
    assert ObservationAdapter(intrinsics={"head": np.eye(3, dtype=np.float32)})(flat).views["head"].resolution == (
        224,
        224,
    )


def test_missing_proprio_is_named(capture):
    with pytest.raises(KeyError, match="proprio"):
        ObservationAdapter().proprio({"task_id": np.array([0])})


# ------------------------------------------------------------------ forward kinematics


def test_fk_reproduces_the_recorded_camera_poses(mask, offsets, gated_rounds):
    """The URDF's root link IS the robot base frame the challenge's poses are in. If it were not, or if the joint
    order were wrong, the cameras FK puts in the base frame would not land on the recorded ones."""
    for round_dir, proprio in gated_rounds:
        with open(round_dir / "capture.json") as f:
            recorded = json.load(f)
        for view, (pos, angle) in camera_pose_errors(mask, recorded, offsets).items():
            assert pos < 1e-3 and angle < np.deg2rad(0.05), f"{round_dir}/{view}: {pos * 1000:.3f} mm"


def test_fk_joint_values_come_from_the_proprio_slices(mask):
    proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
    proprio[PROPRIO_SLICES["arm_left_qpos"]] = np.arange(1, 8) * 0.1
    proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.02, 0.03]
    q = mask.joint_values(proprio)
    np.testing.assert_allclose([q[f"left_arm_joint{i}"] for i in range(1, 8)], np.arange(1, 8) * 0.1)
    assert q["right_gripper_finger_joint1"] == pytest.approx(0.02)
    assert q["torso_joint1"] == 0.0
    with pytest.raises(ValueError, match="finite"):
        mask.joint_values(np.full(PROPRIO_DIM, np.nan))


# ------------------------------------------------------------------ the self-mask, against the privileged one


def mask_scores(mask, round_dir, proprio):
    """Per view: the self-mask against the bridge's privileged robot_mask, over the same link set it used."""
    import h5py

    rows = {}
    with h5py.File(round_dir / "obs.h5", "r") as f:
        groups = {"head": f} | {name: f["views"][name] for name in f.get("views", {})}
        for view, group in groups.items():
            if "robot_mask" not in group or view not in MASK_LINKS:
                continue
            truth = np.asarray(group["robot_mask"][:]).astype(bool)
            if not truth.any():
                continue
            depth = np.asarray(group["depth"][:], dtype=np.float32)
            camera = CameraView(
                name=view,
                rgb=np.asarray(group["rgb"][:]).astype(np.uint8)[..., :3],
                depth=depth[..., 0] if depth.ndim == 3 else depth,
                intrinsics=np.asarray(group["intrinsic_matrix"][:], dtype=np.float32),
                base_from_cam=np.asarray((f if view == "head" else group)["world_from_cam"][:], dtype=np.float32),
            )
            # The bridge ZEROED the depth at its own mask before writing, so the depth test cannot be exercised
            # here: compare the silhouette, which is the part FK and the meshes decide.
            mine = np.isfinite(mask.rendered_depth(camera, proprio, links=MASK_LINKS[view]))
            rows[view] = dict(
                iou=(mine & truth).sum() / (mine | truth).sum(),
                recall=(mine & truth).sum() / truth.sum(),
                precision=(mine & truth).sum() / max(mine.sum(), 1),
                truth=int(truth.sum()),
            )
    return rows


def test_self_mask_matches_the_privileged_robot_mask(mask, gated_rounds):
    """The honest mask against the simulator's own. Measured over the rounds whose configuration is provably the
    rendered one: median IoU above 0.95 per view and recall above 0.95 on average, with the disagreement a band a
    pixel or two wide along the silhouette."""
    import collections

    scores = collections.defaultdict(list)
    for round_dir, proprio in gated_rounds[:20]:
        for view, row in mask_scores(mask, round_dir, proprio).items():
            scores[view].append(row)
    assert set(scores) >= {"head", "left_wrist", "right_wrist"}, f"only {sorted(scores)} compared"
    for view, rows in scores.items():
        median_iou = float(np.median([r["iou"] for r in rows]))
        recall = float(np.mean([r["recall"] for r in rows]))
        assert median_iou > 0.95, f"{view}: median IoU {median_iou:.4f} over {len(rows)} views"
        assert recall > 0.95, f"{view}: mean recall {recall:.4f} over {len(rows)} views"


def test_self_mask_keeps_what_is_in_front_of_the_robot(mask, gated_rounds):
    """A pixel the robot covers whose depth is well in FRONT of the robot's surface is something else -- a held
    object, a hand-over -- and must survive. Only the depth rule can do that; a bare silhouette cannot."""
    round_dir, proprio = gated_rounds[0]
    import h5py

    with h5py.File(round_dir / "obs.h5", "r") as f:
        depth = np.asarray(f["depth"][:], dtype=np.float32)
        view = CameraView(
            name="head",
            rgb=np.asarray(f["rgb"][:]).astype(np.uint8)[..., :3],
            depth=depth[..., 0] if depth.ndim == 3 else depth,
            intrinsics=np.asarray(f["intrinsic_matrix"][:], dtype=np.float32),
            base_from_cam=np.asarray(f["world_from_cam"][:], dtype=np.float32),
        )
    robot_z = mask.rendered_depth(view, proprio)
    covered = np.isfinite(robot_z)
    assert covered.any()
    in_front = np.where(covered, np.maximum(robot_z - 0.2, 0.01), 0.0).astype(np.float32)
    nearer = CameraView(
        name="head", rgb=view.rgb, depth=in_front, intrinsics=view.intrinsics, base_from_cam=view.base_from_cam
    )
    assert not mask.mask(nearer, proprio)[covered].any()
    behind = np.where(covered, robot_z.astype(np.float32), 0.0).astype(np.float32)
    at_surface = CameraView(
        name="head", rgb=view.rgb, depth=behind, intrinsics=view.intrinsics, base_from_cam=view.base_from_cam
    )
    assert mask.mask(at_surface, proprio)[covered].all()


def test_filtered_view_zeroes_only_the_robot(mask, gated_rounds):
    round_dir, proprio = gated_rounds[0]
    recorded = load_oracle_observation(round_dir)
    view = recorded.views["left_wrist"]
    filtered = mask.filtered(view, proprio)
    assert filtered.name == view.name and filtered.depth.dtype == np.float32
    zeroed = (filtered.depth == 0) & (view.depth != 0)
    np.testing.assert_array_equal(filtered.depth[~zeroed], view.depth[~zeroed])
    assert view.depth[view.depth > 0].size == 0 or filtered.unproject().shape[0] <= view.unproject().shape[0]


def test_self_mask_reads_no_privileged_field():
    """The one property that makes it honest: outside its prose, the source has no seg_instance, no simulator link
    state and no simulator import."""
    import ast

    import b1k.selfmask

    tree = ast.parse(Path(b1k.selfmask.__file__).read_text())
    for node in ast.walk(tree):  # drop every docstring; the module explains what it does NOT read
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            root = (getattr(node, "module", None) or node.names[0].name).split(".")[0]
            assert root not in ("omnigibson", "omni", "isaacsim", "carb", "pxr", "torch"), f"selfmask imports {root}"
    code = ast.unparse(tree)
    for forbidden in ("seg_instance", "get_position_orientation", "robot.links", "og.sim"):
        assert forbidden not in code, f"selfmask.py reads {forbidden}"


def _matrix_to_quat_xyzw(rot) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(rot, dtype=np.float64)).as_quat().astype(np.float32)
