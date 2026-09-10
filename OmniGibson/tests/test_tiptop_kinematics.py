"""The wrist-camera look pose: the pure transform helpers, and Lula IK on the R1Pro URDF (skipped without them)."""

from pathlib import Path

import numpy as np
import pytest

from omnigibson.tiptop.kinematics import (
    ArmIK,
    link_from_camera,
    link_pose_for_camera,
    look_at_quat_xyzw,
    look_pose,
    matrix_pose,
    pose_matrix,
)

URDF = Path(__file__).resolve().parents[2] / "datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"
LEFT_ARM = [f"left_arm_joint{i}" for i in range(1, 8)]
Q_HOME = [-1.6312, 0.2636, -1.812, -1.4576, -0.0508, -0.3727, -1.3193]  # r1pro_left's ready posture
FIXED = {f"torso_joint{i}": v for i, v in zip(range(1, 5), [1.025, -1.45, -0.47, 0.0])}
FIXED.update({f"right_arm_joint{i}": 0.0 for i in range(1, 8)})
FIXED.update({f"{side}_gripper_finger_joint{i}": 0.05 for side in ("left", "right") for i in (1, 2)})


def test_look_at_points_the_camera_minus_z_at_the_target():
    quat = look_at_quat_xyzw((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    rot = pose_matrix((0, 0, 1), quat)[:3, :3]
    view = -rot[:, 2]  # the camera's -z axis
    assert np.allclose(view, np.array([1.0, 0.0, -1.0]) / np.sqrt(2))
    assert rot[:, 1] @ np.array([0.0, 0.0, 1.0]) > 0  # +y stays up
    straight_down = look_at_quat_xyzw((0.5, 0.0, 1.0), (0.5, 0.0, 0.0))
    assert np.allclose(-pose_matrix((0, 0, 0), straight_down)[:3, 2], [0.0, 0.0, -1.0])


def test_camera_in_link_composition_round_trips():
    link_pos, link_quat = np.array([0.5, 0.1, 1.0]), look_at_quat_xyzw((0.5, 0.1, 1.0), (1.0, 0.0, 0.5))
    cam_pos, cam_quat = np.array([0.55, 0.1, 1.02]), look_at_quat_xyzw((0.55, 0.1, 1.02), (1.2, -0.1, 0.4))
    link_from_cam = link_from_camera(link_pos, link_quat, cam_pos, cam_quat)
    pos, quat = link_pose_for_camera(cam_pos, cam_quat, link_from_cam)
    assert np.allclose(pos, link_pos, atol=1e-9)
    assert np.allclose(pose_matrix(pos, quat), pose_matrix(link_pos, link_quat), atol=1e-9)
    p, q = matrix_pose(pose_matrix(link_pos, link_quat))
    assert np.allclose(p, link_pos) and np.allclose(pose_matrix(p, q), pose_matrix(link_pos, link_quat))


def test_look_pose_sits_by_the_shoulder_on_the_arms_side_and_faces_the_target():
    target = np.array([0.6, 0.0, 0.85])
    eye_left, quat_left = look_pose(target, shoulder=(0.16, 0.17, 1.23), side=+1)
    eye_right, _ = look_pose(target, shoulder=(0.16, -0.17, 1.23), side=-1)
    assert np.allclose(eye_left, [0.36, 0.47, 1.18]) and np.allclose(eye_right, [0.36, -0.47, 1.18])
    view = -pose_matrix(eye_left, quat_left)[:3, 2]
    assert np.allclose(view, (target - eye_left) / np.linalg.norm(target - eye_left))


@pytest.mark.skipif(not URDF.exists(), reason="the R1Pro URDF is not in this checkout")
def test_lula_solves_a_look_pose_for_the_left_wrist_camera():
    try:
        ik = ArmIK(URDF, LEFT_ARM, FIXED, frame="left_realsense_link")
    except ImportError:
        pytest.skip("lula is not importable in this environment")
    pos_home, _ = ik.fk(Q_HOME)
    assert 0.3 < pos_home[0] < 0.7 and 0.8 < pos_home[2] < 1.2  # the wrist camera ahead of the base, hand height
    # the look pose by the shoulder reaches targets across the workspace (the camera prim sits on the link's origin,
    # turned (0, 0.7071, -0.7071, 0) as in the USD)
    shoulder, _ = ik.fk(Q_HOME, "left_arm_link1")
    link_from_cam = pose_matrix([0.0, 0.0, 0.0], [0.7071, -0.7071, 0.0, 0.0])
    for target in ([0.55, 0.0, 0.8], [0.8, 0.2, 0.55], [0.7, -0.2, 0.3], [0.55, 0.1, 1.0]):
        eye, cam_quat = look_pose(target, shoulder, side=+1)
        q = ik.solve(*link_pose_for_camera(eye, cam_quat, link_from_cam), seed=Q_HOME)
        assert q is not None, target
        cam = pose_matrix(*ik.fk(q)) @ link_from_cam
        to_target = np.asarray(target) - cam[:3, 3]
        assert float(-cam[:3, 2] @ to_target / np.linalg.norm(to_target)) > np.cos(np.radians(15)), target
    # a link target 5 cm further and higher is reached from the ready posture within the tolerances
    pos, quat = ik.fk(Q_HOME)
    q = ik.solve(pos + [0.05, 0.0, 0.05], quat, seed=Q_HOME)
    assert q is not None and len(q) == 7
    pos_after, quat_after = ik.fk(q)
    assert np.linalg.norm(pos_after - (pos + [0.05, 0.0, 0.05])) < 0.01
    assert np.allclose(pose_matrix(pos_after, quat_after)[:3, :3], pose_matrix(pos, quat)[:3, :3], atol=0.15)
    assert ik.solve([3.0, 0.0, 0.0], quat, seed=Q_HOME) is None  # out of reach
