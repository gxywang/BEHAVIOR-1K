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
from omnigibson.tiptop.r1pro import HEAD_VIEWS, HEAD_YAW_JOINT, turned_joints

URDF = Path(__file__).resolve().parents[2] / "datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"
LEFT_ARM = [f"left_arm_joint{i}" for i in range(1, 8)]
TORSO = [f"torso_joint{i}" for i in range(1, 5)]  # planned before the arm in r1pro_left
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


def test_a_head_view_moves_one_torso_joint_and_nothing_else():
    """A head view moves one planned torso joint off the capture posture: yaw re-aims the camera, pitch leans the
    mast it sits on. Everything else in the posture stays."""
    q = [1.2, -1.7, -0.9, 0.0, *Q_HOME]
    for name, (joint, delta) in HEAD_VIEWS.items():
        i = (TORSO + LEFT_ARM).index(joint)
        turned = turned_joints(TORSO + LEFT_ARM, q, joint, delta)
        assert turned[i] == pytest.approx(q[i] + delta), name
        assert turned[:i] == q[:i] and turned[i + 1 :] == q[i + 1 :], name
    assert q == [1.2, -1.7, -0.9, 0.0, *Q_HOME]  # the input is left alone
    assert HEAD_VIEWS["head_left"][1] == -HEAD_VIEWS["head_right"][1] > 0  # left is a positive yaw about z
    assert HEAD_VIEWS["head_up"][1] == -HEAD_VIEWS["head_down"][1] > 0
    assert HEAD_VIEWS["head_up"][0] != HEAD_VIEWS["head_left"][0]  # pitch and yaw are different joints
    with pytest.raises(ValueError):  # an embodiment that locks the torso cannot turn it through q_arm
        turned_joints(LEFT_ARM, Q_HOME, HEAD_YAW_JOINT, 0.5)
    with pytest.raises(ValueError):
        turned_joints(TORSO + LEFT_ARM, Q_HOME, HEAD_YAW_JOINT, 0.5)


# --------------------------------------------------------------- framing a stance's head view
# A head camera roughly where the R1Pro's sits in its base frame -- 1.4 m up, 0.1 m ahead, pitched 30 deg down,
# OpenCV axes (+x right, +y down, +z forward); the measured challenge posture puts it at 1.26 m, and the tests are
# about the geometry, not that pose. 720x720 at 99 deg gives fx = 360 / tan(49.5 deg) = 308.
HEAD_PITCH = np.radians(30.0)
HEAD_K = np.array([[308.0, 0.0, 360.0], [0.0, 308.0, 360.0], [0.0, 0.0, 1.0]])


def head_in_base(pitch: float = HEAD_PITCH) -> np.ndarray:
    forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(forward, right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = right, down, forward
    m[:3, 3] = [0.1, 0.0, 1.4]
    return m


def framing(boxes, x=0.0, y=0.0, yaw=0.0, **kw):
    from omnigibson.tiptop.r1pro import box_corners, frame_objects

    return frame_objects(
        [box_corners(lo, hi) for lo, hi in boxes], HEAD_K, head_in_base(), 0.0, 720, 720, x, y, yaw, **kw
    )


def cube(centre, half=0.03):
    c = np.asarray(centre, dtype=float)
    return (c - half, c + half)


def test_box_corners_are_the_eight_corners_and_their_mean_is_the_centre():
    from omnigibson.tiptop.r1pro import box_corners

    corners = box_corners([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    assert corners.shape == (8, 3)
    assert len({tuple(c) for c in corners}) == 8
    assert np.allclose(corners.mean(axis=0), [0.5, 1.0, 2.0])


def test_an_object_on_a_table_ahead_is_framed_whole():
    why, outside = framing([cube([0.6, 0.0, 0.75])])
    assert why is None and outside == 0.0


def test_an_object_at_the_robots_feet_is_below_the_frame():
    # what the diagnostic saw: a battery 0.2 m ahead of the base projects past the bottom of a 720-row image
    why, _ = framing([cube([0.2, 0.0, 0.75])])
    assert why == "outside the head camera's frame"


def test_an_object_behind_the_robot_is_behind_the_camera():
    why, _ = framing([cube([-0.6, 0.0, 0.75])])
    assert why == "behind the head camera"


def test_yaw_brings_an_object_to_the_side_into_frame():
    aside = cube([0.4, 0.9, 0.75])  # 66 deg to the left of a robot facing +x: outside a 99 deg frame
    assert framing([aside])[0] == "outside the head camera's frame"
    assert framing([aside], yaw=np.radians(66.0))[0] is None


def test_an_object_the_frame_could_hold_whole_is_rejected_when_a_stance_cuts_it():
    table = (np.array([0.35, -0.45, 0.0]), np.array([0.95, 0.45, 0.5]))  # fits in frame, but not from here
    assert framing([table])[0] == "outside the head camera's frame"
    assert framing([table], x=-0.7)[0] is None  # the same object, the robot further back


def test_an_object_too_big_for_the_frame_is_a_penalty_not_a_rejection():
    wall = (np.array([0.3, -2.0, 0.0]), np.array([1.2, 2.0, 1.6]))  # nothing frames this whole
    why, outside = framing([wall])
    assert why is None and outside > 0.0


def test_the_fallback_pass_takes_a_cut_object_as_a_penalty():
    why, outside = framing([cube([0.2, 0.0, 0.75])], strict=False)
    assert why is None and outside > 0.0


def test_standing_further_back_frames_what_standing_close_cuts_off():
    on_the_floor = cube([0.35, 0.0, 0.03])
    assert framing([on_the_floor])[0] == "outside the head camera's frame"
    assert framing([on_the_floor], x=-0.35)[0] is None  # the same object, the robot 0.35 m further back


# --------------------------------------------------------------- the arm on the head camera's line of sight
def blocking(point, eye=(0.1, 0.0, 1.4), target=(0.6, 0.0, 0.75), radius=0.08):
    from omnigibson.tiptop.r1pro import blocks_ray

    return blocks_ray(eye, target, point, radius)


def test_a_link_on_the_line_between_the_camera_and_the_target_blocks_it():
    assert blocking((0.35, 0.0, 1.075))  # halfway along the sight line


def test_a_link_beside_the_line_does_not_block():
    assert not blocking((0.35, 0.3, 1.075))


def test_a_link_behind_the_camera_or_past_the_target_does_not_block():
    assert not blocking((-0.2, 0.0, 1.6))
    assert not blocking((0.9, 0.0, 0.55))


def test_the_radius_is_what_decides_a_near_miss():
    just_off = (0.35, 0.09, 1.075)
    assert not blocking(just_off)
    assert blocking(just_off, radius=0.12)


# The head camera as it actually sits in the base frame at the challenge torso posture, read from a capture
# (runs/bench_toys_head3, `--torso 1.2 -1.7 -0.9 0.0`): 0.44 m ahead of the base, 1.25 m up, pitched 43 deg down,
# 720x720 with fx 306. The framing test is only as good as this pose, so one case is measured rather than made up.
MEASURED_HEAD = np.array(
    [
        [0.0, -0.680984, 0.732298, 0.441770],
        [-1.0, 0.000001, -0.000005, 0.000004],
        [0.0, -0.732298, -0.680984, 1.248709],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
MEASURED_K = np.array([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]])


def measured_framing(boxes, x=0.0, y=0.0, yaw=0.0, **kw):
    from omnigibson.tiptop.r1pro import box_corners, frame_objects

    return frame_objects(
        [box_corners(lo, hi) for lo, hi in boxes], MEASURED_K, MEASURED_HEAD, 0.0, 720, 720, x, y, yaw, **kw
    )


def test_the_measured_head_camera_frames_a_battery_on_a_desk_but_not_one_at_the_robots_feet():
    # the desk in dispose_of_batteries stands at z 0.78 and the head camera sees it from 0.42 m ahead
    assert measured_framing([cube([0.75, 0.0, 0.80])])[0] is None
    assert measured_framing([cube([0.30, 0.0, 0.80])])[0] == "outside the head camera's frame"


def test_the_measured_head_camera_reaches_the_floor_close_in():
    assert measured_framing([cube([0.55, 0.0, 0.04])])[0] is None  # a toy on the floor, within reach


# --------------------------------------------------------------- a link's box on the line of sight
def hits(lo, hi, eye=(0.1, 0.0, 1.4), target=(0.8, 0.0, 0.75)):
    from omnigibson.tiptop.r1pro import segment_hits_box

    return segment_hits_box(eye, target, lo, hi)


def test_a_box_straddling_the_sight_line_is_hit():
    assert hits((0.40, -0.10, 1.00), (0.55, 0.10, 1.15))


def test_a_box_beside_the_sight_line_is_missed():
    assert not hits((0.40, 0.30, 1.00), (0.55, 0.50, 1.15))


def test_a_box_past_the_target_is_missed():
    assert not hits((1.00, -0.10, 0.55), (1.20, 0.10, 0.75))


def test_a_box_behind_the_camera_is_missed():
    assert not hits((-0.50, -0.10, 1.50), (-0.30, 0.10, 1.70))


def test_a_slab_the_line_runs_parallel_to_is_hit_only_when_the_line_is_inside_it():
    flat = ((0.2, -0.5, 1.0), (0.9, 0.5, 1.2))  # the sight line passes through this height band
    assert hits(*flat)
    assert not hits((0.2, -0.5, 1.5), (0.9, 0.5, 1.7))  # the same band, above the line


# --------------------------------------------------------------- the whole arm against a scene box
def arm_hits(points, lo=(0.5, -0.2, 0.6), hi=(0.9, 0.2, 0.9), clearance=0.0):
    from omnigibson.tiptop.r1pro import polyline_hits_box

    return polyline_hits_box(points, lo, hi, clearance)


def test_an_arm_that_reaches_over_a_box_does_not_enter_it():
    over = [(0.2, 0.0, 1.2), (0.5, 0.0, 1.1), (0.8, 0.0, 1.0)]
    assert not arm_hits(over)


def test_an_arm_whose_elbow_crosses_the_box_is_caught_even_when_both_ends_are_clear():
    # both endpoints outside, the segment between them straight through: the case a point test on the hand misses
    through = [(0.2, 0.0, 0.75), (1.2, 0.0, 0.75)]
    assert arm_hits(through)


def test_clearance_catches_a_limb_that_only_grazes():
    beside = [(0.2, 0.0, 0.95), (1.2, 0.0, 0.95)]  # 5 cm above the box's top
    assert not arm_hits(beside)
    assert arm_hits(beside, clearance=0.06)


def test_a_polyline_of_one_point_hits_nothing():
    assert not arm_hits([(0.7, 0.0, 0.75)])


def test_sampling_a_polyline_keeps_the_corners_and_bounds_the_spacing():
    from omnigibson.tiptop.r1pro import sample_polyline

    pts = sample_polyline([(0, 0, 0), (0.1, 0, 0), (0.1, 0.05, 0)], step=0.02)
    assert np.allclose(pts[0], [0, 0, 0]) and np.allclose(pts[-1], [0.1, 0.05, 0])
    assert np.linalg.norm(np.diff(pts, axis=0), axis=1).max() <= 0.02 + 1e-9
    assert any(np.allclose(p, [0.1, 0, 0]) for p in pts)  # the corner itself is a sample


def test_sampling_an_empty_or_single_point_polyline_is_harmless():
    from omnigibson.tiptop.r1pro import sample_polyline

    assert sample_polyline([], step=0.02).shape == (0, 3)
    assert sample_polyline([(1.0, 2.0, 3.0)], step=0.02).shape == (1, 3)


def test_the_head_view_return_tolerance_clears_the_settling_it_must_not_abort_on():
    # measured over one putting_away_toys run (2026-09-13): 32 successful returns spread 0.0000-0.0282 rad, and
    # the three rounds this check aborted were "off by 0.030" -- indistinguishable from settling
    from omnigibson.tiptop.r1pro import HEAD_VIEW_RETURN_TOL, HEAD_VIEWS

    assert HEAD_VIEW_RETURN_TOL > 0.0282 * 2, "the abort threshold must sit clear of normal settling"
    smallest_delta = min(abs(delta) for _, delta in HEAD_VIEWS.values())
    assert HEAD_VIEW_RETURN_TOL < smallest_delta / 2, "but must still catch a torso that did not come back"
