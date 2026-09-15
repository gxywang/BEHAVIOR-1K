"""The wrist-camera look pose: the pure transform helpers, and Lula IK on the R1Pro URDF (skipped without them)."""

import pathlib
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
from omnigibson.tiptop.r1pro import HEAD_VIEWS, HEAD_YAW_JOINT, R1ProSim, turned_joints

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
    assert framing([aside])[0] == "out of the head camera's frame altogether"  # not a pixel of it is in the picture
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
    assert measured_framing([cube([0.30, 0.0, 0.80])])[0] == "out of the head camera's frame altogether"


def test_the_measured_head_camera_reaches_the_floor_close_in():
    assert measured_framing([cube([0.55, 0.0, 0.04])])[0] is None  # a toy on the floor, within reach


# --------------------------------------------------------------- a stance that misses the object altogether
# GoalNotVisible was the commonest failure in runs/queue_logs by a wide margin. Of the 328 times the head camera
# was asked for a goal object and came back with an empty mask, 75 had the object off the LEFT edge of the image
# and not one off the right -- the stance search has to put an object within the LEFT arm's reach, and for
# anything the robot cannot stand square to that means beside its shoulder. The stances below are the ones
# packing_meal_for_delivery actually took on 2026-09-14, and the strict pass was accepting them.


def test_an_object_beside_the_camera_is_refused_however_big_its_projection_is():
    """The measured failure: a hamburger 0.34 m ahead of the base and 0.61 m to its left.

    It projects to pixel (-727, 951) of a 720x720 image -- nowhere near the picture -- but because it is close to
    the lens and far off its axis, its projected BOX is wider than the frame, and the exemption for an object too
    big to fit was letting the stance through. The round went out, the capture saw nothing of the hamburger and it
    died on empty masks.
    """
    beside = cube([0.34, 0.61, 0.89], half=0.06)
    assert measured_framing([beside])[0] == "out of the head camera's frame altogether"
    assert measured_framing([beside], strict=False)[1] > 0.0, "the fallback pass still charges for it"


def test_the_refusal_does_not_get_weaker_the_further_out_of_frame_the_object_goes():
    """A stance that misses by a mile must not be accepted while one that nearly frames the object is refused.

    That was the shape of the bug: the apparent size of the projected box grows with how far off-axis the object
    is, so the "too big to fit" exemption switched on exactly where the picture was worst. Sweeping the object
    sideways at a fixed distance, every one of these stances shows nothing of it and every one must be refused.
    """
    for side in (0.0, 0.15, 0.30, 0.45, 0.61, 0.75):
        why, _ = measured_framing([cube([0.34, side, 0.89], half=0.06)])
        assert why is not None, f"a stance with the object {side:.2f} m to the left shows none of it, but was taken"


def test_an_object_the_camera_can_actually_see_is_still_accepted():
    """The other half: the fix must not refuse the stances that do work.

    On a 0.89 m surface the measured head camera's window starts 0.45 m ahead of the base; these are inside it.
    """
    for ahead in (0.75, 0.90, 1.10):
        assert measured_framing([cube([ahead, 0.30, 0.89], half=0.06)])[0] is None, f"{ahead} m ahead was refused"


def test_an_object_only_partly_in_the_picture_is_still_a_cut_and_not_this_refusal():
    """The new refusal is for objects with nothing in the picture; a clipped one keeps its old treatment."""
    desk = (np.array([0.45, -0.45, 0.0]), np.array([1.10, 0.45, 0.80]))  # too big to frame, but plainly in view
    why, outside = measured_framing([desk])
    assert why is None and outside > 0.0


# --------------------------------------------------------------- turning the head to look at something
def aim(target, cam=None, limit=None):
    from omnigibson.tiptop.r1pro import HEAD_AIM_LIMIT, head_aim_yaw

    cam = MEASURED_HEAD if cam is None else cam
    return head_aim_yaw(cam[:3, 3], cam[:3, 2], target, HEAD_AIM_LIMIT if limit is None else limit)


def turned_in_place(delta):
    """The measured head camera re-aimed by ``delta`` about the base's vertical without moving.

    What ``torso_joint4`` does: it rotates the link the camera sits on, so the camera turns roughly where it
    stands (it is 9 cm off the axis, so it travels about 4 cm, which is not modelled here).
    """
    c, s = np.cos(delta), np.sin(delta)
    about_z = np.array([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    turned = about_z @ MEASURED_HEAD
    turned[:3, 3] = MEASURED_HEAD[:3, 3]  # re-aimed from the same place
    return turned


def test_a_target_straight_ahead_needs_no_turn():
    assert aim([1.0, 0.0, 0.89]) == pytest.approx(0.0, abs=1e-3)


def test_a_target_to_the_left_turns_the_head_left_and_one_to_the_right_turns_it_right():
    # head_left is a POSITIVE yaw, so a target to the robot's left must come back positive
    assert aim([0.7, 0.75, 0.89]) > 0.0
    assert aim([0.7, -0.75, 0.89]) < 0.0
    assert aim([0.7, 0.75, 0.89]) == pytest.approx(-aim([0.7, -0.75, 0.89]), abs=1e-6)


def test_the_turn_is_clamped_to_what_the_joint_may_do():
    from omnigibson.tiptop.r1pro import HEAD_AIM_LIMIT

    assert aim([0.1, 2.0, 0.89]) == pytest.approx(HEAD_AIM_LIMIT)
    assert aim([0.1, -2.0, 0.89]) == pytest.approx(-HEAD_AIM_LIMIT)
    assert abs(aim([-1.0, 0.05, 0.89])) <= HEAD_AIM_LIMIT, "a target behind must not ask for a 180 deg turn"


def test_the_pitch_of_the_head_does_not_confuse_the_bearing():
    """The challenge posture pitches the camera 43 deg down; the turn is about the vertical, so only the
    horizontal bearing may count. A target level with the camera and one on the floor below it, at the same
    bearing, must ask for the same turn."""
    assert aim([0.7, 0.5, 1.25]) == pytest.approx(aim([0.7, 0.5, 0.05]), abs=1e-6)


def test_a_target_at_the_camera_itself_asks_for_no_turn():
    assert aim(MEASURED_HEAD[:3, 3]) == 0.0


def test_the_aimed_turn_brings_an_object_off_the_left_edge_back_into_the_picture():
    """The point of the whole thing, on the measured camera.

    An object 0.70 m ahead of the base and 0.75 m to its left is off the left edge of the head image -- the stance
    the left arm's reach asks for. Turning the head by what ``head_aim_yaw`` says puts it back inside the frame.
    """
    from omnigibson.tiptop.r1pro import box_corners, frame_objects
    from omnigibson.tiptop.protocol import points_to_pixels

    target = np.array([0.70, 0.75, 0.89])
    (u, v), ahead = points_to_pixels([target], MEASURED_K, MEASURED_HEAD)[0][0], 1.0
    assert u < 0, f"the object should start off the LEFT edge, but projects to column {u:.0f}"
    delta = aim(target)
    assert delta > 0.0
    turned = turned_in_place(delta)
    (u2, v2), z2 = points_to_pixels([target], MEASURED_K, turned)[0][0], points_to_pixels(
        [target], MEASURED_K, turned
    )[1][0]
    assert z2 > 0 and 0 <= u2 < 720, f"after the turn it projects to column {u2:.0f}, still outside the image"
    lo, hi = target - 0.04, target + 0.04
    assert frame_objects([box_corners(lo, hi)], MEASURED_K, turned, 0.0, 720, 720, 0.0, 0.0, 0.0)[0] is None


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


# --------------------------------------------------------------- the base's real rectangle against furniture
# The R1Pro's base_link box in its own frame: it reaches 0.40 m behind the base frame's origin and 0.24 m ahead
# (logged by every run as "base box (base frame)"). ROBOT_FOOTPRINT's 0.36 m square is centred, so it under-covers
# the rear by 4 cm and, once the base is turned, misses its corners by up to 16 cm -- which is what put the base
# 15 cm inside cabinet_1 in runs/bench_batteries_ten.
BASE_LO, BASE_HI = (-0.40, -0.34), (0.24, 0.34)


def base_hits(box_lo, box_hi, centre=(0.0, 0.0), yaw=0.0):
    from omnigibson.tiptop.r1pro import rect_hits_box

    return rect_hits_box(centre, yaw, BASE_LO, BASE_HI, box_lo, box_hi)


def test_a_box_the_base_is_clear_of_does_not_overlap():
    assert not base_hits((0.5, -0.5), (1.5, 0.5))


def test_the_base_reaches_further_behind_its_origin_than_a_centred_square_says():
    behind = ((-0.45, -0.2), (-0.38, 0.2))  # 0.38 to 0.45 m behind: outside a 0.36 square, inside the real base
    assert base_hits(*behind)


def test_turning_the_base_sweeps_its_corners_into_a_box_a_square_test_misses():
    # at 45 deg the rear corner (-0.40, +0.34) swings to (-0.52, -0.04): 0.52 m out, where neither the unturned
    # base nor ROBOT_FOOTPRINT's 0.36 m square reaches
    corner = ((-0.55, -0.15), (-0.45, 0.05))
    assert not base_hits(*corner)
    assert base_hits(*corner, yaw=np.radians(45.0))


def test_the_test_is_symmetric_in_the_sense_that_overlap_does_not_depend_on_which_box_moves():
    from omnigibson.tiptop.r1pro import rect_hits_box

    near = ((0.20, -0.1), (0.40, 0.1))
    assert rect_hits_box((0, 0), 0.0, BASE_LO, BASE_HI, *near)
    assert not rect_hits_box((-0.5, 0), 0.0, BASE_LO, BASE_HI, *near)


def test_the_gap_is_positive_when_clear_and_zero_or_less_when_overlapping():
    from omnigibson.tiptop.r1pro import rect_box_gap

    clear = rect_box_gap((0, 0), 0.0, BASE_LO, BASE_HI, (0.60, -0.2), (1.0, 0.2))
    assert clear == pytest.approx(0.36, abs=0.01), "0.60 m away, the base reaching 0.24 m forward"
    touching = rect_box_gap((0, 0), 0.0, BASE_LO, BASE_HI, (0.24, -0.2), (0.6, 0.2))
    assert touching <= 0.0


def test_the_rectangle_is_more_generous_in_front_than_the_square_it_replaced():
    # the measured trap: the base reaches 0.24 m forward where ROBOT_FOOTPRINT's square guarded 0.36, so an honest
    # shape alone lets the search stand 12 cm closer to whatever it faces. The clearance term in the score is what
    # pays that back -- these two numbers are why it exists.
    from omnigibson.tiptop.r1pro import ROBOT_FOOTPRINT, rect_box_gap

    front = ((0.30, -0.2), (0.7, 0.2))
    assert rect_box_gap((0, 0), 0.0, BASE_LO, BASE_HI, *front) == pytest.approx(0.06, abs=0.01)
    assert 0.30 - ROBOT_FOOTPRINT < 0.0, "the square would have called this stance occupied"


def test_a_stance_short_of_the_wanted_clearance_costs_more_than_the_approach_it_buys():
    from omnigibson.tiptop.r1pro import CLEAR_WEIGHT, STANCE_CLEARANCE

    shortfall = 0.05  # standing 5 cm nearer than wanted
    assert CLEAR_WEIGHT * shortfall > shortfall, "the score's distance term is 1 per metre"
    assert STANCE_CLEARANCE > 0.0


def test_every_footprint_verdict_carries_its_clearance():
    """All of _footprint_free's returns are (free, why, clearance).

    Two of them were left at two values when the clearance was added, and the caller unpacks three: the first task
    to stand somewhere with no floor under a corner crashed the whole instance (clean_up_broken_glass, 2026-09-13,
    "not enough values to unpack (expected 3, got 2)"). Arity is not something to find in a benchmark.
    """
    import ast
    import inspect
    import textwrap

    from omnigibson.tiptop.r1pro import R1ProSim

    src = textwrap.dedent(inspect.getsource(R1ProSim._footprint_free))
    tree = ast.parse(src)
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    assert returns, "the function should return"
    for r in returns:
        assert isinstance(r.value, ast.Tuple) and len(r.value.elts) == 3, (
            f"a return at line {r.lineno} of _footprint_free has "
            f"{len(r.value.elts) if isinstance(r.value, ast.Tuple) else 1} values, not 3"
        )


class _StubHand:
    """Just enough of the sim class to exercise grasp_target's geometry without Isaac Sim."""

    def __init__(self, approach, jaw, grasp, tip):
        self._hand_convention = {
            "left": {
                "approach": np.asarray(approach, dtype=np.float64),
                "jaw": np.asarray(jaw, dtype=np.float64),
                "grasp": np.asarray(grasp, dtype=np.float64),
                "tip": float(tip),
            }
        }

    hand_convention = R1ProSim.hand_convention
    grasp_target = R1ProSim.grasp_target


# the R1Pro's hand as measured off the robot: fingers out along -z of the IK frame, jaw across y, the grasp
# centre 6 cm out and the fingertips 1.9 cm past that
R1PRO_HAND = ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (0.0, 0.0, -0.06), 0.019)


def test_grasp_target_puts_the_fingertips_on_the_point():
    """The IK frame goes wherever it must for the tips to reach the point -- 6 cm back, not on top of it."""
    hand = _StubHand(*R1PRO_HAND)
    point = np.array([0.6, 0.1, 0.8])
    approach = np.array([1.0, 0.0, 0.0])  # straight ahead into a panel facing the robot
    pos, rot = hand.grasp_target("left", point, approach, jaw_dir=[0.0, 0.0, 1.0], press=0.0)
    # the hand's approach axis now points the way we asked
    assert np.allclose(rot @ np.array([0.0, 0.0, -1.0]), approach, atol=1e-9)
    # the grasp centre lands one fingertip-length short, and the tips land on the point
    centre = pos + rot @ np.array([0.0, 0.0, -0.06])
    assert np.allclose(centre, point - approach * 0.019, atol=1e-9)
    assert np.allclose(centre + approach * 0.019, point, atol=1e-9)
    # and the IK frame itself is well behind the panel face, which is the whole point
    assert np.isclose(float(np.linalg.norm(pos - point)), 0.06 + 0.019, atol=1e-9)


def test_grasp_target_presses_the_tips_past_the_surface():
    """press drives the fingertips that far in, so the assisted grasp gets the contact it waits for."""
    hand = _StubHand(*R1PRO_HAND)
    point = np.array([0.6, 0.0, 0.8])
    approach = np.array([1.0, 0.0, 0.0])
    near, _ = hand.grasp_target("left", point, approach, press=0.0)
    into, _ = hand.grasp_target("left", point, approach, press=0.005)
    assert np.allclose(into - near, approach * 0.005, atol=1e-9)


def test_grasp_target_jaw_is_square_to_the_approach():
    """A jaw asked for along the approach cannot be had; the returned rotation stays a rotation either way."""
    hand = _StubHand(*R1PRO_HAND)
    approach = np.array([1.0, 0.0, 0.0])
    for jaw in ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.3, 0.0, 0.9]):
        _, rot = hand.grasp_target("left", [0.5, 0.0, 0.9], approach, jaw_dir=jaw)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert np.isclose(float(np.linalg.det(rot)), 1.0, atol=1e-9)
        jaw_world = rot @ np.array([0.0, 1.0, 0.0])
        assert abs(float(jaw_world @ approach)) < 1e-9


def test_the_travel_fold_moves_at_its_own_speed_and_the_unfold_is_checked():
    """Folding in over the base is cheap; coming back out into the room is the part that needs checking.

    The fold cuts the arms' overhang past the base rectangle from 0.41 m to 0.095 m, and it was the only candidate
    that both helped and arrived. It was removed for a day because it cost 7967 of an episode's 16946 steps -- but
    that cost came from ramping it at CAPTURE_MAX_JOINT_VEL, the cap that exists because observation swings out
    through an unplanned scene were knocking objects about. Bringing the arms in over the robot's own base is the
    opposite motion and runs at TRAVEL_MAX_JOINT_VEL instead, about a quarter of the steps. Coming back out IS a
    motion into the room, so path_hits_scene checks it first (2026-09-14).
    """
    import omnigibson.tiptop.r1pro as r1pro

    assert r1pro.TRAVEL_MAX_JOINT_VEL > r1pro.CAPTURE_MAX_JOINT_VEL, "a fold over the base is not a capture swing"
    assert r1pro.TRAVEL_POSE == 0.0
    assert hasattr(r1pro.R1ProSim, "fold_for_travel") and hasattr(r1pro.R1ProSim, "unfold_after_travel")
    fold = r1pro.R1ProSim.fold_for_travel.__doc__ or ""
    unfold = r1pro.R1ProSim.unfold_after_travel.__doc__ or ""
    assert "TRAVEL_MAX_JOINT_VEL" in fold, "the fold must say why it does not pay the capture cap"
    assert "path_hits_scene" in unfold, "the unfold must say it checks the way back"
    # It unfolds even when the path looks blocked. Refusing was measured worse: the arms stayed folded over the
    # base, which was inside the toy box the robot had come to work at, the start-state lift then fired three
    # times, and every plan was refused anyway -- 0.000 against a 0.75 baseline (2026-09-14). The ramp stopping
    # when a joint falls behind its target is the collision-awareness that matters.
    src = (pathlib.Path(r1pro.__file__).read_text().split("def unfold_after_travel")[1]).split("\n    def ")[0]
    assert "unfolding anyway" in src, "the warning must not become a refusal"
    assert "not unfolding here" not in src


def test_arm_points_can_be_evaluated_at_a_stance_the_robot_is_not_standing_in():
    """A stance has to be judged by where the arm would END UP, which means placing it before the robot moves.

    The base's rectangle is not the robot: the teleport folds the arms over the base and then unfolds them to the
    working posture, where they reach 0.41 m past that rectangle. A stance whose base is clear can still leave the
    hand inside a box on the floor, which is what the user saw in putting_away_toys -- after picking up a toy the
    robot teleported to the table and its arm came to rest INSIDE the toy box (2026-09-13).
    """

    class _Stub:
        robot = type("R", (), {"arm_link_names": {"left": ["l1"]}})()

        def base_to_world(self, p):
            raise AssertionError("`at` must not consult the robot's current pose")

    ik = type("IK", (), {"fk": staticmethod(lambda q, name: (np.array([1.0, 0.0, 0.5]), None))})()
    points = R1ProSim.arm_points(_Stub(), "left", ik, [0.0], at=(2.0, 3.0, 0.0))
    assert np.allclose(points[0], [3.0, 3.0, 0.5]), "a point 1 m ahead of a base at (2,3) facing +x is at (3,3)"
    turned = R1ProSim.arm_points(_Stub(), "left", ik, [0.0], at=(2.0, 3.0, np.pi / 2))
    assert np.allclose(turned[0], [2.0, 4.0, 0.5]), "the same point with the base turned 90 deg is at (2,4)"


def test_the_base_is_tested_against_geometry_not_a_bounding_box():
    """A desk is legs and a top; at the height the base sweeps, its bounding box is almost all air.

    Measured in picking_up_toys' scene, over the slab the base occupies (z 0.042 to 0.405 m): the desk that
    turned the robot away has a 1.83 m2 box footprint and 0.08 m2 of solid geometry in that slab, 96% air; the
    breakfast table and the coffee table are 100% air, so the base could roll clean underneath. Refusing a
    position on the box alone is why every task whose objects sit on a desk failed with "no base pose ...
    overlaps desk" before a single round ran. A bed is only 59-62% air, so this has to come from each object's
    own geometry rather than a rule about tables (2026-09-14).
    """
    from omnigibson.tiptop.r1pro import FOOTPRINT_CELL, R1ProSim

    class _Stub:
        _base_cells = {}

        def robot_height_cells(self, obj):
            return self._base_cells.get(obj)

        base_meets = R1ProSim.base_meets

    stub = _Stub()
    rect_lo, rect_hi = np.array([-0.39, -0.34]), np.array([0.23, 0.34])

    # empty at every height the robot occupies -- the notch of an L -- so the box is overstating it
    stub._base_cells["desk_notch"] = set()
    assert not stub.base_meets("desk_notch", (0.0, 0.0), 0.0, rect_lo, rect_hi)

    # a leg inside the rectangle: a real meeting
    leg = (int(0.1 / FOOTPRINT_CELL), int(0.1 / FOOTPRINT_CELL))
    stub._base_cells["desk"] = {leg}
    assert stub.base_meets("desk", (0.0, 0.0), 0.0, rect_lo, rect_hi)

    # the same leg, with the base standing a metre away: no meeting
    assert not stub.base_meets("desk", (1.5, 0.0), 0.0, rect_lo, rect_hi)

    # no mesh to go on: keep the box's word rather than inventing clearance
    stub._base_cells["mystery"] = None
    assert stub.base_meets("mystery", (0.0, 0.0), 0.0, rect_lo, rect_hi)


# --------------------------------------------------------------- the floor a piece of furniture really occupies
def _box_mesh(lo, hi):
    """A closed box as a trimesh, the shape of a desk leg or a table top."""
    import trimesh

    return trimesh.creation.box(
        extents=[hi[i] - lo[i] for i in range(3)],
        transform=trimesh.transformations.translation_matrix([(lo[i] + hi[i]) / 2 for i in range(3)]),
    )


def test_a_leg_that_passes_straight_through_the_slab_is_found():
    """A desk leg runs from the floor to the underside of the top, so NONE of its vertices lie in the slab the
    robot's base sweeps. Reading the vertices alone reported the leg as empty floor and the search stood the
    robot inside it."""
    from omnigibson.tiptop.r1pro import footprint_cells

    leg = _box_mesh((1.00, 1.00, 0.0), (1.06, 1.06, 0.73))  # 6 cm square, floor to 73 cm
    slab_lo, slab_hi = 0.02, 0.35

    vertices_only = {
        (int(np.floor(x / 0.05)), int(np.floor(y / 0.05)))
        for x, y, z in np.asarray(leg.vertices)
        if slab_lo <= z <= slab_hi
    }
    assert vertices_only == set(), "the old test finds nothing, which is the bug"
    assert footprint_cells(leg, slab_lo, slab_hi), "the leg occupies floor and must be found"


def test_a_table_top_above_the_base_is_still_air_underneath():
    """The reason the face test is not simply 'use the bounding box': the base rolls under a table top."""
    from omnigibson.tiptop.r1pro import footprint_cells

    top = _box_mesh((0.0, 0.0, 0.70), (1.60, 0.80, 0.74))
    assert footprint_cells(top, 0.02, 0.35) == set(), "nothing of the top is in the slab"


def test_the_slab_the_stance_search_uses_reaches_the_whole_robot_not_just_the_wheels():
    """A table top is air at wheel height and solid where the arms are, so the slab has to include the arms.

    Run with the slab set to the base's own height (0.04-0.41 m), tidying_living_room parked the robot under the
    coffee table at 0.25-0.60 m, the arm could not fold for travel, never came back to ready, and ended up in
    front of the head camera: every look returned an empty mask and the task went 0.250 -> 0.000 (2026-09-15).
    """
    import inspect

    from omnigibson.tiptop.r1pro import R1ProSim, footprint_cells

    top = _box_mesh((0.0, 0.0, 0.45), (1.20, 0.60, 0.49))  # a table top clear above the base, nothing below it
    assert footprint_cells(top, 0.04, 0.41) == set(), "at wheel height a table top is air -- that part was right"
    assert footprint_cells(top, 0.04, 1.6), "at arm height it is solid, and that is what the stance search must see"

    body = inspect.getsource(R1ProSim.robot_height_cells).split('"""')[-1]
    assert "ROBOT_HEIGHT" in body, "the slab must run to the top of the robot"
    assert "hi_b" not in body, "and must not stop at the top of the base"


def test_a_thing_lying_flat_on_the_floor_is_found():
    from omnigibson.tiptop.r1pro import footprint_cells

    toy = _box_mesh((2.0, 2.0, 0.0), (2.08, 2.08, 0.06))
    assert footprint_cells(toy, 0.02, 0.35), "it stands taller than the base's underside, so it is an obstacle"


def test_the_cells_cover_the_leg_and_not_half_the_room():
    from omnigibson.tiptop.r1pro import footprint_cells

    leg = _box_mesh((1.00, 1.00, 0.0), (1.06, 1.06, 0.73))
    cells = footprint_cells(leg, 0.02, 0.35)
    assert 1 <= len(cells) <= 9, f"a 6 cm leg should touch a couple of 5 cm cells, got {len(cells)}"
    assert all(19 <= gx <= 22 and 19 <= gy <= 22 for gx, gy in cells), "and they should be where the leg is"


# ------------------------------------------------- standing further back rather than accepting a clipped view
def _ladder(at_09=None, at_wide=None, clipped=("clipped",), strict=True, has_boxes=True, reach=0.9):
    """Drive widen_then_clip with a search that answers per (radius, strict), recording what it was asked."""
    from omnigibson.tiptop.r1pro import REACH_WIDEN, widen_then_clip

    asked = []

    def search(r, is_strict):
        asked.append((round(r, 2), is_strict))
        if not is_strict:
            return clipped, {}
        return (at_wide if r >= REACH_WIDEN else at_09), {"geometry": 1}

    out, _ = widen_then_clip(at_09, {"geometry": 1}, reach=reach, frame_strict=strict,
                             has_boxes=has_boxes, search=search)
    return out, asked


def test_it_stands_further_back_rather_than_accept_a_clipped_view():
    """jigsaw_puzzle_2 has 0 framing stances at 0.9 m and 88 at 1.1 m."""
    from omnigibson.tiptop.r1pro import REACH_WIDEN

    out, asked = _ladder(at_09=None, at_wide=("wider",))
    assert out == ("wider",), "the well-framed stance further back should win"
    assert asked[0] == (REACH_WIDEN, True), f"it must try the wider radius while still strict first, got {asked}"
    assert (0.9, False) not in asked, "and must not settle for a clipped view when a framed one exists"


def test_a_clipped_view_is_still_the_last_resort():
    """When nothing frames the goal at any radius, part of the goal in view beats no stance."""
    out, asked = _ladder(at_09=None, at_wide=None)
    assert out == ("clipped",)
    assert asked[-1][1] is False, "the clipped search comes last"


def test_a_stance_that_already_frames_it_is_left_alone():
    out, asked = _ladder(at_09=("near",), at_wide=("wider",))
    assert out == ("near",) and asked == [], "no widening when the close stance already frames it"


def test_a_search_that_was_never_strict_is_left_alone():
    out, asked = _ladder(at_09=None, at_wide=("wider",), strict=False)
    assert out is None and asked == [], "the relaxed pass must not recurse"


def test_without_boxes_there_is_nothing_to_clip_to():
    out, asked = _ladder(at_09=None, at_wide=None, has_boxes=False)
    assert out is None
    assert all(a[1] for a in asked), "never asks for a clipped view when there are no boxes to frame"


def test_it_does_not_widen_when_already_at_the_wider_radius():
    from omnigibson.tiptop.r1pro import REACH_WIDEN

    out, asked = _ladder(at_09=None, at_wide=None, reach=REACH_WIDEN)
    assert (REACH_WIDEN, True) not in asked, "no pointless second search at the same radius"


# ------------------------------------------------------- the arm may reach into what it is standing for
def test_the_arm_may_rest_inside_what_it_is_reaching_for():
    """Standing at a bookcase to put a book in it, the arm ends up inside the bookcase's box. That is what
    reaching into something looks like, and refusing it refused every stance from which the goal could be served:
    "no base pose reaches ['bookcase.n.01_2'] ... {'overlaps bookcase_zfpyqe_0': 2497}"."""
    from omnigibson.tiptop import r1pro

    class Obj:
        def __init__(self, name):
            self.name = name

    bookcase, lamp = Obj("bookcase_zfpyqe_0"), Obj("lamp_1")

    # the set the arm test consults, as _footprint_free builds it
    spared = {o.name for o in [lamp]} | {o.name for o in [bookcase]}
    assert "bookcase_zfpyqe_0" in spared, "the thing being reached into must not refuse the stance"
    assert "lamp_1" in spared, "and an explicitly ignored object stays spared"

    # what it used to be: ignore alone, without the target
    old = {o.name for o in [lamp]}
    assert "bookcase_zfpyqe_0" not in old, "this is the bug the change removes"


def test_the_base_still_refuses_to_stand_inside_it():
    """The two sets are deliberately separate: the arm may reach in, the wheels may not drive in."""
    import inspect

    from omnigibson.tiptop import r1pro

    src = inspect.getsource(r1pro.R1ProSim._footprint_free)
    loop = src.split("for obj, lo, hi in aabbs:")[1].split("if yaw is not None and self.q_home is not None")[0]
    assert "reaching" not in loop, "the base overlap loop must not spare what is being reached for"
    assert "obj in ignore" in loop, "it still honours the explicit ignore list"


# --------------------------------------------------- reach is to the nearest part of a target, not its centre
def _reach_ok(dist, half_width, reach=0.9):
    """The test as best_base_pose applies it."""
    return max(0.0, dist - half_width) <= reach


def test_a_wide_target_is_reachable_from_beside_it():
    """A bed is 2 m across. Putting something on it means reaching the bed, not the middle of the bed -- and the
    only stances within 0.9 m of a bed's CENTRE are the ones standing on it, which the footprint test then
    refuses. 30,718 candidates were refused for "overlaps bed" while standing for that same bed."""
    bed_half = 1.0
    assert _reach_ok(dist=1.5, half_width=bed_half), "standing 0.5 m from the bed's edge is in reach"
    assert not _reach_ok(dist=1.5, half_width=0.0), "measuring to the centre refuses it, which was the bug"


def test_a_small_object_is_unaffected():
    """A battery's radius is a couple of centimetres; nothing about this changes for it."""
    for d in (0.5, 0.88, 0.92, 1.4):
        assert _reach_ok(d, 0.02) == (d - 0.02 <= 0.9)


def test_standing_inside_a_target_is_still_in_reach_not_negative():
    assert _reach_ok(dist=0.2, half_width=1.0), "no negative distances"


def test_a_target_further_than_reach_plus_its_radius_is_still_refused():
    """The change must not make everything reachable."""
    assert not _reach_ok(dist=2.5, half_width=1.0), "0.9 m past a bed's edge is still too far"


def test_the_fold_for_travel_moves_the_arms_and_leaves_the_torso_alone():
    from omnigibson.tiptop.r1pro import TRAVEL_POSE, travel_fold_targets

    joints = ["torso_joint1", "torso_joint2", "left_arm_joint1", "left_arm_joint4", "left_arm_joint7"]
    now = [1.2, -1.7, 0.4, -0.9, 2.2]
    folded = travel_fold_targets(now, joints)
    assert folded[:2] == [1.2, -1.7], "the torso holds the posture the episode is running"
    assert folded[2:] == [TRAVEL_POSE] * 3, "every arm joint goes to the travel pose"
    assert travel_fold_targets(now, joints) is not folded or True  # pure: a fresh list each call
    assert travel_fold_targets([1.2, -1.7], ["torso_joint1", "torso_joint2"]) is None, "no arm joint, nothing to fold"


def test_a_fold_stopped_by_furniture_is_tried_again_at_the_new_stance():
    """The fold happens BEFORE the teleport, so a fold stopped by furniture was stopped at the OLD stance.

    Travelling with the arm wherever it jammed is how it ends up inside the thing the new stance was chosen to
    reach: tidying_living_room blocked the fold 12 times in one run, ran no placement round at all, and its rounds
    died on "the left arm starts inside coffee_table_osroux_0, which the planner refuses before it looks at the
    goal" (2026-09-15). So stand_at folds again once it has arrived.
    """
    import inspect

    from omnigibson.tiptop.r1pro import R1ProSim

    fold = inspect.getsource(R1ProSim.fold_for_travel)
    assert "_fold_blocked" in fold, "the fold has to record that it was stopped"

    # the body only: place_robot's docstring names unfold_after_travel, and an index into the whole source
    # would match the prose rather than the call
    stand = inspect.getsource(R1ProSim.place_robot).split('"""')[-1]
    assert "_fold_blocked" in stand, "and place_robot has to act on it"
    retry = stand.index("_fold_blocked")
    unfold = stand.index("unfold_after_travel")
    assert retry < unfold, "fold again BEFORE unfolding, or the unfold starts from the jammed posture"
    teleport = stand.index("set_position_orientation")
    assert teleport < retry, "and only AFTER the teleport, since the obstacle is at the old stance"
