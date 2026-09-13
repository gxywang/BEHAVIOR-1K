"""Following a joint with the gripper: the geometry of opening a drawer or a door. No simulator.

Opening is the largest single unlock left in the challenge set -- expanding all 100 goals puts the current
vocabulary's ceiling at a mean q_score of 0.415 and open/not-open takes it to 0.636 -- and all of the geometry can
be checked here, which is where a mistake in it should be found.
"""

import numpy as np
import pytest

from omnigibson.tiptop.articulation import (
    OPEN_FRACTION_REACH,
    OPEN_FRACTION_SCORED,
    follow_joint,
    is_open,
    joint_transform,
    opening_travel,
    pose_matrix,
    rotation_about,
)

HANDLE = pose_matrix([0.5, 0.0, 0.9], [0.0, 0.0, 0.0, 1.0])


def test_a_drawer_handle_slides_straight_out_along_the_joint_axis():
    poses = follow_joint(HANDLE, "prismatic", [1.0, 0.0, 0.0], [0, 0, 0], travel=0.30, steps=5)
    assert len(poses) == 5
    assert np.allclose(poses[0], HANDLE), "the path starts where the hand must take hold"
    assert np.allclose(poses[-1][:3, 3], [0.80, 0.0, 0.9])
    assert all(np.allclose(p[:3, :3], np.eye(3)) for p in poses), "a drawer does not turn the hand"
    steps = np.diff([p[0, 3] for p in poses])
    assert np.allclose(steps, steps[0]), "evenly spaced along the slide"


def test_a_door_handle_swings_on_an_arc_and_turns_the_hand_with_it():
    hinge = [0.5, -0.4, 0.9]  # the handle is 0.4 m from the hinge, on a vertical axis
    poses = follow_joint(HANDLE, "revolute", [0.0, 0.0, 1.0], hinge, travel=np.pi / 2, steps=9)
    radius = [float(np.linalg.norm(p[:3, 3] - np.asarray(hinge))) for p in poses]
    assert np.allclose(radius, radius[0], atol=1e-9), "the handle stays on its arc"
    assert radius[0] == pytest.approx(0.4, abs=1e-9)
    turned = poses[-1][:3, :3] @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(turned, [0.0, 1.0, 0.0], atol=1e-9), "the hand turned with the door, a quarter turn"


def test_the_hand_must_turn_on_a_door_or_it_would_have_to_slide_around_the_handle():
    hinge = [0.5, -0.4, 0.9]
    poses = follow_joint(HANDLE, "revolute", [0.0, 0.0, 1.0], hinge, travel=np.pi / 3, steps=4)
    for p in poses[1:]:
        assert not np.allclose(p[:3, :3], np.eye(3))


def test_an_unopenable_joint_is_refused_rather_than_guessed_at():
    with pytest.raises(ValueError, match="does not open"):
        joint_transform("fixed", [1, 0, 0], [0, 0, 0], 0.1)


def test_rotation_about_a_zero_axis_is_the_identity_not_a_crash():
    assert np.allclose(rotation_about([0.0, 0.0, 0.0], 1.0), np.eye(3))


def test_travel_goes_toward_whichever_limit_is_nearer_so_either_hinging_works():
    # a drawer closed at its lower limit opens by going up the range
    assert opening_travel("prismatic", 0.0, 0.4, position=0.0, fraction=0.5) == pytest.approx(0.2)
    # a door closed at its UPPER limit opens by going down it
    assert opening_travel("revolute", -1.6, 0.0, position=0.0, fraction=0.5) == pytest.approx(-0.8)


def test_opening_a_closed_drawer_by_a_fraction_means_that_fraction_of_its_range():
    # store_honey's cabinet: four drawers, range [0, 0.39], all resting closed. 80% open is 0.312 m of travel --
    # the earlier version answered 0.078, being the distance to a target 80% of the way from the FAR limit
    assert opening_travel("prismatic", 0.0, 0.39, position=0.0, fraction=0.80) == pytest.approx(0.312, abs=1e-6)
    assert opening_travel("prismatic", 0.0, 0.39, position=0.0, fraction=OPEN_FRACTION_SCORED) == pytest.approx(
        0.0312, abs=1e-6
    )


def test_a_partly_open_joint_travels_only_the_rest_of_the_way():
    assert opening_travel("prismatic", 0.0, 0.4, position=0.1, fraction=0.5) == pytest.approx(0.1)


def test_travel_never_asks_for_more_than_the_joint_has():
    far = opening_travel("prismatic", 0.0, 0.4, position=0.0, fraction=2.0)
    assert far == pytest.approx(0.4), "clamped to the limit, not 0.8 m into the cabinet"


def test_a_joint_with_no_range_asks_for_no_travel():
    assert opening_travel("prismatic", 0.3, 0.3, position=0.3, fraction=0.8) == 0.0


def test_the_scored_atom_is_far_cheaper_than_reaching_inside():
    assert OPEN_FRACTION_SCORED > 0.05, "OmniGibson flips Open at 5% of the range; clear it"
    assert OPEN_FRACTION_REACH > 4 * OPEN_FRACTION_SCORED, "reaching in is a different, much longer stroke"


def test_is_open_matches_omnigibsons_five_percent_rule_at_both_ends():
    assert not is_open(0.0, 0.4, position=0.01)  # 2.5% out: still closed
    assert is_open(0.0, 0.4, position=0.03)  # 7.5% out: open
    assert not is_open(0.0, 0.4, position=0.39), "a joint resting at its far limit is closed too"


# --------------------------------------------------------------- the handle heuristic
class _Link:
    """A link with just the box the handle heuristic reads."""

    def __init__(self, lo, hi):
        import torch as th

        self.aabb = (th.tensor(lo, dtype=th.float32), th.tensor(hi, dtype=th.float32))


def test_the_handle_sits_on_the_face_that_leads_when_the_drawer_opens():
    from omnigibson.tiptop.articulation import handle_point

    drawer = _Link([0.0, -0.25, 0.5], [0.5, 0.25, 0.65])  # 0.5 m deep, opening along +x
    point = handle_point(drawer, [1.0, 0.0, 0.0], opening_sign=+1.0)
    assert point[0] == pytest.approx(0.5), "the drawer front, not its middle"
    assert point[1] == pytest.approx(0.0) and point[2] == pytest.approx(0.575)


def test_the_handle_follows_the_direction_the_joint_actually_opens():
    from omnigibson.tiptop.articulation import handle_point

    drawer = _Link([0.0, -0.25, 0.5], [0.5, 0.25, 0.65])
    assert handle_point(drawer, [1.0, 0.0, 0.0], opening_sign=-1.0)[0] == pytest.approx(0.0)


def test_a_degenerate_axis_falls_back_to_the_middle_of_the_link():
    from omnigibson.tiptop.articulation import handle_point

    link = _Link([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert np.allclose(handle_point(link, [0.0, 0.0, 0.0], 1.0), [0.5, 0.5, 0.5])
