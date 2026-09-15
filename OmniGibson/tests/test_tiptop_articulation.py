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
        0.312, abs=1e-6
    )


def test_a_partly_open_joint_travels_only_the_rest_of_the_way():
    assert opening_travel("prismatic", 0.0, 0.4, position=0.1, fraction=0.5) == pytest.approx(0.1)


def test_travel_never_asks_for_more_than_the_joint_has():
    far = opening_travel("prismatic", 0.0, 0.4, position=0.0, fraction=2.0)
    assert far == pytest.approx(0.4), "clamped to the limit, not 0.8 m into the cabinet"


def test_a_joint_with_no_range_asks_for_no_travel():
    assert opening_travel("prismatic", 0.3, 0.3, position=0.3, fraction=0.8) == 0.0


def test_both_open_strokes_clear_the_state_threshold_and_stiction():
    """Both fractions ask for a long pull, and the reason is measured rather than assumed.

    This test used to assert the opposite -- that scoring an `open` atom was a much shorter, cheaper stroke than
    reaching inside. Trying it in the simulator refuted that: asked for 8% of store_honey's drawer range (31 mm
    spread over ten steps, 3 mm a step) the drawer moved 7 mm and stayed shut, while 80% (31 mm a step) tracked
    the hand one-to-one to 13 cm, where the ARM ran out of reach. Short steps do not break the joint's stiction.
    """
    assert OPEN_FRACTION_SCORED > 0.05, "OmniGibson flips Open at 5% of the range; clear it"
    assert OPEN_FRACTION_REACH > 0.05
    assert OPEN_FRACTION_SCORED >= 0.5, "a short stroke was measured not to move the drawer at all"
    # 10% of a range is the most that can be asked before the per-step move is under a centimetre on a 0.39 m
    # drawer split into OPEN_PATH_STEPS steps, which is the regime that failed.
    assert OPEN_FRACTION_SCORED * 0.39 / 10 > 0.01, "per-step move must be over a centimetre to break stiction"


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


# --------------------------------------------------------------- grasp orientations for a handle
def test_the_hands_own_orientation_is_offered_first():
    from omnigibson.tiptop.articulation import grasp_orientations

    cur = np.eye(3)
    out = grasp_orientations(cur, [1.0, 0.0, 0.0])
    assert np.allclose(out[0], cur), "what the hand is already holding costs nothing to try"
    assert len(out) > 1, "and it is rarely the one that works"


def test_every_offered_orientation_is_a_rotation():
    from omnigibson.tiptop.articulation import grasp_orientations

    cur = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for rot in grasp_orientations(cur, [0.0, 1.0, 0.3]):
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9), "orthonormal"
        assert float(np.linalg.det(rot)) == pytest.approx(1.0, abs=1e-9), "right handed, not a reflection"


def test_some_offered_orientation_faces_the_way_the_drawer_comes_out():
    from omnigibson.tiptop.articulation import grasp_orientations

    pull = np.array([1.0, 0.0, 0.0])  # the drawer comes toward +x, so a jaw axis should point back along -x
    out = grasp_orientations(np.eye(3), pull)
    assert any(any(np.allclose(rot[:, i], -pull, atol=1e-6) for i in range(3)) for rot in out), (
        "at least one candidate turns an axis of the hand to face the drawer"
    )


def test_a_degenerate_pull_direction_just_offers_what_the_hand_has():
    from omnigibson.tiptop.articulation import grasp_orientations

    assert len(grasp_orientations(np.eye(3), [0.0, 0.0, 0.0])) == 1


def test_the_edge_grip_aims_at_the_top_of_the_panel_and_the_face_grip_at_its_middle():
    """No BEHAVIOR asset marks a handle: store_honey's cabinet is four flat drawer fronts tagged only "openable".

    A parallel jaw has nothing to close on at the middle of a flat face, but it can come down over the panel's top
    edge and close across its thickness, so both are offered and the caller tries the edge first.
    """
    from omnigibson.tiptop.articulation import handle_point

    drawer = _Link([0.0, -0.25, 0.50], [0.5, 0.25, 0.65])  # 15 cm tall front, opening along +x
    face = handle_point(drawer, [1.0, 0.0, 0.0], opening_sign=+1.0, grip="face")
    edge = handle_point(drawer, [1.0, 0.0, 0.0], opening_sign=+1.0, grip="edge")
    assert face[0] == pytest.approx(0.5) and edge[0] == pytest.approx(0.5), "both on the leading face"
    assert face[2] == pytest.approx(0.575), "the face grip aims at the middle of the panel"
    assert edge[2] == pytest.approx(0.635), "the edge grip aims just below its top, where a jaw can pinch"
    assert edge[2] < 0.65, "and inside the panel, not in the air above it"


# --------------------------------------------------------------- the handle, read off the link's own mesh
def _box(lo, hi, n=4):
    """Vertices on the six faces of a box, ``n`` per edge, like a coarse mesh of a panel or a bar."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    xs, ys, zs = (np.linspace(lo[i], hi[i], n) for i in range(3))
    pts = []
    for x in xs:
        for y in ys:
            pts += [[x, y, lo[2]], [x, y, hi[2]]]
    for x in xs:
        for z in zs:
            pts += [[x, lo[1], z], [x, hi[1], z]]
    for y in ys:
        for z in zs:
            pts += [[lo[0], y, z], [hi[0], y, z]]
    return np.asarray(pts, float)


def _drawer_with_rail():
    """store_honey's drawer as measured (bottom_cabinet/slgzfc, asset scale): a 0.426 deep box whose front panel
    spans 0.428 x 0.133, with a rail 0.376 wide x 0.008 tall x 0.010 deep standing 0.024 m proud, near the top."""
    box = _box([-0.13, -0.214, 0.136], [0.273, 0.214, 0.269], n=6)
    rail = _box([0.287, -0.188, 0.232], [0.297, 0.188, 0.240], n=5)
    return np.concatenate([box, rail])


def test_a_drawer_rail_is_read_as_a_bar_with_a_vertical_jaw():
    """The survey that preceded this called store_honey's drawer front "a 9 mm knife-edge lip, nothing a jaw can
    close around". Sliced into 4 mm layers, the front-most centimetre of that link is a 0.376 x 0.008 strip and the
    full 0.428 x 0.133 face lies 2.4 cm behind it: a rail. The densest layer -- what the survey took as the
    panel -- is the rail's own back face."""
    from omnigibson.tiptop.articulation import handle_on

    h = handle_on(_drawer_with_rail(), lead=[1.0, 0.0, 0.0])
    assert h["kind"] == "bar"
    assert h["proud"] == pytest.approx(0.024, abs=0.003)
    assert h["panel"] == pytest.approx(0.273, abs=0.003), "the panel is the first full-face layer, not the densest"
    assert np.allclose(np.abs(h["jaw"]), [0.0, 0.0, 1.0], atol=1e-6), "a horizontal rail is closed on from above and below"
    assert h["span"] == pytest.approx(0.376, abs=0.005) and np.allclose(np.abs(h["along"]), [0.0, 1.0, 0.0], atol=1e-6)
    assert h["point"][2] == pytest.approx(0.236, abs=0.003), "the grasp is on the rail, not the middle of the face"
    assert 0.287 <= h["point"][0] <= 0.297


def test_a_vertical_bar_on_a_door_is_read_as_a_bar_with_a_horizontal_jaw():
    """fridge/petcxr's right door: a bar 0.035 wide x 0.986 tall standing about 6 cm out from a 0.56 x 1.98 panel."""
    from omnigibson.tiptop.articulation import handle_on

    door = _box([0.40, -0.28, -1.0], [0.44, 0.28, 0.98], n=8)
    bar = _box([0.47, 0.20, -0.25], [0.50, 0.235, 0.736], n=6)
    h = handle_on(np.concatenate([door, bar]), lead=[1.0, 0.0, 0.0])
    assert h["kind"] == "bar"
    assert np.allclose(np.abs(h["jaw"]), [0.0, 1.0, 0.0], atol=1e-6), "closed on from the sides"
    assert np.allclose(np.abs(h["along"]), [0.0, 0.0, 1.0], atol=1e-6) and h["span"] == pytest.approx(0.986, abs=0.01)
    assert h["proud"] == pytest.approx(0.06, abs=0.005)


def test_a_flat_panel_offers_its_face_and_nothing_to_close_around():
    from omnigibson.tiptop.articulation import handle_on

    h = handle_on(_box([0.0, -0.2, 0.0], [0.4, 0.2, 0.15], n=6), lead=[1.0, 0.0, 0.0])
    assert h["kind"] == "flat" and h["jaw"] is None
    assert h["point"][0] == pytest.approx(0.4, abs=0.003), "on the front face"
    assert h["point"][1] == pytest.approx(0.0, abs=0.01) and h["point"][2] == pytest.approx(0.075, abs=0.01)


def test_a_shallow_step_is_a_lip_to_press_on_not_a_bar():
    from omnigibson.tiptop.articulation import handle_on

    panel = _box([0.0, -0.2, 0.0], [0.4, 0.2, 0.15], n=6)
    step = _box([0.40, -0.19, 0.12], [0.41, 0.19, 0.13], n=5)  # 1.0 cm proud: too shallow for the pads
    h = handle_on(np.concatenate([panel, step]), lead=[1.0, 0.0, 0.0])
    assert h["kind"] == "lip" and h["jaw"] is None
    assert h["point"][0] == pytest.approx(0.4, abs=0.003), "the press goes on the panel"


def test_a_trim_strip_beside_a_bar_does_not_widen_the_bar():
    """fridge/dszchb: a full-height trim strip 1 cm in front of the panel next to a bar 4.4 cm out. Read over the
    whole proud set they span the door; read over its front half, only the bar is left."""
    from omnigibson.tiptop.articulation import handle_on

    door = _box([0.30, -0.30, -0.6], [0.32, 0.30, 0.8], n=8)
    trim = _box([0.32, 0.27, -0.6], [0.33, 0.30, 0.8], n=6)  # 1 cm proud, the far edge, full height
    bar = _box([0.35, -0.27, 0.0], [0.364, -0.255, 0.24], n=5)  # 4.4 cm proud, the near edge
    h = handle_on(np.concatenate([door, trim, bar]), lead=[1.0, 0.0, 0.0])
    assert h["kind"] == "bar"
    assert h["extent"][0] < 0.03, "the bar's width, not the door's"
    assert h["span"] == pytest.approx(0.24, abs=0.01)


def test_leading_direction_of_a_drawer_is_its_slide_and_of_a_door_its_first_swing():
    from omnigibson.tiptop.articulation import leading_direction

    assert np.allclose(leading_direction("prismatic", [0.0, 1.0, 0.0], [0, 0, 0], [], travel=-0.3), [0.0, -1.0, 0.0])
    # a door hinged on a vertical axis at the origin, its panel along +y: opening (+z rotation) swings it to -x
    door = _box([-0.01, 0.0, 0.0], [0.01, 0.5, 1.0], n=5)
    lead = leading_direction("revolute", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], door, travel=+1.5)
    assert np.allclose(lead, [-1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(leading_direction("revolute", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], door, travel=-1.5), [1.0, 0.0, 0.0])


def test_a_drawer_the_opening_stance_cannot_reach_is_pulled_from_the_looking_stance():
    """open_container picks its stance because the PULL solves from it, and never checks the arm can get to
    where the pull starts.

    store_honey is the case: "15 of 15 pull waypoints solve (80% of the range)" and then "left_arm_joint4 stopped
    following on the way to the standoff (leg 1 of 1, hand 40.1 cm short)" -- 4 attempts over two builds, every
    one of them. The generic looking stance that this replaced opened the same drawer 3 times out of 3, pulling
    13.0, 12.9 and 30.1 cm (2026-09-13/14). So a standoff failure falls back to standing the old way.
    """
    from omnigibson.tiptop.bench import Episode

    class Spec:
        opens = {}

    class Sim:
        arm = "left"
        n_steps = 0
        video_caption = ""

        def __init__(self, second):
            self.second, self.calls, self.stood = second, [], 0

        def open_container(self, arm, name, fraction=None, joint=None, height=None, stand=True):
            self.calls.append(stand)
            if stand:
                return {"opened": False, "why": "left_arm_joint4 stopped following on the way to the standoff "
                                                "(leg 1 of 1, hand 40.1 cm short)"}
            return self.second

    class Ep:
        open_up = Episode.open_up

        def __init__(self, second):
            self.sim, self.spec, self.records = Sim(second), Spec(), []

        def stand_for(self, *names):
            self.sim.stood += 1
            return {}

    # the fallback rescues it
    ep = Ep({"opened": True, "position": 0.30})
    assert ep.open_up("cabinet.n.01_1") is True
    assert ep.sim.calls == [True, False], "try the pull-solving stance first, then the looking one"
    assert ep.sim.stood == 1, "the fallback has to stand again before it pulls"
    assert len(ep.records) == 2 and ep.records[1]["after_standoff_failed"], "both attempts are recorded, once each"

    # the fallback fails too: one record each, no double entry, and the verdict is False
    ep = Ep({"opened": False, "why": "no grasp on cabinet.n.01_1 solves from here"})
    assert ep.open_up("cabinet.n.01_1") is False
    assert len(ep.records) == 2, "each attempt recorded exactly once"

    # a failure that is NOT the standoff does not trigger a second pull
    class SimOther(Sim):
        def open_container(self, arm, name, fraction=None, joint=None, height=None, stand=True):
            self.calls.append(stand)
            return {"opened": False, "why": "no joint of cabinet.n.01_1 has a handle this arm can reach"}

    ep = Ep(None)
    ep.sim = SimOther(None)
    assert ep.open_up("cabinet.n.01_1") is False
    assert ep.sim.calls == [True], "only a standoff failure is worth standing again for"


def test_a_pressed_grasp_that_worked_is_entered_in_the_hand_record():
    """press_grasp physically takes a flat object -- close_on presses until the assist reports it holds -- but
    nothing wrote the robot's OWN hand record, which note_hands writes only after a PLANNER round. So holding()
    said False, Episode.pick threw the success away, and three tasks lost every flat-object pick (2026-09-15).

    The record is written by note_hands' rule: the knowledge source's localization, and the fingers only when
    nothing can localize it. Not from the simulator's grasp assist, which stays diagnostic (scene.check_hands).
    """
    import numpy as np

    from omnigibson.tiptop.bench import Episode

    class Sim:
        arm = "left"

        def __init__(self, finger, hand=(1.0, 0.0, 0.8)):
            self.held_objects, self.finger, self.hand = {}, finger, np.array(hand)

        tracked_label = staticmethod(lambda n: n.replace(".n.01_", "_"))

        def eef_pose_base(self, arm):
            m = np.eye(4)
            m[:3, 3] = self.hand
            return m

        base_to_world = staticmethod(lambda p: np.asarray(p, float))

        def grasp_sensed(self, arm):
            return self.finger > 0.006

        def hands(self):
            return dict(self.held_objects)

    class Knowledge:
        def __init__(self, center):
            self.center = center

        def localize(self, *names):
            if self.center is None:
                raise KeyError(names[0])
            c = np.asarray(self.center, float)
            return {n: {"center": c, "lo": c - 0.03, "hi": c + 0.03} for n in names}

    class Ep:
        note_pressed_grasp = Episode.note_pressed_grasp

        def __init__(self, sim, know):
            self.sim, self.knowledge = sim, know

    # localized at the hand: recorded, whatever the fingers say (sticky closes them through the object)
    ep = Ep(Sim(finger=0.0), Knowledge([1.02, 0.0, 0.78]))
    assert ep.note_pressed_grasp("book.n.01_1") is True
    assert ep.sim.hands() == {"book_1": "left"}, "a pressed grasp that worked must reach holding()"

    # localized far from the hand: not recorded, so pick reports the failure honestly
    ep = Ep(Sim(finger=0.04), Knowledge([3.0, 0.0, 0.1]))
    assert ep.note_pressed_grasp("book.n.01_1") is False
    assert ep.sim.hands() == {}

    # nothing can localize it: the fingers decide, exactly as note_hands falls back
    ep = Ep(Sim(finger=0.04), Knowledge(None))
    assert ep.note_pressed_grasp("book.n.01_1") is True
    ep = Ep(Sim(finger=0.0), Knowledge(None))
    assert ep.note_pressed_grasp("book.n.01_1") is False


def test_the_way_down_to_a_flat_object_may_pass_through_what_it_rests_on():
    """A bookcase's bounding box covers every shelf in it, so reaching onto a book standing in one swept the
    bookcase and the pressed grasp refused its own approach every time."""
    import inspect

    from omnigibson.tiptop.r1pro import R1ProSim

    src = inspect.getsource(R1ProSim.press_grasp)
    assert "spare" in src.split("\n")[0], "the caller has to be able to name what the object is standing on"
    body = src.split('"""')[-1]
    assert "ignore" in body and "spare" in body, "and the sweep check has to honour it"
    assert "if n != obj.name]" not in body, "the old check spared only the object itself"
    # the CALL, not the prose -- the comment above it explains what mesh=False used to do
    call = [ln for ln in body.split("\n") if "path_hits_scene(" in ln]
    assert call and not any("mesh=False" in ln for ln in call), (
        "mesh=False returns box-level hits and returns early, before the filter that drops floors and ceilings: "
        "48 refusals blamed a ceiling for blocking a downward reach"
    )
