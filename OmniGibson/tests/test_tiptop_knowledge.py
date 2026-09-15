"""Pure-python tests for the knowledge sources, the request keys they set, and the task strategies (no Isaac Sim)."""

import numpy as np
import pytest

from omnigibson.tiptop.knowledge import (
    GoalNotVisible,
    OnboardKnowledge,
    OracleKnowledge,
    make_knowledge,
)
from omnigibson.tiptop.bench import Episode
from omnigibson.tiptop.protocol import attach_knowledge, build_request


def _request():
    h, w = 6, 8
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.ones((h, w), dtype=np.float32)
    K = np.array([[10.0, 0, 4], [0, 10.0, 3], [0, 0, 1]], dtype=np.float32)
    return build_request(rgb, depth, K, np.eye(4, dtype=np.float32), "task", np.zeros(11))


def test_attach_knowledge_sets_only_what_is_known():
    req = _request()
    attach_knowledge(req, ["candle_1"], [{"predicate": "holding", "args": ["candle_1"]}])
    assert req["gt_labels"] == ["candle_1"] and req["gt_atoms"] == [{"predicate": "holding", "args": ["candle_1"]}]
    for key in ("gt_masks", "gt_buttons", "held_labels", "in_hand", "workspace_bounds"):
        assert key not in req
    masks = np.zeros((1, 6, 8), bool)
    masks[0, 1:3, 2:4] = True
    attach_knowledge(
        req,
        ["candle_1"],
        [],
        masks=masks,
        buttons={"radio_1_button": {"position": [1, 0, 0.5], "normal": [0, 1, 0], "radius": 0.02}},
        held=["radio_1"],
        in_hand=["candle_1"],
        workspace=[[0.35, -0.8, -0.05], [1.3, 0.8, 1.6]],
    )
    assert req["gt_masks"].dtype == np.uint8 and req["gt_masks"].sum() == 4
    assert req["gt_buttons"]["radio_1_button"]["radius"] == 0.02
    assert req["held_labels"] == ["radio_1"] and req["in_hand"] == ["candle_1"]
    assert req["workspace_bounds"] == [[0.35, -0.8, -0.05], [1.3, 0.8, 1.6]]
    with pytest.raises(ValueError):
        attach_knowledge(req, ["a", "b"], [], masks=masks)  # one mask for two labels
    with pytest.raises(ValueError):
        attach_knowledge(req, [], [], workspace=[[0, 0, 0], [1, 1, 0]])  # lo == hi
    with pytest.raises(ValueError):
        attach_knowledge(req, [], [], buttons={"b": {"position": [0, 0], "normal": [1, 0, 0], "radius": 0.01}})


class _Sim:
    """The bits of a simulator the knowledge sources read: tracked objects, goal translation, masks (per view),
    buttons, hands."""

    def __init__(self, masks, held=None, arm="left", views=None, furniture=()):
        self.masks = masks  # label -> (H, W) bool, the primary view
        self.views = views or {}  # view name -> {label -> (H, W) bool}, the further views
        self.objects = {label: object() for label in masks if label not in furniture}
        self.held_objects = held or {}
        self.arm = arm
        self.eef = {"left": np.eye(4), "right": np.eye(4)}
        self.button_calls = []
        self.furniture = list(furniture)  # what nearby_obstacles offers with --obstacles on
        self.send_obstacles = False

    def tiptop_goal(self, atoms, category_level):
        def name(bddl):
            category, _, index = bddl.partition(".n.01_")
            return category if category_level else f"{category}_{index}"

        predicates = {"inside": "on", "holding": "holding", "toggled_on": "pressed"}
        out = []
        for atom in atoms:
            args = [name(a) for a in atom["args"]]
            if atom["predicate"] == "toggled_on":
                args = [f"{a}_button" for a in args]
            out.append({"predicate": predicates[atom["predicate"]], "args": args})
        task_labels = [label for label in self.masks if label not in self.furniture]  # furniture is not a goal label
        labels = sorted({name(f"{label.rpartition('_')[0]}.n.01_{label.rpartition('_')[2]}") for label in task_labels})
        return labels, out

    def nearby_obstacles(self, exclude=()):
        return [name for name in self.furniture if name not in exclude]

    def object_meshes(self, labels):
        # the real one (TiptopSim.object_meshes) raises on a label it cannot resolve to a simulated object
        missing = [label for label in labels if label not in self.objects and label not in self.furniture]
        if missing:
            raise ValueError(f"no tracked object for labels {missing}")
        return {}

    def oracle_masks(self, request, extras, labels, meshes=None):
        name = request.get("name") or request.get("view_name", "primary")
        masks = self.views[name] if name in self.views else self.masks
        empty = np.zeros(request["depth"].shape, bool)
        return np.stack([masks.get(label, empty) for label in labels])

    def button_hints(self, atoms, category_level=False):
        self.button_calls.append(list(atoms))
        return {"radio_1_button": {"position": [0.7, 0.2, 0.5], "normal": [0, 1, 0], "radius": 0.022}}

    def eef_pose_base(self, arm):
        return self.eef[arm]

    def hands(self):
        return dict(self.held_objects)

    def workspace(self, floor=False):
        return [[0.35, -0.8, -0.05 if floor else 0.25], [1.3, 0.8, 1.6]]


def _masks(**pixels):
    out = {}
    for label, n in pixels.items():
        m = np.zeros((6, 8), bool)
        m.flat[:n] = True
        out[label] = m
    return out


def test_oracle_knowledge_sends_instance_masks_and_every_button_of_the_task():
    goal = [
        {"predicate": "holding", "args": ["radio.n.01_1"]},
        {"predicate": "toggled_on", "args": ["radio.n.01_1"]},
    ]
    sim = _Sim(_masks(radio_1=20, candle_1=0))
    source = make_knowledge("oracle", sim, goal)
    assert source.privileged and source.report() == {"source": "oracle", "privileged": True}
    known = source.describe([goal[0]], _request(), {})
    assert known.labels == ["radio_1"]  # the candle is out of view and dropped
    assert known.masks.shape == (1, 6, 8) and known.atoms == [{"predicate": "holding", "args": ["radio_1"]}]
    assert "radio_1_button" in known.buttons and sim.button_calls == [goal]  # the whole task's buttons, every round
    assert known.in_hand == [] and known.held_labels == [] and known.workspace == sim.workspace()
    req = _request()
    known.attach(req)
    assert req["gt_masks"].shape == (1, 6, 8) and "in_hand" not in req
    # a goal object with no pixels is an error the caller can act on
    with pytest.raises(GoalNotVisible):
        source.describe([{"predicate": "holding", "args": ["candle.n.01_1"]}], _request(), {})


def test_oracle_masks_come_from_every_view_and_a_goal_object_may_hide_from_the_primary_view():
    from omnigibson.tiptop.protocol import add_view

    goal = [{"predicate": "holding", "args": ["candle.n.01_1"]}]
    sim = _Sim(_masks(radio_1=20, candle_1=0), views={"left_wrist": {"candle_1": np.ones((4, 6), bool)}})
    source = make_knowledge("oracle", sim, goal)
    req = _request()
    req["view_name"] = "head"
    add_view(req, "left_wrist", np.zeros((4, 6, 3), np.uint8), np.ones((4, 6), np.float32), np.eye(3), np.eye(4))
    known = source.describe(goal, req, {"views": {"left_wrist": {}}})
    assert known.labels == ["candle_1", "radio_1"]  # the candle is in the wrist view only
    assert known.masks.shape == (2, 6, 8) and not known.masks[0].any() and known.masks[1].sum() == 20
    assert list(known.view_masks) == ["left_wrist"] and known.view_masks["left_wrist"].shape == (2, 4, 6)
    assert known.summary()["view_mask_pixels"] == {"left_wrist": {"candle_1": 24, "radio_1": 0}}
    known.attach(req)
    assert req["gt_masks"].shape == (2, 6, 8) and req["views"][0]["gt_masks"].shape == (2, 4, 6)
    blind = make_knowledge("oracle", _Sim(_masks(radio_1=20, candle_1=0), views={"left_wrist": {}}), goal)
    with pytest.raises(GoalNotVisible, match="left_wrist"):  # hidden in every view
        blind.describe(goal, req, {"views": {"left_wrist": {}}})


def test_knowledge_reports_the_hands_at_its_own_label_level():
    goal = [{"predicate": "inside", "args": ["candle.n.01_4", "wicker_basket.n.01_2"]}]
    sim = _Sim(_masks(candle_4=10, wicker_basket_2=30, radio_1=5), held={"candle_4": "left", "radio_1": "right"})
    oracle = OracleKnowledge(sim, goal)
    known = oracle.describe(goal, _request(), {}, floor=True)
    assert known.in_hand == ["candle_4"] and known.held_labels == ["radio_1"]
    assert known.workspace == [[0.35, -0.8, -0.05], [1.3, 0.8, 1.6]]  # the embodiment's box, down to the floor
    onboard = OnboardKnowledge(sim, goal)
    known = onboard.describe(goal, _request(), {})
    assert known.in_hand == ["candle"] and known.held_labels == ["radio"]  # category level
    assert known.masks is None and known.labels == ["candle", "radio", "wicker_basket"]
    assert known.atoms == [{"predicate": "on", "args": ["candle", "wicker_basket"]}]


def test_onboard_knowledge_asks_for_buttons_and_carries_detections():
    goal = [{"predicate": "toggled_on", "args": ["radio.n.01_1"]}]
    sim = _Sim(_masks(radio_1=20))
    source = OnboardKnowledge(sim, goal)
    assert not source.privileged
    known = source.describe([{"predicate": "holding", "args": ["radio.n.01_1"]}], _request(), {})
    assert known.labels == ["radio", "radio_button"] and known.buttons == {}  # the detector is asked for the button
    source.learned(
        {
            "buttons": {
                "radio_button": {"position": [0.5, 0.1, 0.8], "normal": [0, 1, 0], "radius": 0.01, "source": "detected"}
            }
        }
    )
    close = np.eye(4)
    close[:3, 3] = [0.5, 0.1, 0.9]
    source.picked("radio_1", "left", close)  # tracked label in, the source maps it to its own level
    sim.eef["left"] = np.eye(4)
    sim.eef["left"][:3, 3] = [0.6, 0.1, 1.0]  # the gripper moved 10 cm in x and up
    known = source.describe(goal, _request(), {})
    assert np.allclose(known.buttons["radio_button"]["position"], [0.6, 0.1, 0.9])


def test_make_knowledge_rejects_unknown_sources():
    with pytest.raises(ValueError):
        make_knowledge("gemini", _Sim({}), [])


# ---------------------------------------------------------------- strategies


class _Episode(Episode):
    """A scripted episode: what each round does is decided up front, and everything the runner asks is recorded.
    The retry policy (``pick``, ``achieve``, ``put_down``, ``satisfied``) is the real one; the robot's readings and
    the localization are scripted. An outcome is a set of tokens: ``held`` (the fingers closed on the item),
    ``placed`` (the item ended in or on the target and left the hand), ``failed`` (the round found no plan); an
    empty set is a round that ran and changed nothing, which is what an open-loop press looks like."""

    def __init__(self, outcomes, arms=("left",), on_table=None, positions=None, unreachable=(), rounds=2):
        self.rounds = rounds  # no Episode.__init__: there is no simulator behind this one
        self.outcomes = list(outcomes)
        self.arms = set(arms)
        self.unreachable = set(unreachable)
        self.true = set()
        self.in_hand = set()
        self.calls = []
        self.on_table = set(on_table or ())
        self.positions = positions or {}
        self.floor = "floor.n.01_1"

    def stand_for(self, *names):
        from omnigibson.tiptop.strategies import Unreachable

        self.calls.append(("stand", names))
        if set(names) & self.unreachable:
            raise Unreachable(f"no base pose reaches {names}")

    def has_arm(self, arm):
        return arm in self.arms

    def plan_and_execute(self, atoms, arm="left", floor=False):
        self.calls.append(("round", atoms[0]["predicate"], tuple(atoms[0]["args"]), arm))
        outcome = self.outcomes.pop(0) if self.outcomes else set()
        if "failed" in outcome:
            return {"error": "TiptopPlanningError: no plan"}
        for a in atoms:
            if a["predicate"] == "holding" and "held" in outcome:
                self.in_hand.add(a["args"][0])
            if a["predicate"] in ("inside", "ontop") and a["args"][0] in self.in_hand and "placed" in outcome:
                self.in_hand.discard(a["args"][0])
                self.true.add((a["predicate"], *a["args"]))
                self.on_table.discard(a["args"][0])
        return {}

    def placed(self, item, target):
        return any((p, item, target) in self.true for p in ("inside", "ontop"))

    def near_floor(self, name):
        return name == self.floor

    def on_support(self, item, support):
        return item in self.on_table

    def holding(self, bddl):
        return bddl in self.in_hand

    def held_names(self):
        return sorted(self.in_hand)

    def release(self):
        self.calls.append(("release",))
        self.in_hand.clear()

    def support_of(self, bddl):
        return "table.n.02_1"

    def distance(self, a, b):
        return float(np.linalg.norm(np.subtract(self.positions[a], self.positions[b])))

    def edge_gap(self, item, support):
        return self.positions[item][0]


def _radio(goal):
    from omnigibson.tiptop.strategies import strategy_for

    return strategy_for("turning_on_radio", goal)


def _baskets(goal, attempts):
    from omnigibson.tiptop.strategies import strategy_for

    return strategy_for("assembling_gift_baskets", goal, attempts=attempts)


def test_turn_on_radio_holds_the_radio_while_pressing():
    goal = [{"predicate": "toggled_on", "args": ["radio_receiver.n.01_1"]}]
    ep = _Episode([{"held"}, set()], arms=("left", "right"))
    _radio(goal).run(ep)
    assert ep.calls[0] == ("stand", ("radio_receiver.n.01_1",))
    assert [c[1:] for c in ep.calls[1:]] == [
        ("holding", ("radio_receiver.n.01_1",), "left"),
        ("toggled_on", ("radio_receiver.n.01_1",), "right"),
    ]
    # a failed pick is retried once from a fresh pose; never a press of the free radio
    ep = _Episode([set(), {"held"}, set()], arms=("left", "right"))
    _radio(goal).run(ep)
    rounds = [c[1:] for c in ep.calls if c[0] == "round"]
    assert rounds == [
        ("holding", ("radio_receiver.n.01_1",), "left"),
        ("holding", ("radio_receiver.n.01_1",), "left"),
        ("toggled_on", ("radio_receiver.n.01_1",), "right"),
    ]
    assert [c for c in ep.calls if c[0] == "stand"] == [("stand", ("radio_receiver.n.01_1",))] * 2
    ep = _Episode([set(), set()], arms=("left", "right"))
    _radio(goal).run(ep)
    assert all(c[1] == "holding" for c in ep.calls if c[0] == "round")  # gives up without a press
    with pytest.raises(ValueError):
        _radio(goal).run(_Episode([]))  # one arm only


def test_turn_on_radio_presses_again_once_and_never_puts_the_radio_down():
    goal = [{"predicate": "toggled_on", "args": ["radio_receiver.n.01_1"]}]
    ep = _Episode([{"held"}, {"failed"}, set()], arms=("left", "right"))  # the first press finds no plan
    _radio(goal).run(ep)
    assert [c[1] for c in ep.calls if c[0] == "round"] == ["holding", "toggled_on", "toggled_on"]
    # two presses with no plan end the task: no put-down and re-pick (README, "Kept out of the pipeline")
    ep = _Episode([{"held"}, {"failed"}, {"failed"}, {"placed"}, {"held"}, set()], arms=("left", "right"))
    _radio(goal).run(ep)
    assert [c[1] for c in ep.calls if c[0] == "round"] == ["holding", "toggled_on", "toggled_on"]
    assert ep.holding("radio_receiver.n.01_1")


def test_episode_rounds_are_the_one_retry_policy():
    from omnigibson.tiptop.strategies import atom

    goal = [{"predicate": "toggled_on", "args": ["radio_receiver.n.01_1"]}]
    ep = _Episode([{"held"}, {"failed"}, {"failed"}, set()], arms=("left", "right"), rounds=3)
    _radio(goal).run(ep)
    assert [c[1] for c in ep.calls if c[0] == "round"] == ["holding", "toggled_on", "toggled_on", "toggled_on"]
    ep = _Episode([set(), {"held"}, set()], arms=("left", "right"), rounds=1)  # one pose, then no press
    _radio(goal).run(ep)
    assert [c[1] for c in ep.calls if c[0] == "round"] == ["holding"]
    # a put-down is done when the hand is empty, wherever the object landed; a press with no plan twice is False
    ep = _Episode([{"held"}, {"placed"}, {"failed"}, {"failed"}])
    assert ep.pick("candle.n.01_1") and ep.put_down("candle.n.01_1", "table.n.02_1")
    assert [c[1] for c in ep.calls if c[0] == "round"] == ["holding", "ontop"]
    assert not ep.achieve([atom("toggled_on", "radio_receiver.n.01_1")], arm="right")
    assert len([c for c in ep.calls if c[0] == "round"]) == 4


def test_assemble_gift_baskets_carries_items_and_tries_the_reachable_ones_first():
    goal = [
        {"predicate": "inside", "args": [f"{kind}.n.01_{i}", f"wicker_basket.n.01_{b}"]}
        for kind in ("candle", "bow")
        for i, b in ((1, 1), (2, 2))
    ]
    positions = {
        "table.n.02_1": (0.0, 0.0),
        "wicker_basket.n.01_1": (3.0, 0.0),
        "wicker_basket.n.01_2": (1.0, 0.0),
        "candle.n.01_1": (0.5, 0.0),  # far from the edge
        "candle.n.01_2": (0.1, 0.0),  # nearest the edge: tried first
        "bow.n.01_1": (0.2, 0.0),
        "bow.n.01_2": (0.3, 0.0),
    }
    on_table = [n for n in positions if not n.startswith(("table", "wicker"))]
    # basket 2 (closer) first: candle_2 picked and placed; bow_1's pick fails twice, bow_2 works; basket 1: candle_1, bow_1
    outcomes = [{"held"}, {"placed"}, set(), set(), {"held"}, {"placed"}, {"held"}, {"placed"}, {"held"}, {"placed"}]
    ep = _Episode(outcomes, on_table=on_table, positions=positions)
    _baskets(goal, 2).run(ep)
    rounds = [c for c in ep.calls if c[0] == "round"]
    assert rounds[0][1:3] == ("holding", ("candle.n.01_2",))
    assert rounds[1][1:3] == ("inside", ("candle.n.01_2", "wicker_basket.n.01_2"))
    assert rounds[2][1:3] == rounds[3][1:3] == ("holding", ("bow.n.01_1",))  # retried once from another pose
    assert rounds[4][1:3] == ("holding", ("bow.n.01_2",))
    assert rounds[5][1:3] == ("inside", ("bow.n.01_2", "wicker_basket.n.01_2"))
    assert rounds[6][1:3] == ("holding", ("candle.n.01_1",)) and rounds[8][1:3] == ("holding", ("bow.n.01_1",))
    stands = [c for c in ep.calls if c[0] == "stand"]
    assert stands[0][1] == ("candle.n.01_2",) and stands[1][1] == ("wicker_basket.n.01_2",)  # pick, then carry
    assert ep.true == {
        ("inside", "candle.n.01_2", "wicker_basket.n.01_2"),
        ("inside", "bow.n.01_2", "wicker_basket.n.01_2"),
        ("inside", "candle.n.01_1", "wicker_basket.n.01_1"),
        ("inside", "bow.n.01_1", "wicker_basket.n.01_1"),
    }


def test_assemble_gift_baskets_puts_a_stuck_item_down():
    goal = [{"predicate": "inside", "args": ["candle.n.01_1", "wicker_basket.n.01_1"]}]
    positions = {"table.n.02_1": (0, 0), "wicker_basket.n.01_1": (2, 0), "candle.n.01_1": (0.1, 0)}
    # the place gets the episode's two rounds, then the item is put down where the robot stands
    ep = _Episode([{"held"}, set(), set(), {"placed"}], on_table=["candle.n.01_1"], positions=positions)
    _baskets(goal, 1).run(ep)
    rounds = [c[1:3] for c in ep.calls if c[0] == "round"]
    assert rounds == [
        ("holding", ("candle.n.01_1",)),
        ("inside", ("candle.n.01_1", "wicker_basket.n.01_1")),
        ("inside", ("candle.n.01_1", "wicker_basket.n.01_1")),
        ("ontop", ("candle.n.01_1", "floor.n.01_1")),
    ]
    assert not ep.in_hand


def test_bench_summary_means_the_q_scores():
    from types import SimpleNamespace

    from omnigibson.tiptop.bench import write_summary

    args = SimpleNamespace(
        task_name="turning_on_radio", mode="public_test", knowledge="oracle", grasping_mode="sticky", rounds=2
    )
    results = [
        {
            "instance_id": 301,
            "q_score": {"final": 1.0},
            "success": True,
            "steps": 900,
            "bench": {"reason": "success", "teleports": 1, "wall_time_s": 60},
        },
        {
            "instance_id": 302,
            "q_score": {"final": 0.0},
            "success": False,
            "steps": 3224,
            "bench": {
                "reason": "timeout",
                "teleports": 2,
                "wall_time_s": 200,
                "goal": {"total": 1, "unsatisfied": ["toggled_on(radio_receiver.n.01_1)"]},
                "rounds": [
                    {"round": 1, "atoms": [{"predicate": "holding", "args": ["radio_receiver.n.01_1"]}], "arm": "left"}
                ],
            },
        },
    ]
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        summary = write_summary(Path(d), args, results, 3224)
        assert summary["mean_q_score"] == 0.5 and summary["successes"] == 1 and len(summary["per_instance"]) == 2
        assert summary["per_instance"][0]["what_failed"] == ""
        assert summary["per_instance"][1]["what_failed"] == (
            "1/1 unsatisfied: toggled_on(radio_receiver_1); timeout; rounds run: holding [left] x1"
        )
        assert (Path(d) / "summary.json").exists()


def test_assemble_gift_baskets_skips_unreachable_items_and_puts_an_item_back_for_an_unreachable_basket():
    goal = [
        {"predicate": "inside", "args": ["candle.n.01_1", "wicker_basket.n.01_1"]},
        {"predicate": "inside", "args": ["candle.n.01_2", "wicker_basket.n.01_1"]},
    ]
    positions = {
        "table.n.02_1": (0, 0),
        "wicker_basket.n.01_1": (2, 0),
        "candle.n.01_1": (0.1, 0),
        "candle.n.01_2": (0.2, 0),
    }
    # candle_1 cannot be reached: skipped without a round; candle_2 is picked, the basket is unreachable: put back
    ep = _Episode(
        [{"held"}, {"placed"}],
        on_table=["candle.n.01_1", "candle.n.01_2"],
        positions=positions,
        unreachable={"candle.n.01_1", "wicker_basket.n.01_1"},
    )
    _baskets(goal, 2).run(ep)
    rounds = [c[1:3] for c in ep.calls if c[0] == "round"]
    assert rounds == [("holding", ("candle.n.01_2",)), ("ontop", ("candle.n.01_2", "table.n.02_1"))]
    assert not ep.in_hand


def test_turn_on_radio_gives_up_when_the_radio_is_unreachable():
    goal = [{"predicate": "toggled_on", "args": ["radio_receiver.n.01_1"]}]
    ep = _Episode([], arms=("left", "right"), unreachable={"radio_receiver.n.01_1"})
    _radio(goal).run(ep)
    assert [c for c in ep.calls if c[0] == "round"] == []


def test_assemble_gift_baskets_frees_a_full_hand_before_the_next_pick():
    goal = [
        {"predicate": "inside", "args": ["candle.n.01_1", "wicker_basket.n.01_1"]},
        {"predicate": "inside", "args": ["bow.n.01_1", "wicker_basket.n.01_1"]},
    ]
    positions = {
        "table.n.02_1": (0, 0),
        "wicker_basket.n.01_1": (2, 0),
        "candle.n.01_1": (0.1, 0),
        "bow.n.01_1": (0.2, 0),
    }
    # the candle's place fails (two rounds) and so does the put-down (two rounds): the next transfer starts by
    # putting it down
    ep = _Episode(
        [{"held"}, set(), set(), set(), set(), {"placed"}, {"held"}, {"placed"}],
        on_table=["candle.n.01_1", "bow.n.01_1"],
        positions=positions,
    )
    _baskets(goal, 1).run(ep)
    rounds = [c[1:3] for c in ep.calls if c[0] == "round"]
    inside, floor = ("inside", ("candle.n.01_1", "wicker_basket.n.01_1")), ("ontop", ("candle.n.01_1", "floor.n.01_1"))
    assert rounds[:5] == [("holding", ("candle.n.01_1",)), inside, inside, floor, floor]
    assert rounds[5] == floor  # freed at the start of the bow's transfer
    assert rounds[6:] == [("holding", ("bow.n.01_1",)), ("inside", ("bow.n.01_1", "wicker_basket.n.01_1"))]
    assert not ep.in_hand


def test_assemble_gift_baskets_releases_an_item_no_plan_can_put_down():
    goal = [
        {"predicate": "inside", "args": ["candle.n.01_1", "wicker_basket.n.01_1"]},
        {"predicate": "inside", "args": ["bow.n.01_1", "wicker_basket.n.01_1"]},
    ]
    positions = {
        "table.n.02_1": (0, 0),
        "wicker_basket.n.01_1": (2, 0),
        "candle.n.01_1": (0.1, 0),
        "bow.n.01_1": (0.2, 0),
    }
    # the candle's place fails, the put-down fails, and at the bow's transfer both put-downs fail too (two rounds
    # each): release
    ep = _Episode(
        [{"held"}] + [set()] * 8 + [{"held"}, {"placed"}],
        on_table=["candle.n.01_1", "bow.n.01_1"],
        positions=positions,
    )
    _baskets(goal, 1).run(ep)
    kinds = [c[0] if c[0] != "round" else c[1] for c in ep.calls]
    assert "release" in kinds
    rounds = [c[1:3] for c in ep.calls if c[0] == "round"]
    assert rounds[-2:] == [("holding", ("bow.n.01_1",)), ("inside", ("bow.n.01_1", "wicker_basket.n.01_1"))]
    assert not ep.in_hand


def test_verdict_caption_tells_success_from_failure_and_lists_what_is_missing():
    from omnigibson.tiptop.bench import verdict_caption

    done = {
        "success": True,
        "q_score": 1.0,
        "satisfied": ["toggled_on(radio_receiver.n.01_1)"],
        "unsatisfied": [],
        "total": 1,
    }
    assert verdict_caption("success", True, done) == "RESULT: SUCCESS  q_score 1  1/1 satisfied"
    missing = [f"inside(candle.n.01_{i}, wicker_basket.n.01_1)" for i in range(1, 6)]
    partial = {"success": False, "q_score": 0.6875, "satisfied": ["x"] * 11, "unsatisfied": missing, "total": 16}
    text = verdict_caption("strategy finished", False, partial)
    first, second = text.split("\n")
    assert first == "RESULT: FAILED (strategy finished)  q_score 0.688  11/16 satisfied"
    assert second == "unsatisfied: " + ", ".join(missing[:3]) + " +2 more"


def test_no_knowledge_source_tells_the_executor_when_a_switch_flips():
    """A press runs its planned stroke: the switch's state is the simulator's to know and no source, oracle
    included, hands it to the executor as a stop signal. The oracle gives perception and localization only."""
    from omnigibson.tiptop.knowledge import KnowledgeSource, OnboardKnowledge, OracleKnowledge

    for cls in (KnowledgeSource, OracleKnowledge, OnboardKnowledge):
        assert not hasattr(cls, "press_done")
    assert {"describe", "localize"} <= set(vars(OracleKnowledge))


def test_the_onboard_source_localizes_from_the_planner_reports():
    """Positions the planner reported (base frame) become world-frame boxes of a nominal extent."""
    from omnigibson.tiptop.knowledge import SEEN_HALF_EXTENT, OnboardKnowledge

    sim = _Sim(_masks(radio_1=20))
    sim.base_to_world = lambda p: np.asarray(p, float) + np.array([10.0, 0.0, 0.0])
    sim.tracked_label = lambda name: name.replace(".n.01_", "_")
    source = OnboardKnowledge(sim, [{"predicate": "toggled_on", "args": ["radio.n.01_1"]}])
    with pytest.raises(KeyError):
        source.localize("radio.n.01_1")
    source.learned({"objects": {"radio": {"position": [1.0, 2.0, 0.5], "movable": True}}})
    box = source.localize("radio.n.01_1")["radio.n.01_1"]
    assert np.allclose(box["center"], [11.0, 2.0, 0.5])
    assert np.allclose(box["hi"] - box["lo"], 2 * SEEN_HALF_EXTENT)


def test_what_failed_names_the_unsatisfied_atoms_the_unreachable_objects_and_the_failed_rounds():
    from omnigibson.tiptop.bench import what_failed

    goal = {
        "total": 16,
        "unsatisfied": [
            "inside(bow.n.08_4, wicker_basket.n.01_3)",
            "inside(candle.n.01_2, wicker_basket.n.01_4)",
            "inside(swiss_cheese.n.01_1, wicker_basket.n.01_4)",
            "inside(butter_cookie.n.01_2, wicker_basket.n.01_4)",
        ],
    }
    rounds = [
        {"stand_for": ["bow.n.08_4"], "error": "no free pose", "step": 10},
        {"stand_for": ["bow.n.08_4"], "error": "no free pose", "step": 20},
        {"round": 1, "atoms": [{"predicate": "holding", "args": ["candle.n.01_2"]}], "arm": "left"},
        {
            "round": 2,
            "atoms": [{"predicate": "inside", "args": ["candle.n.01_2", "wicker_basket.n.01_4"]}],
            "arm": "left",
            "error": "TiptopPlanningError: planning failed: no satisfying particles",
        },
        {"round": 3, "atoms": [{"predicate": "ontop", "args": ["candle.n.01_2", "floor.n.01_1"]}], "arm": "left"},
        {"release": True, "step": 300},
    ]
    result = {"success": False, "bench": {"goal": goal, "reason": "strategy finished", "rounds": rounds}}
    text = what_failed(result)
    assert text.split("; ") == [
        "4/16 unsatisfied: inside(bow_4, wicker_basket_3), inside(candle_2, wicker_basket_4), "
        "inside(swiss_cheese_1, wicker_basket_4) ...",
        "no base pose for bow_4 x2",
        "failed rounds: TiptopPlanningError x1",
        "rounds run: holding [left] x1, ontop [left] x1",
        "released an item x1",
    ]
    assert what_failed({"success": True, "bench": {}}) == ""


def test_a_pick_that_closed_on_the_wrong_thing_opens_the_hand_again():
    """A pick that missed its target but stopped the fingers on SOMETHING leaves the hand shut on whatever else
    was there, and nothing later in a run opens it.

    collecting_aluminum_cans round 1 closed on the goal's own ice bucket and carried it for the rest of the
    episode: six of eight teleports reported the gripper touching it at stances 1.2 m apart, two later rounds
    died on "the left arm starts inside ice_bucket_42, which the planner refuses before it looks at the goal",
    and the bucket filled 82% of the wrist camera. 0.833 -> 0.000 (2026-09-15).

    The reading is proprioception -- the gripper command and how far apart the fingers stopped. The simulator's
    grasp assist knows exactly what is in the hand and is deliberately not asked (scene.check_hands logs the
    disagreement and steers nothing).
    """
    from omnigibson.tiptop.run import HOLD_RADIUS, note_hands

    class Sim:
        arm = "left"
        OPEN = 1.0

        def __init__(self, finger):
            self.held_objects, self.bddl_names = {}, {"can_4": "can.n.01_4"}
            self.finger, self.hand, self.opened = finger, np.array([1.0, 0.0, 0.8]), []

        tracked_label = staticmethod(lambda name: name.replace(".n.01_", "_"))

        def eef_pose_base(self, arm):
            m = np.eye(4)
            m[:3, 3] = self.hand
            return m

        base_to_world = staticmethod(lambda p: np.asarray(p, float))

        def grasp_sensed(self, arm):
            return self.finger > 0.006

        def finger_width(self, arm):
            return self.finger

        def hands(self):
            return dict(self.held_objects)

        def check_hands(self):
            pass

        def hold(self, n_steps, gripper=None):
            self.opened.append((n_steps, gripper))

    class Knowledge:
        def __init__(self, center):
            self.center = center

        def localize(self, *names):
            c = np.asarray(self.center, float)
            return {n: {"center": c, "lo": c - 0.03, "hi": c + 0.03} for n in names}

        def picked(self, *a):
            pass

    class Executor:
        close_eef = np.eye(4)

    pick = [{"predicate": "holding", "args": ["can.n.01_4"]}]
    far = [1.0 + HOLD_RADIUS + 0.5, 0.0, 0.78]  # the can is across the floor, where round 1 threw it

    # the bucket case: target elsewhere, fingers 4.6 cm apart -- the hand has something and must let go
    sim = Sim(finger=0.046)
    note_hands(sim, pick, Executor(), Knowledge(far))
    assert sim.hands() == {}, "the can is not in the hand and must not be recorded as held"
    assert len(sim.opened) == 1 and sim.opened[0][1] == Sim.OPEN, "the hand must be opened"

    # closed on nothing at all: there is nothing to drop, so do not spend the steps
    sim = Sim(finger=0.0)
    note_hands(sim, pick, Executor(), Knowledge(far))
    assert sim.hands() == {} and sim.opened == [], "an empty hand needs no opening"

    # a pick that worked is left alone
    sim = Sim(finger=0.046)
    note_hands(sim, pick, Executor(), Knowledge([1.02, 0.0, 0.78]))
    assert sim.hands() == {"can_4": "left"} and sim.opened == [], "do not drop what the pick actually got"


def test_the_hand_record_comes_from_localization_at_the_hand_with_the_fingers_as_fallback():
    """After a pick the object counts as held when the knowledge source localizes it within HOLD_RADIUS of the
    hand; with sticky grasping the fingers close through the object, so their width is only the fallback when
    nothing can localize it. A placement or an object that left the hand clears the record."""
    from omnigibson.tiptop.run import HOLD_RADIUS, note_hands

    class Sim:
        arm = "left"
        OPEN = 1.0

        def __init__(self):
            self.held_objects = {}
            self.bddl_names = {"candle_1": "candle.n.01_1"}
            self.finger = 0.0
            self.hand = np.array([1.0, 0.0, 0.8])
            self.warnings = []
            self.opened = []  # (steps, gripper) of every hand-opening this run asked for

        def hold(self, n_steps, gripper=None):
            self.opened.append((n_steps, gripper))

        def tracked_label(self, name):
            return name.replace(".n.01_", "_")

        def eef_pose_base(self, arm):
            m = np.eye(4)
            m[:3, 3] = self.hand
            return m

        def base_to_world(self, p):
            return np.asarray(p, float)

        def grasp_sensed(self, arm):
            return self.finger > 0.006

        def finger_width(self, arm):
            return self.finger

        def hands(self):
            return dict(self.held_objects)

        def check_hands(self):
            pass

    class Knowledge:
        def __init__(self, center):
            self.center, self.picked_calls = center, []

        def localize(self, *names):
            if self.center is None:
                raise KeyError(names[0])
            c = np.asarray(self.center, float)
            return {n: {"center": c, "lo": c - 0.03, "hi": c + 0.03} for n in names}

        def picked(self, label, arm, eef):
            self.picked_calls.append(label)

    class Executor:
        close_eef = np.eye(4)

    pick = [{"predicate": "holding", "args": ["candle.n.01_1"]}]
    place = [{"predicate": "inside", "args": ["candle.n.01_1", "basket.n.01_1"]}]
    # localized at the hand: held, whatever the fingers say (sticky grasping closes them through the object)
    sim, know = Sim(), Knowledge([1.02, 0.0, 0.78])
    note_hands(sim, pick, Executor(), know)
    assert sim.hands() == {"candle_1": "left"} and know.picked_calls == ["candle_1"]
    # a placement clears it
    note_hands(sim, place, Executor(), know)
    assert sim.hands() == {}
    # localized far from the hand: not held, even with the fingers on something -- and because the fingers ARE
    # on something, the hand is opened rather than left shut on whatever else it caught (see below)
    sim, know = Sim(), Knowledge([1.0 + HOLD_RADIUS + 0.05, 0.0, 0.78])
    sim.finger = 0.03
    note_hands(sim, pick, Executor(), know)
    assert sim.hands() == {}
    assert sim.opened, "the fingers stopped on something that is not the target: open the hand"
    # nothing can localize it: the fingers decide
    sim, know = Sim(), Knowledge(None)
    sim.finger = 0.03
    note_hands(sim, pick, Executor(), know)
    assert sim.hands() == {"candle_1": "left"}
    sim.finger = 0.0
    note_hands(sim, [], Executor(), know)  # a later plan: the fingers are empty now
    assert sim.hands() == {}
    # an object that drifted away from the hand leaves the record
    sim, know = Sim(), Knowledge([1.0, 0.0, 0.8])
    note_hands(sim, pick, Executor(), know)
    know.center = [3.0, 0.0, 0.1]
    note_hands(sim, [], Executor(), know)
    assert sim.hands() == {}


def test_the_oracle_sends_nearby_furniture_to_the_planner_and_only_what_a_view_actually_shows():
    """--obstacles: the room's furniture reaches cuTAMP through held_labels, which it takes as statics.

    Measured 2026-09-15 on putting_dirty_dishes_in_sink: with the flag on, every round died in the capture with
    "no tracked object for labels ['straight_chair_nntxvr_3', ...]" -- nearby_obstacles named scene objects that
    object_meshes could not resolve, so the request was never built and the planner received nothing at all. The
    labels the oracle adds must be ones the simulator can mask and mesh.
    """
    goal = [{"predicate": "holding", "args": ["bowl.n.01_1"]}]
    masks = _masks(bowl_1=20, booth_xzrpar_2=500, bench_xwphjd_3=0)
    sim = _Sim(masks, furniture=("booth_xzrpar_2", "bench_xwphjd_3"))
    source = make_knowledge("oracle", sim, goal)
    off = source.describe(goal, _request(), {})
    assert off.labels == ["bowl_1"] and off.held_labels == []  # default: the planner hears nothing about the room
    sim.send_obstacles = True
    on = source.describe(goal, _request(), {})  # would raise if a furniture label reached object_meshes unresolved
    assert on.held_labels == ["booth_xzrpar_2"]  # a static obstacle, not a movable
    assert on.labels == ["bowl_1", "booth_xzrpar_2"]  # masked like any other label, so it gets a hull
    assert on.masks.shape == (2, 6, 8)
    # the bench is in the room but in none of the views: it carries no pixels, so no hull, so it is not sent --
    # the hole this leaves in the planner's world is the "plan under occlusion" case, still unsolved
    assert "bench_xwphjd_3" not in on.labels and "bench_xwphjd_3" not in on.held_labels


def test_object_meshes_resolves_a_registered_obstacle_and_still_refuses_an_unknown_label():
    """The lookup the crash above was in: furniture is registered as an obstacle, not tracked as a task object."""
    import types

    from omnigibson.tiptop.scene import TiptopSim

    bowl, booth = object(), object()
    sim = types.SimpleNamespace(objects={"bowl_1": bowl}, obstacles={"booth_xzrpar_2": booth})
    sim.tracked_object = types.MethodType(TiptopSim.tracked_object, sim)
    sim.object_trimesh_world = lambda label: f"mesh of {sim.tracked_object(label)!r}"
    sim.tracked_poses_world = lambda labels=None: {}  # no poses to tag these stand-in meshes with
    meshes = TiptopSim.object_meshes(sim, ["bowl_1", "booth_xzrpar_2"])
    assert meshes["bowl_1"] == f"mesh of {bowl!r}" and meshes["booth_xzrpar_2"] == f"mesh of {booth!r}"
    with pytest.raises(ValueError, match="no tracked object for labels"):
        TiptopSim.object_meshes(sim, ["bowl_1", "sideboard_7"])


def test_nearby_obstacles_registers_the_furniture_and_never_offers_an_object_the_task_already_has():
    """The same booth was offered as an obstacle under its scene name while the task tracked it as a movable:
    ``exclude`` carries tiptop labels ('booth_1'), which never match a scene name ('booth_xzrpar_2')."""
    import types

    from omnigibson.tiptop.r1pro import R1ProSim

    def thing(name, category="furniture"):
        return types.SimpleNamespace(name=name, category=category)

    booth, bench, rug, robot = thing("booth_xzrpar_2"), thing("bench_xwphjd_3"), thing("rug_1", "rug"), thing("robot")
    box = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    far = (np.array([9.0, 9.0, 0.0]), np.array([10.0, 10.0, 1.0]))
    sim = types.SimpleNamespace(
        robot=robot,
        objects={"booth_1": booth},  # the task tracks the booth itself, under its tiptop label
        obstacles={"stale_from_the_last_stance": bench},
        base_pose=lambda: (np.array([0.0, 0.0, 0.0]), None),
        scene_aabbs=lambda: [(booth, *box), (bench, *box), (rug, *box), (thing("sideboard_9"), *far)],
    )
    names = R1ProSim.nearby_obstacles(sim, exclude=["booth_1"])
    assert names == ["bench_xwphjd_3"]  # not the booth (already a movable), not the rug, not the far sideboard
    assert sim.obstacles == {"bench_xwphjd_3": bench}  # rebuilt per stance, and resolvable by object_meshes


def test_an_object_in_the_robots_own_hand_does_not_need_a_mask():
    """The visibility gate already exempts two kinds of label that never carry a mask -- the planner's RANSAC
    support plane and a button named by pose. An object in the GRIPPER is the third, for the same reason: it
    reaches the planner through the in_hand carry feature, not through the picture.

    The head camera cannot see into the gripper and the wrist camera is behind it, so every placement round for
    something already held was refused before it was planned: "goal objects ['tile_3'] are not visible in any
    view ['head', 'left_wrist', 'right_wrist'] (empty masks)" with tile_3 in the hand (2026-09-15).
    """
    import inspect

    from omnigibson.tiptop.knowledge import OracleKnowledge

    src = inspect.getsource(OracleKnowledge)
    gate = src[src.index("exempt = ") : src.index("if needed:")]
    assert "carried" in gate, "a held object must be exempt from the mask gate"
    before = src[: src.index("exempt = ")]
    assert "self.hands()" in before, "and the exemption has to read what the hands actually hold"
