"""Pure-python tests for the knowledge sources, the request keys they set, and the task strategies (no Isaac Sim)."""

import numpy as np
import pytest

from omnigibson.tiptop.knowledge import (
    FLOOR_WORKSPACE,
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
        workspace=FLOOR_WORKSPACE,
    )
    assert req["gt_masks"].dtype == np.uint8 and req["gt_masks"].sum() == 4
    assert req["gt_buttons"]["radio_1_button"]["radius"] == 0.02
    assert req["held_labels"] == ["radio_1"] and req["in_hand"] == ["candle_1"]
    assert req["workspace_bounds"] == FLOOR_WORKSPACE
    with pytest.raises(ValueError):
        attach_knowledge(req, ["a", "b"], [], masks=masks)  # one mask for two labels
    with pytest.raises(ValueError):
        attach_knowledge(req, [], [], workspace=[[0, 0, 0], [1, 1, 0]])  # lo == hi
    with pytest.raises(ValueError):
        attach_knowledge(req, [], [], buttons={"b": {"position": [0, 0], "normal": [1, 0, 0], "radius": 0.01}})


class _Sim:
    """The bits of a simulator the knowledge sources read: tracked objects, goal translation, masks, buttons, hands."""

    def __init__(self, masks, held=None, arm="left"):
        self.masks = masks  # label -> (H, W) bool
        self.objects = {label: object() for label in masks}
        self.held_objects = held or {}
        self.arm = arm
        self.eef = {"left": np.eye(4), "right": np.eye(4)}
        self.button_calls = []

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
        labels = sorted({name(f"{label.rpartition('_')[0]}.n.01_{label.rpartition('_')[2]}") for label in self.masks})
        return labels, out

    def oracle_masks(self, request, extras, labels):
        return np.stack([self.masks[label] for label in labels])

    def button_hints(self, atoms, category_level=False):
        self.button_calls.append(list(atoms))
        return {"radio_1_button": {"position": [0.7, 0.2, 0.5], "normal": [0, 1, 0], "radius": 0.022}}

    def eef_pose_base(self, arm):
        return self.eef[arm]

    def hands(self):
        return dict(self.held_objects)


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
    assert known.in_hand == [] and known.held_labels == [] and known.workspace is None
    req = _request()
    known.attach(req)
    assert req["gt_masks"].shape == (1, 6, 8) and "in_hand" not in req
    # a goal object with no pixels is an error the caller can act on
    with pytest.raises(GoalNotVisible):
        source.describe([{"predicate": "holding", "args": ["candle.n.01_1"]}], _request(), {})


def test_knowledge_reports_the_hands_at_its_own_label_level():
    goal = [{"predicate": "inside", "args": ["candle.n.01_4", "wicker_basket.n.01_2"]}]
    sim = _Sim(_masks(candle_4=10, wicker_basket_2=30, radio_1=5), held={"candle_4": "left", "radio_1": "right"})
    oracle = OracleKnowledge(sim, goal)
    known = oracle.describe(goal, _request(), {}, floor=True)
    assert known.in_hand == ["candle_4"] and known.held_labels == ["radio_1"]
    assert known.workspace == FLOOR_WORKSPACE
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
    """A scripted episode: which rounds succeed is decided up front, and everything a strategy asks is recorded.
    The retry policy (``pick``, ``achieve``, ``put_down``) is the real one; the simulator's answers are scripted."""

    def __init__(self, outcomes, arms=("left",), on_table=None, positions=None, unreachable=(), rounds=2):
        self.rounds = rounds  # no Episode.__init__: there is no simulator behind this one
        self.outcomes = list(outcomes)  # per round, in order: a set of BDDL predicates that hold afterwards
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
        for a in atoms:
            if a["predicate"] == "holding" and "held" in outcome:
                self.in_hand.add(a["args"][0])
            if a["predicate"] in ("inside", "ontop") and a["args"][0] in self.in_hand and "placed" in outcome:
                self.in_hand.discard(a["args"][0])
                self.true.add((a["predicate"], *a["args"]))
                self.on_table.discard(a["args"][0])
            if a["predicate"] == "toggled_on" and "toggled" in outcome:
                self.true.add(("toggled_on", a["args"][0]))
        return {}

    def holds(self, predicate, *args):
        return (predicate, *args) in self.true

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


def test_turn_on_radio_holds_the_radio_while_pressing():
    from omnigibson.tiptop.strategies import TurnOnRadio

    goal = [{"predicate": "toggled_on", "args": ["radio_receiver.n.01_1"]}]
    ep = _Episode([{"held"}, {"toggled"}], arms=("left", "right"))
    TurnOnRadio(goal).run(ep)
    assert ep.calls[0] == ("stand", ("radio_receiver.n.01_1",))
    assert [c[1:] for c in ep.calls[1:]] == [
        ("holding", ("radio_receiver.n.01_1",), "left"),
        ("toggled_on", ("radio_receiver.n.01_1",), "right"),
    ]
    # a failed pick is retried once from a fresh pose; never a press of the free radio
    ep = _Episode([set(), {"held"}, {"toggled"}], arms=("left", "right"))
    TurnOnRadio(goal).run(ep)
    rounds = [c[1:] for c in ep.calls if c[0] == "round"]
    assert rounds == [
        ("holding", ("radio_receiver.n.01_1",), "left"),
        ("holding", ("radio_receiver.n.01_1",), "left"),
        ("toggled_on", ("radio_receiver.n.01_1",), "right"),
    ]
    assert [c for c in ep.calls if c[0] == "stand"] == [("stand", ("radio_receiver.n.01_1",))] * 2
    ep = _Episode([set(), set()], arms=("left", "right"))
    TurnOnRadio(goal).run(ep)
    assert all(c[1] == "holding" for c in ep.calls if c[0] == "round")  # gives up without a press
    with pytest.raises(ValueError):
        TurnOnRadio(goal).run(_Episode([]))  # one arm only


def test_turn_on_radio_presses_again_once_and_never_puts_the_radio_down():
    from omnigibson.tiptop.strategies import TurnOnRadio

    goal = [{"predicate": "toggled_on", "args": ["radio_receiver.n.01_1"]}]
    ep = _Episode([{"held"}, set(), {"toggled"}], arms=("left", "right"))  # the first press misses
    TurnOnRadio(goal).run(ep)
    assert [c[1] for c in ep.calls if c[0] == "round"] == ["holding", "toggled_on", "toggled_on"]
    # two misses end the strategy: no put-down and re-pick (README, "Kept out of the pipeline")
    ep = _Episode([{"held"}, set(), set(), {"placed"}, {"held"}, {"toggled"}], arms=("left", "right"))
    TurnOnRadio(goal).run(ep)
    assert [c[1] for c in ep.calls if c[0] == "round"] == ["holding", "toggled_on", "toggled_on"]
    assert ep.holding("radio_receiver.n.01_1")


def test_episode_rounds_are_the_one_retry_policy():
    from omnigibson.tiptop.strategies import TurnOnRadio, atom

    goal = [{"predicate": "toggled_on", "args": ["radio_receiver.n.01_1"]}]
    ep = _Episode([{"held"}, set(), set(), {"toggled"}], arms=("left", "right"), rounds=3)
    TurnOnRadio(goal).run(ep)
    assert [c[1] for c in ep.calls if c[0] == "round"] == ["holding", "toggled_on", "toggled_on", "toggled_on"]
    ep = _Episode([set(), {"held"}, {"toggled"}], arms=("left", "right"), rounds=1)  # one pose, then no press
    TurnOnRadio(goal).run(ep)
    assert [c[1] for c in ep.calls if c[0] == "round"] == ["holding"]
    # a put-down is done when the hand is empty, wherever the object landed; rounds that run out return False
    ep = _Episode([{"held"}, {"placed"}])
    assert ep.pick("candle.n.01_1") and ep.put_down("candle.n.01_1", "table.n.02_1")
    assert [c[1] for c in ep.calls if c[0] == "round"] == ["holding", "ontop"]
    assert not ep.achieve([atom("toggled_on", "radio_receiver.n.01_1")], arm="right")
    assert len([c for c in ep.calls if c[0] == "round"]) == 4


def test_assemble_gift_baskets_carries_items_and_tries_the_reachable_ones_first():
    from omnigibson.tiptop.strategies import AssembleGiftBaskets

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
    AssembleGiftBaskets(goal, attempts=2).run(ep)
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
    from omnigibson.tiptop.strategies import AssembleGiftBaskets

    goal = [{"predicate": "inside", "args": ["candle.n.01_1", "wicker_basket.n.01_1"]}]
    positions = {"table.n.02_1": (0, 0), "wicker_basket.n.01_1": (2, 0), "candle.n.01_1": (0.1, 0)}
    # the place gets the episode's two rounds, then the item is put down where the robot stands
    ep = _Episode([{"held"}, set(), set(), {"placed"}], on_table=["candle.n.01_1"], positions=positions)
    AssembleGiftBaskets(goal, attempts=1).run(ep)
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
    from omnigibson.tiptop.strategies import AssembleGiftBaskets

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
    AssembleGiftBaskets(goal, attempts=2).run(ep)
    rounds = [c[1:3] for c in ep.calls if c[0] == "round"]
    assert rounds == [("holding", ("candle.n.01_2",)), ("ontop", ("candle.n.01_2", "table.n.02_1"))]
    assert not ep.in_hand


def test_turn_on_radio_gives_up_when_the_radio_is_unreachable():
    from omnigibson.tiptop.strategies import TurnOnRadio

    goal = [{"predicate": "toggled_on", "args": ["radio_receiver.n.01_1"]}]
    ep = _Episode([], arms=("left", "right"), unreachable={"radio_receiver.n.01_1"})
    TurnOnRadio(goal).run(ep)
    assert [c for c in ep.calls if c[0] == "round"] == []


def test_assemble_gift_baskets_frees_a_full_hand_before_the_next_pick():
    from omnigibson.tiptop.strategies import AssembleGiftBaskets

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
    AssembleGiftBaskets(goal, attempts=1).run(ep)
    rounds = [c[1:3] for c in ep.calls if c[0] == "round"]
    inside, floor = ("inside", ("candle.n.01_1", "wicker_basket.n.01_1")), ("ontop", ("candle.n.01_1", "floor.n.01_1"))
    assert rounds[:5] == [("holding", ("candle.n.01_1",)), inside, inside, floor, floor]
    assert rounds[5] == floor  # freed at the start of the bow's transfer
    assert rounds[6:] == [("holding", ("bow.n.01_1",)), ("inside", ("bow.n.01_1", "wicker_basket.n.01_1"))]
    assert not ep.in_hand


def test_assemble_gift_baskets_releases_an_item_no_plan_can_put_down():
    from omnigibson.tiptop.strategies import AssembleGiftBaskets

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
    AssembleGiftBaskets(goal, attempts=1).run(ep)
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


def test_only_the_oracle_source_knows_when_a_switch_flips():
    """The executor's press stop signal comes from the knowledge source: the oracle reads the simulator's switch
    state, the onboard source has no such signal (the press runs to its planned depth)."""
    from omnigibson.tiptop.knowledge import OnboardKnowledge, OracleKnowledge

    sim = _Sim(_masks(radio_1=20))
    sim.toggled_now = {"radio.n.01_1": False}
    sim.toggled = lambda bddl: sim.toggled_now[bddl]
    goal = [{"predicate": "toggled_on", "args": ["radio.n.01_1"]}]
    done = OracleKnowledge(sim, goal).press_done(["radio.n.01_1"])
    assert done() is False
    sim.toggled_now["radio.n.01_1"] = True
    assert done() is True
    assert OnboardKnowledge(sim, goal).press_done(["radio.n.01_1"]) is None


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
