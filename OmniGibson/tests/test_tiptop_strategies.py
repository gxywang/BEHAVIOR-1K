"""The generic task runner and the episode's own judgement of rounds (no simulator answers): task descriptions load,
transfers are ordered as the description says, the hand is freed before a pick, a press holds the object first,
and placements are judged by geometry on localized boxes."""

import numpy as np
import pytest

from omnigibson.tiptop.strategies import STRATEGIES, TASKS_DIR, Runner, TaskSpec, Unreachable, atom, strategy_for


def box(center, half=(0.05, 0.05, 0.05)):
    c, h = np.asarray(center, float), np.asarray(half, float)
    return {"center": c, "lo": c - h, "hi": c + h}


class FakeEpisode:
    """An episode whose rounds are scripted: which picks succeed, which places succeed, and where things are."""

    def __init__(self, boxes, pick_ok=(), place_ok=(), unreachable=(), arms=("left", "right")):
        self.boxes = dict(boxes)
        self.pick_ok, self.place_ok, self.unreachable = set(pick_ok), set(place_ok), set(unreachable)
        self.arms = set(arms)
        self.floor = "floor.n.01_1"
        self.hand = None
        self.calls = []

    # moving
    def stand_for(self, *names):
        self.calls.append(("stand_for", names))
        if any(n in self.unreachable for n in names):
            raise Unreachable(f"no base pose reaches {names}")

    def has_arm(self, arm):
        return arm in self.arms

    # rounds
    def pick(self, bddl):
        self.calls.append(("pick", bddl))
        if bddl in self.pick_ok:
            self.hand = bddl
            return True
        return False

    def achieve(self, atoms, arm="left", floor=None):
        self.calls.append(("achieve", tuple(a["predicate"] for a in atoms), tuple(atoms[0]["args"]), arm))
        a = atoms[0]
        if a["predicate"] == "toggled_on":
            return True
        item, target = a["args"]
        if item in self.place_ok:
            self.hand = None
            return True
        return False

    def put_down(self, bddl, support, floor=None):
        self.calls.append(("put_down", bddl, support))
        self.hand = None
        return True

    def release(self):
        self.calls.append(("release",))
        self.hand = None

    def holding(self, bddl):
        return self.hand == bddl

    def held_names(self):
        return [self.hand] if self.hand else []

    # localization
    def position(self, name):
        return self.boxes[name]["center"]

    def distance(self, a, b):
        return float(np.linalg.norm(self.position(a)[:2] - self.position(b)[:2]))

    def on_support(self, item, support):
        from omnigibson.tiptop.bench import placed_over

        return support == self.floor or placed_over(self.boxes[item], self.boxes[support], from_bottom=False)

    def support_of(self, item):
        for name, b in self.boxes.items():
            if name != item and self.on_support(item, name) and "table" in name:
                return name
        return self.floor

    def edge_gap(self, item, support):
        lo, hi, c = self.boxes[support]["lo"], self.boxes[support]["hi"], self.boxes[item]["center"]
        return float(min(c[0] - lo[0], hi[0] - c[0], c[1] - lo[1], hi[1] - c[1]))


def basket_world():
    table = box((0, 0, 0.7), half=(0.6, 0.4, 0.02))
    boxes = {"table.n.02_1": table}
    # items on the table: candle_1 near the edge, candle_2 in the middle; a cookie
    boxes["candle.n.01_1"] = box((0.55, 0, 0.77))
    boxes["candle.n.01_2"] = box((0.0, 0, 0.77))
    boxes["cookie.n.01_1"] = box((0.2, 0.3, 0.77))
    # baskets on the floor: basket_2 is nearer the table than basket_1
    boxes["basket.n.01_1"] = box((3.0, 0, 0.1), half=(0.15, 0.15, 0.1))
    boxes["basket.n.01_2"] = box((1.2, 0, 0.1), half=(0.15, 0.15, 0.1))
    goal = [
        atom("inside", "candle.n.01_1", "basket.n.01_1"),
        atom("inside", "cookie.n.01_1", "basket.n.01_1"),
        atom("inside", "candle.n.01_2", "basket.n.01_2"),
    ]
    return boxes, goal


def test_task_descriptions_load_from_the_tasks_directory():
    assert set(STRATEGIES) >= {"assembling_gift_baskets", "turning_on_radio"}
    baskets = STRATEGIES["assembling_gift_baskets"]
    assert baskets.plan == "transfer" and baskets.attempts_per_kind == 2
    assert baskets.order == {"containers": "nearest_first", "items": "nearest_edge_first"}
    radio = STRATEGIES["turning_on_radio"]
    assert radio.plan == "press" and radio.press == "hold"
    assert all(path.stem in STRATEGIES for path in TASKS_DIR.glob("*.yaml"))


def test_a_description_with_an_unknown_field_or_plan_is_refused(tmp_path):
    bad = tmp_path / "x.yaml"
    bad.write_text("task: x\ninstruction: do x\nrecovery: magic\n")
    with pytest.raises(ValueError, match="unknown fields"):
        TaskSpec.load(bad)
    bad.write_text("task: x\ninstruction: do x\nplan: fly\n")
    with pytest.raises(ValueError, match="plan must be"):
        TaskSpec.load(bad)


def test_transfers_go_container_by_container_nearest_first_and_items_nearest_the_edge_first():
    boxes, goal = basket_world()
    ep = FakeEpisode(
        boxes,
        pick_ok={"candle.n.01_1", "candle.n.01_2", "cookie.n.01_1"},
        place_ok={"candle.n.01_1", "candle.n.01_2", "cookie.n.01_1"},
    )
    strategy_for("assembling_gift_baskets", goal).run(ep)
    picks = [c[1] for c in ep.calls if c[0] == "pick"]
    # basket_2 (nearer the table) first; it wants a candle: candle_1 is nearest the table edge so it goes first
    assert picks[0] == "candle.n.01_1"
    stands = [c[1] for c in ep.calls if c[0] == "stand_for"]
    assert stands[0] == ("basket.n.01_2",)
    # every container got what it wanted: three placements
    assert len([c for c in ep.calls if c[0] == "achieve"]) == 3
    achieved = [c[2] for c in ep.calls if c[0] == "achieve"]
    assert ("candle.n.01_1", "basket.n.01_2") in achieved  # the nearest candle went to the nearest basket


def test_attempts_per_kind_limits_the_items_tried_per_container():
    boxes, goal = basket_world()
    ep = FakeEpisode(boxes, pick_ok=set(), place_ok=set())  # nothing can be picked
    Runner(STRATEGIES["assembling_gift_baskets"], goal, attempts=1).run(ep)
    picks = [c[1] for c in ep.calls if c[0] == "pick"]
    # basket_2 wants one candle: one attempt; basket_1 wants a candle and a cookie: one attempt each
    assert len(picks) == 3


def test_an_unreachable_container_puts_the_item_back_on_its_support():
    boxes, goal = basket_world()
    ep = FakeEpisode(boxes, pick_ok={"candle.n.01_1"}, unreachable={"basket.n.01_2"})
    Runner(STRATEGIES["assembling_gift_baskets"], goal, attempts=1).run(ep)
    assert ("put_down", "candle.n.01_1", "table.n.02_1") in ep.calls
    assert ep.hand is None


def test_a_full_hand_is_emptied_before_the_next_pick():
    boxes, goal = basket_world()
    ep = FakeEpisode(boxes, pick_ok={"candle.n.01_1"}, place_ok=set())
    ep.hand = "cookie.n.01_1"  # left over from an earlier failed place
    Runner(STRATEGIES["assembling_gift_baskets"], goal, attempts=1).run(ep)
    first = [c for c in ep.calls if c[0] in ("put_down", "pick")][0]
    assert first == ("put_down", "cookie.n.01_1", "floor.n.01_1")


def test_a_press_that_holds_picks_first_and_presses_with_the_other_arm():
    ep = FakeEpisode({}, pick_ok={"radio.n.01_1"})
    strategy_for("turning_on_radio", [atom("toggled_on", "radio.n.01_1")]).run(ep)
    assert ep.calls == [
        ("pick", "radio.n.01_1"),
        ("achieve", ("toggled_on",), ("radio.n.01_1",), "right"),
    ]
    ep = FakeEpisode({}, pick_ok={"radio.n.01_1"}, arms=("left",))
    with pytest.raises(ValueError, match="right-arm planner"):
        strategy_for("turning_on_radio", [atom("toggled_on", "radio.n.01_1")]).run(ep)


def test_placement_geometry_judges_on_and_in_from_boxes():
    from omnigibson.tiptop.bench import placed_over

    basket = box((1.0, 0, 0.1), half=(0.15, 0.15, 0.1))  # bottom at 0, top at 0.2
    inside = box((1.02, 0.03, 0.06))  # bottom at 0.01, within the basket
    on_rim = box((1.0, 0, 0.25))  # bottom at 0.2, resting on the rim
    beside = box((1.3, 0, 0.05))  # centre outside the footprint
    assert placed_over(inside, basket, from_bottom=True)
    assert placed_over(on_rim, basket, from_bottom=True)
    assert not placed_over(beside, basket, from_bottom=True)
    assert not placed_over(inside, basket, from_bottom=False)  # not "on top of" the basket
    assert placed_over(on_rim, basket, from_bottom=False)


def test_the_episode_judges_rounds_without_the_simulator():
    from omnigibson.tiptop.bench import Episode

    class Sim:
        arm = "left"
        held_objects = {}

        def hands(self):
            return dict(self.held_objects)

        def tracked_label(self, name):
            return name.replace(".n.01_", "_")

        def task_scope(self):
            return {"table.n.02_1": None, "candle.n.01_1": None, "basket.n.01_1": None, "agent.n.01_1": None}

    class Knowledge:
        def __init__(self, boxes):
            self.boxes = boxes

        def localize(self, *names):
            return {n: self.boxes[n] for n in names}

    ep = Episode.__new__(Episode)
    ep.sim, ep.floor = Sim(), "floor.n.01_1"
    ep.knowledge = Knowledge(
        {
            "table.n.02_1": box((0, 0, 0.7), half=(0.6, 0.4, 0.02)),
            "candle.n.01_1": box((0.1, 0, 0.77)),
            "basket.n.01_1": box((1.0, 0, 0.1), half=(0.15, 0.15, 0.1)),
        }
    )
    assert ep.support_of("candle.n.01_1") == "table.n.02_1"
    assert ep.on_support("candle.n.01_1", "table.n.02_1")
    assert ep.near_floor("basket.n.01_1") and not ep.near_floor("table.n.02_1")
    assert not ep.placed("candle.n.01_1", "basket.n.01_1")
    ep.knowledge.boxes["candle.n.01_1"] = box((1.02, 0, 0.05))
    assert ep.placed("candle.n.01_1", "basket.n.01_1")
    # holding comes from the hand record, a press from the round having run
    assert not ep.satisfied([atom("holding", "candle.n.01_1")])
    ep.sim.held_objects["candle_1"] = "left"
    assert ep.satisfied([atom("holding", "candle.n.01_1")])
    assert not ep.satisfied([atom("toggled_on", "radio.n.01_1")], record={"error": "no plan"})
    assert ep.satisfied([atom("toggled_on", "radio.n.01_1")], record={"round": 3})
    assert ep.satisfied([atom("inside", "candle.n.01_1", "basket.n.01_1")], record={"round": 4})
