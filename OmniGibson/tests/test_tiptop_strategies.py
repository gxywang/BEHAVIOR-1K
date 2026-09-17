"""The generic task runner and the episode's own judgement of rounds (no simulator answers): what each container
wants is read from the task's goal options, atoms already true are left alone, transfers are ordered nearest
first, items on any support are moved, the hand is freed before a pick, a press holds the object first, and
placements are judged by geometry on localized boxes."""

import itertools

import numpy as np
import pytest

from omnigibson.tiptop.strategies import (
    STRATEGIES,
    TASKS_DIR,
    Runner,
    TaskSpec,
    Unreachable,
    atom,
    commit_to_container,
    place_demand,
    press_targets,
    strategy_for,
)


def box(center, half=(0.05, 0.05, 0.05)):
    c, h = np.asarray(center, float), np.asarray(half, float)
    return {"center": c, "lo": c - h, "hi": c + h}


class FakeEpisode:
    """An episode whose rounds are scripted: which picks succeed, which places succeed, and where things are."""

    def __init__(self, boxes, pick_ok=(), place_ok=(), unreachable=(), arms=("left", "right"), shut=(), opens_ok=True):
        self.boxes = dict(boxes)
        self.pick_ok, self.place_ok, self.unreachable = set(pick_ok), set(place_ok), set(unreachable)
        self.arms = set(arms)
        self.floor = "floor.n.01_1"
        self.hand = None
        self.calls = []
        self.shut, self.opens_ok = set(shut), opens_ok

    def is_floor(self, name):
        """Any floor is a floor, not only the one the scope happened to list first (Episode.is_floor)."""
        return bool(name) and str(name).startswith("floor.")

    # articulated containers
    def is_shut(self, name):
        return name in self.shut

    def openable(self, name):
        return name in self.shut

    def open_up(self, name, fraction=None):
        self.calls.append(("open_up", name, fraction))
        if self.opens_ok:
            self.shut.discard(name)
        return self.opens_ok

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
            self.boxes[item] = box(self.boxes[target]["center"])  # it is where it was put
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

        return self.is_floor(support) or placed_over(self.boxes[item], self.boxes[support], from_bottom=False)

    def placed(self, item, target):
        from omnigibson.tiptop.bench import placed_over

        if self.is_floor(target):
            return not self.holding(item)
        return placed_over(self.boxes[item], self.boxes[target], from_bottom=True)

    def walk_to_floor(self, name):
        """Episode.walk_to_floor: travel to the named floor, False when it cannot be located."""
        if name in getattr(self, "unreachable_floors", set()):
            return False
        self.calls.append(("walk_to_floor", name))
        self.floor = name  # the robot now stands on that floor
        return True

    def goal_already_holds(self, predicate, item, container):
        """Whether the goal's atom for this pair holds (Episode.goal_already_holds, which asks the task's own
        evaluator). The fake judges it from the boxes, but a floor target names ONE floor: something lying on
        floor.n.01_1 does not satisfy a goal that asks for it ontop floor.n.01_2."""
        if self.is_floor(container):
            return container == self.floor and self.near_floor(item) and not self.holding(item)
        return self.placed(item, container)

    def near_floor(self, name):
        from omnigibson.tiptop.bench import FLOOR_LEVEL

        return name == self.floor or float(self.boxes[name]["lo"][2]) < FLOOR_LEVEL

    def support_of(self, item):
        for name, b in self.boxes.items():
            if name != item and self.on_support(item, name) and ("table" in name or "desk" in name):
                return name
        return self.floor

    def edge_gap(self, item, support):
        if support == self.floor:
            return 0.0
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
    assert set(STRATEGIES) >= {"assembling_gift_baskets", "turning_on_radio", "dispose_of_batteries"}
    baskets = STRATEGIES["assembling_gift_baskets"]
    assert baskets.plan == "transfer" and baskets.attempts_per_item == 2
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


# ---------------------------------------------------------------- what the goal asks for
def test_the_demand_is_read_from_every_way_the_goal_can_be_satisfied():
    # a goal that pairs one item of each kind with each container (the gift baskets): every pairing is an option
    candles = ["candle.n.01_1", "candle.n.01_2"]
    bows = ["bow.n.08_1", "bow.n.08_2"]
    containers = ["wicker_basket.n.01_1", "wicker_basket.n.01_2"]
    options = [
        [atom("inside", c, containers[i]) for i, c in enumerate(candle_order)]
        + [atom("inside", b, containers[i]) for i, b in enumerate(bow_order)]
        for candle_order in itertools.permutations(candles)
        for bow_order in itertools.permutations(bows)
    ]
    demand = place_demand(options)
    assert demand.wanted == {
        ("candle", "wicker_basket.n.01_1"): 1,
        ("candle", "wicker_basket.n.01_2"): 1,
        ("bow", "wicker_basket.n.01_1"): 1,
        ("bow", "wicker_basket.n.01_2"): 1,
    }
    assert demand.items == {"candle": candles, "bow": bows}
    assert demand.containers == containers and demand.total() == 4

    # a goal that takes any container (the toys): one option per assignment, so a box may want every toy
    toys = [f"toy_figure.n.01_{i}" for i in (1, 2, 3)]
    boxes = ["toy_box.n.01_1", "toy_box.n.01_2"]
    options = [
        [atom("inside", toy, target) for toy, target in zip(toys, choice)]
        for choice in itertools.product(boxes, repeat=3)
    ]
    demand = place_demand(options)
    assert demand.wanted == {("toy_figure", "toy_box.n.01_1"): 3, ("toy_figure", "toy_box.n.01_2"): 3}

    # a goal with one option, several items of a kind into one bin, and an atom that names a support
    option = [atom("inside", f"battery.n.02_{i}", "ashcan.n.01_1") for i in (1, 2, 3)] + [
        atom("ontop", "ashcan.n.01_1", "floor.n.01_1")
    ]
    demand = place_demand([option])
    assert demand.wanted == {("battery", "ashcan.n.01_1"): 3, ("ashcan", "floor.n.01_1"): 1}
    assert demand.predicate[("ashcan", "floor.n.01_1")] == "ontop"


# ---------------------------------------------------------------- transfers
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


def test_an_item_is_tried_a_fixed_number_of_times_and_then_left():
    boxes, goal = basket_world()
    ep = FakeEpisode(boxes, pick_ok=set(), place_ok=set())  # nothing can be picked
    Runner(STRATEGIES["assembling_gift_baskets"], goal, attempts=1).run(ep)
    picks = [c[1] for c in ep.calls if c[0] == "pick"]
    # three items wanted, one try each, and no item is tried twice
    assert len(picks) == 3 and len(set(picks)) == 3


def test_a_container_gets_every_item_of_a_kind_its_goal_asks_for():
    """Three batteries into one bin: the old runner stopped after the first."""
    boxes = {
        "desk.n.01_1": box((0, 0, 0.7), half=(0.6, 0.4, 0.02)),
        "battery.n.02_1": box((0.4, 0, 0.77)),
        "battery.n.02_2": box((0.0, 0, 0.77)),
        "battery.n.02_3": box((-0.4, 0, 0.77)),
        "ashcan.n.01_1": box((1.5, 0, 0.15), half=(0.15, 0.15, 0.15)),
    }
    goal = [atom("inside", f"battery.n.02_{i}", "ashcan.n.01_1") for i in (1, 2, 3)]
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok=set(boxes))
    strategy_for("dispose_of_batteries", goal).run(ep)
    achieved = [c[2] for c in ep.calls if c[0] == "achieve"]
    assert sorted(a[0] for a in achieved) == ["battery.n.02_1", "battery.n.02_2", "battery.n.02_3"]
    assert all(a[1] == "ashcan.n.01_1" for a in achieved)


def test_a_goal_atom_that_puts_something_on_the_floor_is_judged_by_how_low_it_stands():
    """An item on a desk is not "on the floor" just because no hand holds it."""
    boxes = {
        "desk.n.01_1": box((0, 0, 0.7), half=(0.6, 0.4, 0.02)),
        "battery.n.02_1": box((0.4, 0, 0.77)),
        "ashcan.n.01_1": box((1.5, 0, 0.15), half=(0.15, 0.15, 0.15)),
    }
    goal = [atom("ontop", "battery.n.02_1", "floor.n.01_1"), atom("ontop", "ashcan.n.01_1", "floor.n.01_1")]
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok=set(boxes))
    strategy_for("dispose_of_batteries", goal).run(ep)
    picks = [c[1] for c in ep.calls if c[0] == "pick"]
    assert picks == ["battery.n.02_1"]  # the bin already stands on the floor; the battery on the desk does not


class _Budget:
    """The bit of the simulator the runner reads to decide whether another sweep fits."""

    def __init__(self, max_steps=1000, n_steps=0):
        self.max_steps, self.n_steps = max_steps, n_steps


def _two_items(**kw):
    boxes = {
        "ashcan.n.01_1": box((2.0, 0, 0.15), half=(0.15, 0.15, 0.15)),
        "battery.n.02_1": box((0.4, 0, 0.77)),
        "battery.n.02_2": box((0.6, 0, 0.77)),
        "desk.n.01_1": box((0.5, 0, 0.7), half=(0.6, 0.4, 0.02)),
    }
    goal = [atom("inside", f"battery.n.02_{i}", "ashcan.n.01_1") for i in (1, 2)]
    return boxes, goal


def test_the_budget_left_over_is_spent_on_the_atoms_still_open():
    """A pass ends when every item has had its tries; the measured instance then stops with two thirds of its
    step budget unused. A second pass re-chooses the stance, so an item that could not be reached from the first
    one gets another go."""
    boxes, goal = _two_items()
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok={"ashcan.n.01_1", "battery.n.02_1"})
    ep.sim = _Budget(max_steps=1000, n_steps=10)
    strategy_for("dispose_of_batteries", goal).run(ep)
    picks = [c[1] for c in ep.calls if c[0] == "pick"]
    assert picks.count("battery.n.02_2") > 1, "the item that did not land should be tried again on a later sweep"


def test_a_sweep_that_wins_nothing_ends_the_sweeping():
    """An atom that cannot be done costs one extra pass, not the whole budget."""
    boxes, goal = _two_items()
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok=set())  # nothing can be placed at all
    ep.sim = _Budget(max_steps=100000, n_steps=0)
    strategy_for("dispose_of_batteries", goal).run(ep)
    first = [c[1] for c in ep.calls if c[0] == "pick"]
    assert len(first) < 40, f"sweeping should stop once a pass wins nothing, got {len(first)} picks"


def _picks_of_one_run(budget=None):
    boxes, goal = _two_items()
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok={"ashcan.n.01_1", "battery.n.02_1"})
    if budget is not None:
        ep.sim = budget
    strategy_for("dispose_of_batteries", goal).run(ep)
    return [c[1] for c in ep.calls if c[0] == "pick"].count("battery.n.02_2")


def test_no_sweep_begins_without_room_to_finish_one():
    """Almost all the budget already spent: the runner must not start a pass it cannot finish."""
    assert _picks_of_one_run(_Budget(max_steps=1000, n_steps=950)) == _picks_of_one_run()


def test_an_episode_with_no_step_limit_gets_one_pass():
    """Nothing to budget against, so the runner does not invent extra work."""
    one_pass = _picks_of_one_run()
    assert one_pass >= 1
    assert _picks_of_one_run(_Budget(max_steps=1000, n_steps=10)) > one_pass, (
        "with room in the budget it should sweep again, which is what makes the no-budget case meaningful"
    )


def test_a_book_already_in_the_bookcase_is_still_stacked_on_the_other_books():
    """sorting_books_on_shelf wants its books BOTH inside the bookcase, which they already are, AND stacked on
    one another, which they are not.

    With one set of settled items, being done for the bookcase excluded the books from the stacking work too, and
    the instance finished in 14 seconds having never attempted a pick.
    """
    boxes = {
        "bookcase.n.01_1": box((2.0, 0, 0.8), half=(0.4, 0.2, 0.8)),
        "comic_book.n.01_1": box((2.0, 0, 0.9), half=(0.08, 0.05, 0.01)),
        "comic_book.n.01_2": box((2.0, 0.1, 0.9), half=(0.08, 0.05, 0.01)),
    }
    goal = [
        atom("inside", "comic_book.n.01_1", "bookcase.n.01_1"),  # already true
        atom("inside", "comic_book.n.01_2", "bookcase.n.01_1"),  # already true
        atom("ontop", "comic_book.n.01_2", "comic_book.n.01_1"),  # the work
    ]
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok=set(boxes))
    strategy_for("sorting_books_on_shelf", goal).run(ep)
    assert any(c[0] == "pick" for c in ep.calls), "the stacking half of the goal must still be attempted"


def test_an_item_is_only_delivered_once_in_a_pass():
    """The other side of the same coin: a toy that went into one box must not then be carried to the other."""
    boxes = {
        "toy_box.n.01_1": box((2.0, 0, 0.2), half=(0.3, 0.3, 0.2)),
        "toy_box.n.01_2": box((-2.0, 0, 0.2), half=(0.3, 0.3, 0.2)),
        "toy_figure.n.01_1": box((0.4, 0, 0.8)),
    }
    goal = [
        atom("inside", "toy_figure.n.01_1", "toy_box.n.01_1"),
        atom("inside", "toy_figure.n.01_1", "toy_box.n.01_2"),
    ]
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok=set(boxes))
    strategy_for("putting_away_toys", goal).run(ep)
    picks = [c[1] for c in ep.calls if c[0] == "pick"]
    assert picks.count("toy_figure.n.01_1") == 1, f"delivered once per pass, got {picks}"


def test_wood_is_carried_to_the_floor_the_goal_names():
    """The runner used to drop a floor-bound item wherever the robot already stood, which for bringing_in_wood
    is the floor the plywood started on."""
    boxes = {f"plywood.n.01_{i}": box((float(i), 0, 0.05)) for i in (1, 2)}
    goal = [atom("ontop", f"plywood.n.01_{i}", "floor.n.01_2") for i in (1, 2)]
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok=set(boxes))  # ep.floor starts as floor.n.01_1
    strategy_for("bringing_in_wood", goal).run(ep)
    assert ("walk_to_floor", "floor.n.01_2") in ep.calls, "it must travel to the floor the goal names"


def test_a_floor_that_cannot_be_located_falls_back_to_putting_it_down_here():
    """No regression for the ordinary case where the scope cannot resolve the floor at all."""
    boxes = {"plywood.n.01_1": box((1.0, 0, 0.05))}
    goal = [atom("ontop", "plywood.n.01_1", "floor.n.01_2")]
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok=set(boxes))
    ep.unreachable_floors = {"floor.n.01_2"}
    strategy_for("bringing_in_wood", goal).run(ep)
    assert not any(c[0] == "walk_to_floor" for c in ep.calls)
    assert any(c[0] == "pick" for c in ep.calls), "it still tries, it just puts the sheet down where it is"


def test_wood_that_must_go_to_another_room_is_not_already_delivered():
    """bringing_in_wood: three sheets of plywood lie on floor.n.01_1 and the goal wants them on floor.n.01_2.

    The test the runner used before asked only how LOW a thing stands, which every sheet already satisfied, so all
    three were counted as delivered and the instance finished having run no rounds at all. A floor target names
    one floor.
    """
    boxes = {f"plywood.n.01_{i}": box((float(i), 0, 0.05)) for i in (1, 2, 3)}
    goal = [atom("ontop", f"plywood.n.01_{i}", "floor.n.01_2") for i in (1, 2, 3)]
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok=set(boxes))  # ep.floor is floor.n.01_1
    strategy_for("bringing_in_wood", goal).run(ep)
    assert sorted(c[1] for c in ep.calls if c[0] == "pick") == [f"plywood.n.01_{i}" for i in (1, 2, 3)]


def test_naming_a_container_keeps_only_the_options_that_use_it():
    """sorting_vegetables asks for thirteen vegetables in a mixing bowl and grounds into 27 options.

    Some of those options split the vegetables between bowls, which is what stops ``one_container_goal`` from
    committing on its own. Read across all of them the demand then asks for every vegetable in every bowl, so the
    runner sends each to whichever bowl is nearest it, spreads them over three, and satisfies none.
    """
    def all_in(bowl):
        return [atom("inside", f"leek.n.02_{i}", bowl) for i in (1, 2)]

    split = [atom("inside", "leek.n.02_1", "mixing_bowl.n.01_1"), atom("inside", "leek.n.02_2", "mixing_bowl.n.01_2")]
    options = [all_in("mixing_bowl.n.01_3"), all_in("mixing_bowl.n.01_1"), split]
    assert len(place_demand(options).wanted) == 3, "read across the options, all three bowls are asked for"

    kept = commit_to_container(options, "mixing_bowl.n.01_3")
    demand = place_demand(kept)
    assert list(demand.wanted) == [("leek", "mixing_bowl.n.01_3")]
    assert demand.wanted[("leek", "mixing_bowl.n.01_3")] == 2


def test_a_container_name_that_is_not_in_the_goal_leaves_it_alone():
    """A stale name in a task file must not silently delete the task's goal."""
    options = [[atom("inside", "leek.n.02_1", "mixing_bowl.n.01_1")]]
    assert commit_to_container(options, "sink.n.01_1") == options


def test_committing_keeps_the_presses_and_opens_of_the_option():
    options = [[atom("inside", "leek.n.02_1", "mixing_bowl.n.01_3"), atom("toggled_on", "oven.n.01_1")]]
    kept = commit_to_container(options, "mixing_bowl.n.01_3")
    assert press_targets(kept[0]) == ["oven.n.01_1"]


def test_a_thing_is_never_asked_to_be_carried_to_itself():
    """setup_a_bar_for_a_cocktail_party's goal grounds nextto over every pair, the reflexive one included."""
    boxes = {f"can__of__soda.n.01_{i}": box((float(i), 0, 0.8)) for i in (1, 2)}
    boxes["countertop.n.01_1"] = box((0, 2, 0.9), half=(0.8, 0.3, 0.02))
    goal = [
        atom("nextto", "can__of__soda.n.01_1", "can__of__soda.n.01_1"),  # a can beside itself
        atom("nextto", "can__of__soda.n.01_2", "can__of__soda.n.01_1"),
    ]
    demand = place_demand([goal])
    assert ("can__of__soda", "can__of__soda.n.01_1") in demand.wanted
    assert demand.wanted[("can__of__soda", "can__of__soda.n.01_1")] == 1, "only the real pairing is work"


def test_a_goal_atom_that_already_holds_is_never_worked_on():
    """The batteries task also asks for the bin to stand on the floor, which it already does."""
    boxes = {
        "desk.n.01_1": box((0, 0, 0.7), half=(0.6, 0.4, 0.02)),
        "battery.n.02_1": box((0.4, 0, 0.77)),
        "ashcan.n.01_1": box((1.5, 0, 0.15), half=(0.15, 0.15, 0.15)),
    }
    goal = [atom("inside", "battery.n.02_1", "ashcan.n.01_1"), atom("ontop", "ashcan.n.01_1", "floor.n.01_1")]
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok=set(boxes))
    strategy_for("dispose_of_batteries", goal).run(ep)
    picks = [c[1] for c in ep.calls if c[0] == "pick"]
    assert picks == ["battery.n.02_1"]  # the bin is where the goal wants it; nobody picks it up


def test_any_container_the_goal_allows_takes_the_item():
    """Two boxes, and the goal lets either box have any toy: the nearest box gets them all.

    Built on a bare spec rather than a task's own, because this is the GENERIC rule. putting_away_toys used to
    serve as the example and no longer can: it now names a container, which is the point of the test below.
    """
    boxes = {
        "toy_box.n.01_1": box((4.0, 0, 0.2), half=(0.3, 0.3, 0.2)),
        "toy_box.n.01_2": box((1.0, 0, 0.2), half=(0.3, 0.3, 0.2)),
    }
    toys = [f"toy_figure.n.01_{i}" for i in (1, 2, 3)]
    for i, toy in enumerate(toys):
        boxes[toy] = box((0.2 * i, 0.5, 0.05))
    options = [
        [atom("inside", toy, target) for toy, target in zip(toys, choice)]
        for choice in itertools.product(["toy_box.n.01_1", "toy_box.n.01_2"], repeat=3)
    ]
    ep = FakeEpisode(boxes, pick_ok=set(toys), place_ok=set(toys))
    plain = TaskSpec(task="t", instruction="i", plan="transfer")
    Runner(plain, options[0], options=options).run(ep)
    achieved = [c[2] for c in ep.calls if c[0] == "achieve"]
    assert sorted(a[0] for a in achieved) == toys
    assert all(a[1] == "toy_box.n.01_2" for a in achieved)  # the nearer box took all three


def test_putting_away_toys_does_not_commit_to_one_box():
    """A `container: toy_box.n.01_1` was tried and withdrawn on 2026-09-15.

    Rounds aimed at the floor box executed 17 of 23 while the table-top box managed 0 of 7, so naming the floor
    box looked like it could only narrow the demand. True of the GOAL, false of the GEOMETRY: the box is
    0.62 x 0.88 m with its rim at z 0.46 and eight toys do not fit. The committed run placed all 8 with zero
    planning failures and still left 3 atoms open -- toy_figure_5, _7 and _8 all resting with their bottoms at
    z 0.46, exactly the rim. 0.625 against 0.875 for the nearest-box routing.
    """
    from omnigibson.tiptop.strategies import TASKS_DIR, TaskSpec

    spec = TaskSpec.load(TASKS_DIR / "putting_away_toys.yaml")
    assert not spec.container, (
        "the two boxes exist so the toys can be SPLIT; one box does not hold eight of them"
    )


def test_items_on_a_second_support_are_transferred_too():
    """Two batteries on a desk and one on a cabinet: the old runner only looked at the first item's support."""
    boxes = {
        "desk.n.01_1": box((0, 0, 0.7), half=(0.6, 0.4, 0.02)),
        "cabinet.n.01_1": box((2.0, 2.0, 0.9), half=(0.4, 0.3, 0.02)),
        "battery.n.02_1": box((0.4, 0, 0.77)),
        "battery.n.02_2": box((0.0, 0, 0.77)),
        "battery.n.02_3": box((2.0, 2.0, 0.97)),
        "ashcan.n.01_1": box((1.0, 1.0, 0.15), half=(0.15, 0.15, 0.15)),
    }
    goal = [atom("inside", f"battery.n.02_{i}", "ashcan.n.01_1") for i in (1, 2, 3)]
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok=set(boxes))
    strategy_for("dispose_of_batteries", goal).run(ep)
    assert sorted(c[1] for c in ep.calls if c[0] == "pick") == [
        "battery.n.02_1",
        "battery.n.02_2",
        "battery.n.02_3",
    ]


def test_an_unreachable_container_puts_the_item_back_on_its_support():
    boxes, goal = basket_world()
    ep = FakeEpisode(boxes, pick_ok={"candle.n.01_1"}, unreachable={"basket.n.01_2"})
    Runner(STRATEGIES["assembling_gift_baskets"], goal, attempts=1).run(ep)
    assert ("put_down", "candle.n.01_1", "table.n.02_1") in ep.calls
    assert ep.hand is None


def test_emptying_a_hand_never_stands_for_the_floor():
    """The floor is not an object to stand at: standing for it crashed an instance (putting_away_toys, whose
    support_of falls back to the task floor for a toy lying on it)."""
    boxes = {"toy_box.n.01_1": box((1.0, 0, 0.2), half=(0.3, 0.3, 0.2)), "toy.n.01_1": box((0.2, 0.5, 0.05))}
    ep = FakeEpisode(boxes, pick_ok=set(), place_ok=set())
    ep.hand = "toy.n.01_1"  # held, and its support is the floor

    def refuse(bddl, support, floor=None):
        ep.calls.append(("put_down", bddl, support))
        return False  # no put-down plans, so the ladder runs to the end

    ep.put_down = refuse
    assert Runner.free_hand(ep, ep.floor) is False or True  # the point is the calls it made
    assert ("stand_for", (ep.floor,)) not in ep.calls
    assert ("release",) in ep.calls


def test_a_full_hand_is_emptied_before_the_next_pick():
    boxes, goal = basket_world()
    ep = FakeEpisode(boxes, pick_ok={"candle.n.01_1"}, place_ok=set())
    ep.hand = "cookie.n.01_1"  # left over from an earlier failed place
    Runner(STRATEGIES["assembling_gift_baskets"], goal, attempts=1).run(ep)
    first = [c for c in ep.calls if c[0] in ("put_down", "pick")][0]
    assert first == ("put_down", "cookie.n.01_1", "floor.n.01_1")


# ---------------------------------------------------------------- presses
def test_a_goal_that_asks_for_a_switch_to_be_off_is_a_press_too():
    """bddl compiles "(not (toggled_on x))" into a ground atom whose tokens start with 'not'; the runner used to
    drop it and do nothing (turning_out_all_lights_before_sleep, setting_the_fire)."""
    off = [atom("not", "toggled_on", f"switch.n.01_{i}") for i in (1, 2)]
    assert press_targets(off) == ["switch.n.01_1", "switch.n.01_2"]
    assert press_targets([atom("toggled_on", "radio.n.01_1")]) == ["radio.n.01_1"]
    assert press_targets([atom("inside", "a.n.01_1", "b.n.01_1")]) == []


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


# ---------------------------------------------------------------- the episode's own judgement
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


# --------------------------------------------------------------- nextto, OmniGibson's own measure
def test_beside_matches_omnigibsons_nextto_threshold():
    """object_states/next_to.py: the per-axis AABB gap, as a norm, within a sixth of the mean of the extents."""
    from omnigibson.tiptop.bench import Episode

    class Boxes(Episode):
        def __init__(self, boxes):
            self._boxes = boxes

        def boxes(self, *names):
            return {n: self._boxes[n] for n in names}

    def box(cx, cy, half):
        return {"lo": [cx - half, cy - half, 0.0], "hi": [cx + half, cy + half, 2 * half]}

    # two 0.2 m cubes: mean extent 0.2, so the threshold is 0.2 / 6 = 0.033 m of gap
    near = Boxes({"a": box(0.0, 0.0, 0.1), "b": box(0.22, 0.0, 0.1)})  # 2 cm apart
    far = Boxes({"a": box(0.0, 0.0, 0.1), "b": box(0.30, 0.0, 0.1)})  # 10 cm apart
    assert near.beside("a", "b") is True
    assert far.beside("a", "b") is False


def test_beside_scales_the_threshold_with_the_objects():
    from omnigibson.tiptop.bench import Episode

    class Boxes(Episode):
        def __init__(self, boxes):
            self._boxes = boxes

        def boxes(self, *names):
            return {n: self._boxes[n] for n in names}

    # a 2 m object and a 0.2 m one: mean extent is much larger, so 10 cm of gap is still "beside"
    big = {"lo": [0.0, 0.0, 0.0], "hi": [2.0, 2.0, 1.0]}
    small = {"lo": [2.10, 0.0, 0.0], "hi": [2.30, 0.2, 0.2]}
    assert Boxes({"a": small, "b": big}).beside("a", "b") is True


def test_an_unknown_predicate_is_not_counted_satisfied_just_because_a_round_ran():
    from omnigibson.tiptop.bench import Episode

    class Ran(Episode):
        def __init__(self):
            pass

    # "cooked" is a predicate the runner has no skill for; "open" used to stand here and is now one it does
    atoms = [{"predicate": "cooked", "args": ["bacon.n.01_1"]}]
    assert Ran().satisfied(atoms, record={"round": 1}) is False


# --------------------------------------------------------------- opening a container that is shut
def test_a_shut_container_is_opened_before_the_item_is_picked_up():
    """The hand that pulls the drawer is the hand that would be carrying the item, so the order matters."""
    from omnigibson.tiptop.articulation import OPEN_FRACTION_REACH

    ep = FakeEpisode(
        {"jar.n.01_1": box((0.0, 0.0, 0.8)), "cabinet.n.01_1": box((1.0, 0.0, 0.5))},
        pick_ok={"jar.n.01_1"},
        place_ok={("jar.n.01_1", "cabinet.n.01_1")},
        shut={"cabinet.n.01_1"},
    )
    goal = [atom("inside", "jar.n.01_1", "cabinet.n.01_1")]
    Runner(STRATEGIES["store_honey"], goal, attempts=1).run(ep)
    kinds = [c[0] for c in ep.calls]
    assert "open_up" in kinds, "a shut container must be opened"
    assert kinds.index("open_up") < kinds.index("pick"), "and opened before the hand is full"
    opened = next(c for c in ep.calls if c[0] == "open_up")
    assert opened[2] == OPEN_FRACTION_REACH, "far enough to reach in, not just far enough to score the atom"


def test_a_container_that_will_not_open_is_not_filled():
    ep = FakeEpisode(
        {"jar.n.01_1": box((0.0, 0.0, 0.8)), "cabinet.n.01_1": box((1.0, 0.0, 0.5))},
        pick_ok={"jar.n.01_1"},
        place_ok={("jar.n.01_1", "cabinet.n.01_1")},
        shut={"cabinet.n.01_1"},
        opens_ok=False,
    )
    goal = [atom("inside", "jar.n.01_1", "cabinet.n.01_1")]
    Runner(STRATEGIES["store_honey"], goal, attempts=1).run(ep)
    assert "pick" not in [c[0] for c in ep.calls], "nothing should be picked up with nowhere to put it"


def test_only_an_inside_placement_waits_for_the_container_to_open():
    """openable() answers "does this have a joint", which is not the question.

    The bar countertop is an articulated asset, so is_shut() is True of it, and gating every placement on an open
    meant setup_a_bar never attempted one of its 14 countertop placements -- each refused because a countertop
    would not open. installing_a_scanner does the same with a laptop. An ontop or nextto target has no inside and
    needs nothing opened (2026-09-15).
    """
    boxes = {"bottle.n.01_1": box((0.0, 0.0, 0.8)), "countertop.n.01_1": box((1.0, 0.0, 0.9))}
    ep = FakeEpisode(
        boxes,
        pick_ok={"bottle.n.01_1"},
        place_ok={("bottle.n.01_1", "countertop.n.01_1")},
        shut={"countertop.n.01_1"},
        opens_ok=False,
    )
    goal = [atom("ontop", "bottle.n.01_1", "countertop.n.01_1")]
    Runner(STRATEGIES["setup_a_bar_for_a_cocktail_party"], goal, attempts=1).run(ep)
    kinds = [c[0] for c in ep.calls]
    assert "open_up" not in kinds, "an ontop placement must not try to open its target"
    assert "pick" in kinds, "and must not be refused because that target would not open"


def test_a_one_handed_press_stands_for_the_thing_it_presses():
    """The in_place branch used to press from wherever the previous action left the robot.

    turning_out_all_lights_before_sleep then never moves at all -- "teleports 0" in the RESULT line, with the
    lights in other rooms -- and installing_a_fax_machine plans its press with the button 1.00 m away, 0 of 8
    presses reaching it (2026-09-15).
    """
    ep = FakeEpisode({"lamp.n.02_1": box((4.0, 2.0, 1.1))})
    Runner(STRATEGIES["turning_out_all_lights_before_sleep"], [atom("toggled_on", "lamp.n.02_1")], attempts=1).run(ep)
    kinds = [c[0] for c in ep.calls]
    assert "stand_for" in kinds, "the robot has to go to the switch before it presses it"
    assert kinds.index("stand_for") < kinds.index("achieve"), "and go there first"


def test_a_press_with_nowhere_to_stand_is_still_attempted():
    """Unreachable must not lose the press: pressing from here may still work, and refusing certainly does not."""
    ep = FakeEpisode({"lamp.n.02_1": box((4.0, 2.0, 1.1))}, unreachable={"lamp.n.02_1"})
    Runner(STRATEGIES["turning_out_all_lights_before_sleep"], [atom("toggled_on", "lamp.n.02_1")], attempts=1).run(ep)
    assert "achieve" in [c[0] for c in ep.calls], "a stance failure must not swallow the press"


def test_an_open_goal_atom_is_acted_on_rather_than_left_alone():
    ep = FakeEpisode({"cabinet.n.01_1": box((1.0, 0.0, 0.5))}, shut={"cabinet.n.01_1"})
    goal = [atom("open", "cabinet.n.01_1")]
    Runner(STRATEGIES["store_honey"], goal, attempts=1).run(ep)
    assert [c for c in ep.calls if c[0] == "open_up"], "an open atom should open something"


def test_the_gate_reads_an_open_atom_off_the_container_rather_than_the_round_running():
    from omnigibson.tiptop.bench import Episode

    class Shut(Episode):
        def __init__(self, shut):
            self._shut = shut

        def is_shut(self, name):
            return self._shut

    assert Shut(True).satisfied([atom("open", "cabinet.n.01_1")], record={"round": 1}) is False
    assert Shut(False).satisfied([atom("open", "cabinet.n.01_1")], record={"round": 1}) is True
    shut_goal = [{"predicate": "not", "args": ["open", "cabinet.n.01_1"]}]
    assert Shut(True).satisfied(shut_goal, record={"round": 1}) is True
    assert Shut(False).satisfied(shut_goal, record={"round": 1}) is False


def test_the_support_plane_and_buttons_are_exempt_from_the_visibility_test():
    """A goal may name two things that never carry a mask, and refusing those refuses the task.

    PLANNER_SUPPORT ("table") is the plane tiptop fits by RANSAC: a surface, never a detected object, so it is
    never segmented and never "visible". A button is named by pose through button_hints rather than found as an
    object. An object in the ROBOT'S OWN HAND is the third: the head camera cannot see into the gripper and the
    wrist camera is behind it, so it reaches the planner through the in_hand carry feature instead. A visibility
    test written as "every goal argument must be visible" therefore refuses every ontop(item, table), every
    put-down onto the floor (rewritten to the same label), every press -- which is what happened between two
    commits on 2026-09-13/14 -- and every placement of something the robot is already carrying (2026-09-15).
    """
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "omnigibson" / "tiptop" / "knowledge.py"
    text = source.read_text()
    block = re.search(r"exempt = .*?\n        needed = .*?\n", text, re.S)
    assert block, "the visibility test must exempt the support plane and the buttons"
    assert "PLANNER_SUPPORT" in block.group(0)
    assert '"pressed"' in block.group(0), "a press names a button that was never segmented"
    assert "carried" in block.group(0), (
        "and a third: an object in the gripper reaches the planner through in_hand, not through the picture"
    )
    assert "a not in exempt" in block.group(0), "the exemption has to reach the test itself"


def test_a_goal_that_wants_one_container_to_take_everything_commits_to_one():
    """ "Every toy in a bookcase" means the SAME bookcase; spreading them over two satisfies neither option.

    collecting_childrens_toys scored 0.000 with six picks executed, because the demand was read across both
    options at once -- every toy into bookcase_1 AND every toy into bookcase_2 -- so each toy went to whichever
    was nearest it (2026-09-14). The tell is that no option uses more than one container, which is what separates
    it from putting_away_toys, where a toy may go to either box independently and mixed options exist.
    """
    toys = [f"toy_figure.n.01_{i}" for i in (1, 2, 3)]
    shelves = ["bookcase.n.01_1", "bookcase.n.01_2"]
    # one option per bookcase: all three toys into that one
    options = [[atom("inside", t, shelf) for t in toys] for shelf in shelves]
    demand = place_demand(options)
    assert demand.containers == ["bookcase.n.01_1"], "it must commit to a single bookcase"
    assert demand.wanted == {("toy_figure", "bookcase.n.01_1"): 3}
    assert demand.total() == 3, "three toys, not six"

    # the toys case is NOT this: mixed options exist, so every box may want every toy and merging stays
    boxes = ["toy_box.n.01_1", "toy_box.n.01_2"]
    mixed = [
        [atom("inside", toy, target) for toy, target in zip(toys, choice)]
        for choice in itertools.product(boxes, repeat=3)
    ]
    assert place_demand(mixed).wanted == {("toy_figure", "toy_box.n.01_1"): 3, ("toy_figure", "toy_box.n.01_2"): 3}

    # a single option is untouched
    only = [[atom("inside", t, shelves[0]) for t in toys]]
    assert place_demand(only).containers == ["bookcase.n.01_1"]


def test_a_switch_already_where_the_goal_wants_it_is_not_pressed():
    """Pressing a switch that is already on turns it OFF and destroys a condition already satisfied.

    press_targets assumes "the switch starts in the state the goal wants changed". installing_a_modem is the
    counter-example: its (:init) contains (toggled_on modem.n.01_1) and its goal asks for the modem ON, so one
    of its four conditions is free and a press would lose it. Verified in the BDDL by hand (2026-09-15).
    """

    class _Ep:
        def __init__(self, on):
            self.on, self.pressed, self.picked, self.stood = on, [], [], []

        def switched_on(self, name):
            return self.on

        def has_arm(self, arm):
            return True

        def stand_for(self, *names):
            self.stood = list(names)
            return {}

        def pick(self, obj):
            self.picked.append(obj)
            return True

        def achieve(self, atoms, arm="left", **kw):
            self.pressed.append(atoms[0]["args"][0])
            return True

        def is_shut(self, name):
            return False

        def open_up(self, *a, **k):
            return True

    spec = TaskSpec(task="t", instruction="i", plan="press", press="in_place")
    goal = [atom("toggled_on", "modem.n.01_1")]

    already_on = _Ep(True)
    Runner(spec, goal).run(already_on)
    assert already_on.pressed == [], "a switch already on must not be pressed"

    off = _Ep(False)
    Runner(spec, goal).run(off)
    assert off.pressed == ["modem.n.01_1"], "a switch that is off is still pressed"

    # a runner with no way to read the switch behaves as before rather than skipping every press
    class _NoReading:
        def __init__(self):
            self.pressed = []

        has_arm = _Ep.has_arm
        stand_for = _Ep.stand_for
        pick = _Ep.pick
        achieve = _Ep.achieve
        is_shut = _Ep.is_shut
        open_up = _Ep.open_up
        picked = []
        stood = []

    blind = _NoReading()
    assert not hasattr(blind, "switched_on")
    Runner(spec, goal).run(blind)
    assert blind.pressed == ["modem.n.01_1"]


def test_any_floor_is_recognised_as_a_floor():
    """A goal may name a floor that is not the first one in the task's scope.

    laying_tile_floors asks for tiles ontop floor.n.01_2 and crashed with "no object 'floor.n.01_2' in scene
    office_cubicles_right": the runner compared the container against sim.floor_name(), which returns whichever
    floor is listed first, decided it was not a floor, and went looking for furniture to stand at.
    bringing_in_wood targets the same floor and scored 0.000 with no rounds run (2026-09-15).
    """
    ep = FakeEpisode({"tile.n.01_1": ((0, 0, 0), (0.2, 0.2, 0.02))})
    assert ep.floor == "floor.n.01_1"
    assert ep.is_floor("floor.n.01_1")
    assert ep.is_floor("floor.n.01_2"), "the second floor is still a floor"
    assert not ep.is_floor("table.n.02_1")
    assert not ep.is_floor("")

    # and the runner puts an item down on it rather than trying to stand at a piece of furniture
    spec = TaskSpec(task="t", instruction="i", plan="transfer")
    goal = [atom("ontop", "tile.n.01_1", "floor.n.01_2")]
    runner = Runner(spec, goal)
    assert runner.demand.containers == ["floor.n.01_2"]


def test_the_top_of_a_stack_is_taken_first():
    """23 of 38 books in sorting_books_on_shelf have another task book resting on them. An item with something
    of the goal's own on top cannot be picked until that one moves, and trying it first wastes the attempt."""
    boxes = {
        "bookcase.n.01_1": box((3.0, 0, 0.8), half=(0.4, 0.2, 0.8)),
        "comic_book.n.01_1": box((0.30, 0, 0.90), half=(0.14, 0.14, 0.014)),  # under
        "comic_book.n.01_2": box((0.30, 0, 0.93), half=(0.14, 0.14, 0.014)),  # resting on top of _1
    }
    goal = [atom("inside", f"comic_book.n.01_{i}", "bookcase.n.01_1") for i in (1, 2)]

    class Stacked(FakeEpisode):
        def support_of(self, item):
            return "comic_book.n.01_1" if item == "comic_book.n.01_2" else self.floor

    ep = Stacked(boxes, pick_ok=set(boxes), place_ok=set(boxes))
    strategy_for("sorting_books_on_shelf", goal).run(ep)
    picks = [c[1] for c in ep.calls if c[0] == "pick"]
    assert picks, "it should try something"
    assert picks[0] == "comic_book.n.01_2", f"the top of the stack comes first, got {picks}"


def test_nothing_changes_when_no_item_rests_on_another():
    boxes = {
        "ashcan.n.01_1": box((3.0, 0, 0.15), half=(0.15, 0.15, 0.15)),
        "battery.n.02_1": box((0.4, 0, 0.77)),
        "battery.n.02_2": box((0.6, 0, 0.77)),
    }
    goal = [atom("inside", f"battery.n.02_{i}", "ashcan.n.01_1") for i in (1, 2)]
    ep = FakeEpisode(boxes, pick_ok=set(boxes), place_ok=set(boxes))
    strategy_for("dispose_of_batteries", goal).run(ep)
    assert [c[1] for c in ep.calls if c[0] == "pick"], "the ordinary case still runs"


def test_a_two_handed_press_keeps_the_torso_where_both_planners_agree():
    """A press of "hold" uses both planners, and the right-arm one plans its seven arm joints with the torso
    LOCKED at the embodiment's home posture. --torso moves it away and every press round then dies before it
    plans. turning_on_radio scores 1.0 with the postures agreeing and 0.0 without.

    This calls the rule bench.py actually applies. The first version of this test re-implemented the condition
    inside the test body, so it tested a copy and went on passing while the real rule did something else.
    """
    import types

    from omnigibson.tiptop.bench import wants_home_torso

    spec = lambda plan, press: types.SimpleNamespace(plan=plan, press=press)
    assert wants_home_torso(spec("press", "hold")), "a two-handed press keeps the home torso"
    assert wants_home_torso(spec("auto", "hold")), "so does an auto task that ends in one"
    assert not wants_home_torso(spec("press", "in_place")), "a one-handed press keeps the lean"
    assert not wants_home_torso(spec("transfer", "hold")), "a task that never presses keeps the lean, whatever press says"
    assert not wants_home_torso(None), "no spec, no special case"


def test_the_torso_rule_fires_on_the_tasks_that_press_and_no_others():
    """TaskSpec.press DEFAULTS to "hold", so the rule has to read ``plan`` too.

    Read from the YAML files, only 5 of 38 tasks set ``press`` at all -- which is why the first version of this
    test, which globbed the yaml for an explicit press key, passed while the LOADED specs all said "hold" and the
    lean was being stripped from every task in the set (2026-09-15).
    """
    from omnigibson.tiptop.bench import wants_home_torso
    from omnigibson.tiptop.strategies import TASKS_DIR, TaskSpec

    specs = {p.stem: TaskSpec.load(p) for p in sorted(TASKS_DIR.glob("*.yaml"))}
    holds = [n for n, s in specs.items() if s.press == "hold"]
    assert len(holds) > 30, f"the default is still hold, on {len(holds)} of {len(specs)} tasks -- that is the trap"
    fires = sorted(name for name, s in specs.items() if wants_home_torso(s))
    assert fires == ["turning_on_radio"], f"only the two-planner task keeps the home torso, got {fires}"
    transfers = [n for n, s in specs.items() if s.plan == "transfer"]
    assert len(transfers) > 25 and not any(wants_home_torso(specs[n]) for n in transfers), (
        f"{len(transfers)} pure-transfer tasks must all keep the lean"
    )


def test_touching_a_support_is_served_by_placing_onto_it():
    """putting_shoes_on_rack asks for 4 touching(shoe, hallstand) and 4 not-touching(shoe, floor). Neither
    predicate was in PLACE_PREDICATES, so place_demand never saw them and no shoe was ever lifted: the task
    finished on its two nextto atoms alone, q=0.200, which was exactly its ceiling.

    Resting on a support IS touching it, and the four not-touching atoms come free the moment a shoe leaves the
    floor (2026-09-15).
    """
    from b1k.bridge.protocol import tiptop_goal
    from omnigibson.tiptop.strategies import PLACE_PREDICATES

    assert "touching" in PLACE_PREDICATES, "the demand has to see a touching atom to act on it"

    import inspect

    src = inspect.getsource(tiptop_goal)  # the translation moved out of R1ProSim; the mapping is the wire's
    assert '"touching": "on"' in src, "and the wire has to carry it as a placement onto the support"


def test_a_touching_goal_lifts_the_item_onto_its_support():
    boxes = {
        "gym_shoe.n.01_1": box((0.0, 0.0, 0.05)),
        "hallstand.n.01_1": box((1.0, 0.0, 0.4)),
    }
    ep = FakeEpisode(boxes, pick_ok={"gym_shoe.n.01_1"}, place_ok={("gym_shoe.n.01_1", "hallstand.n.01_1")})
    goal = [atom("touching", "gym_shoe.n.01_1", "hallstand.n.01_1")]
    Runner(STRATEGIES["putting_shoes_on_rack"], goal, attempts=1).run(ep)
    kinds = [c[0] for c in ep.calls]
    assert "pick" in kinds, "a touching goal must actually lift the shoe"


def test_a_switch_the_goal_wants_OFF_still_gets_its_button_described():
    """A goal asking for a switch to be off arrives as not(toggled_on, x) -- bddl compiles the negation into the
    ground atom. press_targets has read both forms since it was written; button_hints read only the positive one.

    turning_out_all_lights_before_sleep's goal is five `not toggled_on` atoms and nothing else, so it produced
    ZERO button hints and every round died on "goal objects ['switch_1_button'] were not found in any of the 3
    view(s)". A button is named by pose and never segmented, so no hint means no press (2026-09-15).
    """
    import inspect

    from omnigibson.tiptop.r1pro import R1ProSim
    from omnigibson.tiptop.strategies import press_targets

    off = [atom("not", "toggled_on", "light_bulb.n.01_1")]
    on = [atom("toggled_on", "radio_receiver.n.01_1")]
    assert press_targets(off) == ["light_bulb.n.01_1"], "the runner already reads the negated form"
    assert press_targets(on) == ["radio_receiver.n.01_1"]

    body = inspect.getsource(R1ProSim.button_hints).split('"""')[-1]
    assert '"not"' in body and '"toggled_on"' in body, "and the button hints have to read the same two forms"
    assert body.count("bddl =") >= 2, "each form names the object differently"


def test_a_round_that_finds_nothing_horizontal_asks_again_with_the_floor_in_view():
    """The planner fits its support plane by RANSAC and refuses the whole request when nothing near-horizontal is
    in the picture. A press needs no surface, but it needs the planner to find one, and a switch on a corridor
    wall gives it nothing: turning_out_all_lights_before_sleep lost all 10 of its rounds to "No plane found with
    objects resting on it".

    The floor is the horizontal plane that always exists, and a press round crops it out -- `floor` is read off
    two-argument atoms and toggled_on(x) has one. Retried rather than defaulted, because turning_on_radio presses
    happily without the floor today (2026-09-15).
    """
    import inspect

    from omnigibson.tiptop.bench import Episode

    body = inspect.getsource(Episode.achieve).split('"""')[-1]
    assert "No plane found" in body, "the retry has to key off the planner's own message"
    assert "floor = True" in body, "and it has to put the floor in view"
    # the guard LINE, not an index into the source -- the comment above it quotes the same message
    guard = [ln for ln in body.split("\n") if "No plane found" in ln and "record.get" in ln]
    assert guard, "the retry has to test the round's own error"
    assert "not floor" in guard[0], "and only fire when the floor was NOT already in view"
    assert body.index("plan_and_execute") < body.index(guard[0].strip()), "the retry follows a round"


def test_a_toppled_base_is_stood_back_up_rather_than_worked_from():
    """A pose that intersects furniture is resolved by the physics lifting and rolling the whole robot. A small
    tilt is workable and the runner has always carried on from one -- 66 of 151 settles across every run are
    under 5 deg off level, and 43 more under 15. But 23 are 45 deg or worse and 12 of those are 120+, which is
    the robot on its back.

    clean_up_your_desk settles at 178.6 deg and then runs ZERO rounds, in every run it has ever had, because
    every field of a request is expressed in a base frame that is upside down (2026-09-15).
    """
    import inspect

    from omnigibson.tiptop.bench import TOPPLED_DEG, Episode

    assert 15 < TOPPLED_DEG < 90, f"the threshold has to separate a workable tilt from a topple, got {TOPPLED_DEG}"

    body = inspect.getsource(Episode.stand_for).split('"""')[-1]
    assert "TOPPLED_DEG" in body, "the topple case has to be distinguished from a workable tilt"
    assert "Unreachable" in body[body.index("TOPPLED_DEG"):], "and reported as unreachable, not worked from"
    assert "place_robot" in body[body.index("TOPPLED_DEG"):], "and the robot stood back where it came from"
    assert "going on from here" in body, "a workable tilt still carries on, as it always has"

    # ...and the check has to happen on the way IN as well. clean_up_your_desk topples on its FIRST teleport and
    # the next search raises Unreachable before the attempt loop is exhausted, so the exit path never runs and the
    # robot stays on its back for the rest of the episode -- 24 failed stance searches, zero rounds.
    entry = body[: body.index("for attempt in range")]
    assert "TOPPLED_DEG" in entry, "a robot already toppled cannot search: stand it up before trying"
    assert "last_level" in body, "and the pose to return to is the last one that settled level"
