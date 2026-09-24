"""A failed placement retries its own destination, then a planned floor put-down; the hand is opened where it is
only when no put-down plans, as the last resort before the instance ends with the object in it."""

from types import SimpleNamespace

import pytest

from b1k.bridge.strategies import Runner, TaskSpec, TransferBlocked, Unreachable, atom


ITEM = "bowl.n.01_1"
OTHER = "bowl.n.01_2"
TARGET = "sink.n.01_1"
SOURCE = "booth.n.01_1"


class TransferEpisode:
    """Record high-level actions; successful scripted placements release onto the requested target."""

    floor = "floor.n.01_1"

    def __init__(self, outcomes=(), unreachable=(), held=None):
        self.outcomes = iter(outcomes)
        self.unreachable = set(unreachable)
        self.hand = held
        self.calls = []
        self.completed = set()
        self.shut = False
        self.sim = SimpleNamespace(n_steps=0, max_steps=10000)

    def held_names(self):
        return [] if self.hand is None else [self.hand]

    def holding(self, item):
        return self.hand == item

    def support_of(self, item):
        return SOURCE

    def edge_gap(self, item, support):
        return 0.0 if item == ITEM else 1.0

    def distance(self, item, target):
        return 1.0

    def is_shut(self, target):
        return self.shut

    def is_floor(self, target):
        return target.startswith("floor.")

    def goal_already_holds(self, predicate, item, target):
        return (predicate, item, target) in self.completed

    def pick(self, item):
        self.calls.append(("pick", item))
        assert self.hand is None, "another pickup must not displace an existing grasp"
        self.hand = item
        return True

    def stand_for(self, target):
        self.calls.append(("stand_for", target))
        if target in self.unreachable:
            raise Unreachable(target)

    def walk_to_floor(self, target):
        self.calls.append(("walk_to_floor", target))
        return target not in self.unreachable

    def achieve(self, atoms, **kwargs):
        goal = atoms[0]
        self.calls.append(("achieve", goal["predicate"], *goal["args"]))
        if next(self.outcomes, False):
            self.completed.add((goal["predicate"], *goal["args"]))
            self.hand = None
            return True
        return False

    def put_down(self, item, support, **kwargs):
        self.calls.append(("put_down", item, support))
        return self.achieve([atom("ontop", item, support)])

    def release(self):
        self.calls.append(("release",))  # the last resort, and here nothing comes free: the hand keeps the item

    def open_up(self, target, **kwargs):
        pytest.fail("a retained grasp must not be repurposed to open its destination")

    def has_arm(self, arm):
        return True


def runner(goals=None, attempts=2, plan="transfer"):
    spec = TaskSpec(task="recovery_test", instruction="place the dishes", plan=plan)
    return Runner(spec, goals or [atom("inside", ITEM, TARGET)], attempts=attempts)


def test_exhausted_placement_tries_a_planned_floor_put_down_then_stops():
    ep = TransferEpisode()
    task = runner(
        [atom("inside", ITEM, TARGET), atom("inside", OTHER, TARGET), atom("toggled_on", "lamp.n.02_1")],
        plan="auto",
    )
    with pytest.raises(TransferBlocked, match="object retained in the hand") as failure:
        task.run(ep)
    assert ITEM in str(failure.value) and TARGET in str(failure.value)
    assert ep.hand == ITEM
    assert task.tries[ITEM] == 2
    assert ep.calls == [
        ("pick", ITEM),
        ("stand_for", TARGET),
        ("achieve", "inside", ITEM, TARGET),
        ("stand_for", TARGET),
        ("achieve", "inside", ITEM, TARGET),
        ("put_down", ITEM, ep.floor),
        ("achieve", "ontop", ITEM, ep.floor),
        ("walk_to_floor", ep.floor),
        ("put_down", ITEM, ep.floor),
        ("achieve", "ontop", ITEM, ep.floor),
        ("release",),
    ]


def test_exhausted_placement_frees_the_hand_with_a_planned_put_down_and_goes_on():
    ep = TransferEpisode(outcomes=[False, False, True])
    task = runner()
    assert task.transfer_one(ep, "bowl", TARGET, set()) is None
    assert ep.hand is None and ("ontop", ITEM, ep.floor) in ep.completed
    assert ("inside", ITEM, TARGET) not in ep.completed


def test_successful_destination_retry_reuses_the_existing_grasp():
    ep = TransferEpisode(outcomes=[False, True])
    task = runner()
    assert task.transfer_one(ep, "bowl", TARGET, set()) == ITEM
    assert ep.hand is None
    assert ep.calls.count(("pick", ITEM)) == 1
    assert ep.calls.count(("stand_for", TARGET)) == 2
    assert ep.calls.count(("achieve", "inside", ITEM, TARGET)) == 2
    assert ("inside", ITEM, TARGET) in ep.completed


def test_unreachable_destination_puts_the_item_back_on_its_own_support():
    ep = TransferEpisode(outcomes=[True], unreachable=[TARGET])
    assert runner().transfer_one(ep, "bowl", TARGET, set()) is None
    assert ep.hand is None
    assert ep.calls == [("pick", ITEM), ("stand_for", TARGET), ("put_down", ITEM, SOURCE), ("achieve", "ontop", ITEM, SOURCE)]


def test_other_held_object_prevents_unrelated_transfer_without_any_action():
    ep = TransferEpisode(held=OTHER)
    with pytest.raises(TransferBlocked, match="cannot start pickup"):
        runner().transfer(ep, "inside", ITEM, TARGET, SOURCE)
    assert ep.hand == OTHER
    assert ep.calls == [("put_down", OTHER, ep.floor), ("achieve", "ontop", OTHER, ep.floor), ("walk_to_floor", ep.floor), ("put_down", OTHER, ep.floor), ("achieve", "ontop", OTHER, ep.floor), ("release",)]


def test_free_hand_does_not_use_the_next_items_support():
    ep = TransferEpisode(held=ITEM)
    assert Runner.free_hand(ep) is False
    assert ep.hand == ITEM
    assert ep.calls == [("put_down", ITEM, ep.floor), ("achieve", "ontop", ITEM, ep.floor), ("walk_to_floor", ep.floor), ("put_down", ITEM, ep.floor), ("achieve", "ontop", ITEM, ep.floor), ("release",)]


def test_unreachable_named_floor_never_falls_back_to_the_current_floor():
    target_floor = "floor.n.01_2"
    ep = TransferEpisode(unreachable=[target_floor])
    with pytest.raises(TransferBlocked):
        runner([atom("ontop", ITEM, target_floor)]).run(ep)
    assert ep.hand == ITEM
    # the current floor is only ever a recovery put-down, never counted as the named floor
    assert ep.calls == [
        ("pick", ITEM),
        ("walk_to_floor", target_floor),
        ("walk_to_floor", target_floor),
        ("put_down", ITEM, ep.floor),
        ("achieve", "ontop", ITEM, ep.floor),
        ("walk_to_floor", ep.floor),
        ("put_down", ITEM, ep.floor),
        ("achieve", "ontop", ITEM, ep.floor),
        ("release",),
    ]
    assert ("ontop", ITEM, target_floor) not in ep.completed


def test_a_container_that_closed_while_carrying_does_not_trigger_open():
    ep = TransferEpisode(held=ITEM)
    ep.shut = True
    with pytest.raises(TransferBlocked):
        runner().run(ep)
    assert ep.hand == ITEM
    assert ep.calls == [("put_down", ITEM, ep.floor), ("achieve", "ontop", ITEM, ep.floor), ("walk_to_floor", ep.floor), ("put_down", ITEM, ep.floor), ("achieve", "ontop", ITEM, ep.floor), ("release",)]


@pytest.mark.parametrize("still_held,at_destination", [(True, True), (False, False)])
def test_a_successful_round_alone_cannot_count_as_delivery(still_held, at_destination):
    ep = TransferEpisode()

    def misleading_round(atoms):
        if not still_held:
            ep.hand = None
        if at_destination:
            ep.completed.add(("inside", ITEM, TARGET))
        return True

    ep.achieve = misleading_round
    assert runner().transfer(ep, "inside", ITEM, TARGET, SOURCE) is False


@pytest.mark.parametrize(
    "still_held,at_destination,expected", [(True, True, False), (False, False, False), (False, True, True)]
)
def test_put_down_requires_release_and_evidence_at_the_requested_support(still_held, at_destination, expected):
    from omnigibson.tiptop.bench import Episode

    calls = []

    def achieve(atoms, floor=None, done=None):
        calls.append(atoms)
        return done()

    ep = SimpleNamespace(
        achieve=achieve,
        holding=lambda _: still_held,
        placed=lambda *_: at_destination,
        goal_already_holds=lambda *_: at_destination,
        is_floor=lambda _: False,
    )
    assert Episode.put_down(ep, ITEM, SOURCE) is expected
    assert calls == [[atom("ontop", ITEM, SOURCE)]]


def test_put_down_on_a_named_floor_cannot_succeed_merely_because_the_hand_is_empty():
    from omnigibson.tiptop.bench import Episode

    ep = SimpleNamespace(
        achieve=lambda atoms, floor=None, done=None: done(),
        holding=lambda _: False,
        # This is the historical floor branch of Episode.placed: empty hand alone returns True.
        placed=lambda *_: True,
        goal_already_holds=lambda *_: False,
        is_floor=lambda _: True,
        stance_key=lambda: (1, 2, 3),
    )
    assert Episode.put_down(ep, ITEM, "floor.n.01_2", floor=True) is False
    assert ep.floor_failed_at == {(1, 2, 3)}  # free_hand walks off rather than trying this spot again


def test_held_object_still_touching_its_goal_floor_does_not_count_as_released():
    target_floor = "floor.n.01_2"
    ep = TransferEpisode(held=ITEM)
    # A sticky grasp can attach before lifting, while the physical OnTop relation still holds.
    ep.completed.add(("ontop", ITEM, target_floor))
    assert runner([atom("ontop", ITEM, target_floor)]).transfer(ep, "ontop", ITEM, target_floor, SOURCE) is False
    assert ep.hand == ITEM


def test_free_hand_walks_off_when_the_floor_put_down_already_failed_here():
    from b1k.bridge.strategies import Runner

    calls = []
    held = ["item.n.01_1"]
    ep = SimpleNamespace(
        floor="floor.n.01_1", held_names=lambda: list(held), holding=lambda n: n in held,
        stance_key=lambda: (0, 0, 0), floor_failed_at={(0, 0, 0)},
        put_down=lambda name, support, floor=None: calls.append(("put_down", floor)) or bool(floor and held.clear() is None),
        walk_to_floor=lambda floor: calls.append(("walk",)) or True,
    )
    assert Runner.free_hand(ep) is True
    assert calls == [("walk",), ("put_down", True)]  # no second try from the stance that already failed
