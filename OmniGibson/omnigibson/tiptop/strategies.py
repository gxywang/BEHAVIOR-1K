"""How a challenge task is broken into planning rounds: where to stand and what to ask the planner for. A strategy
runs against an ``Episode`` (bench.py): ``stand_for`` moves the base (a teleport today); ``pick``, ``achieve`` and
``put_down`` run the capture / plan / execute rounds under the one retry policy every task gets (``--rounds``);
and the simulator answers the questions the pipeline cannot yet answer for itself (is the item in the hand, did
it land inside; privileged). A strategy orders the goals and decides what to skip; recovery only one task would
need is not written here (README, "Kept out of the pipeline"). Nothing here knows how the planner is told about
the scene (knowledge.py) or how it plans.
"""

import logging

from omnigibson.tiptop.protocol import bddl_category

log = logging.getLogger(__name__)


class Unreachable(RuntimeError):
    """Raised by an episode's ``stand_for`` when no base pose reaches the named objects; strategies skip them."""


def atom(predicate: str, *args: str) -> dict:
    return {"predicate": predicate, "args": list(args)}


def task_goal_atoms(sim) -> list[dict]:
    """The task's goal as BDDL atoms: the first ground goal option (all options name the same predicates over the
    same objects up to their pairing, which is what a strategy needs to know)."""
    from bddl.condition_evaluation import HEAD

    return [
        atom(head.terms[0], *head.terms[1:])
        for head in sim.env.task.ground_goal_state_options[0]
        if isinstance(head, HEAD)  # a ground atom; the tasks here have no other compiled form in their goal
    ]


class Strategy:
    task = ""
    instruction = ""  # the natural-language task the planner is given

    def __init__(self, goal: list[dict]):
        self.goal = list(goal)

    def run(self, ep) -> None:
        raise NotImplementedError


class TurnOnRadio(Strategy):
    """Pick the radio up with the left hand (the grasp presents the switch) and press the switch with the right.
    The radio must be held: pressed where it stands, a free-standing radio slides away on the table before the
    toggle registers (20 cm across the glass table, 2026-09-09), so this strategy needs the right-arm planner
    (``--press-port``) and refuses to run without it. The pick and the press each get the episode's rounds."""

    task = "turning_on_radio"
    instruction = "pick up the radio and press its button"

    def run(self, ep) -> None:
        (radio,) = [a["args"][0] for a in self.goal if a["predicate"] == "toggled_on"]
        if not ep.has_arm("right"):
            raise ValueError(
                "turning_on_radio needs the right-arm planner (--press-port): the radio is held while pressed"
            )
        if ep.pick(radio):
            ep.achieve([atom("toggled_on", radio)], arm="right")


class AssembleGiftBaskets(Strategy):
    """Every basket gets one item of each kind. Baskets stand on the floor, items on a table; each transfer is a pick
    at the table, a base move to the basket with the item in hand, and a place into the basket.
    Baskets are done nearest to the table first; within a kind, the items still on the table nearest its edge are
    tried first (the reachable ones), ``attempts`` of them per basket. An item still in the hand after a failed
    place is put down where the robot stands, and a hand still full at the next transfer is emptied first."""

    task = "assembling_gift_baskets"
    instruction = "put one candle, one cheese, one cookie and one bow into each wicker basket"

    def __init__(self, goal, attempts: int = 2):
        super().__init__(goal)
        self.attempts = attempts

    def run(self, ep) -> None:
        pairs = [(a["args"][0], a["args"][1]) for a in self.goal if a["predicate"] == "inside" and len(a["args"]) == 2]
        if not pairs:
            raise ValueError("the task goal has no inside(item, container) predicates")
        table = ep.support_of(pairs[0][0])
        baskets = sorted(dict.fromkeys(c for _, c in pairs), key=lambda b: ep.distance(b, table))
        items_by_kind = {}
        for item, _ in pairs:
            items_by_kind.setdefault(bddl_category(item), [])
            if item not in items_by_kind[bddl_category(item)]:
                items_by_kind[bddl_category(item)].append(item)
        log.info(f"{len(baskets)} baskets x {sorted(items_by_kind)}; baskets in order {baskets}")
        placed = set()
        for basket in baskets:
            for kind, items in items_by_kind.items():
                candidates = [i for i in items if i not in placed and ep.on_support(i, table)]
                candidates.sort(key=lambda i: ep.edge_gap(i, table))  # nearest the table's edge first
                for item in candidates[: self.attempts]:
                    if self.transfer(ep, item, basket, table):
                        placed.add(item)
                        break

    def transfer(self, ep, item: str, basket: str, table: str) -> bool:
        if not self.free_hand(ep, table):
            return False
        if not ep.pick(item):
            return False
        try:
            ep.stand_for(basket)  # carrying the item
        except Unreachable as e:
            log.info(f"{basket}: {e}; putting {item} back down")
            ep.put_down(item, table)
            return False
        if ep.achieve([atom("inside", item, basket)], floor=True):
            return True
        if ep.holding(item):  # free the hand: put it down on the floor plane where the robot stands
            log.info(f"{item}: not inside {basket}; putting it down")
            ep.put_down(item, ep.floor, floor=True)
        return False

    @staticmethod
    def free_hand(ep, table: str) -> bool:
        """Put down whatever the hand still holds (a place that failed left it there) before the next pick: on the
        plane where the robot stands, else from a fresh pose at the table, else by opening the hand where it is
        (the item falls; better than a hand that stays full for the rest of the episode). False when it stays."""
        held = ep.held_names()
        if not held:
            return True
        name = held[0]
        log.info(f"{name} is still in the hand; putting it down before the next pick")
        if ep.put_down(name, ep.floor, floor=True):
            return True
        try:
            ep.stand_for(table)
            ep.put_down(name, table)
        except Unreachable as e:
            log.warning(f"{name}: {e}")
        if ep.holding(name):
            log.warning(f"{name}: no put-down plan; releasing it where the robot stands")
            ep.release()
        return not ep.holding(name)


STRATEGIES = {cls.task: cls for cls in (TurnOnRadio, AssembleGiftBaskets)}


def strategy_for(task: str, goal: list[dict], **kwargs) -> Strategy:
    if task not in STRATEGIES:
        raise ValueError(f"no strategy for task {task!r}; known: {sorted(STRATEGIES)}")
    return STRATEGIES[task](goal, **kwargs)
