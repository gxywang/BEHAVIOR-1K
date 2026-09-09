"""How a challenge task is broken into planning rounds: where to stand, what to ask the planner for, what to do
when a round fails. A strategy runs against an ``Episode`` (bench.py): ``stand_for`` moves the base (a teleport
today), ``plan_and_execute`` does one capture / plan / execute round, and the simulator answers the questions the
pipeline cannot yet answer for itself (is the item in the hand, did it land inside; privileged). Nothing here
knows how the planner is told about the scene (knowledge.py) or how it plans.
"""

import logging

log = logging.getLogger(__name__)


class Unreachable(RuntimeError):
    """Raised by an episode's ``stand_for`` when no base pose reaches the named objects; strategies skip them."""


def atom(predicate: str, *args: str) -> dict:
    return {"predicate": predicate, "args": list(args)}


def task_goal_atoms(sim) -> list[dict]:
    """The task's goal as BDDL atoms: the first ground goal option (all options name the same predicates over the
    same objects up to their pairing, which is what a strategy needs to know)."""
    task = sim.env.task
    atoms = []
    for head in task.ground_goal_state_options[0]:
        terms = list(getattr(head, "terms", []))
        if terms:
            atoms.append(atom(terms[0], *terms[1:]))
    return atoms


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
    (``--press-port``) and refuses to run without it. A failed pick is retried once from a fresh base pose. When
    the press finds no plan twice (the grasp left the switch where the right hand cannot reach it), the radio is
    put back on its table and picked up again, once."""

    task = "turning_on_radio"
    instruction = "pick up the radio and press its button"

    def run(self, ep) -> None:
        (radio,) = [a["args"][0] for a in self.goal if a["predicate"] == "toggled_on"]
        if not ep.has_arm("right"):
            raise ValueError(
                "turning_on_radio needs the right-arm planner (--press-port): the radio is held while pressed"
            )
        table = ep.support_of(radio)
        for cycle in range(2):
            if not self.pick(ep, radio):
                return
            for _ in range(2):
                ep.plan_and_execute([atom("toggled_on", radio)], arm="right")
                if ep.holds("toggled_on", radio):
                    return
            if cycle == 0:
                log.info(f"{radio}: no press plan with this grasp; putting it back on {table} to pick it up again")
                ep.plan_and_execute([atom("ontop", radio, table)])
                if ep.holding(radio):
                    log.warning(f"{radio}: still in the hand after the put-down; giving up")
                    return

    @staticmethod
    def pick(ep, radio: str) -> bool:
        """The radio in the left hand after at most two pick rounds from different base poses."""
        for _ in range(2):
            try:
                ep.stand_for(radio)
            except Unreachable as e:
                log.warning(f"{radio}: {e}; giving up")
                return False
            ep.plan_and_execute([atom("holding", radio)])
            if ep.holding(radio):
                return True
        log.warning(f"{radio}: not in the hand after two pick rounds; giving up")
        return False


class AssembleGiftBaskets(Strategy):
    """Every basket gets one item of each kind. Baskets stand on the floor, items on a table; each transfer is a pick
    at the table, a base move to the basket with the item in hand, and a place into the basket. Baskets are done
    nearest to the table first; within a kind, the items nearest the table's edge are tried first (the reachable
    ones), ``attempts`` of them per basket. An item still in the hand after a failed place is put down where the
    robot stands so the hand is free for the next one."""

    task = "assembling_gift_baskets"
    instruction = "put one candle, one cheese, one cookie and one bow into each wicker basket"

    def __init__(self, goal, attempts: int = 2):
        super().__init__(goal)
        self.attempts = attempts

    @staticmethod
    def category(bddl: str) -> str:
        return bddl.split(".n.")[0]

    def run(self, ep) -> None:
        pairs = [(a["args"][0], a["args"][1]) for a in self.goal if a["predicate"] == "inside" and len(a["args"]) == 2]
        if not pairs:
            raise ValueError("the task goal has no inside(item, container) predicates")
        table = ep.support_of(pairs[0][0])
        baskets = sorted(dict.fromkeys(c for _, c in pairs), key=lambda b: ep.distance(b, table))
        items_by_kind = {}
        for item, _ in pairs:
            items_by_kind.setdefault(self.category(item), [])
            if item not in items_by_kind[self.category(item)]:
                items_by_kind[self.category(item)].append(item)
        log.info(f"{len(baskets)} baskets x {sorted(items_by_kind)}; baskets in order {baskets}")
        placed = set()
        for basket in baskets:
            for kind, items in items_by_kind.items():
                candidates = [i for i in items if i not in placed and ep.holds("ontop", i, table)]
                candidates.sort(key=lambda i: ep.edge_gap(i, table))  # nearest the table's edge first
                for item in candidates[: self.attempts]:
                    if self.transfer(ep, item, basket, table):
                        placed.add(item)
                        break

    def transfer(self, ep, item: str, basket: str, table: str) -> bool:
        try:
            ep.stand_for(item)
        except Unreachable as e:
            log.info(f"{item}: {e}")
            return False
        ep.plan_and_execute([atom("holding", item)])
        if not ep.holding(item):
            log.info(f"{item}: not in the hand after the pick round")
            return False
        try:
            ep.stand_for(basket)  # carrying the item
        except Unreachable as e:
            log.info(f"{basket}: {e}; putting {item} back down")
            ep.plan_and_execute([atom("ontop", item, table)])
            return False
        ep.plan_and_execute([atom("inside", item, basket)], floor=True)
        if ep.holds("inside", item, basket):
            return True
        if ep.holding(item):  # free the hand: put it down on the floor plane where the robot stands
            log.info(f"{item}: not inside {basket}; putting it down")
            ep.plan_and_execute([atom("ontop", item, ep.floor)], floor=True)
        return False


STRATEGIES = {cls.task: cls for cls in (TurnOnRadio, AssembleGiftBaskets)}


def strategy_for(task: str, goal: list[dict], **kwargs) -> Strategy:
    if task not in STRATEGIES:
        raise ValueError(f"no strategy for task {task!r}; known: {sorted(STRATEGIES)}")
    return STRATEGIES[task](goal, **kwargs)
