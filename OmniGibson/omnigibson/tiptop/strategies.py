"""How a challenge task is broken into planning rounds: one generic runner, driven by a per-task description.

A task says what it needs in ``tasks/<task>.yaml`` (``TaskSpec``): the instruction the planner is given, which
sub-plan handles its goal atoms (``transfer``: pick each item, carry it to its container, place it; ``press``: press
each object's button, holding the object first when the description says so), and the ordering choices that were
tuned on it. The goal atoms come from the task's BDDL definition at run time (``task_goal_atoms``): the TiPToP
paper had a language model write such goals from an instruction, we read them from the task, and at evaluation
the task id says which definition applies. The ``Runner`` orders the atoms, runs them against an ``Episode``
(bench.py) under the one retry policy every task gets (``--rounds``), and skips what cannot be reached. Nothing
here knows how the planner is told about the scene (knowledge.py), how it plans, or what the simulator thinks:
the episode judges every round from the robot's own readings and from localization.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from omnigibson.tiptop.protocol import bddl_category

log = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
PLACE_PREDICATES = ("inside", "ontop", "on")


class Unreachable(RuntimeError):
    """Raised by an episode's ``stand_for`` when no base pose reaches the named objects; the runner skips them."""


def atom(predicate: str, *args: str) -> dict:
    return {"predicate": predicate, "args": list(args)}


def task_goal_atoms(sim) -> list[dict]:
    """The task's goal as BDDL atoms: the first ground goal option (all options name the same predicates over the
    same objects up to their pairing, which is what a runner needs to know)."""
    from bddl.condition_evaluation import HEAD

    return [
        atom(head.terms[0], *head.terms[1:])
        for head in sim.env.task.ground_goal_state_options[0]
        if isinstance(head, HEAD)  # a ground atom; the tasks here have no other compiled form in their goal
    ]


@dataclass
class TaskSpec:
    """What a task says about itself (``tasks/<task>.yaml``). ``plan``: ``transfer``, ``press`` or ``auto`` (both,
    transfers first). ``press``: ``hold`` (pick the object, press with the other hand) or ``in_place``. ``order``:
    ``containers`` (``nearest_first`` from the items' support, or ``goal``) and ``items`` (``nearest_edge_first`` on
    their support, or ``goal``). ``phrases``: detector words per category, for the onboard source."""

    task: str
    instruction: str
    plan: str = "auto"
    press: str = "hold"
    order: dict = field(default_factory=lambda: {"containers": "nearest_first", "items": "nearest_edge_first"})
    attempts_per_kind: int = 2
    phrases: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "TaskSpec":
        data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown fields {sorted(unknown)}")
        spec = cls(**data)
        if spec.plan not in ("transfer", "press", "auto"):
            raise ValueError(f"{path}: plan must be transfer, press or auto, not {spec.plan!r}")
        if spec.press not in ("hold", "in_place"):
            raise ValueError(f"{path}: press must be hold or in_place, not {spec.press!r}")
        return spec


def load_specs(directory=TASKS_DIR) -> dict:
    """task name -> TaskSpec for every description in the directory."""
    specs = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        spec = TaskSpec.load(path)
        if spec.task != path.stem:
            raise ValueError(f"{path}: describes task {spec.task!r}, expected {path.stem!r}")
        specs[spec.task] = spec
    return specs


STRATEGIES = load_specs()


class Runner:
    """Runs a task's goal atoms against an episode the way its description says."""

    def __init__(self, spec: TaskSpec, goal: list[dict], attempts: int | None = None):
        self.spec = spec
        self.goal = list(goal)
        self.attempts = spec.attempts_per_kind if attempts is None else int(attempts)

    @property
    def instruction(self) -> str:
        return self.spec.instruction

    def run(self, ep) -> None:
        transfers = [
            (a["predicate"], a["args"][0], a["args"][1])
            for a in self.goal
            if a["predicate"] in PLACE_PREDICATES and len(a["args"]) == 2
        ]
        presses = [a["args"][0] for a in self.goal if a["predicate"] == "toggled_on"]
        if self.spec.plan in ("transfer", "auto") and transfers:
            self.run_transfers(ep, transfers)
        elif self.spec.plan == "transfer":
            raise ValueError(f"{self.spec.task}: the goal has no inside/ontop atoms for the transfer plan")
        if self.spec.plan in ("press", "auto") and presses:
            for obj in presses:
                self.run_press(ep, obj)
        elif self.spec.plan == "press":
            raise ValueError(f"{self.spec.task}: the goal has no toggled_on atoms for the press plan")

    # ---------------------------------------------------------------- transfers
    def run_transfers(self, ep, transfers: list[tuple[str, str, str]]) -> None:
        """Every container gets one item of each kind named for it. Containers nearest the items' support first;
        within a kind, the items still on that support nearest its edge first, ``attempts`` of them per container.
        An item still in the hand after a failed place is put down where the robot stands, and a hand still full
        at the next transfer is emptied first."""
        support = ep.support_of(transfers[0][1])
        containers = list(dict.fromkeys(c for _, _, c in transfers))
        if self.spec.order.get("containers", "nearest_first") == "nearest_first":
            containers.sort(key=lambda c: ep.distance(c, support))
        predicate_for = {(i, c): p for p, i, c in transfers}
        kinds = {}
        for _, item, _ in transfers:
            kinds.setdefault(bddl_category(item), [])
            if item not in kinds[bddl_category(item)]:
                kinds[bddl_category(item)].append(item)
        log.info(
            f"{len(containers)} containers x {sorted(kinds)}; containers in order {containers}; items on {support}"
        )
        placed = set()
        for container in containers:
            wanted = {bddl_category(i) for _, i, c in transfers if c == container}
            for kind, items in kinds.items():
                if kind not in wanted:
                    continue
                candidates = [i for i in items if i not in placed and ep.on_support(i, support)]
                if self.spec.order.get("items", "nearest_edge_first") == "nearest_edge_first":
                    candidates.sort(key=lambda i: ep.edge_gap(i, support))
                for item in candidates[: self.attempts]:
                    predicate = predicate_for.get((item, container), "inside")
                    if self.transfer(ep, predicate, item, container, support):
                        placed.add(item)
                        break

    def transfer(self, ep, predicate: str, item: str, container: str, support: str) -> bool:
        if not self.free_hand(ep, support):
            return False
        if not ep.pick(item):
            return False
        try:
            ep.stand_for(container)  # carrying the item
        except Unreachable as e:
            log.info(f"{container}: {e}; putting {item} back down")
            ep.put_down(item, support)
            return False
        if ep.achieve([atom(predicate, item, container)]):
            return True
        if ep.holding(item):  # free the hand: put it down where the robot stands
            log.info(f"{item}: not placed in {container}; putting it down")
            ep.put_down(item, ep.floor)
        return False

    @staticmethod
    def free_hand(ep, support: str) -> bool:
        """Put down whatever the hand still holds (a place that failed left it there) before the next pick: on the
        floor where the robot stands, else from a fresh pose at the items' support, else by opening the hand where
        it is (the item falls; better than a hand that stays full for the rest of the episode). False when it
        stays."""
        held = ep.held_names()
        if not held:
            return True
        name = held[0]
        log.info(f"{name} is still in the hand; putting it down before the next pick")
        if ep.put_down(name, ep.floor):
            return True
        try:
            ep.stand_for(support)
            ep.put_down(name, support)
        except Unreachable as e:
            log.warning(f"{name}: {e}")
        if ep.holding(name):
            log.warning(f"{name}: no put-down plan; releasing it where the robot stands")
            ep.release()
        return not ep.holding(name)

    # ---------------------------------------------------------------- presses
    def run_press(self, ep, obj: str) -> None:
        if self.spec.press == "hold":
            if not ep.has_arm("right"):
                raise ValueError(f"{self.spec.task}: press 'hold' needs the right-arm planner (--press-port)")
            if ep.pick(obj):
                ep.achieve([atom("toggled_on", obj)], arm="right")
        else:
            ep.achieve([atom("toggled_on", obj)])


def strategy_for(task: str, goal: list[dict], **kwargs) -> Runner:
    if task not in STRATEGIES:
        raise ValueError(f"no task description for {task!r} in {TASKS_DIR}; known: {sorted(STRATEGIES)}")
    return Runner(STRATEGIES[task], goal, **kwargs)
