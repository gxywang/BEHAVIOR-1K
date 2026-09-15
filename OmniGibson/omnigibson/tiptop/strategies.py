"""How a challenge task is broken into planning rounds: one generic runner, driven by the task's own goal.

The goal comes from the task's BDDL definition at run time as *ground options*: the ways the goal can be
satisfied, each a list of atoms (the TiPToP paper had a language model write such a goal from an instruction; we
read it from the task, and at evaluation the task id says which definition applies). Reading all the options,
rather than one, is what tells the runner how many items of a kind each container wants and which containers are
interchangeable: four wicker baskets that each want one candle, one bin that wants three batteries, two toy boxes
that will take any toy. ``place_demand`` turns the options into that table; ``Runner.run_transfers`` works through
it (nearest container first, and for each item wanted, the items nearest the edge of whatever they stand on),
and ``Runner.run_press`` handles ``toggled_on`` atoms. Atoms already true when the instance starts are never
worked on.

``tasks/<task>.yaml`` (``TaskSpec``) holds only what cannot be read from the definition: the instruction the
planner is given, whether a press picks the object up first, and how many tries an item gets. The ``Episode``
(bench.py) runs each round under the one retry policy every task gets (``--rounds``) and judges it from the
robot's own readings and localization; nothing here knows how the planner is told about the scene (knowledge.py),
how it plans, or what the simulator thinks.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from omnigibson.tiptop.articulation import OPEN_FRACTION_REACH, OPEN_FRACTION_SCORED
from omnigibson.tiptop.protocol import bddl_category

log = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
# Goal predicates the transfer plan can act on: a pick, a carry and a release at the target. "nextto" joins
# them because the planner has had the operator all along -- cuTAMP's PlaceNear with its Near fluent and
# NearPlacement constraint -- and the bridge already maps nextto -> near on the wire. 16 of the 100 challenge
# tasks name it, and expanding the goals says it lifts the vocabulary's ceiling over the whole set from a mean
# q_score of 0.415 to 0.480 (2026-09-13).
PLACE_PREDICATES = ("inside", "ontop", "on", "nextto")
EXTRA_SWEEPS = 3  # further passes over the atoms still open, while the step budget has room (spend_what_is_left)
BUDGET_FOR_ANOTHER_SWEEP = 0.6  # do not begin a sweep past this share of the episode's steps
GOAL_OPTIONS_READ = 20000  # ground goal options read to learn the demand (assembling_gift_baskets has 331,776)


class Unreachable(RuntimeError):
    """Raised by an episode's ``stand_for`` when no base pose reaches the named objects; the runner skips them."""


def atom(predicate: str, *args: str) -> dict:
    return {"predicate": predicate, "args": list(args)}


def option_atoms(option) -> list[dict]:
    """One ground goal option as atoms; compiled forms other than ground atoms are dropped."""
    from bddl.condition_evaluation import HEAD

    return [atom(head.terms[0], *head.terms[1:]) for head in option if isinstance(head, HEAD)]


def task_goal_atoms(sim) -> list[dict]:
    """The task's goal as BDDL atoms: the first ground option (one way to satisfy the goal)."""
    return option_atoms(sim.env.task.ground_goal_state_options[0])


def task_goal_options(sim, limit: int = GOAL_OPTIONS_READ) -> list[list[dict]]:
    """Every way the task's goal can be satisfied, each as a list of atoms. A goal that pairs things off has one
    option per pairing (331,776 for the gift baskets), so the read is capped: the options are generated in a
    regular order and what the runner takes from them (how many items of a kind a container wants) repeats."""
    options = sim.env.task.ground_goal_state_options
    if len(options) > limit:
        log.info(f"{len(options)} ground goal options; reading the first {limit}")
    return [option_atoms(option) for option in options[:limit]]


@dataclass
class Demand:
    """What the goal asks for, read from its ground options. ``wanted[(kind, container)]``: how many items of that
    kind the container takes (the most any one option puts there). ``items[kind]``: the objects of that kind the
    goal names, in the order it names them. ``predicate[(kind, container)]``: ``inside`` or ``ontop``."""

    wanted: dict = field(default_factory=dict)
    items: dict = field(default_factory=dict)
    predicate: dict = field(default_factory=dict)

    @property
    def containers(self) -> list[str]:
        return list(dict.fromkeys(container for _, container in self.wanted))

    def kinds_for(self, container: str) -> list[str]:
        return [kind for (kind, c) in self.wanted if c == container]

    def total(self) -> int:
        return sum(self.wanted.values())


def option_containers(option: list[dict]) -> list[str]:
    """The containers one ground option places into, in the order it names them."""
    return list(
        dict.fromkeys(a["args"][1] for a in option if a["predicate"] in PLACE_PREDICATES and len(a["args"]) == 2)
    )


def commit_to_container(options: list[list[dict]], container: str) -> list[list[dict]]:
    """Keep only the options whose place atoms all name ``container`` (TaskSpec.container).

    A task whose goal grounds into "any of these containers will do" gives a demand, read across the options,
    that asks for every item in every container. The runner then sends each item to whichever is nearest it and
    satisfies none of them. Naming one container keeps the options that use it alone.

    Atoms that are not placements (a press, an open) are kept exactly as they are, and if nothing matches the
    options are returned untouched rather than emptied -- a name that does not occur in this task must not
    silently delete its goal.
    """
    kept = [
        option
        for option in options
        if all(
            a["args"][1] == container
            for a in option
            if a["predicate"] in PLACE_PREDICATES and len(a["args"]) == 2
        )
        and any(a["predicate"] in PLACE_PREDICATES and a["args"][1:2] == [container] for a in option)
    ]
    if not kept:
        log.warning(f"container {container!r} names no goal option of this task; the goal is left as it is")
        return options
    log.info(f"committing the goal to {container}: {len(kept)} of {len(options)} options use it alone")
    return kept[:1]


def one_container_goal(options: list[list[dict]]) -> list[list[dict]]:
    """Commit to a single option when the goal wants everything in the SAME container.

    "Put every toy in a bookcase", with two bookcases in the room, grounds into one option per bookcase, and
    satisfying either scores the task. Reading demand across both asks for every toy in bookcase_1 AND every toy
    in bookcase_2; the runner then sends each toy to whichever container is nearest it, splits them between the
    two, and satisfies neither option. collecting_childrens_toys is that task, and it scored 0.000 with six picks
    executed (2026-09-14).

    The tell is whether any option spreads its items over more than one container:

      putting_away_toys   each toy may go to EITHER box independently, so the options include mixed ones
                          (toy 1 here, toy 2 there). Merging is right, and it scores 0.8125 that way.
      gift baskets        every option fills all four baskets. Merging is right.
      the bookcase case   every option uses exactly ONE container. Merging is wrong.

    So: several options, none of which uses more than one container, means the goal is asking for one container
    to take everything. Keep the first such option, deterministically. Anything else is left exactly as it was.
    """
    if len(options) < 2:
        return options
    if any(len(option_containers(o)) > 1 for o in options):
        return options
    for option in options:
        if len(option_containers(option)) == 1:
            return [option]
    return options


def place_demand(options: list[list[dict]]) -> Demand:
    """Read the demand off the goal's ground options (see ``Demand``)."""
    demand = Demand()
    for option in one_container_goal(options):
        counts = Counter()
        for a in option:
            if a["predicate"] not in PLACE_PREDICATES or len(a["args"]) != 2:
                continue
            item, container = a["args"]
            if item == container:
                # "nextto(can_of_soda_1, can_of_soda_1)" -- a thing beside itself. bddl grounds the pairing
                # predicate over every pair including the reflexive one; it is true from the start and there is
                # nothing to carry. Counting it made setup_a_bar_for_a_cocktail_party and laying_tile_floors ask
                # for items to be delivered to themselves.
                continue
            kind = bddl_category(item)
            counts[(kind, container)] += 1
            demand.items.setdefault(kind, [])
            if item not in demand.items[kind]:
                demand.items[kind].append(item)
            demand.predicate[(kind, container)] = a["predicate"]
        for key, n in counts.items():
            demand.wanted[key] = max(demand.wanted.get(key, 0), n)
    return demand


def open_targets(goal: list[dict]) -> dict:
    """Objects a goal wants open or shut: {name: True to open, False to close}.

    ``open(x)`` asks for it open; the negation compiles to a flat ``not`` atom the way a switch-off goal does
    (``['not', 'open', 'cabinet.n.01_1']``), and asks for it shut. 23 of the 100 challenge tasks score such an
    atom directly.
    """
    out = {}
    for atom_ in goal:
        args = list(atom_.get("args", []))
        if atom_.get("predicate") == "open" and args:
            out[args[0]] = True
        elif atom_.get("predicate") == "not" and len(args) >= 2 and args[0] == "open":
            out[args[1]] = False
    return out


def press_targets(goal: list[dict]) -> list[str]:
    """The objects whose button the goal asks to be pressed, in goal order. A goal atom that asks for a switch to
    be *off* reaches the runner as ``not(toggled_on, x)`` (a HEAD's flat tokens; bddl compiles the negation into
    the ground atom) and is a press too: the press is open loop either way, and the switch starts in the state the
    goal wants changed."""
    targets = []
    for a in goal:
        if a["predicate"] == "toggled_on" and a["args"]:
            targets.append(a["args"][0])
        elif a["predicate"] == "not" and a["args"][:1] == ["toggled_on"] and len(a["args"]) > 1:
            targets.append(a["args"][1])
    return list(dict.fromkeys(targets))


@dataclass
class TaskSpec:
    """What a task says about itself (``tasks/<task>.yaml``): only what its BDDL definition does not say.
    ``plan``: ``transfer``, ``press`` or ``auto`` (both, transfers first). ``press``: ``hold`` (pick the object,
    press with the other hand) or ``in_place``. ``attempts_per_item``: tries an item gets before the runner moves
    on. ``phrases``: detector words per category, for the onboard source."""

    task: str
    instruction: str
    plan: str = "auto"
    press: str = "hold"
    attempts_per_item: int = 2
    phrases: dict = field(default_factory=dict)
    # Per-container tailoring for opening, because the general skill does not reach every asset. Keyed by the
    # container's name in the goal, e.g.
    #     opens:
    #       cabinet.n.01_1: {joint: j_link_4, height: 0.74, fraction: 0.8}
    # joint: open THIS joint rather than whichever is nearest the hand. height: take hold at this world z on the
    # panel rather than searching up its face. fraction: how much of the joint's range to pull.
    # The user allowed this on 2026-09-13 ("open may not need to be generic to all drawer and cabinet, you can
    # tailor to a certain task a certain drawer"). Every value used is logged, so a score is never mistaken for a
    # general capability -- see the README's "Kept out of the pipeline" list.
    opens: dict = field(default_factory=dict)
    # The one container this goal's items should all go to, when the goal grounds into options the runner cannot
    # choose between. sorting_vegetables asks for thirteen vegetables in a mixing bowl and grounds into 27
    # options; read across all of them the demand asks for every vegetable in EVERY bowl, so the runner sends
    # each one to whichever bowl is nearest it, spreads them over three, and satisfies no option. Naming the
    # container collapses the demand onto it. Only ever narrows the demand -- it cannot invent work.
    container: str = ""

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
    """Runs a task's goal against an episode: transfers from the demand its goal options describe, presses the
    way its description says."""

    def __init__(self, spec: TaskSpec, goal: list[dict], options: list[list[dict]] | None = None, attempts=None):
        self.spec = spec
        self.goal = list(goal)
        self.options = [list(o) for o in options] if options else [list(goal)]
        if spec.container:
            self.options = commit_to_container(self.options, spec.container)
        self.demand = place_demand(self.options)
        self.attempts = spec.attempts_per_item if attempts is None else int(attempts)
        self.tries = Counter()  # item -> transfers attempted for it, over the whole instance

    @property
    def instruction(self) -> str:
        return self.spec.instruction

    def run(self, ep) -> None:
        self.tries.clear()  # one instance's attempts say nothing about the next
        presses = press_targets(self.goal)
        opens = open_targets(self.goal)
        handled = (*PLACE_PREDICATES, "toggled_on", "holding", "open")
        other = {
            a["predicate"]
            for a in self.goal
            if a["predicate"] not in handled and (a["args"][:1] or [""])[0] not in handled
        }
        if other:
            log.warning(f"{self.spec.task}: no sub-plan for goal atoms {sorted(other)}; they are left alone")
        for name, wanted_open in opens.items():
            # A goal that asks for a container open (or shut) in its own right, rather than as the way into one.
            # Before the transfers, since a container this goal wants open is very often the one they fill.
            if ep.is_shut(name) == wanted_open:
                ep.open_up(name, fraction=OPEN_FRACTION_SCORED if wanted_open else 0.0)
        if self.spec.plan in ("transfer", "auto") and self.demand.total():
            self.run_transfers(ep)
            self.spend_what_is_left(ep)
        elif self.spec.plan == "transfer" and not opens:
            raise ValueError(f"{self.spec.task}: the goal has no inside/ontop atoms for the transfer plan")
        if self.spec.plan in ("press", "auto") and presses:
            for obj in presses:
                # Do not press a switch that is already where the goal wants it. press_targets assumes "the
                # switch starts in the state the goal wants changed", and installing_a_modem is the
                # counter-example: its (:init) contains (toggled_on modem.n.01_1) and its goal asks for the modem
                # ON, so a press would turn it OFF and lose a condition the task already had (2026-09-14).
                wanted = any(a["predicate"] == "toggled_on" and a["args"][:1] == [obj] for a in self.goal)
                now = ep.switched_on(obj) if hasattr(ep, "switched_on") else None
                if now is not None and now == wanted:
                    log.info(f"{obj} is already {'on' if now else 'off'}, which is what the goal asks; not pressing it")
                    continue
                self.run_press(ep, obj)
        elif self.spec.plan == "press":
            raise ValueError(f"{self.spec.task}: the goal has no toggled_on atoms for the press plan")

    # ---------------------------------------------------------------- transfers
    def run_transfers(self, ep) -> int:
        """Fill the demand the goal options describe, and return how many items were delivered.

        Containers nearest the items that could go in them first; for each item a container wants, the items of
        that kind still loose, nearest the edge of whatever they stand on (the reachable ones) and nearest the
        container. An item is tried ``attempts`` times in the pass; one still in the hand after a failed place is
        put down, and a hand still full at the next transfer is emptied first."""
        wanted = dict(self.demand.wanted)
        done = self.settled(ep, wanted)
        log.info(
            f"goal demand {sorted((f'{k} x{n} -> {c}') for (k, c), n in wanted.items() if n > 0)}"
            + (f"; already there: {sorted(done)}" if done else "")
        )
        delivered = 0
        for container in self.order_containers(ep, wanted, done):
            for kind in self.demand.kinds_for(container):
                for _ in range(wanted.get((kind, container), 0)):
                    item = self.transfer_one(ep, kind, container, done)
                    if item is None:
                        break
                    done.add(item)
                    delivered += 1
        return delivered

    def spend_what_is_left(self, ep) -> None:
        """Keep sweeping the atoms still open while the step budget has room and a pass is still winning items.

        A pass ends when every item has had its ``attempts_per_item`` tries, and that is usually long before the
        episode's step limit: measured over the 70 instances on disk on 2026-09-15, the mean instance finished
        having used 32% of its budget and only three reached the limit. The rest was simply never claimed. A
        further pass costs steps that were not going to be spent, and it is not a repeat of the same thing -- a
        pass chooses its stance again, and an item that could not be reached or seen from one stance is often
        fine from the next.

        Two guards. A pass that delivers nothing ends it, so an atom that cannot be done costs one pass rather
        than the whole budget. And a pass only starts with room to finish one, so this never turns a run that
        would have ended cleanly into one cut off mid-motion. ``self.tries`` is cleared for each pass, which is
        what gives the item its attempts again; ``settled`` keeps the ones already delivered out of the way.
        """
        sim = getattr(ep, "sim", None)
        limit = getattr(sim, "max_steps", None)
        if not limit:  # no episode limit known: one pass is all the runner can justify
            return
        for sweep in range(1, EXTRA_SWEEPS + 1):
            used = getattr(sim, "n_steps", 0)
            if used > BUDGET_FOR_ANOTHER_SWEEP * limit:
                log.info(f"{used}/{limit} steps used; no room for another sweep")
                return
            if not self.demand.total():
                return
            self.tries.clear()
            log.info(f"sweep {sweep}: {used}/{limit} steps used, going round again for the atoms still open")
            if not self.run_transfers(ep):
                log.info(f"sweep {sweep} delivered nothing; stopping rather than spending the rest of the budget")
                return

    def settled(self, ep, wanted: dict) -> set:
        """Items the goal already has where it wants them when the instance starts (a bin that stands on the floor
        already, an item in its container): they cost nothing and are never worked on.

        Judged by the goal's OWN predicate on the pair, not by a geometric stand-in. The stand-in read a floor
        target as "the item is low and the hand let go of it", which cannot tell one room's floor from another's:
        bringing_in_wood asks for three sheets of plywood ontop floor.n.01_2 and they start on floor.n.01_1, so
        all three were counted as delivered and the instance finished having run no rounds at all, scoring 0.000.
        laying_tile_floors failed the same way. A predicate the evaluator cannot judge leaves the item loose,
        which is the safe direction: the worst case is work that turns out to be unnecessary."""
        done = set()
        for (kind, container), n in list(wanted.items()):
            predicate = self.demand.predicate.get((kind, container), "ontop")
            for item in self.demand.items.get(kind, []):
                if n <= 0:
                    break
                if item in done:
                    continue
                try:
                    there = ep.goal_already_holds(predicate, item, container)
                except (KeyError, NotImplementedError):  # not localized yet: treat it as loose
                    continue
                if not there:
                    continue
                done.add(item)
                n -= 1
                wanted[(kind, container)] = n
        return done

    @staticmethod
    def gap(ep, a: str, b: str) -> float:
        """Distance between two objects, or infinity when one has no box (the task floor has none)."""
        try:
            return ep.distance(a, b)
        except (KeyError, NotImplementedError):
            return float("inf")

    def order_containers(self, ep, wanted: dict, done: set) -> list[str]:
        """Containers that still want something, nearest first: by the distance to the closest item that could go
        in them (the items' support for a table-to-floor task, the room for a scattered one)."""
        containers = [
            c for c in self.demand.containers if any(wanted.get((k, c), 0) > 0 for k in self.demand.kinds_for(c))
        ]

        def near(container: str) -> float:
            gaps = [
                self.gap(ep, container, item)
                for kind in self.demand.kinds_for(container)
                for item in self.demand.items.get(kind, [])
                if item not in done
            ]
            return min(gaps) if gaps else float("inf")

        return sorted(containers, key=near)

    def transfer_one(self, ep, kind: str, container: str, done: set) -> str | None:
        """Move one item of ``kind`` into ``container``; the item moved, or None when none could be."""
        candidates = [i for i in self.demand.items.get(kind, []) if i not in done and self.tries[i] < self.attempts]
        if not candidates:
            return None
        supports = {i: ep.support_of(i) for i in candidates}
        candidates.sort(key=lambda i: (ep.edge_gap(i, supports[i]), self.gap(ep, i, container)))
        predicate = self.demand.predicate.get((kind, container), "inside")
        for item in candidates[: self.attempts]:
            self.tries[item] += 1
            if self.transfer(ep, predicate, item, container, supports[item]):
                return item
        return None

    def transfer(self, ep, predicate: str, item: str, container: str, support: str) -> bool:
        if not self.free_hand(ep, support):
            return False
        # A shut container has no inside to place into: its hull's top is the only surface a placement can find,
        # which is how store_honey put its jar ON the cabinet (2026-09-13). Open it BEFORE picking anything up --
        # the hand that pulls the drawer is the hand that would be carrying the item -- and open it far enough to
        # reach in, which is a much longer stroke than the one that scores an `open` atom.
        if ep.is_shut(container):
            log.info(f"{container} is shut; opening it before fetching {item}")
            if not ep.open_up(container, fraction=OPEN_FRACTION_REACH):
                log.info(f"{container} would not open; {item} has nowhere to go")
                return False
        if not ep.pick(item):
            return False
        if ep.is_floor(container):  # "on the floor": wherever the robot stands is the floor
            return ep.put_down(item, ep.floor)
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
        floor where the robot stands, else from a fresh pose at the item's support, else by opening the hand where
        it is (the item falls; better than a hand that stays full for the rest of the episode). False when it
        stays."""
        held = ep.held_names()
        if not held:
            return True
        name = held[0]
        log.info(f"{name} is still in the hand; putting it down before the next pick")
        if ep.put_down(name, ep.floor):
            return True
        if support != ep.floor:  # there is nowhere to "stand for" the floor, and it has no box to stand by
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


def strategy_for(task: str, goal: list[dict], options=None, **kwargs) -> Runner:
    if task not in STRATEGIES:
        raise ValueError(f"no task description for {task!r} in {TASKS_DIR}; known: {sorted(STRATEGIES)}")
    return Runner(STRATEGIES[task], goal, options=options, **kwargs)
