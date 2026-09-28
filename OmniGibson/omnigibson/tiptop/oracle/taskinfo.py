"""TaskInfo for the pseudo planner (SPEC 5.2 task(); WEEK4_PLAN 3.4): the task the sim loaded, read once per instance
and served through the Runtime, so conn.task() is real on the bench and under the episode host. The goal options are
strategies.task_goal_options(sim) as Facts, the same call the host-built strategy makes (and so the same capped read:
the first ground option, then a Random(0) sample), never re-sampled here: the pseudo planner's Runner and the host's
strategy must see the same options (the equality assert, WEEK4_PLAN 3.3)."""

import logging

from b1k.bridge.strategies import GOAL_OPTIONS_READ, task_goal_options
from b1k.connector.goals import TaskInfo
from b1k.connector.types import Fact
from omnigibson.tiptop.oracle.world import ref

log = logging.getLogger(__name__)
_CACHE: dict = {}  # (name, instance, id(task)) -> (task, task_id, goal_options, scope, floor); the task object is
#                    kept so its id() cannot recur for another task while the entry lives


def atom_to_fact(a: dict) -> Fact:
    """A strategies atom as a Fact: {"predicate": "not", "args": [p, *a]} is Fact(p, a, False)."""
    p, args = a["predicate"], list(a["args"])
    if p == "not":
        return Fact(args[0], tuple(args[1:]), False)
    return Fact(p, tuple(args))


def task_info(sim, planners, max_steps, name: str) -> TaskInfo:
    """The loaded task's TaskInfo: its challenge id, the goal options as Facts, the scope as refs over
    sorted(task_scope()), the floor, the arms with a planner behind them (``planners``: the bench's dict, or the arm
    names), ``max_steps``; source oracle. The options, scope and floor are read once per (task, instance)."""
    task = sim.env.task
    key = (name, getattr(task, "activity_instance_id", None), id(task))
    if key not in _CACHE:
        from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_INDICES  # the challenge's table (B100_task_misc.csv)

        total = len(task.ground_goal_state_options)
        if total > GOAL_OPTIONS_READ:
            log.warning(
                f"{name}: {total} ground goal options, capped at {GOAL_OPTIONS_READ} (the first ground option, then a "
                f"Random(0) sample): goal_options[0] is task_goal_atoms(sim)"
            )
        options = tuple(tuple(atom_to_fact(a) for a in opt) for opt in task_goal_options(sim))
        scope = tuple(ref(sim, n) for n in sorted(sim.task_scope()))
        try:
            floor = ref(sim, sim.floor_name())
        except KeyError:  # no floor in the task's scope
            floor = None
        task_id = TASK_NAMES_TO_INDICES.get(name)
        if task_id is None:
            log.warning(f"{name!r} is not a challenge task: task_id -1")
            task_id = -1
        _CACHE[key] = (task, task_id, options, scope, floor)
    _, task_id, options, scope, floor = _CACHE[key]
    return TaskInfo(task_id, name, options, scope, floor, tuple(planners), max_steps, "oracle")
