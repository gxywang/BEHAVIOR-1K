"""The StepLedger (WEEK4_PLAN 3.3, W4-A2): who stepped the simulator, and who wrote to it, per innermost owner.

Installed on the R1ProSim INSTANCE for one episode, identically under the legacy Runner and the connector host, when
any of --runner-tape, --wstape, --audit or --runner connector is given; with no week-4 flag the bench installs nothing.
It wraps ``sim.step_env`` (the one env.step on the episode path) and counts, per the innermost owner on its stack,
the env.step calls and the ``n_steps`` deltas; ``with ledger.owner(name)`` is how a host names a phase (``rt``,
``observe``, ``go_to``, ``host.build``, ``refresh``), and ``install_episode`` puts ``ep.<member>`` owners on the
Episode instance for its nine writes, as instance attributes, so the Episode's own ``self.stand_for`` inside ``pick``
nests: the steps go to ``ep.stand_for``, the innermost, and each TOP-LEVEL ``ep.*`` call's whole delta is recorded in
``calls``. Anything stepped outside every owner is ``unowned``; steps taken while ``sim.episode_open`` is False (the
video tail) are counted apart, in ``closed``.

Also counted per owner, never changed: the teleports (``place_robot``: the ``sim.teleports`` delta across each
``sim.move_base``, the one place a teleport is counted, so the column sums to the result's ``bench.teleports``; a
refused placement raises before it and ``place_robot_calls`` counts every ``sim.place_robot`` call, refused or not),
``sim.capture`` and ``sim.look_at`` (instance wrappers),
the renderer's ``render`` (``og.sim`` at the bench: ``renders`` are the calls made outside an env step, a capture's
say; the one every ``og.sim.step`` makes of itself is ``renders_in_step``), the robot's ``set_joint_positions``
(instance wrapper), every object class's ``set_position_orientation`` (class-level wrappers, restored on ``finish``)
and writes to the hand record ``sim.held_objects`` (a counting dict in its place; a replacement dict is re-wrapped,
and counted as one write of the owner in flight, at the next step or owner boundary). ``finish()`` restores every
wrapper and returns the rows in counters.py's ledger schema.

Observers (``ledger.observers``) see every env.step's action after it ran, which is how the GripperWatch (oracle/)
reads the gripper commands without a wrapper of its own. Nothing here reads the simulator's truth.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import asdict, dataclass

# The Episode's writes (WEEK4_PLAN 4 rows 18-26; tape.WRITES): each gets an ``ep.<member>`` owner.
EP_MEMBERS = ("pick", "achieve", "put_down", "open_up", "release", "pour", "dwell", "stand_for", "walk_to_floor")
UNOWNED = "unowned"
_MISSING = object()


@dataclass
class Row:
    """One owner's counts. ``writes`` is the write-calls total (counters.py's ledger.jsonl column)."""

    owner: str
    steps: int = 0
    env_step_calls: int = 0
    place_robot: int = 0
    place_robot_calls: int = 0
    capture: int = 0
    look_at: int = 0
    renders: int = 0
    renders_in_step: int = 0
    set_joint_positions: int = 0
    set_position_orientation: int = 0
    held_writes: int = 0

    @property
    def writes(self) -> int:
        return self.set_joint_positions + self.set_position_orientation + self.held_writes

    def to_dict(self) -> dict:
        return {**asdict(self), "writes": self.writes}


@dataclass
class Call:
    """One top-level ``ep.*`` call: its whole step delta, nested owners included."""

    index: int
    owner: str
    call_id: str | None
    step_before: int
    step_after: int | None = None
    env_step_calls: int = 0
    error: str | None = None

    @property
    def steps(self) -> int | None:
        return None if self.step_after is None else self.step_after - self.step_before

    def to_dict(self) -> dict:
        return {**asdict(self), "steps": self.steps}


def _defining_classes(objects, name: str) -> list:
    """For each object's type, the class in its MRO that defines ``name``: the one place a class-level wrapper goes,
    once per class, whatever subclasses share it."""
    seen: dict = {}
    for o in objects:
        for cls in type(o).__mro__:
            if name in vars(cls):
                seen.setdefault(cls, True)
                break
    return list(seen)


class _HeldRecord(dict):
    """``sim.held_objects`` with every write counted by the ledger. Reads are a plain dict's."""

    __slots__ = ("_ledger",)

    def __init__(self, ledger, *a, **k):
        super().__init__(*a, **k)
        self._ledger = ledger

    def __setitem__(self, key, value):
        self._ledger._count("held_writes")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._ledger._count("held_writes")
        super().__delitem__(key)

    def pop(self, *a):
        self._ledger._count("held_writes")
        return super().pop(*a)

    def popitem(self):
        self._ledger._count("held_writes")
        return super().popitem()

    def update(self, *a, **k):
        self._ledger._count("held_writes")
        super().update(*a, **k)

    def clear(self):
        self._ledger._count("held_writes")
        super().clear()

    def setdefault(self, key, default=None):
        if key not in self:
            self._ledger._count("held_writes")
        return super().setdefault(key, default)

    def __reduce__(self):  # a deepcopy or a pickle of the record is a plain dict, not another counter
        return (dict, (dict(self),))


class _StepEnv:
    """The wrapper in ``sim.step_env``'s place; ``calls`` is the env.step count the state digest reads."""

    def __init__(self, ledger, original):
        self.ledger, self.original, self.calls, self.depth = ledger, original, 0, 0
        functools.update_wrapper(self, original)

    def __call__(self, action, *a, **k):
        led, sim = self.ledger, self.ledger.sim
        led._rewrap_held()
        before = int(getattr(sim, "n_steps", 0))
        self.depth += 1
        try:
            return self.original(action, *a, **k)
        finally:  # EpisodeOver is raised after n_steps counted the step (scene.py step_env)
            self.depth -= 1
            self.calls += 1
            after = int(getattr(sim, "n_steps", 0))
            led._stepped(after - before)
            for fn in list(led.observers):
                fn(action, sim)


class StepLedger:
    """See the module docstring. ``renderer``: the object whose ``render`` is counted (``og.sim``); ``robot`` and
    ``objects`` default to ``sim.robot`` and ``sim.objects.values()`` when the sim has them."""

    def __init__(self, sim, renderer=None, robot=_MISSING, objects=_MISSING):
        self.sim = sim
        self.rows: dict[str, Row] = {}
        self.closed = Row("closed")  # env.step calls while episode_open is False, apart from the owners
        self.calls: list[Call] = []
        self.observers: list = []
        self.call_id: str | None = None  # the connector's call id in flight (a host sets it); None under the Runner
        self._stack: list[str] = []
        self._restore: list = []  # (target, name, previous or _MISSING, class_level)
        self._depth: dict = {}  # method name -> the nesting of counted calls in flight (the outermost counts)
        self._ep_depth = 0
        self.n0 = int(getattr(sim, "n_steps", 0))
        self._installed = True
        self.step_env = _StepEnv(self, sim.step_env)
        self._wrap(sim, "step_env", self.step_env)
        for name, column in (("place_robot", "place_robot_calls"), ("capture", "capture"), ("look_at", "look_at")):
            if callable(getattr(sim, name, None)):
                self._count_calls(sim, name, column)
        mover = "move_base" if callable(getattr(sim, "move_base", None)) else "place_robot"
        if callable(getattr(sim, mover, None)):
            self._count_delta(sim, mover, "place_robot", lambda: int(getattr(sim, "teleports", 0)))
        if renderer is not None and hasattr(renderer, "render"):
            self._count_calls(renderer, "render", lambda: "renders_in_step" if self.step_env.depth else "renders")
        robot = getattr(sim, "robot", None) if robot is _MISSING else robot
        if robot is not None and hasattr(robot, "set_joint_positions"):
            self._count_calls(robot, "set_joint_positions", "set_joint_positions")
        objects = list(getattr(sim, "objects", {}).values()) if objects is _MISSING else list(objects or ())
        if robot is not None:
            objects.append(robot)
        for cls in _defining_classes(objects, "set_position_orientation"):
            self._count_calls(cls, "set_position_orientation", "set_position_orientation", class_level=True)
        self._rewrap_held(first=True)

    # ---------------------------------------------------------------- owners
    @property
    def current(self) -> str:
        return self._stack[-1] if self._stack else UNOWNED

    @contextmanager
    def owner(self, name: str):
        self._rewrap_held()
        self._stack.append(name)
        try:
            yield self.row(name)
        finally:
            self._rewrap_held()
            self._stack.pop()

    def row(self, name: str) -> Row:
        row = self.rows.get(name)
        if row is None:
            row = self.rows[name] = Row(name)
        return row

    def install_episode(self, ep) -> None:
        """``ep.<member>`` owners on the Episode instance for every member of EP_MEMBERS it has."""
        for member in EP_MEMBERS:
            fn = getattr(ep, member, None)
            if fn is None or not callable(fn):
                continue
            self._wrap(ep, member, self._owned(f"ep.{member}", fn))

    def _owned(self, name: str, fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            top = self._ep_depth == 0
            self._ep_depth += 1
            rec = None
            if top:
                rec = Call(len(self.calls), name, self.call_id, int(getattr(self.sim, "n_steps", 0)))
                rec._calls0 = self.step_env.calls
                self.calls.append(rec)
            try:
                with self.owner(name):
                    return fn(*args, **kwargs)
            except BaseException as e:
                if rec is not None:
                    rec.error = type(e).__name__
                raise
            finally:
                self._ep_depth -= 1
                if rec is not None:
                    rec.step_after = int(getattr(self.sim, "n_steps", 0))
                    rec.env_step_calls = self.step_env.calls - rec._calls0
                    del rec._calls0

        return call

    # ---------------------------------------------------------------- counting
    def _stepped(self, delta: int) -> None:
        if getattr(self.sim, "episode_open", True):
            row = self.row(self.current)
            row.env_step_calls += 1
            row.steps += delta
        else:
            self.closed.env_step_calls += 1
            self.closed.steps += delta

    def _count(self, column: str) -> None:
        row = self.row(self.current)
        setattr(row, column, getattr(row, column) + 1)

    def _count_calls(self, target, name: str, column, class_level: bool = False) -> None:
        """Count calls of ``target.name`` in ``column`` (or the column ``column()`` names at the call): the outermost
        call only, so a subclass's override that calls ``super()`` through a wrapped base, or a wrapped method calling
        itself, is one write."""
        fn = getattr(target, name)
        depth = self._depth

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            outer = depth.get(name, 0) == 0
            depth[name] = depth.get(name, 0) + 1
            try:
                if outer:
                    self._count(column() if callable(column) else column)
                return fn(*args, **kwargs)
            finally:
                depth[name] -= 1

        self._wrap(target, name, wrapper, class_level=class_level)

    def _count_delta(self, target, name: str, column: str, probe) -> None:
        """Add ``probe()``'s change across each outermost call of ``target.name`` to ``column`` of the owner the call
        began under, whether it returned or raised."""
        fn = getattr(target, name)
        depth, key = self._depth, f"delta:{name}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            outer = depth.get(key, 0) == 0
            depth[key] = depth.get(key, 0) + 1
            row, before = (self.row(self.current), probe()) if outer else (None, None)
            try:
                return fn(*args, **kwargs)
            finally:
                depth[key] -= 1
                if row is not None:
                    setattr(row, column, getattr(row, column) + probe() - before)

        self._wrap(target, name, wrapper)

    def _wrap(self, target, name: str, wrapper, class_level: bool = False) -> None:
        previous = target.__dict__.get(name, _MISSING) if hasattr(target, "__dict__") else _MISSING
        self._restore.append((target, name, previous, class_level))
        setattr(target, name, wrapper)

    def _rewrap_held(self, first: bool = False) -> None:
        held = getattr(self.sim, "held_objects", None)
        if not self._installed or held is None or isinstance(held, _HeldRecord):
            return
        if not first:
            self._count("held_writes")  # someone put a new dict there: one write
        self.sim.held_objects = _HeldRecord(self, held)

    # ---------------------------------------------------------------- the end
    @property
    def owners(self) -> dict:
        return {name: row.to_dict() for name, row in self.rows.items()}

    def totals(self) -> dict:
        """Σ owners against the sim's own count since ``n0``; U0-b's arithmetic (WEEK4_PLAN 5.6)."""
        n_steps = int(getattr(self.sim, "n_steps", 0))
        owned = sum(r.steps for r in self.rows.values())
        calls = sum(r.env_step_calls for r in self.rows.values())
        unowned = self.rows.get(UNOWNED)
        return {
            "n0": self.n0,
            "n_steps": n_steps,
            "owned_steps": owned,
            "owned_env_step_calls": calls,
            "sum_matches_sim": owned == n_steps - self.n0,
            "unowned_steps": 0 if unowned is None else unowned.steps,
            "unowned_env_step_calls": 0 if unowned is None else unowned.env_step_calls,
            "unowned_writes": 0 if unowned is None else unowned.writes,
            "closed_env_step_calls": self.closed.env_step_calls,
        }

    def finish(self) -> dict:
        """Restore every wrapper (once) and return the summary: the rows by owner, the closed row, the top-level
        ``ep.*`` calls and the totals."""
        if self._installed:
            self._installed = False
            for target, name, previous, class_level in reversed(self._restore):
                if previous is _MISSING:
                    try:
                        delattr(target, name)
                    except AttributeError:
                        pass
                else:
                    setattr(target, name, previous)
            self._restore.clear()
            held = getattr(self.sim, "held_objects", None)
            if isinstance(held, _HeldRecord):
                self.sim.held_objects = dict(held)
        return self.summary()

    def summary(self) -> dict:
        return {
            "owners": self.owners,
            "closed": self.closed.to_dict(),
            "calls": [c.to_dict() for c in self.calls],
            "totals": self.totals(),
        }
