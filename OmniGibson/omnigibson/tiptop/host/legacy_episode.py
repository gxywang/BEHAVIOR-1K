"""The episode-mode host pieces of the pseudo planner's Q1 shim (WEEK4_PLAN §3.3, W4-C): the legacy backend that
executes the Runner's literal Episode call taken from the LegacyChannel, the navigator that maps a ``legacy:`` stance
to Episode.stand_for / walk_to_floor, the registry whose precheck is advisory on a backend that owns its own
preconditions, and the EPISODE_SPECS the intent skills are registered in.

LegacyBackend (legacy_skills.py) is the week-3 single-round baseline and stays untouched; this backend is its
sibling for a whole episode: today's multi-round, re-stancing Episode methods, called exactly as the Runner called
them. The typed SkillCall is checked against that literal call (the rebuild check) and never executed itself."""

import inspect
import json
import logging
from collections import Counter
from typing import Any, Callable, ClassVar, Optional

import b1k.runtime.skillrun as skillrun
from b1k.bridge.protocol import bddl_category
from b1k.connector.skills import (
    Code,
    IntentArgs,
    NavResult,
    Precheck,
    SkillResult,
    Stance,
    Status,
    WorldUpdate,
    effects,
)
from b1k.connector.types import ObjRef
from b1k.skills.registry import SkillRegistry
from b1k.skills.specs import SPECS, spec
from b1k.skills.tiptop.backend import wrong_or_not_lifted
from omnigibson.tiptop.host.legacy_skills import FAILED_AS, _atom, _plain, classify
from omnigibson.tiptop.host.teleport_nav import MOVE_TO_STEPS

log = logging.getLogger("omnigibson.tiptop")
LEGACY_KEY = "legacy:"  # Stance.key of the bench-only Episode.stand_for op (b1k/connector/skills.py:186)
INTENTS = ("attach", "stamp", "cut", "heat", "aim", "pour")
# SPECS plus the intents the Runner's achieve() reaches the legacy backend with (WEEK4_PLAN §4 rows 19, 23). They
# are registered here alone, never in SPECS: no native backend serves them.
EPISODE_SPECS = {**SPECS, **{f"intent.{n}": spec(f"intent.{n}", IntentArgs, 1100, default_backend="legacy")
                             for n in INTENTS}}
NO_ENTRY = "no channel entry: the episode host runs only the pseudo planner's calls"


def _n_steps(ep) -> int:
    return int(getattr(getattr(ep, "sim", None), "n_steps", 0) or 0)


def _hands(ep) -> dict:
    hands = getattr(getattr(ep, "sim", None), "hands", None)
    return dict(hands()) if callable(hands) else {}


def _json(x):
    return json.loads(json.dumps(x, default=_plain))


def _bound(ep, method: str, args: tuple, kwargs: dict) -> Any:
    """The call as the Episode method binds it, defaults applied: one shape for the typed rebuild and the literal.
    A call the signature refuses (or a fake with no signature) is returned raw, so it compares unequal to a bound one
    only when it is one."""
    fn = getattr(ep, method, None)
    try:
        b = inspect.signature(fn).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return {"args": list(args), "kwargs": dict(kwargs)}
    b.apply_defaults()
    return dict(b.arguments)


def _pick_kw(kwargs: dict, *names: str) -> dict:
    """The declared extras only: an entry kwarg the typed call cannot carry, taken as the Runner gave it."""
    return {k: kwargs[k] for k in names if k in kwargs}


def rebuild(call, entry: dict, ep) -> Optional[tuple]:
    """The Episode call the typed SkillCall implies, plus the declared extras from the entry (WEEK4_PLAN §4, rows
    18-24): (args, kwargs), or None for a pairing the table does not know."""
    a, skill, method = call.args, call.skill, entry["method"]
    args, kwargs = tuple(entry["args"]), dict(entry["kwargs"])
    if skill == "pick_up" and method == "pick":
        return (a.obj.id,), _pick_kw(kwargs, "into")
    if skill == "place" and method == "achieve":
        return ([_atom(a.obj, r) for r in a.relations],), _pick_kw(kwargs, "arm")
    if skill == "place" and method == "put_down":
        return (a.obj.id, a.relations[0].target.id), _pick_kw(kwargs, "floor")
    if skill == "open" and method == "open_up":
        return (a.target.id,), {"fraction": a.min_fraction}
    if skill == "close" and method == "open_up":
        return (a.target.id,), {"fraction": 0.0}
    if skill == "press" and method == "achieve":
        return ([{"predicate": "toggled_on", "args": [a.target.id]}],), _pick_kw(kwargs, "arm")
    if skill == "release" and method == "release":
        return (), {}
    if skill == "intent.pour" and method == "pour":
        return (a.tool.id, a.target.id), {}
    if skill.startswith("intent.") and method == "achieve":
        atoms = list(args[0]) if args else []  # the entry's atom list is a declared extra: the typed call names only
        #                                        the primary atom's predicate (mode), tool and target, which must agree
        first = atoms[0] if atoms else {}
        names = list(first.get("args", ()))
        agrees = (first.get("predicate") == a.mode and names[:1] == [a.tool.id]
                  and (names[1] if len(names) > 1 else names[0] if names else None) == a.target.id)
        typed = {"predicate": a.mode, "args": [a.tool.id] + ([a.target.id] if a.target.id != a.tool.id else [])}
        return ([first if agrees else typed] + atoms[1:],), _pick_kw(kwargs, "arm")
    if skill == "wait" and method == "dwell":
        return (args[0] if args else kwargs.get("steps"),), {}
    return None


class EpisodeLegacyBackend:
    """Executes the Runner's literal Episode call (the channel entry under the call's id) on the sim thread, stepping
    the sim itself: it ends before its first yield, so it costs 0 Runtime steps, and the host's HostHooks re-seed the
    latch afterwards. The typed call is rebuilt and compared (channel.mismatch), never executed. The outcome is
    recorded on the channel before anything is judged; PASSTHROUGH exceptions (EpisodeOver) propagate, any other
    exception comes back through the channel as the same object and is a FAILED result. Status: the primary
    GoalChecker on effects(call) after the run, else (nothing to judge, or the judge cannot) the literal return and
    the error, as LegacyBackend rules. It never pre-judges a press: the literal call always runs. ``steps`` counts
    the sim steps its calls took, EpisodeOver included, for the host's U0 ledger."""

    name = "legacy"
    captures_in_own_run: ClassVar[bool] = True  # exempt from PERCEPT_REQUIRED: the Episode captures inside its run
    owns_preconditions: ClassVar[bool] = True  # the Episode's own retries and checks: the registry's are advisory

    def __init__(self, ep, channel, observe_now: Callable, classify: Callable = classify):
        self.ep, self.channel, self.observe_now, self.classify = ep, channel, observe_now, classify
        self.steps = 0

    def supports(self, call) -> bool:
        return call.skill in EPISODE_SPECS

    def check(self, call, svc) -> Precheck:
        return Precheck(True)

    def _refs(self, call) -> dict:
        a = call.args
        refs = [getattr(a, n, None) for n in ("obj", "tool", "target")]
        refs += [r.target for r in getattr(a, "relations", ())]
        return {r.id: r for r in refs if isinstance(r, ObjRef)}

    def _ref(self, label: str, known: dict) -> ObjRef:
        names = getattr(getattr(self.ep, "sim", None), "bddl_names", None) or {}
        name = names.get(label, label)
        return known.get(name) or ObjRef(name, bddl_category(name))

    def _check_rebuild(self, call, entry: dict) -> None:
        cid, method, args, kwargs = call.call_id, entry["method"], tuple(entry["args"]), dict(entry["kwargs"])
        literal = {"method": method, **_bound(self.ep, method, args, kwargs)}
        if call.skill == "wait" and method == "dwell":  # the typed steps are the literal's, capped by what is left
            sim = getattr(self.ep, "sim", None)
            lit = int(args[0] if args else kwargs.get("steps"))
            max_steps = getattr(sim, "max_steps", None)
            expect = min(lit, max_steps - _n_steps(self.ep)) if max_steps else lit
            if call.args.steps != expect:
                self.channel.mismatch(cid, {"method": method, "typed_steps": expect}, {**literal, "typed_steps": call.args.steps})
            return
        built = rebuild(call, entry, self.ep)
        if built is None:
            self.channel.mismatch(cid, {"method": method, "skill": call.skill, "pairing": "unknown"}, literal)
            return
        rebuilt = {"method": method, **_bound(self.ep, method, *built)}
        if rebuilt != literal:
            self.channel.mismatch(cid, rebuilt, literal)

    def run(self, call, svc, obs):
        cid = call.call_id
        entry = self.channel.take(cid)
        if entry is None:
            if call.skill != "wait":
                return SkillResult(cid, call.skill, self.name, Status.INFEASIBLE, Code.UNSUPPORTED, "check", NO_ENTRY,
                                   (), {}, svc.goals.primary)
            entry = {"method": "dwell", "args": (call.args.steps,), "kwargs": {}}
        method, args, kwargs = entry["method"], tuple(entry["args"]), dict(entry["kwargs"])
        self._check_rebuild(call, entry)
        # execute the LITERAL call: the Episode steps the sim itself
        records = getattr(self.ep, "records", None)
        n0, r0, hands0 = _n_steps(self.ep), len(records) if records is not None else 0, _hands(self.ep)
        value, err = None, None
        try:
            value = getattr(self.ep, method)(*args, **kwargs)
            self.channel.done(cid, True, value, None)
        except skillrun.PASSTHROUGH:
            raise
        except Exception as e:  # noqa: BLE001 - a legacy crash is a result; the shim re-raises the same object
            self.channel.done(cid, True, None, e)
            err = e
        finally:
            steps = _n_steps(self.ep) - n0
            self.steps += steps
        # a release returns None and raises when it cannot open the hand (LegacyBackend: ``release() or True``)
        legacy_ok = (err is None) if (method == "release" and value is None) else bool(value)
        new_records = _json(records[r0:]) if records is not None else []
        goal = effects(call)
        verdict, verdicts, after = None, {}, None
        if goal:
            try:
                after = self.observe_now()
                b, verdicts = svc.goals.judge(goal, after, who=cid)
                verdict = b.value
            except Exception as e:  # noqa: BLE001 - a judge that cannot: unknown, as verdict None
                log.warning(f"{cid}: the goal judge failed: {type(e).__name__}: {e}")
                verdict, verdicts = None, {}
        ok = (legacy_ok and err is None) if verdict is None else bool(verdict)
        if ok:
            status, code, phase = Status.SUCCEEDED, None, ""
        elif err is not None or not legacy_ok:
            why = err if err is not None else next((r.get("error") or r.get("why") for r in reversed(new_records)
                                                    if isinstance(r, dict) and (r.get("error") or r.get("why"))), None)
            status, code, phase = self.classify(why, call.skill)
            if phase == "execute" and steps == 0:  # no known text and the sim never stepped: nothing executed
                status, code, phase = Status.FAILED, Code.BACKEND_ERROR, None
        else:  # the legacy code says it worked; the GoalChecker says the requested relation does not hold
            status, code, phase = Status.FAILED, FAILED_AS.get(call.skill, Code.PLACED_WRONG), "verify"
            if call.skill == "pick_up":
                try:
                    code = wrong_or_not_lifted(svc, goal, after, None, cid, call.arm or "left")
                except Exception:  # noqa: BLE001 - a panel that cannot judge the holding atom: the plain code
                    code = Code.GRASP_MISSED
        known, hands1 = self._refs(call), _hands(self.ep)
        updates = tuple(WorldUpdate("held", self._ref(label, known), arm, source="oracle")
                        for label, arm in hands1.items() if hands0.get(label) != arm)
        updates += tuple(WorldUpdate("released", self._ref(label, known), arm, source="oracle")
                         for label, arm in hands0.items() if hands1.get(label) != arm)
        ev = {"legacy_ok": legacy_ok, "legacy_return": _json(value), "method": method, "records": new_records}
        return SkillResult(cid, call.skill, self.name, status, code, phase, str(err or ""), goal if ok else (),
                           verdicts, svc.goals.primary, world_updates=updates, steps=steps, requires_sim_clock=True,
                           evidence=ev)
        yield {}  # unreachable: makes run() a generator that ends before its first yield


class EpisodeNavigator:
    """A ``legacy:<names>`` stance is the Runner's stand_for / walk_to_floor (the channel's nav FIFO): the Episode
    teleports itself, on the sim clock, charged MOVE_TO_STEPS per teleport in shadow steps; an exception comes back
    through the channel as the same object, EpisodeOver passes through. Every other stance goes to the
    TeleportNavigator. ``steps`` counts the sim steps its own calls took, for the host's U0 ledger."""

    requires_sim_clock = True

    def __init__(self, ep, channel, teleport_nav, base_pose: Optional[Callable] = None):
        self.ep, self.channel, self.tp, self._base_pose = ep, channel, teleport_nav, base_pose
        self.steps = 0

    def base_pose(self):
        return self.tp.base_pose()

    def propose(self, req, k: int = 8) -> list:
        return self.tp.propose(req, k)

    def apply(self, update) -> None:
        self.tp.apply(update)

    def _landed(self, stance: Stance) -> Stance:
        p = self._base_pose() if self._base_pose is not None else None
        p = getattr(p, "value", p)
        return Stance(stance.key, p or stance.pose, stance.score, stance.why, stance.source)

    def go_to(self, stance: Stance, obs):
        if not stance.key.startswith(LEGACY_KEY):
            return (yield from self.tp.go_to(stance, obs))
        e = self.channel.take_nav()
        names = tuple(stance.key[len(LEGACY_KEY):].split(","))
        assert tuple(e["args"]) == names, f"the nav entry {e['args']} is not the stance's {names}"
        sim = getattr(self.ep, "sim", None)
        t0, n0, value, exc = int(getattr(sim, "teleports", 0) or 0), _n_steps(self.ep), None, None
        try:
            value = getattr(self.ep, e["method"])(*e["args"])
            self.channel.nav_done(True, value, None)
        except skillrun.PASSTHROUGH:
            raise
        except Exception as ex:  # noqa: BLE001 - Unreachable, or a crash: the shim re-raises the same object
            self.channel.nav_done(True, None, ex)
            exc = ex
        finally:
            self.steps += _n_steps(self.ep) - n0
        ok = exc is None and value is not False
        shadow = MOVE_TO_STEPS * (int(getattr(sim, "teleports", 0) or 0) - t0)
        detail = f"{type(exc).__name__}: {exc}" if exc is not None else f"Episode.{e['method']}{names}"
        return NavResult(ok, self._landed(stance), 0, shadow, detail), obs
        yield  # a generator that yields nothing: the Episode stepped the sim itself (requires_sim_clock)


class EpisodeRegistry(SkillRegistry):
    """A call routed to a backend that owns its preconditions (the Episode's own retries and checks) gets
    SkillSpec.check as an ADVISORY: what it would have refused is counted per code in ``advisory`` and the answer is
    always ok. No reach test (an IK request) and no percept rules for it. Every other route gets the full precheck."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.advisory: Counter = Counter()

    def precheck(self, call, svc, resources=None, obs=None) -> Precheck:
        if not getattr(self.backend_for(call, svc), "owns_preconditions", False):
            return super().precheck(call, svc, resources, obs)
        check = self.specs[call.skill].check
        if check is not None:
            try:
                pre = check(call, svc)
            except Exception as e:  # noqa: BLE001 - an advisory never refuses, a check that crashed least of all
                self.advisory[f"error:{type(e).__name__}"] += 1
            else:
                if not pre.ok:
                    self.advisory[pre.code.value if pre.code else "refused"] += 1
        return Precheck(True)
