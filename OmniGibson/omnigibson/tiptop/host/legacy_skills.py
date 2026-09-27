"""Today's Episode methods as a skill backend (SPEC D21, bench only): the legacy baseline every native skill is
gated against, and the one classifier that turns legacy error text into codes."""

import json
from typing import Callable, ClassVar

import b1k.runtime.skillrun as skillrun
from b1k.connector.skills import Code, PlaceArgs, Precheck, Rel, SkillResult, Status, WorldUpdate, effects
from b1k.skills.tiptop.backend import wrong_or_not_lifted

# Legacy error text -> (status, code, phase); the first match wins. The texts are the manip2 corpus' round errors
# and open/close "why"s (2026-09-25), most frequent first.
LEGACY_CODES = (
    ("No satisfying particle", Status.INFEASIBLE, None, "particles"),  # the skill's own: NO_GRASP / NO_PLACEMENT
    ("Motion planning", Status.INFEASIBLE, Code.NO_MOTION, "motion"),
    ("motion validation rejected", Status.INFEASIBLE, Code.EXEC_REFUSED, "motion"),
    ("failed to track the checked path", Status.FAILED, Code.BLOCKED, "execute"),  # the executor stopped mid-way,
    #   held back (fell_behind): the object still in the hand, nothing placed (W3 L4 shoe2 t0: it read placed_wrong)
    ("no valid depth points inside the workspace", Status.INFEASIBLE, Code.NO_STANCE_HERE, "check"),  # in view (the
    #   toy box: 5433 head pixels), out of the planner's workspace: out of reach from here, not unseen (W3 A1/L3)
    ("not visible", Status.PRECONDITION_UNMET, Code.NOT_VISIBLE, "check"),
    ("were not found", Status.PRECONDITION_UNMET, Code.NOT_VISIBLE, "check"),
    ("no stance", Status.INFEASIBLE, Code.NO_STANCE_HERE, "check"),
    ("no base pose", Status.INFEASIBLE, Code.NO_STANCE_HERE, "check"),
    ("no collision-free base destination", Status.INFEASIBLE, Code.NO_STANCE_HERE, "check"),
    ("can be taken hold of", Status.INFEASIBLE, Code.NO_FEATURE, "check"),
    ("no grasp on", Status.INFEASIBLE, Code.NO_MOTION, "check"),
    ("self-collision", Status.INFEASIBLE, Code.NO_MOTION, "motion"),  # the server's S1 precheck: nothing planned
)
NO_SOLUTION = {"pick_up": Code.NO_GRASP}  # else NO_PLACEMENT
FAILED_AS = {"pick_up": Code.GRASP_MISSED, "open": Code.STALLED, "close": Code.STALLED,
             "release": Code.BLOCKED}  # else PLACED_WRONG. release: the hand did not let go


def classify(why, skill: str) -> tuple:
    """(status, code, phase) of a legacy failure from its exception or error text; a failure with no known text
    moved something and did not work: FAILED with the skill's own execution code."""
    text = str(why or "")
    for needle, status, code, phase in LEGACY_CODES:
        if needle in text:
            return status, code or NO_SOLUTION.get(skill, Code.NO_PLACEMENT), phase
    return Status.FAILED, FAILED_AS.get(skill, Code.PLACED_WRONG), "execute"


def _plain(x):
    return x.tolist() if hasattr(x, "tolist") else str(x)


def _atom(obj, r) -> dict:
    return {"predicate": {"on": "ontop", "in": "inside", "next_to": "nextto"}.get(r.rel.value, r.rel.value),
            "args": [obj.id, r.target.id]}


class LegacyBackend:
    """Runs inside the Runtime on the sim thread and steps the sim itself (R1ProSim.step); ends before its first
    yield, so it costs 0 Runtime steps, and the host's HostHooks re-seed the latch afterwards. Never under the
    Evaluator.

    Status comes from the primary GoalChecker on the REQUESTED relations, judged on the observation AFTER the legacy
    code stepped the sim (observe_now); what the legacy code returned goes to evidence["legacy_ok"], its round
    records to evidence["records"], and a relation the legacy wire bends onto on() is named in
    evidence["degraded_to"]: every under and touching, and an in whose target ``has_cavity(target, item)`` says
    the legacy wire sends no compartment floor for. A relation in strict_relations returns UNSUPPORTED instead of
    running. A success names what the hand now holds or let go (WorldUpdate held / released), as a native pick's
    does, so the WorldView the next precheck reads agrees with it.

    single_round (the skill bench): Episode.pick / open_up run ONE round from the case's stance, with no stand_for
    and no hidden push, so the baseline compares with a native call that never moves the base. The pseudo
    planner's Q1 shim keeps single_round=False: today's multi-round, re-stancing behaviour."""

    name = "legacy"
    BENT = (Rel.UNDER, Rel.TOUCHING)
    captures_in_own_run: ClassVar[bool] = True  # exempt from PERCEPT_REQUIRED: it captures inside its own run

    def __init__(self, episode, observe_now: Callable, classify: Callable = classify,
                 has_cavity: Callable = lambda target, item: True, strict_relations: frozenset = frozenset(),
                 single_round: bool = False):
        self.ep, self.classify, self.observe_now, self.has_cavity = episode, classify, observe_now, has_cavity
        self.strict, self.single = frozenset(strict_relations), single_round
        one = {"single_round": True} if single_round else {}
        self.dispatch = {
            "pick_up": lambda c: self.ep.pick(c.args.obj.id, into=None, **one),
            "place": lambda c: self.ep.achieve([_atom(c.args.obj, r) for r in c.args.relations], arm=c.arm or "left"),
            "open": lambda c: self.ep.open_up(c.args.target.id, c.args.min_fraction, joint=c.args.joint, **one),
            "close": lambda c: self.ep.open_up(c.args.target.id, 0.0, joint=c.args.joint, **one),
            "press": lambda c: self.ep.achieve([{"predicate": "toggled_on", "args": [c.args.target.id]}],
                                               arm=c.arm or "left"),
            "release": lambda c: self.ep.release() or True,  # returns None: it raises when it cannot open the hand
        }

    def supports(self, call) -> bool:
        return call.skill in self.dispatch

    def check(self, call, svc) -> Precheck:
        return Precheck(True)

    def _bent(self, call) -> list:
        if not isinstance(call.args, PlaceArgs):
            return []
        return [r.rel for r in call.args.relations
                if r.rel in self.BENT or (r.rel is Rel.IN and not self.has_cavity(r.target, call.args.obj))]

    def run(self, call, svc, obs):
        bent_rels = self._bent(call)
        bent = [f"{r.value}->on" for r in bent_rels]
        if self.strict.intersection(bent_rels):
            return SkillResult(call.call_id, call.skill, self.name, Status.INFEASIBLE, Code.UNSUPPORTED, "check",
                               f"the legacy wire would place {bent} on top", (), {}, svc.goals.primary,
                               evidence={"degraded_to": "on", "relations": bent})
        n0, r0, err, legacy_ok = self.ep.sim.n_steps, len(self.ep.records), None, False
        goal = effects(call)
        if call.skill == "press" and goal and svc.goals.judge(goal, self.observe_now(), who=call.call_id)[0].value is True:
            return SkillResult(call.call_id, call.skill, self.name, Status.SUCCEEDED, None, "", "already in the wanted "
                               "state", goal, {}, svc.goals.primary, requires_sim_clock=True)  # SPEC 6.5: 0 steps
        # the frames before the run, the reference a perception shadow judges holding and lifted against (a
        # legacy pick read None on every row without one); rendered now, as nothing renders until it is read
        ref = self.observe_now().sensors if set(getattr(svc.goals, "checkers", ())) - {"scorer", "none"} else None
        if ref is not None and getattr(ref, "views", None) is None:
            ref = None
        try:
            legacy_ok = bool(self.dispatch[call.skill](call))
        except skillrun.PASSTHROUGH:
            raise
        except Exception as e:  # noqa: BLE001 - a legacy crash is a result
            err = e
        records = json.loads(json.dumps(self.ep.records[r0:], default=_plain))
        after = self.observe_now()
        verdict, verdicts = svc.goals.judge(goal, after, ref, who=call.call_id)  # the frames after the run
        ok = legacy_ok and err is None if verdict.value is None else bool(verdict.value)
        if ok:
            status, code, phase = Status.SUCCEEDED, None, ""
        elif err is not None or not legacy_ok:
            why = err if err is not None else next((r.get("error") or r.get("why") for r in reversed(records)
                                                    if r.get("error") or r.get("why")), None)
            status, code, phase = self.classify(why, call.skill)
            if phase == "execute" and self.ep.sim.n_steps == n0:  # no known text and the sim never stepped: nothing
                status, code, phase = Status.FAILED, Code.BACKEND_ERROR, None  # executed, so no execution code
        else:  # the legacy code says it worked; the GoalChecker says the requested relation does not hold
            status, code, phase = Status.FAILED, FAILED_AS.get(call.skill, Code.PLACED_WRONG), "verify"
            if call.skill == "pick_up":  # as a native pick's: held and still on its support is NOT_LIFTED (the full
                #                          baskets held clear read grasp_missed), another object WRONG_OBJECT
                code = wrong_or_not_lifted(svc, goal, after, ref, call.call_id, call.arm or "left")
        ev = {"legacy_ok": legacy_ok, "single_round": self.single, "records": records}
        if bent:
            ev.update(degraded_to="on", relations=bent)
        arm, obj = call.arm or "left", getattr(call.args, "obj", None)
        kind = {"pick_up": "held", "place": "released", "release": "released"}.get(call.skill) if ok else None
        return SkillResult(call.call_id, call.skill, self.name, status, code, phase, str(err or ""),
                           goal if ok else (), verdicts, svc.goals.primary,
                           world_updates=(WorldUpdate(kind, obj, arm),) if kind else (),
                           steps=self.ep.sim.n_steps - n0, requires_sim_clock=True, evidence=ev)
        yield {}  # unreachable: makes run() a generator that ends before its first yield

