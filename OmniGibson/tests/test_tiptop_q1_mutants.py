"""The Q1 harness's mutants (WEEK4_PLAN §5.2): nine plausible ways to get the pseudo planner's shim or its host pieces
wrong, each applied by monkeypatch, each of which must be caught by E1 (the strategies suite run directly and through
the connector stack, logs compared), by G2 on the tapes the fake suite records, or by a named unit test.

  1  is_shut as ``not is_open``                         (a jointless or unknown container reads shut)
  2  holding from an updates-only ledger                (the legacy backend's hand updates, not the Episode's record)
  3  pick without into                                  (HEAD's LegacyBackend drops it: legacy_skills.py:77)
  4  put_down as a plain achieve                        (the Episode's put_down never runs)
  5  the return read from status, not the channel's literal
  6  a navigator that ignores legacy keys               (every stance to the teleport navigator)
  7  dwell returning SkillResult.steps under a legacy outcome
  8  holding served from held()                         (agrees with the fakes; must fail G2)
  9  a rebuild that reads positional args from the literal  (must fail the rebuild unit test below)

Every detector runs on every mutant, so the table says which ones catch which. The strategies suite runs in-process:
its zero-argument test functions are called under a q1_pytest Session, the direct one once as the baseline. With
Q1_MUTANT_TABLE=PATH the kill matrix is also written there as JSON."""

import dataclasses
import importlib.util
import inspect
import json
import os
from pathlib import Path

import pytest

import b1k.planner.pseudo.shim as shim_mod
import omnigibson.tiptop.host.legacy_episode as le
from b1k.bridge import strategies
from b1k.bridge.strategies import atom
from b1k.connector.skills import (
    CloseArgs,
    IntentArgs,
    OpenArgs,
    PickArgs,
    PlaceArgs,
    PressArgs,
    Rel,
    Relation,
    SkillCall,
    Status,
    WaitArgs,
)
from b1k.connector.types import ObjRef
from b1k.planner.pseudo import tape as tp
from b1k.planner.pseudo.calls import call_skill
from omnigibson.tiptop.host import q1_harness as qh
from omnigibson.tiptop.host import q1_pytest

SUITE = Path(__file__).with_name("test_tiptop_strategies.py")
EOC = shim_mod.EpisodeOverConnector


# ------------------------------------------------------------------------------------------------------ the mutants
def m1_is_shut_as_not_is_open(mp):
    mp.setattr(EOC, "is_shut", lambda self, name: not self._world().is_open(self._ref(name)).value)


def m2_holding_from_an_updates_only_ledger(mp):
    def holding(self, o):
        arms = {}
        for u in self.updates:
            if u.obj is not None and u.obj.id == o.id:
                if u.kind == "held":
                    arms[u.arm] = True
                elif u.kind == "released":
                    arms.pop(u.arm, None)
        return self._b(tuple(arms))

    mp.setattr(qh.EpisodeWorld, "holding", holding)


def m3_pick_without_into(mp):
    def pick(self, bddl, **kw):
        entry = {"method": "pick", "args": (bddl,), "kwargs": {k: v for k, v in kw.items() if k != "into"}}
        return self._run("pick_up", PickArgs(self._ref(bddl)), "left", entry)

    mp.setattr(EOC, "pick", pick)


def m4_put_down_as_a_plain_achieve(mp):
    def put_down(self, bddl, support, **kw):
        atoms = [atom("ontop", bddl, support)]
        return self._run("place", PlaceArgs(self._ref(bddl), (Relation(Rel.ON, self._ref(support)),)), "left",
                         {"method": "achieve", "args": (atoms,), "kwargs": {}})

    mp.setattr(EOC, "put_down", put_down)


def m5_return_from_status(mp):
    def _run(self, skill, args, arm, entry, budget_steps=None):
        self._k += 1
        cid = f"q1-{self._k}"
        self._channel.put(cid, entry)
        r = call_skill(self._conn, SkillCall(skill, args, arm=arm, call_id=cid, budget_steps=budget_steps), self._views)
        if r is None:
            return False
        out = self._channel.outcome(cid)
        if out is not None and out["exc"] is not None:
            raise out["exc"]
        return r.steps if skill == "wait" else r.status is Status.SUCCEEDED

    mp.setattr(EOC, "_run", _run)


def m6_navigator_ignores_legacy_keys(mp):
    def go_to(self, stance, obs):
        return (yield from self.tp.go_to(stance, obs))

    mp.setattr(le.EpisodeNavigator, "go_to", go_to)


def m7_dwell_returns_result_steps(mp):
    def dwell(self, steps):
        c = self._conn.clock()
        left = c.max_steps - c.step if c.max_steps else None
        n = max(0, min(steps, left)) if left is not None else steps
        self._k += 1
        cid = f"q1-{self._k}"
        self._channel.put(cid, {"method": "dwell", "args": (steps,), "kwargs": {}})
        r = call_skill(self._conn, SkillCall("wait", WaitArgs(n), call_id=cid, budget_steps=max(1, n)), self._views)
        return r.steps

    mp.setattr(EOC, "dwell", dwell)


def m8_holding_from_held(mp):
    def holding(self, bddl):
        return any(o.id == bddl for arm in ("left", "right") for o in (self._world().held(arm).value or ()))

    mp.setattr(EOC, "holding", holding)


def m9_rebuild_from_the_literal(mp):
    real = le.rebuild

    def rebuild(call, entry, ep):
        built = real(call, entry, ep)
        return None if built is None else (tuple(entry["args"]), built[1])

    mp.setattr(le, "rebuild", rebuild)


MUTANTS = [m1_is_shut_as_not_is_open, m2_holding_from_an_updates_only_ledger, m3_pick_without_into,
           m4_put_down_as_a_plain_achieve, m5_return_from_status, m6_navigator_ignores_legacy_keys,
           m7_dwell_returns_result_steps, m8_holding_from_held, m9_rebuild_from_the_literal]
MUST = {"m8_holding_from_held": "g2", "m9_rebuild_from_the_literal": "unit"}  # the detector the plan names


# ------------------------------------------------------------------------------------------- the rebuild unit test
class _Ep:
    """Episode writes with their real signatures, every call logged."""

    def __init__(self):
        self.calls = []

    def _did(self, method, args, kwargs):
        self.calls.append((method, args, kwargs))
        return True

    def pick(self, bddl, into=None, single_round=False):
        return self._did("pick", (bddl,), {"into": into})

    def achieve(self, atoms, arm="left", floor=None, done=None):
        return self._did("achieve", (atoms,), {"arm": arm})

    def put_down(self, bddl, support, floor=None):
        return self._did("put_down", (bddl, support), {"floor": floor})

    def open_up(self, name, fraction=None, single_round=False, joint=None):
        return self._did("open_up", (name,), {"fraction": fraction})

    def pour(self, item, target):
        return self._did("pour", (item, target), {})


def _r(name):
    return ObjRef(name, name.partition(".n.")[0])


JAR, APPLE, CAB, TABLE = _r("jar.n.01_1"), _r("apple.n.01_1"), _r("cabinet.n.01_1"), _r("table.n.02_1")
# (the literal entry, a typed call that names another object): every row must be a mismatch, and the literal runs
DISAGREE = [
    (("pick", (JAR.id,), {"into": CAB.id}), SkillCall("pick_up", PickArgs(APPLE), arm="left")),
    (("put_down", (JAR.id, TABLE.id), {}), SkillCall("place", PlaceArgs(JAR, (Relation(Rel.ON, CAB),)), arm="left")),
    (("achieve", ([atom("inside", JAR.id, CAB.id)],), {}),
     SkillCall("place", PlaceArgs(APPLE, (Relation(Rel.IN, CAB),)), arm="left")),
    (("achieve", ([atom("toggled_on", JAR.id)],), {}), SkillCall("press", PressArgs(CAB, want_on=None), arm="left")),
    (("open_up", (CAB.id, 0.8), {}), SkillCall("open", OpenArgs(JAR, min_fraction=0.8), arm="left")),
    (("open_up", (CAB.id,), {"fraction": 0.0}), SkillCall("close", CloseArgs(JAR), arm="left")),
    (("pour", (JAR.id, CAB.id), {}), SkillCall("intent.pour", IntentArgs(JAR, TABLE, "pour"), arm="left")),
]


def test_the_rebuild_check_sees_a_typed_object_the_literal_does_not_name():
    """The rebuild check over the harness stack: a typed call that names another object than the Runner's literal
    call is a typed/literal mismatch, one per row, and the literal is what runs."""
    ep = _Ep()
    with qh.build_episode_connector(ep, qh.members_of(ep)) as h:
        for k, ((method, args, kwargs), call) in enumerate(DISAGREE):
            cid = f"q1-{k + 1}"
            h.channel.put(cid, {"method": method, "args": args, "kwargs": kwargs})
            h.conn.run(dataclasses.replace(call, call_id=cid))
            ran = qh.executed(h.execlog)[-1]
            assert (ran["member"], ran["args"], ran["kwargs"]) == (method, tp.encode(args), tp.encode(kwargs)), \
                (cid, ran)  # the literal call ran, as the Runner made it
        got = [m["call_id"] for m in h.channel.mismatches]
    assert got == [f"q1-{k + 1}" for k in range(len(DISAGREE))], h.channel.mismatches


def test_the_rebuild_check_is_silent_when_the_typed_call_is_the_literal():
    ep = _Ep()
    agree = [(("pick", (JAR.id,), {"into": CAB.id}), SkillCall("pick_up", PickArgs(JAR), arm="left")),
             (("open_up", (CAB.id, 0.8), {}), SkillCall("open", OpenArgs(CAB, min_fraction=0.8), arm="left")),
             (("put_down", (JAR.id, TABLE.id), {"floor": None}),
              SkillCall("place", PlaceArgs(JAR, (Relation(Rel.ON, TABLE),)), arm="left"))]
    with qh.build_episode_connector(ep, qh.members_of(ep)) as h:
        for k, ((method, args, kwargs), call) in enumerate(agree):
            h.channel.put(f"q1-{k + 1}", {"method": method, "args": args, "kwargs": kwargs})
            h.conn.run(dataclasses.replace(call, call_id=f"q1-{k + 1}"))
        assert h.channel.mismatches == [] and h.violations() == []


# --------------------------------------------------------------------------------------------------- the detectors
def _suite_tests():
    spec = importlib.util.spec_from_file_location("q1_mutants_strategies_suite", SUITE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [(n, f) for n, f in vars(mod).items()
            if n.startswith("test_") and inspect.isfunction(f) and not inspect.signature(f).parameters]


def _session(mode, tests, tapes=None) -> tuple:
    """Every test run under a q1_pytest Session in ``mode``: {name: (passed or the error, the E1 log)} and the
    Session."""
    session = q1_pytest.Session(mode, tapes, inspect.unwrap(strategies.Runner.run))
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(strategies.Runner, "run", lambda self, ep: session.run(self, ep))
        for name, fn in tests:
            session.begin(name)
            try:
                fn()
                verdict = "passed"
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:  # noqa: BLE001 - pytest.fail / pytest.raises' Failed included: a failing
                verdict = f"{type(e).__name__}: {str(e)[:200]}"  # suite test is what a mutant is judged by
            out[name] = (verdict, session.runs)
    return out, session


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    tests = _suite_tests()
    assert len(tests) >= 90, f"the suite shrank: {len(tests)} zero-argument tests"
    direct, _ = _session("direct", tests)
    tapes = tmp_path_factory.mktemp("q1_fake_tapes")
    _, rec = _session("record", tests, tapes=str(tapes))
    paths = sorted(tapes.glob("*.json"))
    assert len(paths) == rec.invoked == rec.recorded, (len(paths), rec.invoked, rec.recorded)
    return {"tests": tests, "direct": direct, "tapes": [tp.Tape.load(p) for p in paths]}


def _e1(baseline) -> str:
    """'' when the connector session is the direct one test for test, else what differs first."""
    conn, session = _session("connector", baseline["tests"])
    for name, (verdict, runs) in baseline["direct"].items():
        got = conn[name]
        if got[0] != verdict:
            return f"{name}: {got[0]} (direct: {verdict})"
        if got[1] != runs:
            return f"{name}: the E1 logs differ"
    t = session.totals
    return f"mismatches {t['mismatches']}, violations {t['violations']}" if t["mismatches"] or t["violations"] else ""


def _g2(baseline) -> str:
    bad = [r for r in (qh.replay(t) for t in baseline["tapes"]) if not r["ok"]]
    return f"{len(bad)} of {len(baseline['tapes'])} tapes fail, first {bad[0]['instance']}" if bad else ""


def _unit() -> str:
    try:
        test_the_rebuild_check_sees_a_typed_object_the_literal_does_not_name()
        test_the_rebuild_check_is_silent_when_the_typed_call_is_the_literal()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:200]}"
    return ""


def test_the_unmutated_stack_passes_every_detector(baseline):
    """Without a mutant nothing fires: E1 identical for every test, G2 green on every fake tape, the unit tests
    pass. So a detector that fires under a mutant fires because of it."""
    assert all(v == "passed" for v, _ in baseline["direct"].values())
    assert sum(len(r) for _, r in baseline["direct"].values()) == len(baseline["tapes"]) >= 70
    assert (_e1(baseline), _g2(baseline), _unit()) == ("", "", "")


KILLS = {}


@pytest.mark.parametrize("mutant", MUTANTS, ids=[m.__name__ for m in MUTANTS])
def test_the_mutant_is_killed(mutant, baseline, monkeypatch):
    mutant(monkeypatch)
    row = {"e1": _e1(baseline), "g2": _g2(baseline), "unit": _unit()}
    KILLS[mutant.__name__] = row
    assert any(row.values()), f"{mutant.__name__} survived every detector"
    must = MUST.get(mutant.__name__)
    if must:
        assert row[must], f"{mutant.__name__} must be killed by {must}: {row}"


def test_the_kill_table(baseline):
    """Printed (pytest -s) and, with Q1_MUTANT_TABLE, written: which detector killed which mutant."""
    if len(KILLS) != len(MUTANTS):
        pytest.skip("run with the mutant tests")
    lines = [f"{'mutant':44} {'E1':6} {'G2':6} {'unit':6}"]
    for m in MUTANTS:
        row = KILLS[m.__name__]
        lines.append(f"{m.__name__:44} " + " ".join(f"{'KILL' if row[k] else '-':6}" for k in ("e1", "g2", "unit")))
    killed = sum(any(r.values()) for r in KILLS.values())
    lines.append(f"killed {killed} of {len(MUTANTS)}")
    print("\n" + "\n".join(lines))
    for m in MUTANTS:
        print(f"  {m.__name__}: {KILLS[m.__name__]}")
    if os.environ.get("Q1_MUTANT_TABLE"):
        Path(os.environ["Q1_MUTANT_TABLE"]).write_text(json.dumps({"kills": KILLS, "killed": killed}, indent=1))
    assert killed == len(MUTANTS)
