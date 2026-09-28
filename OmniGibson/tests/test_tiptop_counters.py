"""The ladder's counters (scripts/counters.py, WEEK4_PLAN 5.8) without Isaac Sim: the 8 historical jobs' pins, the
anchored weld regex, the connector sources on synthetic rows and on a week-3 skillbench dir, and the compare mode
(a flag, no flag, a pooled FAIL, a carried-forward task) on synthetic stage pairs."""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "omnigibson" / "tiptop" / "scripts" / "counters.py"
RUNS = Path(os.environ.get("B1K_RUNS", "/home/wding8/projects/BEHAVIOR-1K/runs"))


def _load():
    spec = importlib.util.spec_from_file_location("tiptop_counters", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tiptop_counters"] = mod  # dataclasses resolve the module by name
    spec.loader.exec_module(mod)
    return mod


counters = _load()
Counters, CallOutcome = counters.Counters, counters.CallOutcome

# The 8 historical jobs, in the plan's order: attach, wood, compost, batteries, rearrange, honey manip2c, honey manip2e,
# tidying. Pins: executed 1/5/3/5/7/1/1/6, welds 1/3/2/3/4/1/1/4, on-air 0, open attempts 0/0/0/0/0/2/1/0,
# teleports 2/6/4/8/10/5/3/6, steps 1811/4609/3299/5354/8646/2224/1359/6365.
JOBS = [
    ("manip2_20260925/jobs/attach_a_camera_to_a_tripod_i0_0925_072426", 1, 1, 0, 2, 1811, 0),
    ("manip2_20260925/jobs/bringing_in_wood_i0_0925_084518", 5, 3, 0, 6, 4609, 0),
    ("manip2_20260925/jobs/composting_waste_i0_0925_092501", 3, 2, 0, 4, 3299, 0),
    ("manip1g_20260924/jobs/dispose_of_batteries_i0_0925_032908", 5, 3, 0, 8, 5354, 0),
    ("manip2_20260925/jobs/rearrange_your_room_i0_0925_120034", 7, 4, 0, 10, 8646, 1),
    ("manip2_20260925/superseded/store_honey_i0_0925_062849", 1, 1, 2, 5, 2224, 1),
    ("manip2_20260925/jobs/store_honey_i0_0925_115934", 1, 1, 1, 3, 1359, 0),
    ("manip1g_20260924/jobs/tidying_bedroom_i0_0925_012858", 6, 4, 0, 6, 6365, 0),
]
WEEK3 = RUNS / "skill_arch_20260925" / "week3"


def _needs(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path} is not on this box")


# ------------------------------------------------------------------------------------------ the legacy pins
@pytest.mark.parametrize("job,executed,welds,opens,teleports,steps,rejections", JOBS,
                         ids=[j[0].rsplit("/", 1)[1][:24] for j in JOBS])
def test_the_eight_historical_jobs_pin_the_legacy_counters(job, executed, welds, opens, teleports, steps, rejections):
    _needs(RUNS / job)
    c = counters.extract(RUNS / job)
    assert c.runner == "legacy"
    assert (c.executed, c.welds, c.on_air, c.open_attempts, c.teleports, c.steps) == \
           (executed, welds, 0, opens, teleports, steps)
    # counted apart from the executed motions: every job ended inside a live round, two had a rejected motion
    assert (c.cut_off, c.rejections) == (1, rejections)
    assert (c.teleports_go_to, c.teleports_ep) == (0, teleports)  # the Runner navigates only through ep.*
    assert c.open_successes == min(opens, 1)  # honey manip2c: the first stance was rejected, the second opened
    assert c.notes == []  # the JSON, the round lines and the RESULT line agree


def test_the_pins_cover_the_planned_order_and_values():
    assert [j[1] for j in JOBS] == [1, 5, 3, 5, 7, 1, 1, 6]
    assert [j[2] for j in JOBS] == [1, 3, 2, 3, 4, 1, 1, 4]
    assert [j[3] for j in JOBS] == [0, 0, 0, 0, 0, 2, 1, 0]
    assert [j[4] for j in JOBS] == [2, 6, 4, 8, 10, 5, 3, 6]
    assert [j[5] for j in JOBS] == [1811, 4609, 3299, 5354, 8646, 2224, 1359, 6365]


LOG = """\
2026-09-25 07:29:50,000 b1k.bridge.executor INFO: [2] gripper close: fingers [0.05, 0.05] -> [0.044, 0.022], is_grasping=1
2026-09-25 07:29:56,012 omnigibson.tiptop INFO: round 1 holding(digital_camera.n.01_1) [left]: executed (138.6s)
2026-09-25 07:31:00,000 omnigibson.tiptop INFO: executing plan: 4 trajectories / 313 waypoints / 6.3s planned, gripper events ['open'], gripper closed at the start
2026-09-25 07:31:10,000 b1k.bridge.executor INFO: [2] gripper close: fingers [0.05, 0.05] -> [0.0, 0.0], is_grasping=-1
2026-09-25 07:32:00,000 omnigibson.tiptop INFO: round 3 attached(digital_camera.n.01_1, camera_tripod.n.01_1) [left]: TiptopPlanningError: planning failed: x (1.0s)
2026-09-25 07:32:01,000 omnigibson.tiptop INFO: RESULT instance 301: q_score 1.0 success True steps 1811/5867 (success); teleports 2; 312.2s
"""


def test_the_weld_regex_is_anchored_on_is_grasping():
    assert counters.closes_from_log(LOG) == [1, -1]
    # the mutant: a bare 'gripper close' also matches 'gripper closed at the start' and counts a third close
    unanchored = re.compile(r"gripper close")
    assert len(counters.closes_from_log(LOG, unanchored)) == 3
    assert counters.rounds_from_log(LOG) == [
        {"round": 1, "atoms": "holding(digital_camera.n.01_1)", "arm": "left", "verdict": "executed",
         "seconds": 138.6},
        {"round": 3, "atoms": "attached(digital_camera.n.01_1, camera_tripod.n.01_1)", "arm": "left",
         "verdict": "TiptopPlanningError: planning failed: x", "seconds": 1.0}]
    assert counters.result_line(LOG) == {"instance": 301, "q_score": "1.0", "success": True, "steps": 1811,
                                         "max_steps": 5867, "reason": "success", "teleports": 2, "wall_s": 312.2}


def test_a_legacy_job_dir_from_its_three_sources(tmp_path):
    """The JSON is the authority for rounds (the cut-off round is only there), the log for closes, and a
    disagreement between them is a note, never silent."""
    rounds = [{"stand_for": ["a"], "step": 10},
              {"round": 1, "atoms": [{"predicate": "holding", "args": ["a"]}], "arm": "left", "env_steps": 200},
              {"open": "cab", "opened": False, "why": "rejected", "step": 20},
              {"open": "cab", "opened": True, "step": 30},
              {"close": "cab", "reached": True, "step": 40},
              {"round": 3, "atoms": [], "arm": "left", "env_steps": 0, "error": "motion validation rejected"},
              {"round": 4, "atoms": [], "arm": "left", "error": "GoalNotVisible: x"},
              {"round": 5, "atoms": [], "arm": "left", "error": "episode over"}]
    goal = {"satisfied": ["ontop(a, t)", "ontop(b, t)", "inside(a, box)"], "new": 2, "total": 4}
    _job(tmp_path, rounds=rounds, goal=goal, steps=999, teleports=3, log=LOG)
    c = counters.extract(tmp_path)
    assert c.runner == "legacy" and c.task == "demo"
    assert (c.executed, c.cut_off, c.rejections) == (1, 1, 1)
    assert (c.open_attempts, c.open_successes, c.close_attempts, c.close_successes) == (2, 1, 1, 1)
    assert (c.welds, c.on_air) == (1, 1) and c.on_air_causes == ["unknown: legacy log names no owner"]
    assert (c.teleports, c.teleports_go_to, c.teleports_ep) == (3, 0, 3)
    assert (c.steps, c.delivered_atoms, c.delivered_new, c.goal_total, c.delivered_items) == (999, 3, 2, 4, 2)
    assert c.notes == ["RESULT line steps 1811 != JSON steps 999"]


# ------------------------------------------------------------------------------------------ connector sources
def _job(d: Path, *, rounds=(), goal=None, steps=0, teleports=None, log="", connector=None, task="demo",
         calls=None, ledger=None, gripper=None, tape=None):
    (d / "episode" / "json").mkdir(parents=True, exist_ok=True)
    bench = {"reason": "success", "rounds": list(rounds), "goal": goal or {"satisfied": [], "new": 0, "total": 0}}
    if teleports is not None:
        bench["teleports"] = teleports
    data = {"task": task, "instance_id": 301, "steps": steps, "bench": bench}
    if connector is not None:
        data["connector"] = connector
    (d / "episode" / "json" / f"{task}_301_0.json").write_text(json.dumps(data))
    if log:
        (d / "sim.log").write_text(log)
    for name, rows in (("skill_calls", calls), ("ledger", ledger), ("gripper", gripper), ("runner_tape", tape)):
        if rows is not None:
            (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _row(cid, skill, backend, status, steps, *, scorer=None, effects=(), evidence=None, sim_clock=False):
    return {"__type__": "SkillResult", "call_id": cid, "skill": skill, "backend": backend, "status": status,
            "code": None, "phase": "", "detail": "", "effects": [{"pred": p, "args": list(a), "value": True}
                                                                 for p, a in effects],
            "verdicts": {"scorer": scorer, "perception": None}, "primary": "scorer", "steps": steps,
            "requires_sim_clock": sim_clock, "evidence": evidence or {}}


CALLS = [
    _row("c1", "pick_up", "tiptop", "succeeded", 140, scorer=True, effects=(("holding", ("a", "left")),)),
    _row("c2", "place", "tiptop", "succeeded", 120, scorer=True, effects=(("ontop", ("a", "t")),),
         evidence={"resamples": 1}),  # one motion plus one resample
    _row("c3", "place", "tiptop", "infeasible", 0, scorer=False),  # nothing moved: not an executed motion
    _row("c4", "place", "legacy", "succeeded", 0, scorer=True, sim_clock=True, effects=(("inside", ("b", "box")),),
         evidence={"legacy_ok": True, "records": [
             {"round": 3, "atoms": [], "arm": "left", "env_steps": 0, "error": "motion validation rejected"},
             {"round": 4, "atoms": [], "arm": "left", "env_steps": 210},
             {"stand_for": ["box"], "step": 50}]}),
    _row("c5", "open", "legacy", "succeeded", 0, sim_clock=True, evidence={"legacy_ok": True, "records": [
        {"open": "cab", "opened": False, "why": "rejected", "step": 60}, {"open": "cab", "opened": True, "step": 70}]}),
    _row("c6", "open", "tiptop", "succeeded", 90, scorer=True),
    _row("c7", "close", "tiptop", "failed", 40, scorer=False),
    _row("c8", "place", "tiptop", "succeeded", 100, scorer=False, effects=(("ontop", ("c", "t")),)),  # open loop
]
LEDGER = [{"owner": "rt", "steps": 490, "env_step_calls": 490, "place_robot": 0, "capture": 0, "look_at": 0},
          {"owner": "go_to", "steps": 0, "env_step_calls": 0, "place_robot": 2, "capture": 0, "look_at": 0},
          {"owner": "observe", "steps": 60, "env_step_calls": 60, "place_robot": 0, "capture": 3, "look_at": 3},
          {"owner": "ep.achieve", "steps": 210, "env_step_calls": 210, "place_robot": 1, "capture": 1, "look_at": 1},
          {"owner": "ep.open_up", "steps": 300, "env_step_calls": 300, "place_robot": 2, "capture": 0, "look_at": 0}]
GRIPPER = [{"step": 100, "arm": "left", "event": "close", "is_grasping": 1, "owner": "rt", "call_id": "c1"},
           {"step": 250, "arm": "left", "event": "open", "is_grasping": 0, "owner": "rt", "call_id": "c2"},
           {"step": 400, "arm": "left", "event": "close", "is_grasping": 1, "owner": "ep.achieve", "call_id": "c4"},
           {"step": 700, "arm": "right", "event": "close", "is_grasping": -1, "owner": "rt", "call_id": "c7"}]
BLOCK = {"step": 490, "idle_steps": 0, "charged": {"skill": 490, "observe": 60},
         "live_at_end": {"call_id": "c8", "steps": 100}}


def test_connector_rows_on_synthetic_sources(tmp_path):
    goal = {"satisfied": ["ontop(a, t)", "inside(b, box)"], "new": 2, "total": 3}
    _job(tmp_path, goal=goal, steps=1060, teleports=5, connector=BLOCK, calls=CALLS, ledger=LEDGER, gripper=GRIPPER)
    c = counters.extract(tmp_path)
    assert c.runner == "connector"
    # native runs with steps > 0: c1, c2 (+1 resample), c6, c7, c8 = 6; the legacy row's executed round = 1
    assert c.executed == 7
    assert c.rejections == 1  # the legacy row's rejected round
    assert c.cut_off == 1  # EpisodeOver ended c8, still live at the end
    assert (c.open_attempts, c.open_successes) == (3, 2)  # two legacy open records, one native open
    assert (c.close_attempts, c.close_successes) == (1, 0)
    assert (c.welds, c.on_air) == (2, 1)
    assert c.on_air_causes == [{"step": 700, "arm": "right", "owner": "rt", "call_id": "c7"}]
    assert (c.teleports, c.teleports_go_to, c.teleports_ep) == (5, 2, 3)
    assert (c.steps, c.delivered_atoms, c.delivered_items) == (1060, 2, 2)
    assert (c.native_calls, c.native_by) == (6, {"pick_up@tiptop": 1, "place@tiptop": 3, "open@tiptop": 1,
                                                 "close@tiptop": 1})
    assert c.notes == []
    by = {x.call_id: x for x in c.calls}
    assert len(c.calls) == 8
    assert (by["c2"].returned, by["c2"].scorer, by["c2"].ok, by["c2"].qual) == (True, True, True, "on")
    assert (by["c3"].returned, by["c3"].ok, by["c3"].qual) == (False, False, None)  # nothing to qualify it
    assert (by["c4"].returned, by["c4"].ok, by["c4"].qual, by["c4"].backend) == (True, True, "in", "legacy")
    assert (by["c8"].returned, by["c8"].scorer, by["c8"].ok) == (True, False, False)  # ran, the scorer disagrees


def test_the_runner_tape_qualifies_and_overrides_the_return(tmp_path):
    tape = [{"call_id": "c3", "skill": "place", "qual": "on", "returned": False},
            {"call_id": "c8", "skill": "place", "qual": "on", "returned": False}]  # the Runner saw False
    _job(tmp_path, calls=CALLS, tape=tape)
    by = {x.call_id: x for x in counters.extract(tmp_path).calls}
    assert by["c3"].qual == "on" and by["c8"].returned is False and by["c8"].ok is False


def test_the_connector_block_and_the_log_stand_in_for_missing_jsonl(tmp_path):
    block = dict(BLOCK, ledger={r["owner"]: r for r in LEDGER})
    block.pop("live_at_end")
    _job(tmp_path, steps=1060, teleports=4, connector=block, calls=CALLS, log=LOG)
    c = counters.extract(tmp_path)
    assert c.runner == "connector" and c.cut_off == 0
    assert (c.teleports, c.teleports_go_to, c.teleports_ep) == (5, 2, 3)
    assert (c.welds, c.on_air) == (1, 1) and c.on_air_causes == ["unknown: legacy log names no owner"]
    assert "bench.teleports 4 != ledger place_robot 5" in c.notes
    assert "RESULT line steps 1811 != JSON steps 1060" in c.notes


def test_a_week3_skillbench_dir_native_and_legacy():
    native, legacy = WEEK3 / "A1" / "pick_chopping_native", WEEK3 / "L3" / "gate4" / "honey_legacy"
    _needs(native)
    _needs(legacy)
    n = counters.extract(native)
    assert n.runner == "connector"
    assert (n.native_calls, n.native_by, n.executed) == (5, {"pick_up@tiptop": 5}, 5)  # 5 grasp_missed, all moved
    assert [x.ok for x in n.calls] == [False] * 5 and {x.trial for x in n.calls} == {
        f"pick_up.chopping_wood.e8804.f150_t{i}" for i in range(5)}
    assert (n.welds, n.on_air, n.steps, n.delivered_atoms) == (0, 0, 0, 0)  # no gripper.jsonl, no RESULT line
    assert "no step count (no result JSON, no RESULT line)" in n.notes
    lg = counters.extract(legacy)
    assert (lg.native_calls, lg.executed, lg.rejections) == (0, 5, 0)  # five single-round legacy places
    assert [x.ok for x in lg.calls] == [True] * 5 and {x.backend for x in lg.calls} == {"legacy"}
    assert {x.qual for x in lg.calls} == {"in"}


# ------------------------------------------------------------------------------------------ compare mode
def _c(job, executed, delivered=2, calls=(), **kw):
    c = Counters(job=job, task="t", runner="connector", executed=executed, delivered_atoms=delivered,
                 delivered_items=delivered, welds=delivered, teleports=4, teleports_go_to=4, steps=1000, **kw)
    c.calls = [CallOutcome(f"c{i}", "place", "on", "tiptop", ok, ok) for i, ok in enumerate(calls)]
    return c


def test_fisher_one_sided_matches_scipy_and_the_mde_is_printed():
    scipy = pytest.importorskip("scipy.stats")
    for cn in range(1, 7):
        for tn in range(1, 7):
            for cs in range(cn + 1):
                for ts in range(tn + 1):
                    p = counters.fisher_one_sided(cs, cn, ts, tn)
                    q = scipy.fisher_exact([[ts, tn - ts], [cs, cn - cs]], alternative="less")[1]
                    assert abs(p - q) < 1e-12, (cs, cn, ts, tn)
    # 3 vs 3 with C at 3/3: only 0/3 in T reaches p < 0.10 (p = 0.05), so the MDE is a drop of 1.0
    assert counters.fisher_one_sided(3, 3, 0, 3) == pytest.approx(0.05)
    assert counters.minimum_detectable_effect(3, 3, 3, 0.10) == pytest.approx(1.0)
    # 9 vs 9 with C at 9/9: 5/9 in T reaches it, 6/9 does not (p = 0.103)
    assert counters.minimum_detectable_effect(9, 9, 9, 0.10) == pytest.approx(4 / 9)
    assert counters.minimum_detectable_effect(1, 3, 3, 0.10) is None  # even 0/3 cannot be called against 1/3


def test_a_flag_when_the_treatments_minimum_exceeds_the_controls_maximum():
    C = [_c("c0", 4), _c("c1", 5), _c("c2", 4)]
    T = [_c("t0", 6), _c("t1", 7), _c("t2", 6)]  # 3.0 per atom against at most 2.5
    cmp = counters.compare({"wood": (C, T)}, "place.on")
    flagged = {(f.task, f.counter) for f in cmp.flagged}
    assert flagged == {("wood", "executed")}
    f = cmp.flagged[0]
    assert f.normalised and f.control == [2.0, 2.5, 2.0] and f.treatment == [3.0, 3.5, 3.0]
    assert cmp.n_tests == 11 and cmp.n_informative == 1, "only executed differs between the runs"
    assert cmp.expected_false_flags == pytest.approx(0.05), "0.05 per test that could flag, not per test run"
    assert cmp.raw_tasks == [] and cmp.pooled.fail is False and cmp.pooled.judged is False, "no call: unjudged"


def test_no_flag_when_the_arms_overlap():
    C = [_c("c0", 4), _c("c1", 6), _c("c2", 4)]
    T = [_c("t0", 6), _c("t1", 5), _c("t2", 7)]  # min T 2.5 == max C 3.0? no: 2.5 < 3.0, they overlap
    cmp = counters.compare({"wood": (C, T)}, "place.on")
    assert cmp.flagged == []


def test_counters_compare_raw_when_a_run_delivered_no_atom():
    C = [_c("c0", 4, delivered=2), _c("c1", 4, delivered=2), _c("c2", 4, delivered=2)]
    T = [_c("t0", 5, delivered=0), _c("t1", 5, delivered=2), _c("t2", 5, delivered=2)]
    cmp = counters.compare({"wood": (C, T)}, "place.on")
    assert cmp.raw_tasks == ["wood"]
    f = {fl.counter: fl for fl in cmp.flags}["executed"]
    assert not f.normalised and f.flagged and f.control == [4, 4, 4] and f.treatment == [5, 5, 5]


def test_a_pooled_fail_needs_the_drop_and_the_p_value():
    C = [_c(f"c{i}", 4, calls=(True, True, True)) for i in range(3)]  # 9/9
    T = [_c(f"t{i}", 4, calls=(True, False, False)) for i in range(3)]  # 3/9
    cmp = counters.compare({"wood": (C, T)}, "place.on")
    p = cmp.pooled
    assert (p.c_n, p.c_succ, p.t_n, p.t_succ) == (9, 9, 9, 3)
    assert p.c_rate == 1.0 and p.t_rate == pytest.approx(1 / 3)
    assert p.p < 0.01 and p.fail is True and p.mde == pytest.approx(4 / 9)
    # a drop of 0.5 at n = 2 per arm: p = 0.5, so no FAIL, and the MDE says why
    C2, T2 = [_c("c0", 4, calls=(True, True))], [_c("t0", 4, calls=(True, False))]
    q = counters.compare({"wood": (C2, T2)}, "place.on").pooled
    assert q.c_rate - q.t_rate == pytest.approx(0.5) and q.p == pytest.approx(0.5) and q.fail is False
    assert q.mde is None  # even 0/2 against 2/2 gives p = 1/6: no drop is callable at this n
    # a drop under 0.25 never fails, whatever p
    C3 = [_c(f"c{i}", 4, calls=(True,) * 10) for i in range(3)]
    T3 = [_c(f"t{i}", 4, calls=(True,) * 8 + (False,) * 2) for i in range(3)]
    r = counters.compare({"wood": (C3, T3)}, "place.on").pooled
    assert r.c_rate - r.t_rate == pytest.approx(0.2) and r.p < 0.05 and r.fail is False


def test_the_pooled_test_counts_only_the_switched_skill_and_reports_unqualified_rows():
    C = [_c("c0", 4, calls=(True, True))]
    T = [_c("t0", 4, calls=(True,))]
    T[0].calls += [CallOutcome("x1", "place", "in", "tiptop", False, False),  # another qualifier
                   CallOutcome("x2", "pick_up", None, "tiptop", False, False),  # another skill
                   CallOutcome("x3", "place", None, "tiptop", False, False),  # unqualified: reported, left out
                   CallOutcome("x4", "place", "on", "legacy", False, False)]  # another backend when one is asked
    p = counters.compare({"wood": (C, T)}, "place.on", backend="tiptop").pooled
    assert (p.c_n, p.t_n, p.t_succ, p.unqualified) == (2, 1, 1, 1)
    assert p.judged is False and "no qualifier" in p.why_unjudged, "an unqualified row of place.on: unjudged"
    p2 = counters.compare({"wood": (C, T)}, "place").pooled  # unqualified: every place, any backend
    assert (p2.t_n, p2.t_succ, p2.unqualified) == (4, 1, 0)


def test_a_carried_forward_task_must_show_zero_native_calls_or_report_them():
    clean = Counters(job="attach_native", task="attach", runner="connector", native_calls=0)
    dirty = Counters(job="compost_native", task="compost", runner="connector", native_calls=2,
                     native_by={"press@tiptop": 2})
    cmp = counters.compare({}, "place.on", carried=[("attach", clean), ("compost", dirty)])
    assert cmp.carried == [{"task": "attach", "job": "attach_native", "native_calls": 0, "native_by": {}},
                           {"task": "compost", "job": "compost_native", "native_calls": 2,
                            "native_by": {"press@tiptop": 2}}]
    text = counters.format_comparison(cmp)
    assert "carried attach attach_native: 0 native calls" in text
    assert "carried compost compost_native: 2 native calls {'press@tiptop': 2} (REPORT)" in text
    assert cmp.n_tests == 0 and cmp.expected_false_flags == 0 and cmp.pooled.judged is False


def test_on_air_closes_are_listed_with_their_causes():
    C = [_c("c0", 4)]
    T = [_c("t0", 4, on_air=1, on_air_causes=[{"step": 7, "arm": "left", "owner": "rt", "call_id": "c9"}])]
    cmp = counters.compare({"wood": (C, T)}, "place.on")
    assert cmp.on_air == [{"task": "wood", "arm": "T", "job": "t0",
                           "causes": [{"step": 7, "arm": "left", "owner": "rt", "call_id": "c9"}]}]
    assert "needs a cause" in counters.format_comparison(cmp)


def test_the_command_line_extracts_and_compares(tmp_path, capsys):
    goal = {"satisfied": ["ontop(a, t)", "inside(b, box)"], "new": 2, "total": 3}
    c0, t0 = tmp_path / "c0", tmp_path / "t0"
    _job(c0, goal=goal, steps=1060, teleports=5, connector=BLOCK, calls=CALLS, ledger=LEDGER, gripper=GRIPPER)
    _job(t0, goal=goal, steps=2000, teleports=9, connector=BLOCK, calls=CALLS, ledger=LEDGER, gripper=GRIPPER)
    assert counters.main(["extract", str(c0), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["executed"] == 7 and rows[0]["open_rate"] == pytest.approx(2 / 3) and len(rows[0]["calls"]) == 8
    assert counters.main(["compare", "--skill", "place.on", "--backend", "tiptop",
                          "--pair", "demo", str(c0), str(t0), "--carried", "demo2", str(c0)]) == 0
    out = capsys.readouterr().out
    # c2 ok, c8 not; c3 carries no qualifier (nothing moved, no tape) and is left out; c4 is legacy
    assert "pooled per-call: C 1/2 (0.5), T 1/2 (0.5)" in out and "2 unqualified rows of the skill left out" in out
    assert "UNJUDGED" in out, "c3 is unqualified: the pooled test says nothing, never pass"
    assert "FLAG demo steps: C ['530'] T ['1000']" in out  # per delivered atom
    assert "on-air demo C c0: needs a cause" in out and "carried demo2 c0: 6 native calls" in out


# ------------------------------------------------------------------------------------------------- the fix pass
def test_a_rejection_is_named_by_its_text_and_a_round_that_moved_then_failed_executed(tmp_path):
    """Motion validation rejects a segment just before running it, often after earlier ones ran (bringing_in_wood
    R3: 40 steps). A round that moved and then failed some other way executed, as a native FAILED run that stepped
    does. A round that never planned is a planning failure; one in no category is a note."""
    rounds = [{"round": 1, "atoms": [], "arm": "left", "env_steps": 40, "error": "motion validation rejected Place(p)"},
              {"round": 2, "atoms": [], "arm": "left", "env_steps": 0, "error": "motion validation rejected Pick(p)"},
              {"round": 3, "atoms": [], "arm": "left", "env_steps": 96, "error": "failed to track the checked path"},
              {"round": 4, "atoms": [], "arm": "left", "error": "TiptopPlanningError: No satisfying particles"},
              {"round": 5, "atoms": [], "arm": "left", "env_steps": 120},
              {"round": 6, "atoms": [], "arm": "left"}]
    _job(tmp_path, rounds=rounds, steps=100)
    c = counters.extract(tmp_path)
    assert (c.rejections, c.executed, c.planning_failures, c.cut_off) == (2, 2, 1, 0)
    assert "round 6: no error and no env_steps None: counted in no category" in c.notes


def test_only_the_executors_closes_are_welds_or_on_air_and_both_are_cross_checked(tmp_path):
    gripper = [{"step": 125, "arm": "left", "event": "close", "is_grasping": -1, "owner": "ep.open_up", "call_id": None,
                "via": "sim"},  # store_honey's closed-fist drawer pull
               {"step": 300, "arm": "left", "event": "close", "is_grasping": 1, "owner": "ep.pick", "call_id": None,
                "via": "executor"},
               {"step": 400, "arm": "left", "event": "close", "is_grasping": -1, "owner": "ep.pick", "call_id": None,
                "via": "executor.start"}]
    log = "gripper close: Pick(jar) is_grasping=1\n"
    _job(tmp_path, gripper=gripper, log=log, steps=500)
    c = counters.extract(tmp_path)
    assert (c.welds, c.on_air, c.closes_other) == (1, 0, 2) and not any("sim.log shows" in n for n in c.notes)
    _job(tmp_path, gripper=gripper, log=log + "gripper close: Pick(jar) is_grasping=-1\n", steps=500)
    assert "sim.log shows 1 on-air closes, gripper.jsonl 0" in counters.extract(tmp_path).notes


def test_a_native_call_episode_over_cut_off_is_a_native_call(tmp_path):
    block = {"step": 90, "idle_steps": 0, "charged": {}, "live_at_end": {"call_id": "q1-9", "steps": 40}}
    rows = [_row("q1-1", "pick_up", "legacy", "succeeded", 0, sim_clock=True, evidence={"legacy_ok": True})]
    tape = [{"call_id": "q1-9", "skill": "place", "qual": "on", "returned": None}]
    _job(tmp_path, connector=block, calls=rows, tape=tape, steps=90)
    c = counters.extract(tmp_path)
    assert (c.cut_off, c.native_calls, c.native_by) == (1, 1, {"place@live_at_end": 1})


def test_the_bench_layout_itself_is_read(tmp_path):
    """bench.py writes the connector block at bench.connector and the rows in episode/<task>_<inst>_<rollout>/:
    a connector run read from that layout is a connector run with its native calls (the schema the writer uses)."""
    ep = tmp_path / "episode"
    (ep / "json").mkdir(parents=True)
    block = {"step": 120, "idle_steps": 0, "charged": {"skill": 120}, "live_at_end": None}
    data = {"task": "demo", "instance_id": 301, "steps": 900,
            "bench": {"reason": "success", "rounds": [], "goal": {"satisfied": ["ontop(a, t)"], "new": 1, "total": 1},
                      "teleports": 0, "connector": block}}
    (ep / "json" / "demo_301_0.json").write_text(json.dumps(data))
    inst = ep / "demo_301_0"
    inst.mkdir()
    rows = [_row("q1-1", "place", "tiptop", "succeeded", 120, scorer=True, effects=(("ontop", ("a", "t")),))]
    (inst / "skill_calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = counters.extract(tmp_path)
    assert c.runner == "connector" and (c.native_calls, c.native_by, c.executed) == (1, {"place@tiptop": 1}, 1)


def _rates(job, close_attempts, close_successes, items=2, welds=2):
    return Counters(job=job, task="t", runner="connector", delivered_atoms=2, delivered_items=items, welds=welds,
                    close_attempts=close_attempts, close_successes=close_successes, steps=1000)


def test_a_lower_rate_flags_and_a_higher_one_does_not():
    C = [_rates(f"c{i}", 2, 2) for i in range(3)]  # close_rate 1.0
    low = counters.compare({"honey": (C, [_rates(f"t{i}", 2, 1) for i in range(3)])}, "close.prismatic")
    assert {f.counter for f in low.flagged} == {"close_rate"}
    high = counters.compare({"honey": ([_rates(f"c{i}", 2, 1) for i in range(3)], C)}, "close.prismatic")
    assert high.flagged == [], "a higher success rate is never worse"


def test_welds_are_normalised_by_the_delivered_items():
    C = [_rates(f"c{i}", 0, 0, items=4, welds=4) for i in range(3)]  # 1.0 per item, 2.0 per atom
    T = [_rates(f"t{i}", 0, 0, items=4, welds=6) for i in range(3)]  # 1.5 per item
    f = {x.counter: x for x in counters.compare({"t": (C, T)}, "place.on").flags}["welds"]
    assert f.control == [1.0] * 3 and f.treatment == [1.5] * 3 and f.flagged


def test_a_failed_native_open_is_an_attempt_and_not_a_success(tmp_path):
    rows = [_row("q1-1", "open", "tiptop", "failed", 60, scorer=False), _row("q1-2", "open", "tiptop", "succeeded", 50)]
    _job(tmp_path, calls=rows, steps=110)
    c = counters.extract(tmp_path)
    assert (c.open_attempts, c.open_successes) == (2, 1)


def test_the_backend_filter_is_the_treatments_alone():
    C = [Counters(job=f"c{i}", task="t", delivered_atoms=1) for i in range(3)]
    T = [Counters(job=f"t{i}", task="t", delivered_atoms=1) for i in range(3)]
    for x in C:  # the control runs the line on legacy
        x.calls = [CallOutcome(f"c{k}", "place", "on", "legacy", True, True) for k in range(3)]
    for x in T:
        x.calls = [CallOutcome(f"c{k}", "place", "on", "tiptop", False, False) for k in range(3)]
    p = counters.compare({"t": (C, T)}, "place.on", backend="tiptop").pooled
    assert (p.c_succ, p.c_n, p.t_succ, p.t_n) == (9, 9, 0, 9) and p.fail is True and p.judged


def test_the_mde_is_the_smallest_drop_the_rule_fails_on():
    """The rule fails at a drop >= delta with p < alpha: at C 22/26 against 23 calls, a drop of 0.237 has p < 0.10
    but is under 0.25, so the MDE is the first k/23 that clears both."""
    m = counters.minimum_detectable_effect(22, 26, 23, 0.10, 0.25)
    assert m >= 0.25 and counters.fisher_one_sided(22, 26, round((22 / 26 - m) * 23), 23) < 0.10
    assert counters.minimum_detectable_effect(22, 26, 23, 0.10) < 0.25, "the p threshold alone"
    assert counters.minimum_detectable_effect(21, 21, 21, 0.10, 0.25) >= 0.25
