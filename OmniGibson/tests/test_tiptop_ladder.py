"""W4-I: the HandRefresh proxy in the episode host and the switch ladder's pieces (scripts/ladder.py), without a
simulator. The host fakes come from test_tiptop_episode_host (the stepping FakeSim, the self-stepping FakeEpisode,
FakeProviders and the build helper): a clean PARITY episode never refreshes (every result is legacy), a native hand
result refreshes once under the ledger owner ``refresh`` at 0 steps, and the block reports the count and the popped
labels. The ladder's typing and routing, the branch write, the frame alignment (by the rounds and by the owners),
the E-rep cross-check, the re-stamped tape per replicate, the arm command lines and the gate's verdict pieces are
each pinned on synthetic tapes and rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from b1k.bridge.protocol import packb, unpackb
from b1k.connector.skills import SkillResult, Status
from b1k.planner.pseudo import tape as tp
from omnigibson.tiptop.host import episode_host
from omnigibson.tiptop.host.instruments import UNOWNED, StepLedger
from omnigibson.tiptop.host.wstape import Frame, TapeDir, seed_for

# referenced at use, not imported: without the W4-I change to episode_host.py the HandRefresh tests alone fail
HandRefresh = lambda *a, **k: episode_host.HandRefresh(*a, **k)  # noqa: E731
REFRESH_OWNER = "refresh"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "omnigibson" / "tiptop" / "scripts"))
import counters  # noqa: E402
import ladder  # noqa: E402
import test_tiptop_episode_host as th  # noqa: E402  (the host fakes, beside this file)


def result(skill: str, backend: str, steps: int, status=Status.SUCCEEDED, cid="q1-9", sim_clock=False) -> SkillResult:
    return SkillResult(cid, skill, backend, status, None, "", "", (), {"scorer": True}, "scorer", steps=steps,
                       requires_sim_clock=sim_clock)


class StubConnector:
    """A connector whose run/wait answer the results handed in; everything else is a recorded attribute."""

    def __init__(self, *results):
        self.results, self.ran, self.waited = list(results), [], []

    def run(self, call):
        self.ran.append(call)
        return self.results.pop(0)

    def wait(self, *handles):
        self.waited.append(handles)
        return [self.results.pop(0) for _ in handles]

    def task(self):
        return "task-answer"


class Refresher:
    """providers.refresh_hands as the host binds it: records the ledger's owner at each call; can raise or step."""

    def __init__(self, ledger, popped=("jar_of_honey_1",), raise_=None, sim=None):
        self.ledger, self.popped, self.raise_, self.sim, self.calls = ledger, list(popped), raise_, sim, []

    def __call__(self):
        self.calls.append(self.ledger.current)
        if self.raise_ is not None:
            raise self.raise_
        if self.sim is not None:
            self.sim.step(1)
        return list(self.popped)


# ------------------------------------------------------------------------------------------------- HandRefresh
def test_wants_follows_a_native_hand_result_that_moved_and_nothing_else():
    assert episode_host.HandRefresh.wants(result("pick_up", "tiptop@abc/pick-1", 12))
    assert episode_host.HandRefresh.wants(result("place", "tiptop@abc/place-1", 40, Status.FAILED))
    assert episode_host.HandRefresh.wants(result("release", "scripted", 45)) and episode_host.HandRefresh.wants(result("hold", "scripted", 5))
    assert episode_host.HandRefresh.wants(result("press", "tiptop@abc/press-1", 90))
    assert not episode_host.HandRefresh.wants(result("pick_up", "legacy", 900, sim_clock=True)), "a legacy result, whatever it stepped"
    assert not episode_host.HandRefresh.wants(result("place", "tiptop@abc/place-1", 0, Status.INFEASIBLE)), "nothing moved"
    assert not episode_host.HandRefresh.wants(result("wait", "scripted", 30)) and not episode_host.HandRefresh.wants(result("open", "tiptop@x", 50))
    assert not episode_host.HandRefresh.wants(result("place", "", 0, Status.INFEASIBLE)), "UNSUPPORTED: backend '' at 0 steps"
    assert set(episode_host.HAND_SKILLS) == {"pick_up", "place", "release", "hold", "press"}


def test_a_native_hand_result_refreshes_once_under_the_refresh_owner_at_zero_steps():
    sim = th.FakeSim()
    ledger = StepLedger(sim)
    refresher = Refresher(ledger)
    conn = StubConnector(result("pick_up", "tiptop@abc/pick-1", 12, cid="q1-2"))
    hr = HandRefresh(conn, refresher, ledger)
    r = hr.run("call")
    assert r.call_id == "q1-2" and conn.ran == ["call"]
    assert refresher.calls == [REFRESH_OWNER], "the refresh ran under the ledger owner 'refresh'"
    assert ledger.rows[REFRESH_OWNER].steps == 0 and ledger.rows[REFRESH_OWNER].env_step_calls == 0
    assert ledger.current == UNOWNED, "the owner scope is closed again"
    s = hr.summary()
    assert s == {"count": 1, "popped": ["jar_of_honey_1"], "popped_total": 1, "inside_owner": [], "errors": [], "ok": True,
                 "calls": [{"call_id": "q1-2", "skill": "pick_up", "backend": "tiptop@abc/pick-1", "steps": 12,
                            "popped": ["jar_of_honey_1"], "owner_before": UNOWNED}]}
    assert hr.task() == "task-answer", "every other op passes through"


def test_legacy_results_and_zero_step_results_never_refresh_through_run_or_wait():
    sim = th.FakeSim()
    ledger = StepLedger(sim)
    refresher = Refresher(ledger)
    conn = StubConnector(result("pick_up", "legacy", 900, sim_clock=True), result("place", "legacy", 500, sim_clock=True),
                         result("place", "tiptop@abc/place-1", 0, Status.INFEASIBLE), result("wait", "scripted", 30))
    hr = HandRefresh(conn, refresher, ledger)
    hr.run("pick")
    hr.wait("h1", "h2")
    hr.run("wait")
    assert refresher.calls == [] and hr.summary()["count"] == 0 and hr.summary()["ok"]
    assert REFRESH_OWNER not in ledger.rows, "no refresh owner was ever opened"
    assert conn.waited == [("h1", "h2")]


def test_wait_results_are_looked_at_too_and_each_native_hand_result_counts():
    sim = th.FakeSim()
    ledger = StepLedger(sim)
    refresher = Refresher(ledger, popped=())
    conn = StubConnector(result("place", "tiptop@abc/place-1", 40, cid="q1-3"), result("release", "scripted", 45, cid="q1-4"))
    hr = HandRefresh(conn, refresher, ledger)
    hr.wait("a", "b")
    assert refresher.calls == [REFRESH_OWNER, REFRESH_OWNER]
    assert [c["call_id"] for c in hr.summary()["calls"]] == ["q1-3", "q1-4"] and hr.summary()["popped_total"] == 0


def test_a_refresh_that_raises_or_runs_inside_an_owner_is_recorded_and_fails_the_summary():
    sim = th.FakeSim()
    ledger = StepLedger(sim)
    boom = Refresher(ledger, raise_=RuntimeError("refresh_hands stepped the sim: 3 -> 4"))
    hr = HandRefresh(StubConnector(result("press", "tiptop@abc/press-1", 90, cid="q1-5")), boom, ledger)
    hr.run("press")
    s = hr.summary()
    assert s["count"] == 1 and s["errors"] == ["q1-5: RuntimeError: refresh_hands stepped the sim: 3 -> 4"] and not s["ok"]
    # inside a skill's owner scope (never on the bench: the run has returned) the breach is named
    ok = Refresher(ledger)
    hr2 = HandRefresh(StubConnector(result("pick_up", "tiptop@abc/pick-1", 12, cid="q1-6")), ok, ledger)
    with ledger.owner("ep.pick"):
        hr2.run("pick")
    assert hr2.summary()["inside_owner"] == [{"call_id": "q1-6", "owner": "ep.pick"}] and not hr2.summary()["ok"]
    assert ok.calls == [REFRESH_OWNER], "the refresh owner is the innermost even then"


def test_a_refresh_that_steps_the_sim_lands_on_the_refresh_owner_which_u0b_refuses(tmp_path):
    h = th.build(tmp_path, audit=False)
    sim, ledger = h.sim, h.ledger
    h.providers.refresh_hands = lambda world: Refresher(ledger, sim=sim)()
    h.host.refresh._after(result("place", "tiptop@abc/place-1", 40, cid="q1-7"))
    block, _ = th.run(h)
    assert block["ledger"][REFRESH_OWNER]["steps"] == 1 and block["u0b"]["checks"]["refresh_zero"] is False
    assert not block["ok"] and block["hand_refresh"]["count"] == 1


# ----------------------------------------------------------------------------------- the wiring in the built host
def test_the_host_wires_the_hand_refresh_between_the_audit_and_the_connector(tmp_path):
    h = th.build(tmp_path)
    host = h.host
    assert isinstance(host.refresh, episode_host.HandRefresh) and host.refresh._conn is host.direct
    assert host.audit.conn is host.refresh, "ConnectorAudit(HandRefresh(DirectConnector))"
    assert host.refresh._ledger is h.ledger


def test_the_hand_refresh_is_inert_under_parity(tmp_path):
    """Every result of a PARITY episode is legacy: the providers' refresh_hands is never asked, the block says 0."""
    h = th.build(tmp_path, tape=True)

    def never(world):
        raise AssertionError("refresh_hands was asked on a parity run")

    h.providers.refresh_hands = never
    block, raised = th.run(h)
    assert raised is None and block["ok"], json.dumps(block["g3"])
    assert block["hand_refresh"] == {"count": 0, "popped": [], "popped_total": 0, "calls": [], "inside_owner": [],
                                     "errors": [], "ok": True}
    assert block["g3"]["hand_refresh_ok"] is True
    assert REFRESH_OWNER not in block["ledger"] and block["u0b"]["checks"]["refresh_zero"] is True
    assert all(c["backend"] == "legacy" for c in [json.loads(l) for l in (tmp_path / "skill_calls.jsonl").read_text().splitlines()])


def test_a_native_hand_result_in_the_built_host_is_counted_in_the_block_at_zero_steps(tmp_path):
    h = th.build(tmp_path, audit=False)
    ledger = h.ledger
    seen = []

    def refresh(world):
        seen.append((ledger.current, world is h.host.svc.world))
        return ["jar_of_honey_1"]

    h.providers.refresh_hands = refresh
    h.host.refresh._after(result("pick_up", "tiptop@abc/pick-1", 12, cid="q1-8"))
    block, raised = th.run(h)
    assert raised is None and block["ok"], json.dumps(block["g3"]) + json.dumps(block["u0b"]["checks"])
    assert seen == [(REFRESH_OWNER, True)], "under the refresh owner, over the host's world"
    hr = block["hand_refresh"]
    assert (hr["count"], hr["popped"], hr["popped_total"], hr["ok"]) == (1, ["jar_of_honey_1"], 1, True)
    assert block["ledger"][REFRESH_OWNER] == {**block["ledger"][REFRESH_OWNER], "steps": 0, "env_step_calls": 0}
    assert block["u0b"]["checks"]["refresh_zero"] is True and block["g3"]["hand_refresh_ok"] is True


def test_an_inside_owner_breach_fails_the_block(tmp_path):
    h = th.build(tmp_path, audit=False)
    h.providers.refresh_hands = lambda world: []
    with h.ledger.owner("ep.achieve"):
        h.host.refresh._after(result("place", "tiptop@abc/place-1", 40, cid="q1-9"))
    block, _ = th.run(h)
    assert block["hand_refresh"]["inside_owner"] == [{"call_id": "q1-9", "owner": "ep.achieve"}]
    assert block["g3"]["hand_refresh_ok"] is False and not block["ok"]


# ------------------------------------------------------------------------------------------------- the ladder
def _write(member, args, kwargs, step, ret=True, exc=None):
    r = {"kind": "write", "member": member, "args": list(args), "kwargs": dict(kwargs), "step": list(step), "ret": ret}
    if exc:
        r["exc"] = {"type": exc, "module": "omnigibson.tiptop.scene", "message": exc}
        r.pop("ret")
    return r


INSIDE = {"predicate": "inside", "args": ["jar.n.01_1", "cabinet.n.01_1"]}
ONTOP = {"predicate": "ontop", "args": ["pillow.n.01_1", "bed.n.01_1"]}
RECORDS = [
    {"kind": "read", "member": "holding", "args": ["jar.n.01_1"], "kwargs": {}, "ret": False},
    _write("open_up", ["cabinet.n.01_1"], {"fraction": 0.8}, [90, 400]),  # q1-1: no frame (its own solve)
    _write("pick", ["jar.n.01_1"], {"into": "cabinet.n.01_1"}, [400, 900]),  # q1-2: one round
    _write("walk_to_floor", ["floor.n.01_2"], {}, [900, 1000]),  # navigation: not a skill
    _write("put_down", ["jar.n.01_1", "floor.n.01_2"], {"floor": True}, [1000, 1800]),  # q1-3 place.on: two rounds
    _write("stand_for", ["bed.n.01_1"], {}, [1800, 1900]),
    _write("achieve", [[ONTOP]], {}, [1900, 2600]),  # q1-4 place.on: one round
    _write("achieve", [[INSIDE]], {}, [2600, 3000], ret=False),  # q1-5 place.in: two rounds
    _write("open_up", ["cabinet.n.01_1"], {"fraction": 0.0}, [3000, 3100], exc="EpisodeOver"),  # q1-6 close
]
ROUNDS = [{"round": 1, "step": 500, "env_steps": 220}, {"round": 3, "step": 1100, "env_steps": 0, "error": "motion validation rejected"},
          {"round": 4, "step": 1400, "env_steps": 300}, {"round": 6, "step": 2000, "env_steps": 250},
          {"round": 8, "step": 2650, "error": "TiptopPlanningError: planning failed"}, {"round": 9, "step": 2800, "error": "GoalNotVisible: goal"},
          {"round": 10, "step": 2900, "error": "episode over"}, {"stand_for": ["bed.n.01_1"], "step": 1800}]
INDEX = [{"i": 0, "op": "metadata", "owner": "unowned"}, {"i": 1, "op": "plan", "owner": "ep.pick", "k": 1, "seed": 2301},
         {"i": 2, "op": "plan", "owner": "ep.achieve", "k": 2, "seed": 2302}, {"i": 3, "op": "plan", "owner": "ep.achieve", "k": 3, "seed": 2303},
         {"i": 4, "op": "plan", "owner": "ep.achieve", "k": 4, "seed": 2304}, {"i": 5, "op": "plan", "owner": "ep.achieve", "k": 5, "seed": 2305},
         {"i": 6, "op": "plan", "owner": "ep.achieve", "k": 6, "seed": 2306}]


def _tape() -> tp.Tape:
    return tp.Tape({"task": "demo", "instance": 301}, [dict(r) for r in RECORDS])


def test_writes_are_typed_as_the_shim_types_them_and_numbered_as_it_numbers_them():
    ws = ladder.writes_of(_tape())
    assert [(w.member, w.k, w.skill, w.qual) for w in ws] == [
        ("open_up", 1, "open", None), ("pick", 2, "pick_up", None), ("walk_to_floor", None, None, None),
        ("put_down", 3, "place", "on"), ("stand_for", None, None, None), ("achieve", 4, "place", "on"),
        ("achieve", 5, "place", "in"), ("open_up", 6, "close", None)]
    assert [w.line for w in ws if w.k] == ["open", "pick_up", "place.on", "place.on", "place.in", "close"]
    assert [w.call_id for w in ws][:4] == ["q1-1", "q1-2", None, "q1-3"]
    assert (ws[6].returned, ws[7].returned, ws[7].exc) == (False, None, "EpisodeOver")


def test_stage_routes_and_the_branch_write():
    assert ladder.stage_routes("S1") == ([], ["place.on=tiptop"])
    assert ladder.stage_routes("S2") == (["place.on=tiptop"], ["place.on=tiptop", "close.prismatic=tiptop"])
    ws = ladder.writes_of(_tape())
    c, t = ladder.profile_of([]), ladder.profile_of(["place.on=tiptop"])
    assert ladder.backends_of(ws, c) == ["legacy", "legacy", None, "legacy", None, "legacy", "legacy", "legacy"]
    assert ladder.backends_of(ws, t) == ["legacy", "legacy", None, "tiptop", None, "tiptop", "legacy", "legacy"]
    b = ladder.branch_of(ws, c, t)
    assert (b.write, b.k, b.call_id, b.member, b.line, b.backend_c, b.backend_t) == (3, 3, "q1-3", "put_down", "place.on", "legacy", "tiptop")
    assert b.record == 4, "the record index in the tape (the read comes first)"
    # S2: the close names no joint, so by_joint cannot choose and both arms take the default: no branch
    c2, t2 = ladder.profile_of(["place.on=tiptop"]), ladder.profile_of(["place.on=tiptop", "close.prismatic=tiptop"])
    assert ladder.branch_of(ws, c2, t2) is None
    assert ladder.backends_of(ws, t2)[-1] == "legacy"


def test_frames_align_to_writes_by_the_rounds_and_by_the_owners():
    ws = ladder.writes_of(_tape())
    # rounds: frame j is the j-th requesting round (the GoalNotVisible round asks nothing), placed by its step
    by = ladder.frames_by_rounds(INDEX, ROUNDS, ws)
    assert by == {1: 1, 2: 3, 3: 3, 4: 5, 5: 6, 6: 6}
    b = ladder.branch_of(ws, ladder.profile_of([]), ladder.profile_of(["place.on=tiptop"]))
    assert ladder.frame_of_branch(INDEX, ws, b, ROUNDS) == (2, "rounds")
    # owners alone: the consecutive achieve owners fold into the first write that owns them
    assert ladder.frames_by_write(INDEX, ws) == {1: 1, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3}
    assert ladder.frame_of_branch(INDEX, ws, b, None) == (2, "owner")
    # a later branch (place.in) is where the two methods part: the rounds place it, the owners cannot
    b_in = ladder.Branch(6, 7, 5, "achieve", "place.in", "legacy", "tiptop")
    assert ladder.frame_of_branch(INDEX, ws, b_in, ROUNDS) == (5, "rounds")
    assert ladder.frame_of_branch(INDEX, ws, b_in, None) == (None, None)
    # a count mismatch leaves the rounds out
    assert ladder.frames_by_rounds(INDEX, ROUNDS[:3], ws) == {}
    assert ladder.requesting_rounds(ROUNDS) == [r for r in ROUNDS if "round" in r and not str(r.get("error", "")).startswith("GoalNotVisible")]


def test_the_erep_cross_check_needs_a_matched_prefix():
    rows = [{"i": 0, "op": "metadata", "call_id": None, "matched": True}, {"i": 1, "op": "plan", "call_id": "q1-2", "matched": True},
            {"i": 2, "op": "plan", "call_id": "q1-3", "matched": True}, {"i": 3, "op": "plan", "call_id": "q1-3", "matched": False}]
    assert ladder.frame_of_call(rows, "q1-3") == 2
    assert ladder.frame_of_call(rows, "q1-4") is None, "past the divergence nothing is known"
    rows[1]["matched"] = False
    assert ladder.frame_of_call(rows, "q1-3") is None


def test_the_restamped_tape_changes_the_plan_seeds_alone(tmp_path):
    src, dst = TapeDir(tmp_path / "src"), tmp_path / "r2"
    depth = np.arange(6, dtype=np.float32).reshape(2, 3)
    src.append(Frame(0, "metadata", metadata={"embodiment": {"arms": ["left"]}}), whole=True)
    src.append(Frame(1, "plan", owner="ep.pick", server="127.0.0.1:8804", k=1, seed=seed_for(0, 1),
                     request=packb({"task": "put", "depth": depth, "seed": seed_for(0, 1)}), response='{"success": true}', wall_s=1.5), whole=True)
    src.append(Frame(2, "skill", owner="rt", server="127.0.0.1:8804", k=2,
                     request=packb({"type": "skill", "skill": "place", "seed": 7}), response=b'{"ok": false}'), whole=True)
    src.append(Frame(3, "plan", owner="ep.achieve", server="127.0.0.1:8804", k=3, seed=seed_for(0, 3),
                     request=packb({"task": "put", "seed": seed_for(0, 3)}), response='{"success": false}'), whole=True)
    assert ladder.restamp_tape(tmp_path / "src", dst, 2) == 4
    out = TapeDir(dst)
    assert len(out) == 4 and out.whole
    f0, f1, f2, f3 = (out.load(i) for i in range(4))
    assert f0.op == "metadata" and f0.metadata == {"embodiment": {"arms": ["left"]}} and f0.request is None
    r1 = f1.request_dict()
    assert (r1["seed"], f1.seed, f1.k, f1.owner, f1.wall_s) == (seed_for(2, 1), seed_for(2, 1), 1, "ep.pick", 1.5) and seed_for(2, 1) == 4301
    assert np.array_equal(r1["depth"], depth) and r1["task"] == "put" and f1.response == '{"success": true}'
    assert f2.request_dict() == {"type": "skill", "skill": "place", "seed": 7} and f2.response == b'{"ok": false}' and f2.seed is None
    assert f3.request_dict()["seed"] == seed_for(2, 3) == 4303 and f3.response == '{"success": false}'
    rows = out.rows()
    assert [r["seed"] for r in rows] == [None, 4301, None, 4303] and rows[1]["fields"]["depth"] == src.rows()[1]["fields"]["depth"]
    assert rows[1]["fields"]["seed"] != src.rows()[1]["fields"]["seed"]
    with pytest.raises(FileExistsError):
        ladder.restamp_tape(tmp_path / "src", dst, 2)
    # replicate 0 keeps every seed
    ladder.restamp_tape(tmp_path / "src", tmp_path / "r0", 0)
    assert [unpackb(TapeDir(tmp_path / "r0").load(i).request)["seed"] for i in (1, 3)] == [2301, 2303]


def test_the_arm_command_lines(tmp_path):
    plan = {"tasks": {"bringing_in_wood": {"branch": {"write": 2}, "frame": 2, "tapes": {"0": "/t/r0", "1": "/t/r1", "2": "/t/r2"}},
                      "store_honey": {"branch": None}}}
    specs = ladder.stage_specs("S1", plan)
    assert [(s.task, s.arm, s.rep) for s in specs] == [("bringing_in_wood", a, r) for r in (0, 1, 2) for a in ("C", "T")]
    c, t = specs[0].bench_args(8850), specs[1].bench_args(8851)
    assert "--route" not in c and c[c.index("--routing-profile") + 1] == "parity"
    assert t[t.index("--route") + 1] == "place.on=tiptop" and t.count("--route") == 1
    for a, port in ((c, "8850"), (t, "8851")):
        assert a[a.index("--wstape") + 1] == "replay-live" and a[a.index("--wstape-live-at") + 1] == "2"
        assert a[a.index("--wstape-path") + 1] == "/t/r0" and a[a.index("--replicate") + 1] == "0" and a[a.index("--seed") + 1] == "0"
        assert a[a.index("--port") + 1] == port and a[a.index("--runner") + 1] == "connector" and "--runner-tape" in a
        for flag in ladder.COMMON:
            assert flag in a
    r2 = specs[5].bench_args(8852)
    assert r2[r2.index("--replicate") + 1] == "2" and r2[r2.index("--seed") + 1] == "2" and r2[r2.index("--wstape-path") + 1] == "/t/r2"
    s2 = ladder.stage_specs("S2", {"tasks": {"x": {"branch": {"write": 0}, "frame": 4, "tapes": {"0": "/t", "1": "/t", "2": "/t"}}}}, reps=(0,))
    assert s2[0].routes == ["place.on=tiptop"] and s2[1].routes == ["place.on=tiptop", "close.prismatic=tiptop"]
    st = ladder.strict_spec().bench_args(8999)
    assert st[st.index("--wstape") + 1] == "replay" and "--wstape-live-at" not in st and "--seed" not in st and not ladder.strict_spec().planner
    assert st[st.index("--routing-profile") + 1] == "parity" and st[st.index("--replicate") + 1] == "0"
    w = ladder.witness_spec().bench_args(8860)
    assert w[w.index("--task-name") + 1] == "cook_bacon" and w[w.index("--routing-profile") + 1] == "native"
    assert [w[i + 1] for i, x in enumerate(w) if x == "--route"] == ["press=tiptop", "place.on=tiptop"] and w[w.index("--wstape") + 1] == "record"
    cs = ladder.carried_specs(("dispose_of_batteries",))
    assert cs[0].profile == "native" and cs[0].routes == [] and cs[0].planner and cs[0].wstape == "record"
    text = ladder.launcher_text(specs[1], 8851, 3, Path("/snap/w4ladder"))
    assert "S=/snap/w4ladder" in text and "tiptop-server" in text and "--wstape-live-at 2" in text and "kill -TERM $ppid" in text
    assert "setsid env" in text and "OMP_NUM_THREADS=8" in text and "PYTHONHASHSEED=2300" in text and "CUDA_VISIBLE_DEVICES=$gpu" in text
    st_text = ladder.launcher_text(ladder.strict_spec(), 8999, 1, Path("/snap/w4ladder"))
    assert "tiptop-server" not in st_text and "simslot.sh acquire" in st_text


def test_runner_tape_rows_for_counters_come_from_the_runs_own_tape(tmp_path):
    ep = tmp_path / "episode"
    (ep / "tapes").mkdir(parents=True)
    (ep / "demo_301_0").mkdir()
    _tape().save(ep / "tapes" / "demo_301_0.json")
    p = ladder.write_runner_tape_rows(tmp_path)
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert p == ep / "demo_301_0" / "runner_tape.jsonl"
    assert [(r["call_id"], r["skill"], r["qual"], r["returned"]) for r in rows] == [
        ("q1-1", "open", None, True), ("q1-2", "pick_up", None, True), ("q1-3", "place", "on", True),
        ("q1-4", "place", "on", True), ("q1-5", "place", "in", False), ("q1-6", "close", None, None)]


def test_the_prefix_verdict_forced_at_the_branch_early_within_the_floor_and_early_outside_it():
    replay = [{"i": 0, "op": "metadata", "matched": True, "diffs": []}, {"i": 1, "op": "plan", "matched": True, "diffs": []}]
    run = {"wstape": {"switched_at": 2, "switch_reason": "frame 2 forced live (--wstape-live-at 2)"}, "replay": replay}
    pv = ladder.prefix_verdict(run, 2, None, 36, None, Path("/snap"))
    assert pv["ok"] and pv["forced_at_branch"] and not pv["pre_branch_divergence"] and pv["frames_before_branch"] == 2
    early = {"wstape": {"switched_at": 3, "switch_reason": "frame 3 (plan) differs at depth: ..."},
             "replay": replay + [{"i": 2, "op": "plan", "matched": True, "diffs": []}, {"i": 3, "op": "plan", "matched": False, "diffs": [{"path": "depth"}]}]}
    within = ladder.prefix_verdict(early, 6, 3, 47, None, Path("/snap"))
    assert within["pre_branch_divergence"] and within["within_floor"] and within["ok"] and not within["forced_at_branch"]
    assert within["mismatched_before_branch"] == [{"i": 3, "diffs": ["depth"]}]
    outside = ladder.prefix_verdict(early, 6, None, 47, None, Path("/snap"))
    assert outside["pre_branch_divergence"] and not outside["within_floor"] and not outside["ok"]
    below = ladder.prefix_verdict(early, 6, 5, 47, None, Path("/snap"))
    assert not below["within_floor"] and not below["ok"], "a divergence at frame 3 is not covered by a floor at 5"


def _block(ok=True, **over):
    b = {"ok": ok, "step": 120, "idle_steps": 0, "charged": {"skill": 120, "go_to": 0}, "u0a": {"ok": ok, "L": 0, "X": 0},
         "u0b": {"ok": ok}, "u0c": {"ok": True}, "u0d": {"ok": True}, "build": {"ok": True},
         "rule2": {"ok": True, "rt": {"place_robot": 0, "capture": 0, "look_at": 0}, "d21_debt": {"place_robot": 2}},
         "hand_refresh": {"ok": True, "count": 1, "popped_total": 1}, "reason": "success"}
    b.update(over)
    return b


def test_the_hard_verdict_on_a_treatment_run():
    row = lambda cid, code, backend="tiptop@x/place-1": {"call_id": cid, "skill": "place", "backend": backend, "code": code, "detail": "d"}  # noqa: E731
    good = {"connector": _block(), "json": {"reason": "success"}, "skill_calls": [row("q1-3", None)],
            "gripper": [{"step": 5, "arm": "left", "event": "close", "is_grasping": 1, "owner": "ep.pick", "call_id": "q1-2"}]}
    hv = ladder.hard_verdict(good, "place.on", "T", {"success"})
    assert hv["ok"] and hv["u0"]["exact"] and hv["u0"]["identity"] and hv["on_air"] == [] and not hv["switched_unsupported"]
    unsupported = dict(good, skill_calls=[row("q1-3", "unsupported")])
    assert not ladder.hard_verdict(unsupported, "place.on", "T", {"success"})["ok"]
    assert ladder.hard_verdict(dict(good, skill_calls=[row("q1-3", "unsupported", "legacy")]), "place.on", "T", {"success"})["ok"], "a legacy row is not the switched skill"
    crash = dict(good, json={"reason": "crash: KeyError: x"})
    assert ladder.hard_verdict(crash, "place.on", "T", {"success"})["crash_c_lacks"] and not ladder.hard_verdict(crash, "place.on", "T", {"success"})["ok"]
    assert not ladder.hard_verdict(crash, "place.on", "T", {"crash: KeyError: x"})["crash_c_lacks"], "C crashed the same way"
    on_air = dict(good, gripper=[{"step": 9, "arm": "left", "event": "close", "is_grasping": -1, "owner": "rt", "call_id": "q1-3"}])
    hv = ladder.hard_verdict(on_air, "place.on", "T", {"success"})
    assert hv["on_air"][0]["call_id"] == "q1-3" and hv["on_air_explained"] and hv["ok"]
    unowned = dict(good, gripper=[{"step": 9, "arm": "left", "event": "close", "is_grasping": -1, "owner": None, "call_id": None}])
    assert not ladder.hard_verdict(unowned, "place.on", "T", {"success"})["on_air_explained"]
    broken = dict(good, connector=_block(ok=False, u0b={"ok": False}))
    assert not ladder.hard_verdict(broken, "place.on", "T", {"success"})["ok"]
    refresh = dict(good, connector=_block(hand_refresh={"ok": False, "count": 1, "errors": ["x"]}))
    assert not ladder.hard_verdict(refresh, "place.on", "T", {"success"})["ok"]


def test_the_u0_reading_of_a_block_is_the_identity_and_the_hosts_verdicts():
    u = ladder.u0_of(_block())
    assert u["exact"] and u["identity"] and u["charged"] == {"skill": 120, "go_to": 0}
    assert not ladder.u0_of(_block(step=121))["identity"] and not ladder.u0_of(_block(step=121))["exact"]
    assert not ladder.u0_of(None)["exact"]


# ------------------------------------------------------------------------------- counters.py on the bench's layout
def _bench_job(d: Path, *, connector_block, rounds, gripper, calls, legacy=False):
    """A job dir as bench.py lays it out: episode/json/<task>_301_0.json with bench.connector, the host's rows in
    episode/<task>_301_0/, the GripperWatch rows there under either runner."""
    inst = d / "episode" / "demo_301_0"
    (d / "episode" / "json").mkdir(parents=True)
    inst.mkdir()
    bench = {"reason": "success", "rounds": rounds, "goal": {"satisfied": ["ontop(a, t)"], "new": 1, "total": 1}, "teleports": 2}
    if not legacy:
        bench["connector"] = connector_block
    (d / "episode" / "json" / "demo_301_0.json").write_text(json.dumps({"task": "demo", "instance_id": 301, "steps": 900, "bench": bench}))
    (inst / "gripper.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gripper))
    if not legacy:
        (inst / "skill_calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in calls))
        (inst / "ledger.jsonl").write_text(json.dumps({"owner": "ep.pick", "steps": 900, "env_step_calls": 900, "place_robot": 2, "capture": 1, "look_at": 0, "writes": 1}) + "\n")


ROUNDS_CUT = [{"round": 1, "step": 10, "env_steps": 220, "arm": "left", "atoms": []},
              {"round": 3, "step": 500, "env_steps": None, "error": "episode over", "arm": "left", "atoms": []}]
GRIPPER_ROWS = [{"step": 30, "arm": "left", "event": "close", "is_grasping": 1, "owner": "ep.pick", "call_id": "q1-1", "via": "executor"},
                {"step": 600, "arm": "left", "event": "close", "is_grasping": -1, "owner": "ep.achieve", "call_id": "q1-2", "via": "executor"}]
CALL_ROWS = [{"call_id": "q1-1", "skill": "pick_up", "backend": "legacy", "status": "succeeded", "code": None, "steps": 0, "requires_sim_clock": True,
              "effects": [{"pred": "holding", "args": ["a", "left"], "value": True}], "verdicts": {"scorer": True},
              "evidence": {"legacy_ok": True, "records": [ROUNDS_CUT[0]]}}]


def test_counters_reads_the_benchs_own_layout_as_a_connector_run(tmp_path):
    """bench.connector, the rows in the instance dir, the round EpisodeOver cut off (never a skill row) from
    bench.rounds, the on-air close's owner from gripper.jsonl: the E-rep's counters equal its L-rec's."""
    _bench_job(tmp_path, connector_block={"step": 0, "idle_steps": 0, "charged": {"skill": 0}, "live_at_end": None},
               rounds=ROUNDS_CUT, gripper=GRIPPER_ROWS, calls=CALL_ROWS)
    c = counters.extract(tmp_path)
    assert c.runner == "connector"
    assert (c.executed, c.cut_off, c.welds, c.on_air) == (1, 1, 1, 1)
    assert c.on_air_causes == [{"step": 600, "arm": "left", "owner": "ep.achieve", "call_id": "q1-2"}]
    assert (c.teleports, c.teleports_ep, c.teleports_go_to) == (2, 2, 0) and c.notes == []
    assert [(x.call_id, x.returned, x.scorer) for x in c.calls] == [("q1-1", True, True)]


def test_counters_names_the_owner_of_a_legacy_runs_on_air_close_from_its_gripper_rows(tmp_path):
    _bench_job(tmp_path, connector_block=None, rounds=ROUNDS_CUT, gripper=GRIPPER_ROWS, calls=[], legacy=True)
    c = counters.extract(tmp_path)
    assert c.runner == "legacy" and (c.executed, c.cut_off, c.welds, c.on_air) == (1, 1, 1, 1)
    assert c.on_air_causes == [{"step": 600, "arm": "left", "owner": "ep.achieve", "call_id": "q1-2"}]
