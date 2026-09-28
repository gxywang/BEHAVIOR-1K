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
    host.close("wired", None)  # restores skillrun.PASSTHROUGH for the tests after this one


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
    strict = ladder.strict_spec(Path("/snap/w4ladder"), tape=Path("/t/strict_r0"))
    st = strict.bench_args(8999)
    assert st[st.index("--wstape") + 1] == "replay" and "--wstape-live-at" not in st and "--seed" not in st and not strict.planner
    assert st[st.index("--wstape-path") + 1] == "/t/strict_r0"
    assert st[st.index("--routing-profile") + 1] == "parity" and st[st.index("--replicate") + 1] == "0"
    w = ladder.witness_spec().bench_args(8860)
    assert w[w.index("--task-name") + 1] == "cook_bacon" and w[w.index("--routing-profile") + 1] == "native"
    assert [w[i + 1] for i, x in enumerate(w) if x == "--route"] == ["press=tiptop", "place.on=tiptop"] and w[w.index("--wstape") + 1] == "record"
    cs = ladder.carried_specs(("dispose_of_batteries",))
    assert cs[0].profile == "native" and cs[0].planner and cs[0].wstape == "record"
    assert cs[0].routes == ["place.on=tiptop", "close.prismatic=tiptop"], "NATIVE plus every ladder line: the full native profile"
    ca = cs[0].bench_args(8870)
    assert [ca[i + 1] for i, x in enumerate(ca) if x == "--route"] == ["place.on=tiptop", "close.prismatic=tiptop"]
    text = ladder.launcher_text(specs[1], 8851, 3, Path("/snap/w4ladder"))
    assert "S=/snap/w4ladder" in text and "tiptop-server" in text and "--wstape-live-at 2" in text and "kill -TERM $ppid" in text
    assert "setsid env" in text and "OMP_NUM_THREADS=8" in text and "PYTHONHASHSEED=2300" in text and "CUDA_VISIBLE_DEVICES=$gpu" in text
    st_text = ladder.launcher_text(strict, 8999, 1, Path("/snap/w4ladder"))
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
    assert pv["ok"] and pv["switched_at_branch"] and pv["forced"] and not pv["pre_branch_divergence"] and pv["frames_before_branch"] == 2
    # the T arm: its native request at N differs from the tape at its op and goes live on that, at N all the same
    native = {"wstape": {"switched_at": 2, "switch_reason": "frame 2 (skill) differs at op: plan on the tape, skill in the replay"}, "replay": replay}
    pv = ladder.prefix_verdict(native, 2, None, 36, None, Path("/snap"))
    assert pv["ok"] and pv["switched_at_branch"] and not pv["forced"] and not pv["pre_branch_divergence"]
    early = {"wstape": {"switched_at": 3, "switch_reason": "frame 3 (plan) differs at depth: ..."},
             "replay": replay + [{"i": 2, "op": "plan", "matched": True, "diffs": []}, {"i": 3, "op": "plan", "matched": False, "diffs": [{"path": "depth"}]}]}
    within = ladder.prefix_verdict(early, 6, 3, 47, None, Path("/snap"))
    assert within["pre_branch_divergence"] and within["within_floor"] and within["ok"] and not within["switched_at_branch"]
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


def test_the_derived_tape_reroots_the_served_metadata_to_the_runs_snapshot(tmp_path):
    """check_imports refuses a served metadata frame naming another checkout than the sim's: the derived tape moves
    the recording planner's ``modules`` paths under the run's snapshot and changes nothing else."""
    md = {"server": "tiptop", "robot_type": "r1pro_left", "modules": {"tiptop": "/snap/w4q1/tiptop/tiptop/__init__.py",
                                                                        "cutamp": "/snap/w4q1/tiptop/cutamp/cutamp/__init__.py"}}
    assert ladder.snapshot_root_of(md) == Path("/snap/w4q1") and ladder.snapshot_root_of({}) is None
    out, changed = ladder.reroot_metadata(md, Path("/snap/w4q1"), Path("/snap/w4ladder"))
    assert out["modules"] == {"tiptop": "/snap/w4ladder/tiptop/tiptop/__init__.py", "cutamp": "/snap/w4ladder/tiptop/cutamp/cutamp/__init__.py"}
    assert [c[0] for c in changed] == ["tiptop", "cutamp"] and out["robot_type"] == "r1pro_left" and md["modules"]["tiptop"].startswith("/snap/w4q1")
    assert ladder.reroot_metadata(md, Path("/elsewhere"), Path("/snap/w4ladder"))[1] == [], "another root is left alone"
    src = TapeDir(tmp_path / "src")
    src.append(Frame(0, "metadata", metadata=md), whole=True)
    src.append(Frame(1, "plan", owner="ep.pick", server="127.0.0.1:8804", k=1, seed=2301, request=packb({"task": "put", "seed": 2301}),
                     response='{"success": true}'), whole=True)
    dst = ladder.derived_tape(tmp_path / "src", tmp_path / "r1", 1, Path("/snap/w4ladder"))
    f0, f1 = TapeDir(dst).load(0), TapeDir(dst).load(1)
    assert f0.metadata["modules"]["tiptop"] == "/snap/w4ladder/tiptop/tiptop/__init__.py" and f0.metadata["server"] == "tiptop"
    assert f1.request_dict()["seed"] == seed_for(1, 1) == 3301 and f1.response == '{"success": true}'
    assert ladder.derived_tape(tmp_path / "src", tmp_path / "r1", 1, Path("/snap/w4ladder")) == dst, "made once, reused after"
    assert len(TapeDir(dst)) == 2


# ------------------------------------------------------------------- the G3 items that are PARITY's alone
def test_renders_outside_ep_and_native_requests_gate_a_parity_run_alone(tmp_path):
    """A native route captures through the planner's observe and asks the planner for skills: those renders and
    requests are reported on such a run, never gated (they are the parity items of G3, WEEK4_PLAN 5.4)."""
    h = th.build(tmp_path, audit=False)
    h.ledger.row("observe").renders = 3
    block, _ = th.run(h)
    assert block["g3"]["parity_run"] is True and block["g3"]["renders_outside_ep"] == 3 and not block["ok"]
    h2 = th.build(tmp_path / "native", audit=False, routes=["wait=scripted"])
    h2.ledger.row("observe").renders = 3
    block2, _ = th.run(h2)
    assert block2["g3"]["parity_run"] is False and block2["g3"]["renders_outside_ep"] == 3 and block2["ok"], json.dumps(block2["g3"])
    assert block2["g3"]["u0a"] and block2["g3"]["u0b"] and block2["g3"]["rule2"], "the hard items still gate it"


# ---------------------------------------------------------- the Runner prefix and the cut-off delivery
def test_the_runner_prefix_treats_the_distance_keyerror_message_as_neutral_and_nothing_else():
    """W4-F2's open item 1: the shim's KeyError message differs from the Episode's at every distance read the sim
    cannot answer; Runner.gap maps both to inf, so the pair is neutral. Any other difference is a divergence."""
    lrec = [{"kind": "read", "member": "holding", "args": ["a"], "kwargs": {}, "ret": False},
            {"kind": "read", "member": "distance", "args": ["floor.n.01_2", "a"], "kwargs": {},
             "exc": {"type": "KeyError", "module": "builtins", "message": "'floor.n.01_2'", "args": ["floor.n.01_2"]}},
            {"kind": "write", "member": "pick", "args": ["a"], "kwargs": {}, "step": [0, 10], "ret": True},
            {"kind": "read", "member": "support_of", "args": ["a"], "kwargs": {}, "ret": None},
            {"kind": "write", "member": "put_down", "args": ["a", "f"], "kwargs": {"floor": True}, "step": [10, 20], "ret": True}]
    run = [dict(r) for r in lrec]
    run[1] = dict(run[1], exc={"type": "KeyError", "module": "builtins", "message": "'no distance between floor.n.01_2 and a'",
                                "args": ["no distance between floor.n.01_2 and a"]})
    cmp = ladder.runner_prefix_compare(lrec, run, 4)
    assert cmp == {"compared": 4, "first_divergence": None, "neutral": [1], "identical_before_branch": True}
    run[3] = dict(run[3], ret="table")
    cmp = ladder.runner_prefix_compare(lrec, run, 4)
    assert cmp["first_divergence"]["index"] == 3 and cmp["first_divergence"]["kind"] == "answer" and not cmp["identical_before_branch"]
    assert cmp["neutral"] == [1]
    # a distance read whose args differ, or that answered a value on one side, is not neutral
    other = [dict(r) for r in lrec]
    other[1] = dict(other[1], args=["floor.n.01_1", "a"])
    assert ladder.runner_prefix_compare(lrec, other, 4)["first_divergence"]["index"] == 1
    valued = [dict(r) for r in lrec]
    valued[1] = {"kind": "read", "member": "distance", "args": ["floor.n.01_2", "a"], "kwargs": {}, "ret": 1.5}
    assert ladder.runner_prefix_compare(lrec, valued, 4)["first_divergence"]["index"] == 1
    # the same pair as Tape.load decodes it (the exception an Exc named tuple), and one with a value on one side
    assert ladder.exc_type(tp.Exc("KeyError", "builtins", "'x'", ("x",))) == "KeyError" and ladder.exc_type({"type": "ValueError"}) == "ValueError"
    assert ladder.exc_type(None) is None
    decoded_l = tp.Tape.loads(tp.Tape({}, lrec).dumps()).records
    decoded_r = tp.Tape.loads(tp.Tape({}, run[:2] + lrec[2:]).dumps()).records
    assert ladder.runner_prefix_compare(decoded_l, decoded_r, 4) == {"compared": 4, "first_divergence": None, "neutral": [1], "identical_before_branch": True}
    as_exc_l = [dict(r, exc=tp.Exc(**r["exc"])) if "exc" in r else r for r in lrec]  # the form Tape.load gives a file's records
    as_exc_r = [dict(r, exc=tp.Exc(**r["exc"])) if "exc" in r else r for r in run[:2] + lrec[2:]]
    assert ladder.runner_prefix_compare(as_exc_l, as_exc_r, 4)["neutral"] == [1] and ladder.runner_prefix_compare(as_exc_l, as_exc_r, 4)["identical_before_branch"]
    # only the records before the branch count; a shorter run tape is a divergence
    assert ladder.runner_prefix_compare(lrec, run[:2], 4)["first_divergence"] == {"index": 2, "why": "the run's tape is shorter"}
    assert ladder.runner_prefix_compare(lrec, run[:2], 2)["identical_before_branch"]


def test_a_switched_write_cut_off_at_success_is_a_cut_off_delivery(tmp_path):
    ep = tmp_path / "episode"
    (ep / "tapes").mkdir(parents=True)
    records = [dict(r) for r in RECORDS]
    records[-1] = _write("achieve", [[ONTOP]], {}, [3000, 3100], exc="EpisodeOver")  # the last write: place.on, cut off
    tp.Tape({"task": "demo"}, records).save(ep / "tapes" / "demo_301_0.json")
    run = {"runner_tape": str(ep / "tapes" / "demo_301_0.json"), "json": {"reason": "success"}}
    assert ladder.cut_off_delivery(run, "place.on") == {"call_id": "q1-6", "member": "achieve", "args": [[ONTOP]], "line": "place.on"}
    assert ladder.cut_off_delivery(run, "place.in") is None, "another line"
    assert ladder.cut_off_delivery(dict(run, json={"reason": "episode over after 5000 steps: timeout"}), "place.on") is None
    tp.Tape({"task": "demo"}, RECORDS).save(ep / "tapes" / "demo_301_0.json")  # the last write is a close
    assert ladder.cut_off_delivery(run, "place.on") is None


def test_the_ordinal_pairing_names_the_same_write_once_the_sequences_part(tmp_path):
    """After the branch C's and T's Runner writes differ, so a call id no longer names the same write; the switched
    skill's k-th call does, and the stage-failing cause is judged there, never on another skill sharing an id."""
    def run(inst, rows, cut=None):
        d = tmp_path / inst
        d.mkdir()
        (d / "runner_tape.jsonl").write_text("".join(json.dumps(r) + "\n" for r in
                                                     [{"call_id": r["call_id"], "skill": r["skill"], "qual": r.get("qual"), "args": r.get("args")} for r in rows]))
        return {"inst": str(d), "skill_calls": rows, "cut_off_delivery": cut, "ended": True}
    row = lambda cid, skill, backend, status, code=None, steps=100, scorer=True, qual=None, ok=True: {  # noqa: E731
        "call_id": cid, "skill": skill, "backend": backend, "status": status, "code": code, "steps": steps,
        "verdicts": {"scorer": scorer}, "evidence": {"legacy_ok": ok} if backend == "legacy" else {}, "qual": qual, "effects": []}
    runs = {("C", 0): run("c0", [row("q1-1", "pick_up", "legacy", "succeeded"), row("q1-2", "place", "legacy", "succeeded", qual="on"),
                               row("q1-3", "pick_up", "legacy", "succeeded"), row("q1-4", "place", "legacy", "infeasible", "no_placement", scorer=False, qual="on", ok=False),
                               row("q1-5", "place", "legacy", "succeeded", qual="on")], cut={"call_id": "q1-7", "member": "put_down", "args": [], "line": "place.on"}),
            ("T", 0): run("t0", [row("q1-1", "pick_up", "legacy", "succeeded"), row("q1-2", "place", "tiptop@x/place-1", "succeeded", qual="on"),
                               row("q1-3", "pick_up", "legacy", "succeeded"), row("q1-4", "place", "tiptop@x/place-1", "succeeded", qual="on"),
                               row("q1-6", "pick_up", "legacy", "infeasible", "no_stance_here", steps=0, scorer=False, ok=False)])}
    table = ladder.ordinal_pairing(runs, "place.on")
    assert [r["ordinal"] for r in table] == [1, 2, 3, 4]
    assert (table[0]["C0"]["backend"], table[0]["T0"]["backend"]) == ("legacy", "tiptop@x/place-1")
    assert table[1]["C0"]["code"] == "no_placement" and table[1]["T0"]["status"] == "succeeded"
    assert table[3]["C0"]["status"] == "cut_off" and table[3]["T0"] is None, "C's fourth is the cut-off delivery; T never made one"
    assert table[2]["T0"] is None, "T's pick_up q1-6 is not a place, whatever its id"
    assert ladder.switched_failed_where_c_succeeded(table) == [], "T's places all succeeded: no cause"
    runs[("T", 0)]["skill_calls"][3] = row("q1-4", "place", "tiptop@x/place-1", "failed", "no_placement", scorer=False, qual="on")
    table = ladder.ordinal_pairing(runs, "place.on")
    assert ladder.switched_failed_where_c_succeeded(table) == [], "at ordinal 2 C failed too"
    runs[("C", 0)]["skill_calls"][3] = row("q1-4", "place", "legacy", "succeeded", qual="on")
    table = ladder.ordinal_pairing(runs, "place.on")
    assert ladder.switched_failed_where_c_succeeded(table) == [{"ordinal": 2, "c_ok": [True], "t_failed": [("T0", "no_placement")]}]
    assert ladder.ordinal_pairing(runs, "place.in") == [], "no place.in in either run"


# ------------------------------------------------------------- U0's amended identity and the expected run count
def test_the_u0_reading_is_the_amended_identity_when_episode_over_cuts_a_live_run():
    """WEEK4_PLAN 5.6: step - idle == sum(charged) + L + X with idle 0, and L and X 0 unless EpisodeOver ended the
    episode. bringing_in_wood T1's block: 741 == 536 + 204 + 1, its native place q1-7 still live at the success."""
    cut = _block(step=741, charged={"skill": 536, "go_to": 0, "observe": 0}, u0a={"ok": True, "L": 204, "X": 1, "episode_over": True})
    u = ladder.u0_of(cut)
    assert u["identity"] and u["exact"] and (u["L"], u["X"], u["episode_over"]) == (204, 1, True)
    assert not ladder.u0_of(dict(cut, step=740))["identity"], "a step short of the amended sum"
    assert not ladder.u0_of(dict(cut, u0a={"ok": True, "L": 204, "X": 1, "episode_over": False}))["identity"], \
        "L and X count only when EpisodeOver ended the episode"
    assert not ladder.u0_of(_block(step=125, idle_steps=5))["identity"], "idle_steps must be 0, not subtracted away"


def test_a_stage_with_a_replicate_missing_does_not_pass(tmp_path, monkeypatch):
    """runs_expected is two arms x the replicates on every task with a branch, not the run dirs that happen to
    exist: rearrange_your_room T1 was moved aside after a stall, and 17 of 17 must not read as a complete stage."""
    monkeypatch.setattr(ladder, "OUT", tmp_path)
    monkeypatch.setitem(ladder.STAGES, "S1", ladder.Stage("S1", "place.on=tiptop", None, ("demo",)))
    task = "demo"
    plan = {"tasks": {task: {"branch": {"write": 1, "record": 5, "member": "put_down", "line": "place.on", "call_id": "q1-2",
                                        "backend_c": "legacy", "backend_t": "tiptop"},
                             "frame": 2, "floor": {}, "tape": str(tmp_path / "lrec" / "tapes" / "demo_301_0.json"), "writes": []},
                      "other": {"branch": None}}}
    (tmp_path / "S1").mkdir()
    (tmp_path / "S1" / "plan.json").write_text(json.dumps(plan))
    for arm in ("C", "T"):
        d = ladder.run_dir("S1", task, arm, 0)
        _bench_job(d, connector_block=_block(), rounds=ROUNDS_CUT[:1], gripper=[], calls=CALL_ROWS)
        (d / "job_end.json").write_text('{"ended": 1}')
    g = ladder.gate_stage("S1", Path("/snap"))
    v = g["verdict"]
    assert (v["runs_ended"], v["runs_expected"]) == (2, 6) and not v["pass"]


# ------------------------------------------------------ a C cut-off is a success; on-air causes; after each native call
def test_a_c_row_cut_off_at_success_counts_as_a_success_against_a_failed_treatment():
    """rearrange T2: its fourth place.on failed placed_wrong where every C's fourth was the write EpisodeOver cut
    off at the task's success. That C row delivered the goal, so T2's failure is listed as the switched skill
    failing where C's succeeded."""
    cut = {"call_id": "q1-8", "backend": "(cut off at success)", "status": "cut_off", "code": None, "steps": None, "scorer": True, "returned": None}
    failed = {"call_id": "q1-8", "backend": "tiptop@x/place-1", "status": "failed", "code": "placed_wrong", "steps": 429, "scorer": False, "returned": False}
    row = {"ordinal": 4, "C0": cut, "C1": dict(cut), "T0": dict(cut), "T2": failed}
    assert ladder.switched_failed_where_c_succeeded([row]) == [{"ordinal": 4, "c_ok": [True, True], "t_failed": [("T2", "placed_wrong")]}]
    assert ladder.switched_failed_where_c_succeeded([dict(row, T2=dict(cut))]) == [], "T cut off too: nothing failed"


def test_on_air_causes_name_the_owning_calls_row_and_whether_the_call_held_later():
    run = {"skill_calls": [{"call_id": "q1-6", "skill": "pick_up", "backend": "legacy", "status": "succeeded", "code": None},
                           {"call_id": "q1-7", "skill": "place", "backend": "tiptop@x/place-1", "status": "failed", "code": "placed_wrong"}],
           "gripper": [{"step": 5, "arm": "left", "event": "close", "is_grasping": -1, "owner": "ep.pick", "call_id": "q1-6"},
                       {"step": 6, "arm": "left", "event": "open", "is_grasping": -1, "owner": "ep.pick", "call_id": "q1-6"},
                       {"step": 9, "arm": "left", "event": "close", "is_grasping": 1, "owner": "ep.pick", "call_id": "q1-6"},
                       {"step": 12, "arm": "left", "event": "close", "is_grasping": -1, "owner": "rt", "call_id": "q1-7"}]}
    causes = ladder.on_air_causes(run)
    assert [(c["call_id"], c["skill"], c["native"], c["held_later_in_the_call"]) for c in causes] == \
        [("q1-6", "pick_up", False, True), ("q1-7", "place", True, False)], "an open on air is not a close"


def test_after_native_lists_the_phase_and_the_legacy_calls_up_to_the_next_native_one():
    """bringing_in_wood T2: a native place that ended in phase retreat, then four legacy picks refused with
    no_stance_here at 0 steps."""
    rows = [{"call_id": "q1-1", "skill": "pick_up", "backend": "legacy", "status": "succeeded", "code": None, "steps": 887},
            {"call_id": "q1-2", "skill": "place", "backend": "tiptop@x/place-1", "status": "succeeded", "code": None, "phase": "retreat", "steps": 288}]
    rows += [{"call_id": f"q1-{k}", "skill": "pick_up", "backend": "legacy", "status": "infeasible", "code": "no_stance_here", "steps": 0} for k in (3, 4)]
    rows += [{"call_id": "q1-5", "skill": "place", "backend": "tiptop@x/place-1", "status": "succeeded", "code": None, "phase": "home", "steps": 220}]
    out = ladder.after_native({"skill_calls": rows})
    assert [(x["call_id"], x["phase"], [t[0] for t in x["then"]]) for x in out] == [("q1-2", "retreat", ["q1-3", "q1-4"]), ("q1-5", "home", [])]
    assert out[0]["then"][0] == ("q1-3", "pick_up", "infeasible", "no_stance_here", 0)


def test_a_branched_task_outside_the_stages_list_is_carried_with_its_reason(tmp_path, monkeypatch):
    """attach and composting have a place.on branch in their L-rec (the floor put_down after a failed goal) past
    their A/A floor: they are not S1's arms, the gate says why, and they do not count toward the expected runs."""
    monkeypatch.setattr(ladder, "OUT", tmp_path)
    assert ladder.STAGES["S1"].tasks == ("bringing_in_wood", "rearrange_your_room", "tidying_bedroom")
    assert ladder.STAGES["S2"].tasks == ("store_honey",)
    plan = {"tasks": {"attach_a_camera_to_a_tripod": {"branch": {"write": 5, "record": 47, "member": "put_down", "line": "place.on"},
                                                      "frame": 6, "floor": {"frame": 3}},
                      "dispose_of_batteries": {"branch": None}}}
    (tmp_path / "S1").mkdir()
    (tmp_path / "S1" / "plan.json").write_text(json.dumps(plan))
    g = ladder.gate_stage("S1", Path("/snap"))
    note = g["tasks"]["attach_a_camera_to_a_tripod"]["note"]
    assert note.startswith("not a stage task: carried forward") and "past the floor" in note
    assert g["verdict"]["runs_expected"] == 0 and not g["verdict"]["pass"]
    assert [s.task for s in ladder.stage_specs("S1", {"tasks": {**plan["tasks"], "bringing_in_wood": {"branch": {"write": 2}, "frame": 2, "tapes": {"0": "/t"}}}},
                                                tasks=ladder.STAGES["S1"].tasks, reps=(0,))] == ["bringing_in_wood", "bringing_in_wood"]


def test_the_carried_gate_lists_each_native_call_and_writes_its_table(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "OUT", tmp_path)
    monkeypatch.setattr(ladder, "CARRIED", ("demo",))
    d = ladder.run_dir("carried", "demo", "native", 0)
    native_row = {"call_id": "q1-4", "skill": "place", "backend": "tiptop@x/place-1", "status": "succeeded", "code": None, "phase": "home",
                  "steps": 300, "effects": [{"pred": "ontop", "args": ["a", "floor"], "value": True}], "verdicts": {"scorer": True}}
    _bench_job(d, connector_block=_block(), rounds=ROUNDS_CUT[:1], gripper=[], calls=CALL_ROWS + [native_row])
    (d / "job_end.json").write_text('{"ended": 1}')
    g = ladder.gate_carried(Path("/snap"))
    r = g["runs"]["demo"]
    assert [x["call_id"] for x in r["native_rows"]] == ["q1-4"] and g["verdict"]["native_calls"] == {"demo": 1}
    assert g["verdict"]["runs_ended"] == 1 and g["verdict"]["hard_ok"]
    md = ladder.carried_md(g)
    assert "native: `q1-4` place on tiptop@x/place-1 succeeded/None phase home steps 300" in md and "| demo | True |" in md


def test_the_queue_launches_only_on_the_allowed_cards(monkeypatch):
    assert ladder.parse_gpus("3") == (3,) and ladder.parse_gpus("1,3") == (1, 3)
    for bad in ("2", "3,4", "0", ""):
        with pytest.raises(ValueError):
            ladder.parse_gpus(bad)
    monkeypatch.setattr(ladder, "GPUS", (1, 3))
    monkeypatch.setattr(ladder.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "0, 90000\n1, 90000\n2, 90000\n3, 60000\n"})())
    assert ladder.pick_gpu() == 1
    monkeypatch.setattr(ladder, "GPUS", ladder.parse_gpus("3"))
    assert ladder.gpu_free() == {3: 60000} and ladder.pick_gpu() == 3, "a card left out is never picked, however free"
