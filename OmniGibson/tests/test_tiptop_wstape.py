"""W4-A2: the websocket tape over a fake socket. A fake tiptop server behind ``b1k.bridge.client.connect`` (metadata
on connect, one request in, a JSON answer out, as the real one); the tape records through it, then replays without it:
the same client code gets the same answers, a changed field names its path in a BaseException that no per-round
``except Exception`` can swallow, VOLATILE (rgb, views[*].rgb) never diverges, replay-log logs and goes on,
replay-live switches to the real connection at the first mismatch or at a forced index, health is patched while the
tape serves, and every legacy request is stamped 2300 + 1000 R + k with k counted per server."""

import json
from pathlib import Path

import numpy as np
import pytest

from b1k.bridge import client as bridge_client
from b1k.bridge.client import TiptopClient, move
from b1k.bridge.protocol import packb, unpackb
from omnigibson.tiptop.host import wstape
from omnigibson.tiptop.host.instruments import StepLedger
from omnigibson.tiptop.host.wstape import (
    VOLATILE,
    Frame,
    TapeDiverged,
    TapeExhausted,
    WsTape,
    diff_fields,
    field_digests,
    is_volatile,
    op_of,
    seed_for,
)

METADATA = {
    "server": "tiptop",
    "robot_type": "r1pro",
    "dof": 8,
    "move_supported": True,
    "skills": ["pick"],
    "reach_supported": True,
    "views_supported": True,
    "gt_detections_supported": True,
    "seed": 2300,
    "modules": {"tiptop": "/snap/tiptop/tiptop/__init__.py", "cutamp": "/snap/tiptop/cutamp/cutamp/__init__.py"},
    "embodiment": {"joint_names": [f"j{i}" for i in range(8)]},
}
PLAN = {"version": "1", "q_init": [0.0] * 8, "steps": [{"type": "gripper", "action": "close", "label": "Pick(jar)"}]}


class FakeWs:
    """One server-side connection: metadata first, then one answer per request."""

    def __init__(self, server):
        self.server, self.n_recv, self.sent, self.closed = server, 0, [], False

    def recv(self, timeout=None, **kw):
        self.n_recv += 1
        if self.n_recv == 1:
            return packb(self.server.metadata)
        request = self.sent[-1]
        self.server.answered += 1
        return json.dumps(self.server.answer(request))

    def send(self, payload, **kw):
        self.sent.append(unpackb(payload))
        self.server.requests.append(self.sent[-1])

    def close(self, *a, **k):
        self.closed = True


class FakeServer:
    def __init__(self, metadata=METADATA):
        self.metadata, self.requests, self.connections, self.answered = dict(metadata), [], 0, 0

    def connect(self, uri, **kw):
        self.connections += 1
        return FakeWs(self)

    def answer(self, request) -> dict:
        if request.get("type") == "move":
            return {"success": True, "plan": [], "goal": request["goal_pose"]}
        if request.get("type") == "skill":  # a skill/1 answer that reached the pipeline
            return {"ok": True, "code": None, "phase": "lift", "seed": request.get("seed")}
        return {
            "success": True,
            "plan": PLAN,
            "error": None,
            "save_dir": f"/out/{len(self.requests)}",
            "seed_seen": request.get("seed"),
            "task": request["task"],
            "objects": {},
        }


def request(task="pick up the jar", q=0.0, rgb=0, depth=1.0, seed=None) -> dict:
    r = {
        "task": task,
        "rgb": np.full((4, 4, 3), rgb, dtype=np.uint8),
        "depth": np.full((4, 4), depth, dtype=np.float32),
        "intrinsics": np.eye(3, dtype=np.float32),
        "world_from_cam": np.eye(4, dtype=np.float32),
        "q_init": np.full(8, q, dtype=np.float32),
        "locked_joints": {"j8": 0.1},
        "gt_labels": ["jar"],
        "gt_masks": np.zeros((1, 4, 4), dtype=bool),
        "views": [
            {
                "name": "left_wrist",
                "rgb": np.full((4, 4, 3), rgb, dtype=np.uint8),
                "depth": np.full((4, 4), depth, dtype=np.float32),
            },
            {
                "name": "right_wrist",
                "rgb": np.full((4, 4, 3), rgb + 1, dtype=np.uint8),
                "depth": np.full((4, 4), depth, dtype=np.float32),
            },
        ],
    }
    if seed is not None:
        r["seed"] = seed
    return r


@pytest.fixture
def fake(monkeypatch):
    """The fake server behind the bridge's connect, and health answering "real" so a patched health shows."""
    server = FakeServer()
    monkeypatch.setattr(bridge_client, "connect", server.connect)
    monkeypatch.setattr(TiptopClient, "health", lambda self, timeout_s=3.0: "real")
    return server


def here(server) -> None:
    """The fake planner runs this checkout's tiptop and cutamp (what a live planner must: _check_live)."""
    root = str(wstape.ROOT)
    server.metadata = {**server.metadata, "modules": {"tiptop": f"{root}/tiptop/tiptop/__init__.py",
                                                      "cutamp": f"{root}/tiptop/cutamp/cutamp/__init__.py"}}


def client(port=8765) -> TiptopClient:
    return TiptopClient("localhost", port, expected_robot_type="r1pro", expected_dof=8)


def a_run(c, rgb=0, q=0.0, depth=1.0):
    """The bench's traffic: fetch_metadata, two plans, a move."""
    meta = c.fetch_metadata()
    r1 = c.plan(request(q=q, rgb=rgb, depth=depth))
    r2 = c.plan(request(task="place the jar", q=q, rgb=rgb))
    m = move(
        c,
        {
            "type": "move",
            "q_init": np.zeros(8, dtype=np.float32),
            "locked_joints": {},
            "room": {},
            "goal_link": "left_eef",
            "goal_pose": [1.0, 2.0, 3.0],
        },
    )
    return meta, r1, r2, m


def recorded(tmp_path, fake, ledger=None) -> Path:
    tape_dir = tmp_path / "tape"
    with WsTape("record", tape_dir, ledger=ledger, log_dir=tmp_path / "rec"):
        a_run(client())
    return tape_dir


# ------------------------------------------------------------------------------------------------- record + replay
def test_the_tape_round_trips_over_the_fake_socket(tmp_path, fake):
    tape_dir = tmp_path / "tape"
    with WsTape("record", tape_dir, log_dir=tmp_path / "rec") as t:
        meta, r1, r2, m = a_run(client())
    assert fake.connections == 4 and t.index == 4 and len(t.dir) == 4 and t.dir.whole
    rows = t.dir.rows()
    assert [r["op"] for r in rows] == ["metadata", "plan", "plan", "move"]
    assert [r["k"] for r in rows] == [None, 1, 2, None] and [r["seed"] for r in rows] == [None, 2301, 2302, None]
    assert rows[1]["owner"] == "unowned" and rows[1]["response_ok"] is True and rows[1]["task"] == "pick up the jar"
    assert "views[1].depth" in rows[1]["fields"] and "q_init" in rows[1]["fields"]
    assert fake.requests[0]["seed"] == 2301 and fake.requests[1]["seed"] == 2302, "the server saw the stamp"
    assert r1["seed_seen"] == 2301 and r2["seed_seen"] == 2302
    assert t.summary()["stamped"] == [
        {"server": "localhost:8765", "k": 1, "seed": 2301},
        {"server": "localhost:8765", "k": 2, "seed": 2302},
    ]
    assert bridge_client.connect == fake.connect, "uninstalled"

    # replay: no server at all, the same client code, the same answers
    def no_server(uri, **kw):
        raise AssertionError("a replay opened a real connection")

    bridge_client.connect = no_server
    with WsTape("replay", tape_dir, log_dir=tmp_path / "rep") as t2:
        assert client().health() is True, "health is patched while the tape serves"
        meta2, s1, s2, m2 = a_run(client())
    assert client().health() == "real", "restored"
    root = str(wstape.ROOT)
    assert meta2 == {**meta, "modules": {"tiptop": f"{root}/tiptop/tiptop/__init__.py",
                                         "cutamp": f"{root}/tiptop/cutamp/cutamp/__init__.py"}}, meta2
    assert t2.summary()["rerooted"] == {"from": "/snap", "to": root}, "the tape's planner, mapped onto this checkout"
    assert m2 == m
    assert s1["seed_seen"] == 2301 and s2["task"] == "place the jar" and s1["plan"]["version"] == PLAN["version"]
    assert t2.index == 4 and t2.mismatches == 0 and t2.live is False
    log = [json.loads(l) for l in (tmp_path / "rep" / "wstape_replay.jsonl").read_text().splitlines()]
    assert [(r["i"], r["op"], r["matched"]) for r in log] == [
        (0, "metadata", True),
        (1, "plan", True),
        (2, "plan", True),
        (3, "move", True),
    ]
    assert bridge_client.connect is no_server


def test_a_replay_mismatch_names_the_field_path_in_a_base_exception(tmp_path, fake):
    tape_dir = recorded(tmp_path, fake)
    with WsTape("replay", tape_dir, log_dir=tmp_path / "rep"):
        c = client()
        c.fetch_metadata()
        with pytest.raises(TapeDiverged) as info:
            c.plan(request(q=0.5))
    e = info.value
    assert isinstance(e, BaseException) and not isinstance(e, Exception), "bench.py:426 swallows every Exception"
    assert (e.path, e.index, e.op) == ("q_init", 1, "plan") and "8 of 8 elements differ" in e.detail
    assert "tape diverged at frame 1 (plan): q_init" in str(e)
    try:  # what a per-round catch sees
        raise e
    except Exception:  # noqa: BLE001
        pytest.fail("an except Exception caught it")
    except BaseException:
        pass


def test_a_nested_mismatch_names_the_view_and_the_field(tmp_path, fake):
    tape_dir = recorded(tmp_path, fake)
    with WsTape("replay", tape_dir, log_dir=tmp_path / "rep"):
        c = client()
        c.fetch_metadata()
        with pytest.raises(TapeDiverged) as info:
            c.plan(request(depth=2.0))
    assert info.value.path == "depth", "the first differing field in the tape's order"
    with WsTape("replay", tape_dir, log_dir=tmp_path / "rep2"):
        c = client()
        c.fetch_metadata()
        r = request()
        r["views"][1]["depth"][0, 0] = 9.0
        with pytest.raises(TapeDiverged) as info:
            c.plan(r)
    assert info.value.path == "views[1].depth" and "1 of 16 elements" in info.value.detail


def test_volatile_rgb_never_diverges_and_is_fixed(tmp_path, fake):
    assert VOLATILE == frozenset({"rgb", "views[*].rgb"})
    assert is_volatile("rgb") and is_volatile("views[0].rgb") and is_volatile("views[7].rgb")
    assert not is_volatile("depth") and not is_volatile("views[0].depth") and not is_volatile("gt_masks")
    tape_dir = recorded(tmp_path, fake)
    with WsTape("replay", tape_dir, log_dir=tmp_path / "rep") as t:
        meta, s1, s2, m = a_run(client(), rgb=77)  # every rgb differs, nothing else
    assert t.mismatches == 0 and s1["seed_seen"] == 2301


def test_a_changed_op_or_an_exhausted_tape_is_a_divergence(tmp_path, fake):
    tape_dir = recorded(tmp_path, fake)
    with WsTape("replay", tape_dir, log_dir=tmp_path / "rep"):
        c = client()
        c.fetch_metadata()
        with pytest.raises(TapeDiverged) as info:
            move(
                c,
                {
                    "type": "move",
                    "q_init": np.zeros(8),
                    "locked_joints": {},
                    "room": {},
                    "goal_link": "x",
                    "goal_pose": [],
                },
            )
    assert info.value.path == "op" and "plan on the tape, move in the replay" in info.value.detail
    with WsTape("replay", tape_dir, log_dir=tmp_path / "rep2"):
        c = client()
        a_run(c)
        with pytest.raises(TapeExhausted) as info:
            c.plan(request())
    assert info.value.path == "<end of tape>" and info.value.index == 4 and isinstance(info.value, TapeDiverged)


def test_replay_log_logs_every_diff_and_never_stops(tmp_path, fake):
    tape_dir = recorded(tmp_path, fake)
    with WsTape("replay-log", tape_dir, log_dir=tmp_path / "rep") as t:
        c = client()
        c.fetch_metadata()
        r = request(q=0.5, depth=3.0)
        r["gt_labels"] = ["jar", "lid"]
        s1 = c.plan(r)
        s2 = c.plan(request(task="place the jar"))
    assert s1["seed_seen"] == 2301 and s2["task"] == "place the jar", "served from the tape whatever the diffs"
    assert t.mismatches == 1 and t.live is False
    log = [json.loads(l) for l in (tmp_path / "rep" / "wstape_replay.jsonl").read_text().splitlines()]
    assert [r["matched"] for r in log] == [True, False, True]
    paths = [d["path"] for d in log[1]["diffs"]]
    assert paths == ["depth", "q_init", "gt_labels", "views[0].depth", "views[1].depth"], "the tape's field order"
    assert log[1]["n_diffs"] == 5 and log[1]["diffs"][2]["detail"] == "length 1 on the tape, 2 in the replay"


def test_replay_live_switches_to_the_real_connection_at_the_first_mismatch(tmp_path, fake):
    here(fake)
    tape_dir = recorded(tmp_path, fake)
    fake.requests.clear()
    fake.connections = 0
    with WsTape("replay-live", tape_dir, log_dir=tmp_path / "rep") as t:
        c = client()
        assert c.health() is True
        c.fetch_metadata()
        s1 = c.plan(request())  # matches: from the tape
        assert fake.connections == 0 and t.live is False and s1["save_dir"] == "/out/1"
        s2 = c.plan(request(task="place the jar", q=0.25))  # differs: live from here on
        assert t.live is True and t.switched_at == 2 and "q_init" in t.switch_reason
        assert fake.connections == 1 and fake.requests[-1]["seed"] == 2302, "the stamp of the replayed stream"
        assert s2["seed_seen"] == 2302 and s2["save_dir"] == "/out/1", "the fake's own count: a real answer"
        assert c.health() == "real", "health goes to the real one after the switch"
        m = move(
            c,
            {
                "type": "move",
                "q_init": np.zeros(8, dtype=np.float32),
                "locked_joints": {},
                "room": {},
                "goal_link": "left_eef",
                "goal_pose": [7.0],
            },
        )
        assert fake.connections == 2 and m["goal"] == [7.0]
    live = t.live_dir
    assert live.whole and [r["op"] for r in live.rows()] == ["plan", "move"]
    assert [r["live"] for r in live.rows()] == [True, True] and live.rows()[0]["seed"] == 2302
    assert t.summary()["stamped"] == [
        {"server": "localhost:8765", "k": 1, "seed": 2301},
        {"server": "localhost:8765", "k": 2, "seed": 2302},
    ]
    log = [json.loads(l) for l in (tmp_path / "rep" / "wstape_replay.jsonl").read_text().splitlines()]
    assert [(r["i"], r["matched"], r["live"]) for r in log] == [(0, True, False), (1, True, False), (2, False, True)], (
        "the served frames, then the frame it switched at (live, with its diffs)")
    assert "q_init" in [d["path"] for d in log[2]["diffs"]]


def test_replay_live_switches_at_the_forced_index_when_everything_matches(tmp_path, fake):
    here(fake)
    tape_dir = recorded(tmp_path, fake)
    fake.connections = 0
    with WsTape("replay-live", tape_dir, live_at=2, log_dir=tmp_path / "rep") as t:
        meta, s1, s2, m = a_run(client())
    assert t.switched_at == 2 and "forced live" in t.switch_reason and t.mismatches == 0
    assert fake.connections == 2, "frames 2 and 3 went to the server"
    assert [r["op"] for r in t.live_dir.rows()] == ["plan", "move"]


def test_seed_stamps_follow_the_formula_per_server_and_replicate(tmp_path, fake):
    assert seed_for(0, 1) == 2301 and seed_for(0, 7) == 2307 and seed_for(2, 3) == 4303
    with WsTape("record", tmp_path / "tape", replicate=2, log_dir=tmp_path / "rec") as t:
        left, right = client(8765), client(8770)
        left.plan(request())
        right.plan(request(task="press"))
        left.plan(request(task="place"))
        right.plan(request(task="press again"))
        left.plan(request(task="release"))
    assert [r["seed"] for r in fake.requests] == [4301, 4301, 4302, 4302, 4303], "k per server"
    assert t.k == {"localhost:8765": 3, "localhost:8770": 2}
    assert [(s, k) for s, k, _ in t.stamped] == [
        ("localhost:8765", 1),
        ("localhost:8770", 1),
        ("localhost:8765", 2),
        ("localhost:8770", 2),
        ("localhost:8765", 3),
    ]


def test_a_seed_the_request_carried_is_replaced_and_logged(tmp_path, fake, caplog):
    with WsTape("record", tmp_path / "tape", log_dir=tmp_path / "rec"):
        client().plan(request(seed=5))
    assert fake.requests[0]["seed"] == 2301 and "carried seed 5" in caplog.text


def test_skill_requests_count_toward_k_when_they_reach_the_pipeline():
    """registry.dispatch: a build refusal (UNSUPPORTED, NOT_VISIBLE; phase None) never reaches _run_pipeline, nor does
    articulate's own solve; a refusal from inside the pipeline (pick's TOO_WIDE, phase perception) was counted."""
    t = WsTape.__new__(WsTape)
    t.k = {}

    def skill(i, name, **answer):
        return Frame(
            i, "skill", server="s", request=packb({"type": "skill", "skill": name}), response=json.dumps(answer)
        )

    ran = skill(0, "pick", ok=True, code=None, phase="lift")
    refused = skill(1, "pick", ok=False, code="unsupported", phase=None)
    unseen = skill(2, "pick", ok=False, code="not_visible", phase=None)
    too_wide = skill(3, "pick", ok=False, code="too_wide", phase="perception")
    own = skill(4, "articulate", ok=True, code=None, phase="pull")
    for f in (ran, refused, unseen, too_wide, own, Frame(5, "reach", server="s"), Frame(6, "move", server="s")):
        t._count_skill(f)
    assert t.k == {"s": 2} and (ran.k, too_wide.k) == (1, 2) and (refused.k, unseen.k, own.k) == (None, None, None)


def test_replay_live_goes_live_when_the_run_asks_past_the_end_of_the_tape(tmp_path, fake):
    here(fake)
    tape_dir = recorded(tmp_path, fake)
    fake.connections = 0
    with WsTape("replay-live", tape_dir, log_dir=tmp_path / "rep") as t:
        c = client()
        a_run(c)
        assert fake.connections == 0 and t.live is False
        s3 = c.plan(request(task="one more"))
    assert t.live is True and t.switched_at == 4 and "<end of tape>" in t.switch_reason and t.mismatches == 1
    assert fake.connections == 1 and s3["seed_seen"] == 2303 and fake.requests[-1]["task"] == "one more"
    assert [r["op"] for r in t.live_dir.rows()] == ["plan"]


class TimingOut(FakeWs):
    def recv(self, timeout=None, **kw):
        if self.n_recv == 1:
            raise TimeoutError("timed out while receiving data")
        return super().recv(timeout, **kw)


def test_an_answer_that_never_came_is_raised_again_in_the_replay(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(fake, "connect", lambda uri, **kw: TimingOut(fake))
    monkeypatch.setattr(bridge_client, "connect", fake.connect)
    with WsTape("record", tmp_path / "tape", log_dir=tmp_path / "rec") as t:
        c = client()
        with pytest.raises(TimeoutError):
            c.plan(request())
    [row] = t.dir.rows()
    assert row["op"] == "plan" and row["recv_error"] == [
        1,
        "builtins",
        "TimeoutError",
        "timed out while receiving data",
    ]
    with WsTape("replay", tmp_path / "tape", log_dir=tmp_path / "rep") as t2:
        with pytest.raises(TimeoutError, match="timed out while receiving data"):
            client().plan(request())
    assert t2.index == 1 and t2.mismatches == 0


def test_frames_are_tagged_with_the_ledger_owner_and_call_id(tmp_path, fake):
    class Sim:
        n_steps, episode_open = 0, True

        def step_env(self, action):
            return {}

    led = StepLedger(Sim())
    led.call_id = "q1-2"
    with WsTape("record", tmp_path / "tape", ledger=led, log_dir=tmp_path / "rec") as t:
        c = client()
        with led.owner("ep.pick"):
            c.plan(request())
        c.plan(request(task="x"))
    rows = t.dir.rows()
    assert [(r["owner"], r["call_id"]) for r in rows] == [("ep.pick", "q1-2"), ("unowned", "q1-2")]
    led.finish()


def test_a_log_tape_holds_digests_only_and_cannot_be_replayed(tmp_path, fake):
    with WsTape("log", tmp_path / "tape", log_dir=tmp_path / "rec") as t:
        a_run(client())
    assert len(t.dir) == 4 and not t.dir.whole
    rows = t.dir.rows()
    assert (
        rows[1]["fields"]["depth"] == rows[2]["fields"]["depth"]
        and rows[1]["fields"]["task"] != rows[2]["fields"]["task"]
    )
    with pytest.raises(FileNotFoundError):
        WsTape("replay", tmp_path / "tape", log_dir=tmp_path / "rep")
    with pytest.raises(FileExistsError):
        WsTape("record", tmp_path / "tape", log_dir=tmp_path / "rec2")


def test_the_state_stream_passes_through_untaped(tmp_path, fake):
    with WsTape("record", tmp_path / "tape", log_dir=tmp_path / "rec") as t:
        ws = bridge_client.connect("ws://localhost:8765")
        unpackb(ws.recv(timeout=10.0))
        ws.send(packb({"type": "sim_scene", "objects": {}}))
        ws.send(packb({"type": "sim_state", "q": np.zeros(8)}))
        ws.close()
        client().plan(request())
    assert [r["op"] for r in t.dir.rows()] == ["plan"] and fake.connections == 2


# ------------------------------------------------------------------------------------------------- the diff
def test_diff_fields_walks_dicts_lists_and_arrays_bit_exactly():
    a = {"x": 1, "y": [1.0, {"z": np.arange(3, dtype=np.float32)}], "s": "a", "b": True}
    assert diff_fields(a, a) == []
    b = {"x": 1, "y": [1.0, {"z": np.arange(3, dtype=np.float64)}], "s": "a", "b": True}
    assert diff_fields(a, b) == [("y[1].z", "dtype float32 on the tape, float64 in the replay")]
    c = {"x": 1, "y": [1.0, {"z": np.arange(4, dtype=np.float32)}], "s": "b", "b": 1}
    assert [p for p, _ in diff_fields(a, c)] == ["y[1].z", "s", "b"]
    assert diff_fields({"k": 1}, {}) == [("k", "missing in the replay")] and diff_fields({}, {"k": 1}) == [
        ("k", "not on the tape")
    ]
    assert diff_fields([1, 2], [1]) == [("<root>", "length 2 on the tape, 1 in the replay")]
    assert diff_fields({"f": float("nan")}, {"f": float("nan")}) == [] and diff_fields({"f": 1.0}, {"f": 1}) == []
    z = np.zeros((2, 2), dtype=np.float32)
    o = z.copy()
    o[1, 1] = -0.0
    assert diff_fields({"m": z}, {"m": o}) == [("m", "0 of 4 elements differ (max abs 0.0)")], (
        "-0.0 is another bit pattern"
    )
    assert diff_fields({"rgb": 1, "views": [{"rgb": 1, "d": 2}]}, {"rgb": 2, "views": [{"rgb": 3, "d": 2}]}) == []


def test_op_of_and_field_digests():
    assert op_of(None) == "metadata" and op_of({"task": "x"}) == "plan" and op_of({"type": "move"}) == "move"
    assert op_of({"type": "sim_state"}) == "stream" and op_of({"type": "skill"}) == "skill"
    d = field_digests(request())
    assert {"task", "rgb", "views[0].rgb", "views[1].depth", "gt_masks"} <= set(d)
    assert d["views[0].depth"] == d["views[1].depth"] and d["views[0].rgb"] != d["views[1].rgb"]
    assert wstape.MODES == ("off", "log", "record", "replay", "replay-log", "replay-live")


# ------------------------------------------------------------------------------------------------- scripts/tape_diff.py
def tape_diff():
    import importlib.util

    from omnigibson.tiptop import bench

    script = Path(bench.__file__).parent / "scripts" / "tape_diff.py"
    spec = importlib.util.spec_from_file_location("tiptop_tape_diff", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_stream_diff_finds_the_first_differing_frame_and_reads_a_replay_log(tmp_path, fake):
    td = tape_diff()
    for name, q in (("a", 0.0), ("b", 0.0), ("c", 0.5)):
        with WsTape("log", tmp_path / name / "wstape", log_dir=tmp_path / name):
            a_run(client(), rgb={"a": 0, "b": 9, "c": 0}[name], q=q)
    same = td.stream_report(td.find_stream(tmp_path / "a"), td.find_stream(tmp_path / "b"))
    assert same["prefix"] == 4 and same["first_divergence"] is None, "rgb is VOLATILE"
    assert same["seeds"] == [[2301, 2302], [2301, 2302]]
    other = td.stream_report(td.find_stream(tmp_path / "a"), td.find_stream(tmp_path / "c"))
    assert (
        other["prefix"] == 1
        and other["first_divergence"]["frame"] == 1
        and other["first_divergence"]["fields"] == ["q_init"]
    )
    tape = recorded(tmp_path, fake)
    with WsTape("replay-log", tape, log_dir=tmp_path / "rep"):
        a_run(client(), q=0.5)
    rep = td.stream_report(("index", tape / "index.jsonl"), td.find_stream(tmp_path / "rep"))
    assert rep["prefix"] == 1 and rep["mismatched_frames"] == [1, 2] and rep["first_divergence"]["fields"] == ["q_init"]
    assert td._volatile("views[3].rgb") and td._volatile("rgb") and not td._volatile("views[3].depth")


def test_the_stream_diff_reads_both_replay_logs_a_short_replay_and_the_live_switch(tmp_path, fake):
    """Two replays of one tape: each side's own prefix, the earlier first divergence of the two (B's here, where
    A matched to the end); a replay that stopped early against its tape's index diverges at <end of stream>; a
    replay-live run's switching frame is in its log, live."""
    td = tape_diff()
    tape = recorded(tmp_path, fake)
    bridge_client.connect = lambda uri, **kw: pytest.fail("a replay opened a real connection")
    with WsTape("replay-log", tape, log_dir=tmp_path / "A"):
        a_run(client())
    with WsTape("replay-log", tape, log_dir=tmp_path / "B"):
        a_run(client(), q=0.5)
    both = td.stream_report(td.find_stream(tmp_path / "A"), td.find_stream(tmp_path / "B"))
    assert both["replays"]["a"]["first_divergence"] is None and both["replays"]["b"]["prefix"] == 1
    assert both["first_divergence"]["frame"] == 1 and both["prefix"] == 1, both
    with WsTape("replay-log", tape, log_dir=tmp_path / "S"):
        client().fetch_metadata()
        client().plan(request())  # stops after two of the tape's four frames
    short = td.stream_report(("index", tape / "index.jsonl"), td.find_stream(tmp_path / "S"))
    assert short["first_divergence"]["fields"] == ["<end of stream>"] and short["first_divergence"]["frame"] == 2
    here(fake)
    bridge_client.connect = fake.connect
    live_tape = tmp_path / "live_tape"
    with WsTape("record", live_tape, log_dir=tmp_path / "lrec"):
        a_run(client())
    with WsTape("replay-live", live_tape, live_at=2, log_dir=tmp_path / "L"):
        a_run(client())
    live = td.stream_report(("index", live_tape / "index.jsonl"), td.find_stream(tmp_path / "L"))
    assert live["live_from"] == 2 and live["prefix"] == 2 and live["first_divergence"]["fields"] == ["<forced live>"]


# ------------------------------------------------------------------------------------------------- the fix pass
def raw_skill(port=8765):
    """A skill/1 request as the bench's native backend sends it: through bridge_client.connect, one frame."""
    ws = bridge_client.connect(f"ws://localhost:{port}")
    ws.recv()
    ws.send(packb({"type": "skill", "skill": "pick", "seed": 7}))
    ans = ws.recv()
    ws.close()
    return ans


def test_a_skill_frame_counts_toward_k_under_the_replays_own_port(tmp_path, fake):
    """k is counted per server as the replay talks to it: a tape recorded on :8765 replayed on :8799 stamps the
    plan after a skill with the tape's k (skill frames count under the connection's server, as _stamp does)."""
    with WsTape("record", tmp_path / "tape", log_dir=tmp_path / "rec") as t:
        c = client()
        c.plan(request())
        raw_skill()
        c.plan(request(task="place the jar"))
    assert [r["k"] for r in t.dir.rows()] == [1, 2, 3] and [r["seed"] for r in t.dir.rows()] == [2301, None, 2303]
    bridge_client.connect = lambda uri, **kw: pytest.fail("a replay opened a real connection")
    with WsTape("replay", tmp_path / "tape", log_dir=tmp_path / "rep") as t2:
        c = client(8799)
        c.plan(request())
        raw_skill(8799)
        c.plan(request(task="place the jar"))
    assert t2.mismatches == 0 and t2.k == {"localhost:8799": 3}, t2.k
    assert t2.summary()["servers"] == {"localhost:8765": "localhost:8799"}


def test_a_replay_that_talks_to_another_planner_than_the_tapes_frame_diverges(tmp_path, fake):
    with WsTape("record", tmp_path / "tape", log_dir=tmp_path / "rec"):
        client(8765).plan(request())
        client(8766).plan(request(task="press"))
    bridge_client.connect = lambda uri, **kw: pytest.fail("a replay opened a real connection")
    with WsTape("replay", tmp_path / "tape", log_dir=tmp_path / "ok"):  # both planners moved: paired, no diff
        client(8799).plan(request())
        client(8800).plan(request(task="press"))
    with pytest.raises(TapeDiverged) as e:
        with WsTape("replay", tmp_path / "tape", log_dir=tmp_path / "bad"):
            client(8799).plan(request())
            client(8799).plan(request(task="press"))  # the right-arm frame, sent to the left planner
    assert e.value.path == "server" and e.value.index == 1


def test_a_metadata_only_mismatch_stops_a_strict_replay(tmp_path, fake):
    with WsTape("record", tmp_path / "tape", log_dir=tmp_path / "rec"):
        client().plan(request())
    bridge_client.connect = lambda uri, **kw: pytest.fail("a replay opened a real connection")
    with pytest.raises(TapeDiverged) as e:
        with WsTape("replay", tmp_path / "tape", log_dir=tmp_path / "rep"):
            client().fetch_metadata()  # the tape's frame 0 is a plan
    assert e.value.path == "op" and e.value.index == 0


def test_a_recorded_exception_comes_back_as_its_class_and_message():
    from websockets.exceptions import ConnectionClosedError

    e = wstape.rebuilt_error("websockets.exceptions", "ConnectionClosedError", "no close frame received or sent")
    assert isinstance(e, ConnectionClosedError) and type(e).__name__ == "ConnectionClosedError"
    assert str(e) == "no close frame received or sent" and f"{type(e).__name__}: {e}".startswith("ConnectionClosedError")
    t = wstape.rebuilt_error("builtins", "TimeoutError", "timed out while receiving data")
    assert type(t) is TimeoutError and str(t) == "timed out while receiving data"
    r = wstape.rebuilt_error("nowhere.at_all", "Gone", "x")
    assert type(r) is RuntimeError and str(r) == "nowhere.at_all.Gone: x"


class SendFails(FakeWs):
    def send(self, payload, **kw):
        if not self.server.failed:
            self.server.failed = True
            raise ConnectionResetError("the planner went away mid-send")
        return super().send(payload, **kw)


def test_a_failed_send_is_taped_raised_again_on_replay_and_its_k_is_given_back(tmp_path, fake):
    fake.failed = False
    fake.connect = lambda uri, **kw: (setattr(fake, "connections", fake.connections + 1), SendFails(fake))[1]
    bridge_client.connect = fake.connect
    with WsTape("record", tmp_path / "tape", log_dir=tmp_path / "rec") as t:
        with pytest.raises(ConnectionResetError):
            client().plan(request())
        client().plan(request(task="again"))
    rows = t.dir.rows()
    assert rows[0]["send_error"] == ["builtins", "ConnectionResetError", "the planner went away mid-send"]
    assert fake.requests[0]["seed"] == 2301 and t.k == {"localhost:8765": 1}, "the server saw one request, k 1"
    bridge_client.connect = lambda uri, **kw: pytest.fail("a replay opened a real connection")
    with WsTape("replay", tmp_path / "tape", log_dir=tmp_path / "rep") as t2:
        with pytest.raises(ConnectionResetError, match="went away"):
            client().plan(request())
        client().plan(request(task="again"))
    assert t2.mismatches == 0 and t2.k == {"localhost:8765": 1}


def test_the_live_planner_must_run_this_checkout_and_is_connected_with_retries(tmp_path, fake, monkeypatch):
    here(fake)
    tape_dir = recorded(tmp_path, fake)
    monkeypatch.setattr(wstape, "LIVE_RETRY_WAIT_S", 0.0)
    refusals = []
    real = fake.connect

    def warming(uri, **kw):  # a fresh planner still warming up refuses twice
        if len(refusals) < 2:
            refusals.append(uri)
            raise ConnectionRefusedError("warming up")
        return real(uri, **kw)

    monkeypatch.setattr(bridge_client, "connect", warming)
    with WsTape("replay-live", tape_dir, live_at=1, log_dir=tmp_path / "live") as t:
        client().fetch_metadata()
        client().plan(request())
    assert len(refusals) == 2 and t.switched_at == 1 and t.summary()["live_imports"][0]["ok"] is True
    fake.metadata = {**fake.metadata, "modules": {"tiptop": "/elsewhere/tiptop/tiptop/__init__.py",
                                                   "cutamp": "/elsewhere/tiptop/cutamp/cutamp/__init__.py"}}
    monkeypatch.setattr(bridge_client, "connect", real)
    with pytest.raises(TapeDiverged) as e:
        with WsTape("replay-live", tape_dir, live_at=1, log_dir=tmp_path / "other"):
            client().fetch_metadata()
            client().plan(request())
    assert e.value.path == "imports"
