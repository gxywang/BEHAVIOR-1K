"""G2 (WEEK4_PLAN §5.3; SPEC §7 Q1): every Runner tape replayed offline two ways, with no simulator and no planner.

    pytest OmniGibson/tests/test_tiptop_q1_offline.py -p omnigibson.tiptop.host.q1_pytest --tapes DIR [--q1-log DIR]
    Q1_TAPES=DIR pytest OmniGibson/tests/test_tiptop_q1_offline.py

For each tape under DIR (``*.json``, recursively) the Runner is rebuilt from the header's construction inputs
(q1_harness.rebuild_runner: strategy_for, or the recorded spec) and run
  E0         directly against a TapeEpisode, and
  connector  as Runner -> TapeRecorder(shim) -> q1_harness's stack -> ExecLog -> TapeEpisode.
Both recordings must equal the tape, with every write served in order and no read left unanswerable (every read
that reaches the TapeEpisode is one a Runner member asked); the connector replay also needs 0 typed/literal
mismatches, no entry left on the channel, no Runtime or env step, and the same ending as E0. The connector's own
extra reads (the post-call judge, the advisory precheck, the dwell clip) are answered from the tape without being
consumed, counted and written to ``--q1-log`` when given, never failed on. Without tapes the test is skipped.

The unit tests below the tape test need no tapes: a doctored tape is refused, and a header without the options
cannot be rebuilt."""

import json
import os
import re
from pathlib import Path

import pytest

from b1k.planner.pseudo import tape as tp
from omnigibson.tiptop.host import q1_harness as qh


def _option(config, name):
    try:
        return config.getoption(name)
    except ValueError:  # the q1 plugin is not loaded: its options do not exist
        return None


def pytest_generate_tests(metafunc):
    if "tape_path" not in metafunc.fixturenames:
        return
    where = _option(metafunc.config, "--tapes") or os.environ.get("Q1_TAPES")
    paths = sorted(Path(where).rglob("*.json")) if where else []
    metafunc.parametrize("tape_path", paths or [None], ids=[p.name for p in paths] or ["no-tapes"])


def test_the_tape_replays_the_same_directly_and_through_the_connector_stack(tape_path, request):
    if tape_path is None:
        pytest.skip("no tapes: --tapes DIR (with -p omnigibson.tiptop.host.q1_pytest) or Q1_TAPES=DIR")
    r = qh.replay(tp.Tape.load(tape_path))
    log = _option(request.config, "--q1-log")
    if log:
        Path(log).mkdir(parents=True, exist_ok=True)
        (Path(log) / re.sub(r"[^\w.-]+", "_", f"g2_{tape_path.name}")).write_text(json.dumps(r, indent=1, default=str))
    assert "error" not in r, r["error"]
    assert r["e0"]["ok"], f"E0: {r['e0']}"
    c = r["connector"]
    assert c["diff"] is None, f"the connector replay's tape differs from the recording at {c['diff']}"
    assert c["served"] == r["writes"], f"{c['served']} of {r['writes']} writes served"
    assert not c["unanswerable"], f"reads by Runner members the tape could not answer: {c['unanswerable'][:5]}"
    assert not c["mismatches"], f"typed/literal mismatches: {c['mismatches']}"
    assert not c["violations"], c["violations"]
    assert c["ending"] == r["e0"]["ending"], (c["ending"], r["e0"]["ending"])
    assert c["ok"]


# ------------------------------------------------------------------------------------------------ no tapes needed
class _Ep:
    """A small Episode: the apple is picked and put in the basket."""

    def __init__(self):
        self.floor, self.hand, self.done, self.calls = "floor.n.01_1", None, False, []

    def is_floor(self, name):
        return bool(name) and name.startswith("floor.")

    def holding(self, bddl):
        return self.hand == bddl

    def held_names(self):
        return [self.hand] if self.hand else []

    def support_of(self, item):
        return "table.n.02_1"

    def edge_gap(self, item, support):
        return 0.0

    def distance(self, a, b):
        return 1.0

    def goal_already_holds(self, predicate, item, container=None):
        return self.done

    def is_shut(self, name):
        return False

    def stand_for(self, *names):
        self.calls.append(("stand_for", names))

    def pick(self, bddl, into=None):
        self.calls.append(("pick", bddl, into))
        self.hand = bddl
        return True

    def achieve(self, atoms, arm="left", floor=None):
        self.calls.append(("achieve", atoms[0]["args"]))
        self.hand, self.done = None, True
        return True


def _record():
    from b1k.bridge.strategies import atom, strategy_for

    runner = strategy_for("t", [atom("inside", "apple.n.01_1", "basket.n.01_1")], attempts=1)
    t = tp.Tape(tp.header("t", "unit", "legacy", "fake", 0, strategy=runner, floor="floor.n.01_1"))
    t.header["construction"] = qh.construction(runner)
    runner.run(tp.TapeRecorder(_Ep(), t))
    return tp.Tape.loads(t.dumps())


def test_a_tape_the_bench_probed_replays_both_ways():
    """bench.py's TapeRecorder takes the step and the state digest around every write; the replays' own recorders
    have no sim to probe. G2 compares the records without those diagnostics (G4 compares them) and says how many
    records carried them."""
    import itertools

    from b1k.bridge.strategies import atom, strategy_for

    runner = strategy_for("t", [atom("inside", "apple.n.01_1", "basket.n.01_1")], attempts=1)
    t = tp.Tape(tp.header("t", "unit", "legacy", "parity", 0, strategy=runner, floor="floor.n.01_1"))
    clock = itertools.count(0, 40)
    runner.run(tp.TapeRecorder(_Ep(), t, step_probe=lambda: next(clock), digest=lambda: {"robot": "x"}))
    t = tp.Tape.loads(t.dumps())
    assert all("step" in w and "digest" in w for w in t.writes)
    r = qh.replay(t)
    assert r["ok"] and r["probed_records"] == len(t.writes), r


class _Diverging(_Ep):
    """A strict --wstape replay stops at the pick: TapeDiverged, a BaseException, on the write in flight."""

    def pick(self, bddl, into=None):
        from omnigibson.tiptop.host.wstape import TapeDiverged

        self.calls.append(("pick", bddl, into))
        raise TapeDiverged("depth", 3, "476754 of 518400 elements differ", "plan")


def test_a_tape_that_ends_in_a_tape_divergence_replays_both_ways():
    """bench.py records the TapeDiverged on the write in flight (its TapeRecorder names it in exc_classes): the
    replays tape it too, and both end the way the recording did."""
    from b1k.bridge.strategies import atom, strategy_for
    from omnigibson.tiptop.host.wstape import TapeDiverged

    runner = strategy_for("t", [atom("inside", "apple.n.01_1", "basket.n.01_1")], attempts=1)
    t = tp.Tape(tp.header("t", "unit", "connector", "parity", 0, strategy=runner, floor="floor.n.01_1"))
    with pytest.raises(TapeDiverged):
        runner.run(tp.TapeRecorder(_Diverging(), t, exc_classes=(TapeDiverged,)))
    t = tp.Tape.loads(t.dumps())
    assert t.writes[-1]["exc"].type == "TapeDiverged"
    r = qh.replay(t)
    assert r["ok"], r
    assert (
        r["e0"]["ending"]
        == r["connector"]["ending"]
        == ["TapeDiverged", str(TapeDiverged("depth", 3, "476754 of 518400 elements differ", "plan"))]
    )


def test_a_recorded_tape_replays_the_same_both_ways():
    r = qh.replay(_record())
    assert r["writes"] >= 2 and r["ok"], r
    assert r["connector"]["calls"], "the connector replay ran its writes as skill calls"


def test_a_tape_whose_answer_was_changed_is_refused():
    """A replay that serves the tape faithfully must notice a tape the Runner could not have produced: the pick
    reads as failed, so the Runner retries or moves on, and the tape's next write is not the one it asks for."""
    t = _record()
    pick = next(r for r in t.records if r["kind"] == "write" and r["member"] == "pick")
    pick["ret"] = False
    r = qh.replay(t)
    assert not r["ok"] and not r["connector"]["ok"], r


def test_a_header_in_the_bench_form_is_rebuilt_from_its_options():
    """bench.py --runner-tape writes the goal options and the scope beside ``inputs`` (no construction block): two
    options, so the goal alone would not rebuild it."""
    from b1k.bridge.strategies import atom, strategy_for

    goal = [atom("inside", "apple.n.01_1", "basket.n.01_1")]
    options = [goal, [atom("inside", "apple.n.01_1", "basket.n.01_2")]]
    runner = strategy_for("t", goal, options=options, attempts=1, scope=["apple.n.01_1"])
    t = tp.Tape(tp.header("t", "unit", "legacy", "parity", 0, strategy=runner, floor="floor.n.01_1"))
    t.header["options"], t.header["scope"] = options, list(runner.scope)
    runner.run(tp.TapeRecorder(_Ep(), t))
    t = tp.Tape.loads(t.dumps())
    assert t.header["inputs"]["n_options"] == 2 and "construction" not in t.header
    r = qh.replay(t)
    assert r["ok"], r


def test_a_header_without_the_options_is_not_rebuilt():
    t = _record()
    del t.header["construction"], t.header["options"]  # tape.header() carries the options since the fix pass
    t.header["inputs"] = {**t.header["inputs"], "n_options": 2, "options_sha256": "0" * 64}
    r = qh.replay(t)
    assert not r["ok"] and "construction" in r["error"], r


# ------------------------------------------------------------------------------------------------- the fix pass
class FloorEp:
    """An Episode with a stance_key and a floor put_down that fails, re-stancing inside it (Episode.put_down adds
    the key it stands at AFTER the achieve to floor_failed_at, bench.py put_down)."""

    floor = "floor.n.01_1"

    def __init__(self, key=(10, -3, 5)):
        self.key, self.calls = key, []

    def is_floor(self, name):
        return str(name).startswith("floor.")

    def stance_key(self):
        return self.key

    def holding(self, bddl):
        return True

    def put_down(self, bddl, support, floor=None):
        self.calls.append(("put_down", bddl, support))
        self.key = (self.key[0] + 7, self.key[1] - 2, self.key[2] + 1)  # the achieve inside re-stanced
        self.__dict__.setdefault("floor_failed_at", set()).add(self.stance_key())
        return False


def test_the_worlds_base_pose_round_trips_the_episodes_stance_key_through_the_shim():
    """EpisodeWorld.base_pose is the Episode's stance_key at 10 cm and 15 degrees; the shim's stance_key recovers
    the key exactly (a Pose2(k0, k1, k2) without the scaling, or a None pose, would not)."""
    from b1k.planner.pseudo.shim import EpisodeOverConnector

    for key in ((10, -3, 5), (0, 0, 0), (-123, 45, -11), (7, 7, 12)):
        ep = FloorEp(key)
        with qh.build_episode_connector(ep, qh.members_of(ep)) as h:
            assert EpisodeOverConnector(h.conn, h.channel).stance_key() == key


def test_a_failed_floor_put_down_leaves_the_shims_floor_failed_at_equal_to_the_episodes():
    """The shim adds the stance it reads after the put_down: the key the Episode's put_down re-stanced to, as the
    Episode's own floor_failed_at holds it."""
    from b1k.planner.pseudo.shim import EpisodeOverConnector

    ep = FloorEp()
    with qh.build_episode_connector(ep, qh.members_of(ep)) as h:
        shim = EpisodeOverConnector(h.conn, h.channel, members=qh.members_of(ep) | {"floor_failed_at"})
        assert shim.put_down("plywood.n.01_1", ep.floor) is False
        assert shim.floor_failed_at == ep.floor_failed_at == {(17, -5, 6)}
        assert shim.put_down("plywood.n.01_1", ep.floor) is False
        assert shim.floor_failed_at == ep.floor_failed_at == {(17, -5, 6), (24, -7, 7)}
        assert h.channel.mismatches == []


def test_the_harness_world_turns_an_unperceived_distance_into_none_as_the_oracle_world_does():
    """OracleWorld.distance maps the Episode's KeyError to a None belief and the shim raises the Episode's KeyError
    back (by the name without a box): the harness's world must convert the same way, or E1 and G2 never run the
    shim's None path production takes."""
    from b1k.planner.pseudo.shim import EpisodeOverConnector

    class Far(_Ep):
        def distance(self, a, b):
            if self.is_floor(a) or self.is_floor(b):
                raise KeyError(a if self.is_floor(a) else b)
            return 1.0

    ep = Far()
    with qh.build_episode_connector(ep, qh.members_of(ep)) as h:
        assert h.world.distance(qh.ref("floor.n.01_2"), qh.ref("plywood.n.01_1")).value is None
        with pytest.raises(KeyError) as e:
            EpisodeOverConnector(h.conn, h.channel).distance("plywood.n.01_1", "floor.n.01_2")
    with pytest.raises(KeyError) as legacy:
        ep.distance("plywood.n.01_1", "floor.n.01_2")
    assert e.value.args == legacy.value.args == ("floor.n.01_2",)


def test_peek_never_answers_a_stance_key_across_a_write():
    """A read the connector makes on its own over a tape (an extra) is answered from its own segment. A stance_key with
    no record there reads None and is listed as stale, never the key from before a write that may have re-stanced
    (a failed floor put_down's achieve does, and the shim's floor_failed_at would then take the old stance)."""

    class Ep:
        floor = "floor.n.01_1"

        def __init__(self):
            self.key = (10, -3, 5)

        def stance_key(self):
            return self.key

        def holding(self, bddl):
            return True

        def put_down(self, bddl, support, floor=None):
            self.key = (17, -5, 6)  # the achieve inside re-stanced
            return False

    tape = tp.Tape(tp.header("t", 301, "connector", "parity", 0))
    rec = tp.TapeRecorder(Ep(), tape)
    rec.stance_key()
    rec.put_down("plywood.n.01_1", "floor.n.01_1")
    rec.holding("plywood.n.01_1")
    te = tp.TapeEpisode(tp.Tape.loads(tape.dumps()))
    src = qh.Src(te, qh.members_of_tape(tape), tape=tape)
    assert src.peek("stance_key") == (10, -3, 5), "its own segment answers"
    assert te.put_down("plywood.n.01_1", "floor.n.01_1") is False  # the write: the next segment
    assert src.peek("stance_key") is None, "the key from before the write is not this segment's"
    assert [(s["member"], s["segment"]) for s in src.stale] == [("stance_key", 1)]
    assert src.peek("holding", ("plywood.n.01_1",)) is True, "another member is answered from its segment"
