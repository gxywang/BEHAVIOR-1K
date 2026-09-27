"""pytest plugin: the Runner suite under the tape (WEEK4_PLAN §3.6, §5.2 E0; SPEC §7 Q1, gate item 1).

    pytest OmniGibson/tests/test_tiptop_strategies.py -p omnigibson.tiptop.host.q1_pytest --q1=replay

Without ``--q1`` the plugin is inert. With it, every ``Runner.run(ep)`` of the session runs on a TapeRecorder over
the test's own fake. ``--q1=record`` stops there: the unmodified suite under the recorder. ``--q1=replay`` also
deep-copies the Runner before the run, replays the copy against a TapeEpisode over the recording (after a JSON round
trip, so the codec is on the path), and asserts that the replay's tape equals the recording, that the run ended the
same way (its return, or the same exception class and message), and that the tape answered every read. The session
counts the invocations, the recordings and the faithful replays; the terminal summary reports the three and the
session fails when they differ. ``--q1-tapes DIR`` writes every recording as ``<nodeid>.<k>.json``.

W4-D adds ``--q1=direct`` and ``--q1=connector``.
"""

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import pytest

MODES = ("record", "replay", "direct", "connector")
Q1 = pytest.StashKey()


class Outcome(NamedTuple):
    ret: object
    exc: BaseException | None

    @property
    def ending(self) -> tuple:
        e = self.exc
        return (self.ret, None if e is None else type(e).__name__, None if e is None else str(e))


class Session:
    def __init__(self, mode: str, tapes: str | None, original, log: str | None = None):
        self.mode, self.original = mode, original
        self.tapes = Path(tapes) if tapes else None
        self.log = Path(log) if log else None
        self.invoked = self.recorded = self.replayed = 0
        self.node = ""  # the test running now: the tape's instance
        self.k = 0  # tapes recorded for it
        self.runs: list = []  # the E1 log of the test running now, one entry per Runner.run
        self.diags: list = []
        self.logged = 0  # tests logged
        self.totals: Counter = Counter()  # connector: mismatches, violations, extras, misreads, ref fallbacks
        self._inside = False  # the connector session's own Runner.run, under PseudoPlanner

    @property
    def gated(self) -> bool:
        want = (self.invoked, self.recorded, self.replayed if self.mode == "replay" else self.invoked)
        return len(set(want)) == 1 and not (self.totals["mismatches"] or self.totals["violations"])

    def begin(self, node: str) -> None:
        self.node, self.k, self.runs, self.diags = node, 0, [], []

    def end(self) -> None:
        """Write the test's E1 log (every test, whether it ran the Runner or not)."""
        if self.log is None or self.mode not in ("direct", "connector"):
            return
        self.log.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^\w.-]+", "_", self.node)
        (self.log / f"{name}.json").write_text(json.dumps({"node": self.node, "runs": self.runs}, indent=1,
                                                         sort_keys=True))
        if self.mode == "connector":
            (self.log / f"{name}.diag").write_text(json.dumps(self.diags, indent=1, sort_keys=True, default=str))
        self.logged += 1

    def summary(self) -> dict:
        return {"mode": self.mode, "invoked": self.invoked, "recorded": self.recorded, "replayed": self.replayed,
                "tests_logged": self.logged, **{k: v for k, v in sorted(self.totals.items())}}

    def run(self, strategy, ep):
        from b1k.bridge import strategies
        from b1k.planner.pseudo import tape as tp
        from omnigibson.tiptop.host import q1_harness as qh

        if self._inside:  # PseudoPlanner calling the Runner it was handed
            return self.original(strategy, ep)
        self.invoked += 1
        self.k += 1
        copied = copy.deepcopy(strategy) if self.mode == "replay" else None
        sim = getattr(ep, "sim", None)
        head = tp.header(
            task=strategy.spec.task,
            instance=self.node,
            runner="legacy",
            profile="fake",
            replicate=0,
            strategy=strategy,
            arms=getattr(ep, "arms", None),
            floor=getattr(ep, "floor", None),
            max_steps=getattr(sim, "max_steps", None),
        )
        head["construction"] = qh.construction(strategy)  # what G2 rebuilds the Runner from
        recording = tp.Tape(head)
        if self.mode in ("direct", "connector"):
            return self._e1(strategy, ep, recording)
        outcome = self._run(strategy, tp.TapeRecorder(ep, recording))
        text = recording.dumps()  # the codec is on the path even when no file is written
        self.recorded += 1
        if self.tapes:
            self.tapes.mkdir(parents=True, exist_ok=True)
            name = re.sub(r"[^\w.-]+", "_", self.node)
            (self.tapes / f"{name}.{self.k}.json").write_text(text)
        if self.mode == "replay":
            offline = tp.TapeEpisode(
                tp.Tape.loads(text), exc_classes=(strategies.Unreachable, strategies.TransferBlocked)
            )
            replay = tp.Tape(dict(head))
            again = self._run(copied, tp.TapeRecorder(offline, replay))
            d = tp.diff(recording, replay)
            assert d is None, f"the replay diverged at record {d.index} ({d.kind}): recorded {d.a!r}; replayed {d.b!r}"
            assert (
                again.ending == outcome.ending
            ), f"the replay ended {again.ending!r}; the run ended {outcome.ending!r}"
            assert not offline.unanswerable, f"reads the tape could not answer: {offline.unanswerable[:5]}"
            self.replayed += 1
        if outcome.exc is not None:
            raise outcome.exc
        return outcome.ret

    def _run(self, strategy, ep) -> Outcome:
        try:
            return Outcome(self.original(strategy, ep), None)
        except Exception as e:
            return Outcome(None, e)

    def _e1(self, strategy, ep, recording):
        """One Runner.run of E1: direct on TapeRecorder(ExecLog(fake)), or through the connector stack."""
        from b1k.planner.pseudo import tape as tp
        from b1k.planner.pseudo.planner import PseudoPlanner
        from omnigibson.tiptop.host import q1_harness as qh

        problems = []
        if self.mode == "direct":
            log = qh.ExecLog(ep)
            outcome = self._run(strategy, tp.TapeRecorder(log, recording))
        else:
            members = qh.members_of(ep)
            with qh.build_episode_connector(ep, members, scope=list(strategy.scope), task=strategy.spec.task) as h:
                planner = PseudoPlanner(runner=strategy, channel=h.channel, members=members,
                                        tape=lambda s: tp.TapeRecorder(h.runner_side(s), recording))
                self._inside = True
                try:
                    outcome = Outcome(planner.run(h.conn), None)
                except Exception as e:
                    outcome = Outcome(None, e)
                finally:
                    self._inside = False
                log, diag = h.execlog, h.diagnostics(planner.shim)
            problems = diag["violations"]
            self.totals["mismatches"] += len(diag["mismatches"])
            self.totals["violations"] += len(problems)
            self.totals["extras"] += sum(diag["extras"].values())
            self.totals["misreads"] += len(diag["misreads"])
            self.totals["ref_fallbacks"] += (diag["shim"] or {}).get("ref_fallbacks", 0)
            self.diags.append(diag)
        self.recorded += 1
        e = outcome.exc
        self.runs.append({
            "calls": qh.encode(getattr(ep, "calls", None)),
            "execlog": qh.executed(log),
            "state": qh.state_of(ep),
            "digest": qh.state_digest(ep),
            "tape": tp.encode(recording.records),
            "ending": {"ret": qh.encode(outcome.ret), "exc": None if e is None else [type(e).__name__, str(e)]},
        })
        if self.tapes:
            self.tapes.mkdir(parents=True, exist_ok=True)
            name = re.sub(r"[^\w.-]+", "_", self.node)
            (self.tapes / f"{name}.{self.k}.json").write_text(recording.dumps())
        if problems:
            raise AssertionError(f"q1 connector: {problems}") from e
        if e is not None:
            raise e
        return outcome.ret


def pytest_addoption(parser):
    group = parser.getgroup("q1", "the Runner tape (omnigibson.tiptop.host.q1_pytest)")
    group.addoption(
        "--q1",
        choices=MODES,
        default=None,
        help="run every Runner.run on a TapeRecorder; replay: also replay it offline",
    )
    group.addoption("--q1-tapes", default=None, help="write every recording to this directory")
    group.addoption("--q1-log", default=None, help="direct/connector: write each test's E1 log to this directory")
    group.addoption("--tapes", default=None, help="the Runner tapes test_tiptop_q1_offline.py replays (G2)")


def pytest_configure(config):
    mode = config.getoption("--q1")
    if not mode:
        return
    from b1k.bridge import strategies

    session = Session(mode, config.getoption("--q1-tapes"), strategies.Runner.run, config.getoption("--q1-log"))
    config.stash[Q1] = session

    def run(self, ep):
        return session.run(self, ep)

    run.__wrapped__ = session.original  # inspect.unwrap finds the Runner's own run (the mutants test runs its own)
    strategies.Runner.run = run


def pytest_unconfigure(config):
    session = config.stash.get(Q1, None)
    if session is not None:
        from b1k.bridge import strategies

        strategies.Runner.run = session.original


@pytest.fixture(autouse=True)
def _q1_node(request):
    session = request.config.stash.get(Q1, None)
    if session is not None:
        session.begin(request.node.nodeid)
    yield
    if session is not None:
        session.end()


def pytest_terminal_summary(terminalreporter, config):
    session = config.stash.get(Q1, None)
    if session is not None:
        t = session.totals
        terminalreporter.write_line(
            f"q1 {session.mode}: invoked {session.invoked}, recorded {session.recorded}, replayed {session.replayed}"
            + (f", tests logged {session.logged}" if session.mode in ("direct", "connector") else "")
            + (f", mismatches {t['mismatches']}, violations {t['violations']}, extra reads {t['extras']}, misreads "
               f"{t['misreads']}" if session.mode == "connector" else "")
            + (f", tapes in {session.tapes}" if session.tapes else "")
            + (f", logs in {session.log}" if session.log else "")
            + ("" if session.gated else " -- GATE FAILED")
        )
        if session.log is not None:
            session.log.mkdir(parents=True, exist_ok=True)
            (session.log / "_session.json").write_text(json.dumps(session.summary(), indent=1, sort_keys=True))


def pytest_sessionfinish(session, exitstatus):
    q1 = session.config.stash.get(Q1, None)
    if q1 is not None and not q1.gated:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
