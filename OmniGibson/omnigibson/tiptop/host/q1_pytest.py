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
import re
from pathlib import Path
from typing import NamedTuple

import pytest

MODES = ("record", "replay")
Q1 = pytest.StashKey()


class Outcome(NamedTuple):
    ret: object
    exc: BaseException | None

    @property
    def ending(self) -> tuple:
        e = self.exc
        return (self.ret, None if e is None else type(e).__name__, None if e is None else str(e))


class Session:
    def __init__(self, mode: str, tapes: str | None, original):
        self.mode, self.original = mode, original
        self.tapes = Path(tapes) if tapes else None
        self.invoked = self.recorded = self.replayed = 0
        self.node = ""  # the test running now: the tape's instance
        self.k = 0  # tapes recorded for it

    @property
    def gated(self) -> bool:
        want = (self.invoked, self.recorded, self.replayed if self.mode == "replay" else self.invoked)
        return len(set(want)) == 1

    def run(self, strategy, ep):
        from b1k.bridge import strategies
        from b1k.planner.pseudo import tape as tp

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
        recording = tp.Tape(head)
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


def pytest_addoption(parser):
    group = parser.getgroup("q1", "the Runner tape (omnigibson.tiptop.host.q1_pytest)")
    group.addoption(
        "--q1",
        choices=MODES,
        default=None,
        help="run every Runner.run on a TapeRecorder; replay: also replay it offline",
    )
    group.addoption("--q1-tapes", default=None, help="write every recording to this directory")


def pytest_configure(config):
    mode = config.getoption("--q1")
    if not mode:
        return
    from b1k.bridge import strategies

    session = Session(mode, config.getoption("--q1-tapes"), strategies.Runner.run)
    config.stash[Q1] = session

    def run(self, ep):
        return session.run(self, ep)

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
        session.node, session.k = request.node.nodeid, 0
    yield


def pytest_terminal_summary(terminalreporter, config):
    session = config.stash.get(Q1, None)
    if session is not None:
        terminalreporter.write_line(
            f"q1 {session.mode}: invoked {session.invoked}, recorded {session.recorded}, replayed {session.replayed}"
            + (f", tapes in {session.tapes}" if session.tapes else "")
            + ("" if session.gated else " -- GATE FAILED: the counts differ")
        )


def pytest_sessionfinish(session, exitstatus):
    q1 = session.config.stash.get(Q1, None)
    if q1 is not None and not q1.gated:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
