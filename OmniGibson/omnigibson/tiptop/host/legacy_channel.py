"""The side channel between the pseudo planner's shim and the episode host (WEEK4_PLAN §3.3, W4-B/W4-C contract).

The shim puts the Runner's literal Episode call under the typed SkillCall's id; the host backend takes it, executes
it, and records what came back before any post-processing; the shim reads the outcome and returns the literal
value or re-raises the same exception object. Navigation is a FIFO of stand_for / walk_to_floor entries. The
channel is created by the host and injected into the shim, which never imports it. An entry is
``{"method": str, "args": tuple, "kwargs": dict}``, exactly the Runner's call, nothing added or dropped."""

from collections import Counter, deque
from typing import Any, Optional


class LegacyChannel:
    def __init__(self):
        self._entries: dict[str, dict] = {}
        self._outcomes: dict[str, dict] = {}
        self._nav: deque = deque()
        self._nav_outcome: Optional[dict] = None
        self.mismatches: list = []  # typed/literal disagreements the rebuild check found
        self.exc_counts: Counter = Counter()  # exceptions that came back, by type name

    # -- skill calls ------------------------------------------------------------------------------------------------
    def put(self, call_id: str, entry: dict) -> None:
        self._entries[call_id] = entry

    def take(self, call_id: str) -> Optional[dict]:
        return self._entries.pop(call_id, None)

    def done(self, call_id: str, executed: bool, value: Any, exc: Optional[BaseException]) -> None:
        self._outcomes[call_id] = {"executed": bool(executed), "value": value, "exc": exc}
        if exc is not None:
            self.exc_counts[type(exc).__name__] += 1

    def outcome(self, call_id: str) -> Optional[dict]:
        """None if the host never executed the call, else {"executed": True, "value": ..., "exc": ...}."""
        out = self._outcomes.get(call_id)
        return out if out is not None and out["executed"] else None

    # -- navigation (a FIFO) ----------------------------------------------------------------------------------------
    def put_nav(self, entry: dict) -> None:
        self._nav.append(entry)

    def take_nav(self) -> dict:
        return self._nav.popleft()

    def nav_done(self, executed: bool, value: Any, exc: Optional[BaseException]) -> None:
        self._nav_outcome = {"executed": bool(executed), "value": value, "exc": exc}
        if exc is not None:
            self.exc_counts[type(exc).__name__] += 1

    def nav_outcome(self) -> Optional[dict]:
        out = self._nav_outcome
        return out if out is not None and out["executed"] else None

    # -- records ----------------------------------------------------------------------------------------------------
    def mismatch(self, call_id: str, rebuilt: Any, literal: Any) -> None:
        self.mismatches.append({"call_id": call_id, "rebuilt": rebuilt, "literal": literal})
