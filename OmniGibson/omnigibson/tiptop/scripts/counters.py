"""The ladder's counters (WEEK4_PLAN 5.8): one definition per counter for both runners, and the compare mode.

    python OmniGibson/omnigibson/tiptop/scripts/counters.py extract JOB [JOB ...] [--json]
    python OmniGibson/omnigibson/tiptop/scripts/counters.py compare --skill place.on \\
        --pair TASK C0,C1,C2 T0,T1,T2 [--pair ...] [--carried TASK DIR ...] [--alpha 0.10] [--delta 0.25] [--json]

Needs no simulator, no GPU and nothing outside the standard library: it reads a bench job dir. A counter is
extracted the same way whichever runner produced the job, from these sources, in this order of authority.

Legacy sources (``--runner legacy``, every historical job):
- the result JSON ``episode/json/<task>_<instance>_<rollout>.json``: ``steps``, ``bench.rounds``, ``bench.teleports``
  and ``bench.goal``;
- ``sim.log``: the ``round N <atoms> [arm]: executed|<error> (Ns)`` lines, which lack the cut-off round (the JSON
  carries it, so the JSON is the authority and the log is the cross-check); the executor's
  ``gripper close: ... is_grasping=K`` lines, anchored on ``is_grasping=`` because a bare ``gripper close`` also
  matches the bench's ``gripper closed at the start`` (they are counted over the whole log, so the cut-off round's
  own close is in); and the ``RESULT instance`` line.

Connector sources (``--runner connector``), a schema this module defines; every file is optional and lives in the
job dir (or under ``episode/``):
- ``skill_calls.jsonl``: one ``to_dict(SkillResult)`` row per finished skill call, as the Runtime logs it, plus an
  optional ``trial``. A row is LEGACY when ``backend == "legacy"`` or ``requires_sim_clock`` is true; its
  ``evidence.records`` are the Episode records the call appended (round records with ``env_steps``/``error``, and
  ``open``/``close`` records with ``opened``/``reached``), and they are counted exactly as ``bench.rounds`` are.
  Any other row is NATIVE: it counts one executed motion when ``steps > 0``, plus ``evidence.resamples`` (an int,
  default 0, one per resample inside the call). The Runner-visible return is ``evidence.legacy_ok`` for a legacy row
  and ``status == "succeeded"`` for a native one; ``verdicts.scorer`` is the scorer's verdict right after.
- ``ledger.jsonl``: one row per StepLedger owner: ``{"owner": str, "steps": int, "env_step_calls": int,
  "place_robot": int, "capture": int, "look_at": int, "writes": int}``. A ``place_robot`` is one teleport; the
  ``go_to`` owner's are the planner's, owners named ``ep.<member>`` are teleports inside a legacy call.
- ``gripper.jsonl``: one GripperWatch row per gripper event: ``{"step": int, "arm": "left"|"right",
  "event": "close"|"open", "is_grasping": 1|0|-1, "owner": str, "call_id": str|null}``. A close with
  ``is_grasping == 1`` is a weld and one with ``-1`` an on-air close, whose cause is its ``owner`` and ``call_id``.
  Without this file the executor lines in ``sim.log`` are used, and an on-air close has no owner to name.
- ``runner_tape.jsonl`` (optional): ``{"call_id": str, "skill": str, "qual": str|null, "returned": bool|null}``
  per Runner write, joining a skill row to the Runner's typed call; it qualifies ``place.on`` against ``place.in``.
  Without it the qualifier is read from the row's ``effects`` predicates, and a row with none stays unqualified.
- the result JSON's ``connector`` block: ``step``, ``idle_steps``, ``charged`` (steps per run kind), ``ledger``
  (the same rows as ledger.jsonl, keyed by owner; the fallback when the file is missing) and ``live_at_end``
  (``{"call_id": str, "steps": int}`` when EpisodeOver ended a live native run: the cut-off round).

The counters (WEEK4_PLAN 5.8), each with its direction and what it is normalised by when compared. When any run of a
task delivered no atom, that task's counters are compared raw.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------------------------------------------- the counter table
HIGHER, LOWER, REPORTED, CAUSE = "higher", "lower", "reported", "cause"
ATOMS, ITEMS = "atoms", "items"


@dataclass(frozen=True)
class CounterSpec:
    name: str
    norm: Optional[str]  # ATOMS (delivered goal atoms), ITEMS (delivered items) or None (raw)
    worse: str  # HIGHER, LOWER, REPORTED (never flagged) or CAUSE (any > 0 needs a named cause)
    doc: str


COUNTERS: tuple[CounterSpec, ...] = (
    CounterSpec("executed", ATOMS, HIGHER, "legacy rounds with env_steps > 0 and no error, plus native runs with "
                "steps > 0 (+1 per resample)"),
    CounterSpec("cut_off", None, REPORTED, "1 if EpisodeOver ended a live round"),
    CounterSpec("rejections", ATOMS, HIGHER, "legacy rounds with env_steps == 0 (motion validation rejected)"),
    CounterSpec("welds", ITEMS, HIGHER, "gripper closes with is_grasping=1"),
    CounterSpec("on_air", None, CAUSE, "gripper closes with is_grasping=-1"),
    CounterSpec("open_attempts", None, HIGHER, "open attempts (legacy open records, native open calls)"),
    CounterSpec("open_rate", None, LOWER, "open successes / open attempts (None without an attempt)"),
    CounterSpec("close_attempts", None, HIGHER, "close attempts (legacy close records, native close calls)"),
    CounterSpec("close_rate", None, LOWER, "close successes / close attempts (None without an attempt)"),
    CounterSpec("teleports", ATOMS, HIGHER, "base teleports: go_to plus those inside ep.*"),
    CounterSpec("teleports_go_to", ATOMS, HIGHER, "teleports the planner asked for (go_to)"),
    CounterSpec("teleports_ep", ATOMS, HIGHER, "teleports inside legacy ep.* calls"),
    CounterSpec("steps", ATOMS, HIGHER, "sim.n_steps at the end"),
    CounterSpec("delivered_atoms", None, REPORTED, "goal atoms true at the end (the score)"),
)
SPEC = {c.name: c for c in COUNTERS}

# ------------------------------------------------------------------------------------------------- the log regexes
# Anchored on 'is_grasping=': the bench logs 'executing plan: ... gripper closed at the start', which a bare
# 'gripper close' also matches (the mutant the pins refuse).
CLOSE_RE = re.compile(r"gripper close: .*is_grasping=(-?\d+)")
ROUND_RE = re.compile(r"round (\d+) (.+?) \[(\w+)\]: (.+?) \((\d+(?:\.\d+)?)s\)\s*$")
RESULT_RE = re.compile(r"RESULT instance (\d+): q_score (\S+) success (\w+) steps (\d+)/(\d+) \((.*?)\); "
                       r"teleports (\d+); (\d+(?:\.\d+)?)s")
PLACE_PREDS = {"ontop": "on", "on": "on", "inside": "in", "in": "in", "nextto": "next_to", "next_to": "next_to",
               "under": "under", "touching": "touching", "attached": "attached"}


@dataclass
class CallOutcome:
    """One switched-skill call for the pooled per-call test."""

    call_id: str
    skill: str
    qual: Optional[str]  # place.on -> "on"; None when unqualified
    backend: str
    returned: Optional[bool]  # the Runner-visible return
    scorer: Optional[bool]  # the scorer's verdict right after
    trial: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.returned is True and self.scorer is True


@dataclass
class Counters:
    job: str
    task: str = ""
    runner: str = "legacy"  # "legacy" | "connector"
    reason: str = ""
    executed: int = 0
    cut_off: int = 0
    rejections: int = 0
    welds: int = 0
    on_air: int = 0
    on_air_causes: list = field(default_factory=list)  # [{"step", "arm", "owner", "call_id"}] or "unknown ..."
    open_attempts: int = 0
    open_successes: int = 0
    close_attempts: int = 0
    close_successes: int = 0
    teleports: int = 0
    teleports_go_to: int = 0
    teleports_ep: int = 0
    steps: int = 0
    delivered_atoms: int = 0
    delivered_new: int = 0
    goal_total: int = 0
    delivered_items: int = 0
    native_calls: int = 0
    native_by: dict = field(default_factory=dict)  # "skill@backend" -> count
    calls: list = field(default_factory=list)  # CallOutcome rows (connector runs)
    notes: list = field(default_factory=list)  # cross-check disagreements and missing sources

    @property
    def open_rate(self) -> Optional[float]:
        return None if not self.open_attempts else self.open_successes / self.open_attempts

    @property
    def close_rate(self) -> Optional[float]:
        return None if not self.close_attempts else self.close_successes / self.close_attempts

    def value(self, name: str):
        return getattr(self, name)

    def normalised(self, name: str) -> Optional[float]:
        """The counter divided by what its spec names, or raw when that divisor is 0 (compare mode decides per task
        whether to normalise at all)."""
        v = self.value(name)
        if v is None:
            return None
        norm = SPEC[name].norm
        div = {ATOMS: self.delivered_atoms, ITEMS: self.delivered_items, None: 0}[norm]
        return v / div if div else float(v)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["calls"] = [dataclasses.asdict(c) | {"ok": c.ok} for c in self.calls]
        d["open_rate"], d["close_rate"] = self.open_rate, self.close_rate
        return d


# ------------------------------------------------------------------------------------------------- reading a job dir
def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _find(job: Path, name: str) -> Optional[Path]:
    for p in (job / name, job / "episode" / name):
        if p.exists():
            return p
    return None


def result_json(job: Path, instance: Optional[int] = None) -> Optional[Path]:
    """The bench's result JSON: episode/json/<task>_<instance>_<rollout>.json (one per instance)."""
    files = sorted((job / "episode" / "json").glob("*.json"))
    if instance is not None:
        files = [f for f in files if f.stem.split("_")[-2] == str(instance)]
    return files[0] if files else None


def rounds_from_log(text: str) -> list[dict]:
    """The 'round N atoms [arm]: verdict (Ns)' lines; the cut-off round has none (the JSON carries it)."""
    out = []
    for line in text.splitlines():
        m = ROUND_RE.search(line)
        if m:
            out.append({"round": int(m.group(1)), "atoms": m.group(2), "arm": m.group(3),
                        "verdict": m.group(4), "seconds": float(m.group(5))})
    return out


def closes_from_log(text: str, pattern: re.Pattern = None) -> list[int]:
    """Every executor gripper close's is_grasping value, over the whole log (the cut-off round's included)."""
    pattern = pattern or CLOSE_RE
    return [int(m.group(1)) if m.groups() else 1 for m in (pattern.search(l) for l in text.splitlines()) if m]


def result_line(text: str) -> Optional[dict]:
    for line in text.splitlines():
        m = RESULT_RE.search(line)
        if m:
            return {"instance": int(m.group(1)), "q_score": m.group(2), "success": m.group(3) == "True",
                    "steps": int(m.group(4)), "max_steps": int(m.group(5)), "reason": m.group(6),
                    "teleports": int(m.group(7)), "wall_s": float(m.group(8))}
    return None


def _count_records(c: Counters, records: list[dict]) -> None:
    """Episode records: round records (executed / rejected / cut off) and open / close records. The same for
    bench.rounds and for a legacy skill row's evidence.records."""
    for r in records:
        if "round" in r:
            err = r.get("error")
            if err == "episode over":
                c.cut_off += 1
            elif r.get("env_steps") == 0:
                c.rejections += 1
            elif err is None and (r.get("env_steps") or 0) > 0:
                c.executed += 1
        elif "open" in r:
            c.open_attempts += 1
            c.open_successes += bool(r.get("opened"))
        elif "close" in r:
            c.close_attempts += 1
            c.close_successes += bool(r.get("reached"))


def _count_goal(c: Counters, goal: dict) -> None:
    satisfied = goal.get("satisfied") or []
    c.delivered_atoms, c.goal_total = len(satisfied), int(goal.get("total") or 0)
    c.delivered_new = int(goal.get("new") if goal.get("new") is not None else len(satisfied))
    items = set()
    for atom in satisfied:  # 'pred(a, b)' -> the first argument is the item delivered
        m = re.match(r"\s*\w+\((.*)\)\s*$", atom)
        if m and m.group(1).strip():
            items.add(m.group(1).split(",")[0].strip())
    c.delivered_items = len(items)


def _count_closes(c: Counters, values: list[int], causes: Optional[list] = None) -> None:
    c.welds += sum(1 for v in values if v == 1)
    on_air = [i for i, v in enumerate(values) if v == -1]
    c.on_air += len(on_air)
    if causes is None:
        c.on_air_causes += ["unknown: legacy log names no owner"] * len(on_air)
    else:
        c.on_air_causes += [causes[i] for i in on_air]


def _qual(row: dict, tape: dict) -> Optional[str]:
    t = tape.get(row.get("call_id"))
    if t and t.get("qual") is not None:
        return t["qual"]
    quals = {PLACE_PREDS[e["pred"]] for e in row.get("effects") or () if e.get("pred") in PLACE_PREDS}
    return quals.pop() if len(quals) == 1 else None


def _count_skill_rows(c: Counters, rows: list[dict], tape: dict) -> None:
    for row in rows:
        legacy = row.get("backend") == "legacy" or bool(row.get("requires_sim_clock"))
        ev = row.get("evidence") or {}
        if legacy:
            _count_records(c, ev.get("records") or [])
            if row.get("skill") in ("open", "close") and not any(
                    k in r for r in ev.get("records") or [] for k in ("open", "close")):
                c.notes.append(f"{row.get('call_id')}: legacy {row['skill']} without an open/close record")
            returned = ev.get("legacy_ok")
        else:
            c.native_calls += 1
            key = f"{row.get('skill')}@{row.get('backend')}"
            c.native_by[key] = c.native_by.get(key, 0) + 1
            if (row.get("steps") or 0) > 0:
                c.executed += 1 + int(ev.get("resamples") or 0)
            if row.get("skill") == "open":
                c.open_attempts += 1
                c.open_successes += row.get("status") == "succeeded"
            elif row.get("skill") == "close":
                c.close_attempts += 1
                c.close_successes += row.get("status") == "succeeded"
            returned = row.get("status") == "succeeded"
        t = tape.get(row.get("call_id"))
        if t and t.get("returned") is not None:
            returned = bool(t["returned"])
        c.calls.append(CallOutcome(str(row.get("call_id")), str(row.get("skill")), _qual(row, tape),
                                   str(row.get("backend")), returned, (row.get("verdicts") or {}).get("scorer"),
                                   row.get("trial")))


def _count_ledger(c: Counters, owners: dict) -> None:
    for owner, row in owners.items():
        n = int(row.get("place_robot") or 0)
        if owner == "go_to":
            c.teleports_go_to += n
        elif owner.startswith("ep."):
            c.teleports_ep += n
        elif n:
            c.notes.append(f"ledger: {n} place_robot under owner {owner!r} (neither go_to nor ep.*)")
    c.teleports = c.teleports_go_to + c.teleports_ep


def extract(job: Path, instance: Optional[int] = None) -> Counters:
    """Every counter of one job dir, from the sources the module docstring lists."""
    job = Path(job)
    c = Counters(job=str(job))
    rj = result_json(job, instance)
    data = json.loads(rj.read_text()) if rj else {}
    bench, block = data.get("bench") or {}, data.get("connector")
    calls_path = _find(job, "skill_calls.jsonl")
    c.runner = "connector" if (calls_path or block is not None) else "legacy"
    c.task = data.get("task") or (json.loads((job / "job.json").read_text()).get("task", "")
                                  if (job / "job.json").exists() else "")
    c.reason = str(bench.get("reason") or "")
    log = (job / "sim.log").read_text() if (job / "sim.log").exists() else ""

    # rounds, opens and closes
    if c.runner == "legacy":
        if not rj:
            c.notes.append("no result JSON: rounds from sim.log only (the cut-off round is not in the log)")
            for r in rounds_from_log(log):
                if r["verdict"] == "executed":
                    c.executed += 1
        else:
            _count_records(c, bench.get("rounds") or [])
            if log:
                n = sum(1 for r in rounds_from_log(log) if r["verdict"] == "executed")
                if n != c.executed:
                    c.notes.append(f"sim.log shows {n} executed rounds, the JSON {c.executed}")
    else:
        tape = {}
        tp = _find(job, "runner_tape.jsonl")
        if tp:
            tape = {r["call_id"]: r for r in _read_jsonl(tp)}
        if calls_path:
            _count_skill_rows(c, _read_jsonl(calls_path), tape)
        else:
            c.notes.append("no skill_calls.jsonl")
        if block and block.get("live_at_end"):
            c.cut_off += 1

    # welds and on-air closes
    gp = _find(job, "gripper.jsonl") if c.runner == "connector" else None
    if gp:
        rows = [r for r in _read_jsonl(gp) if r.get("event") == "close"]
        _count_closes(c, [int(r.get("is_grasping")) for r in rows],
                      [{k: r.get(k) for k in ("step", "arm", "owner", "call_id")} for r in rows])
        if log:
            n = sum(1 for v in closes_from_log(log) if v == 1)
            if n != c.welds:
                c.notes.append(f"sim.log shows {n} welds, gripper.jsonl {c.welds}")
    elif log:
        _count_closes(c, closes_from_log(log))
    else:
        c.notes.append("no gripper.jsonl and no sim.log: welds unknown")

    # teleports
    ledger = None
    lp = _find(job, "ledger.jsonl") if c.runner == "connector" else None
    if lp:
        ledger = {r["owner"]: r for r in _read_jsonl(lp)}
    elif block and isinstance(block.get("ledger"), dict):
        ledger = block["ledger"]
    if ledger is not None:
        _count_ledger(c, ledger)
        if bench.get("teleports") is not None and int(bench["teleports"]) != c.teleports:
            c.notes.append(f"bench.teleports {bench['teleports']} != ledger place_robot {c.teleports}")
    else:
        if bench.get("teleports") is not None:
            c.teleports = int(bench["teleports"])
        else:
            rl = result_line(log)
            if rl:
                c.teleports = rl["teleports"]
            else:
                c.notes.append("no teleport count (no bench.teleports, no RESULT line)")
        if c.runner == "legacy":
            c.teleports_ep, c.teleports_go_to = c.teleports, 0  # the Runner navigates only through ep.*
        else:
            c.notes.append("connector run without a ledger: the go_to / ep.* split is unknown")

    # steps and the goal
    rl = result_line(log)
    if data.get("steps") is not None:
        c.steps = int(data["steps"])
        if rl and rl["steps"] != c.steps:
            c.notes.append(f"RESULT line steps {rl['steps']} != JSON steps {c.steps}")
    elif rl:
        c.steps = rl["steps"]
    else:
        c.notes.append("no step count (no result JSON, no RESULT line)")
    if bench.get("goal"):
        _count_goal(c, bench["goal"])
    elif rl is None:
        c.notes.append("no goal block: delivered atoms unknown (counters compare raw)")
    return c


# ------------------------------------------------------------------------------------------------- compare mode
def fisher_one_sided(c_succ: int, c_n: int, t_succ: int, t_n: int) -> float:
    """P(T's successes <= the observed | the margins): the one-sided Fisher exact p for T's rate being lower."""
    N, K, n = c_n + t_n, c_succ + t_succ, t_n
    if N == 0 or n == 0 or c_n == 0:
        return 1.0
    denom = math.comb(N, K)
    lo = max(0, K - (N - n))
    return sum(math.comb(n, k) * math.comb(N - n, K - k) for k in range(lo, min(t_succ, K, n) + 1)) / denom


def minimum_detectable_effect(c_succ: int, c_n: int, t_n: int, alpha: float) -> Optional[float]:
    """The smallest drop in T's rate below C's observed rate that the one-sided test can call at this n: C's rate
    minus the largest T success rate with p < alpha. None when even 0 successes in T would not reach alpha."""
    if c_n == 0 or t_n == 0:
        return None
    for k in range(t_n, -1, -1):
        if fisher_one_sided(c_succ, c_n, k, t_n) < alpha:
            return c_succ / c_n - k / t_n
    return None


@dataclass
class Flag:
    task: str
    counter: str
    control: list
    treatment: list
    normalised: bool
    flagged: bool


@dataclass
class Pooled:
    skill: str
    c_n: int
    c_succ: int
    t_n: int
    t_succ: int
    p: float
    mde: Optional[float]
    fail: bool
    unqualified: int = 0  # rows of the skill that carried no qualifier while one was asked for

    @property
    def c_rate(self) -> Optional[float]:
        return None if not self.c_n else self.c_succ / self.c_n

    @property
    def t_rate(self) -> Optional[float]:
        return None if not self.t_n else self.t_succ / self.t_n


@dataclass
class Comparison:
    skill: str
    alpha: float
    delta: float
    flags: list = field(default_factory=list)  # Flag rows, every (task, counter) tested
    on_air: list = field(default_factory=list)  # {"task", "arm", "job", "causes"} for every run with on-air closes
    raw_tasks: list = field(default_factory=list)  # tasks compared raw (a run delivered no atom)
    pooled: Optional[Pooled] = None
    carried: list = field(default_factory=list)  # {"task", "job", "native_calls", "native_by"}

    @property
    def n_tests(self) -> int:
        return len(self.flags)

    @property
    def expected_false_flags(self) -> float:
        return 0.05 * self.n_tests

    @property
    def flagged(self) -> list:
        return [f for f in self.flags if f.flagged]


def _matches(call: CallOutcome, skill: str, qual: Optional[str], backend: Optional[str]) -> Optional[bool]:
    """True: the call is the switched skill; None: the skill but unqualified (counted, reported); False: another."""
    if call.skill != skill or (backend and call.backend != backend):
        return False
    if qual is None:
        return True
    return None if call.qual is None else call.qual == qual


def compare(pairs: dict, skill: str, *, backend: Optional[str] = None, alpha: float = 0.10, delta: float = 0.25,
            carried: Optional[list] = None) -> Comparison:
    """``pairs``: task -> (control Counters, treatment Counters). ``skill``: the switched skill, ``place`` or
    ``place.on``. ``carried``: (task, Counters) of the tasks carried forward, one NATIVE run each."""
    name, _, qual = skill.partition(".")
    cmp = Comparison(skill, alpha, delta)
    c_n = c_s = t_n = t_s = unq = 0
    for task, (C, T) in pairs.items():
        raw = any(x.delivered_atoms == 0 for x in C + T)
        if raw:
            cmp.raw_tasks.append(task)
        for spec in COUNTERS:
            if spec.worse not in (HIGHER, LOWER):
                continue
            cv = [x.value(spec.name) if raw else x.normalised(spec.name) for x in C]
            tv = [x.value(spec.name) if raw else x.normalised(spec.name) for x in T]
            cs, ts = [v for v in cv if v is not None], [v for v in tv if v is not None]
            if not cs or not ts:
                flagged = False
            elif spec.worse == HIGHER:
                flagged = min(ts) > max(cs)
            else:
                flagged = max(ts) < min(cs)
            cmp.flags.append(Flag(task, spec.name, cv, tv, not raw, flagged))
        for arm, runs in (("C", C), ("T", T)):
            for x in runs:
                if x.on_air:
                    cmp.on_air.append({"task": task, "arm": arm, "job": x.job, "causes": list(x.on_air_causes)})
        for arm, runs in (("C", C), ("T", T)):
            for x in runs:
                for call in x.calls:
                    m = _matches(call, name, qual or None, backend)
                    if m is None:
                        unq += 1
                    if not m:
                        continue
                    if arm == "C":
                        c_n, c_s = c_n + 1, c_s + call.ok
                    else:
                        t_n, t_s = t_n + 1, t_s + call.ok
    p = fisher_one_sided(c_s, c_n, t_s, t_n)
    drop = (c_s / c_n - t_s / t_n) if c_n and t_n else 0.0
    cmp.pooled = Pooled(skill, c_n, c_s, t_n, t_s, p, minimum_detectable_effect(c_s, c_n, t_n, alpha),
                        bool(c_n and t_n and drop >= delta and p < alpha), unq)
    for task, x in carried or []:
        cmp.carried.append({"task": task, "job": x.job, "native_calls": x.native_calls,
                            "native_by": dict(x.native_by)})
    return cmp


# ------------------------------------------------------------------------------------------------- printing
def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.4g}"
    return str(v)


def format_counters(cs: list[Counters]) -> str:
    names = [c.name for c in COUNTERS if c.name not in ("open_rate", "close_rate")]
    names += ["open_successes", "close_successes", "delivered_new", "goal_total", "delivered_items", "native_calls"]
    lines = ["counter".ljust(18) + "  " + "  ".join(Path(c.job).name[:22].rjust(22) for c in cs)]
    for n in names:
        lines.append(n.ljust(18) + "  " + "  ".join(_fmt(c.value(n)).rjust(22) for c in cs))
    for c in cs:
        for note in c.notes:
            lines.append(f"  note {Path(c.job).name}: {note}")
        for cause in c.on_air_causes:
            lines.append(f"  on-air {Path(c.job).name}: {cause}")
    return "\n".join(lines)


def format_comparison(cmp: Comparison) -> str:
    out = [f"switched skill {cmp.skill}: alpha {cmp.alpha}, fail at a drop >= {cmp.delta}"]
    p = cmp.pooled
    if p is not None:
        out.append(f"pooled per-call: C {p.c_succ}/{p.c_n} ({_fmt(p.c_rate)}), T {p.t_succ}/{p.t_n} "
                   f"({_fmt(p.t_rate)}), one-sided Fisher p {p.p:.3f}, MDE at this n "
                   f"{'not reachable' if p.mde is None else _fmt(p.mde)}, "
                   f"{'FAIL' if p.fail else 'pass'}"
                   + (f"; {p.unqualified} unqualified rows of the skill left out" if p.unqualified else ""))
    out.append(f"flag tests {cmp.n_tests}, flagged {len(cmp.flagged)}, expected false flags "
               f"{cmp.expected_false_flags:.2f}")
    for f in cmp.flags:
        if f.flagged:
            out.append(f"  FLAG {f.task} {f.counter}{'' if f.normalised else ' (raw)'}: "
                       f"C {[_fmt(v) for v in f.control]} T {[_fmt(v) for v in f.treatment]}")
    if cmp.raw_tasks:
        out.append(f"compared raw (a run delivered no atom): {', '.join(cmp.raw_tasks)}")
    for row in cmp.on_air:
        out.append(f"  on-air {row['task']} {row['arm']} {Path(row['job']).name}: needs a cause: {row['causes']}")
    for row in cmp.carried:
        verdict = "0 native calls" if not row["native_calls"] else f"{row['native_calls']} native calls " \
                                                                    f"{row['native_by']} (REPORT)"
        out.append(f"  carried {row['task']} {Path(row['job']).name}: {verdict}")
    return "\n".join(out)


def _comparison_dict(cmp: Comparison) -> dict:
    d = dataclasses.asdict(cmp)
    d["n_tests"], d["expected_false_flags"] = cmp.n_tests, cmp.expected_false_flags
    if cmp.pooled is not None:
        d["pooled"]["c_rate"], d["pooled"]["t_rate"] = cmp.pooled.c_rate, cmp.pooled.t_rate
    return d


# ------------------------------------------------------------------------------------------------- command line
def _split(arg: str) -> list[Path]:
    return [Path(p) for p in arg.split(",") if p]


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    ex = sub.add_parser("extract", help="the counters of one or more job dirs")
    ex.add_argument("jobs", nargs="+", type=Path)
    ex.add_argument("--instance", type=int, default=None)
    ex.add_argument("--json", action="store_true")
    cp = sub.add_parser("compare", help="the ladder: control against treatment per task, the pooled per-call test")
    cp.add_argument("--skill", required=True, help="the switched skill, e.g. place or place.on")
    cp.add_argument("--backend", default=None, help="count only calls on this backend (e.g. tiptop)")
    cp.add_argument("--pair", nargs=3, action="append", metavar=("TASK", "C_DIRS", "T_DIRS"), default=[],
                    help="a task's control and treatment job dirs, comma-separated")
    cp.add_argument("--carried", nargs=2, action="append", metavar=("TASK", "DIR"), default=[])
    cp.add_argument("--alpha", type=float, default=0.10)
    cp.add_argument("--delta", type=float, default=0.25)
    cp.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.mode == "extract":
        cs = [extract(j, args.instance) for j in args.jobs]
        print(json.dumps([c.as_dict() for c in cs], indent=1) if args.json else format_counters(cs))
        return 0
    pairs = {t: ([extract(p) for p in _split(c)], [extract(p) for p in _split(tr)]) for t, c, tr in args.pair}
    carried = [(t, extract(Path(d))) for t, d in args.carried]
    cmp = compare(pairs, args.skill, backend=args.backend, alpha=args.alpha, delta=args.delta, carried=carried)
    print(json.dumps(_comparison_dict(cmp), indent=1, default=str) if args.json else format_comparison(cmp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
