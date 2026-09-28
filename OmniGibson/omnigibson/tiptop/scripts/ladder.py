"""The switch ladder (WEEK4_PLAN 5.8; W4-I): native skills routed in one line at a time, each stage a paired
comparison of two arms on every task that calls the stage's line, judged with scripts/counters.py.

    python ladder.py plan  --stage S1                    # the branch per task, the re-stamped tapes, plan.json
    python ladder.py run   --stage S1 [--tasks T ...] [--reps 0,1,2] [--arms C,T]   # the queue (setsid it)
    python ladder.py run   --stage strict|witness|carried
    python ladder.py gate  --stage S1|S2|strict|witness|carried   # GATE.md + gate.json
    python ladder.py status

Run it with the snapshot's python and PYTHONPATH (``$S/b1k/bin/python``, ``PYTHONPATH=$S/OmniGibson:$S/tiptop``), so
the typing and routing it reasons with are the snapshot's; the sims and planners it launches come from ``--snap``.

Stages (WEEK4_PLAN 5.8; BASE.md 5 and legacy_ref/notes.md corrected the list from the L-rec per-call tables):
- S1 ``place.on=tiptop``: C is PARITY, T is PARITY plus the line;
- S2 ``close.prismatic=tiptop``: C is S1's T, T adds the line. No task in the corpus reaches a close (store_honey's
  episode ends inside its second place, and a joint-less close on the four-drawer cabinet routes to the default), so
  ``plan`` finds no branch and the stage is vacuous; store_honey is carried forward.

Arms. Both arms replay-live the task's L-rec websocket tape (legacy_ref/L-rec/<task>/episode/wstape) and are forced
live at the same branch frame N (``--wstape-live-at N``): the first planner request after the first Runner write that
C and T route differently. The branch is computed from the L-rec's Runner tape (the writes, typed as the shim types
them, routed under each arm's profile as SkillRegistry.backend_for routes with no map: by_relation, by_joint on "",
else default) and its wstape index (the frames' ledger owners aligned to the writes in order; the connector runner's
E-rep replay log, which carries the shim's call id on every served frame, is the cross-check when it exists).
Replicates r = 0, 1, 2 per arm: ``--replicate r --seed r``. The bench stamps ``seed = 2300 + 1000 r + k`` on every
legacy plan request and the replay compares the stamp like any other field, so a replicate above 0 would go live at
frame 1 on the seed alone; each replicate is therefore served a copy of the L-rec tape with its plan requests
re-stamped for that replicate (``restamp_tape``: the responses, the L-rec's plans, are untouched), which keeps the
prefix replayed and gives the live tail its own seeds.

Every run: the connector runner, the plan's common flags, ``--runner-tape``, videos on, from the snapshot, at most 5
sims (the week-3 slot pool) and 4 live planners across every track, GPUs 1 and 3 by free memory after nvidia-smi,
setsid, planners stopped by their PID (SIGTERM) when their sim exits, OMP_NUM_THREADS=8 MKL_NUM_THREADS=1
PYTHONHASHSEED=2300. The planner starts before its sim and idles through the replayed prefix.

Gate per stage (WEEK4_PLAN 5.8 a-f), from the runs' own files through counters.py's compare mode: (a) the prefix
(every served frame before N matched and the switch happened at N, or a pre-branch divergence within the task's A/A
floor; the Runner tape identical before the branch write); (b) hard on every run (U0 exact, rule 2, the hand refresh,
no crash reason C lacks, the switched skill never UNSUPPORTED or BACKEND_ERROR where C ran the same write, every
on-air close with its owning call); (c) the pooled per-call success of the switched skill, one-sided Fisher, with the
MDE; (d) the counters per task normalised per delivered atom, min(T) > max(C) a FLAG, each with a cause from the
per-call table; (e) structural shifts named; (f) scores reported. Also the per-call table, the legacy-dependency
list, the hand-refresh count and the D21 debt.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve()
for root in (HERE.parents[4] / "tiptop", HERE.parents[3], HERE.parent):  # the checkout's tiptop, OmniGibson, scripts/
    if root.is_dir() and str(root) not in sys.path:
        sys.path.append(str(root))

import counters  # noqa: E402  (scripts/counters.py, beside this file)
from b1k.connector.skills import (  # noqa: E402
    CloseArgs, IntentArgs, OpenArgs, PickArgs, PlaceArgs, Rel, Relation, ReleaseArgs, SkillCall, WaitArgs,
)
from b1k.connector.types import ObjRef  # noqa: E402
from b1k.planner.pseudo import names, shim  # noqa: E402
from b1k.planner.pseudo import tape as tp  # noqa: E402
from b1k.skills.registry import relation_kind  # noqa: E402

# ------------------------------------------------------------------------------------------------- where things are
WEEK4 = Path("/home/wding8/projects/BEHAVIOR-1K/runs/skill_arch_20260925/week4")
REF, Q1, OUT = WEEK4 / "legacy_ref", WEEK4 / "q1", WEEK4 / "ladder"
LREC = REF / "L-rec"
SNAP = Path("/home/wding8/projects/wt-snap/w4ladder")
PIXI = Path("/home/wding8/projects/wt-manip/tiptop/.pixi/envs/default/bin")
TOOLS = Path("/tmp/claude-1243003/-home-wding8-projects-BEHAVIOR-1K/86e9ce10-28ff-436e-835d-3afd95d0176f/scratchpad/week3/tools")
SLOTS = TOOLS.parent / "simslots"

TASKS = ("attach_a_camera_to_a_tripod", "bringing_in_wood", "composting_waste", "dispose_of_batteries",
         "rearrange_your_room", "store_honey", "tidying_bedroom")
COMMON = ("--knowledge", "oracle", "--grasping-mode", "assisted", "--views", "head", "left_wrist", "right_wrist",
          "--room", "--inside-region", "--no-state-stream", "--instances", "0", "--rounds", "2")
PLANNER_ARGS = ("--num-particles", "256", "--max-planning-time", "40", "--seed", "2300")
REPS = (0, 1, 2)
MAX_SIMS, MAX_PLANNERS, GPUS, MIN_FREE_MIB = 5, 4, (1, 3), 30000
ALLOWED_GPUS = (1, 3)  # the user's rule for this track; 0, 2 and 4-7 are other people's


def parse_gpus(text: str) -> tuple:
    """``--gpus 3``: the cards this queue may use, a subset of ALLOWED_GPUS (a coworker may hold the other one)."""
    gpus = tuple(int(x) for x in text.split(",") if x.strip())
    bad = [g for g in gpus if g not in ALLOWED_GPUS]
    if not gpus or bad:
        raise ValueError(f"--gpus {text!r}: only {ALLOWED_GPUS} may be used (refused {bad or 'an empty list'})")
    return gpus
PORT_BASE, TICK, SETTLE, SIM_TIMEOUT_S = 8850, 30, 120, 21600
WITNESS_TASK, WITNESS_ROUTES = "cook_bacon", ("press=tiptop", "place.on=tiptop")
STRICT_TASK, STRICT_REF = "store_honey", Q1 / "E-rep" / "store_honey_r1"  # W4-F2's E-rep that matched to the end
INSTANCE = 301
UNSUPPORTED_CODES = ("unsupported", "backend_error")
RESULT_RE = re.compile(r"RESULT instance (\d+): q_score (\S+) success (\w+) steps (\d+)/(\d+) \((.*?)\); teleports (\d+); ([\d.]+)s")

# the Episode call each Runner write nests into (the StepLedger's innermost owner on a planner request)
OWNERS = {"pick": ("ep.pick",), "achieve": ("ep.achieve",), "put_down": ("ep.put_down", "ep.achieve"),
          "open_up": ("ep.open_up",), "release": ("ep.release",), "pour": ("ep.pour",), "dwell": ("ep.dwell",)}
NAV = ("stand_for", "walk_to_floor")
FRAME_OPS_IGNORED = ("metadata", "stream")


@dataclass(frozen=True)
class Stage:
    name: str
    line: Optional[str]  # the --route this stage adds; None for the non-ladder stages
    prev: Optional[str]
    tasks: tuple = ()  # the tasks whose arms the stage runs (WEEK4_PLAN 5.8, BASE.md 5)


# attach_a_camera_to_a_tripod and composting_waste call place.on once in their L-rec (a floor put_down after the goal
# failed twice; legacy_ref/calls_table.md), which BASE.md's count from the historical jobs does not have. Their branch
# frames (6 and 9) lie past their A/A floor (frame 3, where every W4-F2 E-rep of either diverged), so an arm pair
# would go live on both arms before the branch and meet the put_down only if the live run repeats the failures: not
# the plan's arms. They are carried forward; the carried run routes the line natively and reports the call if reached.
STAGES = {"S1": Stage("S1", "place.on=tiptop", None, ("bringing_in_wood", "rearrange_your_room", "tidying_bedroom")),
          "S2": Stage("S2", "close.prismatic=tiptop", "S1", ("store_honey",))}
CARRIED = ("attach_a_camera_to_a_tripod", "composting_waste", "dispose_of_batteries", "store_honey")


def stage_routes(name: str) -> tuple[list, list]:
    """(C's --route lines, T's --route lines): C is the previous stage's T, T adds this stage's line."""
    st = STAGES[name]
    c = [] if st.prev is None else stage_routes(st.prev)[1]
    return list(c), list(c) + [st.line]


def profile_of(routes) -> dict:
    from omnigibson.tiptop.host.routing_profiles import PARITY, overlay

    return overlay(PARITY, list(routes))


def native_profile() -> dict:
    from omnigibson.tiptop.host.routing_profiles import native

    return native()


# ------------------------------------------------------------------------------------------------- typing and routing
def ref(name: str) -> ObjRef:
    return ObjRef(name, names.bddl_category(name), False)


def typed_write(member: str, args: tuple, kwargs: dict):
    """(skill, args, arm) as the shim's write methods build them (shim.py: pick, achieve, put_down, open_up, release,
    pour, dwell; WEEK4_PLAN 4 rows 18-24); None for a navigation write (rows 25-26: go_to, never a skill)."""
    if member == "pick":
        return "pick_up", PickArgs(ref(args[0])), "left"
    if member == "achieve":
        skill, a = shim.typed_achieve(list(args[0]), ref)
        return skill, a, kwargs.get("arm", "left")
    if member == "put_down":
        return "place", PlaceArgs(ref(args[0]), (Relation(Rel.ON, ref(args[1])),)), "left"
    if member == "open_up":
        fraction = args[1] if len(args) > 1 else kwargs.get("fraction")
        if fraction == 0.0:
            return "close", CloseArgs(ref(args[0])), "left"
        return "open", OpenArgs(ref(args[0]), min_fraction=fraction), "left"
    if member == "release":
        return "release", ReleaseArgs(), "left"
    if member == "pour":
        return "intent.pour", IntentArgs(ref(args[0]), ref(args[1]), "pour"), "left"
    if member == "dwell":
        return "wait", WaitArgs(int(args[0])), None
    if member in NAV:
        return None
    raise ValueError(f"unknown Runner write {member}")


def route(profile: dict, call: SkillCall, joint_kind: str = "") -> str:
    """The backend, as SkillRegistry.backend_for resolves it with no map (joint_kind "": a joint-less close on a
    piece of several joints takes the default, BASE.md 4.2)."""
    rule = profile.get(call.skill, {})
    if call.backend:
        return call.backend
    rel = relation_kind(call)
    if rule.get("by_relation", {}).get(rel):
        return rule["by_relation"][rel]
    if rule.get("by_joint", {}).get(joint_kind):
        return rule["by_joint"][joint_kind]
    return rule.get("default", "tiptop")


def qualifier(call: SkillCall) -> Optional[str]:
    if call.skill == "place":
        return relation_kind(call) or "mixed"
    return None


@dataclass
class Write:
    i: int  # index among the tape's writes
    record: int  # index in tape.records
    member: str
    args: list
    kwargs: dict
    k: Optional[int]  # the shim's call id q1-k (None for a navigation write)
    skill: Optional[str]
    qual: Optional[str]
    returned: Optional[bool]
    exc: Optional[str]
    step: Optional[list]

    @property
    def call_id(self) -> Optional[str]:
        return None if self.k is None else f"q1-{self.k}"

    @property
    def line(self) -> Optional[str]:
        return None if self.skill is None else (self.skill + (f".{self.qual}" if self.qual else ""))


def writes_of(tape: tp.Tape) -> list[Write]:
    out, k = [], 0
    for i, (record, r) in enumerate((n, x) for n, x in enumerate(tape.records) if x.get("kind") == "write"):
        member, args, kwargs = r["member"], list(r.get("args") or ()), dict(r.get("kwargs") or {})
        tc = typed_write(member, tuple(args), kwargs)
        skill = qual = None
        if tc is not None:
            k += 1
            call = SkillCall(tc[0], tc[1], arm=tc[2])
            skill, qual = call.skill, qualifier(call)
        exc = exc_type(r.get("exc"))
        ret = r.get("ret")
        out.append(Write(i, record, member, args, kwargs, k if tc is not None else None, skill, qual,
                         None if exc is not None or not isinstance(ret, bool) else ret, exc, r.get("step")))
    return out


def call_of(w: Write) -> Optional[SkillCall]:
    tc = typed_write(w.member, tuple(w.args), w.kwargs)
    return None if tc is None else SkillCall(tc[0], tc[1], arm=tc[2])


def backends_of(writes: list[Write], profile: dict) -> list[Optional[str]]:
    return [None if (c := call_of(w)) is None else route(profile, c) for w in writes]


@dataclass
class Branch:
    write: int  # index among the writes
    record: int  # index in tape.records
    k: int
    member: str
    line: str
    backend_c: str
    backend_t: str

    @property
    def call_id(self) -> str:
        return f"q1-{self.k}"


def branch_of(writes: list[Write], profile_c: dict, profile_t: dict) -> Optional[Branch]:
    """The first write the two profiles route differently; None when they agree on every write."""
    bc, bt = backends_of(writes, profile_c), backends_of(writes, profile_t)
    for w, c, t in zip(writes, bc, bt):
        if c != t:
            return Branch(w.i, w.record, w.k, w.member, w.line, c, t)
    return None


# ------------------------------------------------------------------------------------------------- frames
def frames_by_write(index_rows: list[dict], writes: list[Write]) -> dict[int, Optional[int]]:
    """frame index -> the write (index among ``writes``) it was made inside, by the frames' ledger owners aligned to
    the non-navigation writes in order (a frame stays with the current write while its owner is one of that write's
    Episode calls, else moves on to the next write whose calls include it); None for a frame that fits no write."""
    skill_writes = [w for w in writes if w.k is not None]
    out, cur = {}, 0
    for row in index_rows:
        if row.get("op") in FRAME_OPS_IGNORED:
            continue
        owner = row.get("owner")
        j = cur
        while j < len(skill_writes) and owner not in OWNERS.get(skill_writes[j].member, ()):
            j += 1
        if j >= len(skill_writes):
            out[int(row["i"])] = None
            continue
        cur = j
        out[int(row["i"])] = skill_writes[j].i
    return out


NO_REQUEST_ERRORS = ("GoalNotVisible",)  # a round that ends before asking the planner (cook_bacon: 12 of 13 rounds)


def requesting_rounds(rounds: list) -> list[dict]:
    """The Episode's round records that made a planner request, in order: one frame each (every round of the 7
    L-rec runs did: 52 rounds, 52 plan frames; a GoalNotVisible round ends before the request)."""
    return [r for r in rounds or [] if "round" in r and not str(r.get("error") or "").startswith(NO_REQUEST_ERRORS)]


def frames_by_rounds(index_rows: list[dict], rounds: list, writes: list[Write]) -> dict[int, Optional[int]]:
    """frame index -> the write it was made inside, the j-th request frame being the j-th requesting round's, and
    the round's step placing it in the write whose step range holds it. Empty when the counts disagree (the owner
    alignment then decides)."""
    frames = [row for row in index_rows if row.get("op") not in FRAME_OPS_IGNORED]
    reqs = requesting_rounds(rounds)
    if len(frames) != len(reqs):
        return {}
    out = {}
    for row, rnd in zip(frames, reqs):
        step = rnd.get("step")
        w = next((w for w in writes if w.k is not None and w.step and step is not None and w.step[0] <= step < w.step[1]), None)
        out[int(row["i"])] = None if w is None else w.i
    return out


def frame_of_branch(index_rows: list[dict], writes: list[Write], branch: Branch, rounds: list = None) -> tuple:
    """(N, method): the first planner request at or after the branch write; by the rounds when they align with the
    frames, else by the frames' owners."""
    for method, by in (("rounds", frames_by_rounds(index_rows, rounds or [], writes) if rounds else {}),
                       ("owner", frames_by_write(index_rows, writes))):
        cands = [i for i, w in by.items() if w is not None and w >= branch.write]
        if cands:
            return min(cands), method
    return None, None


def frame_of_call(replay_rows: list[dict], call_id: str) -> Optional[int]:
    """The E-rep cross-check: the first served frame carrying the shim's call id, provided every frame before it
    matched the tape (a replay that diverged earlier says nothing about later frames)."""
    for row in replay_rows:
        if row.get("call_id") == call_id:
            return int(row["i"])
        if not row.get("matched", True):
            return None
    return None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------------------------------------- the re-stamped tape
def snapshot_root_of(metadata: dict) -> Optional[Path]:
    """The checkout the recording planner ran from, off its metadata's ``modules.tiptop`` (<root>/tiptop/tiptop/
    __init__.py): what run.check_imports compares with the sim's own root."""
    p = ((metadata or {}).get("modules") or {}).get("tiptop")
    return None if not p else Path(p).parents[2]


def reroot_metadata(metadata: dict, root_from: Path, root_to: Path) -> tuple[dict, list]:
    """The metadata with every ``modules`` path under ``root_from`` moved under ``root_to`` (the planner code at
    both is the same commit: a served metadata frame must name the checkout the sim and the live planner run from,
    or check_imports refuses the mix). (the new metadata, the paths changed)."""
    out, changed = copy.deepcopy(metadata or {}), []
    mods = out.get("modules")
    if isinstance(mods, dict):
        for name, p in list(mods.items()):
            if isinstance(p, str) and Path(p).is_relative_to(root_from):
                mods[name] = str(root_to / Path(p).relative_to(root_from))
                changed.append((name, p, mods[name]))
    return out, changed


def restamp_tape(src: Path, dst: Path, replicate: int, reroot: Optional[tuple] = None) -> int:
    """A copy of the websocket tape at ``src`` with every plan request's ``seed`` re-stamped for ``replicate``
    (wstape.seed_for(replicate, k), k the frame's own pipeline index), the frame's seed and the index row with it,
    and, with ``reroot=(root_from, root_to)``, the metadata frames' ``modules`` paths moved to the snapshot the run
    comes from; everything else, the responses above all, byte for byte the recorded value. The count of frames."""
    from b1k.bridge.protocol import packb
    from omnigibson.tiptop.host.wstape import TapeDir, seed_for

    src_dir, dst_dir = TapeDir(src), TapeDir(dst)
    if dst_dir.index.exists():
        raise FileExistsError(f"{dst_dir.index} exists; a tape is never overwritten")
    rows = src_dir.rows()
    n = 0
    for row in rows:
        frame = src_dir.load(int(row["i"]))
        if frame is None:
            raise FileNotFoundError(f"{src}: frame {row['i']} has no recorded body")
        if frame.op == "plan" and frame.k is not None and frame.request is not None:
            req = frame.request_dict()
            if isinstance(req, dict):
                seed = seed_for(replicate, frame.k)
                req["seed"] = seed
                frame.request, frame.seed = packb(req), seed
        if reroot is not None and frame.metadata is not None:
            frame.metadata, _ = reroot_metadata(frame.metadata, Path(reroot[0]), Path(reroot[1]))
        dst_dir.append(frame, whole=True)
        n += 1
    return n


def derived_tape(src: Path, dst: Path, replicate: int, snap: Path) -> Path:
    """The tape a ladder run is served: ``src`` re-stamped for the replicate and re-rooted from the recording
    snapshot to ``snap``; made once, reused after."""
    from omnigibson.tiptop.host.wstape import TapeDir

    if not (dst / "index.jsonl").exists():
        first = TapeDir(src).load(0)
        root_from = snapshot_root_of(first.metadata if first is not None else None)
        restamp_tape(src, dst, replicate, reroot=None if root_from is None else (root_from, snap))
    return dst


# ------------------------------------------------------------------------------------------------- runs
@dataclass
class RunSpec:
    stage: str
    task: str
    arm: str  # C | T | native | strict | witness
    rep: int
    out: Path
    profile: str  # parity | native
    routes: list
    wstape: str  # replay-live | replay | record
    tape: Optional[Path]  # the tape served (replay modes)
    live_at: Optional[int]
    planner: bool  # a live tiptop-server is started for it
    seed: Optional[int] = None

    @property
    def label(self) -> str:
        return f"ladder-{self.stage}-{self.task}-{self.arm}{self.rep}"

    def bench_args(self, port: int) -> list[str]:
        a = ["--task-name", self.task, "--out-dir", str(self.out / "episode"), "--host", "127.0.0.1", "--port", str(port),
             *COMMON, "--runner", "connector", "--routing-profile", self.profile]
        for r in self.routes:
            a += ["--route", r]
        a += ["--replicate", str(self.rep), "--runner-tape", str(self.out / "episode" / "tapes"), "--wstape", self.wstape]
        if self.tape is not None:
            a += ["--wstape-path", str(self.tape)]
        if self.live_at is not None:
            a += ["--wstape-live-at", str(self.live_at)]
        if self.seed is not None:
            a += ["--seed", str(self.seed)]
        return a


def run_dir(stage: str, task: str, arm: str, rep: int) -> Path:
    return OUT / stage / f"{task}_{arm}{rep}"


def stage_specs(stage: str, plan: dict, tasks=None, reps=REPS, arms=("C", "T")) -> list[RunSpec]:
    c_routes, t_routes = stage_routes(stage)
    specs = []
    for task, p in plan["tasks"].items():
        if tasks and task not in tasks:
            continue
        if p.get("branch") is None:
            continue
        for rep in reps:
            for arm in arms:
                specs.append(RunSpec(stage, task, arm, rep, run_dir(stage, task, arm, rep), "parity",
                                     c_routes if arm == "C" else t_routes, "replay-live",
                                     Path(p["tapes"][str(rep)]), int(p["frame"]), True, seed=rep))
    return specs


def carried_routes() -> list:
    """The ladder's lines on top of NATIVE (the full native profile once every stage is in): routing.yaml does not
    route place.on yet, and close.prismatic is already its line."""
    return stage_routes(list(STAGES)[-1])[1]


def carried_specs(tasks=CARRIED) -> list[RunSpec]:
    return [RunSpec("carried", t, "native", 0, run_dir("carried", t, "native", 0), "native", carried_routes(), "record", None,
                    None, True, seed=0) for t in tasks]


def witness_spec() -> RunSpec:
    return RunSpec("witness", WITNESS_TASK, "witness", 0, run_dir("witness", WITNESS_TASK, "witness", 0), "native",
                   list(WITNESS_ROUTES), "record", None, None, True, seed=0)


def strict_spec(snap: Path = SNAP, tape: Optional[Path] = None) -> RunSpec:
    """The strict E-rep: PARITY, a strict replay of the L-rec tape (re-rooted to ``snap``: the served metadata must
    name the sim's own checkout, or check_imports refuses it), replicate 0, no planner."""
    tape = derived_tape(LREC / STRICT_TASK / "episode" / "wstape", OUT / "strict" / "tapes" / f"{STRICT_TASK}_r0", 0, snap) \
        if tape is None else tape
    return RunSpec("strict", STRICT_TASK, "strict", 0, run_dir("strict", STRICT_TASK, "strict", 0), "parity", [],
                   "replay", tape, None, False)


LAUNCHER = """#!/bin/bash
# {label}: one ladder run from the snapshot {snap} (ladder.py wrote this; WEEK4_PLAN 5.8, W4-I)
S={snap}; OUT={out}; port={port}; gpu={gpu}
export PYTHONHASHSEED=2300 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
mkdir -p $OUT/episode
{planner}
SLOT=$({tools}/simslot.sh acquire {label})
cd $S || exit 1
setsid env -u OMNIGIBSON_GPU_ID CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 \\
  PYTHONHASHSEED=2300 OMNIGIBSON_HEADLESS=1 PYTHONPATH=$S/OmniGibson:$S/tiptop \\
  timeout -k 120 {timeout} $S/b1k/bin/python -m omnigibson.tiptop.bench {bench_args} > $OUT/sim.log 2>&1 < /dev/null &
pid=$!
echo $pid > $OUT/sim.pid
{tools}/simslot.sh bind $SLOT $pid
echo "$SLOT $pid" > $OUT/slot
echo "$(date +%T) {label} sim pid $pid slot $SLOT gpu $gpu port $port"
( while kill -0 $pid 2>/dev/null; do sleep 20; done
  [ "$(cat {slots}/$SLOT/pid 2>/dev/null)" = "$pid" ] && {tools}/simslot.sh release $SLOT
  printf '{{"ended":%s}}\\n' "$(date +%s)" > $OUT/job_end.json
  {stop_planner} ) > /dev/null 2>&1 &
"""

PLANNER_START = """mkdir -p $OUT/planner; cd $OUT/planner || exit 1
setsid env -u LD_LIBRARY_PATH -u OMNIGIBSON_GPU_ID CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \\
  MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 PYTHONHASHSEED=2300 PYTHONPATH=$S/tiptop:$S/tiptop/cutamp PATH={pixi}:$PATH \\
  {pixi}/tiptop-server --config $S/tiptop/tiptop/config/tiptop_sim_r1pro.yml --host 127.0.0.1 --port $port \\
  {planner_args} --rerun-mode disabled > $OUT/planner/planner.log 2>&1 < /dev/null &
echo $! > $OUT/planner/planner.pid
echo "$(date +%T) {label} planner :$port gpu $gpu pid $(cat $OUT/planner/planner.pid)"
for i in $(seq 1 120); do grep -q "server listening" $OUT/planner/planner.log 2>/dev/null && break; sleep 10; done
if ! grep -q "server listening" $OUT/planner/planner.log 2>/dev/null; then
  echo "$(date +%T) {label}: planner :$port NOT listening after 20 min; stopping it"; kill -TERM $(cat $OUT/planner/planner.pid) 2>/dev/null; exit 1
fi
echo "$(date +%T) {label}: planner :$port listening after ~$((i*10)) s"
"""

PLANNER_STOP = """ppid=$(cat $OUT/planner/planner.pid); kill -TERM $ppid 2>/dev/null
  echo "$(date +%T) planner $ppid (:$port) stopped after sim $pid exited" > $OUT/planner_stopped"""


def launcher_text(spec: RunSpec, port: int, gpu: int, snap: Path = SNAP) -> str:
    return LAUNCHER.format(
        label=spec.label, snap=snap, out=spec.out, port=port, gpu=gpu, tools=TOOLS, slots=SLOTS, timeout=SIM_TIMEOUT_S,
        bench_args=" ".join(spec.bench_args(port)),
        planner=PLANNER_START.format(pixi=PIXI, planner_args=" ".join(PLANNER_ARGS), label=spec.label) if spec.planner else "",
        stop_planner=PLANNER_STOP if spec.planner else "true",
    )


# ------------------------------------------------------------------------------------------------- the queue
def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def alive(pid) -> bool:
    try:
        return Path(f"/proc/{int(pid)}").exists()
    except (TypeError, ValueError):
        return False


def live_slots() -> int:
    n = 0
    for s in SLOTS.glob("slot*"):
        pid = s / "pid"
        if pid.exists():
            n += alive(pid.read_text().strip())
        elif time.time() - s.stat().st_mtime <= 900:
            n += 1
    return n


def planners_live() -> int:
    """Live tiptop-server processes of this user, whichever track started them."""
    out = subprocess.run(["pgrep", "-u", os.environ.get("USER", "wding8"), "-f", r"bin/tiptop-server "],
                         capture_output=True, text=True)
    return len([l for l in out.stdout.split() if l.strip()])


def gpu_free() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    free = {}
    for line in out.stdout.splitlines():
        i, f = line.split(",")
        if int(i) in GPUS:
            free[int(i)] = int(f)
    return free


def gpu_owners() -> dict:
    """GPU index -> the users with a compute process on it (nvidia-smi's apps, their pids' owners by ps)."""
    idx = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True)
    by_uuid = {u.strip(): int(i) for i, u in (line.split(",", 1) for line in idx.stdout.splitlines() if "," in line)}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True)
    out: dict = {}
    for line in apps.stdout.splitlines():
        if "," not in line:
            continue
        pid, uuid = (x.strip() for x in line.split(",", 1))
        user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip() or "?"
        out.setdefault(by_uuid.get(uuid), set()).add(user)
    return out


def pick_gpu() -> Optional[int]:
    """The allowed card with the most free memory that no other user has a process on (free memory alone is no
    permission: a coworker's job may leave 30 GB free), and only when it has MIN_FREE_MIB."""
    free, owners, me = gpu_free(), gpu_owners(), os.environ.get("USER", "wding8")
    free = {g: f for g, f in free.items() if not (owners.get(g, set()) - {me})}
    best = max(free, key=free.get) if free else None
    return best if best is not None and free[best] >= MIN_FREE_MIB else None


def m2t2_up(host: str = "127.0.0.1", port: int = 8123, timeout_s: float = 3.0) -> bool:
    """The shared grasp server a live planner asks (WEEK4_PLAN 5.8 item 7): its port accepts a connection."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def ports_in_use() -> set:
    out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
    return {int(m.group(1)) for m in re.finditer(r":(\d+)\s", out)}


def ports_claimed() -> set:
    """Ports named by every ladder run launched and not ended (another queue's planner may not listen yet)."""
    out = set()
    for spec in OUT.glob("*/*/spec.json") if OUT.exists() else ():
        try:
            d = json.loads(spec.read_text())
        except ValueError:
            continue
        if d.get("planner") and not (spec.parent / "job_end.json").exists():
            out.add(int(d.get("port", 0)))
    return out


def next_port(taken: set) -> int:
    used = ports_in_use() | ports_claimed() | taken
    p = PORT_BASE
    while p in used:
        p += 1
    return p


def started(out: Path) -> bool:
    return (out / "sim.pid").exists()


def ended(out: Path) -> bool:
    return (out / "job_end.json").exists()


def result_line(out: Path) -> Optional[str]:
    p = out / "sim.log"
    if not p.exists():
        return None
    for line in p.read_text(errors="replace").splitlines():
        if "RESULT instance" in line:
            return line.split("INFO: ", 1)[-1].strip()
    return None


def launch(spec: RunSpec, port: int, gpu: int, snap: Path, launch_log: Path) -> subprocess.Popen:
    spec.out.mkdir(parents=True, exist_ok=True)
    script = spec.out / "launch.sh"
    script.write_text(launcher_text(spec, port, gpu, snap))
    (spec.out / "spec.json").write_text(json.dumps({**asdict(spec), "out": str(spec.out), "tape": None if spec.tape is None else str(spec.tape),
                                                    "port": port, "gpu": gpu, "snap": str(snap)}, indent=1))
    return subprocess.Popen(["setsid", "bash", str(script)], stdout=open(launch_log, "a"), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def queue(specs: list[RunSpec], snap: Path, launch_log: Path, max_sims: int = MAX_SIMS, max_planners: int = MAX_PLANNERS) -> int:
    """Launch every spec within the limits, one per tick, then a settle wait; report each RESULT line as it lands."""
    pending = [s for s in specs if not started(s.out)]
    log(f"queue up: {len(specs)} runs ({len(specs) - len(pending)} already started); slots {live_slots()}/{max_sims}, "
        f"planners {planners_live()}/{max_planners}, gpu free {gpu_free()}, snap {snap}")
    in_flight: dict = {}
    reported, taken_ports = set(), set()
    while True:
        for label, (proc, spec) in list(in_flight.items()):
            if started(spec.out):
                log(f"{label}: sim pid {(spec.out / 'sim.pid').read_text().strip()} "
                    f"({(spec.out / 'slot').read_text().strip() if (spec.out / 'slot').exists() else '?'})")
                del in_flight[label]
            elif proc.poll() is not None:
                log(f"{label}: launcher exited {proc.returncode} without a sim (see {launch_log})")
                del in_flight[label]
        for spec in specs:
            if ended(spec.out) and spec.label not in reported:
                reported.add(spec.label)
                log(f"{spec.label} ended: {(result_line(spec.out) or 'NO RESULT LINE')[:200]}")
        if all(ended(s.out) for s in specs):
            log("queue done: every run has ended")
            return 0
        sims = live_slots() + len(in_flight)
        planners = planners_live() + sum(1 for _, s in in_flight.values() if s.planner)
        gpu = pick_gpu()
        nxt = next((s for s in pending if s.label not in in_flight), None)
        if nxt is not None and nxt.planner and not m2t2_up():
            log(f"{nxt.label}: M2T2 (127.0.0.1:8123) does not answer; not launching a live planner this tick")
            time.sleep(TICK)
            continue
        if nxt is not None and sims < max_sims and (not nxt.planner or planners < max_planners) and gpu is not None:
            port = next_port(taken_ports)
            taken_ports.add(port)
            log(f"launch {nxt.label} gpu {gpu} port {port} (sims {sims}, planners {planners}, gpu free {gpu_free()})")
            in_flight[nxt.label] = (launch(nxt, port, gpu, snap, launch_log), nxt)
            pending.remove(nxt)
            time.sleep(SETTLE)
            continue
        time.sleep(TICK)


# ------------------------------------------------------------------------------------------------- plan
def floor_of(task: str) -> dict:
    """The task's A/A floor as a request frame and a Runner record: W4-F1's (legacy_ref/aa.json) and, where W4-F2 ran
    extra legacy replays, the extended one (q1/g4.json); the minimum of the two is the floor the prefix check uses."""
    out = {"f1_frame": None, "f1_record": None, "extended_frame": None, "extended_record": None}
    aa = REF / "aa.json"
    if aa.exists():
        fl = (json.loads(aa.read_text()).get(task) or {}).get("floor") or {}
        out["f1_frame"], out["f1_record"] = fl.get("stream_frame"), fl.get("runner_record")
    g4 = Q1 / "g4.json"
    if g4.exists():
        fl = (json.loads(g4.read_text()).get(task) or {}).get("lrep_floor_extended") or {}
        out["extended_frame"], out["extended_record"] = fl.get("stream_frame"), fl.get("runner_record")
    frames = [f for f in (out["f1_frame"], out["extended_frame"]) if f is not None]
    records = [r for r in (out["f1_record"], out["extended_record"]) if r is not None]
    out["frame"], out["record"] = (min(frames) if frames else None), (min(records) if records else None)
    return out


def plan_stage(stage: str, tasks=TASKS, reps=REPS, tapes: bool = True, snap: Path = SNAP) -> dict:
    c_routes, t_routes = stage_routes(stage)
    pc, pt = profile_of(c_routes), profile_of(t_routes)
    plan = {"stage": stage, "line": STAGES[stage].line, "c_routes": c_routes, "t_routes": t_routes,
            "profiles": {"C": pc, "T": pt}, "snap": str(snap), "tasks": {}}
    for task in tasks:
        rec = LREC / task / "episode"
        tape_path = rec / "tapes" / f"{task}_{INSTANCE}_0.json"
        if not tape_path.exists():
            plan["tasks"][task] = {"error": f"no L-rec Runner tape at {tape_path}"}
            continue
        tape = tp.Tape.load(tape_path)
        writes = writes_of(tape)
        bc, bt = backends_of(writes, pc), backends_of(writes, pt)
        table = [{"i": w.i, "record": w.record, "member": w.member, "args": w.args, "kwargs": w.kwargs, "call_id": w.call_id,
                  "line": w.line, "backend_c": c, "backend_t": t, "returned": w.returned, "exc": w.exc, "step": w.step}
                 for w, c, t in zip(writes, bc, bt)]
        br = branch_of(writes, pc, pt)
        entry = {"tape": str(tape_path), "writes": table, "branch": None if br is None else asdict(br) | {"call_id": br.call_id},
                 "line_calls": sum(1 for w, c, t in zip(writes, bc, bt) if c != t), "floor": floor_of(task)}
        if br is not None:
            index_rows = read_jsonl(rec / "wstape" / "index.jsonl")
            js = rec / "json" / f"{task}_{INSTANCE}_0.json"
            rounds = (json.loads(js.read_text()).get("bench") or {}).get("rounds") if js.exists() else []
            n_tape, method = frame_of_branch(index_rows, writes, br, rounds)
            by_rounds, by_owner = frames_by_rounds(index_rows, rounds, writes), frames_by_write(index_rows, writes)
            erep = sorted((Q1 / "E-rep").glob(f"{task}_r*/episode/wstape_replay.jsonl")) if (Q1 / "E-rep").exists() else []
            n_erep = [frame_of_call(read_jsonl(p), br.call_id) for p in erep]
            n_erep = [n for n in n_erep if n is not None]
            entry.update({"frame_by_tape": n_tape, "frame_method": method, "frame_by_erep": sorted(set(n_erep)),
                          "frames_on_tape": len(index_rows), "frames_by_rounds": by_rounds, "frames_by_owner": by_owner,
                          "rounds_align_frames": bool(by_rounds)})
            frame = n_tape if n_tape is not None else (n_erep[0] if n_erep else None)
            entry["frame"] = frame
            entry["frame_agrees"] = None if not n_erep or n_tape is None else all(n == n_tape for n in n_erep)
            if tapes and frame is not None:
                entry["tapes"] = {str(rep): str(derived_tape(rec / "wstape", OUT / stage / "tapes" / f"{task}_r{rep}", rep, snap))
                                  for rep in reps}
                from omnigibson.tiptop.host.wstape import TapeDir

                first = TapeDir(rec / "wstape").load(0)
                entry["tape_reroot"] = [str(snapshot_root_of(first.metadata if first else None)), str(snap)]
        plan["tasks"][task] = entry
    return plan


# ------------------------------------------------------------------------------------------------- reading a run
def parse_result(line: Optional[str]) -> Optional[dict]:
    m = RESULT_RE.search(line or "")
    if not m:
        return None
    return {"q_score": float(m.group(2)), "success": m.group(3) == "True", "steps": int(m.group(4)),
            "max_steps": int(m.group(5)), "reason": m.group(6), "teleports": int(m.group(7)), "wall_s": float(m.group(8))}


def read_run(out: Path) -> dict:
    """Everything the gate reads from one run dir."""
    ep = out / "episode"
    inst = ep / f"{{task}}_{INSTANCE}_0"
    r = {"out": str(out), "ended": ended(out), "started": started(out), "result": None, "connector": None, "json": None}
    try:  # the snapshot the run was launched from (its planner's modules must be under it, D25)
        r["snap"] = json.loads((out / "spec.json").read_text()).get("snap")
    except (OSError, ValueError):
        r["snap"] = None
    lines = (out / "sim.log").read_text(errors="replace").splitlines() if (out / "sim.log").exists() else []
    r["result"] = parse_result(next((l for l in lines if "RESULT instance" in l), None))
    r["imports"] = next((l.split("INFO: ", 1)[1].strip() for l in lines if "INFO: imports:" in l), None)
    ws = [l.split("INFO: wstape: ", 1)[1].strip() for l in lines if "INFO: wstape: {" in l]
    r["wstape"] = json.loads(ws[0]) if ws else None
    r["switch_lines"] = [l.split("WARNING: ", 1)[-1].strip() for l in lines if "switching to live connections" in l]
    r["connector_line"] = next((l.split("INFO: connector: ", 1)[1].strip() for l in lines if "INFO: connector: " in l), None)
    r["crashes"] = [l for l in lines if "crashed" in l and "Traceback" not in l][:5]
    js = sorted((ep / "json").glob("*.json")) if (ep / "json").exists() else []
    if js:
        data = json.loads(js[0].read_text())
        r["json"] = {"steps": data.get("steps"), "success": data.get("success"), "reason": (data.get("bench") or {}).get("reason"),
                     "teleports": (data.get("bench") or {}).get("teleports"), "goal": (data.get("bench") or {}).get("goal"),
                     "rounds": (data.get("bench") or {}).get("rounds"), "video": (data.get("bench") or {}).get("video")}
        r["connector"] = (data.get("bench") or {}).get("connector")
    inst_dirs = sorted(p for p in ep.glob(f"*_{INSTANCE}_0") if p.is_dir()) if ep.exists() else []
    inst = inst_dirs[0] if inst_dirs else inst
    r["inst"] = str(inst)
    r["skill_calls"] = read_jsonl(inst / "skill_calls.jsonl")
    r["gripper"] = read_jsonl(inst / "gripper.jsonl")
    r["audit_ops"] = [x for x in read_jsonl(inst / "audit.jsonl") if x.get("kind") == "op"]
    r["replay"] = read_jsonl(ep / "wstape_replay.jsonl")
    r["live_index"] = read_jsonl(ep / "wstape_live" / "index.jsonl")
    r["record_index"] = read_jsonl(ep / "wstape" / "index.jsonl")
    tapes = sorted((ep / "tapes").glob("*.json")) if (ep / "tapes").exists() else []
    r["runner_tape"] = str(tapes[0]) if tapes else None
    # the live planner's own modules (D25): the first live frame's metadata, which the client received from the
    # real server when the run switched (the served tape's metadata is the recording's, rerooted)
    live = sorted((ep / "wstape_live" / "frames").glob("*.msgpack")) if (ep / "wstape_live" / "frames").exists() else []
    r["live_modules"] = None
    if live:
        from b1k.bridge.protocol import unpackb

        meta = unpackb(live[0].read_bytes()).get("metadata") or {}
        r["live_modules"] = {n: (meta.get("modules") or {}).get(n) for n in ("tiptop", "cutamp")}
    r["planner_log"] = str(out / "planner" / "planner.log") if (out / "planner" / "planner.log").exists() else None
    if r["planner_log"]:
        plines = Path(r["planner_log"]).read_text(errors="replace").splitlines()
        r["planner_imports"] = next((l for l in plines if "imports" in l.lower() and "tiptop" in l), None)
        r["planner_listening"] = any("server listening" in l for l in plines)
    r["video"] = str(ep / "videos" / f"{inst.name}.mp4") if (ep / "videos" / f"{inst.name}.mp4").exists() else None
    return r


def write_runner_tape_rows(out: Path) -> Optional[Path]:
    """counters.py's optional runner_tape.jsonl for a connector run, from its own Runner tape: the shim's call id per
    non-navigation write with the typed skill, its qualifier and the Runner-visible return."""
    ep = out / "episode"
    tapes = sorted((ep / "tapes").glob("*.json")) if (ep / "tapes").exists() else []
    inst = sorted(p for p in ep.glob(f"*_{INSTANCE}_0") if p.is_dir()) if ep.exists() else []
    if not tapes or not inst:
        return None
    rows = [{"call_id": w.call_id, "skill": w.skill, "qual": w.qual, "returned": w.returned, "member": w.member, "exc": w.exc}
            for w in writes_of(tp.Tape.load(tapes[0])) if w.k is not None]
    p = inst[0] / "runner_tape.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def tape_diff(snap: Path, a: Path, b: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": f"{snap}/OmniGibson:{snap}/tiptop", "OMNIGIBSON_HEADLESS": "1", "CUDA_VISIBLE_DEVICES": ""}
    cmd = [f"{snap}/b1k/bin/python", f"{snap}/OmniGibson/omnigibson/tiptop/scripts/tape_diff.py", str(a), str(b), "--json"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    try:
        return json.loads(r.stdout[r.stdout.index("{"):])
    except ValueError:
        return {"error": (r.stdout + r.stderr)[-2000:]}


# ------------------------------------------------------------------------------------------------- verdicts
def u0_of(c: Optional[dict]) -> dict:
    if not c:
        return {"exact": False, "note": "no connector block"}
    charged = c.get("charged") or {}
    ok = lambda k: bool((c.get(k) or {}).get("ok"))  # noqa: E731
    a = c.get("u0a") or {}
    # WEEK4_PLAN 5.6's amended identity: step - idle == sum(charged) + L + X, idle 0, and L and X both 0 unless the
    # episode ended in EpisodeOver (L: the steps of a run still live at the end; X: 1 iff it was raised in env_step)
    live, over, episode_over = int(a.get("L") or 0), int(a.get("X") or 0), bool(a.get("episode_over"))
    out = {"step": c.get("step"), "idle_steps": c.get("idle_steps"), "charged": charged,
           "identity": c.get("step") is not None and c.get("step") - (c.get("idle_steps") or 0) == sum(charged.values()) + live + over
           and not c.get("idle_steps") and (episode_over or (live == 0 and over == 0)),
           "u0a": ok("u0a"), "u0b": ok("u0b"), "u0c": ok("u0c"), "u0d": ok("u0d"), "rule2": ok("rule2"), "build": ok("build"),
           "hand_refresh_ok": bool((c.get("hand_refresh") or {}).get("ok", True)), "block_ok": bool(c.get("ok")),
           "L": a.get("L"), "X": a.get("X"), "episode_over": episode_over, "reason": c.get("reason")}
    out["exact"] = out["identity"] and out["u0a"] and out["u0b"] and out["u0c"] and out["u0d"] and out["build"]
    return out


def rule2_of(c: Optional[dict]) -> dict:
    """The block's rule 2, and the verdict recomputed from its ledger over every owner but ep.*, observe and go_to
    (a native skill's code runs unowned: a block from before the fix pass read ``rt`` alone)."""
    r2 = (c or {}).get("rule2") or {}
    outside = {o: {k: int(v.get(k) or 0) for k in RULE2_KEYS if int(v.get(k) or 0)}
               for o, v in ((c or {}).get("ledger") or {}).items()
               if not o.startswith("ep.") and o not in ("observe", "go_to")}
    outside = {o: v for o, v in outside.items() if v}
    return {"ok": bool(r2.get("ok")) and not outside, "rt": r2.get("rt"), "outside": outside,
            "d21_debt": r2.get("d21_debt")}


RULE2_KEYS = ("place_robot", "place_robot_calls", "capture", "look_at")


def live_imports_ok(run: dict, snap: Path) -> Optional[bool]:
    """D25 for a run that went live: the live planner's tiptop and cutamp under the snapshot. None: never live."""
    if (run.get("wstape") or {}).get("switched_at") is None and run.get("live_modules") is None:
        return None
    mods = run.get("live_modules") or {}
    root = str(Path(run.get("snap") or snap).resolve())  # the run's own snapshot (spec.json) when it names one
    return bool(mods) and all(isinstance(p, str) and str(Path(p).resolve()).startswith(root + "/") for p in mods.values())


NEUTRAL_READ = "distance"  # W4-F2 open item 1: the shim re-raises a KeyError with its own message where the Episode's
#                              carries the name; Runner.gap maps both to inf, so the records differ in prose alone


def neutral_pair(a: dict, b: dict) -> bool:
    """Two Runner-tape records that differ only in a way no decision reads: the same distance read, both raising
    KeyError, with different messages."""
    if a.get("kind") != "read" or b.get("kind") != "read" or a.get("member") != NEUTRAL_READ or b.get("member") != NEUTRAL_READ:
        return False
    if a.get("args") != b.get("args") or a.get("kwargs") != b.get("kwargs"):
        return False
    return exc_type(a.get("exc")) == "KeyError" and exc_type(b.get("exc")) == "KeyError"


def exc_type(e) -> Optional[str]:
    """The recorded exception's class name, from a raw dict or the Exc named tuple Tape.load decodes it into."""
    if e is None:
        return None
    return e.get("type") if isinstance(e, dict) else getattr(e, "type", None)


def runner_prefix_compare(lrec_records: list, run_records: list, branch_record: Optional[int]) -> dict:
    """The run's Runner tape against the L-rec's before the branch write's record: the first record that differs
    beyond a neutral pair, and the neutral pairs met on the way."""
    n = len(lrec_records) if branch_record is None else min(branch_record, len(lrec_records))
    neutral, first = [], None
    for i in range(n):
        if i >= len(run_records):
            first = {"index": i, "why": "the run's tape is shorter"}
            break
        a, b = lrec_records[i], run_records[i]
        if a == b:
            continue
        if neutral_pair(a, b):
            neutral.append(i)
            continue
        first = {"index": i, "kind": "decision" if a.get("kind") == "write" or a.get("member") != b.get("member") or a.get("args") != b.get("args") else "answer",
                 "member": a.get("member"), "a": json.dumps(a, default=str)[:300], "b": json.dumps(b, default=str)[:300]}
        break
    return {"compared": n, "first_divergence": first, "neutral": neutral, "identical_before_branch": first is None}


def prefix_verdict(run: dict, frame: Optional[int], floor_frame: Optional[int], branch_record: Optional[int],
                   lrec_tape: Optional[Path], snap: Path, branch_write: Optional[int] = None) -> dict:
    """(a): the served frames before N matched and the switch happened at N (forced), else within the A/A floor;
    the Runner tape identical to the L-rec's before the branch write's record."""
    ws, replay = run.get("wstape") or {}, run.get("replay") or []
    switched_at, reason = ws.get("switched_at"), ws.get("switch_reason") or ""
    before = [row for row in replay if frame is not None and int(row["i"]) < frame]
    mism = [row for row in before if not row.get("matched", True)]
    # the switch lands at N either way: C's legacy request there is forced live, T's native request differs from the
    # tape at its op (skill or reach against plan) and goes live on that
    out = {"frame": frame, "switched_at": switched_at, "switch_reason": reason, "frames_before_branch": len(before),
           "mismatched_before_branch": [{"i": r["i"], "diffs": [d.get("path") for d in r.get("diffs", [])][:5]} for r in mism],
           "switched_at_branch": switched_at == frame, "forced": "forced" in reason, "floor_frame": floor_frame}
    early = switched_at is not None and frame is not None and switched_at < frame
    out["pre_branch_divergence"] = early or bool(mism)
    out["within_floor"] = (not out["pre_branch_divergence"]) or (floor_frame is not None and switched_at is not None and switched_at >= floor_frame)
    out["runner_prefix"] = None
    if lrec_tape is not None and run.get("runner_tape"):
        d = tape_diff(snap, lrec_tape.parent.parent, Path(run["runner_tape"]).parent.parent)
        runner = d.get("runner") or {}
        fd = runner.get("first_divergence")
        idx = None if fd is None else fd.get("index")
        pw = runner.get("per_write") or []
        before_branch = [w for w in pw if branch_write is not None and w.get("i", 0) < branch_write]
        cmp = runner_prefix_compare(tp.Tape.load(lrec_tape).records, tp.Tape.load(run["runner_tape"]).records, branch_record)
        out["runner_prefix"] = {"first_divergence_raw": idx, "kind_raw": None if fd is None else fd.get("kind"),
                                "records": runner.get("records"), "branch_record": branch_record,
                                "first_divergence": cmp["first_divergence"], "neutral_pairs": cmp["neutral"],
                                "identical_before_branch": cmp["identical_before_branch"],
                                "writes_before_branch": len(before_branch),
                                "writes_before_branch_equal": all(w["delta_a"] == w["delta_b"] and w["digest_before_equal"]
                                                                  and w["digest_after_equal"] for w in before_branch)}
    rp = out["runner_prefix"]
    # the evidence must be there and say what (a) claims: every frame before N served and matched, the switch at N
    # (or, a divergence before it, at or past the A/A floor), the Runner tape compared, and every write before the
    # branch compared on both sides
    served_all = frame is not None and len(before) == frame
    out["served_all_before_branch"] = served_all
    out["writes_before_branch_complete"] = rp is not None and branch_write is not None and rp["writes_before_branch"] == branch_write
    out["ok"] = bool(out["within_floor"] and rp is not None and rp["identical_before_branch"] and rp["writes_before_branch_equal"]
                     and out["writes_before_branch_complete"]
                     and ((served_all and out["switched_at_branch"]) or (out["pre_branch_divergence"] and out["within_floor"]
                                                                          and floor_frame is not None))
                     and not (out["pre_branch_divergence"] and floor_frame is None))
    return out


def hard_verdict(run: dict, switched: Optional[str], arm: str, c_reasons: set, snap: Path = SNAP) -> dict:
    """(b) on one run. A C/T arm fails on a crash reason C lacks; a carried or witness run (no control) on any
    crash. G3's own hard items ride along: no typed/literal mismatch, the Runner's inputs equal, no write the shim
    put on the channel that never ran (a native route's NO_STANCE_HERE with no stance to reach: no skill row, so the
    pooled test would never see it), and a live planner from the snapshot (D25)."""
    c = run.get("connector")
    u0, r2 = u0_of(c), rule2_of(c)
    reason = (run.get("json") or {}).get("reason") or ""
    crash = reason.startswith("crash") or reason.startswith("blocked")
    rows = run.get("skill_calls") or []
    bad = []
    if switched:
        name, _, qual = switched.partition(".")
        for row in rows:
            if row.get("skill") != name or row.get("backend") == "legacy":
                continue
            if str(row.get("code") or "").lower() in UNSUPPORTED_CODES:
                bad.append({"call_id": row.get("call_id"), "code": row.get("code"), "detail": row.get("detail")})
    on_air = [{k: r.get(k) for k in ("step", "arm", "owner", "call_id", "via", "is_grasping")}
              for r in run.get("gripper") or [] if r.get("event") == "close" and r.get("is_grasping") == -1]
    g3 = (c or {}).get("g3") or {}
    shim_counts = (c or {}).get("shim") or {}
    out = {"u0": u0, "rule2": r2, "reason": reason, "crash": crash, "crash_c_lacks": bool(crash and arm == "T" and reason not in c_reasons),
           "crash_uncontrolled": bool(crash and arm not in ("C", "T")),
           "switched_unsupported": bad, "on_air": on_air, "on_air_explained": all(x["owner"] and (x["call_id"] or str(x["owner"]).startswith("ep.")) for x in on_air),
           "hand_refresh": (c or {}).get("hand_refresh"), "epochs": (c or {}).get("epochs"),
           "typed_literal_mismatches": g3.get("typed_literal_mismatches"), "runner_inputs_equal": g3.get("runner_inputs_equal"),
           "unconsumed": shim_counts.get("unconsumed"), "live_imports_ok": live_imports_ok(run, snap)}
    out["ok"] = bool(u0["exact"] and r2["ok"] and u0["hand_refresh_ok"] and not out["crash_c_lacks"] and not out["crash_uncontrolled"]
                     and not bad and out["on_air_explained"] and out["typed_literal_mismatches"] == 0
                     and out["runner_inputs_equal"] is True and not out["unconsumed"] and out["live_imports_ok"] is not False)
    return out


def pairing(plan_task: dict, runs: dict, switched: str) -> list:
    """The switched skill call by call, C against T at the same Runner write (the shim's call id), per replicate."""
    name, _, qual = switched.partition(".")
    rows = []
    for w in plan_task.get("writes") or []:
        if w["line"] != switched:
            continue
        row = {"call_id": w["call_id"], "write": w["i"], "member": w["member"], "args": w["args"], "lrec_returned": w["returned"]}
        for (arm, rep), run in sorted(runs.items()):
            r = next((x for x in run.get("skill_calls") or [] if x.get("call_id") == w["call_id"]), None)
            if r is None:
                row[f"{arm}{rep}"] = None
            else:
                row[f"{arm}{rep}"] = {"skill": r.get("skill"), "backend": r.get("backend"), "status": r.get("status"), "code": r.get("code"),
                                      "steps": r.get("steps"), "scorer": (r.get("verdicts") or {}).get("scorer"),
                                      "returned": (r.get("evidence") or {}).get("legacy_ok") if r.get("backend") == "legacy" else r.get("status") == "succeeded",
                                      "same_write": r.get("skill") == name}
        rows.append(row)
    return rows


def cut_off_delivery(run: dict, switched: str) -> Optional[dict]:
    """A switched-skill write that EpisodeOver cut off at the task's success: it delivered the goal but never
    returned to the Runner and never reached a skill row (the Runtime never finished it), on either arm alike. The
    pre-registered per-call test leaves it out; the secondary line counts it as a success."""
    if not run.get("runner_tape") or ((run.get("json") or {}).get("reason") != "success"):
        return None
    ws = writes_of(tp.Tape.load(run["runner_tape"]))
    last = next((w for w in reversed(ws) if w.k is not None), None)
    if last is None or last.line != switched or last.exc != "EpisodeOver":
        return None
    return {"call_id": last.call_id, "member": last.member, "args": last.args, "line": last.line}


def ordinal_pairing(runs: dict, switched: str) -> list:
    """The switched skill's calls by their order within each run (the first place.on, the second, ...): once the
    arms' write sequences part after the branch, the call ids no longer name the same Runner write, and the ordinal
    is what "the same write" means. Each cell: backend, status/code, steps, the scorer's verdict, the return."""
    name, _, qual = switched.partition(".")
    per_run = {}
    for (arm, rep), run in sorted(runs.items()):
        rows = []
        tape = {r["call_id"]: r for r in read_jsonl(Path(run["inst"]) / "runner_tape.jsonl")} if run.get("inst") else {}
        for r in run.get("skill_calls") or []:
            if r.get("skill") != name:
                continue
            q = (tape.get(r.get("call_id")) or {}).get("qual")
            if q is None:
                quals = {counters.PLACE_PREDS[e["pred"]] for e in r.get("effects") or () if e.get("pred") in counters.PLACE_PREDS}
                q = quals.pop() if len(quals) == 1 else None
            if qual and q != qual:
                continue
            rows.append({"call_id": r.get("call_id"), "backend": r.get("backend"), "status": r.get("status"), "code": r.get("code"),
                         "steps": r.get("steps"), "scorer": (r.get("verdicts") or {}).get("scorer"),
                         "returned": (r.get("evidence") or {}).get("legacy_ok") if r.get("backend") == "legacy" else r.get("status") == "succeeded",
                         "target": (tape.get(r.get("call_id")) or {}).get("args")})
        co = run.get("cut_off_delivery")
        if co and co.get("line") == switched:
            rows.append({"call_id": co["call_id"], "backend": "(cut off at success)", "status": "cut_off", "code": None, "steps": None,
                         "scorer": True, "returned": None})
        per_run[f"{arm}{rep}"] = rows
    n = max((len(v) for v in per_run.values()), default=0)
    return [{"ordinal": j + 1, **{k: (v[j] if j < len(v) else None) for k, v in per_run.items()}} for j in range(n)]


def switched_failed_where_c_succeeded(ordinal: list) -> list:
    """The cause the stage fails on: at an ordinal where a C run's switched call succeeded, a T run's failed. A C row
    EpisodeOver cut off at the task's success delivered its goal, so it counts as a success."""
    out = []
    for row in ordinal:
        c_ok = [(v["returned"] is True and v["scorer"] is True) or v["status"] == "cut_off"
                for k, v in row.items() if k[:1] == "C" and isinstance(v, dict)]
        t = [(k, v) for k, v in row.items() if k[:1] == "T" and isinstance(v, dict)]
        t_bad = [(k, v["code"]) for k, v in t if not (v["returned"] is True and v["scorer"] is True) and v["status"] != "cut_off"]
        if c_ok and any(c_ok) and t_bad:
            out.append({"ordinal": row["ordinal"], "c_ok": c_ok, "t_failed": t_bad})
    return out


def on_air_causes(run: dict) -> list:
    """(b)'s on-air closes (GripperWatch, is_grasping -1), each with its owning call's skill row (skill, backend,
    status/code) and whether a later close inside the same call held (the call's own next round)."""
    rows = {r.get("call_id"): r for r in run.get("skill_calls") or []}
    grip = run.get("gripper") or []
    out = []
    for i, g in enumerate(grip):
        if g.get("event") != "close" or g.get("is_grasping") != -1:
            continue
        r = rows.get(g.get("call_id")) or {}
        out.append({"step": g.get("step"), "arm": g.get("arm"), "owner": g.get("owner"), "call_id": g.get("call_id"),
                    "skill": r.get("skill"), "backend": r.get("backend"), "status": r.get("status"), "code": r.get("code"),
                    "native": bool(r) and r.get("backend") != "legacy",
                    "held_later_in_the_call": any(x.get("event") == "close" and x.get("is_grasping") == 1
                                                  and x.get("call_id") == g.get("call_id") for x in grip[i + 1:])})
    return out


def after_native(run: dict) -> list:
    """Each native skill row with the phase it ended in, then the legacy rows up to the next native one: a posture a
    native run leaves behind shows where the next legacy call pays for it."""
    rows = run.get("skill_calls") or []
    out = []
    for i, r in enumerate(rows):
        if r.get("backend") == "legacy":
            continue
        following = []
        for x in rows[i + 1:]:
            if x.get("backend") != "legacy":
                break
            following.append((x.get("call_id"), x.get("skill"), x.get("status"), x.get("code"), x.get("steps")))
        out.append({"call_id": r.get("call_id"), "skill": r.get("skill"), "status": r.get("status"), "code": r.get("code"),
                    "phase": r.get("phase"), "steps": r.get("steps"), "then": following})
    return out


def legacy_dependencies(run: dict) -> dict:
    """What still ran on legacy in a run: skill rows by skill and qualifier, the qualifier the Runner tape's typed
    call names (runner_tape.jsonl: a failed place has no effects), else the effects'."""
    tape = {r["call_id"]: r for r in read_jsonl(Path(run["inst"]) / "runner_tape.jsonl")} if run.get("inst") else {}
    out = Counter()
    for r in run.get("skill_calls") or []:
        if r.get("backend") == "legacy":
            q = (tape.get(r.get("call_id")) or {}).get("qual")
            if q is None:
                quals = {counters.PLACE_PREDS[e["pred"]] for e in r.get("effects") or () if e.get("pred") in counters.PLACE_PREDS}
                q = quals.pop() if len(quals) == 1 else ""
            out[r.get("skill") + (f".{q}" if q else "")] += 1
    return dict(out)


def write_sequence(run: dict) -> list:
    """The Runner's writes as (call_id, member, line) from the run's Runner tape, for the structural shifts (e)."""
    if not run.get("runner_tape"):
        return []
    return [(w.call_id, w.member, w.line, w.returned, w.exc) for w in writes_of(tp.Tape.load(run["runner_tape"]))]


def structural_shifts(runs: dict, branch_write: int) -> list:
    """T's write sequence after the branch against C's, per replicate: where T made a write C did not (or in another
    order), named by the shim's call and the Runner member."""
    out = []
    reps = sorted({rep for (_, rep) in runs})
    for rep in reps:
        c, t = runs.get(("C", rep)), runs.get(("T", rep))
        if c is None or t is None:
            continue
        sc, st = write_sequence(c)[branch_write:], write_sequence(t)[branch_write:]
        lc, lt = [(m, l) for _, m, l, _, _ in sc], [(m, l) for _, m, l, _, _ in st]
        first = next((i for i, (a, b) in enumerate(zip(lc, lt)) if a != b), None if len(lc) == len(lt) else min(len(lc), len(lt)))
        ct, cc = Counter(lt), Counter(lc)  # a multiset: an extra repeat of a write both arms made is an extra too
        out.append({"rep": rep, "c_writes_after_branch": lc, "t_writes_after_branch": lt, "first_difference": first,
                    "first_difference_write": None if first is None else branch_write + first,
                    "t_extra": sorted((ct - cc).elements()), "c_extra": sorted((cc - ct).elements())})
    return out


# ------------------------------------------------------------------------------------------------- gate: a stage
def gate_stage(stage: str, snap: Path = SNAP) -> dict:
    plan_path = OUT / stage / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else plan_stage(stage, tapes=False)
    switched = STAGES[stage].line.split("=")[0]
    backend = STAGES[stage].line.split("=")[1]
    g = {"stage": stage, "line": STAGES[stage].line, "switched": switched, "snap": str(snap), "tasks": {}, "runs": {}}
    pairs, per_task = {}, {}
    stage_tasks = STAGES[stage].tasks
    # a stage task whose L-rec calls the line but is carried forward (its branch lies past its A/A floor) is a
    # deviation from plan 5.8, which runs arms on every task that calls the line: named, never silent
    g["deviations"] = [{"task": t, "line_calls": p.get("line_calls"), "frame": p.get("frame"),
                        "floor_frame": (p.get("floor") or {}).get("frame")}
                       for t, p in plan["tasks"].items()
                       if p.get("branch") is not None and stage_tasks and t not in stage_tasks and p.get("line_calls")]
    for task, p in plan["tasks"].items():
        if p.get("branch") is None:
            g["tasks"][task] = {"branch": None, "note": "no write routed differently: carried forward"}
            continue
        if stage_tasks and task not in stage_tasks:
            fl = (p.get("floor") or {}).get("frame")
            g["tasks"][task] = {"branch": p["branch"], "frame": p.get("frame"), "note":
                                f"not a stage task: carried forward (branch frame {p.get('frame')}, A/A floor frame {fl}"
                                + (": the branch lies past the floor, so both arms would go live before it)" if fl is not None and p.get("frame") is not None and p["frame"] > fl else ")")}
            continue
        runs = {}
        for rep in REPS:
            for arm in ("C", "T"):
                d = run_dir(stage, task, arm, rep)
                if d.exists():
                    write_runner_tape_rows(d)
                    runs[(arm, rep)] = read_run(d)
        if not runs:
            g["tasks"][task] = {"branch": p["branch"], "note": "no runs"}
            continue
        lrec_tape = Path(p["tape"])
        floor = p.get("floor") or {}
        c_reasons = {(r.get("json") or {}).get("reason") for (a, _), r in runs.items() if a == "C"}
        t_entry = {"branch": p["branch"], "frame": p.get("frame"), "floor": floor, "runs": {}}
        for (arm, rep), run in sorted(runs.items()):
            key = f"{arm}{rep}"
            pv = prefix_verdict(run, p.get("frame"), floor.get("frame"), (p["branch"] or {}).get("record"), lrec_tape, snap,
                                (p["branch"] or {}).get("write"))
            hv = hard_verdict(run, switched, arm, c_reasons, snap)
            t_entry["runs"][key] = {"out": run["out"], "ended": run["ended"], "result": run["result"], "prefix": pv, "hard": hv,
                                    "cut_off_delivery": cut_off_delivery(run, switched),
                                    "imports": run.get("imports"), "planner_imports": run.get("planner_imports"), "video": run.get("video"),
                                    "legacy": legacy_dependencies(run), "hand_refresh": (run.get("connector") or {}).get("hand_refresh"),
                                    "d21_debt": rule2_of(run.get("connector")).get("d21_debt"), "calls": (run.get("connector") or {}).get("calls"),
                                    "charged": (run.get("connector") or {}).get("charged"), "live_index": [(x.get("i"), x.get("op"), x.get("owner"), x.get("call_id")) for x in run.get("live_index") or []][:40],
                                    "on_air_causes": on_air_causes(run), "after_native": after_native(run)}
            g["runs"][f"{task}/{key}"] = t_entry["runs"][key]
        for (arm, rep), run in runs.items():
            run["cut_off_delivery"] = cut_off_delivery(run, switched)
        t_entry["pairing"] = pairing(p, runs, switched)
        t_entry["ordinal"] = ordinal_pairing(runs, switched)
        t_entry["switched_failed_where_c_succeeded"] = switched_failed_where_c_succeeded(t_entry["ordinal"])
        t_entry["structural"] = structural_shifts({k: r for k, r in runs.items() if r["ended"]}, (p["branch"] or {}).get("write", 0))
        ended_ = [(a, rep, run) for (a, rep), run in sorted(runs.items()) if run["ended"]]
        C = [counters.extract(Path(run["out"])) for a, _, run in ended_ if a == "C"]
        T = [counters.extract(Path(run["out"])) for a, _, run in ended_ if a == "T"]
        pairs[task] = (C, T)
        t_entry["counters"] = {"C": [c.as_dict() for c in C], "T": [t.as_dict() for t in T]}
        t_entry["counters_keys"] = {arm: [f"{a}{rep}" for a, rep, _ in ended_ if a == arm] for arm in ("C", "T")}
        per_task[task] = t_entry
        g["tasks"][task] = t_entry
    cmp = counters.compare(pairs, switched, backend=None) if pairs else None
    g["compare"] = None if cmp is None else counters._comparison_dict(cmp)
    g["compare_text"] = None if cmp is None else counters.format_comparison(cmp)
    # the secondary line: the strict counts plus the cut-off deliveries (a success each)
    if cmp is not None and cmp.pooled is not None:
        cut = {"C": 0, "T": 0}
        for task, t in per_task.items():
            for key, r in t["runs"].items():
                co = r.get("cut_off_delivery")
                if co:
                    cut[key[0]] += 1
        pl = cmp.pooled
        c_n, c_s, t_n, t_s = pl.c_n + cut["C"], pl.c_succ + cut["C"], pl.t_n + cut["T"], pl.t_succ + cut["T"]
        p = counters.fisher_one_sided(c_s, c_n, t_s, t_n)
        drop = (c_s / c_n - t_s / t_n) if c_n and t_n else 0.0
        g["compare_with_cut_off"] = {"cut_off_C": cut["C"], "cut_off_T": cut["T"], "c_n": c_n, "c_succ": c_s, "t_n": t_n, "t_succ": t_s,
                                     "c_rate": c_s / c_n if c_n else None, "t_rate": t_s / t_n if t_n else None, "p": p,
                                     "mde": counters.minimum_detectable_effect(c_s, c_n, t_n, cmp.alpha, cmp.delta),
                                     "fail": bool(c_n and t_n and drop >= cmp.delta and p < cmp.alpha)}
    # the causes of the flags, from the pairing: a T failure at a write C's run succeeded on
    flags_fail = []
    if cmp is not None:
        for f in cmp.flagged:
            t_entry = per_task.get(f.task) or {}
            flags_fail.append({"task": f.task, "counter": f.counter, "control": f.control, "treatment": f.treatment,
                               "switched_failed_where_c_succeeded": t_entry.get("switched_failed_where_c_succeeded") or []})
    g["flags"] = flags_fail
    vacuous = not any(p.get("branch") is not None and (not stage_tasks or t in stage_tasks) for t, p in plan["tasks"].items())
    verdict = {
        "vacuous": vacuous,  # no stage task routes any write differently between C and T: nothing to run or judge
        "a_prefix": all(r["prefix"]["ok"] for t in per_task.values() for r in t["runs"].values() if r["ended"]),
        "b_hard": all(r["hard"]["ok"] for t in per_task.values() for r in t["runs"].values() if r["ended"]),
        # an unjudged pooled test (a qualified line with unqualified rows, or an arm with no call) is never a pass
        "c_primary": None if cmp is None or cmp.pooled is None else bool(cmp.pooled.judged and not cmp.pooled.fail),
        "c_judged": None if cmp is None or cmp.pooled is None else cmp.pooled.judged,
        "d_flags_fail": any(x["switched_failed_where_c_succeeded"] for x in flags_fail),
        "runs_ended": sum(1 for t in per_task.values() for r in t["runs"].values() if r["ended"]),
        # two arms x the replicates on every task with a branch, whether or not its run dir exists yet
        "runs_expected": 2 * len(REPS) * sum(1 for t, p in plan["tasks"].items() if p.get("branch") is not None
                                             and (not stage_tasks or t in stage_tasks)),
    }
    verdict["pass"] = bool(verdict["a_prefix"] and verdict["b_hard"] and verdict["c_primary"] is True and not verdict["d_flags_fail"]
                           and verdict["runs_ended"] == verdict["runs_expected"] and verdict["runs_ended"] > 0)
    verdict["outcome"] = "VACUOUS" if vacuous else ("PASS" if verdict["pass"] else "FAIL")
    g["verdict"] = verdict
    return g


def gate_carried(snap: Path = SNAP) -> dict:
    out = {"runs": {}}
    for task in CARRIED:
        d = run_dir("carried", task, "native", 0)
        if not d.exists():
            continue
        write_runner_tape_rows(d)
        run = read_run(d)
        c = counters.extract(d) if run["ended"] else None
        out["runs"][task] = {"out": str(d), "ended": run["ended"], "result": run["result"], "hard": hard_verdict(run, None, "native", set(), snap),
                             "native_calls": None if c is None else c.native_calls, "native_by": None if c is None else c.native_by,
                             "legacy": legacy_dependencies(run), "calls": (run.get("connector") or {}).get("calls"),
                             "hand_refresh": (run.get("connector") or {}).get("hand_refresh"), "imports": run.get("imports"),
                             "planner_imports": run.get("planner_imports"), "video": run.get("video"),
                             "routing_profile": (run.get("connector") or {}).get("routing_profile"), "routes": (run.get("connector") or {}).get("routes"),
                             "native_rows": [{k: r.get(k) for k in ("call_id", "skill", "backend", "status", "code", "phase", "steps")}
                                             for r in run.get("skill_calls") or [] if r.get("backend") != "legacy"],
                             "after_native": after_native(run), "on_air_causes": on_air_causes(run),
                             "d21_debt": rule2_of(run.get("connector")).get("d21_debt")}
    runs = out["runs"].values()
    out["verdict"] = {"runs_ended": sum(1 for r in runs if r["ended"]), "runs_expected": len(CARRIED),
                      "hard_ok": all(r["hard"]["ok"] for r in runs if r["ended"]),
                      "native_calls": {t: len(r["native_rows"]) for t, r in out["runs"].items() if r["ended"]}}
    return out


def carried_md(g: dict) -> str:
    v = g["verdict"]
    o = ["# Carried-forward tasks (W4-I, WEEK4_PLAN 5.8): one NATIVE full-profile run each, replicate 0", "",
         f"Profile: NATIVE (routing.yaml) plus the ladder's lines `{' '.join(carried_routes())}`, live, from the snapshot. "
         f"Runs ended {v['runs_ended']}/{v['runs_expected']}; hard items on every ended run {v['hard_ok']}; "
         f"native calls per task {v['native_calls']} (each reported below).", ""]
    causes = OUT / "carried" / "causes.md"
    if causes.exists():
        o += [causes.read_text().rstrip(), ""]
    o += ["| task | ended | RESULT | U0 exact | rule2 | hand refresh | on-air | native calls | legacy deps | score |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for task, r in g["runs"].items():
        res, hv, hr = r["result"], r["hard"], r.get("hand_refresh") or {}
        res_s = "" if not res else "{}/{} ({}) tp {}".format(res["steps"], res["max_steps"], res["reason"], res["teleports"])
        o.append(f"| {task} | {r['ended']} | {res_s} | {hv['u0']['exact']} | {hv['rule2']['ok']} | {hr.get('count')} / {hr.get('popped_total')} | "
                 f"{len(hv['on_air'])} | {len(r['native_rows'])} | {r['legacy']} | {'' if not res else res['q_score']} |")
    o.append("")
    for task, r in g["runs"].items():
        o += [f"## {task}", "", f"- run `{r['out']}`; routes {r.get('routes')}; profile {r.get('routing_profile')}",
              f"- imports: `{r.get('imports')}`", f"- video: `{r.get('video')}`", f"- D21 debt: {r.get('d21_debt')}"]
        o += [f"- native: `{x['call_id']}` {x['skill']} on {x['backend']} {x['status']}/{x['code']} phase {x['phase']} steps {x['steps']}"
              for x in r["native_rows"]] or ["- native calls: none"]
        o += [f"- after `{x['call_id']}` (phase {x['phase']}): {x['then'] or 'nothing'}" for x in r["after_native"]]
        o += [f"- on-air close step {x['step']}: `{x['owner']}` `{x['call_id']}` = {x['skill']} on {x['backend']} ({x['status']}/{x['code']})"
              for x in r["on_air_causes"]]
        o.append("")
    return "\n".join(o)


def gate_witness(snap: Path = SNAP) -> dict:
    d = run_dir("witness", WITNESS_TASK, "witness", 0)
    if not d.exists():
        return {"out": str(d), "ended": False}
    write_runner_tape_rows(d)
    run = read_run(d)
    c = run.get("connector") or {}
    rows = run.get("skill_calls") or []
    ops = run.get("audit_ops") or []
    # the i-th run op is the i-th skill row; an observe op between the previous run op and the press's is the observe
    run_ops = [i for i, o in enumerate(ops) if o.get("op") == "run"]
    press_rows = [i for i, r in enumerate(rows) if r.get("skill") == "press" and r.get("backend") != "legacy"]
    observe_before_press = []
    for pi in press_rows:
        if pi < len(run_ops):
            lo = run_ops[pi - 1] if pi > 0 else -1
            observe_before_press.append(any(o.get("op") == "observe" for o in ops[lo + 1:run_ops[pi]]))
    native_rows = [r for r in rows if r.get("backend") != "legacy"]
    out = {"out": str(d), "ended": run["ended"], "result": run["result"], "u0": u0_of(c), "rule2": rule2_of(c),
           "charged": c.get("charged"), "calls": c.get("calls"), "routes": c.get("routes"), "routing_profile": c.get("routing_profile"),
           "press_rows": [{"call_id": rows[i].get("call_id"), "backend": rows[i].get("backend"), "status": rows[i].get("status"),
                           "code": rows[i].get("code"), "steps": rows[i].get("steps")} for i in press_rows],
           "observe_before_native_press": observe_before_press, "native_rows": len(native_rows),
           "hand_refresh": c.get("hand_refresh"), "requests": c.get("requests"), "imports": run.get("imports"),
           "planner_imports": run.get("planner_imports"), "video": run.get("video"), "legacy": legacy_dependencies(run),
           "ledger": {k: {kk: v.get(kk) for kk in ("steps", "capture", "place_robot", "look_at")} for k, v in (c.get("ledger") or {}).items()}}
    reason = ((run.get("json") or {}).get("reason") or "")
    out["reason"] = reason
    out["ok"] = bool(out["u0"]["exact"] and out["rule2"]["ok"] and (out["charged"] or {}).get("wait", 0) > 0
                     and press_rows and len(observe_before_press) == len(press_rows) and all(observe_before_press)
                     and not reason.startswith(("crash", "blocked")))
    return out


def gate_strict(snap: Path = SNAP) -> dict:
    """The strict E-rep against W4-F2's: the same comparison F2 made (Runner tape, request stream, rounds, RESULT)."""
    d = run_dir("strict", STRICT_TASK, "strict", 0)
    if not d.exists():
        return {"out": str(d), "ended": False}
    run, ref = read_run(d), read_run(STRICT_REF)
    diff = tape_diff(snap, STRICT_REF / "episode", d / "episode")
    lrec = tape_diff(snap, LREC / STRICT_TASK / "episode", d / "episode")
    strip = lambda rounds: [{k: v for k, v in r.items() if k not in ("seconds", "dir")} for r in rounds or []]  # noqa: E731
    keys = ("steps", "teleports", "q_score", "reason")
    out = {"out": str(d), "ref": str(STRICT_REF), "ended": run["ended"], "result": run["result"], "ref_result": ref["result"],
           "result_equal": bool(run["result"] and ref["result"] and all(run["result"][k] == ref["result"][k] for k in keys)),
           "rounds_equal": strip((run.get("json") or {}).get("rounds")) == strip((ref.get("json") or {}).get("rounds")),
           "runner_tape_vs_f2": (diff.get("runner") or {}).get("first_divergence"), "runner_records": (diff.get("runner") or {}).get("records"),
           # both runs are replays of the same L-rec tape: each one's own log against that tape (tape_diff reads both)
           "stream_vs_f2": (diff.get("stream") or {}).get("first_divergence"),
           "stream_replays": (diff.get("stream") or {}).get("replays"),
           "runner_tape_vs_lrec": (lrec.get("runner") or {}).get("first_divergence"), "stream_vs_lrec": (lrec.get("stream") or {}).get("first_divergence"),
           "replay": [(r.get("i"), r.get("op"), r.get("call_id"), r.get("matched")) for r in run.get("replay") or []],
           "u0": u0_of(run.get("connector")), "hand_refresh": (run.get("connector") or {}).get("hand_refresh"),
           "imports": run.get("imports"), "video": run.get("video")}
    out["identical"] = bool(out["result_equal"] and out["rounds_equal"] and out["runner_records"] and out["runner_tape_vs_f2"] is None
                            and out["stream_vs_f2"] is None and out["stream_replays"] and out["u0"]["exact"]
                            and all(m for _, _, _, m in out["replay"]))
    return out


# ------------------------------------------------------------------------------------------------- GATE.md
def _f(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def gate_md(g: dict) -> str:
    v = g["verdict"]
    o = [f"# {g['stage']}: `{g['line']}` (W4-I switch ladder, WEEK4_PLAN 5.8)", "",
         f"Snapshot `{g['snap']}`. Switched skill `{g['switched']}`. Verdict: **{v.get('outcome') or ('PASS' if v['pass'] else 'FAIL')}** "
         f"(a prefix {v['a_prefix']}, b hard {v['b_hard']}, c primary {v['c_primary']}, "
         f"d flag-caused-by-switch {v['d_flags_fail']}, runs {v['runs_ended']}/{v['runs_expected']}).", ""]
    if v.get("vacuous"):
        o += ["VACUOUS: no stage task routes any Runner write differently between C and T, so there is no arm pair to "
              "run and nothing to judge (not a pass, not a failure).", ""]
    for d in g.get("deviations") or []:
        o += [f"Deviation from WEEK4_PLAN 5.8 (arms on every task that calls the line): {d['task']} calls it "
              f"{d['line_calls']} time(s) in its L-rec but is carried forward (branch frame {d['frame']}, A/A floor frame "
              f"{d['floor_frame']}).", ""]
    causes = OUT / g["stage"] / "causes.md"
    if causes.exists():
        o += [causes.read_text().rstrip(), ""]
    if g.get("compare_text"):
        o += ["## (c) primary and (d) counters (counters.py compare)", "", "```", g["compare_text"], "```", ""]
        cc = g.get("compare_with_cut_off")
        if cc:
            o += [f"Secondary (the strict counts plus the switched-skill writes EpisodeOver cut off at success, one success each; "
                  f"they never reach a skill row on either arm): C {cc['c_succ']}/{cc['c_n']} ({_f(cc['c_rate'])}), T {cc['t_succ']}/{cc['t_n']} "
                  f"({_f(cc['t_rate'])}), one-sided Fisher p {cc['p']:.3f}, MDE {_f(cc['mde'])}, {'FAIL' if cc['fail'] else 'pass'}; "
                  f"cut-off deliveries C {cc['cut_off_C']}, T {cc['cut_off_T']}.", ""]
    for task, t in g["tasks"].items():
        o.append(f"## {task}")
        if t.get("branch") is None or "runs" not in t:
            o += [f"- {t.get('note')}" + ("" if t.get("branch") is None else f" (branch {t['branch']})"), ""]
            continue
        b = t["branch"]
        o += [f"- branch: write {b['write']} `{b['member']}` ({b['line']}) call `{b['call_id']}`, C -> {b['backend_c']}, T -> {b['backend_t']}; "
              f"forced live at frame {t['frame']}; A/A floor frame {t['floor'].get('frame')} (F1 {t['floor'].get('f1_frame')}, extended {t['floor'].get('extended_frame')})", ""]
        o += ["| run | ended | RESULT | prefix (a) | switched_at | hard (b) | U0 exact | rule2 | live planner (D25) | crash | on-air | hand refresh | legacy deps | score |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, r in t["runs"].items():
            res, pv, hv = r["result"], r["prefix"], r["hard"]
            res_s = "" if not res else "{}/{} ({}) tp {}".format(res["steps"], res["max_steps"], res["reason"], res["teleports"])
            within = " (within floor)" if pv["pre_branch_divergence"] and pv["within_floor"] else ""
            hr = r.get("hand_refresh") or {}
            o.append(f"| {key} | {r['ended']} | {res_s} | {pv['ok']}{within} | {pv['switched_at']} | {hv['ok']} | "
                     f"{hv['u0']['exact']} | {hv['rule2']['ok']} | {hv.get('live_imports_ok')} | {hv['reason'] if hv['crash'] else '-'} | {len(hv['on_air'])} | "
                     f"{hr.get('count')} / {hr.get('popped_total')} | {r['legacy']} | {'' if not res else res['q_score']} |")
        on_air = [(key, x) for key, r in t["runs"].items() for x in r.get("on_air_causes") or []]
        o += ["", "On-air closes (b), each with its owning call from GripperWatch and that call's skill row:", ""]
        o += [f"- {key} step {x['step']} {x['arm']}: owner `{x['owner']}` call `{x['call_id']}` = {x['skill']} on {x['backend']} "
              f"({x['status']}/{x['code']}); {'native' if x['native'] else 'not a native run'}; "
              f"{'a later close in the same call held' if x['held_later_in_the_call'] else 'no later close in the call held'}"
              for key, x in on_air] or ["- none"]
        o += ["", "After each native call: the phase it ended in, then the legacy calls up to the next native one (call, skill, status/code, steps):", ""]
        o += [f"- {key} `{x['call_id']}` {x['skill']} {x['status']}/{x['code']} phase {x['phase']} steps {x['steps']} -> {x['then'] or 'nothing'}"
              for key, r in t["runs"].items() for x in r.get("after_native") or []] or ["- no native call"]
        o += ["", "Per-call table (the switched skill at the same Runner write; each cell: backend, status/code, steps, scorer, returned):", ""]
        cols = sorted(k for row in t["pairing"] for k in row if k[:1] in "CT" and k[1:].isdigit())
        cols = sorted(set(cols))
        o += ["| call | write | args | L-rec | " + " | ".join(cols) + " |", "|---|---|---|---|" + "---|" * len(cols)]
        for row in t["pairing"]:
            cells = []
            for k in cols:
                v = row.get(k)
                cells.append("-" if v is None else f"{v['backend']} {v['status']}/{v['code']} {v['steps']} sc {v['scorer']} ret {v['returned']}{'' if v['same_write'] else ' (other skill)'}")
            o.append(f"| {row['call_id']} | {row['write']} {row['member']} | {row['args']} | {row['lrec_returned']} | " + " | ".join(cells) + " |")
        o += ["", "The switched skill by ordinal within each run (the same write once the sequences part; a cut-off row is the write EpisodeOver ended at success):", ""]
        cols_o = sorted({k for row in t["ordinal"] for k in row if k != "ordinal"})
        o += ["| # | " + " | ".join(cols_o) + " |", "|---|" + "---|" * len(cols_o)]
        for row in t["ordinal"]:
            cells = []
            for k in cols_o:
                v = row.get(k)
                cells.append("-" if v is None else f"{v['backend']} {v['status']}/{v['code']} {v['steps']} sc {v['scorer']} ret {v['returned']}")
            o.append(f"| {row['ordinal']} | " + " | ".join(cells) + " |")
        o += ["", f"Switched skill failed where a C run's succeeded (the cause that fails the stage): {t['switched_failed_where_c_succeeded'] or 'none'}"]
        o += ["", "Structural shifts (e), T against C after the branch, per replicate (ended pairs only):", ""]
        for s in t["structural"]:
            o.append(f"- r{s['rep']}: first difference at write {s.get('first_difference_write')} ({s['first_difference']} after the "
                     f"branch); T extra {s['t_extra']}; C extra {s['c_extra']}")
        o += ["", "Counters (raw; compare normalises per delivered atom):", ""]
        names_ = [c.name for c in counters.COUNTERS]
        keys_ = t.get("counters_keys") or {arm: [f"{arm}{i}" for i in range(len(t["counters"][arm]))] for arm in ("C", "T")}
        o += ["| counter | " + " | ".join(keys_["C"]) + " | " + " | ".join(keys_["T"]) + " |",
              "|---|" + "---|" * (len(t['counters']['C']) + len(t['counters']['T']))]
        for n in names_:
            o.append(f"| {n} | " + " | ".join(_f(c.get(n)) for c in t['counters']['C']) + " | " + " | ".join(_f(c.get(n)) for c in t['counters']['T']) + " |")
        o.append("")
        o += ["D21 debt per run: " + "; ".join(f"{k}: {r.get('d21_debt')}" for k, r in t["runs"].items()), ""]
    if g.get("flags"):
        o += ["## Flags and their causes", ""]
        for f in g["flags"]:
            o.append(f"- {f['task']} {f['counter']}: C {f['control']} T {f['treatment']}; switched skill failed where C succeeded: {f['switched_failed_where_c_succeeded'] or 'none'}")
        o.append("")
    return "\n".join(o)


# ------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    global GPUS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("plan", "run", "gate", "status"))
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--reps", default="0,1,2")
    ap.add_argument("--arms", default="C,T")
    ap.add_argument("--snap", type=Path, default=SNAP)
    ap.add_argument("--max-sims", type=int, default=MAX_SIMS)
    ap.add_argument("--max-planners", type=int, default=MAX_PLANNERS)
    ap.add_argument("--gpus", default=",".join(map(str, GPUS)), help="the cards to launch on, a subset of 1,3")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    GPUS = parse_gpus(a.gpus)
    reps = tuple(int(x) for x in a.reps.split(",") if x != "")
    arms = tuple(x for x in a.arms.split(",") if x)
    if a.mode == "plan":
        plan = plan_stage(a.stage, tasks=a.tasks or TASKS, reps=reps, snap=a.snap)
        (OUT / a.stage).mkdir(parents=True, exist_ok=True)
        (OUT / a.stage / "plan.json").write_text(json.dumps(plan, indent=1, default=str))
        for task, p in plan["tasks"].items():
            b = p.get("branch")
            print(f"{task}: " + ("no branch (C and T route every write the same)" if b is None else
                                 f"branch write {b['write']} {b['member']} {b['line']} {b['call_id']} {b['backend_c']}->{b['backend_t']}; "
                                 f"frame {p.get('frame')} (by {p.get('frame_method')}; E-rep {p.get('frame_by_erep')}, agrees {p.get('frame_agrees')}); "
                                 f"line calls {p['line_calls']}; floor {p['floor'].get('frame')}"))
        return 0
    if a.mode == "run":
        (OUT / a.stage).mkdir(parents=True, exist_ok=True)
        if a.stage in STAGES:
            plan = json.loads((OUT / a.stage / "plan.json").read_text())
            specs = stage_specs(a.stage, plan, tasks=a.tasks or STAGES[a.stage].tasks, reps=reps, arms=arms)
        elif a.stage == "carried":
            specs = carried_specs(tuple(a.tasks) if a.tasks else CARRIED)
        elif a.stage == "witness":
            specs = [witness_spec()]
        elif a.stage == "strict":
            specs = [strict_spec(a.snap)]
        else:
            ap.error(f"unknown stage {a.stage}")
        return queue(specs, a.snap, OUT / a.stage / "launchers.log", a.max_sims, a.max_planners)
    if a.mode == "gate":
        if a.stage in STAGES:
            g = gate_stage(a.stage, a.snap)
            (OUT / a.stage / "gate.json").write_text(json.dumps(g, indent=1, default=str))
            (OUT / a.stage / "GATE.md").write_text(gate_md(g))
            print(g["compare_text"] or "no comparison")
            print(json.dumps(g["verdict"], indent=1))
        elif a.stage == "carried":
            g = gate_carried(a.snap)
            (OUT / "carried").mkdir(parents=True, exist_ok=True)
            (OUT / "carried" / "GATE.md").write_text(carried_md(g))
        elif a.stage == "witness":
            g = gate_witness(a.snap)
        elif a.stage == "strict":
            g = gate_strict(a.snap)
        else:
            ap.error(f"unknown stage {a.stage}")
        if a.stage not in STAGES:
            (OUT / a.stage).mkdir(parents=True, exist_ok=True)
            (OUT / a.stage / "gate.json").write_text(json.dumps(g, indent=1, default=str))
            print(json.dumps(g, indent=1, default=str)[:6000])
        return 0
    if a.mode == "status":
        for stage in sorted(p.name for p in OUT.iterdir() if p.is_dir()) if OUT.exists() else []:
            for d in sorted((OUT / stage).glob("*_*")):
                if (d / "spec.json").exists():
                    print(f"{stage}/{d.name}: started {started(d)} ended {ended(d)} {result_line(d) or ''}")
        print(f"slots {live_slots()}/{MAX_SIMS}, planners {planners_live()}/{MAX_PLANNERS}, gpu free {gpu_free()}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
