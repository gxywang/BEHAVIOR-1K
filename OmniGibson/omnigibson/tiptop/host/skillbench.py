"""The Tier 1 skill bench (SPEC §9): one skill call from a fixed start state, through the production path.

One process loads one (task, instance) and, per case, runs the setup (harness-only: the torso and a teleport to the
case's stance, never charged to the skill), dumps the state, and restores it before every trial. A trial is the
one-call TaskPlanner (an observe when the skill asks for a Percept, then the call) over DirectConnector, with the
production Runtime, registry, latch, GoalPanel and the providers of --providers. Seeds go to numpy, torch and the
call. Every trial writes one JSON row (<out>/<case>.jsonl) and asserts the step invariant (U0):
rt.step - rt.idle_steps == sum(rt.charged), with idle_steps == 0, and, for a trial that did not step the sim itself
(legacy), sim.n_steps == rt.step + the steps the planner's captures took: nothing else stepped the sim. summary.csv
has one line per case.

  OMNIGIBSON_HEADLESS=1 python -m omnigibson.tiptop.host.skillbench --out-dir runs/skillbench/pick --port 8970 \\
      --case tiptop/b1k/skills/bench/pick_up.yaml --ids pick_up_freeze_fruit_apple --grasping-mode assisted \\
      --views head left_wrist right_wrist --no-state-stream

--backend legacy runs the baseline; --backend tiptop --collision mesh runs the native skill against today's meshes
(the planner server takes no map voxels until tiptop/skills/voxels.py lands).
"""

import argparse
import contextlib
import copy
import csv
import dataclasses
import importlib
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import b1k.runtime.skillrun as skillrun
from b1k.bridge.strategies import STRATEGIES
from b1k.connector.codec import from_dict
from b1k.connector.observe import ObserveRequest
from b1k.connector.skills import (
    CloseArgs,
    Code,
    OpenArgs,
    PickArgs,
    PlaceArgs,
    ReleaseArgs,
    SkillCall,
    SkillSpec,
    SkillSpecInfo,
)
from b1k.runtime.compose import ACTION_SLICES, PROPRIO_Q
from b1k.runtime.core import Runtime
from b1k.runtime.direct import DirectConnector
from b1k.skills.registry import SkillRegistry, load_routing
from b1k.skills.specs import arm_resources, hand_empty, holding_obj
from b1k.skills.tiptop.backend import TiptopBackend
from b1k.skills.tiptop.builders import pick as pick_request
from omnigibson.tiptop.host.bench_host import BenchHost
from omnigibson.tiptop.host.capture_observer import CaptureObserver
from omnigibson.tiptop.host.legacy_skills import LegacyBackend, _plain
from omnigibson.tiptop.run import (
    add_common,
    add_planner_args,
    apply_embodiment_posture,
    build_r1pro_sim,
    check_imports,
    connect_planners,
    setup_logging,
)

log = logging.getLogger("omnigibson.tiptop")

SETUP_KEYS = ("robot_pose", "torso")  # object_poses, joint_states and held join when a case needs them
# R1ProSim's command state, which the physics state does not carry: what its own steps (a legacy run, the restore's
# hold) command the joints nobody plans (posture), which arm plans, and where the captures look
SIM_FIELDS = ("posture", "locked_nominal", "arm", "other_arm", "planned_joints", "arm_idx", "gripper_idx", "q_home",
              "stance_ready", "look_target", "look_names", "look_arm", "mirror_arm_idx", "mirror_gripper_idx")


def _spec(name: str, args_type: type, budget: int, check=None, needs_percept: bool = False) -> SkillSpec:
    info = SkillSpecInfo(name, "1", (), args_type.__name__, (), (), budget, (), "legacy", needs_percept=needs_percept)
    return SkillSpec(info, args_type, arm_resources, check)


# The week-1 skills' backend-agnostic half. Budgets (report-only until a gate): pick and place from our executed
# rounds (SPEC §1), open/close from store_honey's executed open (475 env steps), release from its bench case.
SPECS = {
    "pick_up": _spec("pick_up", PickArgs, 950, hand_empty, needs_percept=True),
    "place": _spec("place", PlaceArgs, 1100, holding_obj, needs_percept=True),
    "open": _spec("open", OpenArgs, 600, hand_empty),
    "close": _spec("close", CloseArgs, 600, hand_empty),
    "release": _spec("release", ReleaseArgs, 60),
}


def targets(call: SkillCall) -> tuple:
    a = call.args
    named = tuple(o for o in (getattr(a, "obj", None), getattr(a, "target", None)) if o is not None)
    return named + tuple(r.target for r in getattr(a, "relations", ()))


def one_call(conn, call: SkillCall, views: tuple = ("head",)):
    """The bench's TaskPlanner: the one call, after an observe of ``views`` when the skill asks for a Percept (a
    legacy backend captures in its own run and does not). It never calls go_to: the case's stance is the harness's
    setup."""
    if conn.check(call).code is Code.PERCEPT_REQUIRED:
        call = dataclasses.replace(call, percept=conn.observe(ObserveRequest(targets(call), views=views)).id)
    return conn.run(call)


def load_cases(path, ids=None) -> list:
    """A <skill>.yaml: a list of {id, task, instance, setup: {robot_pose, torso}, call: to_dict(SkillCall), n, seeds,
    expect, baseline}; the ones named in ``ids`` when given."""
    with open(path) as f:
        cases = [c for c in yaml.safe_load(f) if ids is None or c["id"] in ids]
    for case in cases:
        unknown = set(case.get("setup") or {}) - set(SETUP_KEYS)
        if unknown:
            raise ValueError(f"{case['id']}: setup {sorted(unknown)} is not on the bench yet (it takes {SETUP_KEYS})")
        case["call"] = from_dict(case["call"], SkillCall)
        case.setdefault("seeds", list(range(case.get("n", 5))))
    return cases


def make_backends(ep, host, svc) -> dict:
    """legacy (the baseline, one round) and tiptop (a RequestBuilder per native skill); a call picks one by its
    backend, then routing.yaml. The legacy wire sends a compartment floor for an in() only under --inside-region
    and only where the geometry has a cavity; anything else it bends onto on(), which its result then says."""
    has_cavity = lambda target, item: bool(getattr(ep.sim, "send_inside", False)) and \
        svc.geometry.cavity(target, item).value is not None
    return {"legacy": LegacyBackend(ep, host.observe_now, has_cavity=has_cavity, single_round=True),
            "tiptop": TiptopBackend({"pick_up": pick_request.build},
                                    checks={"pick_up": pick_request.check})}  # fmt: skip


def run_trial(host, svc, backends: dict, routing: dict, observer, call: SkillCall) -> tuple:
    """One trial: the production Runtime and registry over these providers and backends, DirectConnector on the
    host, the one-call planner. (result, runtime, its skill_calls rows, wall seconds)."""
    calls = []
    rt = Runtime(SkillRegistry(SPECS, backends, routing), svc, host=host, log=calls, observer=observer)
    host.env_wall_s, t0 = 0.0, time.monotonic()
    r = one_call(DirectConnector(rt, host.env_step, host, host.raw()), call, getattr(observer, "views", ("head",)))
    return r, rt, calls, time.monotonic() - t0


def u0(rt, sim_steps=None, observe_steps: int = 0) -> bool:
    """The hard invariant on the one-call planner: every env step carried a live run's action, and (``sim_steps``:
    the sim's own count, for a trial that did not step the sim itself) the sim advanced only by those steps and the
    planner's captures, so nothing stepped it while anything planned."""
    runtime = rt.step - rt.idle_steps == sum(rt.charged.values()) and rt.idle_steps == 0
    return runtime and (sim_steps is None or sim_steps == rt.step + observe_steps)


def latch_gap(rt) -> float:
    """How far (rad) the next composed action would move the trunk or an arm from where it stands: an arm snapping
    back to its pre-run target after a self-stepping run shows here."""
    q = rt.obs.proprio
    return max(float(np.abs(rt.latch.target[ACTION_SLICES[g]] - q[s]).max()) for g, s in PROPRIO_Q.items())


def row(case: dict, trial: int, seed: int, r, rt, sim_steps: int, wall_s: float, env_wall_s: float,
        observe_wall_s: float = 0.0, observe_steps: int = 0) -> dict:
    """planning_wall_s is what is neither an env step nor the planner's capture: planning, resampling and
    verification (for a legacy run, which captures and steps inside itself, all of it)."""
    return {
        "case": case["id"], "trial": trial, "seed": seed, "skill": r.skill, "backend": r.backend,
        "status": r.status.value, "code": None if r.code is None else r.code.value, "phase": r.phase,
        "detail": r.detail, "binding": [dataclasses.asdict(b) for b in r.binding], "primary": r.primary,
        "verdicts": dict(r.verdicts), "steps": r.steps,
        "rt_step": rt.step, "idle_steps": rt.idle_steps, "charged": dict(rt.charged),
        "u0": u0(rt, None if r.requires_sim_clock else sim_steps, observe_steps),  # legacy steps the sim itself
        "sim_steps": sim_steps, "observe_steps": observe_steps, "latch_gap_rad": round(latch_gap(rt), 4),
        "wall_s": round(wall_s, 2),
        "env_wall_s": round(env_wall_s, 2), "observe_wall_s": round(observe_wall_s, 2),
        "planning_wall_s": round(wall_s - env_wall_s - observe_wall_s, 2),
        "oracle_reads": dict(r.oracle_reads), "planner_oracle_reads": dict(rt.planner_oracle_reads),
        "requires_sim_clock": r.requires_sim_clock, "evidence": dict(r.evidence),
    }  # fmt: skip


def summarize(case: dict, rows: list) -> dict:
    steps = [r["steps"] for r in rows]
    expect = case.get("expect") or {}
    return {
        "case": case["id"], "skill": case["call"].skill, "backend": rows[0]["backend"], "n": len(rows),
        "succeeded": sum(r["status"] == "succeeded" for r in rows),
        "rate": round(sum(r["status"] == "succeeded" for r in rows) / len(rows), 3),
        "expect_met": sum(all(r.get(k) == v for k, v in expect.items()) for r in rows),
        "codes": json.dumps(Counter(r["code"] for r in rows)),
        "steps_p50": float(np.percentile(steps, 50)), "steps_p95": float(np.percentile(steps, 95)),
        "agree": sum(all(v is None or v == r["verdicts"][r["primary"]] for v in r["verdicts"].values()) for r in rows),
        "u0_all": all(r["u0"] for r in rows), "requires_sim_clock": sum(r["requires_sim_clock"] for r in rows),
        "wall_s_mean": round(float(np.mean([r["wall_s"] for r in rows])), 1),
    }  # fmt: skip


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(p)
    add_planner_args(p)
    p.add_argument("--case", nargs="+", required=True, help="case files (b1k/skills/bench/<skill>.yaml)")
    p.add_argument("--ids", nargs="+", default=None, help="the cases to run, all of one (task, instance)")
    p.add_argument("--backend", default=None, help="run every call on this backend (the baseline: legacy)")
    p.add_argument("--trials", type=int, default=None, help="run only the first N seeds of each case")
    p.add_argument("--mode", choices=("train", "public_test", "hidden_test"), default="public_test")
    p.add_argument("--rounds", type=int, default=1, help="Episode's rounds (a single_round legacy call runs one)")
    p.add_argument("--providers", default="omnigibson.tiptop.oracle",
                   help="the package whose pseudo_services(episode, planner, routing, collision) builds the Services")
    p.add_argument("--collision", choices=("map", "mesh"), default="map",
                   help="the room a native skill plans against: the map's voxels, or today's physical meshes (the "
                   "A/B; the planner server takes only meshes until tiptop/skills/voxels.py lands)")
    return p.parse_args(argv)


def setup(og, sim, args, case: dict, embodiment: dict) -> tuple:
    """Harness-only, never charged to the skill: the torso, the teleport to the case's stance, the look at the call's
    objects (a round stood for them looked there before it captured), a settle. Returns the state every trial
    restores."""
    s = case.get("setup") or {}
    args.torso = s.get("torso")
    apply_embodiment_posture(sim, args, embodiment)
    if "robot_pose" in s:
        sim.place_robot(*s["robot_pose"], note=f"skill bench setup of {case['id']}")
    sim.look_at(*(o.id for o in targets(case["call"])))
    sim.hold(args.settle_steps, sim.OPEN)
    log.info(f"{case['id']}: setup {s} done")
    return snapshot(og, sim)


def snapshot(og, sim) -> tuple:
    """The physics state and R1ProSim's own command state (SIM_FIELDS): load_state restores only the first."""
    return og.sim.dump_state(serialized=False), {k: copy.deepcopy(getattr(sim, k, None)) for k in SIM_FIELDS}


def restore(og, sim, state: tuple, name: str) -> None:
    physics, fields = state
    og.sim.load_state(physics, serialized=False)
    for k, v in fields.items():
        setattr(sim, k, copy.deepcopy(v))
    sim.seen_boxes, sim.last_gripper, sim.other_gripper = {}, sim.OPEN, sim.OPEN
    sim.hold(1, sim.OPEN)  # a physics step propagates the loaded state (Simulator.load_state)
    sim.begin_episode(name=name)  # counts from zero; clears the hand record


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    cases = [case for path in args.case for case in load_cases(path, args.ids)]
    runs = {(c["task"], int(c["instance"])) for c in cases}
    if len(runs) != 1:
        raise SystemExit(f"one (task, instance) per process; the cases name {sorted(runs)}")
    ((task, instance),) = runs
    spec = STRATEGIES.get(task)
    args.activity, args.embodiment = task, "r1pro"
    args.task = spec.instruction if spec is not None else task.replace("_", " ")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import omnigibson as og
    import torch
    from omnigibson.eval.evaluator import load_task_instance
    from omnigibson.tiptop.bench import Episode
    from omnigibson.tiptop.knowledge import make_knowledge
    from omnigibson.tiptop.scene import EpisodeOver
    from b1k.bridge.strategies import strategy_for, task_goal_atoms, task_goal_options

    skillrun.PASSTHROUGH = (EpisodeOver,)  # the harness's own exception: no skill may swallow it
    providers, routing = importlib.import_module(args.providers), load_routing()
    client, metadata, press_client, press_meta = connect_planners(args)
    check_imports(metadata, press_meta)  # the step-zero gate line: every module from this checkout, or stop here
    planners = {"left": (client, metadata), **({"right": (press_client, press_meta)} if press_client else {})}
    exit_code, summaries = 0, []
    try:
        sim = build_r1pro_sim(args, metadata["embodiment"])
        sim.env.reset()
        load_task_instance(sim.env, sim.robot, instance, mode=args.mode)
        sim.env.reset()
        sim.reset_embodiment(metadata["embodiment"])
        strategy = strategy_for(task, task_goal_atoms(sim), options=task_goal_options(sim),
                                scope=sorted(sim.task_scope()))  # fmt: skip
        knowledge = make_knowledge(args.knowledge, sim, strategy.goal, spec=strategy.spec)
        host, base = BenchHost(sim), snapshot(og, sim)
        for case in cases:
            restore(og, sim, base, case["id"])  # every case starts from the instance as loaded
            state, rows = setup(og, sim, args, case, metadata["embodiment"]), []
            (out / case["id"]).mkdir(parents=True, exist_ok=True)
            for i, seed in enumerate(case["seeds"][: args.trials]):
                name = f"{case['id']}_t{i}"
                restore(og, sim, state, name)
                random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
                ep = Episode(sim, args, planners, knowledge, out / case["id"] / f"t{i}", spec=strategy.spec)
                svc, segmenter = providers.pseudo_services(ep, client, routing, collision=args.collision)
                observer = CaptureObserver(sim, host, segmenter, args.task)
                call = dataclasses.replace(case["call"], seed=seed, backend=args.backend or case["call"].backend)
                video = contextlib.nullcontext() if args.no_video else sim.recording(out / case["id"] / f"{name}.mp4")
                with video:
                    r, rt, calls, wall_s = run_trial(host, svc, make_backends(ep, host, svc), routing, observer, call)
                rows.append(row(case, i, seed, r, rt, sim.n_steps, wall_s, host.env_wall_s, observer.wall_s,
                                observer.steps))
                _append(out / f"{case['id']}.jsonl", [rows[-1]])
                _append(out / "skill_calls.jsonl", [dict(c, trial=name) for c in calls])
                _append(out / "goal_checks.jsonl", [dict(g, trial=name) for g in svc.goals.log])
                log.info(f"TRIAL {name}: {r.status.value} {r.code} steps {r.steps} rt {rt.step} "
                         f"charged {rt.charged} idle {rt.idle_steps} latch gap {rows[-1]['latch_gap_rad']} rad")
                assert rows[-1]["u0"], f"{name} broke the step invariant (U0): {rows[-1]}"
            summaries.append(summarize(case, rows))
    except Exception:
        log.exception("skill bench failed")
        exit_code = 1
    finally:
        if summaries:
            with open(out / "summary.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summaries[0]))
                w.writeheader()
                w.writerows(summaries)
            log.info(f"SUMMARY {summaries}")
        if og.app is not None:
            og.shutdown()
    sys.exit(exit_code)


def _append(path: Path, rows: list) -> None:
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=_plain) + "\n")


if __name__ == "__main__":
    main()
