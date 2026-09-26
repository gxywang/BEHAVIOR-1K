"""The Tier 1 skill bench (SPEC §9): one skill call from a fixed start state, through the production path.

One process loads one task and, per case, its (instance, mode) when that changes, then runs the setup (harness-only,
never charged to the skill: the torso and a teleport to the case's stance, or a human demo's snapshot restored with
its arm posture and the object in its hand), dumps the state, and restores it before every trial. A trial is the
one-call TaskPlanner (an observe when the skill asks for a Percept, then the call) over DirectConnector, with the
production Runtime, registry, latch, GoalPanel and the providers of --providers. Seeds go to numpy, torch and the
call. Every trial writes one JSON row (<out>/<case>.jsonl) and asserts the step invariant (U0):
rt.step - rt.idle_steps == sum(rt.charged), with idle_steps == 0, and, for a trial that did not step the sim itself
(legacy), sim.n_steps == rt.step + the steps the planner's captures took: nothing else stepped the sim. summary.csv
has one line per case. A case with a ``lease`` (a persistent call: hold(left, "here")) is the two-handed shape
(SPEC §5.6): the lease starts before the observe, which then captures from where the head is (aim=False), the call
runs beside it (press(right) through the r1pro_right planner of --press-port), and the lease is aborted after; the
lease is charged every step too, so U0 reads rt.step == sum(rt.charged) - lease.steps. A legacy call runs alone: its
round holds the other hand itself.

  OMNIGIBSON_HEADLESS=1 python -m omnigibson.tiptop.host.skillbench --out-dir runs/skillbench/pick --port 8970 \\
      --case tiptop/b1k/skills/bench/pick_up.yaml --ids pick_up_freeze_fruit_apple --grasping-mode assisted \\
      --views head left_wrist right_wrist --no-state-stream

--backend legacy runs the baseline; --backend tiptop runs the native skill (--collision mesh: against today's meshes
instead of the map's voxels). --setup-only loads, restores and checks every case's setup without a planner or a trial
(setup.jsonl); --selftest N runs SPEC §9's sim self-test on each case's restored state instead of its trials.
"""

import argparse
import contextlib
import copy
import csv
import dataclasses
import importlib
import json
import logging
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

import b1k.runtime.skillrun as skillrun
from b1k.bridge.client import ArmPlanners
from b1k.bridge.protocol import bddl_category
from b1k.bridge.strategies import STRATEGIES
from b1k.connector.codec import from_dict, to_dict
from b1k.connector.api import Unreachable, reach as api_reach
from b1k.connector.observe import ObserveRequest
from b1k.connector.skills import Code, NavResult, SkillCall, WorldUpdate
from b1k.connector.types import ObjRef
from b1k.perception.grasp_sensor import ProprioGraspSensor
from b1k.runtime.compose import ACTION_SLICES, CLOSED, FINGER_Q, OPEN, PROPRIO_Q
from b1k.runtime.core import Runtime
from b1k.runtime.direct import DirectConnector
from b1k.skills.registry import SkillRegistry, load_routing
from b1k.skills.scripted import ScriptedBackend
from b1k.skills.specs import SPECS
from b1k.skills.tiptop import builders
from b1k.skills.tiptop.backend import TiptopBackend
from omnigibson.tiptop.host import demo_cases
from omnigibson.tiptop.host.bench_host import BenchHost
from omnigibson.tiptop.host.capture_observer import CaptureObserver
from omnigibson.tiptop.host.legacy_skills import LegacyBackend, _plain
from omnigibson.tiptop.host.teleport_nav import TeleportNavigator
from omnigibson.tiptop.host.overview import aim_overview
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

SETUP_KEYS = ("robot_pose", "torso", "held", "joint_states", "ready")  # held {arm: sim object name} needs
#                                     demo.snapshot (the object is in the hand there); joint_states {object: {joint:
#                                     value}} (a drawer to close starts open); ready: the planned arm at its planner's
#                                     ready posture, where a pick leaves it; object_poses joins when needed
# R1ProSim's command state, which the physics state does not carry: what its own steps (a legacy run, the restore's
# hold) command the joints nobody plans (posture), which arm plans, where the captures look, both gripper commands
# (a demo's held hand stays closed) and the hand record (what a legacy round knows the hands hold)
SIM_FIELDS = ("posture", "locked_nominal", "arm", "other_arm", "planned_joints", "arm_idx", "gripper_idx", "q_home",
              "stance_ready", "look_target", "look_names", "look_arm", "mirror_arm_idx", "mirror_gripper_idx",
              "last_gripper", "other_gripper", "held_objects")
SELFTEST_STEPS, SELFTEST_RAD = 90, 0.1  # the self-test's scripted trajectory: every planned-arm joint, a sine bump


class CaseSetupError(RuntimeError):
    """The harness could not bring the sim to the case's start state (its reason is the message)."""


def targets(call: SkillCall) -> tuple:
    a = call.args
    named = tuple(o for o in (getattr(a, "obj", None), getattr(a, "target", None)) if o is not None)
    return named + tuple(r.target for r in getattr(a, "relations", ()))


def one_call(conn, call: SkillCall, views: tuple = ("head",), lease: Optional[SkillCall] = None,
             reach: bool = False):
    """The bench's TaskPlanner: the one call, after an observe of ``views`` when the skill asks for a Percept (a
    legacy backend captures in its own run and does not). It calls go_to only for a ``reach`` case (check_stances.yaml):
    b1k.connector.api.reach, the stance the IK service scores best, from a setup stance out of reach; otherwise the
    case's stance is the harness's setup. With a ``lease`` it is hold_and_press (b1k.connector.api): the lease started
    first, the observe from where the head is (a lease holds the trunk read-only), the call beside it, the lease
    aborted after."""
    h = conn.start(lease) if lease is not None else None
    if h is not None and h.call_id in conn.rt.results:  # refused before it ran (not_holding, resource_busy): the
        return conn.rt.results[h.call_id]  # trial is not two-handed, and its row says so, instead of a lone press
    if reach:
        try:
            api_reach(conn, call)
        except Unreachable:  # no candidate, or none the service reaches from: the run answers NO_STANCE_HERE from here
            pass
    if conn.check(call).code is Code.PERCEPT_REQUIRED:
        req = ObserveRequest(targets(call), views=views, aim=lease is None)
        call = dataclasses.replace(call, percept=conn.observe(req).id)
    r = conn.run(call)
    if h is not None:
        conn.abort(h)
    return r


def load_cases(path, ids=None) -> list:
    """A <skill>.yaml: a list of {id, task, instance, setup: {robot_pose, torso, held?, joint_states?}, call:
    to_dict(SkillCall), n, seeds, expect, baseline}, plus, for a case from the human demos (demo_cases.py), mode
    (train), demo {snapshot: a path relative to the case file, arms, fingers, ...}, and ``lease`` (to_dict(SkillCall):
    the hold the call runs beside); the ones named in ``ids`` when
    given. A case with ``skip: <reason>`` is left out, its reason logged."""
    with open(path) as f:
        cases = [c for c in yaml.safe_load(f) if ids is None or c["id"] in ids]
    for case in cases:
        if case.get("skip"):
            log.warning(f"{case['id']}: skipped: {case['skip']}")
    cases = [c for c in cases if not c.get("skip")]
    for case in cases:
        setup, demo = case.get("setup") or {}, case.get("demo") or {}
        unknown = set(setup) - set(SETUP_KEYS)
        if unknown:
            raise ValueError(f"{case['id']}: setup {sorted(unknown)} is not on the bench yet (it takes {SETUP_KEYS})")
        if setup.get("held") and not demo.get("snapshot"):
            raise ValueError(f"{case['id']}: setup.held needs demo.snapshot (the object is in the hand there)")
        if demo.get("snapshot"):
            if not case.get("mode"):
                raise ValueError(f"{case['id']}: a demo.snapshot case names the mode of the instance it was taken on "
                                 "(train for a human demo, public_test for a manip run's stance)")
            demo["snapshot"] = str(Path(path).resolve().parent / demo["snapshot"])
        case["call"] = from_dict(case["call"], SkillCall)
        if case.get("lease"):
            case["lease"] = from_dict(case["lease"], SkillCall)
        case.setdefault("seeds", list(range(case.get("n", 5))))
    return cases


def make_backends(ep, host, svc) -> dict:
    """legacy (the baseline, one round), tiptop (a RequestBuilder per native skill) and scripted, the last two found
    by file (b1k/skills/tiptop/builders/<skill>.py, b1k/skills/scripted/<skill>.py): a new skill is a new module
    there, not an edit here. A call picks one by its backend, then routing.yaml. The legacy wire sends a compartment
    floor for an in() only under --inside-region and only where the geometry has a cavity; anything else it bends
    onto on(), which its result then says."""
    has_cavity = lambda target, item: bool(getattr(ep.sim, "send_inside", False)) and \
        svc.geometry.cavity(target, item).value is not None
    return {"legacy": LegacyBackend(ep, host.observe_now, has_cavity=has_cavity, single_round=True),
            "tiptop": TiptopBackend.from_modules(builders.discover()),  # its hooks (check, stop_state, ...) by its list
            "scripted": ScriptedBackend()}  # fmt: skip


def lease_for(case: dict, call: SkillCall, seed: int, registry: SkillRegistry) -> Optional[SkillCall]:
    """The case's lease beside the call, unless the backend routing.yaml resolves for the call captures in its own run
    (a legacy round holds the other hand itself, and steps the sim under a live lease): by the route, not by
    call.backend, so `press: {default: legacy}` in routing.yaml drops the lease as --backend legacy does."""
    lease = case.get("lease")
    if lease is None or getattr(registry.backend_for(call), "captures_in_own_run", False):
        return None
    return dataclasses.replace(lease, seed=seed)


def run_trial(host, svc, backends: dict, routing: dict, observer, call: SkillCall,
              lease: Optional[SkillCall] = None, nav=None, reach: bool = False) -> tuple:
    """One trial: the production Runtime and registry over these providers and backends, DirectConnector on the
    host, the one-call planner. (result, runtime, its skill_calls rows, wall seconds)."""
    calls = []
    # the Runtime seeds its latch from the host: what the setup left commanded (a demo's held hand stays closed)
    rt = Runtime(SkillRegistry(SPECS, backends, routing), svc, host=host, log=calls, observer=observer, navigator=nav)
    host.env_wall_s, host.frames_wall_s, t0 = 0.0, 0.0, time.monotonic()
    r = one_call(DirectConnector(rt, host.env_step, host, host.raw()), call, getattr(observer, "views", ("head",)),
                 lease, reach)
    return r, rt, calls, time.monotonic() - t0


def u0(rt, sim_steps=None, observe_steps: int = 0, lease_steps: int = 0) -> bool:
    """The hard invariant on the one-call planner: every env step carried a live run's action, and (``sim_steps``:
    the sim's own count, for a trial that did not step the sim itself) the sim advanced only by those steps and the
    planner's captures, so nothing stepped it while anything planned. A lease beside the call is charged on every
    step of the trial too (``lease_steps``): the steps it shared are counted once."""
    runtime = rt.step - rt.idle_steps == sum(rt.charged.values()) - lease_steps and rt.idle_steps == 0
    return runtime and (sim_steps is None or sim_steps == rt.step + observe_steps)


def latch_gap(rt) -> float:
    """How far (rad) the next composed action would move the trunk or an arm from where it stands: an arm snapping
    back to its pre-run target after a self-stepping run shows here."""
    q = rt.obs.proprio
    return max(float(np.abs(rt.latch.target[ACTION_SLICES[g]] - q[s]).max()) for g, s in PROPRIO_Q.items())


def row(case: dict, trial: int, seed: int, r, rt, sim_steps: int, wall_s: float, env_wall_s: float,
        observe_wall_s: float = 0.0, observe_steps: int = 0, lease: Optional[dict] = None,
        frames_wall_s: float = 0.0) -> dict:
    """planning_wall_s is what is neither an env step nor the planner's capture: planning, resampling and
    verification (for a legacy run, which captures and steps inside itself, all of it); ``frames_wall_s``, the part
    of it spent rendering the end-of-run frames the GoalPanel read, is shown beside it. ``lease``: the skill_calls
    row of the lease the call ran beside (aborted = still holding when the call ended; hold_lost = it let go)."""
    return {
        "case": case["id"], "trial": trial, "seed": seed, "skill": r.skill, "backend": r.backend,
        "status": r.status.value, "code": None if r.code is None else r.code.value, "phase": r.phase,
        "detail": r.detail, "binding": [dataclasses.asdict(b) for b in r.binding], "primary": r.primary,
        "verdicts": dict(r.verdicts), "steps": r.steps,
        "rt_step": rt.step, "idle_steps": rt.idle_steps, "charged": dict(rt.charged),
        "u0": u0(rt, None if r.requires_sim_clock else sim_steps, observe_steps,  # legacy steps the sim itself
                 lease["steps"] if lease else 0),
        "lease": None if lease is None else {k: lease[k] for k in ("skill", "backend", "status", "code", "steps")},
        "sim_steps": sim_steps, "observe_steps": observe_steps, "latch_gap_rad": round(latch_gap(rt), 4),
        "wall_s": round(wall_s, 2),
        "env_wall_s": round(env_wall_s, 2), "observe_wall_s": round(observe_wall_s, 2),
        "planning_wall_s": round(wall_s - env_wall_s - observe_wall_s, 2), "frames_wall_s": round(frames_wall_s, 2),
        "oracle_reads": dict(r.oracle_reads), "planner_oracle_reads": dict(rt.planner_oracle_reads),
        "requires_sim_clock": r.requires_sim_clock, "evidence": dict(r.evidence),
        "world_updates": [to_dict(u) for u in r.world_updates],  # an open's joint value and its source (SPEC §6.6)
        "go_to": go_to_rows(rt.results),  # a reach case's teleports: where to, ok or the landing's refusal
    }  # fmt: skip


def go_to_rows(results: dict) -> list:
    """The trial's NavResults (reach(): one per go_to, in order) as rows: the stance, whether the teleport landed, and
    place_robot's reason when it did not."""
    return [{"key": n.stance.key if n.stance else None, "pose": to_dict(n.stance.pose) if n.stance else None,
             "ok": n.ok, "shadow_steps": n.shadow_steps, "detail": n.detail}
            for n in results.values() if isinstance(n, NavResult)]


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
        # the shadow checkers against the primary, over the trials where a shadow judged (None is no agreement: the
        # perception shadow judges the end-of-run Frames the host puts in StepObs.sensors, None where they cannot say)
        "shadow_judged": sum(any(v is not None for n, v in r["verdicts"].items() if n != r["primary"]) for r in rows),
        "agree": sum(any(v is not None for n, v in r["verdicts"].items() if n != r["primary"])
                     and all(v is None or v == r["verdicts"][r["primary"]] for v in r["verdicts"].values()) for r in rows),
        "u0_all": all(r["u0"] for r in rows), "requires_sim_clock": sum(r["requires_sim_clock"] for r in rows),
        "wall_s_mean": round(float(np.mean([r["wall_s"] for r in rows])), 1),
        "lease_kept": sum((r.get("lease") or {}).get("status") == "aborted" for r in rows),  # held to the end
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
    p.add_argument("--setup-only", action="store_true",
                   help="no planner, no trial: load, restore and check every case's setup (setup.jsonl)")
    p.add_argument("--selftest", type=int, default=None, metavar="N",
                   help="SPEC §9's sim self-test instead of the trials: N seeded runs of one scripted trajectory "
                   "from each case's restored state; the proprio and object spread is the sim's noise floor")
    p.add_argument("--finger-settled", type=float, default=None,
                   help="the proprio GraspSensor's settled test (rad/s) for this process, over its own 0.01")
    return p.parse_args(argv)


def held_refs(sim, held: dict, call: SkillCall) -> dict:
    """{arm: ObjRef} of a case's setup.held (sim object names, as the demo recorded them): the call's own ObjRef where
    it names the object, else one built from the BDDL name the task scope gives it (a task object), else the name."""
    named = {o.id: o for o in targets(call)}
    by_name = {obj.name: bddl for bddl, obj in sim.task_scope().items()}
    out = {}
    for arm, name in held.items():
        bddl = by_name.get(name)
        out[arm] = (named.get(bddl) or ObjRef(bddl, bddl_category(bddl), False) if bddl
                    else ObjRef(name, name.rsplit("_", 1)[0], False))  # not a task object: a scene object by name
    return out


def pick_arm(call: SkillCall, held: dict) -> str:
    """The call's arm when the case leaves it to the bench: the hand holding args.obj (place, hold, release), else a
    free hand, left first."""
    if call.arm:
        return call.arm
    obj = getattr(call.args, "obj", None)
    holding = [arm for arm, ref in held.items() if obj is not None and ref.id == obj.id]
    return holding[0] if holding else next((a for a in ("left", "right") if a not in held), "left")


def adopt_demo_posture(sim, host, held: dict) -> None:
    """After a demo restore, R1ProSim's command state is the human's: the idle arm is locked where it stands (its
    planner takes the measured lock, S1) so no hold or capture drives it to the nominal posture, the planned arm's
    ready posture is where it stands (the capture returns there), a holding hand stays commanded closed, and the
    hand record names what it holds, so a legacy round knows and the capture leaves that arm alone."""
    p = host.proprio()
    idle = dict(zip(sim.robot.arm_joint_names[sim.other_arm], p[PROPRIO_Q[f"arm_{sim.other_arm}"]].tolist()))
    sim.posture.update({j: v for j, v in idle.items() if j in sim.posture})
    sim.locked_nominal.update({j: v for j, v in idle.items() if j in sim.locked_nominal})
    sim.stance_ready = [float(v) for v in sim.q_arm()]
    sim.last_gripper = CLOSED if sim.arm in held else OPEN
    sim.other_gripper = CLOSED if sim.other_arm in held else OPEN
    labels = {ref.id: sim.tracked_label(ref.id) for ref in held.values() if ref.id in sim.task_scope()}
    sim.held_objects = {labels[ref.id]: arm for arm, ref in held.items() if ref.id in labels}


def commanded_obs(sim, host):
    """The StepObs the proprio GraspSensor reads: the observation now, with what R1ProSim last commanded."""
    a23 = np.concatenate([sim.commanded_targets()[g] for g in ACTION_SLICES]).astype(np.float32)
    return dataclasses.replace(host.observe_now(), commanded=a23)


def setup(og, sim, args, case: dict, embodiment: dict, host=None) -> tuple:
    """Harness-only, never charged to the skill. A stance case: the torso, the teleport to the case's stance. A demo
    case: the human's snapshot restored (demo_cases.restore: the instance, then every saved object and the robot
    with its grasp constraints), the human's arm posture adopted as the command state, the held hand closed and its
    hold verified by the proprio GraspSensor (CaseSetupError with the reason when it does not hold). Then the look at
    the call's objects (a round stood for them looked there before it captured) and a settle. Returns the state
    every trial restores and {arm: ObjRef} of what the hands hold."""
    s, demo, held = case.get("setup") or {}, case.get("demo") or {}, {}
    if demo.get("snapshot"):
        with open(demo["snapshot"]) as f:
            demo_cases.restore(sim.env, json.load(f), int(case["instance"]), case.get("mode") or args.mode)
        held = held_refs(sim, s.get("held") or {}, case["call"])
        adopt_demo_posture(sim, host or BenchHost(sim), held)
    else:
        args.torso = s.get("torso")
        apply_embodiment_posture(sim, args, embodiment)
        if "robot_pose" in s:
            sim.place_robot(*s["robot_pose"], note=f"skill bench setup of {case['id']}")
    if "robot_pose" in s:  # the video's third-person view, with a line of sight to the robot and the call's target
        named = targets(case["call"])
        aim_overview(sim, *s["robot_pose"], target=named[0].id if named else None)
    if s.get("ready"):  # the holding arm where a pick leaves it (SPEC §5.6: hold "here"): its planner's ready posture
        #                 set outright as apply_posture does, the object welded to the hand following at the settle
        if s["ready"] != sim.arm:
            raise ValueError(f"{case['id']}: ready: {s['ready']} is not on the bench (the {sim.arm} arm is planned)")
        import torch as th

        sim.robot.set_joint_positions(th.tensor(sim.q_home), indices=sim.arm_idx, drive=False)  # no sim read here
        sim.robot.keep_still()
        sim.stance_ready = [float(v) for v in sim.q_home]
    for name, joints in (s.get("joint_states") or {}).items():  # set outright; the settle below propagates it
        for joint, value in joints.items():
            sim.scene_object(name).joints[joint].set_pos(float(value))
    if targets(case["call"]):  # a release names nothing: the head stays where the setup left it
        sim.look_at(*(o.id for o in targets(case["call"])))
    sim.hold(args.settle_steps, sim.last_gripper if demo.get("snapshot") else sim.OPEN)
    if held:
        host, sensor = host or BenchHost(sim), ProprioGraspSensor()
        for _ in range(sensor.WINDOW):  # a welded finger's phantom velocity: the sensor settles by position over a window
            verdicts = {arm: sensor.held(arm, commanded_obs(sim, host)).value for arm in held}
            if None not in verdicts.values():
                break
            sim.hold(1, sim.last_gripper)
        truth, what = demo_cases.held(sim.robot), {a: r.id for a, r in held.items()}  # the sim's record, logged beside
        log.info(f"{case['id']}: held {what}: proprio {verdicts}, sim {truth}")
        missing = [arm for arm, v in verdicts.items() if v is not True]
        if missing:
            raise CaseSetupError(f"{case['id']}: the {missing} hand does not hold {[held[a].id for a in missing]} "
                                 f"after the restore: proprio {verdicts}, sim {truth}")
    log.info(f"{case['id']}: setup {s} done" + (f" from {demo['snapshot']}" if demo.get("snapshot") else ""))
    return snapshot(og, sim), held


def snapshot(og, sim) -> tuple:
    """The physics state and R1ProSim's own command state (SIM_FIELDS): load_state restores only the first."""
    return og.sim.dump_state(serialized=False), {k: copy.deepcopy(getattr(sim, k, None)) for k in SIM_FIELDS}


def restore(og, sim, state: tuple, name: str) -> None:
    physics, fields = state
    og.sim.load_state(physics, serialized=False)
    for k, v in fields.items():
        setattr(sim, k, copy.deepcopy(v))
    sim.seen_boxes = {}
    gripper = fields.get("last_gripper")
    sim.hold(1, sim.OPEN if gripper is None else gripper)  # a physics step propagates the loaded state; the grippers
    #                                                         as the setup left them (a demo's held hand stays closed)
    sim.begin_episode(name=name)  # counts from zero; clears the hand record, which the setup's hands then refill
    sim.held_objects = dict(fields.get("held_objects") or {})


def load_instance(og, sim, instance: int, mode: str, embodiment: dict) -> tuple:
    """One (instance, mode) of the loaded task, as the evaluator loads it, the embodiment planned afresh; the base
    state every case of it starts from."""
    from omnigibson.eval.evaluator import load_task_instance

    sim.env.reset()
    load_task_instance(sim.env, sim.robot, instance, mode=mode)
    sim.env.reset()
    sim.reset_embodiment(embodiment)
    # the lock as apply_posture leaves it: a legacy round of the previous instance may have adopted the other arm
    # (its posture then names the left arm, and the restore's hold raised KeyError 'right_arm_joint1')
    sim.posture = {j: float(v) for j, v in embodiment["locked_joints"].items()}
    sim.locked_nominal = {j: v for j, v in sim.posture.items() if "finger" not in j}
    sim.q_home = [float(v) for v in embodiment["q_home"]]
    sim.held_objects, sim.last_gripper = {}, sim.OPEN  # the last case's held hand (adopt_demo_posture) is not this
    #                                                    instance's: every case of it starts with empty, open hands
    return snapshot(og, sim)


def finger_settled(args) -> None:
    """--finger-settled: the proprio GraspSensor's settled test for this process (its own is 0.01 rad/s). A welded
    finger under a held object reports a velocity that never decays while its position never changes (the radio in
    the left hand at its ready posture: |qvel| 0.010-0.018 over 720 steps, 2026-09-26), so the sensor answers None
    and refuses the hold; a report-only knob until the sensor settles by position (the pick track's item)."""
    if args.finger_settled is not None:
        ProprioGraspSensor.SETTLED = args.finger_settled
        log.warning(f"the proprio GraspSensor's settled test is {args.finger_settled} rad/s in this process")


def spread(rows: list) -> dict:
    """The self-test's noise floor over its trials: per proprio group the widest range of any joint (rad or m), per
    object the largest distance between two trials' positions (m)."""
    p = np.asarray([r["proprio"] for r in rows], dtype=np.float64)
    groups = {**PROPRIO_Q, **{f"gripper_{a}": s for a, s in FINGER_Q.items()}}
    out = {"proprio": {g: float((p[:, s].max(0) - p[:, s].min(0)).max()) for g, s in groups.items()}}
    objs = {}
    for name in rows[0]["objects"]:
        xyz = np.asarray([r["objects"][name] for r in rows if name in r["objects"]], dtype=np.float64)
        objs[name] = float(max(np.linalg.norm(a - b) for a in xyz for b in xyz))
    out["objects_m"] = objs
    return out


def selftest(og, sim, host, state: tuple, case: dict, seeds: list, out: Path) -> dict:
    """SPEC §9's sim self-test (F18): the same scripted trajectory (a sine bump of SELFTEST_RAD on every planned-arm
    joint over SELFTEST_STEPS env steps, from what the restore left commanded) from the same restored state, once
    per seed. The spread of the final proprio and of the task objects' positions is the sim's noise floor: what no
    per-skill gate can claim below."""
    import torch

    rows, arm = [], ACTION_SLICES[f"arm_{sim.arm}"]
    for i, seed in enumerate(seeds):
        restore(og, sim, state, f"{case['id']}_selftest{i}")
        random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
        a = np.concatenate([sim.commanded_targets()[g] for g in ACTION_SLICES]).astype(np.float32)
        start, raw = a[arm].copy(), host.raw()
        for k in range(1, SELFTEST_STEPS + 1):
            a[arm] = start + SELFTEST_RAD * math.sin(math.pi * k / SELFTEST_STEPS)
            raw = host.env_step(a)
        rows.append({"case": case["id"], "trial": i, "seed": seed, "proprio": raw["proprio"].tolist(),
                     "objects": demo_cases.object_positions(sim.env)})  # fmt: skip
    _append(out / "selftest.jsonl", rows)
    result = {"case": case["id"], "n": len(rows), "steps": SELFTEST_STEPS, "rad": SELFTEST_RAD, **spread(rows)}
    log.info(f"SELFTEST {result}")
    return result


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    cases = [case for path in args.case for case in load_cases(path, args.ids)]
    tasks = {c["task"] for c in cases}
    if len(tasks) != 1:
        raise SystemExit(f"one task per process (its instances and modes reload in it); the cases name {sorted(tasks)}")
    (task,) = tasks
    spec = STRATEGIES.get(task)
    args.activity, args.embodiment = task, "r1pro"
    args.task = spec.instruction if spec is not None else task.replace("_", " ")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import omnigibson as og
    import torch
    from omnigibson.eval.evaluator import DISABLED_TRANSITION_RULES
    from omnigibson.tiptop.bench import Episode
    from omnigibson.tiptop.knowledge import make_knowledge
    from omnigibson.tiptop.r1pro import load_embodiment_meta
    from omnigibson.tiptop.scene import EpisodeOver
    from b1k.bridge.strategies import strategy_for, task_goal_atoms, task_goal_options

    for rule in DISABLED_TRANSITION_RULES:  # as the evaluator runs: no recipe rule (their garbage is a particle system),
        rule.ENABLED = False  # and GPU dynamics off as the evaluator has them (make_pizza loads and restores so; on,
    #                           PhysX GPU crashed at the first physics step, W2-P setup sweep)
    skillrun.PASSTHROUGH = (EpisodeOver,)  # the harness's own exception: no skill may swallow it
    providers, routing = importlib.import_module(args.providers), load_routing()
    finger_settled(args)
    if args.setup_only:
        client = metadata = press_client = press_meta = None
        embodiment = load_embodiment_meta()
    else:
        client, metadata, press_client, press_meta = connect_planners(args)
        check_imports(metadata, press_meta)  # the step-zero gate line: every module from this checkout, or stop here
        embodiment = metadata["embodiment"]
    planners = {"left": (client, metadata), **({"right": (press_client, press_meta)} if press_client else {})}
    planner = ArmPlanners(client, press_client) if press_client else client  # a right-arm skill plans on the second
    exit_code, summaries, loaded, strategy = 0, [], None, None
    try:
        sim = build_r1pro_sim(args, embodiment)
        host = BenchHost(sim)
        for case in cases:
            key = (int(case["instance"]), case.get("mode") or args.mode)
            if key != loaded:
                base, loaded = load_instance(og, sim, *key, embodiment), key
                log.info(f"loaded {task} instance {key[0]} ({key[1]})")
            if strategy is None:
                strategy = strategy_for(task, task_goal_atoms(sim), options=task_goal_options(sim),
                                        scope=sorted(sim.task_scope()))  # fmt: skip
                knowledge = make_knowledge(args.knowledge, sim, strategy.goal, spec=strategy.spec)
            restore(og, sim, base, case["id"])  # every case starts from the instance as loaded
            t0 = time.monotonic()
            try:
                (state, held), rows = setup(og, sim, args, case, embodiment, host), []
            except CaseSetupError as e:
                log.error(f"SETUP FAILED {e}")
                _append(out / "setup.jsonl", [{"case": case["id"], "task": task, "instance": key[0], "mode": key[1],
                                               "ok": False, "reason": str(e),
                                               "wall_s": round(time.monotonic() - t0, 1)}])  # fmt: skip
                continue
            _append(out / "setup.jsonl", [{"case": case["id"], "task": task, "instance": key[0], "mode": key[1],
                                           "ok": True, "held": {a: r.id for a, r in held.items()},
                                           "wall_s": round(time.monotonic() - t0, 1)}])  # fmt: skip
            if args.setup_only:
                continue
            (out / case["id"]).mkdir(parents=True, exist_ok=True)
            if args.selftest:
                summaries.append(selftest(og, sim, host, state, case, list(range(args.selftest)), out))
                continue
            for i, seed in enumerate(case["seeds"][: args.trials]):
                name = f"{case['id']}_t{i}"
                restore(og, sim, state, name)
                random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
                ep = Episode(sim, args, planners, knowledge, out / case["id"] / f"t{i}", spec=strategy.spec)
                svc, segmenter = providers.pseudo_services(ep, planner, routing, collision=args.collision)
                host.segmenter = segmenter  # the GoalPanel's end-of-run frames carry this trial's masks
                for arm, ref in held.items():  # what the setup put in the hands, as a pick's result would say it
                    svc.world.apply(WorldUpdate("held", ref, arm))
                observer = CaptureObserver(sim, host, segmenter, args.task)
                nav = TeleportNavigator(sim)  # a reach case's go_to; its teleports' sim steps join the U0 check
                call = dataclasses.replace(case["call"], seed=seed, backend=args.backend or case["call"].backend,
                                           arm=pick_arm(case["call"], held))
                backends = make_backends(ep, host, svc)
                lease = lease_for(case, call, seed, SkillRegistry(SPECS, backends, routing))
                video = contextlib.nullcontext() if args.no_video else sim.recording(out / case["id"] / f"{name}.mp4")
                with video:
                    r, rt, calls, wall_s = run_trial(host, svc, backends, routing, observer, call, lease, nav,
                                                     bool(case.get("reach")))
                held_by = next((c for c in calls if lease is not None and c["skill"] == lease.skill), None)
                rows.append(row(case, i, seed, r, rt, sim.n_steps, wall_s, host.env_wall_s, observer.wall_s,
                                observer.steps + nav.steps, held_by, host.frames_wall_s))
                _append(out / f"{case['id']}.jsonl", [rows[-1]])
                _append(out / "skill_calls.jsonl", [dict(c, trial=name) for c in calls])
                _append(out / "goal_checks.jsonl", [dict(g, trial=name) for g in svc.goals.log])
                log.info(f"TRIAL {name}: {r.status.value} {r.code} steps {r.steps} rt {rt.step} "
                         f"charged {rt.charged} idle {rt.idle_steps} latch gap {rows[-1]['latch_gap_rad']} rad"
                         + (f" lease {held_by['status']} {held_by['code']} {held_by['steps']} steps"
                            if held_by else ""))
                assert rows[-1]["u0"], f"{name} broke the step invariant (U0): {rows[-1]}"
            summaries.append(summarize(case, rows))
    except Exception:
        log.exception("skill bench failed")
        exit_code = 1
    finally:
        if summaries and args.selftest:
            with open(out / "selftest.json", "w") as f:
                json.dump(summaries, f, indent=1)
        elif summaries:
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
