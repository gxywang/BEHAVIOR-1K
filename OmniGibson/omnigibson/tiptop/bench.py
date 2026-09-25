"""Benchmark the pipeline on a challenge task the way the 2026 BEHAVIOR Challenge evaluates a policy.

Same task instances (the public test split, indices 0-9 for reported results), same per-instance timeout (1.5x
the mean human demonstration length, in env steps), same metrics (``TaskMetric``: 1 on success, else the newly
satisfied fraction of the best goal option; ``AgentMetric``: base and end-effector displacement), the same result
JSON per rollout as ``omnigibson.eval.eval``. What differs, and is written into every result: the robot is driven
in-process by a task runner (strategies.py) that teleports the base instead of navigating, and the planner may
be told what the simulator knows (``--knowledge oracle``: masks and button poses). Both are stand-ins for parts of
the pipeline that do not exist yet, so a number from this benchmark is an upper bound for the manipulation part,
not a challenge score.

  OMNIGIBSON_HEADLESS=1 python -m omnigibson.tiptop.bench --task-name turning_on_radio --instances 0 1 2 \\
      --host localhost --port 8765 --knowledge oracle --grasping-mode sticky --out-dir runs/bench_radio
"""

import argparse
import copy
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import omnigibson.utils.transform_utils as T

from omnigibson.tiptop.knowledge import GoalNotVisible
from b1k.bridge.protocol import FLOOR_CATEGORIES, INTENT_PREDICATES, bddl_category
from b1k.bridge.judgement import (  # FLOOR_LEVEL and UNSATISFIED_SHOWN moved with the geometry and the
    FLOOR_LEVEL,  # verdict they parameterize; both are re-exported here, where callers still read them
    UNSATISFIED_SHOWN,  # noqa: F401
    box_distance,
    box_edge_gap,
    boxes_beside,
    highest_support,
    on_the_floor,
    placed_over,
    short_atom,
    verdict_caption,
    what_failed,
)
from b1k.bridge.strategies import (
    PLACE_PREDICATES,
    STRATEGIES,
    TransferBlocked,
    Unreachable,
    atom,
    atom_objects,
    wants_home_torso,
)
from omnigibson.tiptop.run import (
    add_common,
    add_planner_args,
    apply_embodiment_posture,
    atom_text,
    build_r1pro_sim,
    connect_planners,
    live_round,
    open_state_stream,
    setup_logging,
)

log = logging.getLogger("omnigibson.tiptop")

REACH_FAR = 1.1  # base-pose search radius (m) when nothing within the usual 0.9 m works: the torso leans that far
FALL_DROP = 0.05  # m below the teleport height (r1pro.FALL_DROP): falling through a floor that is not there
TOPPLED_DEG = 45.0  # at or past this the base is not tilted, it is toppled: 23 of 151 settles, 12 of them 120+
STANCE_ATTEMPTS = 3  # stances tried before a round works from one that did not settle level
BLIND_LIMIT = 3  # DISTINCT stances a goal object must be invisible from before the runner stops trying for it
EPILOGUE_STEPS = 90  # env steps the final state and the verdict stay on screen after the episode (3 s of video)


class Episode:
    """One task instance as a strategy sees it. The base moves by teleport (``stand_for``); the planner of an arm
    plans one round at a time (``plan_and_execute``, which never raises on a failed round: the runner decides what
    to do next); and judges each round without asking the simulator whether it worked: ``holding`` is the robot's
    own hand record (a plan closed the hand and the fingers stopped on something), a placement is a geometric test
    on where the knowledge source localizes the objects (``placed``: the item's box over the target's), a press
    counts once its planned stroke ran. Positions, distances and supports (``support_of``,
    ``edge_gap``) come from the same localization, which the oracle source reads from the simulator and the onboard
    source from the planner's reports. ``pick``, ``achieve`` and ``put_down`` run the rounds under the one retry
    policy every task gets (``--rounds``); there is no other recovery, in here or in a task description."""

    def __init__(self, sim, args, planners: dict, knowledge, out_dir: Path, spec=None):
        self.sim, self.args, self.planners, self.knowledge, self.out_dir = sim, args, planners, knowledge, out_dir
        self.spec = spec  # the task description, for the values a task names for itself (TaskSpec.opens)
        sim.region_refused = set()  # inside-regions that failed to plan, this episode only
        if getattr(sim, "arm", None) in planners:  # a reach no straight ramp can make (a handle, a shelved book)
            sim.move_planner = planners[sim.arm][0]
        self.rounds = args.rounds
        self.records = []  # one per round, in order
        self.blind = {}  # goal object -> the distinct stances it could not be seen from (BLIND_LIMIT gives up)
        self.stood = {}  # names -> (x, y) poses stood at for them, so a retry gets a different viewpoint
        self.opened_at = {}  # container -> (x, y, yaw) it was opened from: its inside was reached from there
        self.floor = sim.floor_name()
        try:  # the pose the scene started the robot at: level by construction, the first place to right it at
            pos, quat = sim.robot.get_position_orientation()
            self.last_level = (float(pos[0]), float(pos[1]), float(T.quat2euler(quat)[2]))
        except Exception:  # noqa: BLE001 - no simulator behind this episode (the strategy tests)
            self.last_level = None

    # ---------------------------------------------------------------- moving
    def open_up(self, name: str, fraction: float | None = None) -> bool:
        """Stand at ``name`` and open it, by ``fraction`` of its joint's range (the scored atom's worth by default).

        Reading the joint back afterwards is privileged, the way the oracle's masks are: the motion reports what
        the joint says, and at evaluation that verdict would have to come from the hand's own travel and a fresh
        look at the container.
        """
        from b1k.bridge.articulation import OPEN_FRACTION_SCORED, is_open
        from omnigibson.tiptop.articulation import openable_joints

        if fraction == 0.0:  # a close is a push on the moving link, not a pull on its handle (r1pro.push_joint)
            self.sim.video_caption = f"close {name}"
            for j in openable_joints(self.sim.scene_object(name)):
                if is_open(j["lower"], j["upper"], j["position"], closed=j["closed"]):
                    result = self.sim.push_joint(self.sim.arm, name, j, j["closed"])
                    self.records.append({"close": name, **result, "step": self.sim.n_steps})
                    if not result.get("reached"):
                        log.info(f"{name}.{j['name']} did not shut: {result.get('why') or 'the joint stopped short'}")
            return self.is_shut(name)
        # open_container chooses its own stance, in front of the container's leading face where the whole pull
        # solves (r1pro.stance_for_grasp); stand_for's stance is built for looking at things and stood 0.9 m off
        # the drawer fronts (2026-09-14)
        self.sim.video_caption = f"open {name}"
        hint = dict((getattr(self.spec, "opens", None) or {}).get(name) or {})
        if hint:
            log.info(f"{name}: opening with the values this task names for it: {hint}")
        result = self.sim.open_container(
            self.sim.arm,
            name,
            fraction=hint.get("fraction", OPEN_FRACTION_SCORED if fraction is None else fraction),
            joint=hint.get("joint"),
            height=hint.get("height"),
        )
        # The opening stance is chosen because the PULL solves from it (stance_for_grasp), and nothing checks that
        # the arm can get to where the pull starts. store_honey's drawer is the case: the stance reports "15 of 15
        # pull waypoints solve (80% of the range)" and then the hand stops 40 cm short of the standoff. It fails
        # that way every time -- 4 attempts over two builds -- while the generic looking stance this replaced
        # opened the same drawer 3 times out of 3 (13.0, 12.9 and 30.1 cm of travel, 2026-09-13/14). So when the
        # approach is what failed, stand the old way and pull again from there. A fallback, not a replacement:
        # the pull-solving stance is still tried first and still wins wherever it works.
        why_not = str(result.get("why") or "")
        if not result.get("opened") and ("on the way to the standoff" in why_not or "stance rejected" in why_not):
            self.records.append({"open": name, **result, "step": self.sim.n_steps})  # the first attempt's verdict
            log.info(
                f"{name}: the opening stance solves the pull but the arm cannot reach its start "
                f"({result.get('why')}); standing the way the runner stands to look, and pulling from there"
            )
            try:
                self.stand_for(name)
                retry = self.sim.open_container(
                    self.sim.arm,
                    name,
                    fraction=hint.get("fraction", OPEN_FRACTION_SCORED if fraction is None else fraction),
                    joint=hint.get("joint"),
                    height=hint.get("height"),
                    stand=False,
                )
                self.records.append({"open": name, **retry, "step": self.sim.n_steps, "after_standoff_failed": True})
                if retry.get("opened"):
                    log.info(f"{name} opened from the looking stance after the opening stance could not reach it")
                else:
                    log.info(f"{name} did not open from either stance: {retry.get('why') or 'the joint did not move'}")
                return bool(retry.get("opened"))
            except Exception as why:  # noqa: BLE001 - Unreachable, or no stance at all: the first verdict stands
                log.info(f"{name}: no looking stance to fall back to ({type(why).__name__}: {why})")
            return False  # the first attempt failed at the standoff and the fallback did not rescue it
        self.records.append({"open": name, **result, "step": self.sim.n_steps})
        if not result.get("opened"):
            log.info(f"{name} did not open: {result.get('why') or 'the joint did not move'}")
        elif result.get("stance"):
            self.opened_at[name] = tuple(float(v) for v in result["stance"])
        return bool(result.get("opened"))

    def is_floor(self, name: str) -> bool:
        """Whether ``name`` is a floor -- ANY floor, not just the first one in the task's scope.

        ``sim.floor_name()`` returns the first floor it finds and ``self.floor`` holds that one, so comparing a
        goal's container against it by name breaks the moment a task names a different floor. laying_tile_floors
        asks for tiles ontop floor.n.01_2 and crashed with "no object 'floor.n.01_2' in scene
        office_cubicles_right", because the runner did not recognise it as a floor and went looking for a piece
        of furniture to stand at; bringing_in_wood targets the same floor and scored 0.000 with no rounds run
        (2026-09-15). A floor is a floor by category, not by being listed first.
        """
        return bool(name) and bddl_category(name) in FLOOR_CATEGORIES

    def switched_on(self, name: str) -> bool | None:
        """Whether ``name``'s switch is on right now; None when it has no such state.

        Reading it back is privileged in exactly the way the oracle's masks are -- at evaluation this would have
        to come from looking at the thing. It is read for one purpose only: not pressing a switch that is already
        in the state the goal wants. installing_a_modem's (:init) contains (toggled_on modem.n.01_1), so the modem
        starts ON and its goal asks for it ON; pressing it would turn it OFF and destroy a condition the task was
        already being given credit for (2026-09-14).
        """
        from omnigibson.object_states import ToggledOn

        try:
            obj = self.sim.scene_object(name)
            state = obj.states.get(ToggledOn)
            return None if state is None else bool(state.get_value())
        except Exception:  # noqa: BLE001 - no such state, or an object the scope does not name
            return None

    def is_shut(self, name: str) -> bool:
        """Whether every joint of ``name`` that could open is closed."""
        from b1k.bridge.articulation import is_open
        from omnigibson.tiptop.articulation import openable_joints

        try:
            joints = openable_joints(self.sim.scene_object(name))
        except Exception:
            return False
        return bool(joints) and not any(
            is_open(j["lower"], j["upper"], j["position"], closed=j.get("closed")) for j in joints
        )

    def stance_key(self) -> tuple:
        """Where the robot is standing, coarsely: 10 cm and 15 degrees. Two rounds run from the same spot see the
        same things, and two run from different spots do not, which is what the blind count has to distinguish."""
        try:
            pos, quat = self.sim.robot.get_position_orientation()
        except AttributeError:  # a runner driving this without a simulator (the strategy tests) has no stance
            return None
        yaw = float(T.quat2euler(quat)[2])
        return (round(float(pos[0]) / 0.1), round(float(pos[1]) / 0.1), round(yaw / (np.pi / 12)))

    def stand_for(self, *names: str) -> dict:
        """Teleport the base to a pose from which the named objects are in the left arm's reach and in view. A
        second call for the same objects stands somewhere else; when nothing is found within the arm's usual
        reach, the search is widened to ``REACH_FAR`` (the torso leans that far) before giving up.

        The base is read back after it settles (``settled_level``): a pose that intersects furniture is resolved
        by the physics lifting and rolling the whole robot, and since every field of a request is expressed in the
        base frame, a tilted base hands the planner the whole scene tilted. Over runs/bench_batteries_ten only 1
        of the 14 rounds that ran from a base off level executed, against 29 of the 46 that ran from a level one,
        so such a pose is treated as occupied and the search is asked for another (2026-09-13).
        """
        avoid = self.stood.setdefault(names, [])
        # What the landing check refuses is judged at THIS search's posture and load, so it lives for this search
        # only -- the widened retry and another attempt after an off-level settle, not the next round's search.
        refused = []
        self.sim.video_caption = f"teleport: stand for {', '.join(names)}"
        try:  # where the robot stood before the search, which it was working from
            was_pos, was_quat = self.sim.robot.get_position_orientation()
            was = (float(was_pos[0]), float(was_pos[1]), float(T.quat2euler(was_quat)[2]))
        except Exception:  # noqa: BLE001 - no simulator (the strategy tests): nothing to stand back at
            was = None
        # A topple poisons everything after it, not just the round it happened in. base_box and every footprint
        # test are computed in the BASE FRAME, so once the robot is on its back no stance is ever free: 
        # clean_up_your_desk topples on its first teleport at 179.3 deg and then fails all 24 of its stance
        # searches and runs zero rounds. The exit below only stands it up after STANCE_ATTEMPTS are exhausted, and
        # this task never gets that far -- the next search raises Unreachable first. So check on the way IN too,
        # and put it back at the last pose that was known level (2026-09-15).
        upright = self.last_level if getattr(self, "last_level", None) else was
        if upright is not None and was is not None:
            level_now, why_now = self.sim.settled_level(float(was[0]), float(was[1]))
            if not level_now and self.fallen():
                log.warning(f"{why_now} before the search even starts; righting it at ({upright[0]:.2f}, "
                            f"{upright[1]:.2f}) -- every footprint test is computed in the base frame")
                self.right(upright)
        fell = False
        known = self.opened_at.get(names[0]) if len(names) == 1 else None
        for attempt in range(STANCE_ATTEMPTS):
            try:
                pose = None
                if known is not None and all(np.hypot(known[0] - x, known[1] - y) > 0.05 for x, y in avoid):
                    try:  # the stance it was opened from: the door is open and its inside was reached from there
                        self.sim.place_robot(*known, note=f"stand where {names[0]} was opened from")
                        pose = {"x": known[0], "y": known[1], "yaw": known[2]}
                    except RuntimeError as why:
                        log.info(f"the stance {names[0]} was opened from is refused now ({why}); searching")
                if pose is None:
                    pose = self.sim.place_robot_for(*names, avoid=avoid, refused=refused)
            except RuntimeError as e:
                log.info(f"{e}; widening the search to {REACH_FAR} m")
                try:
                    pose = self.sim.place_robot_for(*names, reach=REACH_FAR, avoid=avoid, refused=refused)
                except RuntimeError as far:
                    self.records.append({"stand_for": list(names), "error": str(far), "step": self.sim.n_steps})
                    raise Unreachable(str(far)) from far
            avoid.append((pose["x"], pose["y"]))
            self.sim.hold(self.args.settle_steps, self.sim.last_gripper)  # a held object keeps its gripper closed
            level, why = self.sim.settled_level(pose["x"], pose["y"])
            self.records.append(
                {"stand_for": list(names), "pose": pose, "step": self.sim.n_steps, "level": level, "why": why}
            )
            if level:
                self.last_level = (float(pose["x"]), float(pose["y"]), float(pose["yaw"]))
                return pose
            fell = self.fallen()  # read before right(), whose own settle check replaces last_settle
            if fell and upright is not None:
                # Searching again from a fallen base rejects every candidate (the tests read the live base frame)
                self.right(upright)
            log.warning(
                f"{why} at ({pose['x']:.2f}, {pose['y']:.2f}): the pose is occupied by something the footprint "
                f"test missed"
                + (
                    f"; standing somewhere else ({attempt + 1}/{STANCE_ATTEMPTS})"
                    if attempt + 1 < STANCE_ATTEMPTS
                    else "; out of attempts"
                )
            )
        # Out of attempts. A small tilt is workable and the runner has always carried on from one -- measured over
        # every run, 66 of 151 settles are under 5 deg off level and 43 more under 15. But 23 are 45 deg or worse
        # and 12 of those are 120+, which is the robot on its back: clean_up_your_desk settles at 178.6 deg and
        # then runs ZERO rounds, in every run it has ever had, because every field of a request is expressed in a
        # base frame that is upside down. Standing the robot back where it came from and reporting the objects
        # unreachable is the honest outcome; working from a toppled base is not (2026-09-15).
        if fell:  # the last attempt's reading: after right() the live one is the righted robot, level
            log.warning(f"{why} after {STANCE_ATTEMPTS} attempts; righted rather than working from a fallen base")
            if upright is None:
                log.warning("no level pose is known to right it at")
            raise Unreachable(f"no pose for {list(names)} leaves the base level ({why})")
        log.info(f"out of attempts ({why}), which is workable; going on from here")
        return pose

    def fallen(self) -> bool:
        """Whether the last settle read was a fall or a topple, not a workable tilt (settled_level's numbers)."""
        settle = getattr(self.sim, "last_settle", None) or {}
        return settle.get("drop_m", 0.0) > FALL_DROP or settle.get("tilt_deg", 0.0) >= TOPPLED_DEG

    def right(self, pose) -> None:
        try:
            if self.sim.right_robot(float(pose[0]), float(pose[1]), float(pose[2])):
                self.last_level = tuple(float(v) for v in pose)
        except Exception as why:  # noqa: BLE001 - best effort; the search and the round report what follows
            log.warning(f"could not right the robot ({type(why).__name__}: {why})")

    def has_arm(self, arm: str) -> bool:
        return arm in self.planners

    def use_arm(self, arm: str) -> None:
        if arm != self.sim.arm:
            # the new planner locks this arm at its ready posture; a sticky grasp leaves it wherever it closed
            if not self.sim.return_to_ready(note=f"{self.sim.arm} arm back to ready before the {arm} arm plans"):
                log.warning(f"the {self.sim.arm} arm could not get back to its ready posture before the {arm} arm plans")
            self.sim.adopt_embodiment(self.planners[arm][1]["embodiment"])
            self.sim.move_planner = self.planners[arm][0]

    # ---------------------------------------------------------------- planning rounds
    def plan_and_execute(self, atoms: list[dict], arm: str = "left", floor: bool = False) -> dict:
        """One capture / plan / execute round for ``atoms`` with the planner of ``arm``. A planner failure, an
        object out of view or an execution error is recorded and returned as {"error": ...}; the episode's end
        (``EpisodeOver``) propagates."""
        from omnigibson.tiptop.scene import EpisodeOver

        # What this round MOVES, as opposed to what it moves things relative to. The planner owns a movable and
        # must be able to reach it, so a movable is never geometry to avoid; a fixture is the opposite. Deciding
        # by goal role rather than by size is the correction that made boxing_books plannable: the books were
        # being shipped as obstacles, and cuRobo cannot grasp what it must avoid.
        i = len(self.records)
        # An object no capture can see is not worth another capture. Instance 301 of assembling_gift_baskets spent
        # rounds 7 to 15 -- nine rounds, thirteen minutes, a quarter of its step budget -- on swiss_cheese_2, which
        # was invisible from every stance it tried, and then hit the time limit with eight atoms still open
        # (2026-09-13). The strategy's per-item cap does not cover this: it counts transfers, and each failed
        # transfer starts put-down rounds on the same unseeable object. Giving up on the object frees the budget
        # for atoms that can still be had; seeing it once anywhere clears the count.
        unseeable = [o for o in atom_objects(atoms) if len(self.blind.get(o, ())) >= BLIND_LIMIT]
        if unseeable:
            record = {
                "round": i,
                "atoms": atoms,
                "arm": arm,
                "step": self.sim.n_steps,
                "error": f"GoalNotVisible: skipped, {', '.join(unseeable)} not seen from {BLIND_LIMIT} stances",
                "seconds": 0.0,
            }
            self.records.append(record)
            log.info(f"round {i} {atom_text(atoms)} [{arm}]: {record['error']}")
            return record
        round_dir = self.out_dir / f"r{i:02d}_{arm}_{atoms[0]['predicate']}"
        round_dir.mkdir(parents=True, exist_ok=True)
        self.sim.video_caption = f"round {i}: {atom_text(atoms)} [{arm} arm]"
        record = {"round": i, "atoms": atoms, "arm": arm, "dir": str(round_dir), "step": self.sim.n_steps}
        t0 = time.time()
        try:
            self.use_arm(arm)
            client = self.planners[arm][0]
            client.wait_for_server(timeout_s=300.0)  # a planner relaunched after a CUDA fault comes back in ~1 min
            result = live_round(  # the episode's video covers the round; no clip of its own
                self.sim, self.args, client, round_dir, atoms, self.knowledge, floor=floor, score=False, record=False
            )
            record["env_steps"] = result.get("execution", {}).get("env_steps")
            if result.get("execution", {}).get("error"):
                record["error"] = result["execution"]["error"]
            # the step the plan began executing from: knowledge older than this predates what the plan did
            record["executed_from"] = result.get("execution", {}).get("start_step")
        except EpisodeOver:
            record["error"] = "episode over"
            record["seconds"] = round(time.time() - t0, 1)
            self.records.append(record)
            raise
        except Exception as e:  # noqa: BLE001 - one failed round must not end the instance
            log.exception(f"round {i} {atom_text(atoms)} failed")
            record["error"] = f"{type(e).__name__}: {e}"
            if "No satisfying particles" in str(e) and "time budget" not in str(e):
                # A compartment floor too small for the item, or already full: the next try uses the hull's top,
                # which is what 747 of last week's 768 executed inside-rounds landed with (r1pro.inside_regions).
                # Not on a timeout (the region was never fully tried), and only for pairs this round sent one for.
                refused = self.sim.__dict__.setdefault("region_refused", set())
                sent = getattr(self.sim, "region_sent", set())
                refused.update(tuple(a["args"]) for a in atoms
                               if a["predicate"] == "inside" and len(a["args"]) == 2 and tuple(a["args"]) in sent)
            if isinstance(e, GoalNotVisible):
                for name in atom_objects(atoms):
                    self.blind.setdefault(name, set()).add(self.stance_key())
        else:
            for name in atom_objects(atoms):  # the capture saw them; whatever hid them before is no longer hiding
                self.blind.pop(name, None)
        record["seconds"] = round(time.time() - t0, 1)
        self.records.append(record)
        log.info(f"round {i} {atom_text(atoms)} [{arm}]: {record.get('error') or 'executed'} ({record['seconds']}s)")
        return record

    # ---------------------------------------------------------------- the retry policy, the same for every task
    def satisfied(self, atoms: list[dict], record: dict | None = None) -> bool:
        """Whether every atom holds, judged from the robot's own readings and localization: ``holding`` by the hand
        record, a placement by ``placed``, ``nextto`` by ``beside``, ``open`` by the container's own joint, and a
        press by its round having run without error (the switch's state is the simulator's to know, so a press is
        open loop).

        Reading a joint back is privileged, like the oracle's masks: at evaluation an ``open`` verdict would have
        to come from the hand's travel and a fresh look at the container.

        A predicate the runner does not know is NOT taken to hold because a round ran. That is what this did, and
        it would score every new predicate satisfied the moment a round was attempted -- the first task to name one
        would be reported as solved without anything having been achieved.

        A placement is judged only on knowledge from AFTER the round's plan ran (``record["executed_from"]``). The
        oracle reads the scene live, so this costs it nothing; the onboard source remembers its last look, which
        was the capture the plan was made from, and where the item was before the plan is no evidence of where it
        is now. Older knowledge counts as unknown, and unknown is unfinished: the retry is the fresh look.
        """
        ran = record is not None and not record.get("error")
        after = (record or {}).get("executed_from")
        for a in atoms:
            predicate, args = a["predicate"], a["args"]
            if predicate == "holding":
                ok = self.holding(args[0])
            elif predicate == "nextto" and len(args) == 2:
                ok = self.beside(args[0], args[1], after=after)
            elif predicate in (*PLACE_PREDICATES, "attached") and len(args) == 2:
                # and the task's own evaluator: the geometry takes an item up to 15 cm over the target's top as
                # in it, and 16 of 18 executed inside(x, bookcase) rounds ended on the top board (2026-09-23).
                # under(x, f) has no box geometry (the item is below f's bottom) and attached(x, y) is a snap of
                # meta links no box can see: the evaluator alone
                ok = self.goal_already_holds(predicate, *args) and (
                    predicate in ("under", "attached") or self.placed(args[0], args[1], after=after)
                )
            elif predicate in INTENT_PREDICATES:
                ok = ran  # a stamp, cut, heat or aim is open loop like a press: what it changed is the evaluator's
            elif predicate == "open" and args:
                ok = not self.is_shut(args[0])
            elif predicate == "not" and len(args) >= 2 and args[0] == "open":
                ok = self.is_shut(args[1])
            elif predicate == "toggled_on" or (predicate == "not" and "toggled_on" in args):
                ok = ran  # a press is open loop: it ran, and the switch's state is the simulator's to know
            else:
                log.warning(f"no test for {predicate}({', '.join(args)}); the round counts as unfinished")
                ok = False
            if not ok:
                return False
        return True

    def beside(self, item: str, other: str, after=None) -> bool:
        """``judgement.boxes_beside`` on the boxes the knowledge source localizes; False when either is unknown."""
        boxes = self.boxes(item, other, after=after)
        if item not in boxes or other not in boxes:
            log.info(f"cannot judge nextto({item}, {other}): {self.unknown(boxes, item, other)} not localized")
            return False
        return boxes_beside(boxes[item], boxes[other])

    def achieve(self, atoms: list[dict], arm: str = "left", floor: bool | None = None, done=None) -> bool:
        """Up to ``--rounds`` planning rounds for ``atoms`` with the planner of ``arm``, stopping as soon as
        ``done()`` (default: ``satisfied``); the retry every goal of every task gets, and the only one. ``floor``
        (the planner's workspace reaches the floor) is read off the target when not given: a container or support
        that stands on the floor."""
        if floor is None:  # under(x, f) is a floor placement inside f's footprint, wherever f's bottom is
            floor = any(a["predicate"] == "under" for a in atoms) or self.reaches_floor(
                *[a["args"][1] for a in atoms if len(a["args"]) == 2]
            )
        for attempt in range(self.rounds):
            # A container that has shut again has no interior to place into, and it does shut: store_honey's
            # drawer was pulled to 0.200 of 0.39 and read "every joint is shut" by the placing round, so
            # inside_region refused and the place went back onto the lid. Reopening needs the hand, so a round
            # already carrying the item says so rather than dropping it to try.
            for a in atoms:
                if a["predicate"] == "inside" and len(a.get("args", ())) == 2 and self.is_shut(a["args"][1]):
                    if self.holding(a["args"][0]):
                        log.info(f"{a['args'][1]} is shut and the hand is full; this round cannot reopen it")
                    else:
                        log.info(f"{a['args'][1]} is shut again; opening it before this round")
                        self.open_up(a["args"][1])
            where = self.stance_key()
            record = self.plan_and_execute(atoms, arm=arm, floor=floor)
            # The planner fits its support plane by RANSAC and refuses the whole request when it finds no
            # near-horizontal plane at all (segmentation.py: "No plane found with objects resting on it"). A press
            # does not need a surface, but it does need the planner to find one: a switch on a corridor wall gives
            # it a picture with nothing horizontal in it, and turning_out_all_lights_before_sleep lost all 10 of
            # its rounds that way. The floor is the horizontal plane that always exists, and the press round
            # crops it out -- `floor` is read off two-argument atoms and toggled_on(x) has one.
            # Retried rather than defaulted, because turning_on_radio presses happily without the floor today and
            # a wider workspace is not free (2026-09-15).
            if not floor and "No plane found" in str(record.get("error") or ""):
                log.info("the planner found nothing horizontal to call a table; asking again with the floor in view")
                floor = True
                continue
            if done() if done is not None else self.satisfied(atoms, record):
                return True
            # A round that could not see its goal executed nothing, so the scene is unchanged and the robot has
            # not moved: capturing again from the same spot asks a question already answered, and pays the full
            # price of a capture to hear the same answer. The masks come back identical to a pixel or two. Over
            # the runs read on 2026-09-14 these repeats cost 13% of an episode of putting_away_toys and 38% of
            # one of assembling_gift_baskets -- 45186 env steps across the corpus, several rounds' worth.
            #
            # The blind counter does not catch this any more: it counts DISTINCT stances, so a second look from
            # the same one never increments it. That change was right for its own purpose (an object invisible
            # from here may be visible from somewhere else) and wrong for this one.
            if (
                attempt + 1 < self.rounds
                and str(record.get("error", "")).startswith("GoalNotVisible")
                and where is not None
                and self.stance_key() == where
            ):
                log.info(
                    f"not capturing {atom_text(atoms)} again from the same spot: nothing moved and nothing was seen"
                )
                break
        return False

    def pick(self, bddl: str, into: str | None = None) -> bool:
        """The object in the planned hand after up to ``--rounds`` pick rounds, each from a fresh base pose (a pick
        that fails, no plan or the object hidden, is retried from somewhere else). False when no pose reaches it.
        ``into``: where it is bound for; a roof within the hand stack over that or over where it rests (a shelf,
        a fridge bay: ``sim.side_entry``) has the planner take it from the side, this round only
        (``OracleKnowledge.describe`` -> ``request["side_grasp"]``)."""
        side = any(self.sim.side_entry(bddl, c) for c in (into, self.support_of(bddl)) if c and not self.is_floor(c))
        for _ in range(self.rounds):
            try:
                self.stand_for(bddl)
            except Unreachable as e:
                log.warning(f"{bddl}: {e}")
                return False
            if self.sim.push_face(bddl) is not None:
                # a flat item under a shelf board offers no pinch: slid to the board's edge first (N-push), so the
                # round after it can take the overhang; push_face is None again once it hangs over the edge
                self.achieve([atom("push", bddl)])
            # where the item is *now*: one that was knocked to the floor needs the workspace to reach down to it
            self.sim.side_grasp = {bddl} if side else set()
            self.plan_and_execute([atom("holding", bddl)], floor=self.reaches_floor(bddl))
            self.sim.side_grasp = set()
            if self.holding(bddl):
                return True
            if self.args.grasping_mode != "sticky":
                # the press closes the hand before it touches (press_grasp), so the assisted weld's finger-to-finger
                # ray has nothing to hit (robot.py ~3150): only the planned grasp can take it
                continue
            # M2T2 proposes grasps from the point cloud and a flat object -- a book lying down, a board game --
            # gives it no side a parallel jaw can get under. The sticky fallback reaches a clear standoff,
            # closes, then seeks gentle finger contact with a physical surface. It stops advancing at contact
            # rather than pushing the open gripper into the target.
            try:
                # The pressed grasp physically takes the object -- close_on presses until the assist reports it
                # holds -- but nothing wrote the robot's OWN hand record, which is only ever written by
                # note_hands after a planner round. So holding() was False, the branch failed, and the one grasp
                # built for flat objects had its success thrown away. Three tasks were losing every flat-object
                # pick this way (2026-09-15). The record is written by the same rule note_hands uses: the
                # knowledge source's localization, and the fingers only when nothing can localize it -- not from
                # the simulator's grasp assist, which stays diagnostic (scene.check_hands).
                support = self.support_of(bddl)  # None when it was never perceived: nothing to spare
                pressed_from = self.sim.n_steps
                self.sim.move_planner = self.planners[self.sim.arm][0]  # for an approach no straight ramp can make
                pressed = self.sim.press_grasp(self.sim.arm, bddl, spare=(support,) if support else ())
                taken = getattr(self.sim.robot, "_ag_obj_in_hand", {}).get(self.sim.arm)
                if pressed:
                    self.note_pressed_grasp(bddl, after=pressed_from)
                elif taken is not None and taken is self.sim.scene_object(bddl):
                    # the hand closes before it approaches, so the assist can take the target on the way in and
                    # the close at the precontact pose is then refused (the finger is already in it): the target
                    # IS in the closed hand. It went unrecorded and ended assembling_gift_baskets at step 1031 of
                    # 39090 on "an unconfirmed grasp on pillar_candle_88" (2026-09-22)
                    log.info(f"{bddl}: the assist took it on the way in; recording it by the usual rule")
                    self.note_pressed_grasp(bddl, after=pressed_from)
                if self.holding(bddl):
                    log.info(f"{bddl}: taken by the contact-seeking sticky grasp")
                    self.records.append(
                        {
                            "round": len(self.records),
                            "atoms": [atom("holding", bddl)],
                            "arm": self.sim.arm,
                            "step": self.sim.n_steps,
                            "pressed_grasp": True,
                        }
                    )
                    # A planned pick ends where GoToInitial leaves it, the ready posture; this one ended crouched
                    # over the object with the head camera at 0.75-1.04 m, and the destination stance was searched
                    # from there: sticky carries lost 70% of those searches against 13% from ready, and were put
                    # back 52% of the time against 8% (2026-09-23). Same posture for both, carrying.
                    self.sim.return_to_ready(note=f"ready posture with {bddl} in hand",
                                             allowed_contacts=self.sim.retreat_contacts(bddl))
                    return True
                # Sticky attachment can succeed on the wrong object, or disagree with localization. Let go of it
                # where it is (it was never lifted) before another approach; only a hand that will not let go
                # blocks the episode.
                attached = getattr(self.sim.robot, "_ag_obj_in_hand", {}).get(self.sim.arm)
                if attached is not None:
                    from omnigibson.tiptop.run import DROP_STEPS

                    log.info(f"{bddl}: the assist holds {attached.name}, which the hand record does not; letting go")
                    self.sim.hold(DROP_STEPS, self.sim.OPEN)
                    attached = getattr(self.sim.robot, "_ag_obj_in_hand", {}).get(self.sim.arm)
                    if attached is not None:
                        raise TransferBlocked(
                            f"pickup of {bddl} left an unconfirmed grasp on {attached.name}; object retained"
                        )
                # A failed pressed grasp leaves the hand where the contact seek stopped: down at the floor for a
                # tile, and every stance after it refused for fingers at the floor (laying_tile_floors, 2026-09-22)
                self.sim.return_to_ready(note=f"back to ready after the pressed grasp of {bddl}",
                                         allowed_contacts=self.sim.retreat_contacts(bddl))
                # the closed hand pushed against things on the way back and the assist can take one: left attached
                # and unrecorded it rode into every later landing check (setup_a_bar 302 ran no round, 2026-09-23)
                stray = getattr(self.sim.robot, "_ag_obj_in_hand", {}).get(self.sim.arm)
                if stray is not None and not self.holding(bddl):
                    from omnigibson.tiptop.run import DROP_STEPS

                    log.info(f"{bddl}: the assist took {stray.name} on the way back; letting go")
                    self.sim.hold(DROP_STEPS, self.sim.OPEN)
            except TransferBlocked:
                raise
            except Exception as why:  # noqa: BLE001 - a fallback must not end the instance
                log.warning(f"{bddl}: the pressed grasp failed ({type(why).__name__}: {why})")
        log.warning(f"{bddl}: not in the hand after {self.rounds} pick rounds")
        return False

    def note_pressed_grasp(self, bddl: str, after=None) -> bool:
        """Enter a pressed grasp in the robot's own hand record, by the same rule a planner round uses.

        ``note_hands`` (run.py) decides a pick worked from the knowledge source's localization, falling back to
        the fingers when nothing can localize the object. The pressed grasp goes through none of that -- it is not
        a planner round -- so its success was invisible to ``holding()``. ``after``: the step the grasp started
        at; a localization older than that is from before the hand moved and does not count.
        """
        from omnigibson.tiptop.run import DROP_STEPS, in_hand_by_localization

        try:
            at_hand = (
                in_hand_by_localization(self.sim, self.knowledge, bddl, self.sim.arm, after=after)
                if self.knowledge
                else None
            )
            held = self.sim.grasp_sensed(self.sim.arm) if at_hand is None else at_hand
        except Exception as why:  # noqa: BLE001 - never localized: the fingers decide
            log.info(f"{bddl}: could not localize after the pressed grasp ({type(why).__name__}); reading the fingers")
            held = self.sim.grasp_sensed(self.sim.arm)
        if held:
            self.sim.held_objects[self.sim.tracked_label(bddl)] = self.sim.arm
            log.info(f"hands now hold {self.sim.hands()} after the pressed grasp")
        else:
            log.info(f"{bddl}: the hand pressed onto it but it is not at the hand")
            # The same hazard the planner path has: the fingers stopped on SOMETHING and it is not the target, so
            # the hand is shut on whatever else was under it -- a desk, a shelf, the container. Left shut it gets
            # carried for the rest of the episode (see run.py's note_hands, and 8c8ae6484).
            if self.sim.grasp_sensed(self.sim.arm):
                log.info(f"the {self.sim.arm} hand is shut on something that is not {bddl}; opening it before going on")
                self.sim.hold(DROP_STEPS, self.sim.OPEN)
        return bool(held)

    def pour(self, item: str, target: str) -> bool:
        """Tip what the hand holds out over ``target`` (N-rotate, spec S38): a keep-hold placement of ``item`` on it
        (``pour`` reaches the planner as on(item, target); ``run.do_execute`` cuts the plan with ``keep_holding`` so
        the hand stops above the target, still holding), then the wrist turns, waits and turns back
        (``sim.tilt_wrist``). False when the round did not run or a turn was stopped; the item stays in the hand."""
        if not self.achieve([atom("pour", item, target)]):
            return False
        self.sim.video_caption = f"pour {item} over {target} [{self.sim.arm} arm]"
        return bool(self.sim.tilt_wrist(self.sim.arm))

    def put_down(self, bddl: str, support: str, floor: bool | None = None) -> bool:
        """Place the object on the requested support and verify both placement and release."""
        placed = self.achieve(
            [atom("ontop", bddl, support)],
            floor=floor,
            done=lambda: not self.holding(bddl) and self.goal_already_holds("ontop", bddl, support),
        )
        if not placed and self.is_floor(support):  # free_hand walks off instead of trying the same spot again
            self.__dict__.setdefault("floor_failed_at", set()).add(self.stance_key())
        return placed

    # ---------------------------------------------------------------- the robot's own record
    def holding(self, bddl: str) -> bool:
        """Whether a hand holds the object, by the robot's own record (a plan closed the hand on it and the
        fingers stopped on something; see ``run.note_hands``)."""
        return self.sim.tracked_label(bddl) in self.sim.hands()

    def held_names(self) -> list[str]:
        """BDDL names of the task objects in the hands (the robot's own record)."""
        return [self.sim.bddl_names[label] for label in self.sim.hands() if label in self.sim.bddl_names]

    def release(self, steps: int = 45) -> None:
        """Explicitly open the planned hand and wait ``steps`` for release.

        Not a validated put-down: the runner uses it only as the last resort (``Runner.free_hand``) after both
        planned floor put-downs fail, since an object left in the hand scores nothing and blocks every pick after it.
        """
        self.sim.video_caption = f"release [{self.sim.arm} arm]"
        self.sim.hold(steps, self.sim.OPEN)
        for label, arm in list(self.sim.hands().items()):
            if arm == self.sim.arm:
                self.sim.held_objects.pop(label, None)
        self.records.append({"release": True, "step": self.sim.n_steps, "hands": dict(self.sim.hands())})

    # ---------------------------------------------------------------- localization (the knowledge source's)
    def boxes(self, *bddl_names: str, after=None) -> dict:
        """name -> {center, lo, hi} (world frame) for every named object the knowledge source can localize. The
        floor has no box, and an object the source has never perceived is left out rather than made up: each
        caller below says what an unknown position means for it. ``after``: only knowledge observed after that
        env step counts -- a box that carries ``step`` is a remembered look (the onboard source), one that does not
        is a live reading (the oracle) and always counts."""
        boxes = self.knowledge.localize(*[n for n in bddl_names if not self.is_floor(n)])
        if after is not None:
            boxes = {n: b for n, b in boxes.items() if b.get("step") is None or b["step"] > after}
        return boxes

    @staticmethod
    def unknown(boxes: dict, *names: str) -> str:
        """The names among ``names`` that ``boxes`` does not localize, for a log line."""
        return ", ".join(n for n in names if n not in boxes)

    def position(self, bddl: str) -> np.ndarray:
        return self.boxes(bddl)[bddl]["center"]

    def distance(self, a: str, b: str) -> float:
        """Horizontal distance between two localized objects; KeyError when either has never been perceived
        (``Runner.gap`` reads that as infinitely far)."""
        boxes = self.boxes(a, b)
        return box_distance(boxes[a], boxes[b])

    def placed(self, item: str, target: str, after=None) -> bool:
        """Whether the item ended on or in the target, by geometry: its centre inside the target's footprint and
        its bottom anywhere from 2 cm under the target's bottom (inside a container) to 15 cm above its top (on a
        surface). Onto the floor: the hand let go of it. Unknown (either never perceived, or not since ``after``)
        is not placed: nothing says it is."""
        if self.is_floor(target):
            return not self.holding(item)
        boxes = self.boxes(item, target, after=after)
        if item not in boxes or target not in boxes:
            log.info(
                f"cannot judge whether {item} is in {target}: {self.unknown(boxes, item, target)} not localized"
                + (f" since step {after}" if after is not None else "")
            )
            return False
        return placed_over(boxes[item], boxes[target], from_bottom=True)

    def support_of(self, bddl: str) -> str | None:
        """The BDDL name of the task object the item stands on (the highest one whose footprint holds it, any
        category), else the task's floor; None when the item has never been perceived, since nothing can be said
        about what an object stands on before it has been seen. Candidates never perceived cannot be the answer."""
        names = [n for n in self.sim.task_scope() if n != bddl and not self.is_floor(n) and bddl_category(n) != "agent"]
        boxes = self.boxes(bddl, *names)
        if bddl not in boxes:
            log.info(f"{bddl} has never been perceived; what it stands on is unknown")
            return None
        return highest_support(boxes[bddl], {n: boxes[n] for n in names if n in boxes}) or self.floor

    def walk_to_floor(self, name: str) -> bool:
        """Teleport to somewhere on the floor ``name``, so a thing carried there can be set down on it.

        Returns False when the floor cannot be located; the caller retains the object. A floor is a wide, flat
        object, so the stance search is given
        its centre to aim at rather than the whole of it.
        """
        try:
            obj = self.sim.scene_object(name)
        except Exception as exc:  # noqa: BLE001 - a floor the scope does not resolve to a simulated object
            log.info(f"cannot locate {name} ({exc}); retaining the held item")
            return False
        if obj is None:
            return False
        try:
            self.stand_for(name)
            return True
        except Unreachable as exc:
            log.info(f"no stance reaches {name} ({exc}); retaining the held item")
            return False

    def goal_already_holds(self, predicate: str, *args: str) -> bool:
        """Whether the goal's own atom holds right now, by the task's evaluator (any arity: ``real(x)``,
        ``covered(t, s)``, ``ontop(x, y)``).

        Privileged in the same way ``switched_on`` is. Used to skip completed work and verify the exact
        destination after placement, including named floors and inside versus ontop. The geometric stand-in it replaces could not tell one floor
        from another -- ``near_floor`` asks only how low a thing stands and ``placed`` reads a floor target as
        "the hand let go of it" -- so a task that asks for things to be carried to a DIFFERENT room's floor was
        read as already finished. bringing_in_wood is that task and it ran no rounds at all.

        At evaluation this verdict would have to come from the robot's own localization. The benchmark score
        comes from the simulator's final check independently of this per-transfer verdict.
        """
        try:
            return bool(self.sim.holds(predicate, *args))
        except Exception as exc:  # noqa: BLE001 - an unknown predicate or an object the scope does not name
            log.debug(f"cannot judge {predicate}({', '.join(args)}) yet ({exc}); treating it as still to do")
            return False

    def fixture_for(self, ability: str, near: str | None = None) -> str | None:
        """The scene fixture (fixed base: part of the map) nearest ``near`` (a task object; else the robot) whose
        category's synset has the BDDL ``ability`` ("heatSource", "coldSource"), one with no openable joint first: a
        door gates the oven's and the microwave's heat, a burner needs none (spec 6.1.6). Tracked under its scene
        name so goal atoms can name it (``toggled_on(stove_ykretu_0)`` translates like a task object's); None when
        the scene has none. Where a fixture stands is the map's; what heats is the taxonomy's."""
        from bddl.object_taxonomy import ObjectTaxonomy
        from omnigibson.tiptop.articulation import openable_joints

        taxonomy = ObjectTaxonomy()
        boxes = self.boxes(near) if near else {}
        at = boxes[near]["center"][:2] if near in boxes else self.sim.base_pose()[0][:2].cpu().numpy()
        found = []
        for obj in self.sim.env.scene.objects:
            try:
                synset = taxonomy.get_synset_from_category(obj.category) if getattr(obj, "fixed_base", False) else None
            except ValueError:  # a category the taxonomy maps to more than one synset
                synset = None
            if synset is not None and taxonomy.has_ability(synset, ability):
                centre = obj.aabb_center.cpu().numpy()[:2]
                found.append((bool(openable_joints(obj)), float(np.linalg.norm(centre - at)), obj.name))
        if not found:
            log.info(f"no fixed {ability} in the scene")
            return None
        door, dist, name = min(found)
        self.sim.track(name)
        self.sim.bddl_names[name] = name
        log.info(f"{ability} for {near or 'the robot'}: {name}, {dist:.2f} m away{' (behind a door)' if door else ''}")
        return name

    def after_transition(self) -> list[str]:
        """The BDDL names a transition just created (the halves of a cut), now tracked; [] from a source that cannot
        tell (``KnowledgeSource.appeared``)."""
        return self.knowledge.appeared()

    def dwell(self, steps: int) -> int:
        """Hold still for ``steps`` env steps, or what the episode has left: the clock is the ingredient of a cook
        or a freeze (hot dogs 298-322 steps, spec S39). Returns the steps held."""
        left = self.sim.max_steps - self.sim.n_steps if getattr(self.sim, "max_steps", None) else int(steps)
        steps = max(0, min(int(steps), left))
        self.sim.video_caption = f"dwell {steps} steps"
        self.sim.hold(steps, self.sim.last_gripper)
        return steps

    def near_floor(self, name: str) -> bool | None:
        """Whether a target stands on the floor (its bottom within ``FLOOR_LEVEL`` of z = 0), or is the floor;
        None when it has never been perceived."""
        if self.is_floor(name):
            return True
        boxes = self.boxes(name)
        return on_the_floor(boxes[name]) if name in boxes else None

    def reaches_floor(self, *names: str) -> bool:
        """Whether the planner's workspace must reach the floor for a round on these targets: yes when one stands
        on the floor, and yes when one has never been perceived. Where an unseen object stands is unknown, and
        the round's capture is the look that finds out -- the detector searches only the workspace's projection,
        so the tabletop crop would never even show it a can on the floor. The widest look is the only one that
        finds it wherever it is; once seen, later rounds use its real answer."""
        verdicts = {name: self.near_floor(name) for name in names}
        unseen = [name for name, v in verdicts.items() if v is None]
        if unseen:
            log.info(f"{', '.join(unseen)} never perceived: the workspace reaches the floor so the look covers it")
        return any(v is None or v for v in verdicts.values())

    def edge_gap(self, item: str, support: str | None) -> float:
        """How far the item's centre is from the nearest edge of the support's footprint (small: reachable);
        infinite when the support is unknown (None) or either has never been perceived."""
        if support is None:
            return float("inf")
        if self.is_floor(support):
            return 0.0
        boxes = self.boxes(item, support)
        if item not in boxes or support not in boxes:
            return float("inf")
        return box_edge_gap(boxes[item], boxes[support])


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(p)
    add_planner_args(p)
    p.add_argument("--task-name", required=True, help="challenge task, e.g. turning_on_radio")
    p.add_argument(
        "--instances",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="indices into the task's test instance split (0-9 are the ones the challenge reports)",
    )
    p.add_argument("--mode", choices=("train", "public_test", "hidden_test"), default="public_test")
    p.add_argument(
        "--max-steps", type=int, default=None, help="episode timeout in env steps (default: the challenge's)"
    )
    p.add_argument(
        "--attempts-per-item",
        type=int,
        default=None,
        help="transfers an item gets before it is left (default: the task's)",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="planning rounds a goal gets before the strategy moves on (a pick's rounds each start from a fresh "
        "base pose): the one retry policy, the same for every task",
    )
    p.add_argument(
        "--obstacles",
        action="store_true",
        help="tell the planner about the furniture standing near the robot (held_labels -> cuTAMP statics), so it "
        "plans around the room instead of through it; costs a mask per obstacle in every capture",
    )
    p.add_argument(
        "--summarize",
        action="store_true",
        help="only rewrite summary.json from the result JSONs already in --out-dir/json (no simulation)",
    )
    args = p.parse_args(argv)
    # what run.py's helpers read: the challenge robot, the activity to load, the instruction the planner is given
    args.embodiment = "r1pro"
    args.activity = args.task_name
    spec = STRATEGIES.get(args.task_name)
    args.task = spec.instruction if spec is not None else args.task_name.replace("_", " ")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    out_dir = Path(args.out_dir)
    json_dir, video_dir = out_dir / "json", out_dir / "videos"
    json_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    import omnigibson as og
    from omnigibson.eval.evaluator import load_task_instance, resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import EVAL_TIMEOUT_MULTIPLIER
    from omnigibson.eval.utils.score_utils import load_human_stats
    from omnigibson.metrics import AgentMetric, TaskMetric
    from b1k.bridge.executor import VideoRecorder
    from omnigibson.tiptop.knowledge import make_knowledge
    from omnigibson.tiptop.scene import EpisodeOver
    from b1k.bridge.strategies import strategy_for, task_goal_atoms, task_goal_options

    human = load_human_stats(args.task_name)
    max_steps = args.max_steps or int(human["length"] * EVAL_TIMEOUT_MULTIPLIER)
    if args.summarize:
        results = [json.load(open(path)) for path in sorted(json_dir.glob(f"{args.task_name}_*_*.json"))]
        write_summary(out_dir, args, results, max_steps)
        return
    instance_ids = resolve_instance_ids(args.task_name, args.instances, mode=args.mode)
    log.info(f"{args.task_name}: {args.mode} instances {instance_ids}, timeout {max_steps} env steps")

    exit_code, stream, results = 0, None, []
    try:
        client, metadata, press_client, press_meta = connect_planners(args)
        planners = {"left": (client, metadata)}
        if press_client is not None:
            planners["right"] = (press_client, press_meta)
        sim = build_r1pro_sim(args, metadata["embodiment"], max_steps=max_steps)
        if not args.no_state_stream:
            stream = open_state_stream(f"{args.host}:{args.port}", sim)
        strategy = strategy_for(
            args.task_name,
            task_goal_atoms(sim),
            options=task_goal_options(sim),
            attempts=args.attempts_per_item,
            scope=sorted(sim.task_scope()),  # the objects that exist at reset: a cut's halves are not among them
        )
        for index, instance_id in zip(args.instances, instance_ids):
            t0 = time.time()
            name = f"{args.task_name}_{instance_id}_0"
            inst_dir = out_dir / name
            inst_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"===== instance {instance_id} (index {index}) =====")
            sim.env.reset()
            load_task_instance(sim.env, sim.robot, instance_id, mode=args.mode)
            sim.env.reset()  # episode_steps = 0, as the evaluator does before a rollout
            sim.reset_embodiment(metadata["embodiment"])
            knowledge = make_knowledge(args.knowledge, sim, strategy.goal, spec=strategy.spec)
            metrics = [AgentMetric(human), TaskMetric(human)]
            # from here on every step counts
            sim.begin_episode(metrics, stop_when_done=True, max_steps=max_steps, name=name)
            video = None if args.no_video else VideoRecorder(video_dir / f"{name}.mp4")
            if video is not None:
                sim.recorders.append(video)
            episode, reason = None, None
            try:
                sim.video_caption = f"{args.task_name} instance {instance_id}"
                # A task whose press is "hold" uses BOTH planners, and the right-arm one plans its seven arm
                # joints with the torso LOCKED at the embodiment's home posture. --torso moves the torso away
                # from there, so every press round dies before it plans: "r1pro_right locks torso_joint3 at
                # -0.470 rad but the simulator has it at -0.900". turning_on_radio is the only task in the set
                # that presses this way, and it scores 1.0 with the postures agreeing against 0.0 without
                # (2026-09-15) -- the lean is worth nothing if the hand that presses can never be planned.
                # ``press`` DEFAULTS to "hold" in TaskSpec, so testing it alone fired this guard on all 38 tasks
                # and silently ran every pure-transfer task without the lean for two hours. ``plan`` is what says
                # a task presses at all: only "press" and "auto" ever ask for the right-arm planner, which is
                # five tasks. Found by a review agent reading the code, not by a score -- the test that was
                # supposed to pin this read the task YAML files, where only 5 of 38 set ``press`` at all, so it
                # passed while the runtime did the opposite (2026-09-15).
                posture_args = args
                if wants_home_torso(getattr(strategy, "spec", None)) and getattr(args, "torso", None):
                    posture_args = copy.copy(args)
                    posture_args.torso = None
                    log.info(
                        f"{args.task_name} presses with the other hand, whose planner locks the torso at its home "
                        f"posture; ignoring --torso {list(args.torso)} so both planners agree"
                    )
                apply_embodiment_posture(sim, posture_args, metadata["embodiment"])
                sim.mark_goal_initial()
                episode = Episode(sim, args, planners, knowledge, inst_dir, spec=getattr(strategy, "spec", None))
                strategy.run(episode)
                reason = "strategy finished"
            except EpisodeOver as e:
                reason = e.reason
            except TransferBlocked as e:
                reason = f"blocked: {e}"
                log.warning(reason)
                if episode is not None:
                    episode.records.append(
                        {"blocked_transfer": str(e), "step": sim.n_steps, "hands": dict(sim.hands())}
                    )
            except Exception as e:  # noqa: BLE001 - score what happened and go on to the next instance
                log.exception(f"instance {instance_id} crashed")
                reason = f"crash: {type(e).__name__}: {e}"
            steps = sim.end_episode()  # scored on the state now; the video tail below counts for nothing
            success = bool(sim.env.task.success)
            goal = sim.goal_status()
            aggregated = {}
            for metric in metrics:
                aggregated.update(metric.aggregate(sim.env))
            if video is not None:
                sim.video_caption = verdict_caption(reason, success, goal)
                try:
                    sim.hold(EPILOGUE_STEPS, sim.last_gripper)  # the final state and the verdict stay on screen
                finally:
                    sim.recorders.remove(video)
                    video.close()
            result = {
                "task": args.task_name,
                "instance_id": int(instance_id),
                "rollout_id": 0,
                "steps": steps,
                "success": success,
                **aggregated,
                "bench": {
                    "reason": reason,
                    "max_steps": max_steps,
                    "wall_time_s": round(time.time() - t0, 1),
                    "knowledge": knowledge.report(),
                    "collision_map": "simulator_physical" if sim.send_room else None,
                    "teleports": sim.teleports,
                    "goal": goal,
                    "video": None if video is None else str(Path(video.path).relative_to(out_dir)),
                    "rounds": episode.records if episode is not None else [],
                },
            }
            with open(json_dir / f"{name}.json", "w") as f:
                json.dump(result, f, indent=2, default=float)
            results.append(result)
            log.info(
                f"RESULT instance {instance_id}: q_score {result.get('q_score', {}).get('final')} success {success} "
                f"steps {steps}/{max_steps} ({reason}); teleports {sim.teleports}; {result['bench']['wall_time_s']}s"
                + (f"; {what_failed(result)}" if not success else "")
            )
            write_summary(out_dir, args, results, max_steps)
    except Exception:
        log.exception("benchmark failed")
        exit_code = 1
    finally:
        if stream is not None:
            stream.close()
        if og.app is not None:
            og.shutdown()
    sys.exit(exit_code)


def write_summary(out_dir: Path, args, results: list[dict], max_steps: int) -> dict:
    """summary.json: the mean q_score over the instances run (the challenge's per-task number) and one line each."""
    scores = [float(r.get("q_score", {}).get("final", 0.0)) for r in results]
    summary = {
        "task": args.task_name,
        "mode": args.mode,
        "knowledge": args.knowledge,
        "collision_map": "simulator_physical" if getattr(args, "room", False) else None,
        "grasping_mode": args.grasping_mode,
        "rounds": args.rounds,
        "max_steps": max_steps,
        "instances": len(results),
        "mean_q_score": float(np.mean(scores)) if scores else None,
        "successes": int(sum(r["success"] for r in results)),
        "per_instance": [
            {
                "instance_id": r["instance_id"],
                "q_score": r.get("q_score", {}).get("final"),
                "success": r["success"],
                "steps": r["steps"],
                "reason": r["bench"]["reason"],
                "teleports": r["bench"]["teleports"],
                "wall_time_s": r["bench"]["wall_time_s"],
                "what_failed": what_failed(r),
            }
            for r in results
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log.info(f"SUMMARY {args.task_name}: mean q_score {summary['mean_q_score']} over {len(results)} instances")
    return summary


if __name__ == "__main__":
    main()
