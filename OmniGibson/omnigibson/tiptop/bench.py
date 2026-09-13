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
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from omnigibson.tiptop.protocol import bddl_category
from omnigibson.tiptop.strategies import PLACE_PREDICATES, STRATEGIES, Unreachable, atom
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
STANCE_ATTEMPTS = 3  # stances tried before a round works from one that did not settle level
EPILOGUE_STEPS = 90  # env steps the final state and the verdict stay on screen after the episode (3 s of video)
UNSATISFIED_SHOWN = 3  # goal atoms listed in the verdict; the gift-basket goal has 16
FLOOR_LEVEL = 0.15  # m: a target whose bottom is lower than this stands on the floor (the workspace reaches down)


def short_atom(atom: str) -> str:
    """'inside(bow.n.08_4, wicker_basket.n.01_3)' -> 'inside(bow_4, wicker_basket_3)'."""
    return re.sub(r"\.n\.\d+", "", atom)


def what_failed(result: dict) -> str:
    """One line on why an instance fell short, from its own records: the goal atoms left unsatisfied, the objects no
    base pose reached, the rounds that failed by kind, the rounds that ran (by predicate), releases. Empty on success."""
    if result["success"]:
        return ""
    bench, goal = result["bench"], result["bench"]["goal"]
    rounds = bench["rounds"]
    missing = goal["unsatisfied"]
    parts = [
        f"{len(missing)}/{goal['total']} unsatisfied: "
        + ", ".join(short_atom(a) for a in missing[:UNSATISFIED_SHOWN])
        + (" ..." if len(missing) > UNSATISFIED_SHOWN else "")
    ]
    if bench["reason"] not in ("strategy finished", "success"):
        parts.append(bench["reason"])
    unreachable = Counter(
        ", ".join(short_atom(n) for n in x["stand_for"]) for x in rounds if "stand_for" in x and "error" in x
    )
    if unreachable:
        parts.append("no base pose for " + ", ".join(f"{name} x{n}" for name, n in unreachable.items()))
    failed = Counter(
        x["error"].split(":")[0] for x in rounds if "round" in x and "error" in x and x["error"] != "episode over"
    )
    if failed:
        parts.append("failed rounds: " + ", ".join(f"{kind} x{n}" for kind, n in failed.items()))
    ran = Counter(f"{x['atoms'][0]['predicate']} [{x['arm']}]" for x in rounds if "round" in x and "error" not in x)
    parts.append("rounds run: " + (", ".join(f"{k} x{n}" for k, n in ran.items()) if ran else "none"))
    releases = sum(1 for x in rounds if x.get("release"))
    if releases:
        parts.append(f"released an item x{releases}")
    return "; ".join(parts)


def verdict_caption(reason: str, success: bool, goal: dict) -> str:
    """What the video's tail says (``goal`` as ``goal_status`` reports it): outcome, score, satisfied count, then the
    first unsatisfied atoms, so success and failure can be told apart on screen."""
    head = "RESULT: SUCCESS" if success else f"RESULT: FAILED ({reason})"
    head += f"  q_score {goal['q_score']:.3g}  {len(goal['satisfied'])}/{goal['total']} satisfied"
    missing = goal["unsatisfied"]
    if not missing:
        return head
    more = f" +{len(missing) - UNSATISFIED_SHOWN} more" if len(missing) > UNSATISFIED_SHOWN else ""
    return f"{head}\nunsatisfied: {', '.join(missing[:UNSATISFIED_SHOWN])}{more}"


class Episode:
    """One task instance as a strategy sees it. The base moves by teleport (``stand_for``); the planner of an arm
    plans one round at a time (``plan_and_execute``, which never raises on a failed round: the runner decides what
    to do next); and judges each round without asking the simulator whether it worked: ``holding`` is the robot's
    own hand record (a plan closed the hand and the fingers stopped on something), a placement is a geometric test
    on where the knowledge source localizes the objects (``placed``: the item's box over the target's), a press
    counts once its planned stroke ran. Positions, distances and supports (``on_support``, ``support_of``,
    ``edge_gap``) come from the same localization, which the oracle source reads from the simulator and the onboard
    source from the planner's reports. ``pick``, ``achieve`` and ``put_down`` run the rounds under the one retry
    policy every task gets (``--rounds``); there is no other recovery, in here or in a task description."""

    def __init__(self, sim, args, planners: dict, knowledge, out_dir: Path):
        self.sim, self.args, self.planners, self.knowledge, self.out_dir = sim, args, planners, knowledge, out_dir
        self.rounds = args.rounds
        self.records = []  # one per round, in order
        self.stood = {}  # names -> (x, y) poses stood at for them, so a retry gets a different viewpoint
        self.floor = sim.floor_name()

    # ---------------------------------------------------------------- moving
    def open_up(self, name: str, fraction: float | None = None) -> bool:
        """Stand at ``name`` and open it, by ``fraction`` of its joint's range (the scored atom's worth by default).

        Reading the joint back afterwards is privileged, the way the oracle's masks are: the motion reports what
        the joint says, and at evaluation that verdict would have to come from the hand's own travel and a fresh
        look at the container.
        """
        from omnigibson.tiptop.articulation import OPEN_FRACTION_SCORED

        try:
            self.stand_for(name)
        except Unreachable as e:
            log.info(f"{name}: {e}; cannot reach it to open it")
            self.records.append({"open": name, "error": str(e), "step": self.sim.n_steps})
            return False
        self.sim.video_caption = f"open {name}"
        result = self.sim.open_container(
            self.sim.arm, name, fraction=OPEN_FRACTION_SCORED if fraction is None else fraction
        )
        self.records.append({"open": name, **result, "step": self.sim.n_steps})
        if not result.get("opened"):
            log.info(f"{name} did not open: {result.get('why') or 'the joint did not move'}")
        return bool(result.get("opened"))

    def openable(self, name: str) -> bool:
        """Whether ``name`` has a joint that opens at all (a bin and a basket do not)."""
        from omnigibson.tiptop.articulation import openable_joints

        try:
            return bool(openable_joints(self.sim.scene_object(name)))
        except Exception:
            return False

    def is_shut(self, name: str) -> bool:
        """Whether every joint of ``name`` that could open is closed."""
        from omnigibson.tiptop.articulation import is_open, openable_joints

        try:
            joints = openable_joints(self.sim.scene_object(name))
        except Exception:
            return False
        return bool(joints) and not any(is_open(j["lower"], j["upper"], j["position"]) for j in joints)

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
        self.sim.video_caption = f"teleport: stand for {', '.join(names)}"
        for attempt in range(STANCE_ATTEMPTS):
            try:
                pose = self.sim.place_robot_for(*names, avoid=avoid)
            except RuntimeError as e:
                log.info(f"{e}; widening the search to {REACH_FAR} m")
                try:
                    pose = self.sim.place_robot_for(*names, reach=REACH_FAR, avoid=avoid)
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
                return pose
            log.warning(
                f"{why} at ({pose['x']:.2f}, {pose['y']:.2f}): the pose is occupied by something the footprint "
                f"test missed"
                + (
                    f"; standing somewhere else ({attempt + 1}/{STANCE_ATTEMPTS})"
                    if attempt + 1 < STANCE_ATTEMPTS
                    else "; out of attempts, working from here"
                )
            )
        return pose

    def has_arm(self, arm: str) -> bool:
        return arm in self.planners

    def use_arm(self, arm: str) -> None:
        if arm != self.sim.arm:
            self.sim.adopt_embodiment(self.planners[arm][1]["embodiment"])

    # ---------------------------------------------------------------- planning rounds
    def plan_and_execute(self, atoms: list[dict], arm: str = "left", floor: bool = False) -> dict:
        """One capture / plan / execute round for ``atoms`` with the planner of ``arm``. A planner failure, an
        object out of view or an execution error is recorded and returned as {"error": ...}; the episode's end
        (``EpisodeOver``) propagates."""
        from omnigibson.tiptop.scene import EpisodeOver

        i = len(self.records)
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
        except EpisodeOver:
            record["error"] = "episode over"
            record["seconds"] = round(time.time() - t0, 1)
            self.records.append(record)
            raise
        except Exception as e:  # noqa: BLE001 - one failed round must not end the instance
            log.exception(f"round {i} {atom_text(atoms)} failed")
            record["error"] = f"{type(e).__name__}: {e}"
        record["seconds"] = round(time.time() - t0, 1)
        self.records.append(record)
        log.info(f"round {i} {atom_text(atoms)} [{arm}]: {record.get('error') or 'executed'} ({record['seconds']}s)")
        return record

    # ---------------------------------------------------------------- the retry policy, the same for every task
    def satisfied(self, atoms: list[dict], record: dict | None = None) -> bool:
        """Whether every atom holds, judged from the robot's own readings and localization: ``holding`` by the hand
        record, a placement by ``placed``, ``nextto`` by ``beside``, and a press by its round having run without
        error (the switch's state is the simulator's to know, so a press is open loop).

        A predicate the runner does not know is NOT taken to hold because a round ran. That is what this did, and
        it would score every new predicate satisfied the moment a round was attempted -- the first task to name one
        would be reported as solved without anything having been achieved.
        """
        ran = record is not None and not record.get("error")
        for a in atoms:
            predicate, args = a["predicate"], a["args"]
            if predicate == "holding":
                ok = self.holding(args[0])
            elif predicate == "nextto" and len(args) == 2:
                ok = self.beside(args[0], args[1])
            elif predicate in PLACE_PREDICATES and len(args) == 2:
                ok = self.placed(args[0], args[1])
            elif predicate == "toggled_on" or (predicate == "not" and "toggled_on" in args):
                ok = ran  # a press is open loop: it ran, and the switch's state is the simulator's to know
            else:
                log.warning(f"no test for {predicate}({', '.join(args)}); the round counts as unfinished")
                ok = False
            if not ok:
                return False
        return True

    def beside(self, item: str, other: str) -> bool:
        """OmniGibson's own NextTo measure, on the boxes the knowledge source localizes.

        ``object_states/next_to.py``: the per-axis gap between the two boxes, as a norm, within a sixth of the mean
        of their extents. The simulator's version also asks for horizontal adjacency (a raycast test that the thing
        beside you is not behind something else); this half is the geometry, and the adjacency half is not
        available without the simulator, so a placement that satisfies this can still fail the evaluator's test.
        """
        boxes = self.boxes(item, other)
        a, b = boxes[item], boxes[other]
        gap = np.array([max(0.0, max(a["lo"][d], b["lo"][d]) - min(a["hi"][d], b["hi"][d])) for d in range(3)])
        extents = (np.asarray(a["hi"]) - np.asarray(a["lo"])) + (np.asarray(b["hi"]) - np.asarray(b["lo"]))
        return bool(np.linalg.norm(gap) <= float(np.mean(extents)) / 6.0)

    def achieve(self, atoms: list[dict], arm: str = "left", floor: bool | None = None, done=None) -> bool:
        """Up to ``--rounds`` planning rounds for ``atoms`` with the planner of ``arm``, stopping as soon as
        ``done()`` (default: ``satisfied``); the retry every goal of every task gets, and the only one. ``floor``
        (the planner's workspace reaches the floor) is read off the target when not given: a container or support
        that stands on the floor."""
        if floor is None:
            floor = any(self.near_floor(a["args"][1]) for a in atoms if len(a["args"]) == 2)
        for _ in range(self.rounds):
            record = self.plan_and_execute(atoms, arm=arm, floor=floor)
            if done() if done is not None else self.satisfied(atoms, record):
                return True
        return False

    def pick(self, bddl: str) -> bool:
        """The object in the planned hand after up to ``--rounds`` pick rounds, each from a fresh base pose (a pick
        that fails, no plan or the object hidden, is retried from somewhere else). False when no pose reaches it."""
        for _ in range(self.rounds):
            try:
                self.stand_for(bddl)
            except Unreachable as e:
                log.warning(f"{bddl}: {e}")
                return False
            # where the item is *now*: one that was knocked to the floor needs the workspace to reach down to it
            self.plan_and_execute([atom("holding", bddl)], floor=self.near_floor(bddl))
            if self.holding(bddl):
                return True
        log.warning(f"{bddl}: not in the hand after {self.rounds} pick rounds")
        return False

    def put_down(self, bddl: str, support: str, floor: bool | None = None) -> bool:
        """Put the held object on ``support``; done when the hand is empty, wherever the object landed (the point
        is a free hand)."""
        return self.achieve([atom("ontop", bddl, support)], floor=floor, done=lambda: not self.holding(bddl))

    # ---------------------------------------------------------------- the robot's own record
    def holding(self, bddl: str) -> bool:
        """Whether a hand holds the object, by the robot's own record (a plan closed the hand on it and the
        fingers stopped on something; see ``run.note_hands``)."""
        return self.sim.tracked_label(bddl) in self.sim.hands()

    def held_names(self) -> list[str]:
        """BDDL names of the task objects in the hands (the robot's own record)."""
        return [self.sim.bddl_names[label] for label in self.sim.hands() if label in self.sim.bddl_names]

    def release(self, steps: int = 45) -> None:
        """Open the planned hand where it is and let whatever it holds fall (the last resort when no put-down
        plan exists); the hand is then held open for ``steps`` env steps so the object clears it, and the record
        of that hand is cleared."""
        self.sim.video_caption = f"release [{self.sim.arm} arm]"
        self.sim.hold(steps, self.sim.OPEN)
        for label, arm in list(self.sim.hands().items()):
            if arm == self.sim.arm:
                self.sim.held_objects.pop(label, None)
        self.records.append({"release": True, "step": self.sim.n_steps, "hands": dict(self.sim.hands())})

    # ---------------------------------------------------------------- localization (the knowledge source's)
    def boxes(self, *bddl_names: str) -> dict:
        """name -> {center, lo, hi} (world frame) from the knowledge source; the floor has no box."""
        return self.knowledge.localize(*[n for n in bddl_names if n != self.floor])

    def position(self, bddl: str) -> np.ndarray:
        return self.boxes(bddl)[bddl]["center"]

    def distance(self, a: str, b: str) -> float:
        boxes = self.boxes(a, b)
        return float(np.linalg.norm(boxes[a]["center"][:2] - boxes[b]["center"][:2]))

    def on_support(self, bddl: str, support: str) -> bool:
        """Whether the object stands on the support, by geometry: its centre inside the support's footprint and
        its bottom within 15 cm above the top."""
        if support == self.floor:
            return True
        boxes = self.boxes(bddl, support)
        return placed_over(boxes[bddl], boxes[support], from_bottom=False)

    def placed(self, item: str, target: str) -> bool:
        """Whether the item ended on or in the target, by geometry: its centre inside the target's footprint and
        its bottom anywhere from 2 cm under the target's bottom (inside a container) to 15 cm above its top (on a
        surface). Onto the floor: the hand let go of it."""
        if target == self.floor:
            return not self.holding(item)
        boxes = self.boxes(item, target)
        return placed_over(boxes[item], boxes[target], from_bottom=True)

    def support_of(self, bddl: str) -> str:
        """The BDDL name of the task object the item stands on (the highest one whose footprint holds it, any
        category), else the task's floor."""
        names = [n for n in self.sim.task_scope() if n not in (bddl, self.floor) and bddl_category(n) != "agent"]
        boxes = self.boxes(bddl, *names)
        under = [n for n in names if placed_over(boxes[bddl], boxes[n], from_bottom=False)]
        if not under:
            return self.floor
        return max(under, key=lambda n: float(boxes[n]["hi"][2]))

    def near_floor(self, name: str) -> bool:
        """Whether a target stands on the floor (its bottom within ``FLOOR_LEVEL`` of z = 0), or is the floor."""
        if name == self.floor:
            return True
        return float(self.boxes(name)[name]["lo"][2]) < FLOOR_LEVEL

    def edge_gap(self, item: str, support: str) -> float:
        """How far the item's centre is from the nearest edge of the support's footprint (small: reachable)."""
        if support == self.floor:
            return 0.0
        boxes = self.boxes(item, support)
        lo, hi, c = boxes[support]["lo"], boxes[support]["hi"], boxes[item]["center"]
        return float(min(c[0] - lo[0], hi[0] - c[0], c[1] - lo[1], hi[1] - c[1]))


def placed_over(item: dict, target: dict, from_bottom: bool) -> bool:
    """Geometric "on" (``from_bottom`` False: the item's bottom within -2 cm .. +15 cm of the target's top) or
    "on or in" (True: from 2 cm under the target's bottom to 15 cm over its top), with the item's centre inside
    the target's footprint. Boxes are {center, lo, hi}."""
    c, bottom = item["center"], float(item["lo"][2])
    lo, hi = target["lo"], target["hi"]
    if not (lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1]):
        return False
    low = float(lo[2]) - 0.02 if from_bottom else float(hi[2]) - 0.02
    return low <= bottom <= float(hi[2]) + 0.15


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
        "--summarize",
        action="store_true",
        help="only rewrite summary.json from the result JSONs already in --out-dir/json (no simulation)",
    )
    args = p.parse_args(argv)
    if args.task_name not in STRATEGIES:
        p.error(f"no task description for {args.task_name!r}; known: {sorted(STRATEGIES)}")
    # what run.py's helpers read: the challenge robot, the activity to load, the instruction the planner is given
    args.embodiment = "r1pro"
    args.activity = args.task_name
    args.task = STRATEGIES[args.task_name].instruction
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
    from omnigibson.tiptop.executor import VideoRecorder
    from omnigibson.tiptop.knowledge import make_knowledge
    from omnigibson.tiptop.scene import EpisodeOver
    from omnigibson.tiptop.strategies import strategy_for, task_goal_atoms, task_goal_options

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
            args.task_name, task_goal_atoms(sim), options=task_goal_options(sim), attempts=args.attempts_per_item
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
            knowledge = make_knowledge(args.knowledge, sim, strategy.goal)
            metrics = [AgentMetric(human), TaskMetric(human)]
            sim.begin_episode(metrics, stop_when_done=True, max_steps=max_steps)  # from here on every step counts
            video = None if args.no_video else VideoRecorder(video_dir / f"{name}.mp4")
            if video is not None:
                sim.recorders.append(video)
            episode, reason = None, None
            try:
                sim.video_caption = f"{args.task_name} instance {instance_id}"
                apply_embodiment_posture(sim, args, metadata["embodiment"])
                sim.mark_goal_initial()
                episode = Episode(sim, args, planners, knowledge, inst_dir)
                strategy.run(episode)
                reason = "strategy finished"
            except EpisodeOver as e:
                reason = e.reason
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
