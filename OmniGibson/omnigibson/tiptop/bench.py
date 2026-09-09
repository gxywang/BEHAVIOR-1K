"""Benchmark the pipeline on a challenge task the way the 2026 BEHAVIOR Challenge evaluates a policy.

Same task instances (the public test split, indices 0-9 for reported results), same per-instance timeout (1.5x
the mean human demonstration length, in env steps), same metrics (``TaskMetric``: 1 on success, else the newly
satisfied fraction of the best goal option; ``AgentMetric``: base and end-effector displacement), the same result
JSON per rollout as ``omnigibson.eval.eval``. What differs, and is written into every result: the robot is driven
in-process by a task strategy (strategies.py) that teleports the base instead of navigating, and the planner may
be told what the simulator knows (``--knowledge oracle``: masks and button poses). Both are stand-ins for parts of
the pipeline that do not exist yet, so a number from this benchmark is an upper bound for the manipulation part,
not a challenge score.

  OMNIGIBSON_HEADLESS=1 python -m omnigibson.tiptop.bench --task-name turning_on_radio --instances 0 1 2 \\
      --host localhost --port 8765 --knowledge oracle --grasping-mode sticky --out-dir runs/bench_radio
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from omnigibson.tiptop.strategies import Unreachable
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


class Episode:
    """One task instance as a strategy sees it. The base moves by teleport (``stand_for``); the planner of an arm
    plans one round at a time (``plan_and_execute``, which never raises on a failed round: the strategy decides
    what to do next); and the simulator answers what the pipeline cannot perceive yet (``holds``, ``holding``,
    distances): privileged, and counted in the result."""

    def __init__(self, sim, args, planners: dict, knowledge, out_dir: Path):
        self.sim, self.args, self.planners, self.knowledge, self.out_dir = sim, args, planners, knowledge, out_dir
        self.records = []  # one per round, in order
        self.stood = {}  # names -> (x, y) poses stood at for them, so a retry gets a different viewpoint
        self.floor = next((n for n in sim.task_scope() if n.startswith("floor.")), "floor.n.01_1")

    # ---------------------------------------------------------------- moving
    def stand_for(self, *names: str) -> dict:
        """Teleport the base to a pose from which the named objects are in the left arm's reach and in view. A
        second call for the same objects stands somewhere else; when nothing is found within the arm's usual
        reach, the search is widened to ``REACH_FAR`` (the torso leans that far) before giving up."""
        avoid = self.stood.setdefault(names, [])
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
        self.records.append({"stand_for": list(names), "pose": pose, "step": self.sim.n_steps})
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
        self.sim.video_caption = f"{atom_text(atoms)} [{arm} arm]"
        record = {"round": i, "atoms": atoms, "arm": arm, "dir": str(round_dir), "step": self.sim.n_steps}
        t0 = time.time()
        try:
            self.use_arm(arm)
            client = self.planners[arm][0]
            client.wait_for_server(timeout_s=300.0)  # a planner relaunched after a CUDA fault comes back in ~1 min
            result = live_round(self.sim, self.args, client, round_dir, atoms, self.knowledge, floor=floor, score=False)
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

    # ---------------------------------------------------------------- what the simulator knows (privileged)
    def holds(self, predicate: str, *bddl_names: str) -> bool:
        return self.sim.holds(predicate, *bddl_names)

    def holding(self, bddl: str) -> bool:
        """Whether a hand holds the object (the robot's own knowledge: its grasp assist, else the plans' record)."""
        return self.sim.tracked_label(bddl) in self.sim.hands()

    def position(self, bddl: str) -> np.ndarray:
        return self.sim.scene_object(bddl).aabb_center.cpu().numpy()

    def distance(self, a: str, b: str) -> float:
        return float(np.linalg.norm(self.position(a)[:2] - self.position(b)[:2]))

    def support_of(self, bddl: str) -> str:
        """The BDDL name of the table the object rests on: the task's ontop predicate, else the table whose
        footprint holds the object's centre with its top just under the object (the predicate misses a radio on a
        glass table), else the task's floor."""
        tables = [n for n in self.sim.task_scope() if n.split(".n.")[0] == "table"]
        for name in tables:
            if self.sim.holds("ontop", bddl, name):
                return name
        lo_obj = float(self.sim.scene_object(bddl).aabb[0][2])
        c = self.position(bddl)
        for name in tables:
            lo, hi = [v.cpu().numpy() for v in self.sim.scene_object(name).aabb]
            if lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1] and -0.02 <= lo_obj - hi[2] <= 0.10:
                return name
        return self.floor

    def edge_gap(self, item: str, support: str) -> float:
        """How far the item's centre is from the nearest edge of the support's footprint (small: reachable)."""
        lo, hi = [v.cpu().numpy() for v in self.sim.scene_object(support).aabb]
        c = self.position(item)
        return float(min(c[0] - lo[0], hi[0] - c[0], c[1] - lo[1], hi[1] - c[1]))


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
    p.add_argument("--attempts-per-item", type=int, default=2, help="items of a kind tried per basket")
    p.add_argument(
        "--summarize",
        action="store_true",
        help="only rewrite summary.json from the result JSONs already in --out-dir/json (no simulation)",
    )
    args = p.parse_args(argv)
    args.embodiment = "r1pro"
    args.activity = args.task_name
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
    from omnigibson.tiptop.strategies import strategy_for, task_goal_atoms

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
        strategy = strategy_for(args.task_name, task_goal_atoms(sim), **strategy_kwargs(args))
        args.task = strategy.instruction
        for index, instance_id in zip(args.instances, instance_ids):
            t0 = time.time()
            name = f"{args.task_name}_{instance_id}_0"
            inst_dir = out_dir / name
            inst_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"===== instance {instance_id} (index {index}) =====")
            sim.env.reset()
            load_task_instance(sim.env, sim.robot, instance_id, mode=args.mode)
            sim.env.reset()  # episode_steps = 0, as the evaluator does before a rollout
            sim.held_objects, sim.teleports = {}, 0
            sim.reset_embodiment(metadata["embodiment"])
            knowledge = make_knowledge(args.knowledge, sim, strategy.goal)
            metrics = [AgentMetric(human), TaskMetric(human)]
            sim.begin_episode(metrics, stop_when_done=True)  # from here on every env step counts
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
            finally:
                if video is not None:
                    sim.recorders.remove(video)
                    video.close()
            sim.stop_when_done = False
            success = bool(sim.env.task.success)
            aggregated = {}
            for metric in metrics:
                aggregated.update(metric.aggregate(sim.env))
            result = {
                "task": args.task_name,
                "instance_id": int(instance_id),
                "rollout_id": 0,
                "steps": sim.n_steps,
                "success": success,
                **aggregated,
                "bench": {
                    "reason": reason,
                    "max_steps": max_steps,
                    "wall_time_s": round(time.time() - t0, 1),
                    "knowledge": knowledge.report(),
                    "teleports": sim.teleports,
                    "goal": sim.goal_status(),
                    "rounds": episode.records if episode is not None else [],
                },
            }
            with open(json_dir / f"{name}.json", "w") as f:
                json.dump(result, f, indent=2, default=float)
            results.append(result)
            log.info(
                f"RESULT instance {instance_id}: q_score {result.get('q_score', {}).get('final')} success {success} "
                f"steps {sim.n_steps}/{max_steps} ({reason}); teleports {sim.teleports}; {result['bench']['wall_time_s']}s"
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


def strategy_kwargs(args) -> dict:
    return {"attempts": args.attempts_per_item} if args.task_name == "assembling_gift_baskets" else {}


def write_summary(out_dir: Path, args, results: list[dict], max_steps: int) -> dict:
    """summary.json: the mean q_score over the instances run (the challenge's per-task number) and one line each."""
    scores = [float(r.get("q_score", {}).get("final", 0.0)) for r in results]
    summary = {
        "task": args.task_name,
        "mode": args.mode,
        "knowledge": args.knowledge,
        "grasping_mode": args.grasping_mode,
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
