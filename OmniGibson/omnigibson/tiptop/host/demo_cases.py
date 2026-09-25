"""Skill-bench cases from the B1K human teleop demos (SPEC §9): start one skill from a real human's situation.

The demos are noisy (needless hand-overs, staging), so they give SITUATIONS -- where the robot stood, which object,
which part, the scene state -- never behaviour to imitate.

Restore method (checked in the sim on GPU 3, 2026-09-25):
- an episode starts from TRAINING instance ``task_instance_id`` of its task (episode meta; also
  ``raw_episode_id % 10000 // 10``): ``load_task_instance(env, robot, id, mode="train")`` puts the robot at the
  instance's R1Pro pose with the reset joints, which is the recorded frame 0 (joints within 0.01 rad);
- ``observation.state`` (61) is ``b1k.observation.PROPRIO_SLICES``: base VELOCITY in the robot frame, no base pose;
- replaying the recorded 23-d actions from frame 0 through the evaluator's controllers (base velocity, absolute
  torso/arm joints, gripper -1..1) reproduces the demo: the rendered head depth matches the recorded frame to 1-2 mm
  median (98-99.8 % of pixels within 5 cm) and the human's grasps attach the same object. Dead-reckoning the
  recorded base velocity instead drifts 5-6 deg and up to 13 cm within 15 s (head depth 34-59 % within 5 cm), so a
  set-state restore is only a fallback, even for a first segment where nothing moved before.

Offline (numpy + pyarrow): ``build_catalog`` / ``load_catalog`` / ``pick_starter``. In the sim: ``make_env``,
``materialize``, ``restore``. ``--catalog`` builds catalog.parquet; ``--task T`` (one sim per task, GPU; ``--list``
names the tasks) materializes T's starter cases into cases/ + snapshots/; ``--collect`` merges them into one
``<group>.yaml`` list per group, skillbench's case-file layout.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from b1k.observation import PROPRIO_SLICES

DATASET = Path(os.environ.get("B1K_DEMOS", "/shared/perception/datasets/behavior1k-20k"))
OUT_DIR = Path("/home/wding8/projects/BEHAVIOR-1K/runs/skill_arch_20260925/demo_cases")
FPS = 30
DT = 1.0 / FPS

# human skill_description -> (our skill, relations for place / want_on for press / kind for open-close)
HUMAN_TO_SKILL = {
    "pick up from": ("pick_up", None),
    "place on": ("place", ("on",)),
    "place in": ("place", ("in",)),
    "place under": ("place", ("under",)),
    "place on next to": ("place", ("on", "next_to")),
    "place in next to": ("place", ("in", "next_to")),
    "open door": ("open", "door"),
    "open drawer": ("open", "drawer"),
    "close door": ("close", "door"),
    "close drawer": ("close", "drawer"),
    "press": ("press", None),
    "turn on switch": ("press", True),
    "turn off switch": ("press", False),
    "hold": ("hold", None),
    "push to": ("push", None),
}
# a segment of these does not change the scene (a "move to" inside manipulation_ranges is a carry)
NO_CHANGE = {"move to", "turn to"}
BDDL_PRED = {"on": "ontop", "in": "inside", "under": "under", "next_to": "nextto"}


# ---------------------------------------------------------------- offline: dataset access


@functools.lru_cache(maxsize=None)
def available_tasks() -> dict:
    import yaml

    return yaml.safe_load(open(instances_root() / "metadata" / "available_tasks.yaml"))


def instances_root() -> Path:
    from omnigibson.macros import gm  # cheap: macros only, the app is not launched

    return Path(gm.DATA_PATH) / "2026-challenge-task-instances"


@functools.lru_cache(maxsize=None)
def task_objects(task: str) -> dict:
    """sim object name -> (BDDL instance or None, category, fixed_base) from the task's scene template."""
    scene = available_tasks()[task][0]["scene_model"]
    tpl = json.load(open(instances_root() / "scenes" / scene / "json" / f"{scene}_task_{task}_0_0_template.json"))
    name_to_inst = {v: k for k, v in tpl["metadata"]["task"]["inst_to_name"].items()}
    out = {}
    for name, info in tpl["objects_info"]["init_info"].items():
        a = info["args"]
        out[name] = (name_to_inst.get(name), a.get("category", name.rsplit("_", 1)[0]), bool(a.get("fixed_base")))
    return out


@functools.lru_cache(maxsize=None)
def episodes():
    """The episode meta table (20,000 rows): data file, global index range, instance id."""
    import glob

    import pandas as pd
    import pyarrow.parquet as pq

    cols = ["episode_index", "data/chunk_index", "data/file_index", "raw_episode_id", "task_instance_id",
            "meta/episodes/chunk_index", "meta/episodes/file_index"]
    files = sorted(glob.glob(str(DATASET / "meta/episodes/*/*.parquet")))
    return pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files]).set_index("episode_index")


def episode_arrays(episode_index: int, stop: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(state [T, 61], action [T, 23]) of one episode, frames [0, stop). Reads two columns of one data file."""
    import pyarrow.parquet as pq

    m = episodes().loc[episode_index]
    path = DATASET / f"data/chunk-{m['data/chunk_index']:03d}/file-{m['data/file_index']:03d}.parquet"
    t = pq.read_table(path, columns=["frame_index", "observation.state", "action"],
                      filters=[("episode_index", "=", int(episode_index))])
    order = np.argsort(t.column("frame_index").to_numpy())[:stop]
    state = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))[order]
    action = np.stack(t.column("action").to_numpy(zero_copy_only=False))[order]
    return state.astype(np.float64), action.astype(np.float64)


def dead_reckon(state: np.ndarray, x0: float, y0: float, yaw0: float) -> np.ndarray:
    """[T, 3] world (x, y, yaw) at every frame, integrating the robot-frame base velocity (trapezoid, 1/30 s)."""
    v = state[:, PROPRIO_SLICES["base_qvel"]]
    vm = 0.5 * (v[1:] + v[:-1])
    yaw = yaw0 + np.concatenate([[0.0], np.cumsum(vm[:, 2] * DT)])
    ym = 0.5 * (yaw[1:] + yaw[:-1])
    dx = (np.cos(ym) * vm[:, 0] - np.sin(ym) * vm[:, 1]) * DT
    dy = (np.sin(ym) * vm[:, 0] + np.cos(ym) * vm[:, 1]) * DT
    return np.stack([x0 + np.concatenate([[0.0], np.cumsum(dx)]), y0 + np.concatenate([[0.0], np.cumsum(dy)]), yaw], 1)


def closing_arm(action: np.ndarray, start: int, end: int) -> str | None:
    """The arm whose gripper command goes from open to closed inside [start, end); the first to close wins."""
    first = {}
    for arm, i in (("left", 14), ("right", 22)):
        closed = action[max(start - 1, 0):end, i] < 0
        hits = np.nonzero(closed[1:] & ~closed[:-1])[0]
        if len(hits):
            first[arm] = hits[0]
    return min(first, key=first.get) if first else None


def held_at(action: np.ndarray, frame: int) -> dict:
    """arm -> True when the recorded gripper command is closed just before ``frame``."""
    a = action[max(frame - 1, 0)]
    return {"left": bool(a[14] < 0), "right": bool(a[22] < 0)}


# ---------------------------------------------------------------- offline: the catalog


def _ref(task: str, name: str) -> dict:
    inst, category, fixed = task_objects(task).get(name, (None, name.rsplit("_", 1)[0], False))
    return {"__type__": "ObjRef", "id": inst or name, "category": inst.split(".")[0] if inst else category,
            "fixed": fixed, "part": None}


def skill_call(task: str, human: str, objs: list) -> tuple[dict, list]:
    """The ONE SkillCall (codec dict, b1k.connector.codec.to_dict layout) and its success atoms for a segment."""
    skill, extra = HUMAN_TO_SKILL[human]
    first = _ref(task, objs[0])
    if skill in ("pick_up", "hold"):
        args = {"__type__": "PickArgs", "obj": first, "grasp": "auto", "then": "carry"} if skill == "pick_up" else \
            {"__type__": "HoldArgs", "obj": first, "pose": "carry"}
        atoms = [("holding", [first["id"]], True)]
    elif skill == "place":
        targets = [_ref(task, o) for o in objs[1:1 + len(extra)]]
        rels = [{"__type__": "Relation", "rel": r, "target": t} for r, t in zip(extra, targets)]
        args = {"__type__": "PlaceArgs", "obj": first, "relations": rels, "orient": "any"}
        atoms = [(BDDL_PRED[r], [first["id"], t["id"]], True) for r, t in zip(extra, targets)]
    elif skill in ("open", "close"):
        args = {"__type__": "OpenArgs", "target": first, "joint": None, "purpose": "state", "min_fraction": None} \
            if skill == "open" else {"__type__": "CloseArgs", "target": first, "joint": None}
        atoms = [("open", [first["id"]], skill == "open")]
    elif skill == "press":
        args = {"__type__": "PressArgs", "target": first, "want_on": extra}
        atoms = [("toggled_on", [first["id"]], extra is not False)]
    else:  # push
        args = {"__type__": "PushArgs", "obj": first, "direction": None, "distance": None}
        atoms = []
    call = {"__type__": "SkillCall", "skill": skill, "args": args, "arm": None, "percept": None, "variant": None,
            "backend": None, "freeze_trunk": False, "budget_steps": None, "seed": None, "call_id": ""}
    return call, [{"__type__": "Fact", "pred": p, "args": a, "value": v} for p, a, v in atoms]


def resolve(task: str, name: str) -> str | None:
    """An annotated object as a scene object name. Some annotations name only a category ("apple",
    "electric_refrigerator"): that resolves when one object of the scene has it (a task object first), else None."""
    objs = task_objects(task)
    if name in objs:
        return name
    hits = [n for n, (inst, cat, _) in objs.items() if name in (cat, inst and inst.split(".")[0])]
    scoped = [n for n in hits if objs[n][0]]
    return scoped[0] if len(scoped) == 1 else hits[0] if len(hits) == 1 else None


def build_catalog(out: Path = OUT_DIR / "catalog.parquet"):
    """Every mapped human segment of manipulation_ranges.jsonl, one row each, ranked for bench use: resolved
    objects first, then the fewer scene-changing segments before it (0 = the instance's initial state: only the
    human's navigation is replayed), then the earlier its start frame (a shorter replay)."""
    import pandas as pd

    rows = []
    with open(DATASET / "manipulation_ranges.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            task, before, n_seg = ep["task"], [], 0
            for seg in ep["manipulation_ranges"]:
                human, objs = seg["skill_description"][0], seg["object_id"][0]
                if human in HUMAN_TO_SKILL and all(isinstance(o, str) for o in objs):
                    skill, extra = HUMAN_TO_SKILL[human]
                    names = [resolve(task, o) for o in objs]
                    rows.append({
                        "task": task, "episode_index": ep["episode_index"], "raw_episode_id": ep["raw_episode_id"],
                        "instance": ep["raw_episode_id"] % 10000 // 10, "human_skill": human, "skill": skill,
                        "kind": "+".join(extra) if isinstance(extra, tuple) else extra if isinstance(extra, str) else None,
                        "human_objects": list(objs), "objects": [n or o for n, o in zip(names, objs)],
                        "resolved": all(names), "category": task_objects(task).get(names[0], (0, objs[0]))[1],
                        "start_frame": seg["start_frame"], "end_frame": seg["end_frame"],
                        "n_before": n_seg, "moved_before": sorted(set(before)),
                    })
                if human not in NO_CHANGE:
                    n_seg += 1
                    flat = [x for o in objs for x in ([o] if isinstance(o, str) else o)]
                    before += [resolve(task, o) or o for o in flat if o not in ("left", "right", "robot")]
    cat = pd.DataFrame(rows).sort_values(["skill", "resolved", "n_before", "start_frame"], ascending=[1, 0, 1, 1],
                                         kind="stable").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    cat.to_parquet(out)
    return cat


def load_catalog(path: Path = OUT_DIR / "catalog.parquet"):
    import pandas as pd

    return pd.read_parquet(path)


# ---------------------------------------------------------------- in the sim

HEAD_VIDEO = "observation.depth_linear.zed_link_camera_0"


def recorded_head_depth(episode_index: int, frame: int) -> np.ndarray:
    """The recorded head (zed, 720x720) depth in metres at one frame, decoded from the episode's shard video."""
    import av
    import pyarrow.parquet as pq

    from omnigibson.eval.utils.obs_utils import dequantize_depth

    m = episodes().loc[episode_index]
    meta = DATASET / f"meta/episodes/chunk-{m['meta/episodes/chunk_index']:03d}/file-{m['meta/episodes/file_index']:03d}.parquet"
    cols = [f"videos/{HEAD_VIDEO}/{k}" for k in ("chunk_index", "file_index", "from_timestamp")]
    v = pq.read_table(meta, columns=["episode_index"] + cols, filters=[("episode_index", "=", int(episode_index))])
    chunk, file, t0 = (v.column(c)[0].as_py() for c in cols)
    t = t0 + frame / FPS
    with av.open(str(DATASET / f"videos/{HEAD_VIDEO}/chunk-{chunk:03d}/file-{file:03d}.mp4")) as c:
        s = c.streams.video[0]
        c.seek(int(t / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            if fr.pts * s.time_base >= t - 0.5 / FPS:
                return dequantize_depth(fr.to_ndarray(format="gray12le").astype(np.float64))
    raise ValueError(f"episode {episode_index} frame {frame}: no head depth frame at {t:.3f} s")


def make_env(task: str):
    """The evaluator's environment for ``task`` (eval r1pro.yaml controllers = the demos' action space, base mass,
    head aperture, the challenge's partial room load), with only the head camera's depth rendered."""
    import omnigibson as og
    from omegaconf import OmegaConf

    from omnigibson.eval import evaluator as E  # also sets the evaluator's gm flags
    from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_ROOMS, generate_basic_environment_config

    for rule in E.DISABLED_TRANSITION_RULES:
        rule.ENABLED = False
    task_cfg = available_tasks()[task][0]
    cfg = generate_basic_environment_config(task_name=task, task_cfg=task_cfg)
    cfg["scene"]["load_room_instances"] = TASK_NAMES_TO_ROOMS[task]
    robot = OmegaConf.to_container(OmegaConf.load(E.DEFAULT_ROBOT_CONFIG_PATH))
    robot.pop("eval")
    robot.update(position=task_cfg["robot_start_position"], orientation=task_cfg["robot_start_orientation"],
                 obs_modalities=["depth_linear"], include_sensor_names=["zed_link"])
    robot["sensor_config"]["VisionSensor"]["sensor_kwargs"].update(image_height=720, image_width=720)
    cfg["robots"] = [robot]
    env = og.Environment(configs=cfg)
    r = env.robots[0]
    og.sim.stop()
    r.base_footprint_link.mass = E.EVAL_BASE_LINK_MASS
    og.sim.play()
    head_sensor(r).horizontal_aperture = E.EVAL_HEAD_HORIZONTAL_APERTURE
    return env


def head_sensor(robot):
    return next(s for n, s in robot.sensors.items() if "zed_link" in n)


def _yaw(quat) -> float:
    x, y, z, w = (float(v) for v in quat)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _pose(robot) -> list:
    pos, quat = robot.get_position_orientation()
    return [float(pos[0]), float(pos[1]), float(pos[2]), _yaw(quat)]


def set_robot(robot, state_row: np.ndarray, x: float, y: float, z: float, yaw: float) -> None:
    """Teleport the base and set trunk, arms and fingers to a recorded proprio row."""
    import torch as th

    robot.set_position_orientation(position=th.tensor([x, y, z]),
                                   orientation=th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]))
    groups = [(robot.trunk_control_idx, "trunk_qpos")]
    for arm in ("left", "right"):
        groups += [(robot.arm_control_idx[arm], f"arm_{arm}_qpos"), (robot.gripper_control_idx[arm], f"gripper_{arm}_qpos")]
    idx = th.cat([i for i, _ in groups])
    q = th.tensor(np.concatenate([state_row[PROPRIO_SLICES[k]] for _, k in groups]), dtype=th.float32)
    robot.set_joint_positions(q, indices=idx)
    robot.keep_still()


def proprio_error(robot, state_row: np.ndarray) -> dict:
    """Largest joint error (rad) over trunk and arms, and each eef's position error (m, base frame)."""
    p = {k: v.cpu().numpy() for k, v in robot._get_proprioception_dict().items()}
    joints = ("trunk_qpos", "arm_left_qpos", "arm_right_qpos")
    return {
        "joint_max_rad": float(max(np.abs(p[k] - state_row[PROPRIO_SLICES[k]]).max() for k in joints)),
        "eef_m": {arm: float(np.linalg.norm(p[f"eef_{arm}_pos"] - state_row[PROPRIO_SLICES[f"eef_{arm}_pos"]]))
                  for arm in ("left", "right")},
    }


def depth_error(env, episode_index: int, frame: int) -> dict:
    """Rendered head depth against the recorded frame: median |diff| and the share of pixels within 5 cm (both
    depths clipped to the 10 m video range). A base pose off by a few cm or a degree shows here."""
    import omnigibson as og

    robot = env.robots[0]
    for _ in range(3):
        og.sim.render()
    obs, _ = env.get_obs()
    sensor = head_sensor(robot)
    sim = obs[robot.name][sensor.name]["depth_linear"].cpu().numpy().astype(np.float64)
    rec = recorded_head_depth(episode_index, frame)
    d = np.abs(np.clip(sim, 0.01, 10.0) - rec)
    return {"median_m": round(float(np.median(d)), 4), "within_5cm": round(float((d < 0.05).mean()), 3)}


def held(robot) -> dict:
    return {arm: (o.name if (o := robot._ag_obj_in_hand[arm]) is not None else None) for arm in ("left", "right")}


def materialize(case: dict, env, method: str | None = None, settle: int = 10) -> dict:
    """Bring ``env`` (``make_env(case["task"])``) to the human's situation at the case's start frame and return the
    setup: robot pose + joints, what each hand holds, the fidelity numbers, and the snapshot (object states).

    ``method``: "replay" (default: the recorded actions from frame 0 through the evaluator's controllers; the head
    depth then matches the recording to ~1 mm median) or "set_state" (teleport to the dead-reckoned base pose and set
    the joints; only when nothing moved before, and the dead-reckoned base drifts: 5 deg after 9 s, 13 cm after 14 s,
    so it is a fallback, not the default)."""
    import omnigibson as og
    import torch as th

    from omnigibson.eval.evaluator import load_task_instance

    robot = env.robots[0]
    ep, start = int(case["episode_index"]), int(case["start_frame"])
    method = method or "replay"
    state, action = episode_arrays(ep, stop=start + 1)
    env.reset()
    load_task_instance(env, robot, int(case["instance"]), mode="train")
    env.reset()
    x0, y0, z0, yaw0 = _pose(robot)
    frame0 = proprio_error(robot, state[0])
    reckoned = dead_reckon(state, x0, y0, yaw0)[start]
    t0, grasps = time.time(), []
    if method == "set_state":
        set_robot(robot, state[start], *reckoned[:2], z0, reckoned[2])
    else:
        closes = {f + 15 for i in (14, 22) for f in np.nonzero(np.diff((action[:start, i] < 0).astype(int)) == 1)[0] + 1}
        with og.sim.render_on_step(False):  # physics only; the depth check renders at the end
            for t in range(start):
                env.step(th.tensor(action[t], dtype=th.float32))
                if t in closes:
                    grasps.append({"frame": t, "held": held(robot)})
    for _ in range(settle):
        env.step(hold_action(action, start))
    pose = _pose(robot)
    fid = {
        "method": method, "seconds": round(time.time() - t0, 1), "frame0": frame0,
        "at_start": proprio_error(robot, state[start]),
        "base_vs_reckoned": {"xy_m": round(math.hypot(pose[0] - reckoned[0], pose[1] - reckoned[1]), 4),
                             "yaw_deg": round(math.degrees(math.remainder(pose[3] - reckoned[2], math.tau)), 2)},
        "depth": depth_error(env, ep, start),
        "grasps": grasps,
        "held": held(robot),
        "human_gripper_closed": held_at(action, start),
    }
    names = {getattr(o, "name", None) for o in env.task.object_scope.values()}  # systems and unloaded ones drop out
    names |= set(case["objects"]) | set(case["moved_before"])
    objs = {n: o for n in sorted(names - {robot.name, None}) if (o := env.scene.object_registry("name", n)) is not None}
    snapshot = {n: o.dump_state(serialized=False) for n, o in objs.items()}
    rstate = robot.dump_state(serialized=False)
    rstate["controller_groups"] = {}  # the bench's controllers are its own
    snapshot[robot.name] = rstate
    og.sim.render()
    return {
        "robot_pose": {"x": round(pose[0], 4), "y": round(pose[1], 4), "z": round(pose[2], 4), "yaw": round(pose[3], 4)},
        "joints": {k: [round(float(v), 4) for v in state[start][PROPRIO_SLICES[k]]]
                   for k in ("trunk_qpos", "arm_left_qpos", "arm_right_qpos", "gripper_left_qpos", "gripper_right_qpos")},
        "held": fid["held"],
        "fidelity": fid,
        "snapshot": snapshot,
    }


def hold_action(action: np.ndarray, frame: int):
    """Stand still and keep the human's last joint and gripper commands before ``frame``."""
    import torch as th

    hold = th.tensor(action[max(frame - 1, 0)], dtype=th.float32)
    hold[:3] = 0.0
    return hold


def restore(env, snapshot: dict, instance: int) -> None:
    """The bench side: the instance first, then every saved object and the robot on top of it."""
    from omnigibson.eval.evaluator import load_task_instance
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    robot = env.robots[0]
    env.reset()
    load_task_instance(env, robot, instance, mode="train")
    env.reset()
    for name, state in recursively_convert_to_torch(json.loads(json.dumps(snapshot))).items():
        if name == robot.name:
            state["controller_groups"] = {}
        env.scene.object_registry("name", name).load_state(state, serialized=False)


# ---------------------------------------------------------------- the starter set and the case files

# (file stem, catalog filter): 5-10 cases each, one per task, best-ranked first
STARTER = {
    "pick_up": lambda c: c.skill == "pick_up",
    "place_on": lambda c: (c.skill == "place") & (c.kind == "on"),
    "place_in": lambda c: (c.skill == "place") & (c.kind == "in"),
    "open_drawer": lambda c: (c.skill == "open") & (c.kind == "drawer"),
    "open_door": lambda c: (c.skill == "open") & (c.kind == "door")
    & c.category.str.contains("fridge|refrigerator|cabinet"),
    "press": lambda c: c.skill == "press",
}


def pick_starter(cat, group: str, k: int = 8, min_start: int = 150):
    """The best-ranked resolved cases of one starter group, one per task (two when fewer than 5 tasks have one),
    at most k. A place must follow a pick of the same object. ``min_start``: the human navigated at least 5 s first,
    so the stance is theirs, not the instance's spawn."""
    c = cat[STARTER[group](cat) & cat.resolved & (cat.start_frame >= min_start)]
    if group.startswith("place"):
        c = c[[o[0] in m for o, m in zip(c.objects, c.moved_before)]]
    c = c.sort_values(["n_before", "start_frame"], kind="stable")
    return c.groupby("task", sort=False).head(1 if c.task.nunique() >= 5 else 2).head(k)


def case_id(case) -> str:
    return f"{case['group']}.{case['task']}.e{case['episode_index']}.f{case['start_frame']}"


def case_yaml(case: dict, setup: dict) -> dict:
    """One bench case in SPEC §9's layout. ``setup`` holds what skillbench.py applies today (robot_pose [x, y, yaw],
    torso), plus ``held`` when a hand holds something (the bench refuses that key until it restores a grasp, from
    the snapshot). ``demo`` is where it came from, the rest of the human's posture and the restore's fidelity."""
    state, action = episode_arrays(int(case["episode_index"]), stop=int(case["end_frame"]))
    call, atoms = skill_call(case["task"], case["human_skill"], list(case["objects"]))
    fid, r, j = setup["fidelity"], setup["robot_pose"], setup["joints"]
    held = {a: o for a, o in setup["held"].items() if o}
    return {
        "id": case_id(case), "task": case["task"], "instance": int(case["instance"]), "mode": "train",
        "setup": {"robot_pose": [r["x"], r["y"], r["yaw"]], "torso": j["trunk_qpos"]} | ({"held": held} if held else {}),
        "call": call, "n": 5, "seeds": [0, 1, 2, 3, 4], "expect": {"status": "succeeded"}, "success": atoms,
        "baseline": "legacy",
        "demo": {
            "dataset": DATASET.name, "episode_index": int(case["episode_index"]),
            "raw_episode_id": int(case["raw_episode_id"]), "segment": [int(case["start_frame"]), int(case["end_frame"])],
            "human_skill": case["human_skill"], "human_objects": list(case["human_objects"]),
            "human_arm": closing_arm(action, int(case["start_frame"]), int(case["end_frame"])),
            "moved_before": list(case["moved_before"]), "restore": fid["method"],
            "arms": {a: j[f"arm_{a}_qpos"] for a in ("left", "right")},
            "fingers": {a: j[f"gripper_{a}_qpos"] for a in ("left", "right")},
            "snapshot": f"snapshots/{case_id(case)}.json",
            "fidelity": {k: fid[k] for k in ("at_start", "base_vs_reckoned", "depth", "grasps", "human_gripper_closed",
                                             "restore_check") if k in fid},
        },
        "note": (f"{case['task']} ep {case['episode_index']} f{case['start_frame']}: {case['human_skill']} "
                 f"{' / '.join(case['objects'])}; {fid['method']} of {case['start_frame']} frames through "
                 f"{case['n_before']} earlier segment(s); head depth within 5 cm {fid['depth']['within_5cm']:.0%}; "
                 f"held {held or 'nothing'}"),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--catalog", action="store_true", help="(re)build catalog.parquet and exit")
    p.add_argument("--task", help="materialize the starter cases of this task only (one sim per task)")
    p.add_argument("--list", action="store_true", help="print the starter set's tasks and exit")
    p.add_argument("--collect", action="store_true",
                   help="merge cases/*.yaml into one <group>.yaml list per group (skillbench's case files) + starter.tsv")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    if args.catalog:
        build_catalog(args.out / "catalog.parquet")
        return
    import yaml

    if args.collect:
        groups = {}
        for path in sorted((args.out / "cases").glob("*.yaml")):
            doc = yaml.safe_load(open(path))
            groups.setdefault(doc["id"].split(".")[0], []).append(doc)
        for group, docs in groups.items():
            with open(args.out / f"{group}.yaml", "w") as f:
                yaml.safe_dump(docs, f, sort_keys=False, width=120, default_flow_style=None)
        with open(args.out / "starter.tsv", "w") as f:
            f.writelines(f"{d['id']}\t{d['note']}\n" for docs in groups.values() for d in docs)
        return
    import pandas as pd

    cat = load_catalog(args.out / "catalog.parquet")
    starter = pd.concat([pick_starter(cat, g).assign(group=g) for g in STARTER])
    if args.list:
        print(" ".join(sorted(starter.task.unique())))
        return
    import omnigibson as og
    from omnigibson.utils.config_utils import TorchEncoder

    env = make_env(args.task)
    (args.out / "snapshots").mkdir(parents=True, exist_ok=True)
    (args.out / "cases").mkdir(parents=True, exist_ok=True)
    for case in starter[starter.task == args.task].to_dict("records"):
        setup = materialize(case, env)
        if case["skill"] in ("place", "hold") and case["objects"][0] not in setup["held"].values():
            print("REJECT", case_id(case), f"the replay did not reproduce the grasp: holding {setup['held']}", flush=True)
            continue
        snap = args.out / "snapshots" / f"{case_id(case)}.json"
        with open(snap, "w") as f:
            json.dump(setup["snapshot"], f, cls=TorchEncoder)
        # the bench's fast path: restore the snapshot and stand still, the pose and the hands must come back
        restore(env, json.load(open(snap)), int(case["instance"]))
        action = episode_arrays(int(case["episode_index"]), stop=int(case["start_frame"]))[1]
        for _ in range(10):
            env.step(hold_action(action, int(case["start_frame"])))
        x, y, _, yaw = _pose(env.robots[0])
        r = setup["robot_pose"]
        setup["fidelity"]["restore_check"] = {
            "xy_m": round(math.hypot(x - r["x"], y - r["y"]), 4),
            "yaw_deg": round(math.degrees(math.remainder(yaw - r["yaw"], math.tau)), 2),
            "held_same": held(env.robots[0]) == setup["held"],
        }
        doc = case_yaml(case, setup)
        with open(args.out / "cases" / f"{case_id(case)}.yaml", "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False, width=120, default_flow_style=None)
        print("CASE", doc["id"], json.dumps(doc["demo"]["fidelity"]), doc["note"], flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
