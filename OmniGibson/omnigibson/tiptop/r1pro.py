"""BEHAVIOR-1K R1Pro (the challenge robot) inside a BEHAVIOR scene as a TiPToP client.

TiPToP plans the torso and the left arm with its ``r1pro_left`` embodiment (tiptop/tiptop/embodiments/r1pro.py), a
cuRobo model generated from the very URDF and collision spheres OmniGibson ships for this robot. The right arm and
the fingers are locked in that model, so the simulator holds them at the same values, which it takes from the server
metadata (``embodiment.locked_joints`` / ``joint_names`` / ``q_home``) or, offline, from the embodiment's meta file
in the tiptop submodule.

World frame for TiPToP = the robot's ``base_link`` (floor level), which is both ``robot.get_position_orientation()``
here and the URDF root of the planner model. Capture uses the head camera (``zed_link``) or the left wrist camera;
pixels belonging to the robot itself are removed from the depth so M2T2 never proposes grasps on the robot.
Navigation is out of scope: the robot is placed next to the target furniture (``--near``) or at an explicit pose.
"""

import logging
import math
from pathlib import Path

import numpy as np
import torch as th
import yaml

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.tiptop.scene import OBJECT_PRESETS, TiptopSim, look_at_quat_xyzw

log = logging.getLogger(__name__)

ROBOT_NAME = "robot_r1"
ROBOT_TYPE = "r1pro_left"
CAMERA_LINKS = {"head": "zed_link", "wrist": "left_realsense_link"}
SHADOW_CAM = "tiptop_cam"  # external sensor moved onto the robot camera's pose for each capture (see _capture_obs)
# Capture posture: the ready posture with the left shoulder abducted so the arm swings out to the robot's left, out of
# the head camera's view. In the ready posture the gripper sits in front of the table objects and hides most of them
# (a detector then segments the gripper); probed in Rs_int: mug 3881 px instead of 2005, bowl 8523 instead of 4994,
# 0 robot pixels, no contact. Applied on top of q_home, joint name -> value.
LOOK_ARM = {"left_arm_joint2": 2.0}
LOOK_SETTLE_STEPS = 60
CAPTURE_MAX_RENDERS = 40  # render pairs after moving the capture camera (temporal accumulation)
CAPTURE_CONVERGED_DIFF = 0.25  # mean absolute rgb change (0-255) between consecutive renders that counts as settled
HEAD_APERTURE_MM = 40.0  # BEHAVIOR challenge eval setting (99 deg HFOV); OmniGibson's default 20.995 gives 63 deg
WRIST_APERTURE_MM = 20.995  # OmniGibson VisionSensor default, set explicitly so the shadow camera matches exactly
FLOOR_COVERINGS = ("floors", "ceilings", "paver", "carpet", "rug", "mat", "doormat", "tile")  # stood on, not avoided
ROBOT_HEIGHT = 1.6  # m, top of the head camera with the challenge torso posture is ~1.4
ROBOT_FOOTPRINT = 0.36  # half extent (m) used for free-space checks; base bbox is 0.64 x 0.68
CAMERA_MIN_DIST = 0.65  # nearer objects are cut by the head camera's bottom edge (it meets the table 0.55 m ahead)
TARGET_HALF_WIDTH = 0.22  # containers this wide (basket) hide an item behind them from the head camera
CAMERA_HALF_FOV = math.radians(49.6)  # head camera's true half-FOV: atan(360 / 306) at 720 px, fx 306
FRAMING_PENALTY = 2.0  # score cost per radian an object's edge falls outside the frame (see best_base_pose)
BASE_MASS_KG = 250.0  # omnigibson/eval/evaluator.py sets this for r1/r1pro; keeps the robot upright


def embodiment_meta_path(robot_type: str = ROBOT_TYPE) -> Path:
    """Generated meta file of the tiptop embodiment (tiptop submodule), the offline source of the locked posture."""
    repo = Path(og.__file__).resolve().parents[2]
    return repo / "tiptop" / "tiptop" / "embodiments" / "assets" / "r1pro" / f"{robot_type}_meta.yml"


def load_embodiment_meta(robot_type: str = ROBOT_TYPE) -> dict:
    path = embodiment_meta_path(robot_type)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `pixi run python scripts/make_r1pro_embodiment.py` in tiptop/")
    with open(path) as f:
        return yaml.safe_load(f)


def challenge_task_info(activity: str) -> tuple[str, list[str]]:
    """Scene model and room instances the challenge evaluator loads for a task (its metadata files)."""
    import yaml
    from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_ROOMS
    from omnigibson.macros import gm

    tasks = yaml.safe_load(
        open(Path(gm.DATA_PATH) / "2026-challenge-task-instances" / "metadata" / "available_tasks.yaml")
    )
    if activity not in tasks:
        raise ValueError(f"{activity!r} is not a challenge task; known: {sorted(tasks)[:5]}... ({len(tasks)})")
    return str(tasks[activity][0]["scene_model"]), list(TASK_NAMES_TO_ROOMS[activity])


def bddl_category(bddl_name: str) -> str:
    """'butter_cookie.n.01_2' -> 'butter cookie', 'can__of__soda.n.01_1' -> 'can of soda' (what a detector is asked for)."""
    return bddl_name.split(".n.")[0].replace("__", "_").replace("_", " ")


def bddl_label(bddl_name: str) -> str:
    """'butter_cookie.n.01_2' -> 'butter_cookie_2': the per-instance name used in requests and plans."""
    category, _, index = bddl_name.rpartition("_")
    return f"{bddl_category(category).replace(' ', '_')}_{index}"


def make_r1pro_env_config(
    scene_model: str = "Rs_int",
    load_room_types=None,
    spawn_presets=(),
    grasping_mode: str = "sticky",
    camera: str = "head",
    head_resolution: int = 720,
    wrist_resolution: int = 480,
    head_aperture_mm: float = HEAD_APERTURE_MM,
    not_load_object_categories=("ceilings",),
    activity: str | None = None,
    activity_instance_id: int = 0,
    load_room_instances=None,
    segmentation: bool = True,
) -> dict:
    """OmniGibson config: BEHAVIOR scene + R1Pro with absolute joint controllers on every group.

    Spawned objects start high above the floor and are placed onto furniture by R1ProSim.place_on(). With
    ``activity`` the scene is a challenge task instance (BehaviorTask, pre-sampled objects, the evaluator's rooms).
    ``segmentation=False`` renders rgb + depth only (what the challenge allows); ground-truth masks then come from
    object geometry instead of the annotator, and the robot is not masked out of the depth.

    Cameras: the robot camera TiPToP uses renders rgb only (video); an external "shadow" VisionSensor with the same
    intrinsics provides rgb + depth_linear + seg_instance for the capture frame after being moved onto the robot
    camera's pose. Instance segmentation attached to a robot-mounted camera leaks GPU memory every step in this
    Isaac Sim build and segfaults the synthetic-data graph after ~35 steps; an external camera does not.
    """
    jc = {
        "name": "JointController",
        "motor_type": "position",
        "use_delta_commands": False,
        "use_impedances": False,
        "command_input_limits": None,
        "command_output_limits": None,
    }
    gripper = {
        "name": "MultiFingerGripperController",
        "mode": "binary",
        "command_input_limits": None,
        "command_output_limits": None,
    }
    res = head_resolution if camera == "head" else wrist_resolution
    aperture = head_aperture_mm if camera == "head" else WRIST_APERTURE_MM
    shadow_cam = {
        "sensor_type": "VisionSensor",
        "name": SHADOW_CAM,
        "relative_prim_path": f"/{SHADOW_CAM}",
        "modalities": ["rgb", "depth_linear"] + (["seg_instance"] if segmentation else []),
        "sensor_kwargs": {
            "image_width": res,
            "image_height": res,
            "focal_length": 17.0,
            "horizontal_aperture": aperture,
        },
        "position": [0.0, 0.0, 1.5],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "include_in_obs": False,
    }
    scene = {
        "type": "InteractiveTraversableScene",
        "scene_model": scene_model,
        "include_robots": False,
        "not_load_object_categories": list(not_load_object_categories),
    }
    if load_room_types:
        scene["load_room_types"] = list(load_room_types)
    if load_room_instances:
        scene["load_room_instances"] = list(load_room_instances)
    task = {"type": "DummyTask"}
    if activity:
        task = {
            "type": "BehaviorTask",
            "activity_name": activity,
            "activity_definition_id": 0,
            "activity_instance_id": activity_instance_id,
            "online_object_sampling": False,
            "debug_object_sampling": False,
            "highlight_task_relevant_objects": False,
            "termination_config": {"max_steps": 10**8},
            "reward_config": {"r_potential": 1.0},
            "include_obs": False,
        }
    objects = []
    for i, preset in enumerate(spawn_presets):
        if preset not in OBJECT_PRESETS:
            raise ValueError(f"unknown object preset {preset!r}; known: {sorted(OBJECT_PRESETS)}")
        objects.append({**OBJECT_PRESETS[preset], "name": preset, "position": [0.0, 0.0, 3.0 + 0.3 * i]})
    return {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
            "external_sensors": [shadow_cam],
        },
        "scene": scene,
        "robots": [
            {
                "model": "r1pro",
                "name": ROBOT_NAME,
                "obs_modalities": ["rgb", "proprio"],  # rgb for the video; capture frames come from the shadow camera
                "include_sensor_names": [CAMERA_LINKS[camera]],
                "action_normalize": False,
                "self_collisions": True,
                "grasping_mode": grasping_mode,
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_height": wrist_resolution,
                            "image_width": wrist_resolution,
                            "focal_length": 17.0,
                            "horizontal_aperture": WRIST_APERTURE_MM,
                        }
                    },
                    f"{CAMERA_LINKS['head']}:Camera:0": {
                        "sensor_kwargs": {
                            "image_height": head_resolution,
                            "image_width": head_resolution,
                            "horizontal_aperture": head_aperture_mm,
                        }
                    },
                },
                "controller_config": {
                    "base": {
                        "name": "HolonomicBaseJointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                    "trunk": dict(jc),
                    "arm_left": dict(jc),
                    "arm_right": dict(jc),
                    "gripper_left": dict(gripper),
                    "gripper_right": dict(gripper),
                },
            }
        ],
        "objects": objects,
        "task": task,
    }


class R1ProSim(TiptopSim):
    """R1Pro in a BEHAVIOR scene; the TiptopSim interface (capture / step / q_arm / objects) for the left arm."""

    expect_table_z = None  # no synthetic table at base z = 0: validate_capture only checks the objects
    mask_labels_as_invalid = (ROBOT_NAME,)
    look_arm = LOOK_ARM  # joint overrides on top of q_home for the capture; None: capture in the ready posture

    def __init__(self, config: dict, camera: str = "head"):
        self.config = config
        self.env = og.Environment(configs=config)
        self.robot = self.env.robots[0]
        self.arm = "left"
        self.joint_index = {name: i for i, name in enumerate(self.robot.joints.keys())}
        self.planned_joints = list(self.robot.arm_joint_names[self.arm])  # replaced by apply_posture (torso + arm)
        self.arm_idx = th.tensor([self.joint_index[j] for j in self.planned_joints])
        self.gripper_idx = self.robot.gripper_control_idx[self.arm]
        self.dt = og.sim.get_sim_step_dt()
        self.cam_name = f"{self.robot.name}:{CAMERA_LINKS[camera]}:Camera:0"
        self.robot_cam = self.robot.sensors[self.cam_name]
        self.cam = self.env.external_sensors[SHADOW_CAM]  # capture camera; moved onto robot_cam's pose per frame
        self.objects = {}
        self.bddl_names = {}  # tiptop label -> BDDL instance name for tracked task objects
        self.posture = {}
        self.q_home = None
        self.last_obs = None
        # The challenge evaluator (and JoyLo) give the base 250 kg; with the asset's default mass the leaning
        # challenge torso posture tips the whole robot over backwards.
        self.robot.base_footprint_link.mass = BASE_MASS_KG
        log.info(
            f"R1Pro DOF order: {list(self.joint_index)}; left arm idx {self.arm_idx.tolist()}, "
            f"gripper idx {self.gripper_idx.tolist()}, action dim {self.robot.action_dim}, camera {self.cam_name}"
        )
        assert list(self.robot.controller_order) == [
            "base",
            "trunk",
            "arm_left",
            "gripper_left",
            "arm_right",
            "gripper_right",
        ]
        self.env.reset()

    # ---------------------------------------------------------------- challenge task
    def task_scope(self) -> dict:
        """BDDL instance name -> simulated object for the loaded BehaviorTask (no agent, no floors, no systems)."""
        scope = getattr(self.env.task, "object_scope", None) or {}
        return {
            k: v
            for k, v in scope.items()
            if v is not None and not k.startswith(("agent.", "floor.")) and hasattr(v, "aabb")
        }

    def track_task_objects(self, skip_categories=("table", "floor", "agent")) -> dict:
        """Track every task object under its per-instance label ('candle_1'); furniture the items rest on is skipped."""
        self.bddl_names = {}
        for bddl, obj in self.task_scope().items():
            if bddl_category(bddl) in skip_categories:
                continue
            label = bddl_label(bddl)
            self.objects[label] = obj
            self.bddl_names[label] = bddl
        log.info(f"tracking task objects: {self.bddl_names}")
        return dict(self.bddl_names)

    def tiptop_goal(self, atoms: list[dict], category_level: bool) -> tuple[list[str], list[dict]]:
        """Translate BDDL goal atoms (inside/ontop/on/nextto/holding over BDDL names) for TiPToP.

        Returns the labels the request names and the atoms in TiPToP's predicates. Per instance ('candle_1', with
        ground-truth masks) or per category ('candle': the detector finds every instance and the goal takes the
        best-scoring one, since the task does not care which candle goes into which basket).
        """
        predicates = {"inside": "on", "ontop": "on", "on": "on", "nextto": "near", "holding": "holding"}
        label_of = {bddl: label for label, bddl in self.bddl_names.items()}

        def name(arg):
            if arg not in label_of:
                raise ValueError(f"goal names {arg!r}, which is not a tracked task object: {sorted(label_of)}")
            return bddl_category(arg).replace(" ", "_") if category_level else label_of[arg]

        out = []
        for atom in atoms:
            if atom["predicate"] not in predicates:
                raise ValueError(f"unsupported goal predicate {atom['predicate']!r} ({sorted(predicates)})")
            out.append({"predicate": predicates[atom["predicate"]], "args": [name(a) for a in atom["args"]]})
        if category_level:
            labels = sorted({bddl_category(b).replace(" ", "_") for b in self.bddl_names.values()})
        else:
            labels = sorted(self.bddl_names)
        return labels, out

    def _goal_values(self) -> list[list[bool]]:
        """Truth of every predicate of every ground goal option, evaluating each grounded predicate once.

        forpairs goals ground into every pairing (tens of thousands of options for four baskets); the simulator
        predicates (inside, ontop) are the expensive part, so they are memoized across options.
        """
        task = self.env.task
        leaf_cache, head_cache = {}, {}

        def evaluate(name, *entities):
            key = (name, entities)
            if key not in leaf_cache:
                leaf_cache[key] = task._evaluate_predicate(name, *entities)
            return leaf_cache[key]

        values = []
        for option in task.ground_goal_state_options:
            row = []
            for head in option:
                v = head_cache.get(id(head))
                if v is None:
                    v = head_cache[id(head)] = bool(head.evaluate(evaluate))
                row.append(v)
            values.append(row)
        return values

    @staticmethod
    def _goal_name(head) -> str:
        terms = list(getattr(head, "terms", []))
        return f"{terms[0]}({', '.join(terms[1:])})" if terms else str(head)

    def predicate_holds(self, name: str, *bddl_names: str) -> bool:
        """Evaluate one grounded goal predicate (e.g. inside(item, container)) exactly as the challenge scorer does."""
        task = self.env.task
        terms = [name, *bddl_names]
        for option in task.ground_goal_state_options:
            for head in option:
                if list(getattr(head, "terms", [])) == terms:
                    return bool(head.evaluate(task._evaluate_predicate))
        raise KeyError(f"{name}({', '.join(bddl_names)}) is not a goal predicate of this task")

    def base_hint(self, name: str) -> list[float]:
        """Where a scene/task object is, in the robot base frame (TiPToP's world frame)."""
        obj = self.scene_object(name)
        pos_b, _ = self.to_base(obj.aabb_center, th.tensor([0.0, 0.0, 0.0, 1.0]))
        return [float(v) for v in pos_b]

    def _stage_geometry(self, name: str, support: str):
        """Support AABB, the object's xy half extents and the AABBs of whatever else stands on the support's top."""
        obj, sup = self.scene_object(name), self.scene_object(support)
        lo, hi = [v.cpu().numpy() for v in sup.aabb]
        olo, ohi = [v.cpu().numpy() for v in obj.aabb]
        hx, hy = (ohi[0] - olo[0]) / 2, (ohi[1] - olo[1]) / 2
        skip = {obj, sup, self.robot} | {self.scene_object(n) for n in self.contents_of(name)}
        others = []
        for other in self.env.scene.objects:
            if other in skip:
                continue
            alo, ahi = [v.cpu().numpy() for v in other.aabb]
            if ahi[2] < hi[2] - 0.02 or alo[2] > hi[2] + 0.6:
                continue  # below the top or far above it
            if ahi[0] < lo[0] or alo[0] > hi[0] or ahi[1] < lo[1] or alo[1] > hi[1]:
                continue  # not over the support
            others.append((alo, ahi))
        return lo, hi, hx, hy, others

    def free_spots_on(
        self, name: str, support: str, clearance: float = 0.02, edge: float = 0.04, step: float = 0.03
    ) -> list[tuple[float, float, float]]:
        """Every (x, y, gap) in world xy where ``name`` fits on top of ``support`` without overlapping anything.

        (x, y) is taken as the object's AABB center (a ``step`` grid over the top, ``edge`` in from its rim; place_on
        puts the object's origin there), gap the clearance to the nearest other object on the support (at least
        ``clearance``).
        """
        lo, hi, hx, hy, others = self._stage_geometry(name, support)
        spots = []
        for x in np.arange(lo[0] + edge + hx, hi[0] - edge - hx + 1e-6, step):
            for y in np.arange(lo[1] + edge + hy, hi[1] - edge - hy + 1e-6, step):
                gap = min(
                    (
                        max(alo[0] - (x + hx), (x - hx) - ahi[0], alo[1] - (y + hy), (y - hy) - ahi[1])
                        for alo, ahi in others
                    ),
                    default=1.0,
                )
                if gap < clearance:
                    continue
                spots.append((float(x), float(y), float(gap)))
        return spots

    def free_spot_on(
        self, name: str, support: str, clearance: float = 0.02, edge: float = 0.04, step: float = 0.03, near=None
    ):
        """(dx, dy) from the support's center where ``name`` fits on top of it without overlapping anything.

        Picks the spot with the most clearance, or the free spot closest to ``near`` (world xy) when given.
        """
        spots = self.free_spots_on(name, support, clearance=clearance, edge=edge, step=step)
        if not spots:
            others = self._stage_geometry(name, support)[-1]
            raise RuntimeError(f"no free spot for {name} on {support} ({len(others)} objects on it)")
        if near is not None:
            near = np.asarray(near, dtype=np.float64)  # float64 like the grid (numpy >= 2 would narrow to float32)
            x, y, gap = min(spots, key=lambda s: float(np.hypot(s[0] - near[0], s[1] - near[1])))
        else:
            x, y, gap = max(spots, key=lambda s: s[2])
        lo, hi = [v.cpu().numpy() for v in self.scene_object(support).aabb]
        cx, cy = ((lo + hi) / 2)[:2].tolist()
        log.info(f"free spot for {name} on {support}: ({x:.2f}, {y:.2f}), clearance {gap:.2f} m")
        return x - cx, y - cy

    def contents_of(self, container: str) -> list[str]:
        """Tracked task objects whose AABB center lies inside the container's AABB."""
        lo, hi = [v.cpu().numpy() for v in self.scene_object(container).aabb]
        inside = []
        for label, bddl in self.bddl_names.items():
            if bddl == container:
                continue
            c = self.objects[label].aabb_center.cpu().numpy()
            if np.all(c > lo) and np.all(c < hi):
                inside.append(bddl)
        return inside

    def move_with_contents(self, container: str, position, orientation=None) -> None:
        """Teleport a container and whatever sits inside it by the same offset (stand-in for carrying it)."""
        obj = self.scene_object(container)
        pos0, quat0 = obj.get_position_orientation()
        delta = th.as_tensor(position, dtype=th.float32) - pos0
        contents = self.contents_of(container)
        for name in [container, *contents]:
            o = self.scene_object(name)
            p, q = o.get_position_orientation()
            quat = q if name != container or orientation is None else th.as_tensor(orientation, dtype=th.float32)
            o.set_position_orientation(position=p + delta, orientation=quat)
            o.keep_still()
        log.info(f"moved {container} with {contents} by {np.round(delta.numpy(), 3).tolist()}")

    def place_on_with_contents(self, container: str, support: str, dx: float, dy: float, lift: float = 0.01) -> None:
        """place_on() for a container that may already hold items: they travel with it."""
        obj, sup = self.scene_object(container), self.scene_object(support)
        lo, hi = [v.cpu().numpy() for v in sup.aabb]
        olo, ohi = [v.cpu().numpy() for v in obj.aabb]
        center = (lo + hi) / 2
        pos = obj.get_position_orientation()[0].numpy()
        z = hi[2] + lift + (pos[2] - olo[2])  # keep the object's own origin height above its bottom
        self.move_with_contents(container, [center[0] + dx, center[1] + dy, z])

    def mark_goal_initial(self) -> None:
        """Remember which goal predicates already hold, as the challenge metric does (no credit for those)."""
        self.goal_initial = self._goal_values()

    def goal_status(self) -> dict:
        """Challenge-style score: 1 on full success, else the best goal option's newly satisfied fraction."""
        task = self.env.task
        values = self._goal_values()
        initial = getattr(self, "goal_initial", None) or [[False] * len(v) for v in values]
        options = task.ground_goal_state_options
        best_i, best_new = 0, -1
        for i, (now, was) in enumerate(zip(values, initial)):
            new = sum(int(v and not v0) for v, v0 in zip(now, was))
            if new > best_new:
                best_i, best_new = i, new
        success = any(all(row) for row in values)
        total = len(options[best_i]) if options else 0
        q = 1.0 if success else (best_new / total if total else 0.0)
        return {
            "success": success,
            "q_score": q,
            "options": len(options),
            "satisfied": [self._goal_name(h) for h, v in zip(options[best_i], values[best_i]) if v] if options else [],
            "unsatisfied": [self._goal_name(h) for h, v in zip(options[best_i], values[best_i]) if not v]
            if options
            else [],
            "new": best_new,
            "total": total,
        }

    # ---------------------------------------------------------------- scene setup
    def scene_object(self, name: str):
        obj = self.task_scope().get(name) or self.env.scene.object_registry("name", name)
        if obj is None:
            names = sorted(o.name for o in self.env.scene.objects)
            raise ValueError(f"no object {name!r} in scene {self.config['scene']['scene_model']}; objects: {names}")
        return obj

    def object_names(self) -> list[str]:
        return list(self.objects)

    def track(self, *names: str) -> None:
        """Objects whose masks/poses go to TiPToP and into the success check (spawned or scene objects)."""
        for name in names:
            self.objects[name] = self.scene_object(name)

    def place_on(self, name: str, support: str, dx: float = 0.0, dy: float = 0.0, lift: float = 0.01) -> None:
        """Drop a tracked object onto the top of a piece of furniture (AABB top + offsets)."""
        obj, sup = self.scene_object(name), self.scene_object(support)
        lo, hi = [v.cpu().numpy() for v in sup.aabb]
        olo, ohi = [v.cpu().numpy() for v in obj.aabb]
        center = (lo + hi) / 2
        pos = th.tensor([center[0] + dx, center[1] + dy, hi[2] + (ohi[2] - olo[2]) / 2 + lift], dtype=th.float32)
        obj.set_position_orientation(position=pos, orientation=th.tensor([0.0, 0.0, 0.0, 1.0]))
        obj.keep_still()
        self.objects[name] = obj
        log.info(f"placed {name} on {support} at {np.round(pos.numpy(), 3).tolist()} (support top z={hi[2]:.3f})")

    def scene_aabbs(self) -> list[tuple]:
        """(object, lo, hi) for every scene object but the robot: one AABB query each (the AABB is recomputed from
        the collision meshes on every access, ~50 ms for a house scene), to reuse across footprint checks."""
        return [(o, *[v.cpu().numpy() for v in o.aabb]) for o in self.env.scene.objects if o is not self.robot]

    def _footprint_free(self, x: float, y: float, ignore, aabbs=None) -> tuple[bool, str]:
        """Floor under the whole footprint, inside a room, and no other object's AABB overlapping the footprint.

        ``aabbs``: a scene_aabbs() snapshot to test against (taken here otherwise; nothing moves during a search).
        """
        r = ROBOT_FOOTPRINT
        corners = [(x + sx * r, y + sy * r) for sx in (-1, 1) for sy in (-1, 1)] + [(x, y)]
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        floors = [(lo, hi) for o, lo, hi in aabbs if getattr(o, "category", "") == "floors"]
        for cx, cy in corners:
            on_floor = False
            for lo, hi in floors:
                if lo[0] <= cx <= hi[0] and lo[1] <= cy <= hi[1]:
                    on_floor = True
                    break
            if not on_floor:
                return False, f"no floor under ({cx:.2f}, {cy:.2f})"
        seg_map = getattr(self.env.scene, "seg_map", None)
        if seg_map is not None:
            try:
                room = seg_map.get_room_instance_by_point(th.tensor([x, y]))
            except Exception:  # noqa: BLE001 - the map lookup is a best-effort filter
                room = "unknown"
            if room is None:
                return False, "outside every room"
        for obj, lo, hi in aabbs:
            if obj is self.robot or obj in ignore or getattr(obj, "category", "") in FLOOR_COVERINGS:
                continue
            if (hi[0] - lo[0]) * (hi[1] - lo[1]) > 20.0:
                continue  # house-sized AABBs (merged walls, roof, ceilings) say nothing; the floor test handles walls
            if lo[2] > ROBOT_HEIGHT:
                continue  # entirely above the robot (roof, lamps)
            if hi[2] - lo[2] < 0.08 and lo[2] < 0.05:
                continue  # flat floor coverings (pavers, rugs, mats) are stood on, not avoided
            if lo[0] < x + r and hi[0] > x - r and lo[1] < y + r and hi[1] > y - r and hi[2] > 0.05:
                return False, f"overlaps {obj.name}"
        return True, "free"

    def place_robot_near(self, support: str, side: str = "auto", standoff: float = 0.30, ignore_names=()) -> dict:
        """Put the robot next to a piece of furniture, facing it ("navigation done" stand-in).

        side: -x/+x/-y/+y = which side of the furniture's AABB the robot stands on; auto = first free one.
        ignore_names: objects that do not count as obstacles (e.g. spawned presets still parked in the air).
        """
        sup = self.scene_object(support)
        ignore = (sup, *[self.scene_object(n) for n in ignore_names])
        lo, hi = [v.cpu().numpy() for v in sup.aabb]
        c = (lo + hi) / 2
        d = standoff + ROBOT_FOOTPRINT
        cands = {
            "-x": (lo[0] - d, c[1], 0.0),
            "+x": (hi[0] + d, c[1], math.pi),
            "-y": (c[0], lo[1] - d, math.pi / 2),
            "+y": (c[0], hi[1] + d, -math.pi / 2),
        }
        for s in [side] if side != "auto" else ["-x", "+x", "-y", "+y"]:
            x, y, yaw = cands[s]
            ok, why = self._footprint_free(x, y, ignore=ignore)
            log.info(f"candidate {s} side of {support} at ({x:.2f}, {y:.2f}): {why}")
            if side == "auto" and not ok:
                continue
            return self.place_robot(x, y, yaw, note=f"{s} side of {support}")
        raise RuntimeError(f"no free side around {support}; pass --side or --robot-pose")

    def xy_radius(self, name: str) -> float:
        """Circumscribed xy radius of a scene object, for keeping its edges inside the head camera's view."""
        obj = self.scene_object(name)
        lo, hi = obj.aabb
        ex, ey = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        return 0.5 * math.hypot(ex, ey)

    def best_base_pose(
        self, p_item_xy, p_target_xy, ignore=(), reach: float = 0.9, aabbs=None, half_widths=(0.0, 0.0)
    ) -> tuple[tuple | None, dict]:
        """Best base pose with two points (world xy) both ahead and to the left, within the left arm's reach.

        Candidates on rings around the two points' midpoint, facing it; scored by the farther point's distance and
        how far left both are (lower is better), rejected when a point is behind the robot, well to its right, out of
        reach, outside the head camera's view, or the footprint is not free (``ignore``: objects that do not count;
        ``aabbs``: a scene_aabbs() snapshot to reuse across searches, taken here otherwise).

        ``half_widths``: each point's xy radius, so the view test can keep the object's *edges* in frame and not just
        its centre; 0 (the default) reproduces the point test for callers that pass bare positions.
        Returns ((score, x, y, yaw, dist, side) or None, rejection counts by reason).
        """
        p_item, p_target = np.asarray(p_item_xy)[:2], np.asarray(p_target_xy)[:2]
        mid = (p_item + p_target) / 2
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        best, rejected, footprint = None, {}, {}  # footprint: (x, y) -> _footprint_free result (yaw-independent)
        for radius in np.arange(0.35, 0.95, 0.05):
            for angle in np.arange(0.0, 2 * np.pi, np.pi / 18):
                x, y = mid + radius * np.array([np.cos(angle), np.sin(angle)])
                for yaw_offset in np.arange(-np.pi / 3, np.pi / 3 + 1e-6, np.pi / 12):
                    yaw = np.arctan2(mid[1] - y, mid[0] - x) + yaw_offset
                    fwd, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
                    rel = [(p - np.array([x, y])) for p in (p_item, p_target)]
                    ahead = [float(r @ fwd) for r in rel]
                    side = [float(r @ left) for r in rel]
                    dist = [float(np.linalg.norm(r)) for r in rel]
                    # the torso can turn, so a little to the right is acceptable; well to the left is preferred;
                    # both must be inside the head camera's view (about +-45 deg of forward; the camera sees +-50)
                    if min(ahead) < 0.15 or min(side) < -0.3 or max(dist) > reach:
                        rejected["geometry"] = rejected.get("geometry", 0) + 1
                        continue
                    if min(dist) < CAMERA_MIN_DIST:  # below the head camera's frame (a cookie at 0.42 m: empty mask)
                        rejected["too close for the camera"] = rejected.get("too close for the camera", 0) + 1
                        continue
                    # a container nearer than the item and in line with it hides the item (empty mask): keep their
                    # bearings apart by the container's angular half-width plus a margin for the item
                    bearing = [math.atan2(sd, ah) for ah, sd in zip(ahead, side)]
                    apart = abs(bearing[0] - bearing[1])
                    if dist[1] < dist[0] + 0.05 and apart < math.atan(TARGET_HALF_WIDTH / dist[1]) + math.atan(
                        0.06 / dist[0]
                    ):
                        rejected["container hides the item"] = rejected.get("container hides the item", 0) + 1
                        continue
                    if max(abs(sd) / max(ah, 1e-6) for ah, sd in zip(ahead, side)) > 1.0:
                        rejected["outside camera view"] = rejected.get("outside camera view", 0) + 1
                        continue
                    # Prefer poses that keep each object's *edges* in frame, not just its centre. A mask cut by the
                    # image border reconstructs into a hull that runs past the real object, and the planner then
                    # places into that phantom part: on 2026-09-04 a basket whose centre sat at 43 deg had its edge
                    # at 61 deg, lost a third of its width off the left of the image, and the cookie was released
                    # 3 cm outside the rim. This is a penalty rather than a rejection because for some item/container
                    # pairs no pose frames both -- the item is then simply out of reach of a single base pose (see
                    # README, "what stands between this and the full task").
                    clipped = sum(
                        max(0.0, abs(br) + math.atan2(hw, max(d, 1e-6)) - CAMERA_HALF_FOV)
                        for br, d, hw in zip(bearing, dist, half_widths)
                    )
                    score = (
                        max(dist)
                        + 0.5 * max(0.0, 0.15 - min(side))
                        + 0.1 * abs(yaw_offset)
                        + FRAMING_PENALTY * clipped
                    )
                    if best is None or score < best[0]:
                        key = (float(x), float(y))
                        if key not in footprint:
                            footprint[key] = self._footprint_free(x, y, ignore, aabbs=aabbs)
                        free, why = footprint[key]
                        if free:
                            best = (score, x, y, yaw, dist, side)
                        else:
                            rejected[why] = rejected.get(why, 0) + 1
        return best, rejected

    def place_robot_for(self, item: str, target: str, ignore_names=(), reach: float = 0.9) -> dict:
        """Stand where ITEM and TARGET are both in the left arm's reach ("navigation done"): best pose + place_robot.

        ignore_names: objects that do not count as obstacles, resolved like place_robot_near's (unknown names raise).
        """
        p_item = self.scene_object(item).aabb_center.cpu().numpy()[:2]
        p_target = self.scene_object(target).aabb_center.cpu().numpy()[:2]
        ignore = [self.scene_object(n) for n in ignore_names]
        half_widths = (self.xy_radius(item), self.xy_radius(target))
        best, rejected = self.best_base_pose(
            p_item, p_target, ignore=ignore, reach=reach, half_widths=half_widths
        )
        if best is None:
            raise RuntimeError(
                f"no base pose reaches both {item!r} and {target!r} within {reach} m (objects {np.round(p_item, 2)}, "
                f"{np.round(p_target, 2)}; rejections {dict(sorted(rejected.items(), key=lambda kv: -kv[1])[:6])})"
            )
        score, x, y, yaw, dist, side = best
        log.info(
            f"standing for {item} + {target}: ({x:.2f}, {y:.2f}) yaw {np.degrees(yaw):.0f} deg, "
            f"distances {np.round(dist, 2).tolist()} m, left offsets {np.round(side, 2).tolist()} m"
        )
        return self.place_robot(float(x), float(y), float(yaw), note=f"stand for {item} + {target}")

    def place_robot(self, x: float, y: float, yaw: float, note: str = "") -> dict:
        quat = T.euler2quat(th.tensor([0.0, 0.0, float(yaw)]))
        self.robot.set_position_orientation(position=th.tensor([x, y, 0.0]), orientation=quat)
        self.robot.keep_still()
        if not gm.HEADLESS:
            eye = (x - 2.2 * math.cos(yaw) - 1.0 * math.sin(yaw), y - 2.2 * math.sin(yaw) + 1.0 * math.cos(yaw), 2.0)
            target = (x + 0.8 * math.cos(yaw), y + 0.8 * math.sin(yaw), 0.8)
            og.sim.viewer_camera.set_position_orientation(
                position=th.tensor(eye), orientation=th.tensor(look_at_quat_xyzw(eye, target))
            )
        log.info(f"robot placed at ({x:.2f}, {y:.2f}) yaw {math.degrees(yaw):.0f} deg {note}")
        return {"x": float(x), "y": float(y), "yaw": float(yaw)}

    def apply_posture(self, locked: dict, q_home, settle_steps: int = 30, tol: float = 0.03, joint_names=None) -> None:
        """Hold the joints the planner locks (right arm, fingers, torso if not planned) and go to q_home.

        joint_names: the planner's joint order (embodiment metadata), e.g. torso_joint1..4 + left_arm_joint1..7; the
        simulator's planned-joint vector (q_arm / actions / sim_state) follows that order from here on.
        """
        unknown = [j for j in list(locked) + list(joint_names or []) if j not in self.joint_index]
        if unknown:
            raise ValueError(f"joints unknown to the simulator: {unknown}")
        if joint_names:
            self.planned_joints = list(joint_names)
            self.arm_idx = th.tensor([self.joint_index[j] for j in self.planned_joints])
        assert len(q_home) == len(self.planned_joints), (len(q_home), self.planned_joints)
        self.posture = {j: float(v) for j, v in locked.items()}
        self.q_home = [float(v) for v in q_home]
        q = self.robot.get_joint_positions().clone()
        for j, v in self.posture.items():
            q[self.joint_index[j]] = v
        for j, v in zip(self.planned_joints, self.q_home):
            q[self.joint_index[j]] = v
        self.robot.set_joint_positions(q, drive=False)
        self.robot.keep_still()
        self.hold(settle_steps, self.OPEN, q_arm=self.q_home)
        now = self.robot.get_joint_positions()
        errs = {j: float(abs(now[self.joint_index[j]] - v)) for j, v in self.posture.items() if "finger" not in j}
        worst = max(errs, key=errs.get)
        base_z = float(self.robot.get_position_orientation()[0][2])
        cam_z = float(self.robot_cam.get_position_orientation()[0][2])
        log.info(
            f"posture applied: worst locked-joint error {errs[worst]:.4f} rad on {worst}; base z {base_z:.3f}, camera z {cam_z:.3f}"
        )
        if cam_z < 0.8:
            raise RuntimeError(f"robot is not upright (camera at z={cam_z:.2f} m); check base mass / posture")
        if errs[worst] > tol:
            raise RuntimeError(
                f"simulator does not hold the planner's locked posture: {worst} off by {errs[worst]:.3f} rad"
            )

    # ---------------------------------------------------------------- observation
    def capture(self, task: str, gt_labels=None, gt_atoms=None) -> tuple[dict, dict]:
        """Look with the arm out of the head camera's view, then return to the ready posture the plan starts from."""
        if self.look_arm is None:
            return super().capture(task, gt_labels=gt_labels, gt_atoms=gt_atoms)
        ready = list(self.q_home)
        unknown = set(self.look_arm) - set(self.planned_joints)
        assert not unknown, f"look posture names joints the planner does not move: {unknown}"
        look = [float(self.look_arm.get(j, v)) for j, v in zip(self.planned_joints, ready)]
        self.hold(LOOK_SETTLE_STEPS, self.OPEN, q_arm=look)
        request, extras = super().capture(task, gt_labels=gt_labels, gt_atoms=gt_atoms)
        self.hold(LOOK_SETTLE_STEPS, self.OPEN, q_arm=ready)
        q_ready = self.q_arm()
        lag = float(np.abs(q_ready - np.asarray(ready)).max())
        if lag > 0.03:
            raise RuntimeError(f"arm did not return to the ready posture after the capture (max error {lag:.3f} rad)")
        request["q_init"] = np.asarray(q_ready, dtype=np.float32)  # the plan starts here, not at the look posture
        extras["q_look"] = [float(v) for v in look]
        log.info(f"captured in the look posture; plan starts from the ready posture (max error {lag:.4f} rad)")
        return request, extras

    # ---------------------------------------------------------------- stepping
    def action(self, q_arm, gripper: float) -> dict:
        """23-D action: planned joints (torso + left arm, by name) to their targets, locked joints to their values."""
        targets = dict(self.posture)
        targets.update(zip(self.planned_joints, np.asarray(q_arm, dtype=np.float32).tolist()))
        idx = self.robot.controller_action_idx
        a = th.zeros(self.robot.action_dim, dtype=th.float32)
        a[idx["trunk"]] = th.tensor([targets[j] for j in self.robot.trunk_joint_names], dtype=th.float32)
        a[idx["arm_left"]] = th.tensor([targets[j] for j in self.robot.arm_joint_names["left"]], dtype=th.float32)
        a[idx["gripper_left"]] = float(gripper)
        a[idx["arm_right"]] = th.tensor([targets[j] for j in self.robot.arm_joint_names["right"]], dtype=th.float32)
        a[idx["gripper_right"]] = self.OPEN
        # base: HolonomicBaseJointController in position mode takes deltas, zeros hold the base still
        return {self.robot.name: a}

    def _capture_obs(self) -> tuple[dict, dict]:
        """Move the shadow camera onto the robot camera and render one rgb + depth + segmentation frame."""
        pos, quat = self.robot_cam.get_position_orientation()
        self.cam.set_position_orientation(position=pos, orientation=quat)
        # The renderer accumulates frames over time: after the camera jumps (base teleport, look posture) the first
        # frames are a ghost of the previous view, so render until two consecutive frames agree.
        previous = None
        for i in range(CAPTURE_MAX_RENDERS):
            og.sim.render()
            og.sim.render()
            rgb = self.cam.get_obs()[0]["rgb"][..., :3].to(th.float32)
            if previous is not None and float((rgb - previous).abs().mean()) < CAPTURE_CONVERGED_DIFF:
                log.info(f"capture converged after {2 * (i + 1)} renders")
                break
            previous = rgb
        else:
            log.warning(f"capture did not converge after {2 * CAPTURE_MAX_RENDERS} renders; using the last frame")
        k_robot, k_shadow = self.robot_cam.intrinsic_matrix.cpu().numpy(), self.cam.intrinsic_matrix.cpu().numpy()
        if not np.allclose(k_robot, k_shadow, atol=0.5):
            raise RuntimeError(f"shadow camera intrinsics {k_shadow.tolist()} != robot camera {k_robot.tolist()}")
        p2, q2 = self.cam.get_position_orientation()
        assert th.allclose(p2, pos, atol=1e-4) and th.allclose(
            q2.abs(), quat.abs(), atol=1e-4
        ), "shadow camera did not move"
        return self.cam.get_obs()

    def camera_rgb(self) -> np.ndarray | None:
        if self.last_obs is None or self.robot.name not in self.last_obs:
            return None
        return self.last_obs[self.robot.name][self.cam_name]["rgb"][..., :3].cpu().numpy().astype(np.uint8)
