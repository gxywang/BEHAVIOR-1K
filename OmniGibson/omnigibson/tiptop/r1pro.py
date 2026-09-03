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
HEAD_APERTURE_MM = 40.0  # BEHAVIOR challenge eval setting (99 deg HFOV); OmniGibson's default 20.995 gives 63 deg
WRIST_APERTURE_MM = 20.995  # OmniGibson VisionSensor default, set explicitly so the shadow camera matches exactly
ROBOT_FOOTPRINT = 0.36  # half extent (m) used for free-space checks; base bbox is 0.64 x 0.68
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
) -> dict:
    """OmniGibson config: BEHAVIOR scene + R1Pro with absolute joint controllers on every group.

    Spawned objects start high above the floor and are placed onto furniture by R1ProSim.place_on().

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
        "modalities": ["rgb", "depth_linear", "seg_instance"],
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
        "task": {"type": "DummyTask"},
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

    # ---------------------------------------------------------------- scene setup
    def scene_object(self, name: str):
        obj = self.env.scene.object_registry("name", name)
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

    def _footprint_free(self, x: float, y: float, ignore) -> tuple[bool, str]:
        """Floor under the whole footprint, inside a room, and no other object's AABB overlapping the footprint."""
        r = ROBOT_FOOTPRINT
        corners = [(x + sx * r, y + sy * r) for sx in (-1, 1) for sy in (-1, 1)] + [(x, y)]
        floors = [o for o in self.env.scene.objects if getattr(o, "category", "") == "floors"]
        for cx, cy in corners:
            on_floor = False
            for f in floors:
                lo, hi = [v.cpu().numpy() for v in f.aabb]
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
        for obj in self.env.scene.objects:
            if obj is self.robot or obj in ignore or getattr(obj, "category", "") in ("floors", "ceilings"):
                continue
            lo, hi = [v.cpu().numpy() for v in obj.aabb]
            if getattr(obj, "category", "") == "walls" and (hi[0] - lo[0]) * (hi[1] - lo[1]) > 20.0:
                continue  # merged whole-house wall objects have a useless AABB; the floor test handles walls
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
        for _ in range(4):
            og.sim.render()
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
