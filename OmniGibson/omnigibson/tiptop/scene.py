"""Tabletop scene with a Franka Panda for TiPToP, plus observation capture in TiPToP's conventions.

Conventions handled here (the parts that silently break a TiPToP integration):
- TiPToP's world frame is the ROBOT BASE frame (cuRobo base_link ``panda_link0``); all poses are expressed there.
- TiPToP expects an OpenCV camera (+x right, +y down, +z forward) while OmniGibson/USD cameras look down -z with +y up.
- TiPToP expects z-depth (distance to the image plane) = OmniGibson ``depth_linear``, not ``depth`` (ray length).
- OmniGibson quaternions are (x, y, z, w); the droid H5 layout stores (w, x, y, z).
"""

import logging

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.tiptop.protocol import DROID_CAMERA_KWARGS, DROID_Q_INIT, build_request, depth_to_points

log = logging.getLogger(__name__)

TABLE_HEIGHT = 0.75  # world z of the table top; the robot base sits on it, so base-frame z = 0 there
FINGER_OPEN = 0.04
CAMERA_NAME = "tiptop_cam"
VIEWER_EYE = (1.9, -1.6, 1.7)  # Isaac Sim viewport pose in GUI mode, world frame
VIEWER_TARGET = (0.45, 0.0, 0.8)

# Objects with ground-truth instance segmentation; names double as the labels used in goal atoms.
OBJECT_PRESETS = {
    "mug": {
        "type": "DatasetObject",
        "category": "mug",
        "model": "ycbmug",
        "position": [0.50, 0.17, TABLE_HEIGHT + 0.06],
    },
    "bowl": {
        "type": "DatasetObject",
        "category": "bowl",
        "model": "ycbbwl",
        "position": [0.56, -0.15, TABLE_HEIGHT + 0.04],
    },
    "apple": {
        "type": "DatasetObject",
        "category": "apple",
        "model": "agveuv",
        "position": [0.42, -0.02, TABLE_HEIGHT + 0.05],
    },
    "banana": {
        "type": "DatasetObject",
        "category": "banana",
        "model": "verqwv",
        "position": [0.62, 0.05, TABLE_HEIGHT + 0.04],
    },
}


def look_at_quat_xyzw(eye, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """Orientation (x, y, z, w) of a USD/OpenGL camera at ``eye`` looking at ``target`` (camera -z axis = view dir)."""
    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight along `up`: pick any horizontal axis as the image x axis
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    rot = np.stack([right, cam_up, -forward], axis=1)
    return T.mat2quat(th.tensor(rot, dtype=th.float32)).tolist()


def make_env_config(
    objects=("mug", "bowl"),
    q_init=DROID_Q_INIT,
    grasping_mode: str = "physical",
    camera_pos=(0.95, -0.65, TABLE_HEIGHT + 0.70),
    camera_target=(0.50, 0.0, TABLE_HEIGHT),
    camera_kwargs: dict | None = None,
    table_height: float = TABLE_HEIGHT,
) -> dict:
    """OmniGibson config: empty scene, table, Franka Panda (base at the table height), external RGB-D camera."""
    object_cfgs = []
    for name in objects:
        if name not in OBJECT_PRESETS:
            raise ValueError(f"unknown object preset {name!r}; known: {sorted(OBJECT_PRESETS)}")
        object_cfgs.append({"name": name, **OBJECT_PRESETS[name]})
    return {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
            "external_sensors": [
                {
                    "sensor_type": "VisionSensor",
                    "name": CAMERA_NAME,
                    "relative_prim_path": f"/{CAMERA_NAME}",
                    "modalities": ["rgb", "depth_linear", "seg_instance"],
                    "sensor_kwargs": dict(DROID_CAMERA_KWARGS, **(camera_kwargs or {})),
                    "position": list(camera_pos),
                    "orientation": look_at_quat_xyzw(camera_pos, camera_target),
                    "include_in_obs": True,
                }
            ],
        },
        "scene": {"type": "Scene", "use_floor_plane": True, "floor_plane_visible": True},
        "robots": [
            {
                "model": "franka",
                "name": "robot0",
                "end_effector": "gripper",
                "position": [0.0, 0.0, table_height],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "obs_modalities": ["proprio"],
                "action_normalize": False,
                "grasping_mode": grasping_mode,
                "self_collisions": True,
                "reset_joint_pos": [float(q) for q in q_init] + [FINGER_OPEN, FINGER_OPEN],
                "controller_config": {
                    "arm_0": {
                        "name": "JointController",
                        "motor_type": "position",
                        "use_delta_commands": False,
                        "use_impedances": False,
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                    "gripper_0": {
                        "name": "MultiFingerGripperController",
                        "mode": "binary",
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                },
            }
        ],
        "objects": [
            {
                "type": "PrimitiveObject",
                "name": "table",
                "primitive_type": "Cube",
                "fixed_base": True,
                "size": 1.0,
                "scale": [1.4, 1.4, 0.05],
                "rgba": [0.62, 0.52, 0.40, 1.0],
                "position": [0.30, 0.0, table_height - 0.025],
            },
            *object_cfgs,
        ],
        "task": {"type": "DummyTask"},
    }


class TiptopSim:
    """Owns the OmniGibson environment and produces TiPToP observations / executes joint targets."""

    OPEN, CLOSE = 1.0, -1.0  # MultiFingerGripperController binary: command >= 0 opens, < 0 closes

    def __init__(self, config: dict):
        self.config = config
        self.env = og.Environment(configs=config)
        if not gm.HEADLESS:  # GUI: aim the Isaac Sim viewport at the table (the TiPToP camera is a separate sensor)
            og.sim.viewer_camera.set_position_orientation(
                position=th.tensor(VIEWER_EYE), orientation=th.tensor(look_at_quat_xyzw(VIEWER_EYE, VIEWER_TARGET))
            )
        self.robot = self.env.robots[0]
        self.cam = self.env.external_sensors[CAMERA_NAME]
        arm = self.robot.default_arm
        self.arm_idx = self.robot.arm_control_idx[arm]
        self.gripper_idx = self.robot.gripper_control_idx[arm]
        self.dt = og.sim.get_sim_step_dt()
        self.objects = {name: self.env.scene.object_registry("name", name) for name in self.object_names()}
        self.last_obs = None
        joint_names = list(self.robot.joints.keys())
        log.info(
            f"robot DOF order: {joint_names}; arm idx {self.arm_idx.tolist()}, gripper idx {self.gripper_idx.tolist()}"
        )
        assert list(self.robot.controller_order) == ["arm_0", "gripper_0"], self.robot.controller_order
        assert self.robot.action_dim == len(self.arm_idx) + 1, self.robot.action_dim
        self.env.reset()

    def set_finger_max_effort(self, effort_n: float) -> None:
        """Raise the finger drive force (the Franka USD ships 20 N; the real hand delivers 70 N continuous)."""
        for joint_name in self.robot.finger_joint_names[self.robot.default_arm]:
            self.robot.joints[joint_name].max_effort = float(effort_n)
        log.info(f"finger max effort set to {effort_n} N")

    def object_names(self) -> list[str]:
        return [o["name"] for o in self.config["objects"] if o["name"] != "table"]

    # ---------------------------------------------------------------- state
    def q_arm(self) -> np.ndarray:
        return self.robot.get_joint_positions()[self.arm_idx].cpu().numpy().astype(np.float32)

    def q_fingers(self) -> np.ndarray:
        return self.robot.get_joint_positions()[self.gripper_idx].cpu().numpy().astype(np.float32)

    def base_pose(self):
        return self.robot.get_position_orientation()

    def to_base(self, pos, quat):
        """Express a world pose (xyzw quaternion) in the robot base frame."""
        base_pos, base_quat = self.base_pose()
        return T.relative_pose_transform(
            th.as_tensor(pos, dtype=th.float32), th.as_tensor(quat, dtype=th.float32), base_pos, base_quat
        )

    def object_poses_world(self) -> dict:
        return {name: [p.tolist() for p in obj.get_position_orientation()] for name, obj in self.objects.items()}

    def apply_object_poses(self, poses: dict) -> None:
        for name, (pos, quat) in poses.items():
            obj = self.objects[name]
            obj.set_position_orientation(position=th.tensor(pos), orientation=th.tensor(quat))
            obj.keep_still()

    # ---------------------------------------------------------------- stepping
    def action(self, q_arm, gripper: float) -> dict:
        a = th.cat([th.as_tensor(np.asarray(q_arm, dtype=np.float32)), th.tensor([float(gripper)])])
        return {self.robot.name: a}

    def step(self, q_arm, gripper: float):
        self.last_obs = self.env.step(self.action(q_arm, gripper))[0]
        return self.last_obs

    def hold(self, n_steps: int, gripper: float = OPEN, q_arm=None):
        q = self.q_arm() if q_arm is None else q_arm
        for _ in range(n_steps):
            self.step(q, gripper)

    def camera_rgb(self) -> np.ndarray | None:
        if self.last_obs is None or "external" not in self.last_obs:
            return None
        return self.last_obs["external"][CAMERA_NAME]["rgb"][..., :3].cpu().numpy().astype(np.uint8)

    # ---------------------------------------------------------------- observation
    def capture(self, task: str, gt_labels=None, gt_atoms=None) -> tuple[dict, dict]:
        """Render and assemble a TiPToP request (plus extras for H5/validation) in the robot base frame."""
        for _ in range(3):
            og.sim.render()
        obs, info = self.cam.get_obs()
        rgb = obs["rgb"][..., :3].cpu().numpy().astype(np.uint8)
        depth = obs["depth_linear"].cpu().numpy().astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0.0
        seg = obs["seg_instance"].cpu().numpy()
        id_to_name = {int(k): str(v) for k, v in info["seg_instance"].items()}
        intrinsics = self.cam.intrinsic_matrix.cpu().numpy().astype(np.float32)

        cam_pos, cam_quat = self.cam.get_position_orientation()  # world, USD camera axes
        cam_quat_cv = T.quat_multiply(cam_quat, th.tensor([1.0, 0.0, 0.0, 0.0]))  # 180 deg about camera x -> OpenCV
        cam_pos_b, cam_quat_b = self.to_base(cam_pos, cam_quat_cv)
        world_from_cam = T.pose2mat((cam_pos_b, cam_quat_b)).cpu().numpy().astype(np.float32)

        gt = None
        if gt_labels:
            masks = np.stack([self.instance_mask(seg, id_to_name, label) for label in gt_labels])
            gt = {"labels": list(gt_labels), "masks": masks, "atoms": list(gt_atoms or [])}
        request = build_request(rgb, depth, intrinsics, world_from_cam, task, self.q_arm(), gt=gt)

        object_poses_base = {}
        for name, obj in self.objects.items():
            pos_b, quat_b = self.to_base(*obj.get_position_orientation())
            aabb_center_b, _ = self.to_base(obj.aabb_center, th.tensor([0.0, 0.0, 0.0, 1.0]))
            object_poses_base[name] = {
                "pos": pos_b.tolist(),
                "quat_xyzw": quat_b.tolist(),
                "aabb_center": aabb_center_b.tolist(),
            }
        extras = {
            "cam_pos_base": cam_pos_b.cpu().numpy().tolist(),
            "cam_quat_wxyz_ros": T.convert_quat(cam_quat_b, to="wxyz").cpu().numpy().tolist(),
            "cam_pos_world": cam_pos.cpu().numpy().tolist(),
            "cam_quat_xyzw_world_usd": cam_quat.cpu().numpy().tolist(),
            "base_pos_world": self.base_pose()[0].cpu().numpy().tolist(),
            "base_quat_xyzw_world": self.base_pose()[1].cpu().numpy().tolist(),
            "seg_instance": seg,
            "id_to_name": id_to_name,
            "object_poses_base": object_poses_base,
            "object_poses_world": self.object_poses_world(),
            "q_fingers": self.q_fingers().tolist(),
            "sim_dt": self.dt,
        }
        return request, extras

    @staticmethod
    def instance_mask(seg: np.ndarray, id_to_name: dict, label: str) -> np.ndarray:
        ids = [i for i, name in id_to_name.items() if name == label]
        if not ids:
            raise ValueError(
                f"object {label!r} is not visible in the instance segmentation (labels: {sorted(set(id_to_name.values()))})"
            )
        return np.isin(seg, ids)

    def validate_capture(self, request: dict, extras: dict) -> dict:
        """Numerically check the frame conventions: table at base z=0, object mask centroids near their true poses."""
        pts = depth_to_points(request["depth"], request["intrinsics"], request["world_from_cam"])
        seg, id_to_name = extras["seg_instance"], extras["id_to_name"]
        report = {"camera_view_axis_base": request["world_from_cam"][:3, 2].tolist()}
        table_ids = [i for i, n in id_to_name.items() if n == "table"]
        table_pts = pts[np.isin(seg, table_ids)] if table_ids else np.zeros((0, 3))
        table_pts = table_pts[np.isfinite(table_pts).all(axis=1)]
        if len(table_pts):
            report["table_z_base_median"] = float(np.median(table_pts[:, 2]))
            report["table_z_base_p05_p95"] = [
                float(np.percentile(table_pts[:, 2], 5)),
                float(np.percentile(table_pts[:, 2], 95)),
            ]
        else:
            report["table_z_base_median"] = float("nan")
        for name, pose in extras["object_poses_base"].items():
            mask = self.instance_mask(seg, id_to_name, name)
            obj_pts = pts[mask]
            obj_pts = obj_pts[np.isfinite(obj_pts).all(axis=1)]
            centroid = obj_pts.mean(axis=0)
            err = centroid - np.asarray(pose["aabb_center"])
            report[f"{name}_visible_centroid_base"] = centroid.tolist()
            report[f"{name}_aabb_center_base"] = pose["aabb_center"]
            report[f"{name}_centroid_error_xy_m"] = float(np.linalg.norm(err[:2]))
            report[f"{name}_pixels"] = int(mask.sum())
        problems = []
        table_z = report["table_z_base_median"]
        if not np.isfinite(table_z):
            problems.append("table is not visible in the instance segmentation (no valid depth pixels)")
        elif abs(table_z) > 0.02:
            problems.append(f"table top is at base z={table_z:.3f} m, expected ~0")
        for name in extras["object_poses_base"]:
            if report[f"{name}_centroid_error_xy_m"] > 0.06:
                problems.append(
                    f"{name} mask centroid is {report[f'{name}_centroid_error_xy_m']:.3f} m from its true xy"
                )
        if report["camera_view_axis_base"][2] > -0.2:
            problems.append("camera optical axis is not pointing downward in the base frame")
        report["problems"] = problems
        return report
