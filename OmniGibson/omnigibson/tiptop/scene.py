"""Tabletop scene with a Franka Panda for TiPToP, plus observation capture in TiPToP's conventions.

Conventions handled here (the parts that silently break a TiPToP integration):
- TiPToP's world frame is the ROBOT BASE frame (cuRobo base_link ``panda_link0``); all poses are expressed there.
- TiPToP expects an OpenCV camera (+x right, +y down, +z forward) while OmniGibson/USD cameras look down -z with +y up.
- TiPToP expects z-depth (distance to the image plane) = OmniGibson ``depth_linear``, not ``depth`` (ray length).
- OmniGibson quaternions are (x, y, z, w); the droid H5 layout stores (w, x, y, z).

The simulator also mirrors itself into the planner's Rerun view (``state_stream``, see client.SimStateStream): its
own object meshes under their task names, the planned joints, the finger opening and two camera images, a few times
per simulated second, from ``step``.
"""

import io
import logging
import time
from contextlib import contextmanager

import numpy as np
import torch as th
import trimesh

from b1k.bridge.kinematics import look_at_quat_xyzw as _look_at_quat_xyzw
from b1k.bridge.protocol import (
    DROID_CAMERA_KWARGS,
    DROID_Q_INIT,
    add_view,
    build_request,
    canonical_object_name,
    capture_views,
    depth_to_points,
    points_to_pixels,
    rerun_name,
)

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.tiptop.gt_masks import masks_from_geometry, meshes_at_view_poses
from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh

log = logging.getLogger(__name__)

TABLE_HEIGHT = 0.75  # world z of the table top; the robot base sits on it, so base-frame z = 0 there
FINGER_OPEN = 0.04
FINGER_CONTACT = 0.006  # m between the fingers when closed: more than this and they stopped on an object
CAMERA_NAME = "tiptop_cam"
OVERVIEW_CAM = "overview_cam"  # third-person rgb camera for the Rerun mirror, aimed at the workspace
OVERVIEW_SIZE = (640, 360)
VIEWER_EYE = (1.9, -1.6, 1.7)  # Isaac Sim viewport / overview camera pose in the Panda scene, world frame
VIEWER_TARGET = (0.45, 0.0, 0.8)
STREAM_MAX_FACES = 4000  # meshes sent to the Rerun mirror are decimated to this many triangles
STREAM_IMAGE_MAX_PX = 480  # longest side of the JPEGs in the mirror
STREAM_JPEG_QUALITY = 75

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


def decimated(vertices, faces, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """The mesh reduced to about ``target_faces`` triangles (open3d's quadric decimation)."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    ).simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def look_at_quat_xyzw(eye, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """``kinematics.look_at_quat_xyzw`` as a list (sensor configs and tensors take it as is)."""
    return _look_at_quat_xyzw(eye, target, up).tolist()


def overview_cam_config(eye=(0.0, 0.0, 3.0), target=(1.0, 0.0, 0.0)) -> dict:
    """External rgb sensor for the Rerun mirror; not in the observation, read on demand by ``stream_images``."""
    return {
        "sensor_type": "VisionSensor",
        "name": OVERVIEW_CAM,
        "relative_prim_path": f"/{OVERVIEW_CAM}",
        "modalities": ["rgb"],
        "sensor_kwargs": {
            "image_width": OVERVIEW_SIZE[0],
            "image_height": OVERVIEW_SIZE[1],
            "focal_length": 17.0,
            "horizontal_aperture": 40.0,
        },
        "position": list(eye),
        "orientation": look_at_quat_xyzw(eye, target),
        "include_in_obs": False,
    }


def jpeg_bytes(rgb: np.ndarray, max_px: int = STREAM_IMAGE_MAX_PX, quality: int = STREAM_JPEG_QUALITY) -> bytes:
    """(H, W, 3) uint8 -> JPEG, downscaled so the longer side is at most ``max_px``."""
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
    if max(img.size) > max_px:
        scale = max_px / max(img.size)
        img = img.resize((max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


CAPTURE_CAMERA_EYE = (0.95, -0.65, TABLE_HEIGHT + 0.70)  # the DROID-style external capture camera, world frame
CAPTURE_CAMERA_TARGET = (0.50, 0.0, TABLE_HEIGHT)


def make_env_config(objects=("mug", "bowl"), grasping_mode: str = "physical") -> dict:
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
                    "sensor_kwargs": dict(DROID_CAMERA_KWARGS),
                    "position": list(CAPTURE_CAMERA_EYE),
                    "orientation": look_at_quat_xyzw(CAPTURE_CAMERA_EYE, CAPTURE_CAMERA_TARGET),
                    "include_in_obs": True,
                },
                overview_cam_config(VIEWER_EYE, VIEWER_TARGET),
            ],
        },
        "scene": {"type": "Scene", "use_floor_plane": True, "floor_plane_visible": True},
        "robots": [
            {
                "model": "franka",
                "name": "robot0",
                "end_effector": "gripper",
                "position": [0.0, 0.0, TABLE_HEIGHT],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "obs_modalities": ["proprio"],
                "action_normalize": False,
                "grasping_mode": grasping_mode,
                "self_collisions": True,
                "reset_joint_pos": [float(q) for q in DROID_Q_INIT] + [FINGER_OPEN, FINGER_OPEN],
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
                "position": [0.30, 0.0, TABLE_HEIGHT - 0.025],
            },
            *object_cfgs,
        ],
        "task": {"type": "DummyTask"},
    }


class EpisodeOver(Exception):
    """The task reported the episode done (its goal holds, or its step limit passed) while the pipeline was still
    stepping; raised from ``TiptopSim.step`` when ``stop_when_done`` is set, so a benchmark scores at that step the
    way the challenge evaluator stops there. ``reason`` is "success", "timeout" or "terminated"."""

    def __init__(self, reason: str, steps: int):
        super().__init__(f"episode over after {steps} steps: {reason}")
        self.reason, self.steps = reason, steps


class TiptopSim:
    """Owns the OmniGibson environment and produces TiPToP observations / executes joint targets."""

    OPEN, CLOSE = 1.0, -1.0  # MultiFingerGripperController binary: command >= 0 opens, < 0 closes
    expect_table_z = 0.0  # validate_capture: expected table top height in the base frame (None = no table check)
    mask_labels_as_invalid = ()  # instance labels whose pixels get depth 0 (e.g. the robot seen by its own camera)
    # validate_capture: warn below this fraction of an object's projected AABB being inside the image. Measured on
    # the gift-basket task: 0.21 for the clipped basket that made the planner drop the cookie outside it, 0.72 for
    # the same basket framed in the round that succeeded. The AABB over-estimates the silhouette, so keep it low.
    FRAME_COVERAGE_MIN = 0.5
    gt_mask_tol = 0.008  # geometry masks: surface distance (m) within which a depth pixel belongs to an object
    STREAM_CAMERA = "cam"  # name of the capture camera's image in the Rerun mirror
    # The base-frame box the planner works in (its crop of every view): the tabletop ahead of the robot, as TiPToP's
    # sim config has it; ``workspace`` lowers it to the floor for a container standing there. An embodiment moves the
    # near edge past its own footprint (R1ProSim).
    WORKSPACE = ((0.05, -0.80, 0.25), (1.30, 0.80, 1.60))
    FLOOR_Z = -0.05

    def __init__(self, config: dict):
        self.config = config
        self.env = og.Environment(configs=config)
        if not gm.HEADLESS:  # GUI: aim the Isaac Sim viewport at the table (the TiPToP camera is a separate sensor)
            og.sim.viewer_camera.set_position_orientation(
                position=th.tensor(VIEWER_EYE), orientation=th.tensor(look_at_quat_xyzw(VIEWER_EYE, VIEWER_TARGET))
            )
        self.robot = self.env.robots[0]
        self.cam = self.env.external_sensors[CAMERA_NAME]
        self.overview = self.env.external_sensors.get(OVERVIEW_CAM)
        arm = self.robot.default_arm
        self.arm_idx = self.robot.arm_control_idx[arm]
        self.gripper_idx = self.robot.gripper_control_idx[arm]
        self.dt = og.sim.get_sim_step_dt()
        self.objects = {name: self.env.scene.object_registry("name", name) for name in self.object_names()}
        self.context = {"table": self.env.scene.object_registry("name", "table")}  # furniture shown in the mirror
        self.obstacles = {}  # label -> furniture sent to the planner as a static obstacle (see tracked_object)
        self._init_state()
        joint_names = list(self.robot.joints.keys())
        log.info(
            f"robot DOF order: {joint_names}; arm idx {self.arm_idx.tolist()}, gripper idx {self.gripper_idx.tolist()}"
        )
        assert list(self.robot.controller_order) == ["arm_0", "gripper_0"], self.robot.controller_order
        assert self.robot.action_dim == len(self.arm_idx) + 1, self.robot.action_dim
        self.env.reset()

    def _init_state(self) -> None:
        """Per-episode state shared by every embodiment (R1ProSim builds its own scene and calls this too)."""
        self.state_stream = None  # client.SimStateStream once attached; fed from step()
        self._link_meshes = {}  # (link prim path, max faces) -> its mesh in the link frame (link_trimesh_world)
        self.arm = self.robot.default_arm  # the arm the plans move (R1ProSim: the planner embodiment's arm)
        self.held_objects = {}  # tracked label -> arm, for objects a plan picked up (they move with that gripper)
        self.teleports = 0  # base teleports so far (the navigation stand-in; the benchmark reports the count)
        self._in_hand_object = None  # the robot's own grasp search while block_grasping() has it wrapped
        self.recorders = []  # executor.VideoRecorder instances, fed from step(); frame_caption() is stamped on each
        self.metrics = []  # omnigibson.metrics.MetricBase instances fed from step() (the challenge's own scoring)
        self.n_steps = 0  # env steps taken: holds, captures and plans all count, as they do for the challenge's timeout
        self.max_steps = None  # the episode's step limit when known (shown in the frame caption)
        self.episode_open = True  # env steps count (n_steps, metrics) until end_episode(); a video tail does not
        self.stop_when_done = False  # raise EpisodeOver when the task reports success or its step limit (benchmark)
        self.video_caption = None  # what the robot is doing now, one line, set by the driver (run.py, bench.py)
        self.last_obs = None
        self.last_gripper = self.OPEN
        self.last_capture_rgb = None  # the last frame capture() rendered, for saving next to an error
        self.capture_object_aabb_min_z = {}  # where each tracked object rested at the last capture
        self._stream_meshes = {}

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

    def base_to_world(self, pos_b) -> np.ndarray:
        """A base-frame point in the world frame (the inverse of ``to_base`` for positions)."""
        base_pos, base_quat = self.base_pose()
        rot = T.quat2mat(base_quat).cpu().numpy().astype(np.float64)
        return rot @ np.asarray(pos_b, dtype=np.float64) + base_pos.cpu().numpy().astype(np.float64)

    def object_poses_world(self) -> dict:
        return {name: [p.tolist() for p in obj.get_position_orientation()] for name, obj in self.objects.items()}

    def object_poses_base_mats(self) -> dict:
        """Base-frame 4x4 pose of every mirrored object (tracked objects and context furniture), by Rerun name."""
        mats = {}
        for name, obj in (*self.objects.items(), *self.context.items()):
            pos_b, quat_b = self.to_base(*obj.get_position_orientation())
            mats[rerun_name(name)] = T.pose2mat((pos_b, quat_b)).cpu().numpy().astype(np.float32)
        return mats

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
        self.last_gripper = float(gripper)
        action = self.action(q_arm, gripper)
        self.last_obs, reward, terminated, truncated, info = self.env.step(action)
        if self.episode_open:
            self.n_steps += 1
            for metric in self.metrics:
                metric.step(self.env, action[self.robot.name], self.last_obs, reward, terminated, truncated, info)
        if self.state_stream is not None:
            self.state_stream.on_step(self)
        if self.recorders:
            views, caption = None, self.frame_caption()
            for recorder in self.recorders:
                if recorder.due():
                    views = self.video_views() if views is None else views
                    recorder.write(views, caption)
        if self.stop_when_done and (terminated or truncated):
            success = bool((info or {}).get("done", {}).get("success", False))
            raise EpisodeOver("success" if success else "timeout" if truncated else "terminated", self.n_steps)
        return self.last_obs

    @contextmanager
    def recording(self, path):
        """Record every env step inside the block to the video at ``path`` (``executor.VideoRecorder``, fed from
        ``step``); the file is closed on the way out, whatever ended the block."""
        from b1k.bridge.executor import VideoRecorder

        recorder = VideoRecorder(path)
        self.recorders.append(recorder)
        try:
            yield recorder
        finally:
            self.recorders.remove(recorder)
            recorder.close()

    def frame_caption(self) -> str:
        """What is stamped on a video frame: the driver's ``video_caption`` line, then the env step count (over the
        episode's limit when known), so a viewer can tell where in the episode a frame is."""
        steps = f"step {self.n_steps}" + (f"/{self.max_steps}" if self.max_steps else "")
        return f"{self.video_caption}\n{steps}" if self.video_caption else steps

    def begin_episode(self, metrics=(), stop_when_done: bool = False, max_steps: int | None = None) -> None:
        """Start counting from zero for a scored episode: env steps, the metrics fed from ``step`` (each is reset on
        the environment first) and whether the task's own done signal ends the episode (``EpisodeOver``)."""
        self.n_steps = 0
        self.max_steps = max_steps
        self.held_objects, self.teleports, self.blocked_swings = {}, 0, 0
        self.metrics = list(metrics)
        for metric in self.metrics:
            metric.reset(self.env)
        self.stop_when_done = stop_when_done
        self.episode_open = True

    def end_episode(self) -> int:
        """The episode is over: env steps from here on (a video tail showing the final state) are neither counted
        nor scored and no longer raise ``EpisodeOver``. Returns the episode's step count."""
        self.episode_open = False
        self.stop_when_done = False
        return self.n_steps

    def workspace(self, floor: bool = False) -> list:
        """The box the planner works in for a request (base frame, [[x0, y0, z0], [x1, y1, z1]]): ``WORKSPACE``,
        reaching the floor when the target stands on it."""
        (x0, y0, z0), (x1, y1, z1) = self.WORKSPACE
        return [[x0, y0, self.FLOOR_Z if floor else z0], [x1, y1, z1]]

    def hands(self) -> dict:
        """{tracked label: arm} of what the hands hold now, by the robot's own record: a plan that closed a hand on
        an object with the fingers stopping on something (``grasp_sensed``) put it there, a plan that let go or a
        release took it out (``run.note_hands``). Never the simulator's grasp assist: that record is compared with
        this one in the log (``check_hands``) and steers nothing."""
        return dict(self.held_objects)

    def check_hands(self) -> None:
        """Log where the robot's own record and the simulator's grasp assist disagree (diagnostics only)."""
        grasped = self.grasped_labels()
        if grasped is not None and grasped != self.held_objects:
            log.warning(
                f"hand record {self.held_objects or 'empty'} differs from the grasp assist's {grasped or 'empty'}"
            )

    def finger_width(self, arm: str | None = None) -> float:
        """How far apart the fingers of ``arm`` (default: the planned arm) are, in metres: the sum of the finger
        joint positions (each 0 closed .. FINGER_OPEN open)."""
        order = list(self.robot.joints.keys())
        names = self.robot.finger_joint_names[arm or self.robot.default_arm]
        q = self.robot.get_joint_positions()
        return float(sum(float(q[order.index(j)]) for j in names))

    def gripper_command(self, arm: str | None = None) -> float:
        """The last gripper command sent to ``arm`` (OPEN or CLOSE): the planned arm's, or the other arm's kept
        command on a two-armed robot."""
        planned = getattr(self, "arm", None)
        if arm is None or arm == planned or planned is None:
            return self.last_gripper
        return getattr(self, "other_gripper", self.OPEN)

    def grasp_sensed(self, arm: str | None = None) -> bool:
        """Whether the hand of ``arm`` is closed on something, from the robot's own readings: the gripper was
        commanded closed and the fingers stopped more than ``FINGER_CONTACT`` apart (closed on nothing they meet)."""
        return self.gripper_command(arm) < 0 and self.finger_width(arm) > FINGER_CONTACT

    def grasped_labels(self) -> dict | None:
        """{tracked label: arm} of the tracked objects the robot's grasp assist holds right now (sticky or assisted
        grasping): what the robot knows it carries, read from its own gripper. None in physical grasping mode,
        where there is no such record (the robot keeps the dict, empty, in every mode)."""
        if self.robot.grasping_mode == "physical":
            return None
        by_obj = {obj: label for label, obj in self.objects.items()}
        return {
            by_obj[obj]: arm for arm, obj in self.robot._ag_obj_in_hand.items() if obj is not None and obj in by_obj
        }

    def mirror_q(self) -> np.ndarray:
        """Planned joints for the Rerun mirror (the embodiment the mirror was attached with; see R1ProSim)."""
        return self.q_arm()

    def mirror_fingers(self) -> np.ndarray:
        return self.q_fingers()

    def hold(self, n_steps: int, gripper: float = OPEN, q_arm=None):
        q = self.q_arm() if q_arm is None else q_arm
        for _ in range(n_steps):
            self.step(q, gripper)

    def eef_pose_base(self, arm: str | None = None) -> np.ndarray:
        """4x4 base-frame pose of the arm's end-effector link (the robot's default arm when ``arm`` is None)."""
        link = self.robot.eef_links[arm or self.robot.default_arm]
        pos, quat = self.to_base(*link.get_position_orientation())
        mat = np.eye(4)
        mat[:3, :3] = T.quat2mat(quat).cpu().numpy()
        mat[:3, 3] = pos.cpu().numpy()
        return mat

    def block_grasping(self, arm: str | None = None) -> None:
        """Keep OmniGibson's assisted/sticky grasp off ``arm`` (None: every arm) until ``unblock_grasping``: a closed
        gripper pressing a button must not pick up what it touches (a grasp starts after 0.3 s of finger contact
        while the gripper is commanded closed). Wraps the robot's per-arm candidate search; a grasp the arm already
        holds is not released."""
        robot = self.robot
        if self._in_hand_object is None:
            self._in_hand_object = robot._calculate_in_hand_object
        original = self._in_hand_object

        def blocked(*args, **kwargs):  # OmniGibson calls it as _calculate_in_hand_object(arm=arm)
            candidate = kwargs["arm"] if "arm" in kwargs else (args[0] if args else "default")
            if arm is None or candidate == arm:
                return None
            return original(*args, **kwargs)

        robot._calculate_in_hand_object = blocked
        log.info(f"grasping blocked for the {arm or 'whole robot'} (a press must not attach the button's object)")

    def unblock_grasping(self) -> None:
        if self._in_hand_object is not None:
            self.robot._calculate_in_hand_object = self._in_hand_object
            self._in_hand_object = None

    def camera_rgb(self) -> np.ndarray | None:
        if self.last_obs is None or "external" not in self.last_obs:
            return None
        return self.last_obs["external"][CAMERA_NAME]["rgb"][..., :3].cpu().numpy().astype(np.uint8)

    # ---------------------------------------------------------------- Rerun mirror
    def aim_overview(self, eye, target) -> None:
        if self.overview is not None:
            self.overview.set_position_orientation(
                position=th.tensor(eye, dtype=th.float32), orientation=th.tensor(look_at_quat_xyzw(eye, target))
            )

    def video_views(self) -> dict:
        """rgb views for the video and the mirror, {name: (H, W, 3) uint8}: the capture camera as the robot sees it
        (first), then the overview camera."""
        views = {}
        rgb = self.camera_rgb()
        if rgb is not None:
            views[self.STREAM_CAMERA] = rgb
        if self.overview is not None:
            over = self.overview.get_obs()[0].get("rgb")
            if over is not None and over.numel():
                views["overview"] = over.cpu().numpy().astype(np.uint8)[..., :3]
        return views

    def stream_images(self) -> dict:
        """JPEGs for the mirror: ``video_views``."""
        return {name: jpeg_bytes(rgb) for name, rgb in self.video_views().items()}

    def stream_scene(self) -> dict:
        """Every mirrored object's mesh in its own frame plus its current base-frame pose (see SimStateStream)."""
        t0 = time.time()
        poses = self.object_poses_base_mats()
        scene = {}
        for kind, group in (("object", self.objects), ("context", self.context)):
            for name, obj in group.items():
                key = rerun_name(name)
                if key not in self._stream_meshes:
                    try:
                        self._stream_meshes[key] = self.mesh_local(obj)
                    except Exception as e:  # noqa: BLE001 - a missing mesh only costs its picture in the viewer
                        log.warning(f"no mesh for {name!r} in the Rerun mirror ({e})")
                        self._stream_meshes[key] = None
                if self._stream_meshes[key] is None:
                    continue
                vertices, faces = self._stream_meshes[key]
                scene[key] = {"vertices": vertices, "faces": faces, "pose": poses[key], "kind": kind}
        log.info(
            f"Rerun mirror: {len(scene)} meshes, {sum(len(m['faces']) for m in scene.values())} triangles "
            f"({time.time() - t0:.1f}s)"
        )
        return scene

    def mesh_local(self, obj) -> tuple[np.ndarray, np.ndarray]:
        """(vertices (N, 3) f32, faces (M, 3) i32) of an object in its own frame, decimated to STREAM_MAX_FACES."""
        tm = self.trimesh_world(obj)
        pos, quat = obj.get_position_orientation()
        rot = T.quat2mat(quat).cpu().numpy().astype(np.float64)
        vertices = (np.asarray(tm.vertices, dtype=np.float64) - pos.cpu().numpy()) @ rot
        faces = np.asarray(tm.faces, dtype=np.int64)
        if len(faces) > STREAM_MAX_FACES:
            vertices, faces = decimated(vertices, faces, STREAM_MAX_FACES)
        return vertices.astype(np.float32), faces.astype(np.int32)

    # ---------------------------------------------------------------- observation
    primary_view = "cam"  # the capture camera's view name (R1ProSim: the --camera choice)
    extra_views = ()  # further views captured with it and fused by the planner (R1ProSim: the wrist cameras)

    def view_sensor(self, name: str):
        """The sensor a view is rendered through (its intrinsics and pose describe the view)."""
        return self.cam

    def _capture_obs(self, name: str) -> tuple[dict, dict]:
        """One rendered frame with rgb, depth_linear and seg_instance from the capture camera."""
        for _ in range(3):
            og.sim.render()
        return self.cam.get_obs()

    def robot_self_mask(self, name: str, depth, intrinsics, cam_pos_world, cam_quat_cv_world) -> np.ndarray | None:
        """The robot's own pixels in a view, from its link meshes; None when this simulator's cameras never see it."""
        return None

    def view_frame(self, name: str) -> tuple[dict, dict]:
        """One view of the capture: what the wire carries for it (rgb, z-depth, intrinsics, the OpenCV camera pose in
        the robot base frame, the robot's own pixels as ``robot_mask``, None when there are none) and what the
        knowledge sources, the validation and the files need (the camera pose in the world frame, the instance
        segmentation when it was rendered). The robot's pixels are zeroed in the depth."""
        obs, info = self._capture_obs(name)
        sensor = self.view_sensor(name)
        rgb = obs["rgb"][..., :3].cpu().numpy().astype(np.uint8)
        depth = obs["depth_linear"].cpu().numpy().astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0.0
        seg = obs["seg_instance"].cpu().numpy() if "seg_instance" in obs else None
        id_to_name = {int(k): str(v) for k, v in info["seg_instance"].items()} if seg is not None else {}
        intrinsics = sensor.intrinsic_matrix.cpu().numpy().astype(np.float32)
        cam_pos, cam_quat = sensor.get_position_orientation()  # world, USD camera axes
        cam_quat_cv = T.quat_multiply(cam_quat, th.tensor([1.0, 0.0, 0.0, 0.0]))  # 180 deg about camera x -> OpenCV
        cam_pos_b, cam_quat_b = self.to_base(cam_pos, cam_quat_cv)
        world_from_cam = T.pose2mat((cam_pos_b, cam_quat_b)).cpu().numpy().astype(np.float32)
        robot_mask = np.zeros(depth.shape, dtype=bool)  # self-filter: the robot's own body seen by its camera
        if seg is not None:
            for label in self.mask_labels_as_invalid:
                ids = [i for i, n in id_to_name.items() if n == label]
                if ids:
                    robot_mask |= np.isin(seg, ids)
        else:
            own = self.robot_self_mask(name, depth, intrinsics, cam_pos, cam_quat_cv)
            if own is not None:
                robot_mask |= own
        if robot_mask.any():
            depth[robot_mask] = 0.0
            log.info(f"{name}: masked {int(robot_mask.sum())} pixels of the robot out of the depth")
        view = {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "world_from_cam": world_from_cam,
            "robot_mask": robot_mask if robot_mask.any() else None,
        }
        view_extras = {
            "cam_pos_base": cam_pos_b.cpu().numpy().tolist(),
            "cam_quat_wxyz_ros": T.convert_quat(cam_quat_b, to="wxyz").cpu().numpy().tolist(),
            "cam_pos_world": cam_pos.cpu().numpy().tolist(),
            "cam_quat_xyzw_world_usd": cam_quat.cpu().numpy().tolist(),
            "cam_quat_xyzw_world_cv": cam_quat_cv.cpu().numpy().tolist(),
            "seg_instance": seg,
            "id_to_name": id_to_name,
            # Where every tracked object was at the moment THIS view rendered, as 4x4 world poses. A capture turns
            # the torso between head views, and whatever is in the gripper travels with it, so one pose per capture
            # is wrong for anything the robot carries; oracle_masks moves the capture's meshes here before masking
            # this view. Deliberately NOT "object_poses_world": that key is the capture's own (pos, quat) pairs,
            # which capture() writes over the primary view's extras and run.py reads back to restore a scene.
            "object_pose_mats_at_render": self.tracked_poses_world(),
        }
        return view, view_extras

    def capture(self, task: str) -> tuple[dict, dict]:
        """Render every view (the primary, then ``extra_views``) and assemble a TiPToP request from the observation
        alone (per view rgb, z-depth, intrinsics, the OpenCV camera pose in the robot base frame; the planned
        joints), plus ``extras`` for the H5 file, the validation and the knowledge sources (``knowledge.py`` attaches
        labels, masks and buttons afterwards): the camera poses in the world frame, the instance segmentation when
        it was rendered (the primary's at the top level, the others under ``views``), every tracked object's pose.
        """
        view, extras = self.view_frame(self.primary_view)
        request = build_request(
            view["rgb"], view["depth"], view["intrinsics"], view["world_from_cam"], task, self.q_arm()
        )
        request["view_name"] = self.primary_view
        if view["robot_mask"] is not None:
            request["robot_mask"] = view["robot_mask"]  # the server keeps SAM2 off these pixels (occluding gripper)
        self.last_capture_rgb = view["rgb"]
        extras["views"] = {}
        for name in self.extra_views:
            view, view_extras = self.view_frame(name)
            if not np.any(view["depth"] > 0):
                # Every pixel is the robot's own (the self-mask zeroes them) or has no return: the view carries no
                # geometry at all and the planner can only waste time on it. Seen when a capture swing is blocked
                # and the arm stays at the ready posture, where its wrist camera looks at the robot's own body --
                # all 230400 pixels of a left wrist view masked (2026-09-13, dispose_of_batteries).
                log.warning(f"{name}: nothing but the robot in this view (no valid depth); not sent")
                continue
            add_view(
                request,
                name,
                view["rgb"],
                view["depth"],
                view["intrinsics"],
                view["world_from_cam"],
                robot_mask=view["robot_mask"],
            )
            extras["views"][name] = view_extras

        object_poses_base = {}
        self.capture_object_aabb_min_z = {name: float(obj.aabb[0][2]) for name, obj in self.objects.items()}
        for name, obj in self.objects.items():
            pos_b, quat_b = self.to_base(*obj.get_position_orientation())
            aabb_center_b, _ = self.to_base(obj.aabb_center, th.tensor([0.0, 0.0, 0.0, 1.0]))
            lo, hi = obj.aabb
            identity = th.tensor([0.0, 0.0, 0.0, 1.0])
            corners_b = [
                self.to_base(th.tensor([x, y, z]), identity)[0].tolist()
                for x in (float(lo[0]), float(hi[0]))
                for y in (float(lo[1]), float(hi[1]))
                for z in (float(lo[2]), float(hi[2]))
            ]
            object_poses_base[name] = {
                "pos": pos_b.tolist(),
                "quat_xyzw": quat_b.tolist(),
                "aabb_center": aabb_center_b.tolist(),
                "aabb_corners": corners_b,  # for the frame-coverage check in validate_capture
            }
        extras.update(
            {
                "base_pos_world": self.base_pose()[0].cpu().numpy().tolist(),
                "base_quat_xyzw_world": self.base_pose()[1].cpu().numpy().tolist(),
                "object_poses_base": object_poses_base,
                "object_poses_world": self.object_poses_world(),
                "q_fingers": self.q_fingers().tolist(),
                "sim_dt": self.dt,
            }
        )
        return request, extras

    def tracked_object(self, label: str):
        """The simulated object a request label stands for, or None: a tracked task object, else a piece of
        furniture registered as an obstacle for this stance (``nearby_obstacles``). Obstacles are deliberately
        kept out of ``self.objects`` -- they are geometry for the planner to avoid, not things the episode
        poses, checks or frames, and every capture's ``object_poses_base`` (and with it the frame-coverage
        check) is built from ``self.objects`` alone."""
        return self.objects.get(label) or self.obstacles.get(label)

    def tracked_poses_world(self, labels=None) -> dict:
        """{label: 4x4 world pose} of the tracked task objects (all of them, or ``labels``). Obstacles are left out
        on purpose: they are furniture, they do not move, and they are not registered until after the capture."""
        names = list(self.objects) if labels is None else [l for l in labels if l in self.objects]
        return {
            name: T.pose2mat(self.objects[name].get_position_orientation()).cpu().numpy().astype(np.float64).tolist()
            for name in names
        }

    def posed_for_view(self, meshes: dict, view_extras: dict) -> dict:
        """``meshes`` moved to where each object was when this view rendered (``gt_masks.meshes_at_view_poses``)."""
        return meshes_at_view_poses(meshes, view_extras.get("object_pose_mats_at_render") or {}, log=log)

    def object_meshes(self, labels: list[str]) -> dict:
        """{label: trimesh} of tracked objects at their current poses, world frame: the masks of every view of one
        capture come from the same meshes (privileged), each tagged with the pose it was built at so a view that
        saw the object somewhere else can move it back (``posed_for_view``)."""
        missing = [label for label in labels if self.tracked_object(label) is None]
        if missing:
            raise ValueError(
                f"no tracked object for labels {missing} (tracked: {sorted(self.objects)}"
                + (f"; obstacles: {sorted(self.obstacles)}" if self.obstacles else "")
                + ")"
            )
        built = self.tracked_poses_world(labels)
        meshes = {}
        for label in labels:
            mesh = self.object_trimesh_world(label)
            if label in built:
                mesh.metadata = dict(mesh.metadata or {}, world_from_obj=built[label])
            meshes[label] = mesh
        return meshes

    def oracle_masks(self, view: dict, view_extras: dict, labels: list[str], meshes: dict | None = None) -> np.ndarray:
        """(N, H, W) bool masks of tracked objects for one view of a capture (the request itself with the capture's
        extras, or an entry of ``views`` with ``extras["views"][name]``; privileged: simulator truth). Isaac's
        instance segmentation when it was rendered, else from the objects' meshes (``meshes``, else built here) and
        the depth image (``geometry_masks``). A label out of view gets an all-False row."""
        seg = view_extras["seg_instance"]
        if seg is not None:
            id_to_name = view_extras["id_to_name"]
            masks = []
            for label in labels:  # tracked objects may carry a different simulator name (task objects)
                obj = self.tracked_object(label)
                sim_name = obj.name if obj is not None else label
                ids = [i for i, n in id_to_name.items() if n == sim_name]
                masks.append(np.isin(seg, ids) if ids else np.zeros(view["depth"].shape, dtype=bool))
            return np.stack(masks)
        cam_pos = th.tensor(view_extras["cam_pos_world"], dtype=th.float32)
        cam_quat_cv = th.tensor(view_extras["cam_quat_xyzw_world_cv"], dtype=th.float32)
        meshes = self.posed_for_view(self.object_meshes(labels) if meshes is None else meshes, view_extras)
        return self.geometry_masks(view["depth"], view["intrinsics"], cam_pos, cam_quat_cv, labels, meshes=meshes)

    def tiptop_goal(self, atoms: list[dict], category_level: bool) -> tuple[list[str], list[dict]]:
        """The request labels and TiPToP atoms for goal atoms over tracked object names (spawned presets: the
        names are the labels and the predicates are TiPToP's already). R1ProSim translates BDDL names instead."""
        labels = sorted({a for atom in atoms for a in atom["args"] if a in self.objects})
        return labels, [dict(atom) for atom in atoms]

    def button_hints(self, atoms: list[dict], category_level: bool = False) -> dict:
        """Toggle buttons of the goal's objects by pose (privileged); none in the tabletop scene."""
        return {}

    def tracked_label(self, name: str) -> str:
        """The tracked label of an object named in a goal atom (the name itself for spawned presets)."""
        return name

    def link_trimesh_world(self, link, max_faces: int | None = None) -> trimesh.Trimesh | None:
        """Every visual mesh of one link as one trimesh in the WORLD frame (current pose); None for a link without
        meshes. Links without visual meshes contribute their collision meshes. A link is rigid, so its mesh is read
        from USD once, kept in the link's own frame (decimated to ``max_faces`` when asked: the robot's links are
        only a self-mask), and placed by the link's current pose; poses come from the Fabric hierarchy, so the
        result is current after teleports / set_position_orientation without a physics step."""
        key = (link.prim_path, max_faces)
        if key not in self._link_meshes:
            # meta-link volumes (particleapplier, slicer, fluidsource, ...) sit in visual_meshes with purpose
            # "guide": never rendered, so they must not claim depth pixels; collision meshes are all "guide" and
            # stay unfiltered
            geoms = {k: g for k, g in link.visual_meshes.items() if g.purpose != "guide"} or link.collision_meshes
            parts = [
                mesh_prim_to_trimesh_mesh(geom.prim, include_normals=False, include_texcoord=False, world_frame=True)
                for geom in geoms.values()
            ]
            local = None
            if parts:
                world = trimesh.util.concatenate(parts)
                pos, quat = link.get_position_orientation()
                local = world.copy()
                local.apply_transform(np.linalg.inv(T.pose2mat((pos, quat)).cpu().numpy().astype(np.float64)))
                if max_faces is not None and len(local.faces) > max_faces:
                    local = trimesh.Trimesh(*decimated(local.vertices, local.faces, max_faces), process=False)
            self._link_meshes[key] = local
        local = self._link_meshes[key]
        if local is None:
            return None
        world = local.copy()
        world.apply_transform(T.pose2mat(link.get_position_orientation()).cpu().numpy().astype(np.float64))
        return world

    def trimesh_world(self, obj) -> trimesh.Trimesh:
        """Every visual mesh of every link of ``obj`` as one trimesh in the WORLD frame (``link_trimesh_world``)."""
        parts = [mesh for link in obj.links.values() if (mesh := self.link_trimesh_world(link)) is not None]
        if not parts:
            raise ValueError(f"object {obj.name!r} has no visual or collision meshes")
        return trimesh.util.concatenate(parts)

    def object_trimesh_world(self, name: str) -> trimesh.Trimesh:
        return self.trimesh_world(self.tracked_object(name))

    def geometry_masks(
        self, depth, intrinsics, cam_pos_world, cam_quat_cv_world, labels, tol: float | None = None, meshes=None
    ) -> np.ndarray:
        """(N, H, W) bool oracle masks for ``labels`` from the rendered depth and the objects' meshes.

        Computed in the WORLD frame: the meshes come from ``object_meshes`` (or ``meshes``, the same built once for
        every view of a capture) and the camera pose is the world pose of the OpenCV camera frame (same 180
        deg-about-x conversion as the base-frame ``world_from_cam`` of the request, minus the base transform), so
        no vertex transform into the base frame is needed. ``tol`` defaults to ``self.gt_mask_tol`` (see
        ``masks_from_geometry`` for the contact-halo trade-off).
        """
        meshes = self.object_meshes(labels) if meshes is None else meshes
        world_from_cam_w = T.pose2mat((cam_pos_world, cam_quat_cv_world)).cpu().numpy().astype(np.float64)
        masks = masks_from_geometry(
            depth, intrinsics, world_from_cam_w, meshes, tol=self.gt_mask_tol if tol is None else tol
        )
        return np.stack([masks[label] for label in labels])  # all-False rows for objects out of view

    @staticmethod
    def instance_mask(seg: np.ndarray, id_to_name: dict, label: str) -> np.ndarray:
        ids = [i for i, name in id_to_name.items() if name == label]
        if not ids:
            raise ValueError(
                f"object {label!r} is not visible in the instance segmentation (labels: {sorted(set(id_to_name.values()))})"
            )
        return np.isin(seg, ids)

    @staticmethod
    def _names_goal_object(name: str, goal_args) -> bool:
        """Does tracked object ``name`` correspond to one of the goal atoms' arguments?

        Atoms carry request labels: BDDL instance names for a task (``candle.n.01_2``), the per-instance label for
        spawned objects (``candle_2``), or a bare category with ``--knowledge onboard`` (``candle``), which matches any instance.
        """
        nc, ni = canonical_object_name(name)
        for arg in goal_args:
            ac, ai = canonical_object_name(arg)
            if ac == nc and (not ai or not ni or ai == ni):
                return True
        return False

    @staticmethod
    def frame_coverage(request: dict, extras: dict) -> dict:
        """Fraction of each object's projected AABB that falls inside the image (1.0 = fully framed).

        Objects cut by the image border are the silent failure mode of the whole pipeline: the server reconstructs
        the visible sliver into a convex hull that runs *past* the real object, cuTAMP happily satisfies its
        StablePlacement constraint inside that phantom volume, and the item is released beside the container. Needs
        no segmentation, so it also covers ``--knowledge onboard`` captures, where nothing else checks the frame.
        """
        h, w = request["depth"].shape
        coverage = {}
        for name, pose in extras["object_poses_base"].items():
            corners = pose.get("aabb_corners")
            if not corners:
                continue
            px, z = points_to_pixels(corners, request["intrinsics"], request["world_from_cam"])
            if (z <= 0).any():  # straddles the image plane: the projection is meaningless
                continue
            u0, v0 = px.min(axis=0)
            u1, v1 = px.max(axis=0)
            box = max(u1 - u0, 1e-6) * max(v1 - v0, 1e-6)
            inside = max(min(u1, w) - max(u0, 0.0), 0.0) * max(min(v1, h) - max(v0, 0.0), 0.0)
            coverage[name] = float(inside / box)
        return coverage

    def validate_capture(self, request: dict, extras: dict) -> dict:
        """Numerically check the frame conventions: table at base z=0, object mask centroids near their true poses."""
        pts = depth_to_points(request["depth"], request["intrinsics"], request["world_from_cam"])
        seg, id_to_name = extras["seg_instance"], extras["id_to_name"]
        report = {"camera_view_axis_base": request["world_from_cam"][:3, 2].tolist()}
        # Objects at least partly in view but cut by the image border (measured 2026-09-04: a basket at 0.21
        # coverage reconstructed 8 cm too long and the cookie was released 3 cm outside its rim; the same basket
        # fully framed in the next round, 0.72, worked). Runs with or without segmentation.
        coverage = self.frame_coverage(request, extras)
        report["frame_coverage"] = coverage
        views = capture_views(request, extras)
        report["view_coverage"] = {name: self.frame_coverage(view, extras) for name, view, _ in views[1:]}
        # Only the goal's own objects matter: every capture of a crowded table clips something at the edge, and a
        # warning per clipped bystander would drown the one that actually breaks the plan. An object another view
        # frames whole is not clipped.
        goal_args = {a for atom in request.get("gt_atoms") or [] for a in atom.get("args", [])}
        clipped, seen = [], set()
        for name, c in sorted(coverage.items(), key=lambda kv: kv[1]):
            c = max([c, *(vc.get(name, 0.0) for vc in report["view_coverage"].values())])
            # with no atoms to filter by, report every clipped object rather than staying silent
            if not (self.FRAME_COVERAGE_MIN > c > 0.0):
                continue
            if goal_args and not self._names_goal_object(name, goal_args):
                continue
            key = canonical_object_name(name)  # 'candle.n.01_2' and 'candle_2' are the same object
            if key in seen:
                continue
            seen.add(key)
            clipped.append(
                f"{name} is cut by the image border ({100 * c:.0f}% of its projected extent is in frame); its "
                f"reconstructed hull will run past the real object"
            )
        if seg is None:  # rgb + depth only: nothing to compare masks against
            report["note"] = "no instance segmentation rendered; mask checks skipped"
            report["problems"] = clipped
            return report
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
            mask = self.instance_mask(seg, id_to_name, self.objects[name].name if name in self.objects else name)
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
        if self.expect_table_z is None:
            pass  # scene-based embodiments have no synthetic table at a known base height
        elif not np.isfinite(table_z):
            problems.append("table is not visible in the instance segmentation (no valid depth pixels)")
        elif abs(table_z - self.expect_table_z) > 0.02:
            problems.append(f"table top is at base z={table_z:.3f} m, expected ~{self.expect_table_z}")
        for name in extras["object_poses_base"]:
            if report[f"{name}_centroid_error_xy_m"] > 0.06:
                problems.append(
                    f"{name} mask centroid is {report[f'{name}_centroid_error_xy_m']:.3f} m from its true xy"
                )
        if report["camera_view_axis_base"][2] > -0.2:
            problems.append("camera optical axis is not pointing downward in the base frame")
        report["problems"] = problems + clipped
        return report
