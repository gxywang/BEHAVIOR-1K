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

import numpy as np
import torch as th
import trimesh

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.tiptop.gt_masks import masks_from_geometry
from omnigibson.tiptop.protocol import (
    DROID_CAMERA_KWARGS,
    DROID_Q_INIT,
    build_request,
    depth_to_points,
    points_to_pixels,
)
from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh

log = logging.getLogger(__name__)

TABLE_HEIGHT = 0.75  # world z of the table top; the robot base sits on it, so base-frame z = 0 there
FINGER_OPEN = 0.04
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


def canonical_object_name(name: str) -> tuple:
    """('candle.n.01_2' | 'candle_2') -> ('candle', '2'); a bare category -> ('candle', '')."""
    name = name.split(".n.")[0] + ("_" + name.rsplit("_", 1)[1] if ".n." in name else "")
    name = name.replace("__", "_")
    category, _, index = name.rpartition("_")
    return (category, index) if category and index.isdigit() else (name, "")


def rerun_name(name: str) -> str:
    """Entity-path-safe object name for the Rerun mirror ('table.n.02_1' -> 'table_n_02_1')."""
    return name.replace(".", "_").replace(" ", "_").replace("/", "_")


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
        self.state_stream = None  # client.SimStateStream once attached; fed from step()
        self.n_steps = 0
        self.last_obs = None
        self.last_capture_rgb = None  # set by capture(); read by run.py when a goal object is out of frame
        self.capture_object_aabb_min_z = {}
        self._stream_meshes = {}
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

    @property
    def sim_time(self) -> float:
        return self.n_steps * self.dt

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
        self.last_obs = self.env.step(self.action(q_arm, gripper))[0]
        self.n_steps += 1
        if self.state_stream is not None:
            self.state_stream.on_step(self)
        return self.last_obs

    def hold(self, n_steps: int, gripper: float = OPEN, q_arm=None):
        q = self.q_arm() if q_arm is None else q_arm
        for _ in range(n_steps):
            self.step(q, gripper)

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

    def stream_images(self) -> dict:
        """JPEGs for the mirror: the capture camera as the robot sees it, and the overview camera."""
        images = {}
        rgb = self.camera_rgb()
        if rgb is not None:
            images[self.STREAM_CAMERA] = jpeg_bytes(rgb)
        if self.overview is not None:
            over = self.overview.get_obs()[0].get("rgb")
            if over is not None and over.numel():
                images["overview"] = jpeg_bytes(over.cpu().numpy().astype(np.uint8))
        return images

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
            import open3d as o3d

            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
            ).simplify_quadric_decimation(target_number_of_triangles=STREAM_MAX_FACES)
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        return vertices.astype(np.float32), faces.astype(np.int32)

    # ---------------------------------------------------------------- observation
    def _capture_obs(self) -> tuple[dict, dict]:
        """One rendered frame with rgb, depth_linear and seg_instance from the capture camera."""
        for _ in range(3):
            og.sim.render()
        return self.cam.get_obs()

    def capture(self, task: str, gt_labels=None, gt_atoms=None) -> tuple[dict, dict]:
        """Render and assemble a TiPToP request (plus extras for H5/validation) in the robot base frame.

        ``gt_labels`` are request names, normally keys of ``self.objects``. With instance segmentation a label that is
        not a tracked object falls back to the raw segmentation name; the geometry path (no ``seg_instance`` rendered)
        needs the object's meshes and raises for such labels.
        """
        obs, info = self._capture_obs()
        rgb = obs["rgb"][..., :3].cpu().numpy().astype(np.uint8)
        depth = obs["depth_linear"].cpu().numpy().astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0.0
        seg = obs["seg_instance"].cpu().numpy() if "seg_instance" in obs else None
        id_to_name = {int(k): str(v) for k, v in info["seg_instance"].items()} if seg is not None else {}
        intrinsics = self.cam.intrinsic_matrix.cpu().numpy().astype(np.float32)
        robot_mask = np.zeros(depth.shape, dtype=bool)  # self-filter: the robot's own body seen by its camera
        for label in self.mask_labels_as_invalid if seg is not None else ():
            ids = [i for i, n in id_to_name.items() if n == label]
            if ids:
                masked = np.isin(seg, ids)
                depth[masked] = 0.0
                robot_mask |= masked
                log.info(f"masked {int(masked.sum())} pixels of {label!r} out of the depth")

        cam_pos, cam_quat = self.cam.get_position_orientation()  # world, USD camera axes
        cam_quat_cv = T.quat_multiply(cam_quat, th.tensor([1.0, 0.0, 0.0, 0.0]))  # 180 deg about camera x -> OpenCV
        cam_pos_b, cam_quat_b = self.to_base(cam_pos, cam_quat_cv)
        world_from_cam = T.pose2mat((cam_pos_b, cam_quat_b)).cpu().numpy().astype(np.float32)

        gt = None
        if gt_labels and seg is not None:
            # labels are request names; tracked objects may carry a different simulator name (task objects)
            masks = []
            for label in gt_labels:
                sim_name = self.objects[label].name if label in self.objects else label
                ids = [i for i, n in id_to_name.items() if n == sim_name]
                masks.append(np.isin(seg, ids) if ids else np.zeros(depth.shape, dtype=bool))
            masks = np.stack(masks)
            source = "instance segmentation"
        elif gt_labels:  # no seg_instance rendered (segfaults in house scenes): oracle masks from depth + meshes
            masks = self.geometry_masks(depth, intrinsics, cam_pos, cam_quat_cv, gt_labels)
            source = "geometry (depth + object meshes)"
        if gt_labels:
            # a task tracks every object it names, most of them out of view; send the visible ones, but the goal's
            # objects must be in the frame
            counts = {label: int(m.sum()) for label, m in zip(gt_labels, masks)}
            visible = [label for label in gt_labels if counts[label]]
            hidden = [label for label in gt_labels if not counts[label]]
            needed = sorted({a for atom in (gt_atoms or []) for a in atom["args"] if a in hidden})
            if needed:
                self.last_capture_rgb = rgb  # for the caller to save alongside the error
                raise ValueError(f"goal objects {needed} are not visible in the capture (empty masks)")
            log.info(
                f"ground-truth masks from {source}: pixels per label "
                f"{ {label: counts[label] for label in visible} }; not in view: {hidden or 'none'}"
            )
            gt = {
                "labels": visible,
                "masks": masks[[gt_labels.index(l) for l in visible]],
                "atoms": list(gt_atoms or []),
            }
        request = build_request(rgb, depth, intrinsics, world_from_cam, task, self.q_arm(), gt=gt)
        if robot_mask.any():
            request["robot_mask"] = robot_mask  # the server keeps SAM2 off these pixels (occluding gripper)

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
    def trimesh_world(obj) -> trimesh.Trimesh:
        """Every visual mesh of every link of ``obj`` as one trimesh in the WORLD frame (current pose).

        Links without visual meshes contribute their collision meshes. Poses come from the Fabric hierarchy, so the
        result is current after teleports / set_position_orientation without a physics step.
        """
        parts = []
        for link in obj.links.values():
            # meta-link volumes (particleapplier, slicer, fluidsource, ...) sit in visual_meshes with purpose "guide":
            # never rendered, so they must not claim depth pixels; collision meshes are all "guide" and stay unfiltered
            geoms = {k: g for k, g in link.visual_meshes.items() if g.purpose != "guide"} or link.collision_meshes
            for geom in geoms.values():
                parts.append(
                    mesh_prim_to_trimesh_mesh(
                        geom.prim, include_normals=False, include_texcoord=False, world_frame=True
                    )
                )
        if not parts:
            raise ValueError(f"object {obj.name!r} has no visual or collision meshes")
        return trimesh.util.concatenate(parts)

    def object_trimesh_world(self, name: str) -> trimesh.Trimesh:
        return self.trimesh_world(self.objects[name])

    def geometry_masks(
        self, depth, intrinsics, cam_pos_world, cam_quat_cv_world, labels, tol: float | None = None
    ) -> np.ndarray:
        """(N, H, W) bool oracle masks for ``labels`` from the rendered depth and the objects' meshes.

        Computed in the WORLD frame: the meshes come from ``object_trimesh_world`` and the camera pose is the world
        pose of the OpenCV camera frame (same 180 deg-about-x conversion as the base-frame ``world_from_cam`` of the
        request, minus the base transform), so no vertex transform into the base frame is needed. ``tol`` defaults to
        ``self.gt_mask_tol`` (see ``masks_from_geometry`` for the contact-halo trade-off).
        """
        missing = [label for label in labels if label not in self.objects]
        if missing:
            raise ValueError(f"no tracked object for labels {missing} (tracked: {sorted(self.objects)})")
        world_from_cam_w = T.pose2mat((cam_pos_world, cam_quat_cv_world)).cpu().numpy().astype(np.float64)
        meshes = {label: self.object_trimesh_world(label) for label in labels}
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
        spawned objects (``candle_2``), or a bare category with ``--no-gt`` (``candle``), which matches any instance.
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
        no segmentation, so it also covers ``--no-gt`` captures, where nothing else checks the frame.
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
        # Only the goal's own objects matter: every capture of a crowded table clips something at the edge, and a
        # warning per clipped bystander would drown the one that actually breaks the plan.
        goal_args = {a for atom in request.get("gt_atoms") or [] for a in atom.get("args", [])}
        clipped, seen = [], set()
        for name, c in sorted(coverage.items(), key=lambda kv: kv[1]):
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
