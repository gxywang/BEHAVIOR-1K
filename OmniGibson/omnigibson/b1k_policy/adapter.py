"""The simulator side of the observation seam: a live evaluator observation -> the policy's observation type.

``b1k/observation.py`` in the policy repo says what an observation IS (:class:`~b1k.observation.SensorObservation`
and its privileged sibling :class:`~b1k.observation.OracleObservation`). This module is the other half: it turns
what ``omnigibson.eval.evaluator.Evaluator`` actually hands a policy into one of those. It is deliberately NOT
under ``omnigibson/tiptop/`` -- that tree moves out to the policy repo, this is the piece that stays behind -- and
it is the only place in the submission allowed to import omnigibson. Even here, only
:class:`OracleObservationAdapter` does: the sensor-only path needs nothing but the flat observation dict.

WHAT THE EVALUATOR GIVES, AND WHAT HAD TO BE ADDED
--------------------------------------------------
``Evaluator._preprocess_obs`` flattens the environment's observation to ``{"<robot>::<sensor>::<modality>": tensor}``
and adds two keys of its own. Field by field:

* **proprio** -- ``obs["<robot>::proprio"]``, 61 numbers. Its layout is ``r1pro.yaml``'s ``proprio_obs`` list read
  through ``Robot._get_proprioception_dict``: base_qvel 3, then per arm qpos 7 / qvel 7 / eef pos 3 / eef quat 4 /
  gripper qpos 2 / qvel 2 (left, then right), then trunk qpos 4 / qvel 4. 3 + 25 + 25 + 8 = 61, and field for
  field that is ``PROPRIO_SLICES``. :func:`check_proprio_layout` re-derives it from the YAML at run time so the
  two cannot drift apart silently.
* **task_id** -- ``obs["task_id"]``, an int64 tensor of one element (``evaluator.py``: ``TASK_NAMES_TO_INDICES``).
* **rgb** -- rendered RGBA, (H, W, 4); the alpha channel is dropped, as ``tiptop/scene.py`` already does.
* **depth** -- ``depth_linear`` ONLY: z in the camera frame. OmniGibson also offers a modality literally named
  ``depth``, which is RAY LENGTH; unprojecting that with the pinhole model puts every off-axis point centimetres
  short. Asking for it here is an error, not a fallback.
* **base_from_cam** -- ``obs["<robot>::cam_rel_poses"]``, one flat 21-vector, 7 numbers per camera (pos xyz, quat
  xyzw) in ``r1pro.yaml``'s ``eval.camera_sensor_names`` insertion order, which is **left_wrist, right_wrist,
  head** -- neither alphabetical nor head-first. Split by the tested ``cam_rel_poses_to_base_from_cam``.
  **The evaluator's camera poses are USD camera axes** (-z forward, +y up): it reads them straight off
  ``VisionSensor.get_position_orientation`` / ``cameraViewTransform``, neither of which converts. ``CameraView``
  wants OpenCV (+z along the view axis). :data:`OPENCV_FROM_USD` is that 180 degree turn about the camera x axis,
  and it is applied here, once. Checked against a recorded capture, whose ``extras`` stores both conventions:
  ``usd^-1 * cv`` is exactly 180 degrees about x for all three cameras.
* **intrinsics** -- not in the observation at all. Taken from ``b1k.observation.CAMERA_INTRINSICS``, which is what
  the evaluator's own cameras produce: ``fx = focal_length * width / horizontal_aperture``, and OmniGibson's
  VisionSensor defaults (focal 17.0, aperture 20.995) with ``Evaluator``'s head override
  (``EVAL_HEAD_HORIZONTAL_APERTURE = 40.0``) give 17 * 720 / 40 = 306.0 for the head and 17 * 480 / 20.995 =
  388.664 for a wrist. Both scale with the image width, so an observation at another resolution would silently
  carry the wrong focal length: :meth:`ObservationAdapter.views` refuses one instead.
* **the robot's own pixels** -- not in the observation either, and the whole reason ``selfmask.py`` exists. Pass a
  :class:`~omnigibson.b1k_policy.selfmask.R1ProSelfMask` and the depth is filtered with forward kinematics from
  the proprioception above; pass None and the robot's arms stay in the depth, where the planner will meet them as
  obstacles.
"""

from pathlib import Path

import numpy as np
import yaml

from b1k.observation import (
    CAM_REL_POSE_ORDER,
    CAMERA_INTRINSICS,
    HEAD_RESOLUTION,
    PROPRIO_DIM,
    PROPRIO_SLICES,
    WRIST_RESOLUTION,
    CameraView,
    OracleFields,
    OracleObject,
    OracleObservation,
    SensorObservation,
    cam_rel_poses_to_base_from_cam,
    pose_to_matrix,
)

# OpenCV camera <- USD camera: 180 degrees about the camera's x axis. USD is -z forward / +y up, OpenCV is
# +z forward / +y down. Right-multiply a USD base_from_cam by this to get the OpenCV one.
OPENCV_FROM_USD = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

# How many numbers each proprio_obs entry of r1pro.yaml contributes, for check_proprio_layout. R1Pro: 7 joints per
# arm, 2 finger joints per gripper, 4 trunk joints, and a 3-vector base velocity in the base's own frame.
PROPRIO_FIELD_DIMS = {name: s.stop - s.start for name, s in PROPRIO_SLICES.items()}

# The resolutions CAMERA_INTRINSICS is for; the eval RGBDFullResWrapper sets exactly these.
VIEW_RESOLUTIONS = {"head": HEAD_RESOLUTION, "left_wrist": WRIST_RESOLUTION, "right_wrist": WRIST_RESOLUTION}


def eval_config_path() -> Path:
    """``omnigibson/eval/r1pro.yaml``, the challenge's own robot config."""
    return Path(__file__).resolve().parents[1] / "eval" / "r1pro.yaml"


def check_proprio_layout(config_path=None) -> dict:
    """Re-derive the proprioception layout from ``r1pro.yaml`` and check it against ``PROPRIO_SLICES``.

    Returns ``{field: slice}``. Raises when the challenge's ``proprio_obs`` list, in its own order, does not tile
    the 61 numbers the way the policy package expects -- which would mean every ``proprio_field`` call is reading
    the wrong slice.
    """
    with open(config_path or eval_config_path()) as f:
        fields = list(yaml.safe_load(f)["proprio_obs"])
    if fields != list(PROPRIO_SLICES):
        raise ValueError(f"r1pro.yaml proprio_obs is {fields}, PROPRIO_SLICES is {list(PROPRIO_SLICES)}")
    layout, start = {}, 0
    for name in fields:
        layout[name] = slice(start, start + PROPRIO_FIELD_DIMS[name])
        start += PROPRIO_FIELD_DIMS[name]
    if start != PROPRIO_DIM or layout != dict(PROPRIO_SLICES):
        raise ValueError(f"proprio layout from r1pro.yaml sums to {start} as {layout}, expected {dict(PROPRIO_SLICES)}")
    return layout


def camera_sensor_names(config_path=None) -> dict:
    """``r1pro.yaml``'s ``eval.camera_sensor_names``, and the check that its ORDER is ``CAM_REL_POSE_ORDER``.

    ``Evaluator._preprocess_obs`` concatenates one camera pose per entry of this mapping, in insertion order, so
    the order of a YAML mapping is load-bearing: get it wrong and every camera pose is attributed to the wrong
    camera, which no shape check would catch.
    """
    with open(config_path or eval_config_path()) as f:
        names = yaml.safe_load(f)["eval"]["camera_sensor_names"]
    if tuple(names) != tuple(CAM_REL_POSE_ORDER):
        raise ValueError(f"r1pro.yaml camera order is {tuple(names)}, CAM_REL_POSE_ORDER is {CAM_REL_POSE_ORDER}")
    return dict(names)


def _numpy(value) -> np.ndarray:
    """A torch tensor or array-like as a numpy array, without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


class ObservationAdapter:
    """A flat evaluator observation dict -> :class:`~b1k.observation.SensorObservation`. Nothing privileged.

    Pure in the sense that matters: it reads the dict the evaluator already built and never touches the scene, so
    it is exercised offline against recorded captures.
    """

    def __init__(self, robot_name: str = "robot_r1", sensors=None, intrinsics=None, self_mask=None):
        """
        Args:
            robot_name: the evaluator's robot name, the prefix of every observation key.
            sensors: ``{view name: sensor name}``; ``r1pro.yaml``'s own mapping when None.
            intrinsics: ``{view name: (3, 3)}``; ``CAMERA_INTRINSICS`` when None.
            self_mask: a :class:`~omnigibson.b1k_policy.selfmask.R1ProSelfMask`, or None to leave the robot's own
                body in the depth.
        """
        self.robot_name = robot_name
        self.sensors = dict(sensors) if sensors is not None else camera_sensor_names()
        self.intrinsics = dict(intrinsics) if intrinsics is not None else dict(CAMERA_INTRINSICS)
        self._own_intrinsics = intrinsics is None  # then the resolution they are for is not negotiable
        self.self_mask = self_mask
        check_proprio_layout()

    def proprio(self, obs) -> np.ndarray:
        key = f"{self.robot_name}::proprio"
        if key not in obs:
            raise KeyError(f"{key!r} is not in the observation; r1pro.yaml must list 'proprio' in obs_modalities")
        return _numpy(obs[key]).astype(np.float32).reshape(-1)

    def task_id(self, obs) -> int:
        return int(_numpy(obs["task_id"]).reshape(-1)[0])

    def base_from_cam(self, obs) -> dict:
        """``{view name: (4, 4) OpenCV base_from_cam}`` from the evaluator's flat, USD-convention cam_rel_poses."""
        key = f"{self.robot_name}::cam_rel_poses"
        if key not in obs:
            raise KeyError(f"{key!r} is not in the observation; Evaluator._preprocess_obs adds it")
        usd = cam_rel_poses_to_base_from_cam(_numpy(obs[key]))
        return {name: (mat @ OPENCV_FROM_USD).astype(np.float32) for name, mat in usd.items()}

    def views(self, obs) -> dict:
        """``{view name: CameraView}`` for every camera the observation actually carries."""
        poses = self.base_from_cam(obs)
        views = {}
        for name, sensor in self.sensors.items():
            prefix = f"{self.robot_name}::{sensor}"
            if f"{prefix}::rgb" not in obs:
                continue
            if f"{prefix}::depth_linear" not in obs:
                offered = sorted(k.rsplit("::", 1)[1] for k in obs if k.startswith(f"{prefix}::"))
                raise KeyError(
                    f"{name}: no {prefix}::depth_linear. The eval wrapper must enable 'depth_linear' (z to the "
                    f"image plane); the modality named 'depth' is RAY LENGTH and is wrong here. Offered: {offered}"
                )
            rgb = _numpy(obs[f"{prefix}::rgb"])[..., :3].astype(np.uint8)  # the evaluator renders RGBA
            expected = VIEW_RESOLUTIONS.get(name)
            if self._own_intrinsics and expected is not None and rgb.shape[:2] != expected:
                raise ValueError(
                    f"{name}: rendered at {rgb.shape[:2]}, but CAMERA_INTRINSICS is for {expected} and the focal "
                    "length scales with the image width. Use the RGBDFullResWrapper, or pass intrinsics."
                )
            depth = np.nan_to_num(_numpy(obs[f"{prefix}::depth_linear"]).astype(np.float32), nan=0.0, posinf=0.0)
            depth[depth < 0] = 0.0
            views[name] = CameraView(
                name=name, rgb=rgb, depth=depth, intrinsics=self.intrinsics[name], base_from_cam=poses[name]
            )
        if not views:
            raise KeyError(f"no camera in the observation for any of {sorted(self.sensors)}")
        return views

    def __call__(self, obs) -> SensorObservation:
        proprio = self.proprio(obs)
        views = self.views(obs)
        if self.self_mask is not None:
            views = {name: self.self_mask.filtered(view, proprio) for name, view in views.items()}
        source = "evaluator" if self.self_mask is None else "evaluator:self-masked"
        return SensorObservation(views=views, proprio=proprio, task_id=self.task_id(obs), source=source)


class OracleObservationAdapter(ObservationAdapter):
    """The same instant plus everything the simulator knows and a real robot does not. PRIVILEGED, bring-up only.

    Everything it reads beyond :class:`ObservationAdapter` comes from the live scene and lands in
    ``OracleObservation.oracle``: the robot's true pose in the world, the task objects' poses and boxes, and --
    the point of the thing -- ``robot_masks`` from ``seg_instance``, which is the ground truth ``selfmask.py``
    is measured against.
    """

    def __init__(self, env, robot_name=None, **kwargs):
        """``env``: the live ``og.Environment`` (or eval wrapper). Its first robot is the one observed."""
        self.env = env
        self.robot = env.robots[0]
        super().__init__(robot_name=robot_name or self.robot.name, **kwargs)

    def _task_objects(self) -> dict:
        """``{BDDL instance name: object}`` for the loaded task: no agent, no floors, no systems (they have no pose)."""
        scope = getattr(self.env.task, "object_scope", None) or {}
        return {
            name: obj
            for name, obj in scope.items()
            if obj is not None and hasattr(obj, "aabb") and not name.startswith(("agent.", "floor."))
        }

    def _world_from_base(self) -> np.ndarray:
        pos, quat = (_numpy(v) for v in self.robot.get_position_orientation())
        return pose_to_matrix(pos, quat)

    def _objects(self, world_from_base) -> dict:
        """``{label: OracleObject}``: world pose from the scene, bounding box brought into the robot base frame."""
        base_from_world = np.linalg.inv(world_from_base)

        def to_base(points):
            points = np.atleast_2d(np.asarray(points, dtype=np.float64))
            return points @ base_from_world[:3, :3].T + base_from_world[:3, 3]

        objects = {}
        for label, obj in self._task_objects().items():
            pos, quat = (_numpy(v) for v in obj.get_position_orientation())
            low, high = (_numpy(v).astype(np.float64).reshape(3) for v in obj.aabb)
            corners = np.array(
                [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
            )
            objects[label] = OracleObject(
                pos_world=np.asarray(pos, dtype=np.float32),
                quat_xyzw_world=np.asarray(quat, dtype=np.float32),
                aabb_center_base=to_base(_numpy(obj.aabb_center))[0].astype(np.float32),
                aabb_corners_base=to_base(corners).astype(np.float32),
            )
        return objects

    def _masks(self, views: dict) -> tuple:
        """``(labels, instance_masks, robot_masks)`` from each camera's ``seg_instance``; empty when it is off.

        ``seg_instance`` is useless without the id -> name table that goes with it, and the evaluator's flattened
        dict drops it, so this asks the sensors directly (``VisionSensor.get_obs`` returns both). No render is
        forced: the modality was filled by the step the caller is adapting. Segmentation names objects by
        ``obj.name``; the labels here are the task's BDDL instance names, mapped through ``_task_objects``.
        """
        task_objects = self._task_objects()
        labels = tuple(sorted(task_objects))
        name_to_label = {obj.name: label for label, obj in task_objects.items()}
        instance_masks, robot_masks = {}, {}
        for view_name, sensor_name in self.sensors.items():
            if view_name not in views or sensor_name not in self.robot.sensors:
                continue
            sensor = self.robot.sensors[sensor_name]
            if "seg_instance" not in sensor.modalities:
                continue
            sensor_obs, info = sensor.get_obs()
            seg = _numpy(sensor_obs["seg_instance"])
            id_to_name = {int(k): str(v) for k, v in info["seg_instance"].items()}
            robot_ids = [i for i, n in id_to_name.items() if n == self.robot.name]
            robot_masks[view_name] = np.isin(seg, robot_ids)
            per_label = {label: np.zeros(seg.shape, dtype=bool) for label in labels}
            for i, n in id_to_name.items():
                if n in name_to_label:
                    per_label[name_to_label[n]] |= seg == i
            instance_masks[view_name] = per_label
        return labels, instance_masks, robot_masks

    def __call__(self, obs) -> OracleObservation:
        proprio = self.proprio(obs)
        views = self.views(obs)
        labels, instance_masks, robot_masks = self._masks(views)
        if self.self_mask is not None:
            views = {name: self.self_mask.filtered(view, proprio) for name, view in views.items()}
        world_from_base = self._world_from_base()
        return OracleObservation(
            views=views,
            proprio=proprio,
            task_id=self.task_id(obs),
            source="evaluator:oracle",
            oracle=OracleFields(
                world_from_base=world_from_base,
                labels=labels,
                instance_masks=instance_masks,
                robot_masks=robot_masks,
                objects=self._objects(world_from_base.astype(np.float64)),
            ),
        )
