"""Wire and file formats shared with a TiPToP planning server.

No OmniGibson imports: this module is usable (and unit-tested) without Isaac Sim. Formats mirror
``tiptop/tiptop_websocket_server.py``, ``tiptop/planning.py`` and ``tiptop/tiptop_offline.py`` (v0.3.0) and the
reference IsaacLab client in tiptop-robot/droid-sim-evals.
"""

import json
from pathlib import Path

import h5py
import msgpack
import numpy as np

# Plan JSON schema produced by tiptop.planning.serialize_plan; the client accepts any 1.x version.
SUPPORTED_PLAN_MAJOR = 1

# Camera used by the DROID IsaacLab reference simulation: 1280x720, 2.8 mm focal length, 5.376 mm aperture,
# i.e. fx = fy = 1280 * 2.8 / 5.376 = 666.67 px, principal point at the image center.
DROID_CAMERA_KWARGS = {"image_width": 1280, "image_height": 720, "focal_length": 2.8, "horizontal_aperture": 5.376}

# Franka joint configuration the DROID reference observations start from (radians, panda_joint1..7).
DROID_Q_INIT = np.array([0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0], dtype=np.float32)

REQUEST_KEYS = ("rgb", "depth", "intrinsics", "world_from_cam", "task", "q_init")
GT_KEYS = ("gt_labels", "gt_masks", "gt_atoms")


# --------------------------------------------------------------------------------------------------------------------
# msgpack-numpy wire format (what tiptop's msgpack_numpy.unpackb expects), hand-rolled so the sim side only needs msgpack
# --------------------------------------------------------------------------------------------------------------------
def _encode(obj):
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {b"nd": True, b"type": arr.dtype.str, b"kind": b"", b"shape": list(arr.shape), b"data": arr.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot msgpack-encode {type(obj)}")


def _decode(obj):
    for nd_key, type_key, shape_key, data_key in ((b"nd", b"type", b"shape", b"data"), ("nd", "type", "shape", "data")):
        if obj.get(nd_key) is True:
            dtype = obj[type_key]
            dtype = dtype.decode() if isinstance(dtype, bytes) else dtype
            return np.ndarray(buffer=obj[data_key], dtype=np.dtype(dtype), shape=tuple(obj[shape_key])).copy()
    return obj


def packb(obj) -> bytes:
    """Serialize a dict that may contain numpy arrays exactly like msgpack_numpy.Packer().pack does."""
    return msgpack.packb(obj, default=_encode, use_bin_type=True)


def unpackb(data: bytes):
    """Deserialize msgpack (numpy arrays restored), like msgpack_numpy.unpackb."""
    return msgpack.unpackb(data, object_hook=_decode, raw=False, strict_map_key=False)


# --------------------------------------------------------------------------------------------------------------------
# Request / response
# --------------------------------------------------------------------------------------------------------------------
def build_request(rgb, depth, intrinsics, world_from_cam, task: str, q_init, gt: dict | None = None) -> dict:
    """Validate and assemble one planning request.

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        depth: (H, W) float32 z-depth in metres (distance to the image plane); invalid pixels must be 0.
        intrinsics: (3, 3) float32 pinhole matrix for the same resolution.
        world_from_cam: (4, 4) float32 pose of the OpenCV-convention camera frame (+x right, +y down, +z forward)
            expressed in the ROBOT BASE frame, which is TiPToP's world frame.
        task: natural-language instruction.
        q_init: (7,) float32 arm joint positions (panda_joint1..7) the plan must start from.
        gt: optional ground-truth perception {labels: [str], masks: (N, H, W) bool, atoms: [{predicate, args}]};
            when present the server skips Gemini + SAM2 (requires the tiptop fork with the gt_* hook).
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"depth {depth.shape} does not match rgb {rgb.shape[:2]}")
    if not np.all(np.isfinite(depth)) or depth.min() < 0:
        raise ValueError("depth must be finite and non-negative (use 0 for invalid pixels)")
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    world_from_cam = np.asarray(world_from_cam, dtype=np.float32)
    q_init = np.asarray(q_init, dtype=np.float32).reshape(-1)
    if intrinsics.shape != (3, 3) or world_from_cam.shape != (4, 4):
        raise ValueError("intrinsics must be (3, 3) and world_from_cam (4, 4)")
    if not np.allclose(world_from_cam[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("world_from_cam is not a homogeneous transform")
    if not task:
        raise ValueError("task must be a non-empty string")
    request = {
        "rgb": rgb,
        "depth": depth,
        "intrinsics": intrinsics,
        "world_from_cam": world_from_cam,
        "task": str(task),
        "q_init": q_init,
    }
    if gt is not None:
        labels = [str(label) for label in gt["labels"]]
        masks = np.asarray(gt["masks"]).astype(np.uint8)
        if masks.shape != (len(labels), *rgb.shape[:2]):
            raise ValueError(f"gt masks must be ({len(labels)}, H, W), got {masks.shape}")
        request["gt_labels"] = labels
        request["gt_masks"] = masks
        request["gt_atoms"] = [
            {"predicate": str(atom["predicate"]), "args": [str(arg) for arg in atom["args"]]}
            for atom in gt.get("atoms", [])
        ]
    return request


def parse_plan(plan: dict) -> dict:
    """Validate a TiPToP plan dict (response['plan'] or a tiptop_plan.json) and convert arrays to float32."""
    if not isinstance(plan, dict) or "steps" not in plan:
        raise ValueError("plan must be a dict with a 'steps' list")
    version = str(plan.get("version", "1.0.0"))
    major = int(version.split(".")[0])
    if major != SUPPORTED_PLAN_MAJOR:
        raise ValueError(f"unsupported plan schema version {version} (supported major: {SUPPORTED_PLAN_MAJOR})")
    steps = []
    for i, step in enumerate(plan["steps"]):
        kind = step.get("type")
        if kind == "trajectory":
            positions = np.asarray(step["positions"], dtype=np.float32)
            if positions.ndim != 2 or positions.shape[0] < 1:
                raise ValueError(f"step {i}: positions must be (N, dof), got {positions.shape}")
            dt = float(step.get("dt", 0.02))
            if dt <= 0:
                raise ValueError(f"step {i}: dt must be positive")
            velocities = step.get("velocities")
            steps.append(
                {
                    "type": "trajectory",
                    "label": str(step.get("label", "")),
                    "positions": positions,
                    "velocities": None if velocities is None else np.asarray(velocities, dtype=np.float32),
                    "dt": dt,
                }
            )
        elif kind == "gripper":
            action = step.get("action")
            if action not in ("open", "close"):
                raise ValueError(f"step {i}: gripper action must be open/close, got {action!r}")
            steps.append({"type": "gripper", "label": str(step.get("label", "")), "action": action})
        elif kind == "metadata":
            continue
        else:
            raise ValueError(f"step {i}: unknown step type {kind!r}")
    q_init = plan.get("q_init")
    return {
        "version": version,
        "q_init": None if q_init is None else np.asarray(q_init, dtype=np.float32),
        "steps": steps,
    }


def plan_summary(plan: dict) -> str:
    n_traj = sum(s["type"] == "trajectory" for s in plan["steps"])
    n_wp = sum(len(s["positions"]) for s in plan["steps"] if s["type"] == "trajectory")
    duration = sum(len(s["positions"]) * s["dt"] for s in plan["steps"] if s["type"] == "trajectory")
    grippers = [s["action"] for s in plan["steps"] if s["type"] == "gripper"]
    return f"{n_traj} trajectories / {n_wp} waypoints / {duration:.1f}s planned, gripper events {grippers}"


def load_plan_json(path) -> dict:
    with open(path) as f:
        return parse_plan(json.load(f))


# --------------------------------------------------------------------------------------------------------------------
# Offline H5 observation (droid-sim-evals layout consumed by `tiptop-h5`)
# --------------------------------------------------------------------------------------------------------------------
def save_observation_h5(path, request: dict, cam_pos_base, cam_quat_wxyz_ros, extra: dict | None = None) -> None:
    """Write an observation in the droid-sim-evals H5 layout plus optional ground-truth datasets.

    ``cam_pos_base`` / ``cam_quat_wxyz_ros`` describe the OpenCV camera frame in the robot base frame (they duplicate
    request['world_from_cam']). The attribute ``pos_w_z_offset_m = 0`` tells the tiptop fork's loader that the stored
    position is physically exact (the stock loader adds a 1.5 cm DROID calibration offset otherwise).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("rgb", data=request["rgb"])
        f.create_dataset("depth", data=request["depth"][..., None].astype(np.float32))
        f.create_dataset("intrinsic_matrix", data=request["intrinsics"].astype(np.float32))
        f.create_dataset("pos_w", data=np.asarray(cam_pos_base, dtype=np.float32))
        f.create_dataset("quat_w_ros", data=np.asarray(cam_quat_wxyz_ros, dtype=np.float32))
        f.create_dataset("q_init", data=request["q_init"].astype(np.float32))
        f.create_dataset("world_from_cam", data=request["world_from_cam"].astype(np.float32))
        f.create_dataset("task", data=request["task"])
        f.attrs["pos_w_z_offset_m"] = 0.0
        f.attrs["source"] = "omnigibson.tiptop"
        if "gt_masks" in request:
            f.create_dataset("gt_masks", data=request["gt_masks"].astype(np.uint8), compression="gzip")
            f.create_dataset("gt_labels", data=np.array(request["gt_labels"], dtype=h5py.string_dtype()))
            f.create_dataset("gt_atoms", data=json.dumps(request["gt_atoms"]))
        if "robot_mask" in request:
            f.create_dataset("robot_mask", data=request["robot_mask"].astype(np.uint8), compression="gzip")
        for key, value in (extra or {}).items():
            f.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value


def load_observation_h5(path) -> dict:
    """Read an H5 written by save_observation_h5 (or by droid-sim-evals) back into a request-like dict."""
    with h5py.File(path, "r") as f:
        depth = f["depth"][:]
        depth = depth[..., 0] if depth.ndim == 3 else depth
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if "world_from_cam" in f:
            world_from_cam = f["world_from_cam"][:]
        else:
            # droid-sim-evals files: same reconstruction and default +1.5 cm camera z offset as tiptop's loader
            w, x, y, z = [float(v) for v in f["quat_w_ros"][:]]
            world_from_cam = np.eye(4, dtype=np.float32)
            world_from_cam[:3, :3] = quat_wxyz_to_matrix(np.array([w, x, y, z]))
            world_from_cam[:3, 3] = f["pos_w"][:]
            world_from_cam[2, 3] += float(f.attrs.get("pos_w_z_offset_m", 0.015))
        task = f["task"][()] if "task" in f else ""
        task = task.decode() if isinstance(task, bytes) else str(task)
        obs = {
            "rgb": f["rgb"][:].astype(np.uint8),
            "depth": depth,
            "intrinsics": f["intrinsic_matrix"][:].astype(np.float32),
            "world_from_cam": world_from_cam.astype(np.float32),
            "task": task,
            "q_init": np.asarray(f["q_init"][()], dtype=np.float32).reshape(-1),
        }
        if "gt_masks" in f:
            obs["gt_masks"] = f["gt_masks"][:]
            obs["gt_labels"] = [s.decode() if isinstance(s, bytes) else str(s) for s in f["gt_labels"][:]]
            atoms = f["gt_atoms"][()]
            obs["gt_atoms"] = json.loads(atoms.decode() if isinstance(atoms, bytes) else atoms)
        if "robot_mask" in f:
            obs["robot_mask"] = f["robot_mask"][:].astype(bool)
    return obs


# --------------------------------------------------------------------------------------------------------------------
# Small geometry helpers (numpy only)
# --------------------------------------------------------------------------------------------------------------------
def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def depth_to_points(depth, intrinsics, world_from_cam=None) -> np.ndarray:
    """Unproject a z-depth image (OpenCV pinhole) to (H, W, 3) points; invalid (0) pixels map to NaN."""
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = depth.astype(np.float64)
    pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
    if world_from_cam is not None:
        pts = pts @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
    pts[z <= 0] = np.nan
    return pts


def resample_trajectory(positions, dt: float, target_dt: float) -> np.ndarray:
    """Linearly resample an (N, dof) trajectory sampled every ``dt`` seconds onto ``target_dt``, keeping the end point."""
    positions = np.asarray(positions, dtype=np.float32)
    if len(positions) == 1:
        return positions.copy()
    t = np.arange(len(positions)) * dt
    tq = np.arange(0.0, t[-1], target_dt)
    if len(tq) == 0 or tq[-1] < t[-1] - 1e-9:
        tq = np.append(tq, t[-1])
    return np.stack([np.interp(tq, t, positions[:, j]) for j in range(positions.shape[1])], axis=1).astype(np.float32)
