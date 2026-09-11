"""Wire and file formats shared with a TiPToP planning server.

No OmniGibson imports: this module is usable (and unit-tested) without Isaac Sim. Formats mirror
``tiptop/tiptop_websocket_server.py``, ``tiptop/planning.py`` and ``tiptop/tiptop_offline.py`` at the submodule
commit BEHAVIOR-1K pins (the simulator-side additions since tiptop v0.3.0 -- gt_* keys, robot_mask, held_labels,
in_hand, workspace_bounds, the ``objects`` response, the mirror messages -- are described in tiptop/docs/simulation.md and listed in
tiptop/CHANGELOG.md) and the reference IsaacLab client in tiptop-robot/droid-sim-evals.
"""

import json
from pathlib import Path

import h5py
import msgpack
import numpy as np

# Plan JSON schema produced by tiptop.planning.serialize_plan; the client accepts any 1.x version.
SUPPORTED_PLAN_MAJOR = 1
MATCH_MAX_DIST = 0.08  # perceived hull to simulated object (m): farther apart is a different object

# Camera used by the DROID IsaacLab reference simulation: 1280x720, 2.8 mm focal length, 5.376 mm aperture,
# i.e. fx = fy = 1280 * 2.8 / 5.376 = 666.67 px, principal point at the image center.
DROID_CAMERA_KWARGS = {"image_width": 1280, "image_height": 720, "focal_length": 2.8, "horizontal_aperture": 5.376}

# Franka joint configuration the DROID reference observations start from (radians, panda_joint1..7).
DROID_Q_INIT = np.array([0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0], dtype=np.float32)


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
def _frame(rgb, depth, intrinsics, world_from_cam) -> tuple:
    """The validated arrays of one view (the conventions of ``build_request``)."""
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
    if intrinsics.shape != (3, 3) or world_from_cam.shape != (4, 4):
        raise ValueError("intrinsics must be (3, 3) and world_from_cam (4, 4)")
    if not np.allclose(world_from_cam[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("world_from_cam is not a homogeneous transform")
    return rgb, depth, intrinsics, world_from_cam


def build_request(rgb, depth, intrinsics, world_from_cam, task: str, q_init, gt: dict | None = None) -> dict:
    """Validate and assemble one planning request: the observation alone (its primary view; ``add_view`` adds
    further views of the same instant).

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        depth: (H, W) float32 z-depth in metres (distance to the image plane); invalid pixels must be 0.
        intrinsics: (3, 3) float32 pinhole matrix for the same resolution.
        world_from_cam: (4, 4) float32 pose of the OpenCV-convention camera frame (+x right, +y down, +z forward)
            expressed in the ROBOT BASE frame, which is TiPToP's world frame.
        task: natural-language instruction.
        q_init: (dof,) float32 joint positions of the planned joints the plan must start from.
        gt: optional {labels, masks, atoms}, forwarded to ``attach_knowledge`` (kept for the offline H5 path and
            older callers; the client normally attaches what it knows afterwards, see ``knowledge.py``).
    """
    rgb, depth, intrinsics, world_from_cam = _frame(rgb, depth, intrinsics, world_from_cam)
    q_init = np.asarray(q_init, dtype=np.float32).reshape(-1)
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
        attach_knowledge(request, gt["labels"], gt.get("atoms", []), masks=gt["masks"])
    return request


def add_view(request: dict, name: str, rgb, depth, intrinsics, world_from_cam, robot_mask=None) -> dict:
    """Append a further view of the same instant to a request (``views``): the conventions of the request's own
    image (its primary view, named by ``view_name``), at the view's own resolution. ``robot_mask`` (H, W) bool
    marks the robot's own pixels in that view. Returns the view dict (per-view masks are attached by
    ``attach_knowledge``)."""
    rgb, depth, intrinsics, world_from_cam = _frame(rgb, depth, intrinsics, world_from_cam)
    name = str(name)
    taken = {request.get("view_name", "primary"), *(view["name"] for view in request.get("views", []))}
    if not name or name in taken:
        raise ValueError(f"view name {name!r} is empty or already in the request ({sorted(taken)})")
    view = {"name": name, "rgb": rgb, "depth": depth, "intrinsics": intrinsics, "world_from_cam": world_from_cam}
    if robot_mask is not None:
        robot_mask = np.asarray(robot_mask, dtype=bool)
        if robot_mask.shape != rgb.shape[:2]:
            raise ValueError(f"view {name!r}: robot_mask {robot_mask.shape} does not match rgb {rgb.shape[:2]}")
        view["robot_mask"] = robot_mask
    request.setdefault("views", []).append(view)
    return view


def capture_views(request: dict, extras: dict) -> list[tuple[str, dict, dict]]:
    """Every view of a capture as (name, view dict, view extras), the primary first: the request itself with the
    capture's extras, then each of ``request["views"]`` with ``extras["views"][name]``."""
    primary = str(request.get("view_name", "primary"))
    return [(primary, request, extras)] + [
        (view["name"], view, extras["views"][view["name"]]) for view in request.get("views", [])
    ]


def attach_knowledge(
    request: dict,
    labels,
    atoms,
    masks=None,
    buttons: dict | None = None,
    held=(),
    in_hand=(),
    workspace=None,
    view_masks: dict | None = None,
) -> dict:
    """Add what the client knows about the scene to a request built by ``build_request`` (validated, in place).

    The wire keys are the server's (``gt_*`` historically; a ``gt_`` key does not mean simulator truth):
        gt_labels: the object names the planner works with; with masks one per mask, else what the detector is
            asked for (category names, and ``<object>_button`` for a toggle button to find on that object).
        gt_atoms: the goal, [{"predicate", "args"}] over those names.
        gt_masks: optional (N, H, W) bool instance masks aligned with ``labels``; the server then skips its
            detector. An all-False row is a label this image does not show; a label the goal names must have
            pixels in at least one view of the request. ``view_masks`` ({view name: (N, H_v, W_v)}) are the same
            for the further views (``add_view``), aligned with the same ``labels``.
        gt_buttons: optional {label: {position, normal, radius}} poses of toggle buttons in the base frame (given
            outright by an oracle, or carried from an earlier detection).
        held_labels: objects in a hand the plan does not move: obstacles the planner must not try to pick up.
        in_hand: objects in the planned hand: the plan starts holding them (a carry: pick here, place after moving).
        workspace_bounds: optional [[x0, y0, z0], [x1, y1, z1]] base-frame box the planner should work in for this
            request instead of its configured tabletop crop (a container on the floor needs the floor in it).
    """
    labels = [str(label) for label in labels]
    request["gt_labels"] = labels
    request["gt_atoms"] = [
        {"predicate": str(atom["predicate"]), "args": [str(arg) for arg in atom["args"]]} for atom in atoms
    ]
    if masks is not None:
        masks = np.asarray(masks).astype(np.uint8)
        if masks.shape != (len(labels), *request["rgb"].shape[:2]):
            raise ValueError(f"masks must be ({len(labels)}, H, W), got {masks.shape}")
        request["gt_masks"] = masks
    if view_masks:
        views = {view["name"]: view for view in request.get("views", [])}
        for name, view_mask in view_masks.items():
            if name not in views:
                raise ValueError(f"view_masks name {name!r} is not a view of the request ({sorted(views)})")
            view_mask = np.asarray(view_mask).astype(np.uint8)
            if view_mask.shape != (len(labels), *views[name]["rgb"].shape[:2]):
                raise ValueError(f"view {name!r}: masks must be ({len(labels)}, H, W), got {view_mask.shape}")
            views[name]["gt_masks"] = view_mask
    if buttons:
        for label, button in buttons.items():
            if len(button["position"]) != 3 or len(button["normal"]) != 3 or float(button["radius"]) <= 0:
                raise ValueError(f"button {label!r} needs a 3-vector position and normal and a positive radius")
        request["gt_buttons"] = {
            str(label): {
                "position": [float(v) for v in b["position"]],
                "normal": [float(v) for v in b["normal"]],
                "radius": float(b["radius"]),
            }
            for label, b in buttons.items()
        }
    if held:
        request["held_labels"] = sorted(str(label) for label in held)
    if in_hand:
        request["in_hand"] = sorted(str(label) for label in in_hand)
    if workspace is not None:
        box = np.asarray(workspace, dtype=np.float64)
        if box.shape != (2, 3) or not np.all(box[0] < box[1]):
            raise ValueError(f"workspace must be [[x0, y0, z0], [x1, y1, z1]] with lo < hi, got {workspace}")
        request["workspace_bounds"] = box.tolist()
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
    gripper_init = plan.get("gripper_init")  # the gripper state the plan assumes at q_init (schema 1.1.0)
    if gripper_init not in (None, "open", "closed"):
        raise ValueError(f"gripper_init must be 'open' or 'closed', got {gripper_init!r}")
    return {
        "version": version,
        "q_init": None if q_init is None else np.asarray(q_init, dtype=np.float32),
        "gripper_init": gripper_init,
        "steps": steps,
    }


def plan_summary(plan: dict) -> str:
    n_traj = sum(s["type"] == "trajectory" for s in plan["steps"])
    n_wp = sum(len(s["positions"]) for s in plan["steps"] if s["type"] == "trajectory")
    duration = sum(len(s["positions"]) * s["dt"] for s in plan["steps"] if s["type"] == "trajectory")
    grippers = [s["action"] for s in plan["steps"] if s["type"] == "gripper"]
    start = f", gripper {plan['gripper_init']} at the start" if plan.get("gripper_init") else ""
    return f"{n_traj} trajectories / {n_wp} waypoints / {duration:.1f}s planned, gripper events {grippers}{start}"


def load_plan_json(path) -> dict:
    with open(path) as f:
        return parse_plan(json.load(f))


# --------------------------------------------------------------------------------------------------------------------
# Offline H5 observation (droid-sim-evals layout consumed by `tiptop-h5`)
# --------------------------------------------------------------------------------------------------------------------
KNOWLEDGE_JSON_KEYS = ("gt_buttons", "held_labels", "in_hand", "workspace_bounds")  # stored as JSON text datasets


def save_observation_h5(path, request: dict, cam_pos_base, cam_quat_wxyz_ros, extra: dict | None = None) -> None:
    """Write an observation in the droid-sim-evals H5 layout plus the knowledge keys of the request, so the file is
    the whole request (``request_from_observation`` rebuilds it).

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
        if "gt_labels" in request:  # what the client knew (attach_knowledge), so the file is the whole request
            f.create_dataset("gt_labels", data=np.array(request["gt_labels"], dtype=h5py.string_dtype()))
            f.create_dataset("gt_atoms", data=json.dumps(request["gt_atoms"]))
        if "gt_masks" in request:
            f.create_dataset("gt_masks", data=request["gt_masks"].astype(np.uint8), compression="gzip")
        for key in KNOWLEDGE_JSON_KEYS:
            if key in request:
                f.create_dataset(key, data=json.dumps(request[key]))
        if "robot_mask" in request:
            f.create_dataset("robot_mask", data=request["robot_mask"].astype(np.uint8), compression="gzip")
        if "view_name" in request:
            f.attrs["view_name"] = str(request["view_name"])
        for view in request.get("views", []):  # further views of the instant: one group each, the same layout
            g = f.create_group(f"views/{view['name']}")
            g.create_dataset("rgb", data=view["rgb"])
            g.create_dataset("depth", data=view["depth"][..., None].astype(np.float32))
            g.create_dataset("intrinsic_matrix", data=view["intrinsics"].astype(np.float32))
            g.create_dataset("world_from_cam", data=view["world_from_cam"].astype(np.float32))
            for key in ("robot_mask", "gt_masks"):
                if key in view:
                    g.create_dataset(key, data=view[key].astype(np.uint8), compression="gzip")
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
        if "gt_labels" in f:
            obs["gt_labels"] = [s.decode() if isinstance(s, bytes) else str(s) for s in f["gt_labels"][:]]
            obs["gt_atoms"] = _json_dataset(f["gt_atoms"])
        if "gt_masks" in f:
            obs["gt_masks"] = f["gt_masks"][:]
        for key in KNOWLEDGE_JSON_KEYS:
            if key in f:
                obs[key] = _json_dataset(f[key])
        if "robot_mask" in f:
            obs["robot_mask"] = f["robot_mask"][:].astype(bool)
        if "view_name" in f.attrs:
            obs["view_name"] = str(f.attrs["view_name"])
        if "views" in f:
            obs["views"] = []
            for name in f["views"]:
                g = f["views"][name]
                depth_v = g["depth"][:]
                depth_v = depth_v[..., 0] if depth_v.ndim == 3 else depth_v
                view = {
                    "name": str(name),
                    "rgb": g["rgb"][:].astype(np.uint8),
                    "depth": np.nan_to_num(depth_v.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                    "intrinsics": g["intrinsic_matrix"][:].astype(np.float32),
                    "world_from_cam": g["world_from_cam"][:].astype(np.float32),
                }
                if "robot_mask" in g:
                    view["robot_mask"] = g["robot_mask"][:].astype(bool)
                if "gt_masks" in g:
                    view["gt_masks"] = g["gt_masks"][:]
                obs["views"].append(view)
    return obs


def _json_dataset(dataset):
    raw = dataset[()]
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


def request_from_observation(obs: dict) -> dict:
    """The planner request a saved observation was (``load_observation_h5``): the frame, and the knowledge keys the
    file has, validated the way ``attach_knowledge`` validates a live request. Replaying a round is
    ``client.plan(request_from_observation(load_observation_h5(round_dir / "obs.h5")))``."""
    request = {key: obs[key] for key in ("rgb", "depth", "intrinsics", "world_from_cam", "task", "q_init")}
    if "view_name" in obs:
        request["view_name"] = obs["view_name"]
    for view in obs.get("views", []):
        add_view(
            request,
            view["name"],
            view["rgb"],
            view["depth"],
            view["intrinsics"],
            view["world_from_cam"],
            robot_mask=view.get("robot_mask"),
        )
    if "gt_labels" in obs:
        attach_knowledge(
            request,
            obs["gt_labels"],
            obs["gt_atoms"],
            masks=obs.get("gt_masks"),
            buttons=obs.get("gt_buttons"),
            held=obs.get("held_labels", ()),
            in_hand=obs.get("in_hand", ()),
            workspace=obs.get("workspace_bounds"),
            view_masks={view["name"]: view["gt_masks"] for view in obs.get("views", []) if "gt_masks" in view},
        )
    return request


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


def points_to_pixels(points, intrinsics, world_from_cam) -> tuple:
    """Project (N, 3) points given in ``world_from_cam``'s reference frame to (N, 2) pixels; also returns camera z."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cam = (pts - world_from_cam[:3, 3]) @ world_from_cam[:3, :3]  # inverse rigid transform
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    z = np.where(np.abs(cam[:, 2]) < 1e-9, 1e-9, cam[:, 2])
    return np.stack([fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy], axis=-1), cam[:, 2]


def match_objects(perceived: dict, simulated: dict, max_dist=MATCH_MAX_DIST) -> dict:
    """Pair perceived objects with simulated ones by position ({name: [x, y, z]}, both in the same frame).

    Perception numbers instances by box size ("candle_2" is the second-largest candle the detector found) and
    the simulator by task instance, so a name never identifies an object across the two; a position does. Greedy by
    distance, each simulated object used once. Returns {perceived name: {"sim": simulated name or None, "dist":
    metres to it, or to the nearest simulated object when none is within ``max_dist``}}; a perceived object without
    a partner is a false detection or a hull that landed somewhere else. ``max_dist`` is one distance, or one per
    simulated name: a hull built from one view of a big object is centred well above the object (its underside is
    never seen), so a large object gets a larger allowance.
    """
    tolerance = max_dist if isinstance(max_dist, dict) else {name: float(max_dist) for name in simulated}
    pairs = sorted(
        (float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(s, dtype=np.float64))), p_name, s_name)
        for p_name, p in perceived.items()
        for s_name, s in simulated.items()
    )
    out = {name: {"sim": None, "dist": None} for name in perceived}
    used = set()
    for dist, p_name, s_name in pairs:
        if out[p_name]["dist"] is None:
            out[p_name]["dist"] = dist  # the nearest overall, for the report when nothing is close enough
        if out[p_name]["sim"] is None and s_name not in used and dist <= tolerance[s_name]:
            out[p_name] = {"sim": s_name, "dist": dist}
            used.add(s_name)
    return out


def bddl_category(name: str) -> str:
    """The category of a BDDL instance name: 'wicker_basket.n.01_2' -> 'wicker_basket' (a bare category is itself)."""
    return name.partition(".n.")[0]


def canonical_object_name(name: str) -> tuple:
    """('candle.n.01_2' | 'candle_2') -> ('candle', '2'); a bare category or a BDDL name without an instance -> (.., '')."""
    if ".n." in name:
        category, _, rest = name.partition(".n.")  # 'candle', '01_2'
        index = rest.rpartition("_")[2] if "_" in rest else ""  # '2', or '' for 'candle.n.01'
        name = category.replace("__", "_") + (f"_{index}" if index else "")
    category, _, index = name.rpartition("_")
    return (category, index) if category and index.isdigit() else (name, "")


def rerun_name(name: str) -> str:
    """Entity-path-safe object name for the Rerun mirror ('table.n.02_1' -> 'table_n_02_1')."""
    return name.replace(".", "_").replace(" ", "_").replace("/", "_")


def face_normal_local(vertices, point) -> np.ndarray:
    """Outward unit normal of the face of ``vertices``' bounding box that ``point`` (same frame) is nearest to.

    Which face of an object a button sits on: the box's six faces are candidates and the one with the smallest gap
    to the point wins (a button 2 mm inside the +x face beats the top face 10 cm away).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    p = np.asarray(point, dtype=np.float64)
    gaps = np.concatenate([p - lo, hi - p])  # to the -x -y -z faces, then the +x +y +z faces
    k = int(np.argmin(gaps))
    normal = np.zeros(3)
    normal[k % 3] = 1.0 if k >= 3 else -1.0
    return normal


def joint_ramp(start, goal, max_step: float) -> np.ndarray:
    """Joint targets from ``start`` to ``goal`` (dof,) in equal steps of at most ``max_step`` per joint, the last one
    ``goal`` itself: (n, dof), n >= 1. A target that jumps makes a position controller slam the joints; commanding
    this ramp one row per control step bounds every joint's speed to ``max_step`` per step."""
    start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    n = max(1, int(np.ceil(np.abs(goal - start).max() / max_step))) if max_step > 0 else 1
    t = np.arange(1, n + 1, dtype=np.float64)[:, None] / n
    return (start + t * (goal - start)).astype(np.float32)


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
