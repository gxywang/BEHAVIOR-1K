"""Opening and closing an articulated container: the geometry of following a joint with the gripper.

A drawer or a door is not a pick and place. The hand takes hold of a point on the moving link and then has to
travel the path that link's joint allows -- a straight slide for a drawer, an arc about a hinge for a door -- while
the link comes with it. This module is the pure half of that: given where the handle is and what the joint is, it
says where the gripper must be at each step. Nothing here touches the simulator or the planner, so it is all
testable without a GPU; ``r1pro.py`` reads the joint (``openable_hint``) and executes the path.

Why it is worth having: expanding all 100 challenge goals says the pipeline's vocabulary caps a run at a mean
q_score of 0.415 with 28 tasks fully expressible, and adding open/not-open takes that to 0.636 and 50 tasks -- the
largest single unlock in the set. 23 tasks score an ``open`` atom directly, 24 need a container opened before
anything can be put inside it, and 23 start a goal item inside one.
"""

import math

import numpy as np

# OmniGibson's Open state (object_states/open_state.py) calls a joint open at 5% of its range, for both revolute
# and prismatic joints. So the SCORED atom is cheap -- a drawer 2-3 cm out already satisfies it -- while reaching
# INTO the container needs the full stroke. The two are different targets and the caller says which it wants.
OPEN_FRACTION_SCORED = 0.08  # a little past the 5% the state flips at, for the atom alone
OPEN_FRACTION_REACH = 0.80  # enough of the range to put something in


def pose_matrix(position, quat_xyzw) -> np.ndarray:
    """A 4x4 from a position and an (x, y, z, w) quaternion."""
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    m[:3, 3] = np.asarray(position, dtype=np.float64)
    return m


def rotation_about(axis, angle: float) -> np.ndarray:
    """Rodrigues: a 3x3 rotation of ``angle`` radians about a unit ``axis``."""
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return np.eye(3)
    a = a / n
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def joint_transform(joint_type: str, axis, origin, travel: float) -> np.ndarray:
    """The 4x4 a link's own frame undergoes when its joint moves by ``travel``.

    ``prismatic``: ``travel`` metres along ``axis``. ``revolute``: ``travel`` radians about ``axis`` through
    ``origin``. Both in whatever frame the axis and origin are given in; the caller keeps everything in one frame.
    """
    kind = str(joint_type).lower()
    out = np.eye(4)
    if kind.startswith("prismatic"):
        a = np.asarray(axis, dtype=np.float64)
        out[:3, 3] = a / max(float(np.linalg.norm(a)), 1e-9) * float(travel)
        return out
    if kind.startswith("revolute") or kind.startswith("continuous"):
        rot = rotation_about(axis, float(travel))
        o = np.asarray(origin, dtype=np.float64)
        out[:3, :3] = rot
        out[:3, 3] = o - rot @ o
        return out
    raise ValueError(f"joint type {joint_type!r} does not open: expected prismatic or revolute")


def follow_joint(handle_pose, joint_type: str, axis, origin, travel: float, steps: int = 8) -> list:
    """Gripper poses that carry the handle from where it is to ``travel`` along its joint, ``steps`` of them.

    The first pose is where the handle is now (so the caller can grasp there) and the last is the opened one. For
    a drawer the poses translate; for a door they swing, and the gripper TURNS with the door -- a hand that keeps
    its orientation while the door rotates would have to slide around the handle, which a real grasp cannot do.
    """
    start = np.asarray(handle_pose, dtype=np.float64).reshape(4, 4)
    out = []
    for i in range(max(2, steps)):
        t = float(travel) * i / (max(2, steps) - 1)
        out.append(joint_transform(joint_type, axis, origin, t) @ start)
    return out


def grasp_orientations(current_rot, pull_direction) -> list:
    """Rotations to try for taking hold of a handle, best guess first.

    Nothing says which way the R1Pro's jaw must face to hold a drawer front, and the orientation the hand happens
    to be carrying is rarely one the arm can reach at the handle -- the first smoke test on store_honey's cabinet
    failed with "no inverse kinematics for step 1 of 10" for exactly that. So the caller is given several: the
    hand's own orientation, and orientations that turn its three axes to face along the pull, each with a quarter
    turn about that direction so the fingers can close either way across the handle. The one that solves is logged,
    which is how the right convention gets learned rather than assumed.
    """
    cur = np.asarray(current_rot, dtype=np.float64).reshape(3, 3)
    pull = np.asarray(pull_direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(pull))
    if n < 1e-9:
        return [cur]
    pull = pull / n
    out = [cur]
    for axis_index in range(3):  # turn the hand's x, y or z to face the way the drawer comes out
        a = cur[:, axis_index]
        v = np.cross(a, -pull)
        c = float(np.dot(a, -pull))
        if np.linalg.norm(v) < 1e-9:
            aligned = cur if c > 0 else rotation_about(cur[:, (axis_index + 1) % 3], math.pi) @ cur
        else:
            k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            aligned = (np.eye(3) + k + k @ k * (1.0 / (1.0 + c))) @ cur
        for turn in (0.0, math.pi / 2):
            out.append(rotation_about(-pull, turn) @ aligned)
    return out


def opening_travel(joint_type: str, lower: float, upper: float, position: float, fraction: float) -> float:
    """How far the joint must move from ``position`` to reach ``fraction`` of its range, signed.

    A joint is opened toward whichever limit is further from where it rests, which is how a closed drawer (at its
    lower limit) and a door hung the other way both open without the caller knowing which.
    """
    lo, hi = float(min(lower, upper)), float(max(lower, upper))
    span = hi - lo
    if span < 1e-6:
        return 0.0
    here = float(position)
    # Open AWAY from the limit the joint is resting at. Picking whichever target was nearer -- which this did --
    # sends a drawer closed at 0.0 to 0.078 of its 0.39 range and calls that 80% open, because 0.078 is nearer to
    # 0.0 than 0.312 is. The probe on store_honey's cabinet is what showed it (2026-09-13).
    target = lo + fraction * span if abs(here - lo) <= abs(here - hi) else hi - fraction * span
    return float(min(max(target, lo), hi) - here)


def is_open(lower: float, upper: float, position: float, threshold: float = 0.05) -> bool:
    """OmniGibson's own test (object_states/open_state.py): the joint is off its resting limit by ``threshold``
    of its range. Both ends count, since a joint can rest at either."""
    lo, hi = float(min(lower, upper)), float(max(lower, upper))
    if hi - lo < 1e-6:
        return False
    return min(abs(float(position) - lo), abs(float(position) - hi)) > threshold * (hi - lo)


# ---------------------------------------------------------------- reading a joint out of a scene
AXIS_VECTORS = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
OPENABLE_TYPES = ("revolute", "prismatic")


def openable_joints(obj) -> list:
    """Every joint of ``obj`` that could open, as dicts in the WORLD frame.

    Each: name, kind (revolute/prismatic), axis, origin, lower, upper, position, link (the moving link's name).
    The axis a joint reports is a letter in its own local frame, so it is composed with the parent link's pose to
    get a world direction; the origin likewise. Joints with no range are skipped -- a fixed joint cannot open.

    Privileged, like the button poses the oracle knowledge source sends: at evaluation the same fields would have
    to come from perception. It is written as a reader so that the rest of the skill does not care which.
    """
    import omnigibson.utils.transform_utils as T
    import torch as th

    out = []
    for name, joint in (getattr(obj, "joints", None) or {}).items():
        try:
            kind = str(joint.joint_type).lower()
            if not any(k in kind for k in OPENABLE_TYPES):
                continue
            lower, upper = float(joint.lower_limit), float(joint.upper_limit)
            if not np.isfinite([lower, upper]).all() or abs(upper - lower) < 1e-6:
                continue
            position = float(np.asarray(joint.get_state()[0]).reshape(-1)[0])
            local = np.asarray(AXIS_VECTORS.get(str(joint.axis).upper(), (1.0, 0.0, 0.0)), dtype=np.float64)
            parent_key = str(joint.body0).split("/")[-1] if getattr(joint, "body0", None) else ""
            child_key = str(joint.body1).split("/")[-1] if getattr(joint, "body1", None) else ""
            parent = obj.links.get(parent_key)
            child = obj.links.get(child_key)
            frame = parent if parent is not None else child
            if frame is None:
                continue
            pos, quat = frame.get_position_orientation()
            rot = T.quat2mat(th.as_tensor(quat)).cpu().numpy().astype(np.float64)
            out.append(
                {
                    "name": name,
                    "kind": "revolute" if "revolute" in kind or "continuous" in kind else "prismatic",
                    "axis": rot @ local,
                    "origin": np.asarray(pos.cpu().numpy(), dtype=np.float64),
                    "lower": lower,
                    "upper": upper,
                    "position": position,
                    # the KEY in obj.links, not the link's .name: they differ, and the caller looks the link up
                    # by key. The smoke test on store_honey's cabinet failed with "has no link to take hold of"
                    # for exactly this (2026-09-13).
                    "link": child_key if child is not None else "",
                }
            )
        except Exception as e:  # a joint that does not answer is not one to open
            log_name = getattr(obj, "name", "?")
            print(f"openable_joints: skipping {log_name}.{name}: {type(e).__name__}: {e}")
    return out


def handle_point(link, axis, opening_sign: float) -> np.ndarray:
    """A point to take hold of on a moving link: the middle of the face that leads when the joint opens.

    Nothing in the dataset marks a handle, so this takes the link's own box and steps to the face furthest along
    the direction the link travels -- the drawer front, the door's swinging edge -- which is where a handle is when
    there is one and a reasonable place to push or pull when there is not.
    """
    lo, hi = (v.cpu().numpy().astype(np.float64) for v in link.aabb)
    centre = (lo + hi) / 2.0
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return centre
    a = a / n * (1.0 if opening_sign >= 0 else -1.0)
    half = (hi - lo) / 2.0
    return centre + a * float(np.abs(half @ a))
