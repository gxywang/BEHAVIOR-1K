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


def opening_travel(joint_type: str, lower: float, upper: float, position: float, fraction: float) -> float:
    """How far the joint must move from ``position`` to reach ``fraction`` of its range, signed.

    A joint is opened toward whichever limit is further from where it rests, which is how a closed drawer (at its
    lower limit) and a door hung the other way both open without the caller knowing which.
    """
    lo, hi = float(min(lower, upper)), float(max(lower, upper))
    if hi - lo < 1e-6:
        return 0.0
    target_hi, target_lo = lo + fraction * (hi - lo), hi - fraction * (hi - lo)
    to_hi, to_lo = target_hi - float(position), target_lo - float(position)
    return to_hi if abs(to_hi) <= abs(to_lo) else to_lo


def is_open(lower: float, upper: float, position: float, threshold: float = 0.05) -> bool:
    """OmniGibson's own test (object_states/open_state.py): the joint is off its resting limit by ``threshold``
    of its range. Both ends count, since a joint can rest at either."""
    lo, hi = float(min(lower, upper)), float(max(lower, upper))
    if hi - lo < 1e-6:
        return False
    return min(abs(float(position) - lo), abs(float(position) - hi)) > threshold * (hi - lo)
