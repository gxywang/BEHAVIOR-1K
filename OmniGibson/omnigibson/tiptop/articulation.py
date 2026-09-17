"""Reading an articulated container out of a live scene: the two halves of the opening skill that need a prim.

The geometry of following a joint with the gripper -- where the hand must be at each step, which way the joint
leads, where on the moving link there is something a jaw can close on -- moved to ``b1k.bridge.articulation``
and is re-exported below, so callers of this module are unchanged. What is left is the pair that reads a live
object: ``openable_joints`` walks an object's joints and their parent links, and ``handle_point`` reads a link's
AABB. ``r1pro.py`` calls both, then hands the numbers to the pure half.
"""

import numpy as np

from b1k.bridge.articulation import (  # noqa: F401
    HANDLE_BAR_MIN,
    HANDLE_DEPTH_RANGE,
    HANDLE_JAW,
    HANDLE_JAW_MARGIN,
    HANDLE_LAYER,
    HANDLE_PANEL_COVER,
    HANDLE_PROUD_MIN,
    OPEN_FRACTION_REACH,
    OPEN_FRACTION_SCORED,
    follow_joint,
    grasp_orientations,
    handle_on,
    is_open,
    joint_transform,
    leading_direction,
    opening_travel,
    pose_matrix,
    rotation_about,
)

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


def handle_point(link, axis, opening_sign: float, grip: str = "face", inset: float = 0.015) -> np.ndarray:
    """A point to take hold of on a moving link, and there is rarely anything built to take hold of.

    Nothing in the BEHAVIOR assets marks a handle: `bottom_cabinet/slgzfc/misc/metadata.json`, the cabinet of
    store_honey, has link_tags {link_1..4: ["openable"]} and nothing else, and its drawer fronts are flat panels.
    So two grips are offered and the caller tries both:

    ``face``   the middle of the face that leads when the joint opens -- the drawer front, the door's swinging
               edge. Right when a handle protrudes there, and impossible on a flat panel, since a parallel jaw
               has nothing to close on.
    ``edge``   the middle of the leading face's TOP edge, ``inset`` below the top surface, which a parallel jaw
               can pinch: it comes down over the panel and closes across its thickness. This is the one that can
               work on a flat drawer front.
    """
    lo, hi = (v.cpu().numpy().astype(np.float64) for v in link.aabb)
    centre = (lo + hi) / 2.0
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return centre
    a = a / n * (1.0 if opening_sign >= 0 else -1.0)
    half = (hi - lo) / 2.0
    point = centre + a * float(np.abs(half @ a))
    if grip == "edge":
        point[2] = float(hi[2]) - float(inset)
    return point
