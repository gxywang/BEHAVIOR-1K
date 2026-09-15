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
OPEN_FRACTION_SCORED = 0.80  # measured: a short stroke does not move the drawer at all (see below)
# Why the scored fraction is the same long stroke as the reach one. Asked for 8% of store_honey's drawer
# range (31 mm over ten steps, 3 mm a step) the drawer moved 7 mm and stayed shut; asked for 80% (312 mm,
# 31 mm a step) it tracked the hand one-to-one -- 0.031, 0.066, 0.101, 0.134 against 0.031, 0.063, 0.094,
# 0.125 -- until the ARM ran out of reach at 13 cm, which is well past the 5% of range the `open` state
# flips at. Short steps do not break the joint's stiction; long ones do. Being stopped early costs nothing,
# because whatever travel was achieved is kept (2026-09-13).
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


# ---------------------------------------------------------------- where the handle is, from the mesh
# Measured off the decrypted assets (scratchpad/dump_link_verts.py + probe_handles.py, 2026-09-14), asset scale 1,
# vertex layers 4 mm thick behind the leading face:
#   bottom_cabinet/slgzfc (store_honey)  full-face layer (0.428 x 0.133) at 2.4 cm; in front of it a strip 0.376 wide
#                                        x 0.008 tall x 0.010 deep near the drawer's top: a horizontal RAIL 2.4 cm proud
#                                        (2.8 cm, 1.2 cm tall, 0.95 m wide at the instance's scale [1.15, 2.52, 1.55]).
#                                        The survey that called it "a 9 mm lip" took the densest plane as the panel,
#                                        and the densest plane is the rail's own BACK face.
#   bottom_cabinet/bamfsz                the same rail, 0.376 x 0.005, 2.4 cm proud.
#   bottom_cabinet/rhdbzv                a rail 1.061 wide x 0.008 tall along the top edge, 2.0 cm proud.
#   bottom_cabinet/jhymlr                a tab 0.087 wide x 0.002 tall, 1.2 cm proud of the full-face layer.
#   fridge/petcxr                        both doors: a vertical bar 0.035 wide x 0.986 tall, ~6 cm proud.
#   fridge/dszchb                        a vertical bar 0.014 x 0.241, 4.4 cm proud.   microwave/hjjxmi: 0.014 x 0.106.
# So every drawer and door in the test scenes has SOMETHING a parallel jaw can close around, and none of it is
# marked anywhere: no handle link, no link tag, no meta link. It is read off the moving link's own vertices.
HANDLE_LAYER = 0.004  # m: thickness of one vertex layer behind the leading face
HANDLE_DEPTH_RANGE = 0.15  # m behind the front that is searched for the panel face
HANDLE_PANEL_COVER = 0.7  # a layer spanning this fraction of the link's cross-section both ways is the panel
HANDLE_PROUD_MIN = 0.008  # m in front of the panel before a vertex counts as standing proud of it
HANDLE_BAR_MIN = 0.012  # m proud before a proud part can be closed around rather than merely touched
HANDLE_JAW = 0.10  # m: the R1Pro jaw fully open (embodiment gripper.max_width_m)
HANDLE_JAW_MARGIN = 0.02  # m the proud part must fit inside the jaw by


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def leading_direction(joint_type: str, axis, origin, vertices, travel: float) -> np.ndarray:
    """Unit vector the moving link's opening face looks along: the way its handle sets off when the joint opens.

    A drawer's is its slide, signed by the travel. A door's is the direction its centroid first moves in,
    axis x (centroid - hinge) x sign(travel), which is the panel's outward normal to within the angle the handle
    and the shelves inside the door pull the centroid round: 1 deg on fridge/petcxr's right door, 6 deg on its
    left one. Estimating the normal from the vertices instead was tried and was worse on every door measured
    (2026-09-14): a PCA normal is skewed by the shelves behind the panel (the right door's 6.0 cm bar read as 3.6
    and 2.8 cm), the longest forward-facing hull edge tilts with the handle, and the sharpest vertex layer is the
    door's dense BACK. And petcxr's left door has no flat front to find: its front-most vertices range over 17 cm
    across the width, so it reads "flat" with an implausible depth whatever the direction, and the caller treats
    that as a face to press on.
    """
    kind = str(joint_type).lower()
    a = _unit(axis)
    sign = 1.0 if float(travel) >= 0 else -1.0
    if kind.startswith("prismatic"):
        return a * sign
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    centroid = v.mean(axis=0) if len(v) else np.asarray(origin, dtype=np.float64)
    return _unit(np.cross(a, centroid - np.asarray(origin, dtype=np.float64)) * sign)


def handle_on(vertices, lead, up=(0.0, 0.0, 1.0), jaw: float = HANDLE_JAW) -> dict:
    """What a moving link offers a parallel jaw on its leading face, read off its own vertices.

    ``vertices``: the link's mesh vertices (any frame); ``lead``: the direction its face looks along
    (``leading_direction``). The vertices are sliced into ``HANDLE_LAYER`` thick layers behind the front-most one,
    and the PANEL is the first layer from the front that spans the link's cross-section both ways
    (``HANDLE_PANEL_COVER``) -- not the densest layer, which on store_honey's drawers is the back of the rail
    handle and made a 2.4 cm rail read as a 9 mm lip. Whatever stands more than ``HANDLE_PROUD_MIN`` in front of
    the panel is the proud part, and its shape says what to do with it:

      bar   narrow enough to fit in the jaw across one axis and at least ``HANDLE_BAR_MIN`` proud: close AROUND
            it. ``point`` is its centre, ``jaw`` the unit vector across its narrow axis, ``along`` its long axis
            with ``span`` its length and ``ends`` (lo, hi) along it, so the caller may slide the grasp along it.
      lip   proud but too shallow to close around (a step, a bevel): the pressed-face grasp at ``point`` on the
            panel; ``lip_point`` is the middle of the proud part.
      flat  nothing proud, or an overlay as wide as the face: the pressed-face grasp at ``point``, on the
            front-most surface.

    Also ``panel`` and ``front`` (scalars along ``lead``), ``proud`` (front - panel), ``face_centre`` and
    ``face_extent`` (side, up) of the panel layer, and for a bar ``bar_depth`` (its extent along ``lead``).
    Pure numpy, so it is checked offline against the decrypted assets rather than in a simulator.
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    lead = _unit(lead)
    side = np.cross(lead, _unit(up))
    if float(np.linalg.norm(side)) < 1e-6:  # a lid that opens straight up: any horizontal axis will do
        side = np.cross(lead, [1.0, 0.0, 0.0])
    side = _unit(side)
    up2 = _unit(np.cross(side, lead))
    if len(v) < 3:
        centre = v.mean(axis=0) if len(v) else np.zeros(3)
        return {
            "kind": "flat", "point": centre, "panel": float(centre @ lead), "front": float(centre @ lead),
            "proud": 0.0, "face_centre": centre, "face_extent": (0.0, 0.0), "extent": (0.0, 0.0, 0.0),
            "jaw": None, "lead": lead, "side": side, "up": up2,
        }  # fmt: skip
    s = v @ lead
    front = float(s.max())
    width, height = float(np.ptp(v @ side)), float(np.ptp(v @ up2))
    panel, densest, densest_count = None, None, -1
    edge = front + HANDLE_LAYER / 2.0
    while edge - HANDLE_LAYER > front - HANDLE_DEPTH_RANGE:
        sel = v[(s >= edge - HANDLE_LAYER) & (s < edge)]
        edge -= HANDLE_LAYER
        if len(sel) < 4:
            continue
        if len(sel) > densest_count:
            densest, densest_count = float(np.mean(sel @ lead)), len(sel)
        if np.ptp(sel @ side) >= HANDLE_PANEL_COVER * width and np.ptp(sel @ up2) >= HANDLE_PANEL_COVER * height:
            panel = float(np.mean(sel @ lead))  # the layer's own mean, not its bin centre: 2 mm matter on a lip
            break
    if panel is None:
        panel = densest if densest is not None else front
    face = v[np.abs(s - panel) <= HANDLE_LAYER]
    if not len(face):
        face = v
    face_centre = face.mean(axis=0)
    face_centre = face_centre + lead * (panel - float(face_centre @ lead))
    face_extent = (float(np.ptp(face @ side)), float(np.ptp(face @ up2)))
    # The SHAPE of the proud part is read off its front half only: fridge/dszchb carries a full-height trim strip
    # 1 cm in front of its panel beside a bar 4.4 cm out, and together they span the whole door (2026-09-14).
    proud_verts = v[s > panel + max(HANDLE_PROUD_MIN, (front - panel) / 2.0)]
    out = {
        "panel": float(panel), "front": front, "proud": float(front - panel), "face_centre": face_centre,
        "face_extent": face_extent, "lead": lead, "side": side, "up": up2, "jaw": None,
    }  # fmt: skip
    if len(proud_verts) < 3:
        out.update(kind="flat", point=face_centre, proud=0.0, extent=(0.0, 0.0, 0.0))
        return out
    e_side, e_up, e_deep = (float(np.ptp(proud_verts @ d)) for d in (side, up2, lead))
    out["extent"] = (e_side, e_up, e_deep)
    proud_centre = proud_verts.mean(axis=0)
    narrow_is_side = e_side <= e_up
    narrow = e_side if narrow_is_side else e_up
    if front - panel >= HANDLE_BAR_MIN and narrow <= jaw - HANDLE_JAW_MARGIN:
        along = up2 if narrow_is_side else side
        along_s = proud_verts @ along
        out.update(
            kind="bar", point=proud_centre, jaw=side if narrow_is_side else up2, along=along,
            span=float(np.ptp(along_s)), ends=(float(along_s.min()), float(along_s.max())), bar_depth=e_deep,
        )  # fmt: skip
        return out
    if front - panel < HANDLE_BAR_MIN:
        out.update(kind="lip", point=face_centre, lip_point=proud_centre)
        return out
    # an overlay as wide as the face: the face to press on is the overlay's own front
    out.update(kind="flat", point=face_centre + lead * (front - panel))
    return out
