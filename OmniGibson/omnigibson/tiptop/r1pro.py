"""BEHAVIOR-1K R1Pro (the challenge robot) inside a BEHAVIOR scene as a TiPToP client.

TiPToP plans the torso and the left arm with its ``r1pro_left`` embodiment (tiptop/tiptop/embodiments/r1pro.py), a
cuRobo model generated from the very URDF and collision spheres OmniGibson ships for this robot. The right arm and
the fingers are locked in that model, so the simulator holds them at the same values, which it takes from the server
metadata (``embodiment.locked_joints`` / ``joint_names`` / ``q_home``) or, offline, from the embodiment's meta file
in the tiptop submodule.

World frame for TiPToP = the robot's ``base_link`` (floor level), which is both ``robot.get_position_orientation()``
here and the URDF root of the planner model. Capture uses the head camera (``zed_link``) or the left wrist camera,
in a look posture that swings the arm out of the camera's view; with ``--seg-instance`` the robot's own pixels are
also removed from the depth. Navigation is a stand-in: ``best_base_pose`` chooses where to stand for a set of
objects (in reach, in the camera's view, on free floor), and ``place_robot`` teleports the base there. What the
simulator knows and the robot could not (object poses, button poses, masks, a switch's state) is read only by
``knowledge.py``'s oracle source and by the base-pose search; both are privileged and say so.
"""

import logging
import math
import re
from pathlib import Path

import numpy as np
import torch as th
import yaml
from bddl.condition_evaluation import HEAD

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects.usd_object import USDObject
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.tiptop.gt_masks import masks_from_geometry, points_within_tol
from omnigibson.tiptop.kinematics import ArmIK, link_from_camera, link_pose_for_camera, look_pose
from omnigibson.tiptop.protocol import (
    add_view,
    bddl_category,
    face_normal_local,
    joint_ramp,
    points_to_pixels,
    via_configuration,
)
from omnigibson.tiptop.scene import (
    CAMERA_NAME,
    OBJECT_PRESETS,
    OVERVIEW_CAM,
    TiptopSim,
    look_at_quat_xyzw,
    overview_cam_config,
)

log = logging.getLogger(__name__)

ROBOT_NAME = "robot_r1"
ROBOT_TYPE = "r1pro_left"
# Capture views: name -> the robot camera's link. Every "head_*" is the head camera again with the torso moved
# (HEAD_VIEWS): an alternative to swinging the wrist cameras, which needs no room beside the robot.
CAMERA_LINKS = {
    "head": "zed_link",
    "left_wrist": "left_realsense_link",
    "right_wrist": "right_realsense_link",
    "head_left": "zed_link",
    "head_right": "zed_link",
    "head_up": "zed_link",
    "head_down": "zed_link",
}
VIEW_OPTICS = {  # which shadow camera renders a view
    "head": "head",
    "left_wrist": "wrist",
    "right_wrist": "wrist",
    "head_left": "head",
    "head_right": "head",
    "head_up": "head",
    "head_down": "head",
}
DEFAULT_VIEWS = ("left_wrist", "right_wrist")  # captured with the head camera and fused by the planner
# Two ways to move the head camera, both through the torso (the base never turns) and both bringing the arms
# along. Yaw: torso_joint4 rotates about the link the camera sits on, so +-29 degrees re-aims it from the same
# place (the camera is 9 cm off that axis, so it travels about 4 cm); three yaw views span roughly 150 degrees
# with the 99 degree head camera. In the leaning challenge posture the yaw axis is 23 degrees off the base's z
# (measured 2026-09-11). Pitch: torso_joint3 is below the camera's mast, so +-17 degrees moves the camera about
# 15 cm as well as tilting it, which is what gives a second viewpoint of a container's inside rather than the
# same viewpoint re-aimed. Either way the view's camera pose is read from the simulator as it is rendered.
HEAD_VIEW_YAW = 0.5  # rad the torso turns for a sideways head view
HEAD_VIEW_PITCH = 0.3  # rad the torso leans for a head view from higher or lower
HEAD_YAW_JOINT = "torso_joint4"  # the joint above the head camera: turning it aims the camera, moving it 4 cm
HEAD_PITCH_JOINT = "torso_joint3"  # the joint below the camera's 0.48 m mast: leaning it moves the camera ~15 cm
# view name -> (planned joint, how far it moves from the posture's target for that view). A yaw view re-aims the
# head from the same place; a pitch view moves it, which is what gives a second view of a container's inside.
HEAD_VIEWS = {
    "head_left": (HEAD_YAW_JOINT, HEAD_VIEW_YAW),
    "head_right": (HEAD_YAW_JOINT, -HEAD_VIEW_YAW),
    "head_up": (HEAD_PITCH_JOINT, HEAD_VIEW_PITCH),
    "head_down": (HEAD_PITCH_JOINT, -HEAD_VIEW_PITCH),
}
HEAD_VIEW_SETTLE_STEPS = 30  # after a yaw ramp: only the torso moved (the arms settle for LOOK_SETTLE_STEPS)
# The external capture sensors, one per optics: moved onto the robot camera's pose per view (_capture_obs). The
# robot's own cameras render rgb only (video, mirror); depth and segmentation come from these.
SHADOW_CAMS = {"head": CAMERA_NAME, "wrist": "tiptop_wrist_cam"}
SHADOW_CAM = SHADOW_CAMS["head"]
# Capture posture: the ready posture with the left shoulder abducted so the arm swings out to the robot's left, out of
# the head camera's view. In the ready posture the gripper sits in front of the table objects and hides most of them
# (a detector then segments the gripper); probed in Rs_int: mug 3881 px instead of 2005, bowl 8523 instead of 4994,
# 0 robot pixels, no contact. Applied on top of q_home, joint name -> value.
LOOK_ARM = {"left_arm_joint2": 2.0}
# Fractions of that sweep to try, smallest first, when the planned arm has to be taken out of the head camera's
# way and no look pose was found. The full sweep is a 1.74 rad shoulder abduction from the challenge home posture
# that carries the whole arm across the room, and it is the motion that swept a battery off a desk (2026-09-12);
# the purpose is only to clear the camera's line, so the smallest swing that does is taken instead.
LOOK_ARM_FRACTIONS = (0.3, 0.5, 0.75, 1.0)
LOOK_SETTLE_STEPS = 60
# A capture posture is reached by ramping the joint targets at no more than this speed (rad/s), one interpolated
# target per control step: a step change of the targets makes the position controller slam the arms, which shakes
# the whole robot and can shift the objects the capture is about to look at. Well under the arm joints' 7 rad/s.
CAPTURE_MAX_JOINT_VEL = 0.6
RAMP_BLOCK_TOL = 0.1  # rad: a ramped joint this far from its target is not following the ramp (blocked); logged
RAMP_BLOCK_STEPS = 5  # consecutive steps behind that tolerance before the ramp calls it blocked and stops
RAMP_MOVING_EPS = 1e-3  # rad: a joint whose target moves less than this over a ramp is being held, not ramped
# A wrist camera's look configuration must keep these links of its arm (name suffixes) out of the base's box, inflated
# by BASE_CLEARANCE: Lula IK knows no collisions, and with the torso leaning the first offset put the right hand on the
# base top, where it stayed for the whole capture (2026-09-11, assembling_gift_baskets: finger and wrist-camera links
# in contact with base_link, the joints slipping round it at their velocity limit).
HAND_LINKS = (
    "arm_link4",
    "arm_link6",
    "realsense_link",
    "gripper_link",
    "gripper_finger_link1",
    "gripper_finger_link2",
)
BASE_CLEARANCE = 0.10  # m
SCENE_CLEARANCE = 0.03  # m: a look configuration keeps the hand links this far outside every scene object's box
BLOCKED_SWINGS_MAX = 2  # capture swings stopped against something before an instance gives up on look poses
ELBOW = 3  # index of the elbow in an arm's joint list: folded before a capture swing, straightened after it
LOOK_TOL = 0.03  # rad: an arm this far from its look posture after settling is blocked; from the ready posture, wrong
# How far the torso may be from where it started after the head views before the round is abandoned. It is a
# separate number from LOOK_TOL because it is a different question, and because sharing LOOK_TOL's 0.03 put the
# abort threshold *inside* the normal settling distribution: over one putting_away_toys run (2026-09-13) the 32
# returns that succeeded spread 0.0000-0.0282 rad, and the three rounds lost to this check were "off by 0.030".
# Those were noise, not a torso that failed to return -- a real failure leaves a large fraction of the head view's
# own 0.3 rad delta behind, which this still catches.
HEAD_VIEW_RETURN_TOL = 0.10
# An arm link whose origin passes this close (m) to the head camera's line of sight to the look target is taken to
# block it. About the half width of the gripper, which is the widest thing on the arm; the test is on link origins,
# so a link is a point and this radius stands in for its mesh. The arm in front of the target does not merely darken
# it: the self-mask zeroes the robot's own pixels, so the target comes back with no depth and an empty mask.
LOOK_BLOCK_RADIUS = 0.08
# The whole arm against the scene (arm_hits_scene): the arm is taken as the polyline through its link origins and a
# scene box is grown by ARM_RADIUS before the segments are tested against it, which stands in for the limbs' own
# thickness -- the R1Pro's upper arm and forearm are about 0.1 m across. PATH_SAMPLES configurations are checked
# along a motion, because the endpoint being clear says nothing about what the arm sweeps through on the way.
ARM_RADIUS = 0.06
PATH_SAMPLES = 9
ARM_SAMPLE_STEP = 0.02  # m between the points taken along the arm when it is measured against an object's mesh
# A box is a loose model of furniture: measured on 2026-09-13 (dispose_of_batteries, scratchpad/arm_clearance.py),
# at the READY posture -- where the arm has just teleported in and is touching nothing -- the boxes of a swivel
# chair and a desk both contain the hand, and the existing hand-only test (links_in_scene) calls it a collision.
# So a box decides nothing on its own: it is the cheap prefilter, and the object's own mesh decides.
ARM_MESH_CHECK = True
DEFAULT_LOOK_TARGET = (0.6, 0.0, 0.85)  # base frame: what the wrist cameras look at when no base pose was chosen
# The planner's box starts this far ahead of the base frame: past the base (its front collision spheres reach x 0.25)
# and the leaning torso, so the support plane the wrist cameras see beside the robot never runs under it. The head
# camera alone never saw the table nearer than 0.40 m; the wrist cameras see the floor from 0.14 m and the base's top
# at table height, and a support cuboid reaching there puts the robot's start posture in collision (pass 4).
WORKSPACE_NEAR = 0.35
SELF_MASK_FACES = (
    2000  # a robot link's mesh is decimated to this for the self-mask (it only marks the robot's own pixels)
)
# Where a wrist camera looks at the target from: (ahead, aside on the arm's side, up) from its arm's shoulder (m), the
# first the arm can reach (Lula IK on the R1Pro URDF, 2026-09-10: the first alone reaches every test target in the
# upright posture; with the torso leaning the second takes most, and the four together reach nine targets in ten)
LOOK_OFFSETS = ((0.2, 0.3, -0.05), (0.1, 0.25, -0.1), (0.0, 0.25, -0.1), (0.1, 0.15, -0.3))
# Where a held object is put so the head camera sees it (base frame, the y on the holding arm's side): about
# 0.25 m ahead of the camera and 0.3 m below it, which is on the head camera's optical axis in the challenge
# posture. Only used when the capture has no wrist view to look at the hand: at the ready posture the gripper is
# below the head camera's frame, so a carry planned from head views alone has no picture of what it carries
# (putting_away_toys with --views head_up head_down, 2026-09-12).
PRESENT_POINT = (0.62, 0.18, 1.00)
# Tried in order from the point itself: higher and further out (over a container the robot stands at), wider to
# the arm's side, lower and nearer. Each stays within about 0.45 m of the head camera and inside its frame.
PRESENT_OFFSETS = (
    (0.0, 0.0, 0.0),
    (0.08, 0.0, 0.12),
    (0.16, 0.0, 0.22),
    (0.0, 0.10, 0.06),
    (0.10, 0.12, 0.16),
    (-0.06, 0.0, -0.08),
    (0.0, -0.08, 0.04),
    (-0.10, 0.10, 0.10),
)
CAPTURE_MAX_RENDERS = 40  # render pairs after moving the capture camera (temporal accumulation)
CAPTURE_CONVERGED_DIFF = 0.25  # mean absolute rgb change (0-255) between consecutive renders that counts as settled
HEAD_APERTURE_MM = 40.0  # BEHAVIOR challenge eval setting (99 deg HFOV); OmniGibson's default 20.995 gives 63 deg
WRIST_APERTURE_MM = 20.995  # OmniGibson VisionSensor default, set explicitly so the shadow camera matches exactly
FLOOR_COVERINGS = ("floors", "ceilings", "paver", "carpet", "rug", "mat", "doormat", "tile")  # stood on, not avoided
ROBOT_HEIGHT = 1.6  # m, top of the head camera with the challenge torso posture is ~1.4
ROBOT_FOOTPRINT = 0.36  # half extent (m) used for free-space checks; base bbox is 0.64 x 0.68
CAMERA_MIN_MARGIN = 0.08  # added to where the bottom image edge meets an object's support: room to be whole.
# Raised to 0.15 m on 2026-09-12 on the theory that a battery kept coming out of the capture with an empty mask
# because it sat just past the frame's bottom edge, and put back: the wider margin moved the stance from 0.55 m
# to 0.60 m and the mask was still empty (runs/bench_batteries_6 against _5), so the object is hidden at those
# stances for another reason and the tighter margin only costs reach. What actually recovers the round is the
# retry from a different pose, which finds the battery every time (493 mask pixels at the stance that works).

TARGET_HALF_WIDTH = 0.22  # containers this wide (basket) hide an item behind them from the head camera
FRAMING_PENALTY = 2.0  # score cost per radian an object's edge falls outside the frame (see best_base_pose)
# best_base_pose projects the objects themselves into the head camera a candidate stance would have (see
# frame_objects). The image border is taken FRAME_MARGIN_PX pixels in, so an object counted as framed has a little
# room either side rather than touching the edge, and every pixel an object still falls outside costs
# FRAMING_PENALTY_PX: 0.006 is FRAMING_PENALTY per radian divided by the head camera's focal length in pixels
# (308 at 99 deg over 720), so a clipped object keeps about the weight the angle measure gave it.
FRAME_MARGIN_PX = 12
FRAMING_PENALTY_PX = 0.006
# best_base_pose: the candidate grid around the objects' centroid and the score terms (lower is better)
RING_START, RING_STEP = 0.25, 0.05  # m, rings out to the arm's reach
RING_ANGLE_STEP = np.pi / 18  # 10 deg around the centroid
YAW_OFFSETS = np.arange(-np.pi / 3, np.pi / 3 + 1e-6, np.pi / 12)  # facing the centroid +-60 deg, 15 deg steps
MIN_AHEAD = 0.15  # m every object must be ahead of the base at least
MIN_SIDE = -0.30  # m to the right at most (the torso can turn a little); further left is preferred
SIDE_TARGET, SIDE_WEIGHT = 0.15, 0.5  # objects less than SIDE_TARGET m to the left cost SIDE_WEIGHT per m short
YAW_WEIGHT = 0.1  # per radian of turning away from the centroid
HIDE_DEPTH, HIDE_MARGIN = 0.05, 0.06  # m: a container nearer by less than HIDE_DEPTH hides an item behind it, as
# seen from the camera, when their bearings are within its angular half-width plus an item margin of HIDE_MARGIN
# _footprint_free: what an AABB in the footprint means
HOUSE_AABB_AREA = 20.0  # m^2; larger boxes are merged walls, roofs or ceilings and say nothing about the floor
FLAT_COVERING_HEIGHT, GROUND_CLEARANCE = 0.08, 0.05  # m; boxes flatter than that on the ground are stood on
# place_robot: the overview camera in the base's frame, (eye dx, eye dy, eye z, target dx, target z); "shoulder" looks
# over the left shoulder at the workspace, "front" looks back at the chest so both hands and what they hold are in view
OVERVIEW_OFFSETS = {"shoulder": (-1.5, 1.1, 1.7, 0.7, 0.55), "front": (1.15, -0.75, 1.35, 0.3, 0.9)}
AVOID_RADIUS = 0.15  # a retried base pose must be at least this far (m) from the ones tried before
BASE_MASS_KG = 250.0  # omnigibson/eval/evaluator.py sets this for r1/r1pro; keeps the robot upright
SUPPORT_CATEGORIES = ("table", "floor")  # BDDL supports a goal may name; the planner knows the plane under the objects
PLANNER_SUPPORT = "table"  # the planner's label for that plane (tiptop's RANSAC "table", a floor when standing at one)


def blocks_ray(eye, target, point, radius: float) -> bool:
    """Whether ``point`` sits within ``radius`` of the segment from ``eye`` to ``target`` (all in one frame).

    A point beside the line, behind the eye or past the target does not block the view of the target.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    ray = np.asarray(target, dtype=np.float64).reshape(3) - eye
    length = float(np.linalg.norm(ray))
    if length < 1e-6:
        return False
    rel = np.asarray(point, dtype=np.float64).reshape(3) - eye
    along = float(rel @ (ray / length))
    return 0.0 < along < length and float(np.linalg.norm(rel - along * ray / length)) < radius


def segment_hits_box(eye, target, lo, hi) -> bool:
    """Whether the segment from ``eye`` to ``target`` passes through the axis-aligned box (slab method).

    Exact for a box, and a link's box is what the camera actually sees of it -- unlike a link origin, which is a
    point and misses a gripper whose fingers reach well past it.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    ray = np.asarray(target, dtype=np.float64).reshape(3) - eye
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    near, far = 0.0, 1.0  # the segment as a fraction of ray
    for axis in range(3):
        if abs(ray[axis]) < 1e-12:
            if eye[axis] < lo[axis] or eye[axis] > hi[axis]:
                return False  # parallel to this slab and outside it
            continue
        t1 = (lo[axis] - eye[axis]) / ray[axis]
        t2 = (hi[axis] - eye[axis]) / ray[axis]
        near, far = max(near, min(t1, t2)), min(far, max(t1, t2))
        if near > far:
            return False
    return True


def polyline_hits_box(points, lo, hi, clearance: float = 0.0) -> bool:
    """Whether a polyline (a list of points, in order) enters the axis-aligned box grown by ``clearance``.

    The arm modelled as the chain through its link origins: a limb is a segment, and growing the box stands in for
    the limb's own thickness.
    """
    low = np.asarray(lo, dtype=np.float64).reshape(3) - clearance
    high = np.asarray(hi, dtype=np.float64).reshape(3) + clearance
    return any(segment_hits_box(a, b, low, high) for a, b in zip(points, points[1:]))


def sample_polyline(points, step: float) -> np.ndarray:
    """Points along a polyline at no more than ``step`` apart, the corners included; (N, 3)."""
    points = [np.asarray(p, dtype=np.float64).reshape(3) for p in points]
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        n = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / max(step, 1e-6))))
        out += [a + (b - a) * (i / n) for i in range(1, n + 1)]
    return np.asarray(out, dtype=np.float64)


def box_corners(lo, hi) -> np.ndarray:
    """The 8 corners of an axis-aligned box given as its low and high xyz, as (8, 3)."""
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    return np.array(
        [[hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]] for i in range(8)],
        dtype=np.float64,
    )


def frame_objects(
    corners, intrinsics, base_from_cam, base_z, width, height, x, y, yaw, strict=True, margin_px=FRAME_MARGIN_PX
):
    """Project the objects into the head camera a stance at (``x``, ``y``, ``yaw``) would have.

    The head camera is rigid with the base for a given torso posture, so its pose in the base frame
    (``base_from_cam``, OpenCV axes, from ``R1ProSim.head_camera_in_base``) is the same wherever the robot stands:
    putting each object's world box into the candidate's base frame and projecting through it is exactly the
    picture the capture will take. ``corners``: one (8, 3) world box per object (``box_corners``). ``base_z``: the
    world height of the base frame, so a world z becomes a base z.

    Returns (reason to reject the stance or None, pixels an object's corners fall outside the image). A stance is
    rejected when an object is behind the camera, or when an object small enough to fit inside the frame at that
    distance is cut by its border -- the measured cause of a lost round is a battery projecting to row 791 of a
    720-row head image (2026-09-12, dispose_of_batteries), and an item the planner has to grasp is worth nothing
    half seen. An object too big to fit is only penalised by the pixels it falls outside: no stance frames a toy
    box and the toy beside it whole, and a clipped container is worth more than no stance at all. ``strict=False``
    turns every cut into a penalty: what ``best_base_pose`` falls back to when no stance frames the objects whole.
    """
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    here = np.array([x, y], dtype=np.float64)
    outside = 0.0
    for box in corners:
        rel = np.asarray(box, dtype=np.float64)[:, :2] - here
        pts = np.stack([rel @ fwd, rel @ left, np.asarray(box, dtype=np.float64)[:, 2] - base_z], axis=-1)
        px, z = points_to_pixels(pts, intrinsics, base_from_cam)
        if np.any(z <= 0):
            return "behind the head camera", 0.0
        lo, hi = px.min(axis=0), px.max(axis=0)
        low = np.array([margin_px, margin_px], dtype=np.float64)
        high = np.array([width - 1 - margin_px, height - 1 - margin_px], dtype=np.float64)
        cut = float(np.sum(np.maximum(0.0, low - lo) + np.maximum(0.0, hi - high)))
        if strict and cut > 0.0 and np.all(hi - lo <= high - low):  # it would fit in the frame; this stance cuts it
            return "outside the head camera's frame", 0.0
        outside += cut
    return None, outside


def embodiment_meta_path(robot_type: str = ROBOT_TYPE) -> Path:
    """Generated meta file of the tiptop embodiment (tiptop submodule), the offline source of the locked posture."""
    repo = Path(og.__file__).resolve().parents[2]
    return repo / "tiptop" / "tiptop" / "embodiments" / "assets" / "r1pro" / f"{robot_type}_meta.yml"


def load_embodiment_meta(robot_type: str = ROBOT_TYPE) -> dict:
    path = embodiment_meta_path(robot_type)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `pixi run python scripts/make_r1pro_embodiment.py` in tiptop/")
    with open(path) as f:
        return yaml.safe_load(f)


def challenge_task_info(activity: str) -> tuple[str, list[str]]:
    """Scene model and room instances the challenge evaluator loads for a task (its metadata files)."""
    from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_ROOMS

    tasks = yaml.safe_load(
        open(Path(gm.DATA_PATH) / "2026-challenge-task-instances" / "metadata" / "available_tasks.yaml")
    )
    if activity not in tasks:
        raise ValueError(f"{activity!r} is not a challenge task; known: {sorted(tasks)[:5]}... ({len(tasks)})")
    return str(tasks[activity][0]["scene_model"]), list(TASK_NAMES_TO_ROOMS[activity])


def bddl_predicate_class(name: str):
    """The BDDL predicate class for a predicate name as written in a task definition ('ontop', 'toggled_on')."""
    from omnigibson.utils.bddl_utils import PREDICATE_TO_STATE

    key = name.replace("_", "").lower()
    for cls in PREDICATE_TO_STATE:
        if cls.__name__.lower() == key:
            return cls
    raise ValueError(f"unknown BDDL predicate {name!r}; known: {sorted(c.__name__ for c in PREDICATE_TO_STATE)}")


def detector_phrase(bddl_name: str) -> str:
    """'butter_cookie.n.01_2' -> 'butter cookie', 'can__of__soda.n.01_1' -> 'can of soda' (what a detector is asked for)."""
    return bddl_category(bddl_name).replace("__", "_").replace("_", " ")


def label_category(bddl_name: str) -> str:
    """'butter_cookie.n.01_2' -> 'butter_cookie', 'can__of__soda.n.01_1' -> 'can_of_soda': the category as it appears
    in request labels (single underscores)."""
    return detector_phrase(bddl_name).replace(" ", "_")


def bddl_label(bddl_name: str) -> str:
    """'butter_cookie.n.01_2' -> 'butter_cookie_2': the per-instance name used in requests and plans."""
    category, _, index = bddl_name.rpartition("_")
    return f"{label_category(category)}_{index}"


def make_r1pro_env_config(
    scene_model: str = "Rs_int",
    load_room_types=None,
    spawn_presets=(),
    grasping_mode: str = "sticky",
    camera: str = "head",
    views=DEFAULT_VIEWS,
    head_resolution: int = 720,
    wrist_resolution: int = 480,
    head_aperture_mm: float = HEAD_APERTURE_MM,
    not_load_object_categories=("ceilings",),
    activity: str | None = None,
    activity_instance_id: int = 0,
    load_room_instances=None,
    segmentation: bool = True,
    max_steps: int = 10**8,
) -> dict:
    """OmniGibson config: BEHAVIOR scene + R1Pro with absolute joint controllers on every group.

    Spawned objects start high above the floor and are placed onto furniture by R1ProSim.place_on(). With
    ``activity`` the scene is a challenge task instance (BehaviorTask, pre-sampled objects, the evaluator's rooms).
    ``segmentation=False`` renders rgb + depth only (what the challenge allows); oracle masks then come from object
    geometry instead of the annotator, and the robot is not masked out of the depth. ``max_steps`` is the task's own
    timeout (env steps), the challenge's 1.5x mean human demonstration length in a benchmark; effectively none
    otherwise.

    Cameras: the robot cameras render rgb only (video); an external "shadow" VisionSensor per optics (head:
    ``head_resolution`` / ``head_aperture_mm``; wrist: ``wrist_resolution`` / ``WRIST_APERTURE_MM``) provides rgb +
    depth_linear + seg_instance for a capture view after being moved onto the robot camera's pose; ``camera`` is
    the primary view, ``views`` the further ones (``CAMERA_LINKS``). Instance segmentation attached to a
    robot-mounted camera leaks GPU memory every step in this Isaac Sim build and segfaults the synthetic-data
    graph after ~35 steps; an external camera does not.
    """
    jc = {
        "name": "JointController",
        "motor_type": "position",
        "use_delta_commands": False,
        "use_impedances": False,
        "command_input_limits": None,
        "command_output_limits": None,
    }
    gripper = {
        "name": "MultiFingerGripperController",
        "mode": "binary",
        "command_input_limits": None,
        "command_output_limits": None,
    }
    unknown = [v for v in (camera, *views) if v not in CAMERA_LINKS]
    if unknown:
        raise ValueError(f"unknown camera views {unknown} (known: {sorted(CAMERA_LINKS)})")
    if camera in HEAD_VIEWS:
        raise ValueError(f"{camera!r} is the head camera turned; the primary view is taken at torso yaw 0 ('head')")
    optics = {"head": (head_resolution, head_aperture_mm), "wrist": (wrist_resolution, WRIST_APERTURE_MM)}
    shadow_cams = [
        {
            "sensor_type": "VisionSensor",
            "name": SHADOW_CAMS[kind],
            "relative_prim_path": f"/{SHADOW_CAMS[kind]}",
            "modalities": ["rgb", "depth_linear"] + (["seg_instance"] if segmentation else []),
            "sensor_kwargs": {
                "image_width": optics[kind][0],
                "image_height": optics[kind][0],
                "focal_length": 17.0,
                "horizontal_aperture": optics[kind][1],
            },
            "position": [0.0, 0.0, 1.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "include_in_obs": False,
        }
        for kind in ("head", "wrist")
        if kind in {VIEW_OPTICS[v] for v in (camera, *views)}
    ]
    scene = {
        "type": "InteractiveTraversableScene",
        "scene_model": scene_model,
        "include_robots": False,
        "not_load_object_categories": list(not_load_object_categories),
    }
    if load_room_types:
        scene["load_room_types"] = list(load_room_types)
    if load_room_instances:
        scene["load_room_instances"] = list(load_room_instances)
    task = {"type": "DummyTask"}
    if activity:
        task = {
            "type": "BehaviorTask",
            "activity_name": activity,
            "activity_definition_id": 0,
            "activity_instance_id": activity_instance_id,
            "online_object_sampling": False,
            "debug_object_sampling": False,
            "highlight_task_relevant_objects": False,
            "termination_config": {"max_steps": int(max_steps)},
            "reward_config": {"r_potential": 1.0},
            "include_obs": False,
        }
    objects = []
    for i, preset in enumerate(spawn_presets):
        if preset not in OBJECT_PRESETS:
            raise ValueError(f"unknown object preset {preset!r}; known: {sorted(OBJECT_PRESETS)}")
        objects.append({**OBJECT_PRESETS[preset], "name": preset, "position": [0.0, 0.0, 3.0 + 0.3 * i]})
    return {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
            "external_sensors": [*shadow_cams, overview_cam_config()],  # the overview is aimed by place_robot
        },
        "scene": scene,
        "robots": [
            {
                "model": "r1pro",
                "name": ROBOT_NAME,
                "obs_modalities": [
                    "rgb",
                    "proprio",
                ],  # rgb for the video + mirror; capture frames come from the shadow cameras
                "include_sensor_names": sorted(set(CAMERA_LINKS.values())),
                "action_normalize": False,
                "self_collisions": True,
                "grasping_mode": grasping_mode,
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_height": wrist_resolution,
                            "image_width": wrist_resolution,
                            "focal_length": 17.0,
                            "horizontal_aperture": WRIST_APERTURE_MM,
                        }
                    },
                    f"{CAMERA_LINKS['head']}:Camera:0": {
                        "sensor_kwargs": {
                            "image_height": head_resolution,
                            "image_width": head_resolution,
                            "horizontal_aperture": head_aperture_mm,
                        }
                    },
                },
                "controller_config": {
                    "base": {
                        "name": "HolonomicBaseJointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                    "trunk": dict(jc),
                    "arm_left": dict(jc),
                    "arm_right": dict(jc),
                    "gripper_left": dict(gripper),
                    "gripper_right": dict(gripper),
                },
            }
        ],
        "objects": objects,
        "task": task,
    }


def _intrinsics(sensor, tries: int = 10) -> np.ndarray:
    """The sensor's intrinsic matrix as an array; renders and retries while the render product has not produced
    camera parameters yet (OmniGibson asserts on the degenerate matrix it reads then; seen once, right after the
    posture was applied, with the GPU nearly full)."""
    for i in range(tries):
        try:
            return sensor.intrinsic_matrix.cpu().numpy()
        except AssertionError:
            log.warning(f"{sensor.name}: no camera parameters yet (render {i + 1}/{tries})")
            og.sim.render()
    return sensor.intrinsic_matrix.cpu().numpy()


def turned_joints(planned_joints, q_arm, joint: str, delta: float) -> list[float]:
    """``q_arm`` (the planned joints' targets, in ``planned_joints`` order) with ``joint`` moved by ``delta``: the
    torso turned or leaned for a head view, every other joint where it is. The joint must be planned
    (``r1pro_left`` plans the torso); a locked joint's value lives in the posture instead."""
    planned_joints = list(planned_joints)
    if joint not in planned_joints:
        raise ValueError(f"{joint} is not a planned joint ({planned_joints}); a turned head view needs it")
    if len(q_arm) != len(planned_joints):
        raise ValueError(f"{len(q_arm)} joint targets for {len(planned_joints)} planned joints")
    q = [float(v) for v in q_arm]
    q[planned_joints.index(joint)] += float(delta)
    return q


class R1ProSim(TiptopSim):
    """R1Pro in a BEHAVIOR scene; the TiptopSim interface (capture / step / q_arm / objects) for the left arm."""

    WORKSPACE = ((WORKSPACE_NEAR, -0.80, 0.25), (1.30, 0.80, 1.60))

    expect_table_z = None  # no synthetic table at base z = 0: validate_capture only checks the objects
    mask_labels_as_invalid = (ROBOT_NAME,)

    def __init__(
        self,
        config: dict,
        camera: str = "head",
        views=DEFAULT_VIEWS,
        overview_view: str = "shoulder",
        look_arm=LOOK_ARM,
    ):
        """``camera``: the primary view; ``views``: the further views of every capture (``CAMERA_LINKS``; the
        environment must have been configured with the same, ``make_r1pro_env_config``); ``overview_view``: where
        ``place_robot`` puts the overview camera (``OVERVIEW_OFFSETS``); ``look_arm``: joint overrides on top of
        q_home for the capture, None to capture in the ready posture."""
        if camera in HEAD_VIEWS:
            raise ValueError(f"{camera!r} is the head camera turned; the primary view is taken at torso yaw 0 ('head')")
        self.config = config
        self.overview_view = overview_view
        self.look_arm = look_arm
        self.primary_view = camera
        self.extra_views = tuple(v for v in views if v != camera)
        self.env = og.Environment(configs=config)
        self.robot = self.env.robots[0]
        self.arm = "left"
        self.goal_initial = None  # the goal predicates' values when the episode began (mark_goal_initial)
        self.other_arm = "right"
        self.other_gripper = self.OPEN  # the arm that is not planned keeps this gripper command (see adopt_embodiment)
        self.last_gripper = self.OPEN
        self.mirror_arm_idx = None  # set when the planned arm changes: the Rerun mirror keeps the first embodiment
        self.mirror_gripper_idx = None
        self.joint_index = {name: i for i, name in enumerate(self.robot.joints.keys())}
        self.planned_joints = list(self.robot.arm_joint_names[self.arm])  # replaced by apply_posture (torso + arm)
        self.arm_idx = th.tensor([self.joint_index[j] for j in self.planned_joints])
        self.gripper_idx = self.robot.gripper_control_idx[self.arm]
        self.dt = og.sim.get_sim_step_dt()
        sensor_names = {name: f"{self.robot.name}:{link}:Camera:0" for name, link in CAMERA_LINKS.items()}
        self.robot_cam_names = {n: s for n, s in sensor_names.items() if n not in HEAD_VIEWS}  # one per camera
        self.robot_cams = {name: self.robot.sensors[sensor] for name, sensor in sensor_names.items()}  # per view
        self.cam_name = self.robot_cam_names[camera]
        self.robot_cam = self.robot_cams[camera]  # the primary view's camera: the base-pose search frames with it
        self.STREAM_CAMERA = f"{camera}_cam"  # the capture camera's image in the Rerun mirror
        self.shadows = {
            kind: self.env.external_sensors[name]
            for kind, name in SHADOW_CAMS.items()
            if name in self.env.external_sensors
        }
        self.cam = self.shadows[VIEW_OPTICS[camera]]  # capture camera; moved onto robot_cam's pose per frame
        # the wrist cameras' constant poses in their links, for the look poses (wrist_look), and the URDF's joints
        self.camera_in_link = {}
        for arm in ("left", "right"):
            link = self.robot.links[CAMERA_LINKS[f"{arm}_wrist"]]
            self.camera_in_link[arm] = link_from_camera(
                *[v.cpu().numpy() for v in link.get_position_orientation()],
                *[v.cpu().numpy() for v in self.robot_cams[f"{arm}_wrist"].get_position_orientation()],
            )
        self.urdf_joints = set(re.findall(r'<joint name="([^"]+)"', Path(self.robot.urdf_path).read_text()))
        self.look_target = None  # base-frame point the wrist cameras look at in a capture (place_robot_for sets it)
        self.look_names = ()  # the objects it was chosen for: one of them in a hand is looked at there instead
        self._base_box = None  # base_link's bounding box in the base frame (constant; measured on first use)
        self.overview = self.env.external_sensors.get(OVERVIEW_CAM)
        self.objects = {}
        self.context = {}  # furniture shown in the Rerun mirror (track_context)
        self.bddl_names = {}  # tiptop label -> BDDL instance name for tracked task objects
        self.posture = {}
        self.q_home = None
        self.blocked_swings = 0  # capture swings stopped against something this instance (see capture)
        self._scene_meshes = {}  # object name -> (box stamp, world trimesh) for the arm's collision check
        self._init_state()
        # The challenge evaluator (and JoyLo) give the base 250 kg; with the asset's default mass the leaning
        # challenge torso posture tips the whole robot over backwards.
        self.robot.base_footprint_link.mass = BASE_MASS_KG
        log.info(
            f"R1Pro DOF order: {list(self.joint_index)}; left arm idx {self.arm_idx.tolist()}, "
            f"gripper idx {self.gripper_idx.tolist()}, action dim {self.robot.action_dim}, camera {self.cam_name}"
        )
        assert list(self.robot.controller_order) == [
            "base",
            "trunk",
            "arm_left",
            "gripper_left",
            "arm_right",
            "gripper_right",
        ]
        self.env.reset()

    # ---------------------------------------------------------------- challenge task
    def task_scope(self) -> dict:
        """BDDL instance name -> simulated object for the loaded BehaviorTask (no agent, no floors, no systems)."""
        task = self.env.task
        scope = task.object_scope if isinstance(task, BehaviorTask) else {}  # a DummyTask has no scope
        return {
            k: v
            for k, v in scope.items()
            if isinstance(v, USDObject) and not k.startswith(("agent.", "floor."))  # systems have no pose
        }

    def floor_name(self) -> str:
        """The BDDL name of the loaded task's floor (``task_scope`` leaves floors out; a strategy puts things down
        on it when nothing else will do)."""
        names = [name for name in self.env.task.object_scope if name.startswith("floor.")]
        if not names:
            raise KeyError(f"the task {self.config['task'].get('activity_name')!r} has no floor in its scope")
        return names[0]

    def track_task_objects(self, skip_categories=("table", "floor", "agent")) -> dict:
        """Track every task object under its per-instance label ('candle_1'); furniture the items rest on is skipped."""
        self.bddl_names = {}
        for bddl, obj in self.task_scope().items():
            if label_category(bddl) in skip_categories:
                continue
            label = bddl_label(bddl)
            self.objects[label] = obj
            self.bddl_names[label] = bddl
        log.info(f"tracking task objects: {self.bddl_names}")
        return dict(self.bddl_names)

    def tiptop_goal(self, atoms: list[dict], category_level: bool) -> tuple[list[str], list[dict]]:
        """Translate BDDL goal atoms (inside/ontop/on/nextto/holding/toggled_on over BDDL names) for TiPToP.

        Returns the labels the request names and the atoms in TiPToP's predicates. Per instance ('candle_1', with
        oracle masks) or per category ('candle': the detector finds every instance and the goal takes the
        best-scoring one, since the task does not care which candle goes into which basket). toggled_on(obj)
        becomes pressed(<label>_button): the button is described by pose (button_hints) or found by the detector.
        A table or floor named as a support becomes the planner's own support plane, "table" (whatever horizontal
        plane the objects in view rest on), so an object can be put down where the robot stands.
        """
        predicates = {
            "inside": "on",
            "ontop": "on",
            "on": "on",
            "nextto": "near",
            "holding": "holding",
            "toggled_on": "pressed",
        }
        label_of = {bddl: label for label, bddl in self.bddl_names.items()}

        def name(arg):
            if label_category(arg) in SUPPORT_CATEGORIES:
                return PLANNER_SUPPORT  # the planner's name for the support plane under the objects it sees
            if arg not in label_of:
                raise ValueError(f"goal names {arg!r}, which is not a tracked task object: {sorted(label_of)}")
            return label_category(arg) if category_level else label_of[arg]

        out = []
        for atom in atoms:
            if atom["predicate"] not in predicates:
                raise ValueError(f"unsupported goal predicate {atom['predicate']!r} ({sorted(predicates)})")
            args = [name(a) for a in atom["args"]]
            if atom["predicate"] == "toggled_on":
                args = [self.button_label(a) for a in args]
            out.append({"predicate": predicates[atom["predicate"]], "args": args})
        if category_level:
            labels = sorted({label_category(b) for b in self.bddl_names.values()})
        else:
            labels = sorted(self.bddl_names)
        return labels, out

    def _goal_values(self) -> list[list[bool]]:
        """Truth of every predicate of every ground goal option, evaluating each grounded predicate once.

        forpairs goals ground into every pairing (331,776 options for four baskets); the simulator
        predicates (inside, ontop) are the expensive part, so they are memoized across options.
        """
        task = self.env.task
        leaf_cache, head_cache = {}, {}

        def evaluate(name, *entities):
            key = (name, entities)
            if key not in leaf_cache:
                leaf_cache[key] = task._evaluate_predicate(name, *entities)
            return leaf_cache[key]

        values = []
        for option in task.ground_goal_state_options:
            row = []
            for head in option:
                v = head_cache.get(id(head))
                if v is None:
                    v = head_cache[id(head)] = bool(head.evaluate(evaluate))
                row.append(v)
            values.append(row)
        return values

    @staticmethod
    def _goal_name(head) -> str:
        """'inside(candle.n.01_1, wicker_basket.n.01_2)' for a ground atom; other compiled forms print as they are."""
        if not isinstance(head, HEAD):
            return str(head)
        return f"{head.terms[0]}({', '.join(head.terms[1:])})"

    def holds(self, predicate: str, *bddl_names: str) -> bool:
        """Evaluate any BDDL predicate ("ontop", "inside", "toggled_on", ...) over task objects with the task's own
        evaluator (privileged: the simulator's object states; a benchmark strategy uses it where the pipeline has
        no perception of its own yet)."""
        return bool(self.env.task._evaluate_predicate(bddl_predicate_class(predicate), *bddl_names))

    @staticmethod
    def button_label(label: str) -> str:
        """Request label of an object's toggle button ('radio_receiver_1' -> 'radio_receiver_1_button')."""
        return f"{label}_button"

    def button_world(self, bddl: str) -> tuple[np.ndarray, np.ndarray, float]:
        """A toggle button as the simulator knows it (privileged): world position, the outward unit normal of the
        object face it sits on (from the object's own mesh), and the radius within which ToggledOn counts a finger."""
        from omnigibson.object_states import ToggledOn

        obj = self.scene_object(bddl)
        if ToggledOn not in obj.states:
            raise ValueError(f"{bddl} has no toggle button (no ToggledOn state)")
        state = obj.states[ToggledOn]
        pos_w = state.link.get_position_orientation()[0].cpu().numpy().astype(np.float64)
        obj_pos, obj_quat = obj.get_position_orientation()
        rot = T.quat2mat(obj_quat).cpu().numpy().astype(np.float64)
        vertices, _ = self.mesh_local(obj)  # the object's own frame
        p_local = rot.T @ (pos_w - obj_pos.cpu().numpy().astype(np.float64))
        n_world = rot @ face_normal_local(vertices, p_local)
        radius = float(th.min(state.visual_marker.extent * state.scale * state.link.scale))
        return pos_w, n_world, radius

    def button_hints(self, atoms: list[dict], category_level: bool = False) -> dict:
        """The toggle button of every toggled_on goal object, for the request's gt_buttons (privileged): base-frame
        position, outward normal of the face it sits on, and the radius within which ToggledOn counts a finger."""
        out = {}
        for atom in atoms:
            if atom["predicate"] != "toggled_on":
                continue
            (bddl,) = atom["args"]
            pos_w, n_world, radius = self.button_world(bddl)
            identity = th.tensor([0.0, 0.0, 0.0, 1.0])
            pos_b, _ = self.to_base(th.tensor(pos_w, dtype=th.float32), identity)
            tip_b, _ = self.to_base(th.tensor(pos_w + n_world, dtype=th.float32), identity)
            label = label_category(bddl) if category_level else self.label_of(bddl)
            out[self.button_label(label)] = {
                "position": [float(v) for v in pos_b],
                "normal": [float(v) for v in (tip_b - pos_b)],
                "radius": radius,
            }
            log.info(
                f"button of {bddl}: {self.button_label(label)} at {np.round(out[self.button_label(label)]['position'], 3).tolist()} "
                f"(base), normal {np.round(out[self.button_label(label)]['normal'], 2).tolist()}, "
                f"radius {out[self.button_label(label)]['radius']:.3f} m"
            )
        return out

    def tracked_label(self, name: str) -> str:
        """The tracked label of an object named in a goal atom: BDDL name -> per-instance label; a label stays."""
        return self.label_of(name) if name in self.bddl_names.values() else name

    def label_of(self, bddl: str) -> str:
        """Request label of a tracked task object ('radio_receiver.n.01_1' -> 'radio_receiver_1')."""
        for label, name in self.bddl_names.items():
            if name == bddl:
                return label
        raise KeyError(f"{bddl} is not a tracked task object: {sorted(self.bddl_names.values())}")

    def toggled(self, bddl: str) -> bool:
        """The switch's ToggledOn state (privileged: the task's own predicate; read by the oracle knowledge source)."""
        from omnigibson.object_states import ToggledOn

        return bool(self.scene_object(bddl).states[ToggledOn].get_value())

    def press_state(self, bddl: str) -> dict:
        """ToggledOn value and how many consecutive steps a finger has been on the button (5 flip it)."""
        from omnigibson.object_states import ToggledOn

        state = self.scene_object(bddl).states[ToggledOn]
        return {"toggled_on": bool(state.get_value()), "finger_on_button_steps": int(state.robot_can_toggle_steps)}

    def mark_goal_initial(self) -> None:
        """Remember which goal predicates already hold, as the challenge metric does (no credit for those)."""
        self.goal_initial = self._goal_values()

    def goal_status(self) -> dict:
        """Challenge-style score: 1 on full success, else the best goal option's newly satisfied fraction."""
        task = self.env.task
        values = self._goal_values()
        initial = self.goal_initial or [[False] * len(v) for v in values]
        options = task.ground_goal_state_options
        best_i, best_new = 0, -1
        for i, (now, was) in enumerate(zip(values, initial)):
            new = sum(int(v and not v0) for v, v0 in zip(now, was))
            if new > best_new:
                best_i, best_new = i, new
        success = any(all(row) for row in values)
        total = len(options[best_i]) if options else 0
        q = 1.0 if success else (best_new / total if total else 0.0)
        return {
            "success": success,
            "q_score": q,
            "options": len(options),
            "satisfied": [self._goal_name(h) for h, v in zip(options[best_i], values[best_i]) if v] if options else [],
            "unsatisfied": [self._goal_name(h) for h, v in zip(options[best_i], values[best_i]) if not v]
            if options
            else [],
            "new": best_new,
            "total": total,
        }

    # ---------------------------------------------------------------- scene setup
    def scene_object(self, name: str):
        obj = self.task_scope().get(name) or self.env.scene.object_registry("name", name)
        if obj is None:
            names = sorted(o.name for o in self.env.scene.objects)
            raise ValueError(f"no object {name!r} in scene {self.config['scene']['scene_model']}; objects: {names}")
        return obj

    def object_names(self) -> list[str]:
        return list(self.objects)

    def track(self, *names: str) -> None:
        """Objects whose masks/poses go to TiPToP and into the success check (spawned or scene objects)."""
        for name in names:
            self.objects[name] = self.scene_object(name)

    def track_context(self, *names: str) -> None:
        """Furniture drawn in the Rerun mirror for orientation (never sent to the planner)."""
        for name in names:
            if name:
                self.context[name] = self.scene_object(name)

    def place_on(self, name: str, support: str, dx: float = 0.0, dy: float = 0.0, lift: float = 0.01) -> None:
        """Drop a tracked object onto the top of a piece of furniture (AABB top + offsets)."""
        obj, sup = self.scene_object(name), self.scene_object(support)
        lo, hi = [v.cpu().numpy() for v in sup.aabb]
        olo, ohi = [v.cpu().numpy() for v in obj.aabb]
        center = (lo + hi) / 2
        pos = th.tensor([center[0] + dx, center[1] + dy, hi[2] + (ohi[2] - olo[2]) / 2 + lift], dtype=th.float32)
        obj.set_position_orientation(position=pos, orientation=th.tensor([0.0, 0.0, 0.0, 1.0]))
        obj.keep_still()
        if obj not in self.objects.values():  # task objects are already tracked under their label
            self.objects[name] = obj
        log.info(f"placed {name} on {support} at {np.round(pos.numpy(), 3).tolist()} (support top z={hi[2]:.3f})")

    def scene_aabbs(self) -> list[tuple]:
        """(object, lo, hi) for every scene object but the robot: one AABB query each (the AABB is recomputed from
        the collision meshes on every access, ~50 ms for a house scene), to reuse across footprint checks."""
        return [(o, *[v.cpu().numpy() for v in o.aabb]) for o in self.env.scene.objects if o is not self.robot]

    def _footprint_free(self, x: float, y: float, ignore, aabbs=None) -> tuple[bool, str]:
        """Floor under the whole footprint, inside a room, and no other object's AABB overlapping the footprint.

        ``aabbs``: a scene_aabbs() snapshot to test against (taken here otherwise; nothing moves during a search).
        """
        r = ROBOT_FOOTPRINT
        corners = [(x + sx * r, y + sy * r) for sx in (-1, 1) for sy in (-1, 1)] + [(x, y)]
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        floors = [(lo, hi) for o, lo, hi in aabbs if o.category == "floors"]
        for cx, cy in corners:
            on_floor = False
            for lo, hi in floors:
                if lo[0] <= cx <= hi[0] and lo[1] <= cy <= hi[1]:
                    on_floor = True
                    break
            if not on_floor:
                return False, f"no floor under ({cx:.2f}, {cy:.2f})"
        try:
            room = self.env.scene.seg_map.get_room_instance_by_point(th.tensor([x, y]))
        except Exception:  # noqa: BLE001 - a point off the map's raster raises inside the lookup; it is a filter only
            room = "unknown"
        if room is None:
            return False, "outside every room"
        for obj, lo, hi in aabbs:
            if obj is self.robot or obj in ignore or obj.category in FLOOR_COVERINGS:
                continue
            if (hi[0] - lo[0]) * (hi[1] - lo[1]) > HOUSE_AABB_AREA:
                continue  # merged walls, roof, ceilings say nothing; the floor test handles walls
            if lo[2] > ROBOT_HEIGHT:
                continue  # entirely above the robot (roof, lamps)
            if hi[2] - lo[2] < FLAT_COVERING_HEIGHT and lo[2] < GROUND_CLEARANCE:
                continue  # flat floor coverings (pavers, rugs, mats) are stood on, not avoided
            if lo[0] < x + r and hi[0] > x - r and lo[1] < y + r and hi[1] > y - r and hi[2] > GROUND_CLEARANCE:
                return False, f"overlaps {obj.name}"
        return True, "free"

    def place_robot_near(self, support: str, side: str = "auto", standoff: float = 0.30, ignore_names=()) -> dict:
        """Put the robot next to a piece of furniture, facing it ("navigation done" stand-in).

        side: -x/+x/-y/+y = which side of the furniture's AABB the robot stands on; auto = first free one.
        ignore_names: objects that do not count as obstacles (e.g. spawned presets still parked in the air).
        """
        sup = self.scene_object(support)
        ignore = (sup, *[self.scene_object(n) for n in ignore_names])
        lo, hi = [v.cpu().numpy() for v in sup.aabb]
        c = (lo + hi) / 2
        d = standoff + ROBOT_FOOTPRINT
        cands = {
            "-x": (lo[0] - d, c[1], 0.0),
            "+x": (hi[0] + d, c[1], math.pi),
            "-y": (c[0], lo[1] - d, math.pi / 2),
            "+y": (c[0], hi[1] + d, -math.pi / 2),
        }
        for s in [side] if side != "auto" else ["-x", "+x", "-y", "+y"]:
            x, y, yaw = cands[s]
            ok, why = self._footprint_free(x, y, ignore=ignore)
            log.info(f"candidate {s} side of {support} at ({x:.2f}, {y:.2f}): {why}")
            if side == "auto" and not ok:
                continue
            return self.place_robot(x, y, yaw, note=f"{s} side of {support}")
        raise RuntimeError(f"no free side around {support}; pass --side or --robot-pose")

    def xy_radius(self, name: str) -> float:
        """Circumscribed xy radius of a scene object, for keeping its edges inside the head camera's view."""
        obj = self.scene_object(name)
        lo, hi = obj.aabb
        ex, ey = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        return 0.5 * math.hypot(ex, ey)

    def best_base_pose(
        self,
        points_xy,
        ignore=(),
        reach: float = 0.9,
        aabbs=None,
        half_widths=None,
        support_z=None,
        avoid=(),
        boxes=None,
        frame_strict: bool = True,
    ) -> tuple[tuple | None, dict]:
        """Best base pose with every point (world xy; the last one is the container) ahead and to the left, within
        the left arm's reach.

        Candidates on rings (0.25 m to ``reach``) around the points' centroid, facing it; scored by the farthest point's distance and
        how far left the points are (lower is better), rejected when a point is behind the robot, well to its right,
        out of reach, outside the head camera's view, hidden behind the container, or the footprint is not free
        (``ignore``: objects that do not count; ``aabbs``: a scene_aabbs() snapshot to reuse across searches, taken
        here otherwise).

        ``boxes``: each object's world AABB as (low xyz, high xyz). Given them, whether a candidate frames an object
        is decided by projecting the object into the head camera that candidate would have (``frame_objects``)
        rather than by the angle measures below -- the same projection the masks use, so the test is exact. A
        candidate that cuts an object the frame could hold whole is rejected; when no candidate frames them all,
        the search runs again with ``frame_strict=False``, where a cut only costs score.

        ``half_widths``, ``support_z``: the angle measures used when no boxes are passed. Each point's xy radius, so
        the view test can keep the object's *edges* in frame and not just its centre (None reproduces the point
        test), and the world height each object stands at (one value, or one per point) so the head camera's reach
        for it can be measured (``camera_floor_distance``; None skips that test). ``avoid``: (x, y) poses already
        tried; candidates within ``AVOID_RADIUS`` of one are rejected, so a retry gets a different viewpoint.
        Returns ((score, x, y, yaw, dists, sides) or None, rejection counts by reason).
        """
        pts = [np.asarray(p, dtype=np.float64)[:2] for p in points_xy]
        if not pts:
            raise ValueError("best_base_pose needs at least one point (the last one is the target)")
        half_widths = [0.0] * len(pts) if half_widths is None else list(half_widths)
        if support_z is None:
            min_dists = [0.0] * len(pts)
        else:
            heights = [support_z] * len(pts) if np.isscalar(support_z) else list(support_z)
            min_dists = [self.camera_floor_distance(float(z)) + CAMERA_MIN_MARGIN for z in heights]
        half_fov = math.atan2(self.robot_cam.image_width / 2, float(_intrinsics(self.robot_cam)[0, 0]))
        view = None
        if boxes is not None:
            k, base_from_cam, base_z = self.head_camera_in_base()
            view = dict(
                corners=[box_corners(lo, hi) for lo, hi in boxes],
                intrinsics=k,
                base_from_cam=base_from_cam,
                base_z=base_z,
                width=int(self.robot_cam.image_width),
                height=int(self.robot_cam.image_height),
                strict=frame_strict,
            )
        t = len(pts) - 1  # the target (container) is last
        mid = np.mean(pts, axis=0)
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        best, rejected, footprint = None, {}, {}  # footprint: (x, y) -> _footprint_free result (yaw-independent)
        for radius in np.arange(RING_START, reach + RING_STEP, RING_STEP):  # rings out to the reach itself
            for angle in np.arange(0.0, 2 * np.pi, RING_ANGLE_STEP):
                x, y = mid + radius * np.array([np.cos(angle), np.sin(angle)])
                if any(np.hypot(x - ax, y - ay) < AVOID_RADIUS for ax, ay in avoid):
                    rejected["tried before"] = rejected.get("tried before", 0) + 1
                    continue
                for yaw_offset in YAW_OFFSETS:
                    yaw = np.arctan2(mid[1] - y, mid[0] - x) + yaw_offset
                    fwd, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
                    rel = [p - np.array([x, y]) for p in pts]
                    ahead = [float(r @ fwd) for r in rel]
                    side = [float(r @ left) for r in rel]
                    dist = [float(np.linalg.norm(r)) for r in rel]
                    # the torso can turn, so a little to the right is acceptable; well to the left is preferred;
                    # all must be inside the head camera's view (about +-45 deg of forward; the camera sees +-50)
                    if min(ahead) < MIN_AHEAD or min(side) < MIN_SIDE or max(dist) > reach:
                        rejected["geometry"] = rejected.get("geometry", 0) + 1
                        continue
                    # cut by the bottom of the head camera's frame. The measure is how far AHEAD the object is,
                    # which is what camera_floor_distance returns; the radial distance flattered a stance with the
                    # object off to one side, and a battery 0.48 m ahead of a base whose camera sits 0.44 m ahead
                    # of it came out of the capture with an empty mask (2026-09-12, dispose_of_batteries).
                    if view is None and any(a < m for a, m in zip(ahead, min_dists)):
                        rejected["too close for the camera"] = rejected.get("too close for the camera", 0) + 1
                        continue
                    # a container nearer than an item and in line with it hides the item (empty mask): keep their
                    # bearings apart by the container's angular half-width plus a margin for the item
                    bearing = [math.atan2(sd, ah) for ah, sd in zip(ahead, side)]
                    if any(
                        dist[t] < dist[i] + HIDE_DEPTH
                        and abs(bearing[i] - bearing[t])
                        < math.atan(TARGET_HALF_WIDTH / dist[t]) + math.atan(HIDE_MARGIN / dist[i])
                        for i in range(t)
                    ):
                        rejected["container hides the item"] = rejected.get("container hides the item", 0) + 1
                        continue
                    if view is None and max(abs(sd) / max(ah, 1e-6) for ah, sd in zip(ahead, side)) > 1.0:
                        rejected["outside camera view"] = rejected.get("outside camera view", 0) + 1
                        continue
                    if view is not None:
                        why, off_frame = frame_objects(**view, x=x, y=y, yaw=yaw)
                        if why:
                            rejected[why] = rejected.get(why, 0) + 1
                            continue
                    # Prefer poses that keep each object's *edges* in frame, not just its centre. A mask cut by the
                    # image border reconstructs into a hull that runs past the real object, and the planner then
                    # places into that phantom part: on 2026-09-04 a basket whose centre sat at 43 deg had its edge
                    # at 61 deg, lost a third of its width off the left of the image, and the cookie was released
                    # 3 cm outside the rim. This is a penalty rather than a rejection because for some item/container
                    # pairs no pose frames both -- the item is then simply out of reach of a single base pose (see
                    # README, "what stands between this and the full task").
                    clipped = sum(
                        max(0.0, abs(br) + math.atan2(hw, max(d, 1e-6)) - half_fov)
                        for br, d, hw in zip(bearing, dist, half_widths)
                    )
                    score = (
                        max(dist)
                        + SIDE_WEIGHT * max(0.0, SIDE_TARGET - min(side))
                        + YAW_WEIGHT * abs(yaw_offset)
                        + (FRAMING_PENALTY_PX * off_frame if view is not None else FRAMING_PENALTY * clipped)
                    )
                    if best is None or score < best[0]:
                        key = (float(x), float(y))
                        if key not in footprint:
                            footprint[key] = self._footprint_free(x, y, ignore, aabbs=aabbs)
                        free, why = footprint[key]
                        if free:
                            best = (score, x, y, yaw, dist, side)
                        else:
                            rejected[why] = rejected.get(why, 0) + 1
        if best is None and boxes is not None and frame_strict:
            log.info(
                f"no stance frames every object whole ({dict(sorted(rejected.items(), key=lambda kv: -kv[1])[:4])}); allowing a clipped one"
            )
            return self.best_base_pose(
                points_xy,
                ignore=ignore,
                reach=reach,
                aabbs=aabbs,
                half_widths=half_widths,
                support_z=support_z,
                avoid=avoid,
                boxes=boxes,
                frame_strict=False,
            )
        return best, rejected

    def hidden_from_here(self, names, aabbs=None) -> dict:
        """For each object named, the scene objects standing between the head camera (as it is now) and it.

        The stance search asks whether an object is in frame; this asks whether anything is in the way, which is a
        different question and the one `battery_3` fails in every run of `dispose_of_batteries` -- it sits on a
        cabinet whose own top edge hides it at the grazing angle a 1.26 m camera has on a 1.02 m surface. The box
        is the prefilter and the object's mesh decides, as in ``arm_hits_scene``. An object's own support does not
        false-positive: the ray to a thing standing on a surface stays above that surface until it arrives.
        """
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        eye = self.base_to_world(self.head_camera_in_base()[1][:3, 3])
        held = {self.objects[label] for label in self.hands() if label in self.objects}
        out = {}
        for name in names:
            obj = self.scene_object(name)
            target = obj.aabb_center.cpu().numpy().astype(np.float64)
            blockers = []
            for other, lo, hi in aabbs:
                if other is obj or other in held or other.category in FLOOR_COVERINGS:
                    continue
                if not segment_hits_box(eye, target, lo, hi):
                    continue
                try:
                    mesh = self.scene_mesh(other)
                except Exception:
                    blockers.append(other.name)  # no mesh; the box is all there is to go on
                    continue
                ray = sample_polyline([eye, target], ARM_SAMPLE_STEP)[:-1]  # not the target itself
                if bool(points_within_tol(mesh, ray, 0.0).any()):
                    blockers.append(other.name)
            if blockers:
                out[name] = blockers
        return out

    def place_robot_for(self, *names: str, ignore_names=(), reach: float = 0.9, avoid=()) -> dict:
        """Stand where every item and the target (the last name) are in the left arm's reach ("navigation done").

        A single name is a target with no items (a one-object task such as turning_on_radio). ignore_names:
        objects that do not count as obstacles, resolved like place_robot_near's (unknown names raise). Objects
        the robot holds never count (they travel with it). ``avoid``: (x, y) poses not to stand at again.
        """
        if not names:
            raise ValueError("place_robot_for needs [ITEM,...,]TARGET")
        objects = [self.scene_object(n) for n in names]
        points = [o.aabb_center.cpu().numpy()[:2] for o in objects]
        support_z = [float(o.aabb[0][2]) for o in objects]  # each object's bottom: the top of what it stands on
        ignore = [self.scene_object(n) for n in ignore_names] + [self.objects[l] for l in (self.grasped_labels() or {})]
        log.info(
            f"head camera at z {float(self.robot_cam.get_position_orientation()[0][2]):.2f} m sees a surface at "
            f"z {min(support_z):.2f} from {self.camera_floor_distance(min(support_z)):.2f} m ahead"
        )
        best, rejected = self.best_base_pose(
            points,
            ignore=ignore,
            reach=reach,
            half_widths=[self.xy_radius(n) for n in names],
            support_z=support_z,
            avoid=avoid,
            boxes=[(o.aabb[0].cpu().numpy(), o.aabb[1].cpu().numpy()) for o in objects],
        )
        if best is None:
            raise RuntimeError(
                f"no base pose reaches {list(names)} within {reach} m (objects "
                f"{[np.round(p, 2).tolist() for p in points]}; rejections "
                f"{dict(sorted(rejected.items(), key=lambda kv: -kv[1])[:6])})"
            )
        score, x, y, yaw, dist, side = best
        log.info(
            f"standing for {' + '.join(names)}: ({x:.2f}, {y:.2f}) yaw {np.degrees(yaw):.0f} deg, "
            f"distances {np.round(dist, 2).tolist()} m, left offsets {np.round(side, 2).tolist()} m"
        )
        pose = self.place_robot(float(x), float(y), float(yaw), note=f"stand for {' + '.join(names)}")
        hidden = self.hidden_from_here(names)
        if hidden:
            log.warning(
                "from this stance the head camera's line to "
                + ", ".join(f"{name} passes through {blockers}" for name, blockers in hidden.items())
                + " -- the capture will see nothing of it unless another view does"
            )
        centre = th.stack([o.aabb_center for o in objects]).mean(dim=0)  # what the wrist cameras look at
        self.look_target = self.to_base(centre, th.tensor([0.0, 0.0, 0.0, 1.0]))[0].cpu().numpy()
        self.look_names = tuple(names)
        return pose

    def place_robot(self, x: float, y: float, yaw: float, note: str = "") -> dict:
        """Teleport the base to a floor pose (the navigation stand-in). OmniGibson moves an object held by the
        grasp assist along with the robot, so a carried object stays in the gripper."""
        quat = T.euler2quat(th.tensor([0.0, 0.0, float(yaw)]))
        self.robot.set_position_orientation(position=th.tensor([x, y, 0.0]), orientation=quat)
        self.robot.keep_still()
        self.teleports += 1
        self.look_target, self.look_names = None, ()  # a base-frame target from the previous pose means nothing here
        # third-person view for the overview camera (video, Rerun mirror) and the Isaac Sim viewport when there is
        # one: over the robot's left shoulder at the workspace ("shoulder"), or from ahead and to the right looking
        # back at the chest, where both hands and what they hold are in view ("front", the two-hands demo)
        dx, dy, z, tx, tz = OVERVIEW_OFFSETS[self.overview_view]
        eye = (x + dx * math.cos(yaw) - dy * math.sin(yaw), y + dx * math.sin(yaw) + dy * math.cos(yaw), z)
        target = (x + tx * math.cos(yaw), y + tx * math.sin(yaw), tz)
        self.aim_overview(eye, target)
        if not gm.HEADLESS:
            og.sim.viewer_camera.set_position_orientation(
                position=th.tensor(eye), orientation=th.tensor(look_at_quat_xyzw(eye, target))
            )
        log.info(f"robot placed at ({x:.2f}, {y:.2f}) yaw {math.degrees(yaw):.0f} deg {note}")
        return {"x": float(x), "y": float(y), "yaw": float(yaw)}

    def apply_posture(self, locked: dict, q_home, settle_steps: int = 30, tol: float = 0.03, joint_names=None) -> None:
        """Hold the joints the planner locks (right arm, fingers, torso if not planned) and go to q_home.

        joint_names: the planner's joint order (embodiment metadata), e.g. torso_joint1..4 + left_arm_joint1..7; the
        simulator's planned-joint vector (q_arm / actions / sim_state) follows that order from here on.
        """
        unknown = [j for j in list(locked) + list(joint_names or []) if j not in self.joint_index]
        if unknown:
            raise ValueError(f"joints unknown to the simulator: {unknown}")
        if joint_names:
            self.planned_joints = list(joint_names)
            self.arm_idx = th.tensor([self.joint_index[j] for j in self.planned_joints])
        assert len(q_home) == len(self.planned_joints), (len(q_home), self.planned_joints)
        self.posture = {j: float(v) for j, v in locked.items()}
        self.q_home = [float(v) for v in q_home]
        q = self.robot.get_joint_positions().clone()
        for j, v in self.posture.items():
            q[self.joint_index[j]] = v
        for j, v in zip(self.planned_joints, self.q_home):
            q[self.joint_index[j]] = v
        self.robot.set_joint_positions(q, drive=False)
        self.robot.keep_still()
        self.hold(settle_steps, self.OPEN, q_arm=self.q_home)
        now = self.robot.get_joint_positions()
        errs = {j: float(abs(now[self.joint_index[j]] - v)) for j, v in self.posture.items() if "finger" not in j}
        worst = max(errs, key=errs.get)
        base_z = float(self.robot.get_position_orientation()[0][2])
        cam_z = float(self.robot_cam.get_position_orientation()[0][2])
        log.info(
            f"posture applied: worst locked-joint error {errs[worst]:.4f} rad on {worst}; base z {base_z:.3f}, camera z {cam_z:.3f}"
        )
        if cam_z < 0.8:
            raise RuntimeError(f"robot is not upright (camera at z={cam_z:.2f} m); check base mass / posture")
        if errs[worst] > tol:
            raise RuntimeError(
                f"simulator does not hold the planner's locked posture: {worst} off by {errs[worst]:.3f} rad"
            )

    def reset_embodiment(self, embodiment: dict) -> None:
        """Plan ``embodiment``'s arm from the start of a fresh episode, no questions asked (``apply_posture`` follows
        and teleports the joints): the other arm's gripper opens, the look posture is back, the mirror follows."""
        self.arm = embodiment["arm"]
        self.other_arm = "right" if self.arm == "left" else "left"
        self.other_gripper = self.OPEN
        self.planned_joints = list(embodiment["joint_names"])
        self.arm_idx = th.tensor([self.joint_index[j] for j in self.planned_joints])
        self.gripper_idx = self.robot.gripper_control_idx[self.arm]
        self.look_arm = LOOK_ARM
        self.mirror_arm_idx = self.mirror_gripper_idx = None

    def adopt_embodiment(self, embodiment: dict, tol: float = 0.05) -> None:
        """Plan another arm from here on without moving anything: e.g. ``r1pro_right`` after the left hand picked
        something up. The joints the new embodiment locks (torso, the other arm) must already be where it expects
        them within ``tol`` (a loaded wrist settles up to ~0.035 rad short of its target under a held object; the
        new planner only uses these values for the other arm's own collision spheres); fingers are the gripper
        state and are not checked. The arm that planned so far keeps its last gripper command (a held object stays
        held) and its joints are held at their current values; the adopted arm resumes the command it was left
        with (until 2026-09-09 it inherited the other arm's, so a left hand holding the radio was commanded open
        by the next left plan and dropped it).
        A capture still poses the free arm for its wrist camera (the held arm never moves), and the Rerun mirror
        keeps reporting the first embodiment's joints."""
        arm = embodiment["arm"]
        if arm == self.arm:
            return
        locked = {j: float(v) for j, v in embodiment["locked_joints"].items()}
        unknown = [j for j in list(locked) + list(embodiment["joint_names"]) if j not in self.joint_index]
        if unknown:
            raise ValueError(f"joints unknown to the simulator: {unknown}")
        q = self.robot.get_joint_positions()
        errs = {j: abs(float(q[self.joint_index[j]]) - v) for j, v in locked.items() if "finger" not in j}
        worst = max(errs, key=errs.get)
        if errs[worst] > tol:
            raise RuntimeError(
                f"{embodiment['robot_type']} locks {worst} at {locked[worst]:.3f} rad but the simulator has it at "
                f"{float(q[self.joint_index[worst]]):.3f} (off by {errs[worst]:.3f} > {tol})"
            )
        if self.mirror_arm_idx is None:
            self.mirror_arm_idx, self.mirror_gripper_idx = self.arm_idx, self.gripper_idx
        self.other_arm, self.other_gripper, self.last_gripper = self.arm, self.last_gripper, self.other_gripper
        self.arm = arm
        self.planned_joints = list(embodiment["joint_names"])
        self.arm_idx = th.tensor([self.joint_index[j] for j in self.planned_joints])
        self.gripper_idx = self.robot.gripper_control_idx[arm]
        self.posture = {j: float(q[self.joint_index[j]]) for j in locked if "finger" not in j}  # hold, do not move
        self.q_home = [float(v) for v in embodiment["q_home"]]
        log.info(
            f"planning the {arm} arm from here on ({embodiment['robot_type']}: {len(self.planned_joints)} joints); "
            f"the {self.other_arm} arm holds its posture with gripper command {self.other_gripper:+.0f}; "
            f"worst locked-joint error {errs[worst]:.4f} rad on {worst}"
        )

    def mirror_q(self) -> np.ndarray:
        idx = self.arm_idx if self.mirror_arm_idx is None else self.mirror_arm_idx
        return self.robot.get_joint_positions()[idx].cpu().numpy()

    def mirror_fingers(self) -> np.ndarray:
        idx = self.gripper_idx if self.mirror_gripper_idx is None else self.mirror_gripper_idx
        return self.robot.get_joint_positions()[idx].cpu().numpy()

    def camera_floor_distance(self, surface_z: float) -> float:
        """How far ahead of the base (m) the head camera's bottom image edge meets a horizontal surface at ``surface_z``.

        Objects nearer than this are cut off at the bottom of the capture. Depends on the torso posture (camera
        height and pitch) and nothing else, so it holds for any base pose once the posture is applied.
        """
        pos, quat = self.robot_cam.get_position_orientation()
        k = _intrinsics(self.robot_cam)
        bottom = np.array(
            [0.0, -(self.robot_cam.image_height - k[1, 2]) / k[1, 1], -1.0]
        )  # USD camera: -z forward, +y up
        d = T.quat2mat(quat).cpu().numpy() @ bottom
        if d[2] >= -1e-6:
            return float("inf")  # the bottom edge never meets the surface
        hit = pos.cpu().numpy() + (surface_z - float(pos[2])) / d[2] * d
        ahead, _ = self.to_base(th.tensor(hit, dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0]))
        return float(ahead[0])

    def head_camera_in_base(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Head camera intrinsics, its OpenCV pose (4, 4) in the robot base frame, and the base frame's world z.

        Built exactly as a captured view's ``world_from_cam`` is (scene.view_frame), which is also a base-frame
        pose. The camera is rigid with the base for a given torso posture, so this holds for any base pose: it is
        what lets ``best_base_pose`` project an object into a stance it has not taken yet.
        """
        k = _intrinsics(self.robot_cam).astype(np.float64)
        pos, quat = self.robot_cam.get_position_orientation()
        quat_cv = T.quat_multiply(quat, th.tensor([1.0, 0.0, 0.0, 0.0]))  # 180 deg about camera x -> OpenCV
        pos_b, quat_b = self.to_base(pos, quat_cv)
        base_from_cam = T.pose2mat((pos_b, quat_b)).cpu().numpy().astype(np.float64)
        return k, base_from_cam, float(self.base_pose()[0][2])

    # ---------------------------------------------------------------- observation
    def arm_ik(self, arm: str, frame: str | None = None) -> ArmIK:
        """Inverse kinematics for ``arm``'s joints with every other joint held where it is now, solving for
        ``frame`` (its wrist camera's link by default)."""
        joints = list(self.robot.arm_joint_names[arm])
        q = self.robot.get_joint_positions()
        fixed = {
            name: float(q[i]) for name, i in self.joint_index.items() if name in self.urdf_joints and name not in joints
        }
        return ArmIK(self.robot.urdf_path, joints, fixed, frame=frame or CAMERA_LINKS[f"{arm}_wrist"])

    def in_head_frame(self, ik: ArmIK, q, frame: str) -> bool:
        """Whether ``frame`` at arm joints ``q`` would land inside the head camera's image as it stands now.

        What a presentation is *for*: the point of lifting a carried object is that the head camera sees it, and
        the present points are fixed offsets chosen for one torso posture. The posture varies in a run -- the head
        camera stood between 1.26 m and 1.37 m over runs/bench_toys_7 -- so whether a given point is still in
        frame is worth asking rather than assuming. All nine rounds that run lost to empty masks were place
        rounds, every one of them a carried object the capture could not see.
        """
        try:
            pos, _ = ik.fk(q, frame)
        except Exception:
            return True  # no forward kinematics for it: do not let this decide anything
        k, base_from_cam, _ = self.head_camera_in_base()
        px, z = points_to_pixels([np.asarray(pos, dtype=np.float64)], k, base_from_cam)
        (u, v), ahead = px[0], float(z[0])
        return bool(ahead > 0 and 0 <= u < self.robot_cam.image_width and 0 <= v < self.robot_cam.image_height)

    def present_held(self, arm: str, aabbs=None, ignore=()) -> np.ndarray | None:
        """Joints of ``arm`` that hold what it is carrying in front of the head camera (``PRESENT_POINT`` on its
        own side, the roomiest of the ``PRESENT_OFFSETS`` it reaches, clear of the base), the gripper
        keeping the orientation it grasped with so the object is not turned in the hand. ``ignore``: objects the
        round is about, which do not count as obstacles here -- the robot stands at the container it is going to
        place into, and the place motion enters it anyway with the planner's own collision geometry, so refusing
        every pose near it leaves nothing (putting_away_toys at a toy box, 2026-09-12). None when no
        configuration does.

        Ranked by ``path_contacts`` for the same reason ``wrist_look`` is: with head views alone there is no wrist
        camera to see a carried object with, so every carry round presents it -- and in runs/bench_toys_6, 28 of
        the 29 presentations were stopped against something. Taking the first offset that reaches is what made
        that the task's dominant collision."""
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        skip = {self.scene_object(n) for n in ignore}
        aabbs = [row for row in aabbs if row[0] not in skip]
        ik = self.arm_ik(arm, frame=f"{arm}_gripper_link")
        joints = list(self.robot.arm_joint_names[arm])
        q = self.robot.get_joint_positions()
        seed = [float(q[self.joint_index[j]]) for j in joints]
        pose = self.eef_pose_base(arm)
        quat_xyzw = T.mat2quat(th.tensor(pose[:3, :3], dtype=th.float32)).cpu().numpy()
        side = 1.0 if arm == "left" else -1.0
        base = np.array([PRESENT_POINT[0], side * PRESENT_POINT[1], PRESENT_POINT[2]], dtype=np.float64)
        here = np.asarray(seed, dtype=np.float64)
        candidates, unreachable = [], 0
        for offset in PRESENT_OFFSETS:
            target = base + np.array([offset[0], side * offset[1], offset[2]], dtype=np.float64)
            # the orientation is loose: what matters is that the object is in the picture, not how it is held
            solution = ik.solve(target, quat_xyzw, seed=seed, tolerance_pos=0.04, tolerance_rad=1.2)
            if solution is None:
                unreachable += 1
                continue
            inside = self.links_in_base_box(arm, ik, solution)
            if inside:
                log.info(f"{arm} arm: presenting at {np.round(target, 2).tolist()} puts {inside} at the base; skipped")
                continue
            touched, objects = self.path_contacts(arm, ik, here, solution, aabbs)
            candidates.append(
                (
                    not self.in_head_frame(ik, solution, f"{arm}_gripper_link"),
                    touched,
                    len(candidates),
                    target,
                    solution,
                    objects,
                )
            )
        if not candidates:
            # Which of the two it is matters: an arm that cannot reach any present point from where it stands is a
            # different problem from one whose every reach is refused. The seed is where the arm actually is, and
            # the capture's own changes move that -- so say it.
            log.warning(
                f"{arm} arm: none of the {len(PRESENT_OFFSETS)} present points work from "
                f"{np.round(here, 2).tolist()}: {unreachable} out of reach, "
                f"{len(PRESENT_OFFSETS) - unreachable} refused at the base"
            )
            return None
        out_of_frame, touched, _, target, solution, objects = min(candidates)
        if out_of_frame:
            log.warning(
                f"{arm} arm: no present point puts what it holds inside the head camera's frame; using "
                f"{np.round(target, 2).tolist()} anyway"
            )
        if touched:
            log.info(
                f"{arm} arm: presenting what it holds at {np.round(target, 2).tolist()}, the roomiest of "
                f"{len(candidates)}, still passing within {ARM_RADIUS} m of {objects} at {touched} point(s)"
            )
        else:
            log.info(
                f"{arm} arm: presenting what it holds at {np.round(target, 2).tolist()}, clear of the scene the "
                f"whole way ({len(candidates)} reachable)"
            )
        return solution

    def wrist_look(self, arm: str, target, aabbs=None) -> np.ndarray | None:
        """Joints of ``arm`` that point its wrist camera at ``target`` (base frame) from beside its own shoulder
        (``kinematics.look_pose``, the first of ``LOOK_OFFSETS`` the arm reaches) and keep the hand clear of the
        base and of every scene object, every other joint where it is; None when no configuration does."""
        joints = list(self.robot.arm_joint_names[arm])
        q = self.robot.get_joint_positions()
        ik = self.arm_ik(arm)
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        shoulder, _ = self.to_base(*self.robot.links[self.robot.arm_link_names[arm][0]].get_position_orientation())
        seed = [float(q[self.joint_index[j]]) for j in joints]
        here = np.asarray(seed, dtype=np.float64)
        candidates = []
        for offset in LOOK_OFFSETS:
            eye, cam_quat = look_pose(target, shoulder.cpu().numpy(), side=1 if arm == "left" else -1, offset=offset)
            solution = ik.solve(*link_pose_for_camera(eye, cam_quat, self.camera_in_link[arm]), seed=seed)
            if solution is None:
                continue
            inside = self.links_in_base_box(arm, ik, solution)
            if inside:
                log.info(
                    f"{arm} arm: look offset {offset} puts {inside} within {BASE_CLEARANCE} m of the base; skipped"
                )
                continue
            in_the_way = self.links_before_camera(arm, ik, solution, target)
            if in_the_way:
                log.info(
                    f"{arm} arm: look offset {offset} puts {in_the_way} between the head camera and the target; "
                    "skipped (the head view would lose it to the self-mask)"
                )
                continue
            # How much of the arm passes within ARM_RADIUS of the furniture on the way there and back. Not a
            # rejection: the arm is near the furniture it works at whatever it does, so the offsets are ranked
            # and the roomiest is taken. The runs of 2026-09-13 show what the unranked choice costs -- the first
            # offset that solved was taken, and the arm met a desk on the way to it.
            touched, objects = self.path_contacts(arm, ik, here, solution, aabbs)
            candidates.append((touched, len(candidates), offset, solution, objects))
        if not candidates:
            return None
        touched, _, offset, solution, objects = min(candidates)
        if touched:
            log.info(
                f"{arm} arm: look offset {offset} is the roomiest of {len(candidates)}, and still takes the arm "
                f"within {ARM_RADIUS} m of {objects} at {touched} point(s) on the way"
            )
        elif len(candidates) > 1:
            log.info(f"{arm} arm: look offset {offset} chosen, clear of the scene the whole way")
        return solution

    def base_box(self) -> np.ndarray:
        """(min, max) corners of base_link's collision bounding box in the base frame, inflated by nothing."""
        if self._base_box is None:
            lo, hi = self.robot.links["base_link"].aabb
            unit = th.tensor([0.0, 0.0, 0.0, 1.0])
            corners = [
                self.to_base(th.tensor([float(x), float(y), float(z)]), unit)[0].cpu().numpy()
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            ]
            pts = np.asarray(corners, dtype=np.float64)
            self._base_box = np.stack([pts.min(axis=0), pts.max(axis=0)])
            log.info(f"base box (base frame): {np.round(self._base_box, 2).tolist()}")
        return self._base_box

    def links_in_scene(self, arm: str, ik: ArmIK, q, aabbs=None) -> list[str]:
        """Where the arm's hand links (``HAND_LINKS``) at joints ``q`` would sit inside a scene object's box
        (inflated by ``SCENE_CLEARANCE``): "left_gripper_link in desk_1". Objects the hands hold travel with the
        arm and do not count. A capture swing into furniture is what this exists to refuse: in a cubicle the look
        pose put the wrist against the desk, the joints stopped following the ramp, and the arm swept a battery
        onto the floor on the way (2026-09-12)."""
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        held = {self.objects[label] for label in self.hands() if label in self.objects}
        hits = []
        for suffix in HAND_LINKS:
            pos, _ = ik.fk(q, f"{arm}_{suffix}")
            point = np.asarray(self.base_to_world(np.asarray(pos, dtype=np.float64)), dtype=np.float64)
            for obj, lo, hi in aabbs:
                if obj in held:
                    continue
                if np.all(point > np.asarray(lo) - SCENE_CLEARANCE) and np.all(
                    point < np.asarray(hi) + SCENE_CLEARANCE
                ):
                    hits.append(f"{arm}_{suffix} in {obj.name}")
                    break
        return hits

    def links_before_camera(self, arm: str, ik: ArmIK, q, target) -> list[str]:
        """Links of ``arm`` at joints ``q`` that stand on the head camera's line of sight to ``target`` (base frame).

        Measured cause of a lost round (2026-09-12, dispose_of_batteries): battery_1 projected *inside* the head
        image both times the capture missed it, at 0.61 m and 0.64 m, and the depth at its pixel read 0.41 m and
        then 0.01 m. A near-zero depth is the robot's own pixels, which the self-mask zeroes -- the arm was between
        the head camera and the battery. Link origins against ``LOOK_BLOCK_RADIUS``; links Lula's description does
        not carry are skipped rather than guessed at.
        """
        eye = self.head_camera_in_base()[1][:3, 3]
        hits = []
        for link in self.robot.arm_link_names[arm]:
            try:
                pos, _ = ik.fk(q, link)
            except Exception:
                continue
            if blocks_ray(eye, target, pos, LOOK_BLOCK_RADIUS):
                hits.append(link)
        return hits

    def arm_points(self, arm: str, ik: ArmIK, q) -> list[np.ndarray]:
        """World positions of ``arm``'s link origins at joints ``q``, shoulder to fingertips, in order."""
        points = []
        for name in list(self.robot.arm_link_names[arm]) + [f"{arm}_{suffix}" for suffix in HAND_LINKS]:
            try:
                pos, _ = ik.fk(q, name)
            except Exception:
                continue  # a link Lula's description does not carry
            points.append(np.asarray(self.base_to_world(np.asarray(pos, dtype=np.float64)), dtype=np.float64))
        return points

    def arm_hits_scene(self, arm: str, ik: ArmIK, q, aabbs=None, clearance: float = ARM_RADIUS) -> list[str]:
        """Scene objects the whole arm reaches into at joints ``q``: "desk_1".

        The arm is the polyline through its link origins (``arm_points``) and each scene box is grown by
        ``clearance`` for the limbs' thickness, so a segment crossing the grown box means the arm is in it.
        ``links_in_scene`` tests only the hand's links, and only as points -- but the joints the runs report
        pushing against furniture are the shoulder and elbow (left_arm_joint4, left_arm_joint5, right_arm_joint3),
        which no test covered. Objects a hand holds travel with the arm and do not count.

        A box is a coarse model of a piece of furniture and the polyline a coarse model of an arm, so this says
        "the arm is inside that object's box", not "the arm is in contact". Before it decides anything, check what
        it reports at the ready posture in a real scene: a detector that fires where the arm plainly is not
        touching anything would only cost the capture its look poses.
        """
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        held = {self.objects[label] for label in self.hands() if label in self.objects}
        points = self.arm_points(arm, ik, q)
        near = [
            (obj, lo, hi) for obj, lo, hi in aabbs if obj not in held and polyline_hits_box(points, lo, hi, clearance)
        ]
        if not (ARM_MESH_CHECK and near):
            return [obj.name for obj, _, _ in near]
        samples = sample_polyline(points, ARM_SAMPLE_STEP)
        hits = []
        for obj, _, _ in near:  # the box only says "look closer"; the object's own surface decides
            try:
                mesh = self.scene_mesh(obj)
            except Exception:
                hits.append(obj.name)  # no mesh to check against: keep the box's word
                continue
            if bool(points_within_tol(mesh, samples, clearance).any()):
                hits.append(obj.name)
        return hits

    def scene_mesh(self, obj):
        """World-frame trimesh of a scene object, kept until the object moves (its box is the stamp).

        Furniture never moves, so this is built once per object per run; a task object that is carried about gets
        a new mesh when its box changes. Building one concatenates every link's visual mesh, which is far too slow
        to do per candidate -- hence the box prefilter in ``arm_hits_scene``.
        """
        lo, hi = (v.cpu().numpy() for v in obj.aabb)
        stamp = (tuple(np.round(lo, 4)), tuple(np.round(hi, 4)))
        cached = self._scene_meshes.get(obj.name)
        if cached is None or cached[0] != stamp:
            self._scene_meshes[obj.name] = (stamp, self.trimesh_world(obj))
        return self._scene_meshes[obj.name][1]

    def path_contacts(self, arm: str, ik: ArmIK, q_from, q_to, aabbs=None, samples: int = PATH_SAMPLES) -> tuple:
        """How much of the arm comes within ``ARM_RADIUS`` of the scene along the straight joint-space path from
        ``q_from`` to ``q_to``: (number of arm sample points that do, the objects they belong to).

        A count rather than a verdict, because there is no honest threshold: at the ready posture in front of a
        desk the arm is already within 6 cm of it (measured 2026-09-13, scratchpad/arm_clearance.py), and a robot
        working at a desk is *supposed* to be near it. Ranking candidate look poses by this number picks the one
        with the most room without ever deciding that none of them is usable.

        One mesh query per nearby object for the whole path: the meshes are cached (``scene_mesh``) but preparing
        one for a query is not, so the arm's points for every sampled configuration go in together.
        """
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        held = {self.objects[label] for label in self.hands() if label in self.objects}
        points, boxes = [], []
        for t in np.linspace(0.0, 1.0, max(2, samples)):
            arm_points = self.arm_points(arm, ik, q_from + t * (q_to - q_from))
            if not arm_points:
                continue
            points.append(sample_polyline(arm_points, ARM_SAMPLE_STEP))
            boxes.append(arm_points)
        if not points:
            return 0, []
        samples_all = np.concatenate(points)
        touched, objects = 0, []
        for obj, lo, hi in aabbs:
            if obj in held or not any(polyline_hits_box(b, lo, hi, ARM_RADIUS) for b in boxes):
                continue
            try:
                near = points_within_tol(self.scene_mesh(obj), samples_all, ARM_RADIUS)
            except Exception:
                continue  # no mesh to measure against; the box alone decides nothing (see arm_hits_scene)
            if near.any():
                touched += int(near.sum())
                objects.append(obj.name)
        return touched, objects

    def path_hits_scene(self, arm: str, ik: ArmIK, q_from, q_to, aabbs=None, samples: int = PATH_SAMPLES) -> list[str]:
        """Scene objects the arm reaches into anywhere along the straight joint-space path from ``q_from`` to
        ``q_to``, sampled at ``samples`` configurations (the ends included).

        The bridge's own capture motion is a straight ramp in joint space (``ramp_to``), so this is the path the
        arm really takes. Checking only the destination is what let a swing sweep a battery off a desk on the way
        (2026-09-12) and what leaves a run with tens of "the arm is pushing against something" (2026-09-13).
        """
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        hits = []
        for t in np.linspace(0.0, 1.0, max(2, samples)):
            for name in self.arm_hits_scene(arm, ik, q_from + t * (q_to - q_from), aabbs):
                if name not in hits:
                    hits.append(name)
        return hits

    def links_in_base_box(self, arm: str, ik: ArmIK, q) -> list[str]:
        """The arm's hand links (``HAND_LINKS``) whose frame origin at arm joints ``q`` lies inside the base's box
        inflated by ``BASE_CLEARANCE``."""
        lo, hi = self.base_box()
        inside = []
        for suffix in HAND_LINKS:
            pos, _ = ik.fk(q, f"{arm}_{suffix}")
            if np.all(pos > lo - BASE_CLEARANCE) and np.all(pos < hi + BASE_CLEARANCE):
                inside.append(f"{arm}_{suffix}")
        return inside

    def ramp_arms(
        self, q_arm, posture: dict, arms, gripper: float, settle_steps: int, elbow_first: bool, note: str = ""
    ) -> bool:
        """``ramp_to`` the targets through a via configuration: for each arm in ``arms`` the elbow alone has moved
        (``elbow_first``, a swing out: the hand rises before it travels) or every joint but the elbow has (a swing
        back: the hand travels folded and straightens last). Both legs ramp at the capture speed; the via is not
        held. False when a leg stopped against something (the arms are then wherever that left them)."""
        q = self.robot.get_joint_positions()
        names = list(self.planned_joints) + list(posture)
        now = {j: float(q[self.joint_index[j]]) for j in names}
        goal = dict(zip(names, [float(v) for v in q_arm] + [float(posture[j]) for j in posture]))
        via = via_configuration(names, now, goal, {a: self.robot.arm_joint_names[a] for a in arms}, ELBOW, elbow_first)
        if any(abs(via[j] - now[j]) > 1e-3 for j in names):
            if self.ramp_to([via[j] for j in self.planned_joints], {j: via[j] for j in posture}, gripper, 0, note):
                self.hold(settle_steps, gripper)  # blocked on the first leg; do not drive the second into it
                return False
        return self.ramp_to(q_arm, posture, gripper, settle_steps, note) is None

    def ramp_to(self, q_arm, posture: dict, gripper: float, settle_steps: int, note: str = "") -> tuple | None:
        """Move the planned joints to ``q_arm`` and the locked joints to ``posture`` together, every joint at no more
        than ``CAPTURE_MAX_JOINT_VEL``: one interpolated target per control step from where the joints are now, then
        ``settle_steps`` holding the targets. ``self.posture`` follows the ramp and ends at ``posture``. ``note``
        names the motion in the log when it is blocked, so a run says which motion met the obstacle rather than
        only which joint did (the user watched a video of arms knocking objects about, 2026-09-13).

        A joint that falls more than ``RAMP_BLOCK_TOL`` behind its target is pushing against something, and the ramp
        stops there and holds where the joints actually are rather than leaning on it for the rest of the path
        (a capture swing in a cubicle sweeps what is on the desk onto the floor, 2026-09-12). Returns None when the
        joints followed, else (joint, step, lag)."""
        now = self.robot.get_joint_positions()
        names = list(self.planned_joints) + list(posture)
        start = [float(now[self.joint_index[j]]) for j in names]
        goal = [float(v) for v in q_arm] + [float(posture[j]) for j in posture]
        path = joint_ramp(start, goal, CAPTURE_MAX_JOINT_VEL * self.dt)
        k = len(self.planned_joints)
        idx = [self.joint_index[j] for j in names]
        # Only the joints this ramp actually MOVES can tell it it is blocked. The fingers are driven by the gripper
        # command, not the ramp; and a joint whose target does not change is merely being held, so its lag measures
        # whether it can hold itself -- an arm sagging against furniture, say -- not whether this motion is obstructed.
        # Measured 2026-09-13 (putting_away_toys, runs/bench_toys_5): once a head view stopped commanding the arm to
        # the nominal posture and held it where it stood, head-view ramps began reporting themselves blocked on
        # left_arm_joint1/4/5/7 -- joints a head view does not move. Each such abort left the torso partway, and the
        # round then died on "the torso did not return after the head views (off by 0.130 / 0.221 rad)".
        moving = np.abs(np.asarray(goal) - np.asarray(start)) > RAMP_MOVING_EPS
        ramped = moving & np.array(["finger" not in j for j in names])
        last, fastest, culprit, blocked, behind = np.asarray(start), 0.0, "", None, 0
        sagged, sagged_joint = 0.0, ""  # the worst a HELD joint drifted from where it was asked to stay
        for i, q in enumerate(path):
            self.posture = {j: float(v) for j, v in zip(posture, q[k:])}
            self.step(q[:k], gripper)
            measured = self.robot.get_joint_positions()[idx].cpu().numpy().astype(np.float64)
            speed = np.abs(measured - last) / self.dt
            if speed.max() > fastest:
                j = int(speed.argmax())
                fastest = float(speed[j])
                culprit = f" ({names[j]} at step {i + 1}: {last[j]:+.3f} -> {measured[j]:+.3f} rad, target {q[j]:+.3f})"
            lag = np.where(ramped, np.abs(measured - q), 0.0)
            last = measured
            # One step behind is a graze the arm slips past (the right arm over the base does it on most basket
            # captures); RAMP_BLOCK_STEPS in a row is something the arm is not going to get past.
            drift = np.where(~moving & np.array(["finger" not in j for j in names]), np.abs(measured - q), 0.0)
            if drift.max() > sagged:
                sagged = float(drift.max())
                sagged_joint = names[int(drift.argmax())]
            behind = behind + 1 if lag.max() > RAMP_BLOCK_TOL else 0
            if behind >= RAMP_BLOCK_STEPS:
                j = int(lag.argmax())
                blocked = (names[j], i + 1, float(lag[j]))
                break  # stop pushing: the rest of the path would only lean harder on whatever is in the way
        if blocked is not None:
            log.warning(
                f"{blocked[0]} stopped following the ramp at step {blocked[1]} of {len(path)} ({blocked[2]:.2f} rad "
                f"behind its target for {RAMP_BLOCK_STEPS} steps): the arm is pushing against something, so the "
                f"ramp stopped there [motion: {note or 'unnamed'}, env step {self.n_steps}]"
            )
            held = self.robot.get_joint_positions()
            self.posture = {j: float(held[self.joint_index[j]]) for j in posture}
            q_hold = [float(held[self.joint_index[j]]) for j in self.planned_joints]
        else:
            self.posture = {j: float(v) for j, v in posture.items()}
            q_hold = [float(v) for v in q_arm]
        self.hold(settle_steps, gripper, q_arm=q_hold)
        if sagged > RAMP_BLOCK_TOL:
            log.warning(
                f"{sagged_joint} drifted {sagged:.2f} rad from where it was asked to stay during this ramp "
                f"({note or 'unnamed'}): it is not holding itself, though this motion does not move it"
            )
        log.info(
            f"joints ramped over {len(path)} steps at up to {CAPTURE_MAX_JOINT_VEL} rad/s commanded, "
            f"{fastest:.2f} rad/s measured{culprit}, then {settle_steps} settle steps"
        )
        return blocked

    def capture(self, task: str) -> tuple[dict, dict]:
        """Every view in one posture: each free arm whose wrist camera is a view points it at the look target
        (``wrist_look``: what the base pose was chosen for, at the hand that holds it once it has been picked up;
        a held object otherwise, when nothing was stood for), which also takes the arm out of the head camera's
        frame. An arm that holds something stays where it is:
        the held object is what the next plan is about and must be seen, and the gripper keeps its command. The
        planned arm swings out of view (``look_arm``) when no look configuration exists; with ``look_arm`` None
        nothing moves. The head views of ``HEAD_VIEWS`` come last, the torso turned to their yaw with the arms
        as they are (``_capture_views``). The plan starts from the ready posture the arms return to."""
        ready = list(self.q_home) if self.q_home is not None else [float(v) for v in self.q_arm()]
        if self.look_arm is None:
            return self._capture_views(task, ready)
        # This room stops the arms: stop trying and stop nudging things over. Not while a hand holds something,
        # though: the other arm's wrist camera is what sees the held object, and without it the place round has no
        # view of what it is carrying and fails on empty masks (putting_away_toys, 2026-09-12).
        if self.blocked_swings >= BLOCKED_SWINGS_MAX and not self.hands():
            return self._capture_views(task, ready)
        hands = self.hands()
        held_arms = set(hands.values())
        holding_arms = sorted(hands[self.tracked_label(n)] for n in self.look_names if self.tracked_label(n) in hands)
        if holding_arms or (held_arms and self.look_target is None):
            target = self.eef_pose_base((holding_arms or sorted(held_arms))[0])[:3, 3]
        else:
            target = np.asarray(DEFAULT_LOOK_TARGET if self.look_target is None else self.look_target, dtype=np.float64)
        look = list(ready)  # the planned joints during the capture
        posture = dict(self.posture)  # the other arm's joints during the capture
        moved = {}  # arm -> {joint: value}
        aabbs = self.scene_aabbs()  # one snapshot for both arms' look configurations (nothing moves meanwhile)
        views = (self.primary_view, *self.extra_views)
        # A held object is seen by the other arm's wrist camera. With no wrist view in the capture there is
        # nothing to look with, and at the ready posture the gripper sits below the head camera's frame, so the
        # holding arm presents what it carries instead of staying put.
        present = held_arms and not any(v.endswith("_wrist") for v in views)
        for arm in ("left", "right"):
            holding = arm in held_arms
            if holding and not present:
                continue
            if not holding and arm != self.arm and f"{arm}_wrist" not in views:
                continue
            # An arm is posed for one of two reasons: its wrist camera is one of the views, or it stands in the
            # head camera's way. The planned arm was posed whatever the views were, so with head views alone
            # (--views head head_up head_down) it could be swung with no camera to aim. How often: over the 23
            # saved captures of runs/bench_toys_4, 2 swung, 7 tried and were blocked, and 14 never reached this
            # branch at all -- so this is a real motion for nothing, but a rare one, not the source of that run's
            # 26 blocked ramps (those were the head-view ramps; see _capture_views).
            if not holding and f"{arm}_wrist" not in views:
                in_the_way = self.links_on_sight_line(arm, target)
                if not in_the_way:
                    log.info(
                        f"{arm} arm: no wrist view in this capture and it is clear of the head camera's line to "
                        f"{np.round(target, 2).tolist()}; left where it is"
                    )
                    continue
                log.info(f"{arm} arm: {in_the_way} in the head camera's way; posing it out of the line")
            joints = list(self.robot.arm_joint_names[arm])
            q = self.present_held(arm, aabbs, self.look_names) if holding else self.wrist_look(arm, target, aabbs)
            if q is not None:
                moved[arm] = {j: float(v) for j, v in zip(joints, q)}
            elif holding:
                log.warning(f"{arm} arm: no configuration presents what it holds to the head camera; it stays put")
            elif arm == self.arm and any(j in self.look_arm for j in joints):
                ik = self.arm_ik(arm)
                q_now = self.robot.get_joint_positions()
                start = [float(q_now[self.joint_index[j]]) for j in joints]
                chosen, why = None, "the arm is in the way and no swing clears it"
                for fraction in LOOK_ARM_FRACTIONS:
                    swung = [
                        v + fraction * (float(self.look_arm[j]) - v) if j in self.look_arm else v
                        for j, v in zip(joints, start)
                    ]
                    if self.links_in_base_box(arm, ik, swung):
                        continue
                    blocked = self.links_in_scene(arm, ik, swung, aabbs)
                    if blocked:
                        why = f"swinging it out of view puts {blocked}"
                        continue
                    if self.links_before_camera(arm, ik, swung, target):
                        why = "the swing does not take it off the head camera's line"
                        continue  # it is still in the way: swing further
                    chosen, fraction_taken = swung, fraction
                    break
                if chosen is None:
                    log.warning(
                        f"{arm} arm: no look configuration for {np.round(target, 2).tolist()}, and {why}; it stays "
                        "where it is"
                    )
                else:
                    touched, objects = self.path_contacts(arm, ik, start, chosen, aabbs)
                    log.warning(
                        f"{arm} arm: no look configuration for {np.round(target, 2).tolist()}; swinging "
                        f"{fraction_taken:.0%} of the way out of view"
                        + (f", passing within {ARM_RADIUS} m of {objects} at {touched} point(s)" if touched else "")
                    )
                    moved[arm] = {j: float(v) for j, v in zip(joints, chosen)}
            else:
                log.warning(
                    f"{arm} arm: no look configuration for {np.round(target, 2).tolist()}; it stays where it is"
                )
            if arm == self.arm:
                look = [moved.get(arm, {}).get(j, v) for j, v in zip(self.planned_joints, ready)]
            else:
                posture.update(moved.get(arm, {}))
        if not moved:
            return self._capture_views(task, ready)
        original = self.posture
        struck = False  # whether this capture has already counted a blocked swing
        if self.ramp_arms(
            look,
            posture,
            sorted(moved),
            self.last_gripper,
            LOOK_SETTLE_STEPS,
            elbow_first=True,
            note="capture swing out",
        ):
            now = self.robot.get_joint_positions()
            for arm, targets in moved.items():
                lag = max(abs(float(now[self.joint_index[j]]) - v) for j, v in targets.items())
                if lag > LOOK_TOL:
                    log.warning(f"{arm} arm is {lag:.3f} rad short of its look posture (blocked?); capturing anyway")
            request, extras = self._capture_views(task, look)
            self._log_wrist_framing(request, moved)
            self.ramp_arms(
                ready,
                original,
                sorted(moved),
                self.last_gripper,
                LOOK_SETTLE_STEPS,
                elbow_first=False,
                note="return to ready after the capture",
            )
        else:  # it met something on the way out: go back first, then capture from where the arms rest
            struck = True  # this capture has already spent its strike; being stuck afterwards is the same event
            self.blocked_swings += 1
            log.warning("the capture swing stopped against something; capturing from the ready posture instead")
            self.ramp_arms(
                ready,
                original,
                sorted(moved),
                self.last_gripper,
                LOOK_SETTLE_STEPS,
                elbow_first=False,
                note="return to ready after a blocked swing",
            )
            moved, look = {}, list(ready)
            request, extras = self._capture_views(task, ready)
        q_ready = self.q_arm()
        lag = float(np.abs(q_ready - np.asarray(ready)).max())
        if lag > LOOK_TOL:  # it caught on something on the way back: try the direct path before giving up on it
            log.warning(f"arm {lag:.3f} rad short of the ready posture after the capture; ramping straight back")
            self.ramp_to(ready, original, self.last_gripper, LOOK_SETTLE_STEPS, note="straight back to ready")
            q_ready = self.q_arm()
            lag = float(np.abs(q_ready - np.asarray(ready)).max())
        if lag > LOOK_TOL:
            # The arm is resting against something. The plan is asked for from where the arm actually is
            # (``q_init`` below is the measured posture), so the round goes ahead rather than being lost; what
            # this costs is the ready posture's clean start, and the room gets one strike (BLOCKED_SWINGS_MAX) --
            # one strike per capture, not two. A blocked swing leaves the arm short of ready almost by
            # definition, so counting both spent the whole allowance on a single bad capture and turned the wrist
            # look poses off for the rest of the instance (measured 2026-09-13, batteries8.log instance 301).
            if not struck:
                self.blocked_swings += 1
            log.warning(
                f"arm {lag:.3f} rad from the ready posture after the capture and stuck there; planning from where "
                f"it is ({self.blocked_swings} blocked swing(s) this instance)"
            )
        now = self.robot.get_joint_positions()
        for arm in moved:
            if arm != self.arm:
                back = max(abs(float(now[self.joint_index[j]]) - original[j]) for j in self.robot.arm_joint_names[arm])
                if back > LOOK_TOL:
                    log.warning(f"{arm} arm is {back:.3f} rad from its locked posture after the capture")
        request["q_init"] = np.asarray(q_ready, dtype=np.float32)  # the plan starts here, not at the look posture
        extras["q_look"] = moved
        log.info(
            f"captured with {sorted(moved)} arm(s) posed; plan starts from the ready posture (max error {lag:.4f} rad)"
        )
        return request, extras

    def log_blocked_sight(self) -> list[str]:
        """Say which of the robot's own arm links stand between the head camera and what this capture is about.

        The self-mask zeroes the robot's own pixels, so an arm on that line does not darken the target, it deletes
        it: the mask comes back empty and the round is lost. The look poses are already filtered for this
        (``wrist_look``), but an arm that never moved -- no look configuration, or a swing that stopped against
        something and went back to the ready posture -- is not, and that is the posture most captures end in.
        Measured on 2026-09-13 (dispose_of_batteries, `scratchpad/stance_check.py`): the *same* stance, with the
        battery at the same pixel 0.67 m away, gave an empty mask in one capture and 499 pixels in the next; the
        stance was identical and the arms were not. The test is the segment against each arm link's own box
        (``segment_hits_box``), which is what the camera sees of the link -- ``links_before_camera`` has to use
        link origins because it judges a posture the arm has not taken yet. An arm holding something is skipped,
        since what it holds is usually the look target itself.
        """
        if self.look_target is None:
            return []
        held_arms = set(self.hands().values())
        blocking = []
        for arm in ("left", "right"):
            if arm in held_arms:
                continue
            hits = self.links_on_sight_line(arm, self.look_target)
            if hits:
                log.warning(f"{arm} arm stands between the head camera and the look target: {hits}")
            blocking += hits
        return blocking

    def links_on_sight_line(self, arm: str, target) -> list[str]:
        """Links of ``arm``, as they stand now, whose own box the head camera's line to ``target`` (base frame)
        passes through."""
        eye = self.base_to_world(self.head_camera_in_base()[1][:3, 3])
        goal = self.base_to_world(np.asarray(target, dtype=np.float64))
        hits = []
        for name in self.robot.arm_link_names[arm]:
            link = self.robot.links.get(name)
            if link is None:
                continue
            lo, hi = link.aabb
            if segment_hits_box(eye, goal, lo.cpu().numpy(), hi.cpu().numpy()):
                hits.append(name)
        return hits

    def _capture_views(self, task: str, q_arm) -> tuple[dict, dict]:
        """``TiptopSim.capture`` for the primary view and the wrist views, where the joints stand now (``q_arm``: the
        planned joints' targets, at torso yaw 0), then each head view of ``HEAD_VIEWS`` among ``extra_views``
        with the torso ramped to its yaw (``ramp_to``; every other joint stays where the capture found it, so a head
        view moves the torso and nothing else), and the torso ramped back. The base frame does not turn with the torso, so a view's
        camera pose, read from the simulator as it is rendered, is right as it is."""
        self.log_blocked_sight()
        head_views = [v for v in self.extra_views if v in HEAD_VIEWS]
        extra_views = self.extra_views
        self.extra_views = tuple(v for v in extra_views if v not in HEAD_VIEWS)
        try:
            request, extras = super().capture(task)
        finally:
            self.extra_views = extra_views
        if not head_views:
            return request, extras
        q_arm = [float(v) for v in q_arm]
        # A head view moves the torso and NOTHING else. It used to be built on the posture the caller intended,
        # so every head view also re-commanded the arm to that posture -- and an arm that was somewhere else,
        # because a swing had been blocked or a grasp had left it short, was dragged there against whatever had
        # stopped it. 24 of the 28 blocked ramps in the putting_away_toys run of 2026-09-13 were head-view ramps,
        # and the joint that blocked was always an arm joint being dragged, never the torso. Building the view on
        # the MEASURED joints leaves the arm where it is.
        where_it_is = [float(v) for v in self.q_arm()]
        for name in head_views:
            joint, delta = HEAD_VIEWS[name]
            self.ramp_to(
                turned_joints(self.planned_joints, where_it_is, joint, delta),
                self.posture,
                self.last_gripper,
                HEAD_VIEW_SETTLE_STEPS,
                note=f"torso to the {name} view",
            )
            view, view_extras = self.view_frame(name)
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
            log.info(
                f"head view {name}: {joint} moved {math.degrees(delta):+.0f} deg, camera at base "
                f"{np.round(view_extras['cam_pos_base'], 2).tolist()}"
            )
        # Back the same way: the torso to where it was, every other joint left where the views found it.
        back_to = list(where_it_is)
        for name in head_views:
            joint = HEAD_VIEWS[name][0]
            if joint in self.planned_joints:
                back_to[self.planned_joints.index(joint)] = q_arm[self.planned_joints.index(joint)]
        self.ramp_to(back_to, self.posture, self.last_gripper, HEAD_VIEW_SETTLE_STEPS, note="back from a head view")
        moved_joints = {HEAD_VIEWS[name][0] for name in head_views}
        back = max(
            abs(float(self.q_arm()[self.planned_joints.index(j)]) - q_arm[self.planned_joints.index(j)])
            for j in moved_joints
        )
        if back > HEAD_VIEW_RETURN_TOL:
            raise RuntimeError(f"the torso did not return after the head views (off by {back:.3f} rad)")
        request["q_init"] = self.q_arm()  # the plan starts here, not at a turned head view
        log.info(f"head views {head_views} taken; {sorted(moved_joints)} back (error {back:.4f} rad)")
        return request, extras

    def _log_wrist_framing(self, request: dict, arms) -> None:
        """Where the posed wrist cameras stand in the primary view's frame (an arm inside it hides the workspace)."""
        h, w = request["depth"].shape
        for arm in arms:
            pos_b, _ = self.to_base(*self.robot_cams[f"{arm}_wrist"].get_position_orientation())
            px, z = points_to_pixels([pos_b.cpu().numpy()], request["intrinsics"], request["world_from_cam"])
            (u, v), inside = px[0], bool(z[0] > 0 and 0 <= px[0][0] < w and 0 <= px[0][1] < h)
            log.info(
                f"{arm} wrist camera at base {np.round(pos_b.cpu().numpy(), 2).tolist()}: "
                + (
                    f"inside the {self.primary_view} frame at pixel ({u:.0f}, {v:.0f})"
                    if inside
                    else f"outside the {self.primary_view} frame"
                )
            )

    # ---------------------------------------------------------------- stepping
    def action(self, q_arm, gripper: float) -> dict:
        """23-D action: planned joints (torso + left arm, by name) to their targets, locked joints to their values."""
        targets = dict(self.posture)
        targets.update(zip(self.planned_joints, np.asarray(q_arm, dtype=np.float32).tolist()))
        idx = self.robot.controller_action_idx
        a = th.zeros(self.robot.action_dim, dtype=th.float32)
        a[idx["trunk"]] = th.tensor([targets[j] for j in self.robot.trunk_joint_names], dtype=th.float32)
        a[idx["arm_left"]] = th.tensor([targets[j] for j in self.robot.arm_joint_names["left"]], dtype=th.float32)
        a[idx[f"gripper_{self.arm}"]] = float(gripper)
        a[idx["arm_right"]] = th.tensor([targets[j] for j in self.robot.arm_joint_names["right"]], dtype=th.float32)
        a[idx[f"gripper_{self.other_arm}"]] = float(self.other_gripper)
        # base: HolonomicBaseJointController in position mode takes deltas, zeros hold the base still
        return {self.robot.name: a}

    def view_sensor(self, name: str):
        return self.shadows[VIEW_OPTICS[name]]

    def _capture_obs(self, name: str) -> tuple[dict, dict]:
        """Move the view's shadow camera onto its robot camera and render one rgb + depth (+ segmentation) frame."""
        robot_cam, shadow = self.robot_cams[name], self.view_sensor(name)
        pos, quat = robot_cam.get_position_orientation()
        shadow.set_position_orientation(position=pos, orientation=quat)
        # The renderer accumulates frames over time: after the camera jumps (base teleport, look posture) the first
        # frames are a ghost of the previous view, so render until two consecutive frames agree.
        previous = None
        for i in range(CAPTURE_MAX_RENDERS):
            og.sim.render()
            og.sim.render()
            rgb = shadow.get_obs()[0]["rgb"][..., :3].to(th.float32)
            if previous is not None and float((rgb - previous).abs().mean()) < CAPTURE_CONVERGED_DIFF:
                log.info(f"{name}: capture converged after {2 * (i + 1)} renders")
                break
            previous = rgb
        else:
            log.warning(
                f"{name}: capture did not converge after {2 * CAPTURE_MAX_RENDERS} renders; using the last frame"
            )
        k_robot, k_shadow = _intrinsics(robot_cam), _intrinsics(shadow)
        if not np.allclose(k_robot, k_shadow, atol=0.5):
            raise RuntimeError(
                f"{name}: shadow camera intrinsics {k_shadow.tolist()} != robot camera {k_robot.tolist()}"
            )
        p2, q2 = shadow.get_position_orientation()
        assert th.allclose(p2, pos, atol=1e-4) and th.allclose(q2.abs(), quat.abs(), atol=1e-4), (
            "shadow camera did not move"
        )
        return shadow.get_obs()

    def robot_self_mask(self, name: str, depth, intrinsics, cam_pos_world, cam_quat_cv_world) -> np.ndarray:
        """The robot's own pixels in a view from its link meshes: the viewing arm's links, gripper and fingers for
        a wrist view (the camera looks along its own gripper), both arms' for the head view."""
        arms = ["left", "right"] if VIEW_OPTICS[name] == "head" else [name.split("_")[0]]
        meshes = {}
        for arm in arms:
            for link_name in (
                *self.robot.arm_link_names[arm],
                *self.robot.gripper_link_names[arm],
                *self.robot.finger_link_names[arm],
            ):
                mesh = self.link_trimesh_world(self.robot.links[link_name], max_faces=SELF_MASK_FACES)
                if mesh is not None:
                    meshes[link_name] = mesh
        world_from_cam_w = T.pose2mat((cam_pos_world, cam_quat_cv_world)).cpu().numpy().astype(np.float64)
        masks = masks_from_geometry(depth, intrinsics, world_from_cam_w, meshes, tol=self.gt_mask_tol)
        return np.any(np.stack(list(masks.values())), axis=0) if masks else np.zeros(depth.shape, dtype=bool)

    def camera_rgb(self) -> np.ndarray | None:
        return self._robot_rgb(self.cam_name)

    def _robot_rgb(self, sensor_name: str) -> np.ndarray | None:
        if self.last_obs is None or sensor_name not in self.last_obs.get(self.robot.name, {}):
            return None
        return self.last_obs[self.robot.name][sensor_name]["rgb"][..., :3].cpu().numpy().astype(np.uint8)

    def video_views(self) -> dict:
        """The capture camera, the overview and the other robot cameras (video and Rerun mirror)."""
        views = super().video_views()
        for name, sensor in self.robot_cam_names.items():
            rgb = self._robot_rgb(sensor) if name != self.primary_view else None
            if rgb is not None:
                views[f"{name}_cam"] = rgb
        return views
