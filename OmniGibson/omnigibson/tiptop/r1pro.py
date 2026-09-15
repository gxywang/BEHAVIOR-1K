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

import trimesh
import numpy as np
import torch as th
import yaml
from bddl.condition_evaluation import HEAD

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects.usd_object import USDObject
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.tiptop.articulation import (
    OPEN_FRACTION_SCORED,
    follow_joint,
    handle_on,
    is_open,
    leading_direction,
    openable_joints,
    opening_travel,
    pose_matrix,
)
from omnigibson.tiptop.gt_masks import masks_from_geometry, points_within_tol
from omnigibson.tiptop.kinematics import ArmIK, link_from_camera, link_pose_for_camera, look_pose
from omnigibson.tiptop.protocol import (
    add_view,
    bddl_category,
    face_normal_local,
    joint_ramp,
    points_to_pixels,
    reach_candidates,
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
    "head_aim": "zed_link",
}
VIEW_OPTICS = {  # which shadow camera renders a view
    "head": "head",
    "left_wrist": "wrist",
    "right_wrist": "wrist",
    "head_left": "head",
    "head_right": "head",
    "head_up": "head",
    "head_down": "head",
    "head_aim": "head",
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
# The same yaw joint, turned by however much it takes to put what the stance was chosen for in the middle of the
# frame, instead of by a fixed amount. The head camera sees about +-50 degrees and the stance search has to put an
# object within the LEFT arm's reach, which for anything the robot cannot stand square to means well off to one
# side: of the 328 head-camera looks at a goal object that came back with an empty mask across runs/queue_logs,
# 75 had the object off the LEFT edge of the image and not one off the right, and 67 of those 75 come back inside
# the frame with a turn this joint can make. The turn is commanded from the camera's pose as it stands and the
# resulting pose is read back from the simulator when the view is rendered, so an imperfect aim costs a little
# centring and nothing else.
HEAD_AIM_VIEW = "head_aim"
HEAD_AIM_LIMIT = 0.6  # rad the torso may turn to aim (a little past the fixed +-0.5 rad views)
HEAD_AIM_MIN = 0.12  # rad: a smaller turn is not worth a ramp and a second render
TURNED_HEAD_VIEWS = (HEAD_AIM_VIEW,)  # head views beyond HEAD_VIEWS: same camera, same optics, computed turn
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
# Folding the arms in over the base, and putting them back, moves over the robot's own footprint rather than out
# through the room, so it is not what the capture cap is for and does not pay its price. At the capture speed a
# fold was 91 steps each way and 47% of an episode's budget; at this speed it is about a quarter of that.
TRAVEL_MAX_JOINT_VEL = 2.5
TRAVEL_POSE = 0.0  # every planned <arm>_arm_joint<n> goes here for the teleport; the torso is left alone
TRAVEL_SETTLE_STEPS = 12
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
# Lifts tried, in radians, to get the arm out of a surface before a plan is asked for from where it stands.
# Smallest first, so the posture moves as little as it must; see clear_start_posture for why this exists.
START_LIFTS = (0.15, 0.3, 0.5, 0.8)
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
# Furniture sent to the planner as static obstacles (see nearby_obstacles). Close enough to matter, big enough
# to be a fixture, and capped because every one of them costs a mask in the capture and a hull in perception.
OBSTACLE_REACH = 2.5  # m from the base; beyond this the arm cannot reach it anyway
OBSTACLE_LIMIT = 8
OBSTACLE_MIN_SIZE = 0.30  # m on its longest axis
HOUSE_AABB_AREA = 20.0  # m^2; larger boxes are merged walls, roofs or ceilings and say nothing about the floor
# What the base can drive over is decided by the base's own underside (see _footprint_free), not by a guess at
# how thin a thing is: the 8 cm rule that used to live here exempted the toys a task has to pick up.
# place_robot: the overview camera in the base's frame, (eye dx, eye dy, eye z, target dx, target z); "shoulder" looks
# over the left shoulder at the workspace, "front" looks back at the chest so both hands and what they hold are in view
OVERVIEW_OFFSETS = {"shoulder": (-1.5, 1.1, 1.7, 0.7, 0.55), "front": (1.15, -0.75, 1.35, 0.3, 0.9)}
FOOTPRINT_CELL = 0.05  # m: the grid the base-height geometry of a scene object is measured on
AVOID_RADIUS = 0.15  # a retried base pose must be at least this far (m) from the ones tried before
TILT_LIMIT_DEG = 1.0  # a base that settles further off level than this is fighting something it was put in
SHIFT_LIMIT = 0.02  # m it may slide while settling before the same is true
# The robot no longer folds before a teleport. What that fold cost, and why the stance search replaced it rather
# than the fold being tuned, is written up in place_robot. Measured on 2026-09-13 by ramping to each candidate and
# reading back the overhang past the base's own rectangle (x -0.39..0.24, y -0.34..0.34) and whether the ramp
# finished: the working posture leaves 41.2 cm; a torso fold to -2.25 leaves 29.2 cm and never arrives; every
# planned arm joint to zero leaves 9.5 cm and does arrive. That last one was adopted, and then measured to cost
# 7967 of an episode's 16946 steps, which is what removed it.
FOLD_OVERHANG = 0.07  # m the folded upper body still reaches beyond the base: what the teleport actually lands as
# Room the stance search wants between the robot and everything it is not there to touch, and what a metre short
# of it costs in the score. Without this the score is indifferent between standing 1 mm from a cabinet and 6 cm
# from it and takes the nearer one, because it is 6 cm closer to the object: the filter decides everything and the
# objective pushes straight back against it. A clearance term also covers every residual error in the shapes at
# once -- box against mesh, the arms the footprint does not model, the base settling -- which a bigger footprint
# does not. At 4.0 per metre a 5 cm shortfall costs 0.20 against the 0.05 of approach it buys.
STANCE_CLEARANCE = 0.05
CLEAR_WEIGHT = 4.0
# Opening a container: how finely the joint's path is followed, and the holds around it. The steps matter more
# than they look -- the hand is holding the link, so a coarse path drags it through poses its joint does not
# allow and the grasp is what gives way.
OPEN_PATH_STEPS = 16  # waypoints of the pull, the grasp included; every one is solved before the hand moves
OPEN_SETTLE_STEPS = 15
OPEN_GRASP_STEPS = 25  # closing on the handle before any pulling starts
OPEN_APPROACH = 0.20  # m off the grip point along the pull: where the hand waits before it comes straight in
# The reach and the pull run at this joint speed rather than CAPTURE_MAX_JOINT_VEL. At 0.6 rad/s the ramp to a
# drawer's standoff reported torso_joint1 0.18 rad behind its target 18 steps into a 114-step ramp from a stance
# 0.9 m off the cabinet, and torso_joint2 0.14 rad behind at step 51 of 120 the day before -- the torso carries the
# whole upper body and does not track 0.6 rad/s, and the block detector reads that lag as a collision (2026-09-14).
OPEN_MAX_JOINT_VEL = 0.3
OPEN_JUMP_TOL = 0.8  # rad: a joint moving further than this between two adjacent waypoints is flipping, not pulling
OPEN_FOLLOW_TOL = 0.02  # m the hand may get ahead of a drawer by before the pull is called lost
OPEN_FOLLOW_TOL_RAD = 0.05  # the same for a door, in radians of the hinge
OPEN_MIN_FRACTION = 0.20  # of the joint's range: the shortest pull worth making when the arm cannot follow it all
OPEN_STANCE_TRIES = 120  # (stance, grasp) pairs solved for before a container is given up on
# A reach ramp that reports a joint behind its target at its LAST step is not a collision on the way: the baseline
# stopped 'left_arm_joint1 0.12 rad behind at step 121 of 122' and this code 'left_arm_joint2 0.13 rad behind at
# step 153 of 154' (2026-09-14), a shoulder settling short of a stretched configuration. What matters is where the
# HAND ended up, so the reach goes on when it is within this of the standoff, and the approach re-targets from there.
OPEN_REACH_SLACK = 0.06  # m
OPEN_GRASP_TOLERANCE_RAD = 0.15  # rad the grasp pose may be off in orientation: the fingertips are 7.8 cm from the
# IK frame, so 0.5 rad (the tolerance the rest of the pipeline uses) lets them wander 3.7 cm, and a rail is 1.2 cm tall
BAR_TIP_CLEARANCE = 0.005  # m the fingertips stop short of the panel behind a bar
BAR_TIP_DEPTH = 0.03  # m behind a bar's front the fingertips go at most, so the pads hold it and the assist's ray
# between them crosses it (fridge/petcxr's bar stands 6 cm out; the pads are 1.8 cm long)
BAR_END_INSET = 0.03  # m kept clear of a bar's ends
HAND_BODY_CLEARANCE = 0.01  # m the hand's link origins keep from the container's own body (the hand works at it)
STANCE_AHEAD = (0.55, 0.65, 0.45, 0.75, 0.85)  # m the handle is ahead of the base, first choice first
STANCE_SIDE = (0.15, 0.25, 0.05, 0.35, -0.05)  # m the handle is to the LEFT of the base (the left arm opens)
STANCE_YAWS = (0.0, 0.26, -0.26)  # rad off square to the face
# How far the fingertips are driven past the surface they are taking hold of. The assisted grasp the
# simulator uses fires on finger CONTACT, so the tips have to reach the panel; a few millimetres of overlap
# makes the contact certain without the panel pushing the arm off its target.
GRASP_PRESS = 0.005
# The arm IK tolerance used when closing on something. The pipeline's usual 2 cm is wider than the press
# above, so a "solved" grasp can sit clear of the surface with nothing between the fingers.
OPEN_GRASP_TOLERANCE = 0.004
# Pressing in until the fingers actually touch. The arm does not land exactly where it is commanded -- with
# the fingertips sent 5 mm INTO store_honey's drawer front they settled 3 mm clear of it, about 8 mm of
# tracking error against a 5 mm press -- and the assist fires on contact, so a press smaller than that error
# grasps nothing. Rather than guess a constant big enough, the hand presses deeper a step at a time and stops
# as soon as the fingers report contact. The cap keeps it inside a drawer front's thickness (15.5 mm here).
GRASP_NUDGE = 0.008  # m deeper per attempt
GRASP_NUDGES = 3  # attempts after the first, so at most 24 mm past where the fingertips were first sent
# Places to try taking hold of one panel, spread up its face. A fridge door is 2 m tall and only a band of it
# is in the arm's reach, so one point per joint is one guess; these are offered nearest the hand's own height
# first, which is both the likeliest to solve and the least the arm has to travel.
# An object thinner than this on its smallest axis is flat: M2T2 has no side to propose a grasp on, so the
# hand is pressed onto its top face instead (press_grasp). A hardback is about 3 cm, a board game about 5.
FLAT_THICKNESS = 0.06
GRASP_COLUMN_SAMPLES = 5
GRASP_COLUMN_INSET = 0.08  # m kept clear of the panel's top and bottom edges, so the jaw lands on the face
GRASP_FACE_SAMPLES = 5  # rays across each direction of a face when looking for the surface to close on
GRASP_FACE_FRACTION = 0.35  # how far across the face they spread, as a fraction of its half-extent
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


def rect_box_gap(centre, yaw: float, rect_lo, rect_hi, box_lo, box_hi) -> float:
    """How much room there is between a rectangle at ``centre`` turned by ``yaw`` and an axis-aligned box, in xy.

    Positive is clearance in metres; zero or less means they overlap. ``rect_lo``/``rect_hi``: the rectangle's own
    extent in its frame, which for the robot's base is NOT centred on the origin -- it reaches further behind the
    base frame than in front of it. Separating-axis test over the four axes (the two world axes and the two the
    rectangle turns to); the widest separation is the clearance, and no separation at all means overlap.

    The number matters as much as the verdict. Swapping the old centred 0.36 m square for this rectangle tightens
    the robot's rear by up to 16 cm and LOOSENS its front by 12 cm (the base reaches only 0.24 m forward), and the
    front is the side that faces the furniture -- so with nothing in the score to want clearance, an honest shape
    alone would just let the search stand 12 cm closer (measured over the 61 stances of runs/bench_batteries_ten).
    """
    c = np.asarray(centre, dtype=np.float64).reshape(2)
    lo = np.asarray(rect_lo, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi, dtype=np.float64).reshape(2)
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    rect = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) @ rot.T + c
    box_lo = np.asarray(box_lo, dtype=np.float64).reshape(-1)[:2]
    box_hi = np.asarray(box_hi, dtype=np.float64).reshape(-1)[:2]
    box = np.array([[x, y] for x in (box_lo[0], box_hi[0]) for y in (box_lo[1], box_hi[1])])
    gap = -np.inf
    for axis in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), rot[:, 0], rot[:, 1]):
        a, b = rect @ axis, box @ axis
        gap = max(gap, float(b.min() - a.max()), float(a.min() - b.max()))
    return gap


def rect_hits_box(centre, yaw: float, rect_lo, rect_hi, box_lo, box_hi) -> bool:
    """Whether a turned rectangle overlaps an axis-aligned box (``rect_box_gap`` at or below zero)."""
    return rect_box_gap(centre, yaw, rect_lo, rect_hi, box_lo, box_hi) <= 0.0


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
    box and the toy beside it whole, and a clipped container is worth more than no stance at all -- but that
    exemption applies only to an object some of which is in the picture, because an object close to the camera and
    far off its axis projects to a box bigger than the frame while landing entirely outside it. ``strict=False``
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
        ahead = z > 0
        if not np.any(ahead):
            return "behind the head camera", 0.0
        if not np.all(ahead):
            # Part of the box is behind the camera plane. Requiring ALL of it in front is unsatisfiable for
            # anything taller than the camera that the robot has to stand within arm's reach of: standing to open
            # a fridge, 4310 of the 5832 candidate stances were thrown out as "behind the head camera" and the
            # round never happened (2026-09-13). A partly-seen box is a CUT, which is what the strict pass rejects
            # and the fallback pass merely charges for -- the same treatment as a box cut by the image edge.
            if strict:
                return "cut by the head camera's near plane", 0.0
            px, z = px[ahead], z[ahead]
        lo, hi = px.min(axis=0), px.max(axis=0)
        low = np.array([margin_px, margin_px], dtype=np.float64)
        high = np.array([width - 1 - margin_px, height - 1 - margin_px], dtype=np.float64)
        cut = float(np.sum(np.maximum(0.0, low - lo) + np.maximum(0.0, hi - high)))
        # Not one pixel of it lands in the picture. This has to be judged before the "too big to fit" exemption
        # below, because apparent size is not physical size: an object close to the camera and far off its axis
        # projects to a HUGE box precisely because it is nearly beside the lens, so the exemption was letting
        # through exactly the stances that see nothing. Measured against the real head camera (fixture of
        # bench_batteries_8): a hamburger 0.34 m ahead of the base and 0.61 m to its left -- the stance
        # packing_meal_for_delivery actually took on 2026-09-14 -- projects to pixel (-727, 951) of a 720x720
        # image and the strict pass ACCEPTED it, while the same object 0.15 m to the left, far better framed, was
        # rejected. The round went out, the capture saw nothing and it died on empty masks.
        if strict and not (np.all(lo <= high) and np.all(hi >= low)):
            return "out of the head camera's frame altogether", 0.0
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
    if camera in HEAD_VIEWS or camera in TURNED_HEAD_VIEWS:
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


def head_aim_yaw(cam_pos_base, cam_forward_base, target_base, limit: float = HEAD_AIM_LIMIT) -> float:
    """How far to turn the torso's yaw joint to bring ``target_base`` into the middle of the head camera's frame.

    The camera's position and optical axis and the target are all in the robot base frame. The joint turns the
    camera about the base's vertical, so only the horizontal bearing can be corrected: the answer is the angle
    from where the camera looks now to where the target lies, wrapped to (-pi, pi] and clamped to +-``limit``.
    Positive is to the robot's left, the same sign as ``HEAD_VIEWS["head_left"]``.

    The camera's own pitch is left alone. A head camera pitched 43 degrees down (the challenge posture) still has
    three quarters of its optical axis in the horizontal plane, which is what the bearing is taken from; a camera
    looking straight down has no bearing to speak of and gets no turn.
    """
    here = np.asarray(cam_pos_base, dtype=np.float64)[:2]
    fwd = np.asarray(cam_forward_base, dtype=np.float64)[:2]
    to_target = np.asarray(target_base, dtype=np.float64)[:2] - here
    if np.linalg.norm(fwd) < 1e-6 or np.linalg.norm(to_target) < 1e-6:
        return 0.0
    delta = math.atan2(to_target[1], to_target[0]) - math.atan2(fwd[1], fwd[0])
    delta = (delta + math.pi) % (2 * math.pi) - math.pi
    return float(np.clip(delta, -abs(limit), abs(limit)))


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
        if camera in HEAD_VIEWS or camera in TURNED_HEAD_VIEWS:
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
        turned = set(HEAD_VIEWS) | set(TURNED_HEAD_VIEWS)  # the head camera re-aimed, not cameras of their own
        self.robot_cam_names = {n: s for n, s in sensor_names.items() if n not in turned}  # one per camera
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
        self._hand_convention = {}  # arm -> how its hand approaches and closes (constant; measured on first use)
        self._stance_iks = {}  # arm -> the IK the stance search reuses (built once, not per candidate)
        self._base_cells = {}  # object name -> the floor squares it really occupies at base height
        self.send_obstacles = False  # tell the planner about the room's furniture (--obstacles); measured, not assumed
        self.overview = self.env.external_sensors.get(OVERVIEW_CAM)
        self.objects = {}
        self.context = {}  # furniture shown in the Rerun mirror (track_context)
        self.obstacles = {}  # furniture sent to the planner as a static obstacle this stance (nearby_obstacles)
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

    def base_height_cells(self, obj) -> set | None:
        """Which ``FOOTPRINT_CELL`` squares of floor this object actually occupies at the height the BASE sweeps.

        A bounding box is not the object. A desk is legs and a top, and at the height the robot's base occupies
        there is almost nothing in between -- measured in picking_up_toys' scene, where the desk that turned the
        robot away has a 1.83 m2 box footprint and 0.08 m2 of solid geometry in that slab, 96% air; the breakfast
        table and the coffee table are 100% air, so the base could roll clean underneath them. Testing the base
        against the box refuses positions where there is nothing at all, which is why every task whose objects sit
        on a desk failed with "no base pose ... overlaps desk" before a single round ran (2026-09-14).

        It is NOT true of everything: a bed is 59-62% air, so this has to come from each object's own geometry
        rather than a rule about tables. Returns None when the object has no mesh, and the caller then keeps the
        box's word.

        Computed once per object -- the mesh is cached and nothing moves during a stance search -- because doing
        it per candidate is what took an earlier version of this search from a median of 1.0 s to 48.8 s.
        """
        if obj.name in self._base_cells:
            return self._base_cells[obj.name]
        cells = None
        try:
            mesh = self.scene_mesh(obj)
            points = np.asarray(mesh.vertices, dtype=np.float64)
            lo_b, hi_b = self.base_box()
            floor = float(self.base_pose()[0][2])
            slab = points[(points[:, 2] >= floor + float(lo_b[2])) & (points[:, 2] <= floor + float(hi_b[2]))]
            cells = {(int(np.floor(x / FOOTPRINT_CELL)), int(np.floor(y / FOOTPRINT_CELL))) for x, y in slab[:, :2]}
        except Exception as why:  # noqa: BLE001 - no mesh, or an unreadable one: the box stands
            log.debug(f"no base-height geometry for {obj.name} ({type(why).__name__}); keeping its box")
            cells = None
        self._base_cells[obj.name] = cells
        return cells

    def base_meets(self, obj, centre, yaw, rect_lo, rect_hi) -> bool:
        """Whether the base's rectangle actually meets ``obj``, rather than merely meeting its bounding box."""
        cells = self.base_height_cells(obj)
        if cells is None:
            return True  # nothing better to go on than the box, which the caller has already found overlapping
        if not cells:
            return False  # the object has no geometry at all in the slab the base sweeps: it passes under it
        cx, cy = float(centre[0]), float(centre[1])
        c, sn = math.cos(float(yaw)), math.sin(float(yaw))
        reach = float(max(abs(rect_lo[0]), abs(rect_hi[0]), abs(rect_lo[1]), abs(rect_hi[1]))) + FOOTPRINT_CELL
        for gx, gy in cells:
            x = (gx + 0.5) * FOOTPRINT_CELL - cx
            y = (gy + 0.5) * FOOTPRINT_CELL - cy
            if abs(x) > reach or abs(y) > reach:
                continue
            fx, fy = x * c + y * sn, -x * sn + y * c  # the cell in the base's own frame
            if rect_lo[0] <= fx <= rect_hi[0] and rect_lo[1] <= fy <= rect_hi[1]:
                return True
        return False

    def _stance_ik(self, arm: str) -> ArmIK:
        """Arm IK kept for the stance search. Building one per candidate would cost more than the search itself."""
        if arm not in self._stance_iks:
            self._stance_iks[arm] = self.arm_ik(arm, frame=f"{arm}_gripper_link")
        return self._stance_iks[arm]

    def _footprint_free(
        self, x: float, y: float, ignore, aabbs=None, yaw: float | None = None, arms: bool = True
    ) -> tuple[bool, str]:
        """Floor under the whole footprint, inside a room, and no object's box overlapping the base.

        ``arms``: also refuse a stance where an arm at its CURRENT posture would rest inside something (below).
        Off for an opening stance, whose arms stay folded over the base after the teleport and go from there
        to the handle by a checked path: with the working posture they would sit 0.41 m past the base, i.e.
        inside the very drawer front the stance is chosen to reach, and every stance near it was refused.

        With ``yaw``, the base is tested as the rectangle it actually is, turned to face that way
        (``base_box``/``rect_hits_box``); without one, as the ``ROBOT_FOOTPRINT`` square, which is what callers
        that have no yaw yet get. The square is centred and yaw-independent, and the base is neither: it reaches
        0.40 m behind the base frame and 0.24 m ahead, so a turned base puts its rear corner 0.52 m out where the
        square guards 0.36. Measured over runs/bench_batteries_ten: **23 of the 60 stances taken overlapped a
        piece of furniture**, up to 0.15 m into a cabinet and 0.05 m into a desk, which is what the user saw in
        the videos as the robot standing too close and clipping the cabinet (2026-09-13).

        ``aabbs``: a scene_aabbs() snapshot to test against (taken here otherwise; nothing moves during a search).
        """
        r = ROBOT_FOOTPRINT
        rect = (
            self.base_box()[:, :2] + np.array([[-FOLD_OVERHANG] * 2, [FOLD_OVERHANG] * 2]) if yaw is not None else None
        )
        corners = [(x + sx * r, y + sy * r) for sx in (-1, 1) for sy in (-1, 1)] + [(x, y)]
        if rect is not None:  # the floor test wants the real corners too
            rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            corners = [
                tuple(np.array([cx, cy]) @ rot.T + np.array([x, y]))
                for cx in (rect[0][0], rect[1][0])
                for cy in (rect[0][1], rect[1][1])
            ] + [(x, y)]
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        underside = float(self.base_pose()[0][2]) + float(self.base_box()[0][2])  # world z the base clears
        clearance = float("inf")  # the least room to any obstacle, so the score can prefer a stance with some
        floors = [(lo, hi) for o, lo, hi in aabbs if o.category == "floors"]
        for cx, cy in corners:
            on_floor = False
            for lo, hi in floors:
                if lo[0] <= cx <= hi[0] and lo[1] <= cy <= hi[1]:
                    on_floor = True
                    break
            if not on_floor:
                return False, f"no floor under ({cx:.2f}, {cy:.2f})", 0.0
        try:
            room = self.env.scene.seg_map.get_room_instance_by_point(th.tensor([x, y]))
        except Exception:  # noqa: BLE001 - a point off the map's raster raises inside the lookup; it is a filter only
            room = "unknown"
        if room is None:
            return False, "outside every room", 0.0
        for obj, lo, hi in aabbs:
            if obj is self.robot or obj in ignore or obj.category in FLOOR_COVERINGS:
                continue
            if (hi[0] - lo[0]) * (hi[1] - lo[1]) > HOUSE_AABB_AREA:
                continue  # merged walls, roof, ceilings say nothing; the floor test handles walls
            if lo[2] > ROBOT_HEIGHT:
                continue  # entirely above the robot (roof, lamps)
            if hi[2] <= underside:
                continue  # it passes under the base: a rug, a threshold, a cable. Anything standing taller than
                # the base's underside is an obstacle, however thin. The old test exempted everything under 8 cm
                # lying on the floor, which is written for pavers and mats and matches the task's own objects: 7
                # of the 8 toy figures of runs/bench_toys_7 are 0.043-0.079 m thick and were exempt in 32 to 40 of
                # the 41 captures each, and the base ended up 28 cm inside toy_figure_6's box, riding on it at
                # -4.5 deg of roll (2026-09-13).
            if rect is not None:
                gap = rect_box_gap((x, y), yaw, rect[0], rect[1], lo, hi)
                if gap <= 0.0:
                    # The box says they meet; ask the object itself. A desk is legs and a top, and at the height
                    # the base sweeps it is 96% air -- a coffee table is 100% air, so the base can roll right
                    # under it. Refusing on the box turned the robot away from every desk in the house.
                    if not self.base_meets(obj, (x, y), yaw, rect[0], rect[1]):
                        continue
                    return False, f"overlaps {obj.name}", 0.0
                clearance = min(clearance, gap)
            elif lo[0] < x + r and hi[0] > x - r and lo[1] < y + r and hi[1] > y - r:
                return False, f"overlaps {obj.name}", 0.0
        # The base's rectangle is not the robot: with the working posture the arms reach 0.41 m past it, so a
        # stance whose
        # base is clear can still leave the hand inside a box on the floor. The user watched exactly that in
        # putting_away_toys: after picking up a toy the robot teleported to the table and its arm came to rest
        # INSIDE the toy box (2026-09-13). The arms are tested in 3D, at the posture they will unfold to and at
        # the pose being judged, so a stance is refused for where the arm ENDS UP rather than only for where the
        # wheels are. Objects being stood for are in ``ignore``: the arm is meant to reach those.
        if yaw is not None and self.q_home is not None and arms:
            spared = {o.name for o in ignore}
            joints = self.robot.get_joint_positions()
            for arm in self.robot.arm_names:
                try:
                    ik = self._stance_ik(arm)
                    q = [float(joints[self.joint_index[j]]) for j in self.robot.arm_joint_names[arm]]
                except Exception:  # noqa: BLE001 - no description for this arm; the base test still stands
                    continue
                inside = [
                    n for n in self.arm_hits_scene(arm, ik, q, aabbs=aabbs, at=(x, y, float(yaw))) if n not in spared
                ]
                if inside:
                    return False, f"the {arm} arm would come to rest in {inside[0]}", 0.0
        return True, "free", float(clearance)

    def settled_level(self, x: float, y: float, tilt_deg: float = TILT_LIMIT_DEG, shift: float = SHIFT_LIMIT) -> tuple:
        """(level, why) for where the base actually settled after being teleported to ``x``, ``y``.

        Nothing used to read the base back. When a commanded pose intersects furniture the physics resolves it by
        lifting and rolling the whole robot, and the round then runs from a base frame that is not level -- which
        every field of the request is expressed in, so the planner receives the whole scene tilted against gravity.
        Measured over runs/bench_batteries_ten: 15 of 61 rounds settled off the nominal, up to 0.101 m of lift and
        20.4 deg of roll, and **1 of those 14 rounds executed against 29 of the 46 level ones** (2026-09-13).

        Roll and pitch and the xy shift are used rather than the height, because their nominal value is exactly
        zero while the settled height is a property of the scene's floor.
        """
        pos, quat = self.base_pose()
        roll, pitch, _ = T.quat2euler(quat)
        tilt = math.degrees(max(abs(float(roll)), abs(float(pitch))))
        moved = float(np.hypot(float(pos[0]) - x, float(pos[1]) - y))
        if tilt > tilt_deg:
            return False, f"the base settled {tilt:.1f} deg off level"
        if moved > shift:
            return False, f"the base slid {moved * 100:.0f} cm from where it was put"
        return True, ""

    def surface_point(self, link, point, direction, reach: float = 0.5):
        """First point of ``link``'s own surface that a ray along ``direction`` meets, starting ``reach`` outside
        ``point``. Returns ``point`` unchanged when the link has no mesh or the ray misses it. World frame.

        A bounding box is not a surface. store_honey's drawer fronts have their box face 3.0 cm in FRONT of the
        panel -- the box picks up a lip at the drawer's outer edges, and every one of the four drawers reports the
        same face -- so a hand sent to the box face closed on empty air 2.6 cm short of the panel and the assist
        had nothing to take hold of ("the fingers touch nothing at all", 2026-09-13). Taking the point off the
        geometry works whatever the shape: a flat drawer front, a door, a handle that protrudes.
        """
        mesh = self.link_trimesh_world(link)
        if mesh is None or not len(mesh.faces):
            return np.asarray(point, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        origin = np.asarray(point, dtype=np.float64) - d * reach
        try:
            hits, _, _ = mesh.ray.intersects_location(ray_origins=origin.reshape(1, 3), ray_directions=d.reshape(1, 3))
        except Exception as why:  # noqa: BLE001 - a missing ray engine must not stop the round
            log.debug(f"ray against {link.name} failed ({type(why).__name__}); using the box face")
            return np.asarray(point, dtype=np.float64)
        if not len(hits):
            return np.asarray(point, dtype=np.float64)
        along = (np.asarray(hits, dtype=np.float64) - origin) @ d
        return np.asarray(hits, dtype=np.float64)[int(np.argmin(along))]

    def hand_convention(self, arm: str) -> dict:
        """Which way this hand approaches, which way its jaw closes, and where the grasp actually happens.

        All of it measured off the robot rather than assumed, because the assumption was wrong: the frame the arm
        IK solves for, ``<arm>_gripper_link``, is 6 cm BEHIND the point the fingers close on (``<arm>_eef_link``),
        so every pose aimed straight at a surface put the fingers 6 cm inside it. That is what stopped the drawer
        smoke tests, blocked one step after the approach every time (2026-09-13).

        Returned in the IK frame's own axes, so it holds at any arm configuration:
          approach  unit vector from the IK frame towards the fingertips (the way the hand goes in)
          jaw       unit vector between the two fingers (the way the jaw closes), square to ``approach``
          grasp     offset of the grasp centre, i.e. where a held object ends up
          tip       how far the fingertips reach past that grasp centre, along ``approach``
        """
        if arm in self._hand_convention:
            return self._hand_convention[arm]
        link = self.robot.links[f"{arm}_gripper_link"]
        pos, quat = link.get_position_orientation()
        rot = T.quat2mat(quat).cpu().numpy().astype(np.float64)
        origin = pos.cpu().numpy().astype(np.float64)

        def to_frame(points):
            return (rot.T @ (np.asarray(points, dtype=np.float64) - origin).T).T

        centroids, cloud = [], []
        for finger in self.robot.finger_links[arm]:
            mesh = self.link_trimesh_world(finger)
            if mesh is None or not len(mesh.vertices):
                continue
            local = to_frame(mesh.vertices)
            centroids.append(local.mean(axis=0))
            cloud.append(local)
        if len(centroids) < 2:
            raise RuntimeError(f"{arm} hand has fewer than two fingers with meshes; cannot measure its convention")
        # The fingers sit either side of the approach axis, so their midpoint lies on it and the line between them
        # is the jaw. Both come out of the meshes, so a differently built hand measures differently and still works.
        middle = np.mean(centroids, axis=0)
        approach = middle / max(float(np.linalg.norm(middle)), 1e-9)
        jaw = centroids[0] - centroids[1]
        jaw = jaw - approach * float(jaw @ approach)
        jaw = jaw / max(float(np.linalg.norm(jaw)), 1e-9)
        eef = self.robot.eef_links.get(arm)
        if eef is not None and eef.prim_path != link.prim_path:
            grasp = to_frame([eef.get_position_orientation()[0].cpu().numpy()])[0]
        else:  # no separate end-effector frame: the grasp happens between the fingertips
            grasp = approach * float(np.concatenate(cloud) @ approach).max() * 0.5
        tip = float(np.max(np.concatenate(cloud) @ approach) - grasp @ approach)
        out = {"approach": approach, "jaw": jaw, "grasp": grasp, "tip": tip}
        self._hand_convention[arm] = out
        log.info(
            f"{arm} hand: approach {np.round(approach, 3).tolist()}, jaw {np.round(jaw, 3).tolist()}, "
            f"grasp centre {np.round(grasp, 3).tolist()} ({float(np.linalg.norm(grasp)):.3f} m out), "
            f"fingertips {tip:.3f} m past it -- all in the {arm}_gripper_link frame"
        )
        return out

    def grasp_point_on(self, link, aim_world, approach_world, back_off: float = 0.35) -> tuple:
        """Where a link's surface actually is along the line the hand comes in on, and how far it stands proud.

        A bounding box is not a surface. store_honey's drawer link measures x[1.3301, 1.8221] and the hand aimed at
        1.3301 as the front of the panel -- but a ray down the approach at the middle of that drawer first meets it
        at 1.3599, so the fingertips stopped 2.6 cm short, in clear air, and the assist had nothing to hold: "the
        fingers touch nothing at all" (2026-09-13). Whatever stands 3 cm proud of the panel sets the box and is not
        where the hand was going.

        Rays are cast over a patch of the face and the most protruding hit wins, so a handle is taken hold of when
        the asset has one and the flat panel when it does not -- without either being named anywhere.

        Returns (point, hits, proud): the world point to close on, how many rays found the link, and how far the
        chosen point stands in front of the flattest one. Falls back to ``aim_world`` when no ray finds it.
        """
        mesh = self.link_trimesh_world(link)
        aim = np.asarray(aim_world, dtype=np.float64)
        direction = np.asarray(approach_world, dtype=np.float64)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
        if mesh is None or not len(mesh.faces):
            return aim, 0, 0.0
        # Two directions across the face, and how far the link reaches along each, so the patch scales to the part
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(direction, up)
        if float(np.linalg.norm(side)) < 1e-6:
            side = np.cross(direction, [1.0, 0.0, 0.0])
        side = side / max(float(np.linalg.norm(side)), 1e-9)
        up = np.cross(side, direction)
        lo, hi = (v.cpu().numpy().astype(np.float64) for v in link.aabb)
        half = (hi - lo) / 2.0
        reach_side = float(np.abs(half @ side)) * GRASP_FACE_FRACTION
        reach_up = float(np.abs(half @ up)) * GRASP_FACE_FRACTION
        best = None
        hits = 0
        for a in np.linspace(-reach_side, reach_side, GRASP_FACE_SAMPLES):
            for b in np.linspace(-reach_up, reach_up, GRASP_FACE_SAMPLES):
                start = aim + side * a + up * b - direction * back_off
                where, _, _ = mesh.ray.intersects_location([start], [direction])
                if not len(where):
                    continue
                hits += 1
                travel = [(float((w - start) @ direction), np.asarray(w, dtype=np.float64)) for w in where]
                travel.sort(key=lambda row: row[0])
                # nearest along the approach, and nearest the middle of the patch when two stand equally proud
                key = (round(travel[0][0], 3), abs(a) + abs(b))
                if best is None or key < best[0]:
                    best = (key, travel[0][0], travel[0][1])
        if best is None:
            return aim, 0, 0.0
        aim_travel = float((aim - (aim - direction * back_off)) @ direction)
        return best[2], hits, aim_travel - best[1]

    def grasp_target(self, arm: str, point, approach_dir, jaw_dir=None, press: float = GRASP_PRESS):
        """Pose for the arm IK that closes this hand on ``point``, coming in along ``approach_dir``.

        Takes the point the FINGERTIPS should reach and returns where the IK frame has to be, which are 6 cm and a
        rotation apart (``hand_convention``). ``press`` drives the tips that far past the point so the contact the
        assisted grasp waits for actually happens. Base frame in, base frame out.
        """
        hand = self.hand_convention(arm)
        a_dir = np.asarray(approach_dir, dtype=np.float64)
        a_dir = a_dir / max(float(np.linalg.norm(a_dir)), 1e-9)
        j_dir = np.asarray(jaw_dir if jaw_dir is not None else [0.0, 0.0, 1.0], dtype=np.float64)
        j_dir = j_dir - a_dir * float(j_dir @ a_dir)
        if float(np.linalg.norm(j_dir)) < 1e-6:  # asked for a jaw along the approach: any square direction will do
            j_dir = np.cross(a_dir, [1.0, 0.0, 0.0])
            if float(np.linalg.norm(j_dir)) < 1e-6:
                j_dir = np.cross(a_dir, [0.0, 1.0, 0.0])
        j_dir = j_dir / max(float(np.linalg.norm(j_dir)), 1e-9)
        local = np.stack([hand["approach"], hand["jaw"], np.cross(hand["approach"], hand["jaw"])], axis=1)
        world = np.stack([a_dir, j_dir, np.cross(a_dir, j_dir)], axis=1)
        rot = world @ local.T
        # tips ``press`` past the point, so the grasp centre sits back by the fingertip reach less the press
        centre = np.asarray(point, dtype=np.float64) + a_dir * (press - hand["tip"])
        return centre - rot @ hand["grasp"], rot

    def close_on(self, arm: str, ik, obj, link_name: str, pose, into, seed, joints_of, nudges: int = GRASP_NUDGES) -> tuple:
        """Close the hand on ``obj``'s link and press in until the assist takes hold. (seed, held) afterwards.

        The assist fires on finger CONTACT, and the arm does not stop exactly where it is told, so a fixed press
        depth either misses (and grasps nothing) or is chosen big enough to shove the thing being grasped. This
        presses a step at a time along ``into`` and stops at the first contact, which needs no constant to be
        right and reports the gap it measured when it fails.
        """
        panel = self.link_trimesh_world(obj.links[link_name])
        for attempt in range(nudges + 1):
            self.hold(OPEN_GRASP_STEPS, self.CLOSE)
            held = self.robot._ag_obj_in_hand.get(arm)
            if held is not None and held.name == obj.name:
                log.info(f"the assist has {obj.name} after {attempt + 1} press(es)")
                return seed, True
            gap = ""
            try:
                touching, _ = self.robot._find_gripper_contacts(arm=arm)
                gaps = []
                for finger in self.robot.finger_links[arm]:
                    mesh = self.link_trimesh_world(finger)
                    if mesh is None or panel is None or not len(panel.faces):
                        continue
                    _, dist, _ = trimesh.proximity.closest_point(panel, np.asarray(mesh.vertices, dtype=np.float64))
                    gaps.append(float(np.min(dist)))
                gap = f"fingers {'touch ' + str(len(touching)) + ' thing(s)' if touching else 'touch nothing'}" + (
                    f", nearest the panel by {min(gaps) * 100:.1f} cm" if gaps else ""
                )
            except Exception as why:  # noqa: BLE001 - a diagnostic must not replace the failure
                gap = f"could not measure the gap ({type(why).__name__})"
            if attempt == nudges:
                log.info(f"no hold on {obj.name} after {attempt + 1} presses; {gap}")
                return seed, False
            deeper = np.asarray(pose, dtype=np.float64).copy()
            deeper[:3, 3] = deeper[:3, 3] + np.asarray(into, dtype=np.float64) * GRASP_NUDGE * (attempt + 1)
            quat = T.mat2quat(th.tensor(deeper[:3, :3], dtype=th.float32)).cpu().numpy()
            solution = ik.solve(deeper[:3, 3], quat, seed=seed, tolerance_pos=OPEN_GRASP_TOLERANCE, tolerance_rad=0.5)
            if solution is None:
                log.info(
                    f"no hold on {obj.name}; cannot press {GRASP_NUDGE * (attempt + 1) * 100:.1f} cm deeper; {gap}"
                )
                return seed, False
            log.info(f"{gap}; pressing {GRASP_NUDGE * (attempt + 1) * 100:.1f} cm deeper")
            targets = [float(v) for v in self.q_arm()]
            for name_j, value in zip(joints_of, solution):
                if name_j in self.planned_joints:
                    targets[self.planned_joints.index(name_j)] = float(value)
            self.ramp_to(targets, self.posture, self.CLOSE, OPEN_SETTLE_STEPS, note="press onto the panel")
            seed = [float(v) for v in solution]
        return seed, False

    # ---------------------------------------------------------------- opening a container
    def _targets_from(self, joints_of, solution) -> list[float]:
        """The planned-joint vector with an IK solution (over ``joints_of``) written into it."""
        targets = [float(v) for v in self.q_arm()]
        for name_j, value in zip(joints_of, solution):
            if name_j in self.planned_joints:
                targets[self.planned_joints.index(name_j)] = float(value)
        return targets

    def container_grasps(self, obj, joints, fraction: float, hand_world, height: float | None = None) -> list:
        """Every way of taking hold of ``obj``'s openable links, best first: one dict per (joint, point on the
        handle), each with the joint, its signed ``travel``, the handle's ``kind`` (``articulation.handle_on``),
        the world point the FINGERTIPS go to (``tips``), the approach direction (into the face, ``into``), the
        jaw directions to try (``jaws``), the press depth for ``grasp_target`` and the leading direction ``lead``.

        On a bar the tips go ``BAR_TIP_CLEARANCE`` short of the panel behind it, but no more than
        ``BAR_TIP_DEPTH`` behind the bar's front so it is the pads that hold it and the assist's ray between the
        pads still crosses it; the grasp is offered at the bar's middle first and then out along it. On a lip or
        a flat panel the pressed-face grasp of ``close_on`` is offered at a column of heights (the old behaviour).
        """
        out = []
        for j in joints:
            travel = opening_travel(j["kind"], j["lower"], j["upper"], j["position"], fraction)
            link = obj.links.get(j["link"])
            if link is None or abs(travel) < 1e-4:
                continue
            mesh = self.link_trimesh_world(link)
            if mesh is None or not len(mesh.vertices):
                continue
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            lead = leading_direction(j["kind"], j["axis"], j["origin"], verts, travel)
            h = handle_on(verts, lead)
            log.info(
                f"{obj.name}.{j['name']} ({j['kind']}, travel {travel:+.3f}): leading face looks along "
                f"{np.round(lead, 2).tolist()}; its handle is a {h['kind'].upper()} standing {h['proud'] * 100:.1f} cm "
                f"proud (proud part {np.round(h['extent'], 3).tolist()} m, face {np.round(h['face_extent'], 2).tolist()} m)"
            )
            into = -lead
            if h["kind"] == "bar":
                along, (lo_a, hi_a) = h["along"], h["ends"]
                mid = (lo_a + hi_a) / 2.0
                half = max(0.0, (hi_a - lo_a) / 2.0 - BAR_END_INSET)
                offsets = sorted({float(np.clip(o, -half, half)) for o in (0.0, 0.15, -0.15, 0.30, -0.30)}, key=abs)
                tips_s = max(h["panel"] + BAR_TIP_CLEARANCE, h["front"] - BAR_TIP_DEPTH)
                if height is not None:
                    log.info(f"{obj.name}.{j['name']}: the task names z {height:.3f}; a bar sets its own height, ignored")
                for o in offsets:
                    p = np.asarray(h["point"], dtype=np.float64) + along * (mid + o - float(h["point"] @ along))
                    p = p + lead * (tips_s - float(p @ lead))
                    # nearest the hand's own height first (a bank of drawers: the one the arm reaches with the
                    # least bending), then nearest the bar's middle
                    rank = abs(float(p[2] - hand_world[2])) + abs(o)
                    out.append(
                        dict(joint=j, travel=travel, kind="bar", tips=p, into=into, jaws=[h["jaw"], -h["jaw"]],
                             press=0.0, lead=lead, rank=(0, rank), handle=h, nudges=1)
                    )  # fmt: skip
                continue
            # a lip or a flat panel: press both pads on the face, at heights along it
            face_c = np.asarray(h["point"], dtype=np.float64)
            lo_z, hi_z = float(link.aabb[0][2]), float(link.aabb[1][2])
            band = max(0.0, (hi_z - lo_z) / 2.0 - GRASP_COLUMN_INSET)
            middle = (lo_z + hi_z) / 2.0
            if height is not None:
                heights = [float(height)]
                log.info(f"{obj.name}.{j['name']}: this task names z {height:.3f} to take hold at")
            else:
                heights = sorted(
                    {round(float(np.clip(z, lo_z + GRASP_COLUMN_INSET, hi_z - GRASP_COLUMN_INSET)), 4)
                     for z in np.linspace(middle - band, middle + band, GRASP_COLUMN_SAMPLES)}
                )  # fmt: skip
            uprights = [np.array([0.0, 0.0, 1.0]), h["side"]]
            for z in heights:
                point = np.array([face_c[0], face_c[1], z], dtype=np.float64)
                on_surface = self.surface_point(link, point, into)
                out.append(
                    dict(joint=j, travel=travel, kind=h["kind"], tips=on_surface, into=into, jaws=uprights,
                         press=GRASP_PRESS, lead=lead, rank=(1, abs(float(on_surface[2] - hand_world[2]))),
                         handle=h, nudges=GRASP_NUDGES)
                )  # fmt: skip
        out.sort(key=lambda g: g["rank"])
        return out

    def _grasp_pose_base(self, arm: str, grasp: dict, jaw_world, base_pose=None) -> tuple:
        """(position, rotation 3x3) of the IK frame, in the base frame, for a grasp dict and a jaw direction.

        ``base_pose``: (x, y, yaw) of a base the robot is NOT at yet, so a stance can be judged before it is
        taken; the robot's own pose otherwise."""
        if base_pose is None:
            unit = th.tensor([0.0, 0.0, 0.0, 1.0])
            origin = self.to_base(th.tensor([0.0, 0.0, 0.0]), unit)[0].cpu().numpy()
            tips = self.to_base(th.tensor(grasp["tips"], dtype=th.float32), unit)[0].cpu().numpy()
            into = self.to_base(th.tensor(grasp["into"], dtype=th.float32), unit)[0].cpu().numpy() - origin
            jaw = self.to_base(th.tensor(jaw_world, dtype=th.float32), unit)[0].cpu().numpy() - origin
        else:
            x, y, yaw = base_pose
            c, s = math.cos(float(yaw)), math.sin(float(yaw))
            rot_t = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            base_z = float(self.base_pose()[0][2])
            tips = rot_t @ (np.asarray(grasp["tips"], dtype=np.float64) - np.array([x, y, base_z]))
            into = rot_t @ np.asarray(grasp["into"], dtype=np.float64)
            jaw = rot_t @ np.asarray(jaw_world, dtype=np.float64)
        return self.grasp_target(arm, tips, into, jaw, press=grasp["press"])

    def _pull_poses(self, grasp: dict, start_pose, base_pose=None) -> tuple[list, np.ndarray, np.ndarray]:
        """The gripper poses of the pull (base frame): the grasp pose carried along the joint by follow_joint."""
        j = grasp["joint"]
        if base_pose is None:
            unit = th.tensor([0.0, 0.0, 0.0, 1.0])
            origin = self.to_base(th.tensor([0.0, 0.0, 0.0]), unit)[0].cpu().numpy()
            axis = self.to_base(th.tensor(j["axis"], dtype=th.float32), unit)[0].cpu().numpy() - origin
            hinge = self.to_base(th.tensor(j["origin"], dtype=th.float32), unit)[0].cpu().numpy()
        else:
            x, y, yaw = base_pose
            c, s = math.cos(float(yaw)), math.sin(float(yaw))
            rot_t = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            base_z = float(self.base_pose()[0][2])
            axis = rot_t @ np.asarray(j["axis"], dtype=np.float64)
            hinge = rot_t @ (np.asarray(j["origin"], dtype=np.float64) - np.array([x, y, base_z]))
        return follow_joint(start_pose, j["kind"], axis, hinge, grasp["travel"], steps=OPEN_PATH_STEPS), axis, hinge

    def solve_pull(self, ik, grasp: dict, jaw_world, seed, base_pose=None, aabbs=None, arm: str = "left", obj=None):
        """IK for the whole motion of one grasp -- the pre-grasp standoff, the grasp, and every waypoint of the
        pull -- solved in order, each seeded by the last, with every configuration checked against the scene
        (the container excepted). None when the standoff or the grasp has no solution or stands in something;
        otherwise a dict with the solutions, how many pull waypoints solved (``reached``), the poses and why it
        stopped short. A path that solves only part of the way is kept when it opens at least
        ``OPEN_MIN_FRACTION`` of the range: the scored atom flips at 5%, and a door's 86 deg arc is beyond
        any fixed stance.
        """
        where, rot = self._grasp_pose_base(arm, grasp, jaw_world, base_pose)
        quat = T.mat2quat(th.tensor(rot, dtype=th.float32)).cpu().numpy()
        grasp_pose = pose_matrix(where, quat)
        j = grasp["joint"]
        if base_pose is None:
            unit = th.tensor([0.0, 0.0, 0.0, 1.0])
            origin = self.to_base(th.tensor([0.0, 0.0, 0.0]), unit)[0].cpu().numpy()
            lead_b = self.to_base(th.tensor(grasp["lead"], dtype=th.float32), unit)[0].cpu().numpy() - origin
        else:
            c, s = math.cos(float(base_pose[2])), math.sin(float(base_pose[2]))
            lead_b = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(grasp["lead"], dtype=np.float64)
        standoff = grasp_pose.copy()
        standoff[:3, 3] = standoff[:3, 3] + lead_b * OPEN_APPROACH
        at = None if base_pose is None else tuple(float(v) for v in base_pose)
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        clear = [row for row in aabbs if obj is None or row[0] is not obj]
        body = self.container_body(obj, j["link"]) if obj is not None else None
        solutions, poses = [], [standoff, grasp_pose]
        pull, _, _ = self._pull_poses(grasp, grasp_pose, base_pose)
        poses += pull[1:]
        q = list(seed)
        why = ""
        for i, pose in enumerate(poses):
            quat_i = T.mat2quat(th.tensor(pose[:3, :3], dtype=th.float32)).cpu().numpy()
            solution = None
            for tol_p, tol_r in ((OPEN_GRASP_TOLERANCE, OPEN_GRASP_TOLERANCE_RAD), (0.01, 0.3)):
                solution = ik.solve(pose[:3, 3], quat_i, seed=q, tolerance_pos=tol_p, tolerance_rad=tol_r)
                if solution is not None:
                    break
            if solution is None:
                why = f"no inverse kinematics for {'the standoff' if i == 0 else 'the grasp' if i == 1 else f'pull waypoint {i - 1} of {len(pull) - 1}'}"
                break
            if solutions and float(np.max(np.abs(np.asarray(solution) - np.asarray(solutions[-1])))) > OPEN_JUMP_TOL:
                why = f"the arm would flip {float(np.max(np.abs(np.asarray(solution) - np.asarray(solutions[-1])))):.2f} rad between waypoints at {i}"
                break
            hits = self.arm_hits_scene(arm, ik, solution, aabbs=clear, at=at, mesh=True)
            if not hits and self.body_hits(arm, ik, solution, body, at=at):
                hits = [f"{obj.name}'s body"]
            if hits:
                why = f"the arm at {'the standoff' if i == 0 else 'the grasp' if i == 1 else f'pull waypoint {i - 1}'} would be in {hits[0]}"
                break
            solutions.append([float(v) for v in solution])
            q = solutions[-1]
        if len(solutions) < 2:
            return None, why
        reached = len(solutions) - 2  # pull waypoints solved
        span = abs(float(j["upper"] - j["lower"]))
        fraction_done = abs(grasp["travel"]) * reached / (len(pull) - 1) / max(span, 1e-9)
        if reached < len(pull) - 1 and fraction_done < OPEN_MIN_FRACTION:
            return None, f"{why}; only {fraction_done * 100:.0f}% of the range is reachable"
        return dict(solutions=solutions, poses=poses[: len(solutions)], reached=reached, why=why, lead=lead_b,
                    grasp_pose=grasp_pose, fraction=fraction_done), why  # fmt: skip

    def stance_for_grasp(self, obj, grasps: list, arm: str = "left", limit: int = OPEN_STANCE_TRIES) -> tuple:
        """A base pose in front of the container's leading face from which the whole opening motion solves and
        the arm is clear of the scene, and the grasp it was found for: ((x, y, yaw), grasp, jaw_world) or
        (None, None, None).

        Candidates put the handle ``STANCE_AHEAD`` ahead of the base and ``STANCE_SIDE`` to its left, facing the
        face; each is refused for its footprint first (``_footprint_free``: the base must not overlap the
        container or anything else), then by ``solve_pull`` at that pose. This replaces standing by
        ``best_base_pose`` at the container's centroid, which is built for looking at things and stood the rig
        0.9 m from a drawer front, at furniture backs and against walls.
        """
        aabbs = self.scene_aabbs()
        ik = self.arm_ik(arm, frame=f"{arm}_gripper_link", with_torso=True)
        joints_of = self.ik_joint_names(arm, with_torso=True)
        seed = [float(self.q_home[self.planned_joints.index(j)]) if j in self.planned_joints else 0.0 for j in joints_of]
        tried, footprints = 0, {}
        refused = {}
        for grasp in grasps:
            lead = np.asarray(grasp["lead"], dtype=np.float64).copy()
            lead[2] = 0.0
            if float(np.linalg.norm(lead)) < 1e-6:  # a lid: come at it from where the robot stands
                here = self.base_pose()[0].cpu().numpy().astype(np.float64)
                lead = here - np.asarray(grasp["tips"], dtype=np.float64)
                lead[2] = 0.0
            fwd = -lead / max(float(np.linalg.norm(lead)), 1e-9)
            tips = np.asarray(grasp["tips"], dtype=np.float64)
            for ahead in STANCE_AHEAD:
                for side in STANCE_SIDE:
                    for dyaw in STANCE_YAWS:
                        yaw = math.atan2(fwd[1], fwd[0]) + dyaw
                        c, s = math.cos(yaw), math.sin(yaw)
                        f2, l2 = np.array([c, s]), np.array([-s, c])
                        xy = tips[:2] - f2 * ahead - l2 * side
                        key = (round(float(xy[0]), 3), round(float(xy[1]), 3), round(yaw, 3))
                        if key not in footprints:
                            footprints[key] = self._footprint_free(
                                float(xy[0]), float(xy[1]), [], aabbs=aabbs, yaw=yaw, arms=False
                            )
                        free, why, _ = footprints[key]
                        if not free:
                            refused[why] = refused.get(why, 0) + 1
                            continue
                        for jaw in grasp["jaws"]:
                            tried += 1
                            plan, why = self.solve_pull(ik, grasp, jaw, seed, base_pose=(float(xy[0]), float(xy[1]), yaw),
                                                        aabbs=aabbs, arm=arm, obj=obj)  # fmt: skip
                            if plan is not None:
                                log.info(
                                    f"stance for {obj.name}.{grasp['joint']['name']}: ({xy[0]:.2f}, {xy[1]:.2f}) yaw "
                                    f"{math.degrees(yaw):.0f} deg, handle {ahead:.2f} m ahead and {side:+.2f} m left; "
                                    f"{plan['reached']} of {OPEN_PATH_STEPS - 1} pull waypoints solve "
                                    f"({plan['fraction'] * 100:.0f}% of the range) after {tried} tries"
                                    + (f"; stops because {plan['why']}" if plan["why"] else "")
                                )
                                return (float(xy[0]), float(xy[1]), float(yaw)), grasp, jaw
                            refused[why.split(" at ")[0][:60]] = refused.get(why.split(" at ")[0][:60], 0) + 1
                            if tried >= limit:
                                log.info(f"no stance for {obj.name} after {tried} tries: {dict(sorted(refused.items(), key=lambda kv: -kv[1])[:5])}")
                                return None, None, None
        log.info(f"no stance for {obj.name} after {tried} tries: {dict(sorted(refused.items(), key=lambda kv: -kv[1])[:5])}")
        return None, None, None

    def reach_plan(self, arm: str, ik, joints_of, q_to, exclude=(), aabbs=None, body=None) -> list:
        """Legs (each a full solution over ``joints_of``) that take the arm from where it is to ``q_to`` without
        sweeping through the scene, chosen among: straight; via the ready posture; torso first then the arm; the
        arm first then the torso; elbow first. Each leg is a straight ramp in joint space, which is what
        ``ramp_to`` executes, so ``path_hits_scene`` judges the motion the arm will really make. The first clean
        plan wins; when none is clean the plan with the fewest objects swept is taken and logged, because
        refusing to move at all was measured to be worse than moving through a box (2026-09-14).
        """
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        clear = [row for row in aabbs if row[0].name not in exclude]
        q = self.robot.get_joint_positions()
        q_from = [float(q[self.joint_index[j]]) for j in joints_of]
        home = [float(self.q_home[self.planned_joints.index(j)]) if j in self.planned_joints else q_from[i]
                for i, j in enumerate(joints_of)]  # fmt: skip
        # An arm still in its travel fold hangs beside the base with the hand low, and a straight joint-space line
        # from there to a standoff swings the forearm through whatever the robot is standing at (the other
        # drawer fronts of jhymlr's cabinet, 2026-09-14). Unfolding to the ready posture first is the motion every
        # teleport already makes, so from the fold it is the first plan offered.
        folded = all(abs(q_from[i]) < 0.05 for i, j in enumerate(joints_of) if "_arm_joint" in j)
        best = None
        for name, legs in reach_candidates(joints_of, q_from, q_to, home, elbow=ELBOW, prefer_home=folded):
            swept, start = [], q_from
            for leg in legs:
                for hit in self.path_hits_scene(arm, ik, start, leg, aabbs=clear, mesh=True):
                    if hit not in swept:
                        swept.append(hit)
                if self.path_hits_body(arm, ik, start, leg, body) and "the container's body" not in swept:
                    swept.append("the container's body")
                start = leg
            if not swept:
                log.info(f"reach: {name} is clear")
                return legs
            if best is None or len(swept) < len(best[2]):
                best = (name, legs, swept)
        log.info(f"reach: no plan is clear; {best[0]} sweeps the least ({best[2]})")
        return best[1]

    def follow_pull(self, arm: str, joints_of, plan: dict, obj, grasp: dict) -> tuple[int, str]:
        """Stream the pre-solved pull waypoints as one continuous motion at ``OPEN_MAX_JOINT_VEL`` with the gripper
        closed, and after each waypoint compare what the hand did with what the container's joint did: a hand
        that has moved on while the joint stays put has lost its hold, and the pull stops there rather than
        dragging an empty hand along the path. (waypoints completed, why it stopped)."""
        j = grasp["joint"]
        link = self.robot.eef_links[arm]
        start = link.get_position_orientation()[0].cpu().numpy().astype(np.float64)
        joint0 = next((k["position"] for k in openable_joints(obj) if k["name"] == j["name"]), j["position"])
        lead = np.asarray(grasp["lead"], dtype=np.float64)
        axis = np.asarray(j["axis"], dtype=np.float64)
        hinge = np.asarray(j["origin"], dtype=np.float64)
        done, why = 0, ""
        pulls = plan["solutions"][2:]
        for i, solution in enumerate(pulls):
            stopped = self.ramp_to(
                self._targets_from(joints_of, solution), self.posture, self.CLOSE, 0,
                note=f"pull {obj.name} waypoint {i + 1} of {len(pulls)}", max_vel=OPEN_MAX_JOINT_VEL,
            )  # fmt: skip
            now = link.get_position_orientation()[0].cpu().numpy().astype(np.float64)
            joint_now = next((k["position"] for k in openable_joints(obj) if k["name"] == j["name"]), joint0)
            if j["kind"] == "prismatic":
                hand, unit_s = float((now - start) @ lead), "m"
            else:
                r0, r1 = start - hinge, now - hinge
                r0, r1 = r0 - axis * (r0 @ axis), r1 - axis * (r1 @ axis)
                hand = float(math.atan2(float(np.cross(r0, r1) @ axis), float(r0 @ r1)))
                unit_s = "rad"
            moved = float(joint_now - joint0)
            asked = grasp["travel"] * (i + 1) / (OPEN_PATH_STEPS - 1)
            held = self.robot._ag_obj_in_hand.get(arm)
            log.info(
                f"  pull {i + 1}/{len(pulls)}: asked {asked:+.3f}, hand moved {hand:+.3f} {unit_s}, {j['name']} moved "
                f"{moved:+.3f} {unit_s}{'' if held is not None else ' -- the assist holds nothing'}"
            )
            tol = OPEN_FOLLOW_TOL if j["kind"] == "prismatic" else OPEN_FOLLOW_TOL_RAD
            if abs(hand) - abs(moved) > tol and abs(hand) > tol:
                why = f"hand moved {hand:+.3f} {unit_s} but {j['name']} only {moved:+.3f}: the hold was lost at waypoint {i + 1}"
                break
            done = i + 1
            if stopped is not None:
                why = f"{stopped[0]} stopped following at waypoint {i + 1} of {len(pulls)}"
                break
        return done, why

    def open_container(
        self,
        arm: str,
        name: str,
        fraction: float = OPEN_FRACTION_SCORED,
        joint: str | None = None,
        height: float | None = None,
        stand: bool = True,
    ) -> dict:
        """Take hold of a container's handle and follow its joint, opening it by ``fraction`` of its range.

        The one motion in the pipeline where the gripper has to follow a path rather than reach a pose: a drawer
        slides, a door swings, and the link comes with the hand. In order:
          1. the handle: read off the moving link's own mesh (``articulation.handle_on``) -- a bar is closed
             around, a flat face is pressed on; no asset marks one;
          2. the stance (``stand``): a base pose in front of the leading face from which the standoff, the
             grasp and the pull all solve with the arm clear of the scene (``stance_for_grasp``);
          3. the reach: the way from the current posture to the standoff is chosen among several by
             ``path_hits_scene`` (``reach_plan``), then the hand comes straight in along the pull;
          4. the pull: the pre-solved waypoints of ``follow_joint`` streamed as one motion, the container's joint
             read at each and the pull stopped where the hand moves on without it (``follow_pull``);
          5. release and retreat back along the pull.
        Returns what happened, for the round's record. The joint comes from the simulator (``openable_joints``),
        privileged in the way the oracle's button poses are.
        """
        obj = self.scene_object(name)
        joints = openable_joints(obj)
        if not joints:
            return {"opened": False, "why": f"{name} has no joint that opens"}
        if joint is not None:
            named = [j for j in joints if j["name"] == joint]
            if not named:
                log.warning(f"{name}: this task names joint {joint!r}, which it does not have: {[j['name'] for j in joints]}")
            else:
                log.info(f"{name}: this task names joint {joint!r}; opening that one")
                joints = named
        hand_world = self.robot.eef_links[arm].get_position_orientation()[0].cpu().numpy().astype(np.float64)
        grasps = self.container_grasps(obj, joints, fraction, hand_world, height=height)
        if not grasps:
            already = any(is_open(j["lower"], j["upper"], j["position"]) for j in joints)
            return {"opened": already, "why": "already open" if already else f"no joint of {name} can be taken hold of"}
        ik = self.arm_ik(arm, frame=f"{arm}_gripper_link", with_torso=True)
        joints_of = self.ik_joint_names(arm, with_torso=True)
        chosen, jaw_world, plan = None, None, None
        if stand:
            pose, chosen, jaw_world = self.stance_for_grasp(obj, grasps, arm=arm)
            if pose is None:
                return {"opened": False, "why": f"no stance in front of {name} lets the arm reach its handle and pull"}
            self.place_robot(*pose, note=f"stand to open {name}", unfold=False)
            self.hold(OPEN_SETTLE_STEPS, self.OPEN)
        # solve from where the robot actually stands (a teleport settles a little off the pose asked for), seeded
        # from the ready posture as the stance search was, not from the travel fold the arms are in now
        q = self.robot.get_joint_positions()
        seed = [
            float(self.q_home[self.planned_joints.index(j)]) if self.q_home and j in self.planned_joints
            else float(q[self.joint_index[j]])
            for j in joints_of
        ]  # fmt: skip
        aabbs = self.scene_aabbs()
        candidates = [(chosen, jaw_world)] if chosen is not None else []
        candidates += [(g, jaw) for g in grasps for jaw in g["jaws"] if g is not chosen]
        for g, jaw in candidates[: OPEN_STANCE_TRIES]:
            plan, why = self.solve_pull(ik, g, jaw, seed, aabbs=aabbs, arm=arm, obj=obj)
            if plan is not None:
                chosen, jaw_world = g, jaw
                break
            log.info(f"{name}.{g['joint']['name']}: {why}; trying the next grasp")
        if plan is None:
            return {"opened": False, "why": f"no grasp on {name} solves from here"}
        j = chosen["joint"]
        log.info(
            f"taking hold of {name}.{j['name']} by its {chosen['kind']} at {np.round(chosen['tips'], 3).tolist()} "
            f"(fingertips), jaw {np.round(jaw_world, 2).tolist()}; {plan['reached']} of {OPEN_PATH_STEPS - 1} "
            f"pull waypoints solve" + (f" ({plan['why']})" if plan["why"] else "")
        )
        # 3. reach the standoff, collision-checked, then come straight in
        legs = self.reach_plan(
            arm, ik, joints_of, plan["solutions"][0], exclude=(obj.name,), aabbs=aabbs, body=self.container_body(obj, j["link"])
        )
        for k, leg in enumerate(legs):
            stopped = self.ramp_to(self._targets_from(joints_of, leg), self.posture, self.OPEN, OPEN_SETTLE_STEPS,
                                   note=f"reach the standoff of {name} (leg {k + 1} of {len(legs)})", max_vel=OPEN_MAX_JOINT_VEL)  # fmt: skip
            if stopped is not None:
                q_now = self.robot.get_joint_positions()
                measured = [float(q_now[self.joint_index[jn]]) for jn in joints_of]
                off = float(np.linalg.norm(ik.fk(measured, f"{arm}_gripper_link")[0] - ik.fk(leg, f"{arm}_gripper_link")[0]))
                if off > OPEN_REACH_SLACK or k + 1 < len(legs):
                    self.hold(OPEN_SETTLE_STEPS, self.OPEN)
                    return {"opened": False, "joint": j["name"], "position": j["position"],
                            "why": f"{stopped[0]} stopped following on the way to the standoff (leg {k + 1} of {len(legs)}, "
                                   f"hand {off * 100:.1f} cm short)"}  # fmt: skip
                log.info(f"{stopped[0]} settled {stopped[2]:.2f} rad short at the end of the reach; the hand is {off * 100:.1f} cm off the standoff, going on")
        stopped = self.ramp_to(self._targets_from(joints_of, plan["solutions"][1]), self.posture, self.OPEN,
                               OPEN_SETTLE_STEPS, note=f"approach the handle of {name}", max_vel=OPEN_MAX_JOINT_VEL)  # fmt: skip
        if stopped is not None:
            log.info(f"{stopped[0]} stopped {stopped[2]:.2f} rad short on the approach; closing where the hand is")
        seed, grabbed = self.close_on(arm, ik, obj, j["link"], plan["grasp_pose"], -plan["lead"], plan["solutions"][1],
                                      joints_of, nudges=chosen["nudges"])  # fmt: skip
        if not grabbed:
            log.info(f"nothing to pull on: the assist never took hold of {name}.{j['name']}")
        # 4. the pull
        done, blocked = self.follow_pull(arm, joints_of, plan, obj, chosen)
        # 5. let go and back off along the pull, checked like everything else
        self.hold(OPEN_SETTLE_STEPS, self.OPEN)
        back = None
        try:
            pos_now, quat_now = ik.fk(plan["solutions"][1 + done], f"{arm}_gripper_link")
            retreat = ik.solve(np.asarray(pos_now) + plan["lead"] * OPEN_APPROACH, quat_now, seed=plan["solutions"][1 + done],
                               tolerance_pos=0.02, tolerance_rad=0.3)  # fmt: skip
            if retreat is not None and not self.arm_hits_scene(arm, ik, retreat, aabbs=[r for r in aabbs if r[0] is not obj]):
                back = retreat
        except Exception as why_r:  # noqa: BLE001 - the retreat is a courtesy
            log.info(f"no retreat solved ({type(why_r).__name__}: {why_r})")
        if back is not None:
            self.ramp_to(self._targets_from(joints_of, back), self.posture, self.OPEN, OPEN_SETTLE_STEPS,
                         note=f"back off from {name}", max_vel=OPEN_MAX_JOINT_VEL)  # fmt: skip
        after = openable_joints(obj)
        now = next((k for k in after if k["name"] == j["name"]), j)
        opened = is_open(now["lower"], now["upper"], now["position"])
        log.info(
            f"{name}.{j['name']} ({j['kind']}): asked for {chosen['travel']:+.3f}, pulled {done} of {plan['reached']} "
            f"solved waypoints ({OPEN_PATH_STEPS - 1} in the full path), joint now {now['position']:.3f} of "
            f"[{now['lower']:.2f}, {now['upper']:.2f}] -- {'OPEN' if opened else 'still closed'}"
            + (f"; {blocked}" if blocked else "")
        )
        return {"opened": opened, "why": blocked, "joint": j["name"], "position": now["position"], "grip": chosen["kind"],
                "waypoints": done, "solved": plan["reached"], "held": bool(grabbed)}  # fmt: skip

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
            ok, why, _ = self._footprint_free(x, y, ignore=ignore, yaw=float(yaw))
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
        Returns ((score, x, y, yaw, dists, sides, clearance) or None, rejection counts by reason).
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
        best, rejected, footprint = None, {}, {}  # footprint: (x, y, yaw) -> _footprint_free result
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
                        key = (float(x), float(y), round(float(yaw), 3))  # the base's box turns with the yaw
                        if key not in footprint:
                            footprint[key] = self._footprint_free(x, y, ignore, aabbs=aabbs, yaw=float(yaw))
                        free, why, clearance = footprint[key]
                        if not free:
                            rejected[why] = rejected.get(why, 0) + 1
                        else:
                            # Standing closer is worth 1 per metre to the score above, so without this the search
                            # takes every centimetre the filter allows and stops a hair from the furniture.
                            score += CLEAR_WEIGHT * max(0.0, STANCE_CLEARANCE - clearance)
                            if best is None or score < best[0]:
                                best = (score, x, y, yaw, dist, side, clearance)
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
        score, x, y, yaw, dist, side, clearance = best
        log.info(
            f"standing for {' + '.join(names)}: ({x:.2f}, {y:.2f}) yaw {np.degrees(yaw):.0f} deg, "
            f"distances {np.round(dist, 2).tolist()} m, left offsets {np.round(side, 2).tolist()} m, "
            f"{clearance:.2f} m of room to the nearest obstacle"
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

    def fold_for_travel(self) -> list | None:
        """Bring the arms in over the base before the base teleports; the posture to unfold back to, or None.

        The robot materialises at the new stance rather than driving to it, so the posture it lands in is the whole
        question. With the working posture the arms sit 0.41 m past the base's own rectangle; with every planned
        arm joint at zero, 0.095 m -- four times better, and the only candidate that actually arrives (2026-09-13).

        Removed for a day because it cost 47% of an episode's steps, which was the wrong half to remove: that cost
        was in ramping it at the CAPTURE speed cap, which exists for observation swings out through a scene nobody
        has planned. Folding in over the robot's own base is the opposite motion, so it runs at
        TRAVEL_MAX_JOINT_VEL and costs about a quarter as much.
        """
        if self.q_home is None or not self.planned_joints:
            return None
        here = [float(v) for v in self.q_arm()]
        folded = list(here)
        moved = False
        for index, joint in enumerate(self.planned_joints):
            if "_arm_joint" in joint:
                folded[index] = float(TRAVEL_POSE)
                moved = True
        if not moved:
            return None
        blocked = self.ramp_to(
            folded,
            self.posture,
            self.last_gripper,
            TRAVEL_SETTLE_STEPS,
            note="fold for travel",
            max_vel=TRAVEL_MAX_JOINT_VEL,
        )
        if blocked is not None:
            log.warning(f"the fold before the teleport was stopped by {blocked[0]}; travelling as the robot stands")
        return here

    def unfold_after_travel(self, targets) -> None:
        """Come back out of the travel fold at the new stance, checking the way out before taking it.

        Folding in is safe because it moves over the robot's own base. Coming out is the opposite -- the arms go
        into a room the robot has only just arrived in -- so the path is tested first with ``path_hits_scene`` and
        a warning says what is in the way. It unfolds regardless: the ramp stops the moment a joint falls behind
        its target, which is the collision-awareness that matters, and refusing to unfold was measured to be worse
        than unfolding carefully (see below).
        """
        if targets is None:
            return
        arm = self.arm if self.arm in self.robot.arm_names else self.robot.arm_names[0]
        try:
            ik = self._stance_ik(arm)
            names = self.robot.arm_joint_names[arm]
            now = dict(zip(self.planned_joints, [float(v) for v in self.q_arm()]))
            want = dict(zip(self.planned_joints, [float(v) for v in targets]))
            swept = self.path_hits_scene(
                arm, ik, [now.get(j, 0.0) for j in names], [want.get(j, 0.0) for j in names], mesh=False
            )
        except Exception:  # noqa: BLE001 - no description for this arm: unfold as before rather than stay folded
            swept = []
        if swept:
            # Say so, but still go. Refusing to unfold was measured to be worse than unfolding carefully: in
            # putting_away_toys the arms stayed folded over the base, which is INSIDE the toy box the robot had
            # come to work at, so the start-state lift then fired three times trying to get them out and every
            # plan was refused anyway -- 0.000 against a 0.75 baseline (2026-09-14). The ramp is already
            # collision-aware in the way that matters: it stops the moment a joint falls behind its target
            # instead of leaning on the obstacle. What this check is worth is the warning, and one day a choice
            # between paths; it is not worth staying folded for.
            log.warning(
                f"the way back to the working posture passes through {swept[0]}; unfolding anyway, and the ramp "
                "will stop if a joint meets it"
            )
        blocked = self.ramp_to(
            targets,
            self.posture,
            self.last_gripper,
            TRAVEL_SETTLE_STEPS,
            note="unfold after travel",
            max_vel=TRAVEL_MAX_JOINT_VEL,
        )
        if blocked is not None:
            log.warning(
                f"unfolding after the teleport was stopped by {blocked[0]}: the posture the round works from is "
                "not the one it asked for, and something is in the way of it here"
            )

    def place_robot(self, x: float, y: float, yaw: float, note: str = "", unfold: bool = True) -> dict:
        """Teleport the base to a floor pose (the navigation stand-in). OmniGibson moves an object held by the
        grasp assist along with the robot, so a carried object stays in the gripper.

        The arms come in over the base for the teleport (``fold_for_travel``) and go back out only if the way back
        is clear (``unfold_after_travel``). A teleport does not sweep -- it materialises the robot wherever it
        lands -- so the landing posture is the whole of the question: with the working posture the arms sit 0.41 m
        past the base's own rectangle and 0.095 m folded.

        The fold was taken out for a day because it cost 7967 of an episode's 16946 steps, and that was the wrong
        half to remove. The cost was in ramping it at the CAPTURE speed cap of 0.6 rad/s, which is there because
        observation swings were knocking objects about -- a motion out through a scene nobody has planned. Bringing
        the arms in over the robot's own base is the opposite motion, so it runs at its own speed and costs about a
        quarter as much. Coming back out IS a motion into the room, so it is collision-checked first (2026-09-14).
        """
        unfold_to = self.fold_for_travel()
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
        if unfold:  # an opening stance keeps the arms folded: reach_plan chooses the way out to the handle
            self.unfold_after_travel(unfold_to)
        log.info(f"robot placed at ({x:.2f}, {y:.2f}) yaw {math.degrees(yaw):.0f} deg {note}")
        self.log_teleport_contacts()
        return {"x": float(x), "y": float(y), "yaw": float(yaw)}

    def log_teleport_contacts(self) -> dict:
        """MEASUREMENT ONLY (dev/stance, 2026-09-14): what the robot is physically touching right after a teleport
        and its unfold, by robot link, read from the physics' contact matrix. Floors are left out (the wheels stand
        on them). Logs one line and changes nothing; a failure inside is logged and swallowed."""
        try:
            from omnigibson.utils.usd_utils import RigidContactAPI

            links = set(self.robot.link_prim_paths)
            found = {}
            for current_only in (True, False):
                pairs = RigidContactAPI.get_contact_pairs(
                    scene_idx=self.robot.scene.idx, query_set=links, with_set=None, current_only=current_only
                )
                for link, other in pairs:
                    if other in links:
                        continue
                    parts = other.split("/")  # /World/scene_0/<object>/<link>
                    name = parts[3] if len(parts) > 3 else other
                    obj = self.env.scene.object_registry("name", name)
                    if obj is not None and getattr(obj, "category", "") == "floors":
                        continue
                    found.setdefault(name, {"now": set(), "recent": set()})["now" if current_only else "recent"].add(
                        link.rsplit("/", 1)[-1]
                    )
        except Exception as e:  # noqa: BLE001 - a probe must never end a run
            log.info(f"teleport contact probe unavailable: {e!r}")
            return {}
        if not found:
            log.info("after the teleport the robot touches nothing but the floor [teleport contact probe]")
            return {}
        parts = []
        for name, by in sorted(found.items()):
            now = sorted(by["now"])
            recent = sorted(by["recent"] - by["now"])
            parts.append(f"{name}: now {now}" + (f", during the unfold {recent}" if recent else ""))
        log.warning(f"after the teleport the robot touches {'; '.join(parts)} [teleport contact probe]")
        return {name: sorted(by["now"] | by["recent"]) for name, by in found.items()}

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
    def ik_joint_names(self, arm: str, with_torso: bool = False) -> list[str]:
        """The joints an ``arm_ik`` solves for, in the simulator's own DOF order (torso first, then the arm).

        A caller that solves with the torso has to seed and read back the same list, so both come from here.
        """
        trunk = list(getattr(self.robot, "trunk_joint_names", []) or []) if with_torso else []
        return [j for j in trunk if j in self.urdf_joints] + list(self.robot.arm_joint_names[arm])

    def arm_ik(self, arm: str, frame: str | None = None, with_torso: bool = False) -> ArmIK:
        """Inverse kinematics for ``arm``'s joints with every other joint held where it is now, solving for
        ``frame`` (its wrist camera's link by default).

        ``with_torso``: solve for the four torso joints as well, 11 degrees of freedom instead of 7. The torso is
        NOT locked -- the embodiment plans all four of its joints and locks only the right arm and the fingers
        (r1pro_left_meta.yml), so the planner bends the torso whenever it needs to. Only this IK was holding it
        still, which is why every skill the bridge owns has the reach of a fixed torso: a drawer 22 cm off the
        floor and a handle at 1.2 m both came back "no orientation reaches its face" while the planner would have
        had no trouble. Off by default, because a caller that turns it on must seed and read back the longer
        joint list (``ik_joint_names``).
        """
        joints = self.ik_joint_names(arm, with_torso)
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
        """(min, max) corners of base_link's own box in the base frame, whatever the base's yaw.

        Measured from the link's mesh vertices rather than from its world AABB: re-bounding a world AABB in a
        turned frame inflates it, and this box was reported as 0.64 x 0.68 m square-on and 0.98 x 0.98 m at an
        angle in the same code (2026-09-13). It is not centred on the base frame -- the base reaches about 0.40 m
        behind the origin and 0.24 m ahead.
        """
        if self._base_box is None:
            link = self.robot.links["base_link"]
            pts = None
            mesh = self.link_trimesh_world(link)  # the link's own geometry, exact whatever the base's yaw
            if mesh is not None and len(mesh.vertices):
                unit = th.tensor([0.0, 0.0, 0.0, 1.0])
                pts = np.asarray(
                    [
                        self.to_base(th.tensor(v, dtype=th.float32), unit)[0].cpu().numpy()
                        for v in np.asarray(mesh.vertices, dtype=np.float64)
                    ],
                    dtype=np.float64,
                )
            if pts is None:  # no meshes: fall back to the world AABB's corners, which a turned base inflates
                lo, hi = link.aabb
                unit = th.tensor([0.0, 0.0, 0.0, 1.0])
                pts = np.asarray(
                    [
                        self.to_base(th.tensor([float(x), float(y), float(z)]), unit)[0].cpu().numpy()
                        for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])
                    ],
                    dtype=np.float64,
                )
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

    def arm_points(self, arm: str, ik: ArmIK, q, at=None) -> list[np.ndarray]:
        """World positions of ``arm``'s link origins at joints ``q``, shoulder to fingertips, in order.

        ``at``: an (x, y, yaw) base pose to evaluate them at instead of the base's current one, so a stance can be
        judged by where the arm would END UP before the robot is put there.
        """
        names = list(self.robot.arm_link_names[arm]) + [f"{arm}_{suffix}" for suffix in HAND_LINKS]
        return R1ProSim._link_points(self, ik, q, names, at)

    def container_body(self, obj, moving: str):
        """World mesh of ``obj``'s links other than ``moving``: the cabinet around the drawer being opened, the
        fridge around its door. Kept until the object moves (its box is the stamp), like ``scene_mesh``.

        Every check of the opening motion has to exempt the moving link -- the hand is meant to reach it and it
        travels with the hand -- but exempting the whole container with it left the cabinet's body out of every
        check, and the first reach that ``path_hits_scene`` called clear stopped 82 steps in with the elbow 0.17
        rad behind its target on the way to a drawer front (2026-09-14).
        """
        lo, hi = (v.cpu().numpy() for v in obj.aabb)
        key = f"{obj.name}#body-{moving}"
        stamp = (tuple(np.round(lo, 4)), tuple(np.round(hi, 4)))
        cached = self._scene_meshes.get(key)
        if cached is None or cached[0] != stamp:
            parts = [
                mesh
                for name, link in obj.links.items()
                if name != moving and (mesh := self.link_trimesh_world(link)) is not None and len(mesh.faces)
            ]
            self._scene_meshes[key] = (stamp, trimesh.util.concatenate(parts) if parts else None)
        return self._scene_meshes[key][1]

    def body_hits(self, arm: str, ik: ArmIK, q, body, at=None) -> bool:
        """Whether the arm at joints ``q`` reaches into ``body`` (a container's mesh less its moving link): the
        arm's own links as a polyline within ``ARM_RADIUS`` of it, the hand's link origins within
        ``HAND_BODY_CLEARANCE`` -- the hand is working at the container, so it is allowed close."""
        if body is None:
            return False
        limbs = self._link_points(ik, q, list(self.robot.arm_link_names[arm]), at)
        if len(limbs) >= 2 and bool(points_within_tol(body, sample_polyline(limbs, ARM_SAMPLE_STEP), ARM_RADIUS).any()):
            return True
        hand = self._link_points(ik, q, [f"{arm}_{suffix}" for suffix in HAND_LINKS], at)
        return bool(hand) and bool(points_within_tol(body, np.asarray(hand), HAND_BODY_CLEARANCE).any())

    def path_hits_body(self, arm: str, ik: ArmIK, q_from, q_to, body, samples: int = PATH_SAMPLES) -> bool:
        """``body_hits`` anywhere along the straight joint-space path, sampled like ``path_hits_scene``."""
        if body is None:
            return False
        q_from, q_to = np.asarray(q_from, dtype=np.float64), np.asarray(q_to, dtype=np.float64)
        return any(self.body_hits(arm, ik, q_from + t * (q_to - q_from), body) for t in np.linspace(0.0, 1.0, max(2, samples)))

    def _link_points(self, ik: ArmIK, q, names, at=None) -> list[np.ndarray]:
        """World positions of the named links' origins at joints ``q`` (``arm_points`` for any link list)."""
        points = []
        for name in names:
            try:
                pos, _ = ik.fk(q, name)
            except Exception:
                continue  # a link Lula's description does not carry
            p = np.asarray(pos, dtype=np.float64)
            if at is None:
                points.append(np.asarray(self.base_to_world(p), dtype=np.float64))
            else:
                x, y, yaw = at
                c, sn = math.cos(float(yaw)), math.sin(float(yaw))
                points.append(np.array([x + c * p[0] - sn * p[1], y + sn * p[0] + c * p[1], p[2]], dtype=np.float64))
        return points

    def arm_hits_scene(
        self, arm: str, ik: ArmIK, q, aabbs=None, clearance: float = ARM_RADIUS, at=None, mesh: bool = True
    ) -> list[str]:
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
        points = self.arm_points(arm, ik, q, at=at)
        near = [
            (obj, lo, hi) for obj, lo, hi in aabbs if obj not in held and polyline_hits_box(points, lo, hi, clearance)
        ]
        if not (ARM_MESH_CHECK and mesh and near):
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

    def nearby_obstacles(self, exclude=(), reach: float = OBSTACLE_REACH, limit: int = OBSTACLE_LIMIT) -> list[str]:
        """Tracked names of the furniture standing close enough to get in the way of a plan, nearest first.

        The planner's collision world holds the task's own objects and one fitted table plane, and nothing else in
        the room -- so cuTAMP plans straight through the furniture it was never told about, and the bridge has
        been second-guessing it afterwards. These labels go out in ``held_labels``, which cuTAMP takes as statics:
        obstacles it plans around and cannot pick up. That is a change to WHAT THE PLANNER IS TOLD rather than to
        how the bridge moves, which is the right place for it; the bridge has no business re-deciding a motion the
        planner already planned.

        Floors, ceilings and rugs are left out (they are stood on, not avoided), as are merged walls and roofs
        (``HOUSE_AABB_AREA``), anything wholly above the robot, and anything small enough to be a task object
        rather than a fixture. A hull is only built for a label the capture also masks, so the caller must add
        these to the labels it segments.

        WHAT THIS ACTUALLY DELIVERS, measured 2026-09-15 on putting_dirty_dishes_in_sink (instance 301, oracle,
        head/head_up/head_down, the 4 rounds that reached the planner): of 32 labels offered here, 17 carried
        pixels and were sent, 15 of those were dropped by the planner as "no hull in this frame", and TWO became
        obstacles in cuTAMP's world -- the same bench both times ("In the other hand (obstacles):
        ['bench_xwphjd_3']", 157813 and 284216 points). The drops happen in the planner's own reconstruction, for
        two reasons that have nothing to do with which labels this picks: hulls_from_points drops every label with
        no points above the fitted table top (z = 0.742 here), which is every bench, chair and seat in a diner
        below the table; and the perception crop (``workspace``, [[0.35, -0.8, 0.25], [1.3, 0.8, 1.6]] in the base
        frame) throws away the points of anything not directly in front of the robot, which is how walls and window
        blinds arrive with "only 0 valid depth points". So this channel can only ever hand over furniture that
        stands above the work surface inside the workspace box. Giving the planner the geometry the bridge already
        has needs a channel that is not perception -- a request key cuTAMP turns into statics directly -- and that
        is a change on the planner's side of the wire.
        """
        here = (
            self.base_pose()[0][:2].cpu().numpy()
            if hasattr(self.base_pose()[0], "cpu")
            else np.asarray(self.base_pose()[0][:2], dtype=np.float64)
        )
        # The task's own objects are never obstacles: the planner already has them, as movables it may pick up,
        # under their tiptop labels. ``exclude`` carries those labels ('booth_1'), which never match a scene name
        # ('booth_xzrpar_2'), so the same booth was offered twice -- once to pick up and once to plan around.
        # Sparing by object identity is what the caller meant; ``exclude`` still spares extra scene names.
        spared = set(exclude) | {self.robot.name}
        tracked = set(map(id, self.objects.values()))
        rows = []
        for obj, lo, hi in self.scene_aabbs():
            if obj is self.robot or obj.name in spared or id(obj) in tracked or obj.category in FLOOR_COVERINGS:
                continue
            if (hi[0] - lo[0]) * (hi[1] - lo[1]) > HOUSE_AABB_AREA:
                continue  # merged walls, roofs, ceilings
            if lo[2] > ROBOT_HEIGHT:
                continue  # entirely overhead
            extent = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
            if float(np.max(extent)) < OBSTACLE_MIN_SIZE:
                continue  # small enough to be something to pick up, not a fixture to plan around
            centre = (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64))[:2] / 2.0
            gap = float(np.linalg.norm(centre - here))
            if gap <= reach:
                rows.append((gap, obj.name, obj))
        rows.sort(key=lambda row: (row[0], row[1]))
        # Register them so the capture can mask them and ``object_meshes`` can build their geometry: a label the
        # oracle could not resolve to an object reached ``object_meshes`` and raised there, which killed every
        # round the flag was on. They stay out of ``self.objects`` (see TiptopSim.tracked_object) and are rebuilt
        # per call, because which furniture is near depends on where the base is standing.
        self.obstacles = {name: obj for _, name, obj in rows[:limit]}
        return list(self.obstacles)

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

    def path_hits_scene(
        self, arm: str, ik: ArmIK, q_from, q_to, aabbs=None, samples: int = PATH_SAMPLES, mesh: bool = True
    ) -> list[str]:
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
            for name in self.arm_hits_scene(arm, ik, q_from + t * (q_to - q_from), aabbs, mesh=mesh):
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

    def ramp_to(
        self, q_arm, posture: dict, gripper: float, settle_steps: int, note: str = "", max_vel: float | None = None
    ) -> tuple | None:
        """Move the planned joints to ``q_arm`` and the locked joints to ``posture`` together, every joint at no more
        than ``CAPTURE_MAX_JOINT_VEL``: one interpolated target per control step from where the joints are now, then
        ``settle_steps`` holding the targets, at ``max_vel`` rad/s (default ``CAPTURE_MAX_JOINT_VEL``: the cap that
        exists because observation swings were knocking objects about -- a motion through a scene nobody has
        planned. A motion that stays over the robot's own base, like folding for a teleport, is not that and can
        pass its own speed). ``self.posture`` follows the ramp and ends at ``posture``. ``note``
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
        speed = CAPTURE_MAX_JOINT_VEL if max_vel is None else float(max_vel)
        path = joint_ramp(start, goal, speed * self.dt)
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
            rates = np.abs(measured - last) / self.dt
            if rates.max() > fastest:
                j = int(rates.argmax())
                fastest = float(rates[j])
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
            f"joints ramped over {len(path)} steps at up to {speed} rad/s commanded, "
            f"{fastest:.2f} rad/s measured{culprit}, then {settle_steps} settle steps"
        )
        return blocked

    def look_at_point(self, hands: dict | None = None) -> np.ndarray:
        """The base-frame point a capture aims its cameras at.

        The gripper that holds what this round is about, once it has been picked up -- the held object is what the
        next plan has to see; a held object otherwise, when nothing was stood for; else what the base pose was
        chosen for; else straight ahead. ``hands``: a ``hands()`` snapshot to reuse.
        """
        hands = self.hands() if hands is None else hands
        held_arms = set(hands.values())
        holding_arms = sorted(hands[self.tracked_label(n)] for n in self.look_names if self.tracked_label(n) in hands)
        if holding_arms or (held_arms and self.look_target is None):
            return np.asarray(self.eef_pose_base((holding_arms or sorted(held_arms))[0])[:3, 3], dtype=np.float64)
        return np.asarray(DEFAULT_LOOK_TARGET if self.look_target is None else self.look_target, dtype=np.float64)

    def head_view_turn(self, name: str) -> tuple[str, float] | None:
        """The joint a head view turns and how far, or None when this view is not worth taking.

        The fixed views of ``HEAD_VIEWS`` always turn by their own constant. The aimed view works out for itself
        how far the torso has to turn to put ``look_at_point`` in the middle of the head camera's frame, from the
        camera's pose as it stands right now, and is skipped when the head is already pointing near enough at it
        (a ramp and a second render cost steps, and a turn of a few degrees buys nothing).
        """
        if name in HEAD_VIEWS:
            return HEAD_VIEWS[name]
        if name != HEAD_AIM_VIEW:
            raise ValueError(f"{name!r} is not a head view ({sorted(set(HEAD_VIEWS) | set(TURNED_HEAD_VIEWS))})")
        _, base_from_cam, _ = self.head_camera_in_base()
        target = self.look_at_point()
        delta = head_aim_yaw(base_from_cam[:3, 3], base_from_cam[:3, 2], target, HEAD_AIM_LIMIT)
        if abs(delta) < HEAD_AIM_MIN:
            log.info(
                f"head aim: {np.round(target, 2).tolist()} is already {math.degrees(delta):+.0f} deg off the head "
                f"camera's axis; no turn taken"
            )
            return None
        log.info(
            f"head aim: turning {HEAD_YAW_JOINT} {math.degrees(delta):+.0f} deg to look at "
            f"{np.round(target, 2).tolist()}"
        )
        return HEAD_YAW_JOINT, delta

    def capture(self, task: str) -> tuple[dict, dict]:
        """Every view in one posture: each free arm whose wrist camera is a view points it at the look target
        (``wrist_look``: what the base pose was chosen for, at the hand that holds it once it has been picked up;
        a held object otherwise, when nothing was stood for), which also takes the arm out of the head camera's
        frame. An arm that holds something stays where it is:
        the held object is what the next plan is about and must be seen, and the gripper keeps its command. The
        planned arm swings out of view (``look_arm``) when no look configuration exists; with ``look_arm`` None
        nothing moves. The head views come last, the torso turned to their yaw with the arms as they are
        (``_capture_views``). The plan starts from the ready posture the arms return to."""
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
        target = self.look_at_point(hands)
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
        # The plan starts here, not at the look posture -- and not inside the furniture either: a start state
        # in collision is refused before the goal is considered (``clear_start_posture``).
        q_ready = self.clear_start_posture(self.arm, q_ready)
        request["q_init"] = np.asarray(q_ready, dtype=np.float32)
        extras["q_look"] = moved
        log.info(
            f"captured with {sorted(moved)} arm(s) posed; plan starts from the ready posture (max error {lag:.4f} rad)"
        )
        return request, extras

    def press_grasp(self, arm: str, name: str) -> bool:
        """Take hold of a flat object by pressing the open hand onto it and closing; whether the assist holds it.

        M2T2 proposes grasps from the point cloud, and a book lying flat on a table gives it nothing usable --
        there is no side a parallel jaw can get under. The round then fails before the arm moves, which reads as a
        planning failure rather than as "this cannot be grasped that way". The user's instruction (2026-09-14):
        close the gripper on the book and let the assisted grasp take it.

        This is the drawer panel's grasp turned upwards: come straight down, put the fingertips on the top face,
        and press in a step at a time until the assist reports it holds (``close_on``). It solves with the torso,
        since a book on a low shelf is outside a fixed-torso workspace, and it refuses a path that sweeps the
        furniture rather than discovering it by collision.

        Returns False without moving when the object is not flat -- this is a fallback for the shape M2T2 cannot
        serve, not a replacement for it.
        """
        obj = self.scene_object(name)
        lo, hi = (v.cpu().numpy().astype(np.float64) for v in obj.aabb)
        extent = hi - lo
        if float(np.min(extent)) > FLAT_THICKNESS:
            log.info(f"{name} is {np.round(extent, 3).tolist()} m: not flat enough to need the pressed grasp")
            return False
        top = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, float(hi[2])], dtype=np.float64)
        unit = th.tensor([0.0, 0.0, 0.0, 1.0])
        top_base = self.to_base(th.tensor(top, dtype=th.float32), unit)[0].cpu().numpy()
        ik = self.arm_ik(arm, frame=f"{arm}_gripper_link", with_torso=True)
        joints_of = self.ik_joint_names(arm, with_torso=True)
        q = self.robot.get_joint_positions()
        seed = [float(q[self.joint_index[j]]) for j in joints_of]
        aabbs = self.scene_aabbs()
        down = np.array([0.0, 0.0, -1.0])
        # The jaw closes across the object's narrower horizontal axis, so the fingers meet over it rather than
        # along it; a book is gripped across its width, not its length.
        across = np.array([1.0, 0.0, 0.0]) if extent[0] <= extent[1] else np.array([0.0, 1.0, 0.0])
        for jaw in (across, np.array([across[1], across[0], 0.0])):
            where, rot = self.grasp_target(arm, top_base, down, jaw)
            quat = T.mat2quat(th.tensor(rot, dtype=th.float32)).cpu().numpy()
            solution = None
            for tolerance in (OPEN_GRASP_TOLERANCE, 0.02):
                solution = ik.solve(where, quat, seed=seed, tolerance_pos=tolerance, tolerance_rad=0.5)
                if solution is not None:
                    break
            if solution is None:
                continue
            swept = [n for n in self.path_hits_scene(arm, ik, seed, solution, aabbs=aabbs, mesh=False) if n != obj.name]
            if swept:
                log.info(f"{name}: the way down to it sweeps through {swept[0]}; trying the other jaw direction")
                continue
            log.info(f"pressing the hand onto {name} at {np.round(top, 3).tolist()} to take hold of it")
            targets = [float(v) for v in self.q_arm()]
            for joint_name, value in zip(joints_of, solution):
                if joint_name in self.planned_joints:
                    targets[self.planned_joints.index(joint_name)] = float(value)
            self.ramp_to(targets, self.posture, self.OPEN, OPEN_SETTLE_STEPS, note=f"down onto {name}")
            pose = pose_matrix(where, quat)
            _, held = self.close_on(
                arm, ik, obj, obj.root_link_name, pose, down, [float(v) for v in solution], joints_of
            )
            if held:
                return True
        log.info(f"{name}: the pressed grasp found no way onto it")
        return False

    def clear_start_posture(self, arm: str, q_ready):
        """Lift ``arm`` out of whatever it is resting in before a plan is asked for from there; the posture to send.

        The plan starts at ``q_init``, and cuRobo refuses a start state that is in collision -- without ever
        looking at the goal. Across the planner's own saved logs that refusal, INVALID_START_STATE_WORLD_COLLISION,
        appears 6657 times against 1355 IK_FAIL: start states in collision outnumber unreachable goals five to one.
        Each one costs the whole round, and it costs it 32 times over, because the verdict does not depend on which
        grasp particle is being tried, so every refinement attempt returns the same message.

        The cause is a height coincidence rather than a broken perception. With the challenge torso posture the
        elbow and forearm sit at roughly counter height, and the planner's support is a thin slab whose top is
        sunk 2 cm under the surface it detected, with no activation distance -- so at a counter or a cabinet the
        margin between "fine" and "the whole plan is refused" is millimetres, while at a floor or a low table it
        is tens of centimetres. This is the same failure the workspace crop already guards for the BASE
        (``WORKSPACE_NEAR``, "a support cuboid reaching there puts the robot's start posture in collision"); the
        reasoning was never extended to the arm.

        Lifting is the right direction: it moves away from the surface, so it is the motion least likely to be
        blocked in turn. A posture that is already clear is returned untouched, which is every floor and low-table
        task, so this costs nothing where nothing is wrong.
        """
        if arm not in self.robot.arm_names:
            return q_ready
        try:
            ik = self._stance_ik(arm)
        except Exception:  # noqa: BLE001 - no description for this arm: send the posture as it is
            return q_ready
        aabbs = self.scene_aabbs()
        held = {self.objects[label] for label in self.hands() if label in self.objects}
        names = [n for n in self.planned_joints if n.startswith(f"{arm}_arm_joint")]
        if not names:
            return q_ready

        def arm_q(q):
            by_name = dict(zip(self.planned_joints, [float(v) for v in q]))
            return [by_name.get(j, 0.0) for j in self.robot.arm_joint_names[arm]]

        def hits(q):
            return [
                n
                for n in self.arm_hits_scene(arm, ik, arm_q(q), aabbs=aabbs, mesh=False)
                if n not in {o.name for o in held}
            ]

        inside = hits(q_ready)
        if not inside:
            return q_ready
        index = {n: self.planned_joints.index(n) for n in names}
        shoulder, elbow = f"{arm}_arm_joint2", f"{arm}_arm_joint4"
        best = None
        for lift in START_LIFTS:
            for joints in ((shoulder,), (elbow,), (shoulder, elbow)):
                if any(j not in index for j in joints):
                    continue
                for sign in (-1.0, 1.0):
                    q = [float(v) for v in q_ready]
                    for j in joints:
                        q[index[j]] = float(q[index[j]]) + sign * lift
                    if hits(q):
                        continue
                    best = (q, joints, sign * lift)
                    break
                if best:
                    break
            if best:
                break
        if best is None:
            log.warning(
                f"the {arm} arm starts inside {inside[0]} and no lift up to {max(START_LIFTS):.2f} rad frees it; "
                "asking for the plan from here, which the planner will probably refuse"
            )
            return q_ready
        q, joints, delta = best
        log.info(
            f"the {arm} arm starts inside {inside[0]}, which the planner refuses before it looks at the goal; "
            f"lifting {'+'.join(joints)} by {delta:+.2f} rad to start clear"
        )
        self.ramp_to(q, self.posture, self.last_gripper, LOOK_SETTLE_STEPS, note="lift clear before planning")
        return self.q_arm()

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
        planned joints' targets, at torso yaw 0), then each head view among ``extra_views`` -- the fixed turns of
        ``HEAD_VIEWS`` and the aimed one of ``TURNED_HEAD_VIEWS`` -- with the torso ramped to the yaw
        ``head_view_turn`` gives it (``ramp_to``; every other joint stays where the capture found it, so a head
        view moves the torso and nothing else), and the torso ramped back. The base frame does not turn with the torso, so a view's
        camera pose, read from the simulator as it is rendered, is right as it is."""
        self.log_blocked_sight()
        head_views = [v for v in self.extra_views if v in HEAD_VIEWS or v in TURNED_HEAD_VIEWS]
        extra_views = self.extra_views
        self.extra_views = tuple(v for v in extra_views if v not in head_views)
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
        # Each view's turn is settled before any of them is taken, so a view that turns out not to be worth taking
        # (the aimed view when the head already points at the target) never reaches the ramp-back bookkeeping below.
        turns = [(name, turn) for name in head_views if (turn := self.head_view_turn(name)) is not None]
        head_views = [name for name, _ in turns]
        if not head_views:
            return request, extras
        for name, (joint, delta) in turns:
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
        for _, (joint, _delta) in turns:
            if joint in self.planned_joints:
                back_to[self.planned_joints.index(joint)] = q_arm[self.planned_joints.index(joint)]
        self.ramp_to(back_to, self.posture, self.last_gripper, HEAD_VIEW_SETTLE_STEPS, note="back from a head view")
        moved_joints = {joint for _, (joint, _delta) in turns}
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
