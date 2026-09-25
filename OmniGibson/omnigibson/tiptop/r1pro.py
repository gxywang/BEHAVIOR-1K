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

import itertools
import logging
import math
import re
from pathlib import Path

import trimesh
import numpy as np
import torch as th
import yaml
from bddl.condition_evaluation import HEAD

from b1k.bridge.articulation import (
    OPEN_FRACTION_SCORED,
    follow_joint,
    handle_on,
    is_open,
    leading_direction,
    opening_travel,
    pose_matrix,
)
from b1k.bridge.executor import leash
from b1k.bridge.kinematics import link_from_camera, link_pose_for_camera, look_pose
from b1k.bridge.geometry import (
    FOOTPRINT_CELL,
    HEAD_AIM_LIMIT,
    ROBOT_FOOTPRINT,
    ROBOT_HEIGHT,
    HeadCamera,
    best_base_pose as search_base_poses,
    blocks_ray,
    corner_off_floor,
    footprint_blockers,
    footprint_cells,
    footprint_corners,
    head_aim_yaw,
    polyline_hits_box,
    rect_box_gap,
    sample_polyline,
    segment_hits_box,
    travel_fold_targets,
    turned_joints,
)

# Re-exported and not used here: the stance search took its constants and the rest of its geometry with it,
# and test_tiptop_kinematics.py, the scripts and the unmerged dev/* branches read them out of this module.
from b1k.bridge.geometry import (  # noqa: F401
    AVOID_RADIUS,
    CLEAR_WEIGHT,
    FLOOR_COVERINGS,
    FOOTPRINT_FACE_CELLS,
    FRAMING_PENALTY,
    FRAMING_PENALTY_PX,
    FRAME_MARGIN_PX,
    HOUSE_AABB_AREA,
    HIDE_DEPTH,
    HIDE_MARGIN,
    MIN_AHEAD,
    MIN_SIDE,
    REACH_WIDEN,
    RING_ANGLE_STEP,
    RING_START,
    RING_STEP,
    SIDE_TARGET,
    SIDE_WEIGHT,
    STANCE_CLEARANCE,
    TARGET_HALF_WIDTH,
    TRAVEL_POSE,
    YAW_OFFSETS,
    YAW_WEIGHT,
    box_corners,
    frame_objects,
    rect_hits_box,
    widen_then_clip,
)
from b1k.bridge.protocol import (
    FLOOR_CATEGORIES,
    INTENT_PREDICATES,
    PLANNER_SUPPORT,
    SUPPORT_CATEGORIES,
    add_view,
    bddl_category,
    bddl_label,
    button_label,
    detector_phrase,
    face_normal_local,
    joint_ramp,
    label_category,
    points_to_pixels,
    reach_candidates,
    tiptop_goal as translate_goal,
    via_configuration,
)

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects.usd_object import USDObject
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.tiptop.articulation import openable_joints
from omnigibson.tiptop.gt_masks import masks_from_geometry, points_within_tol
from omnigibson.tiptop.kinematics import ArmIK
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
# DO NOT USE THIS VIEW YET -- measured broken on 2026-09-15, in picking_up_toys 301 (runs/vis_toys_aim).
# The turn itself is right: the log says torso_joint4 moved +34 deg to look at [0.64, 0.64, 0.79], which is the
# object, and the camera came back 4 cm from where it started, which is what a yaw of a camera 9 cm off its axis
# should do. What comes back is a BLACK frame whose depth is 3 mm to 35 mm across all 720x720 pixels: at this
# torso posture the turn buries the head camera in geometry. Two black frames then satisfy the capture's
# convergence test after 4 renders (a real head view takes 20), every mask in the view is empty, and -- worst --
# the view is NOT dropped by the "nothing but the robot in this view" guard, because a uniform 3 cm of depth
# counts as valid. So the planner is handed a view of a wall 3 cm away. It is opt-in (`--views ... head_aim`) and
# off by default, so nothing uses it unless asked; leave it that way until the collision is understood.
#
# Also true, and separate: a turned head view's masks are wrong for anything the robot is HOLDING (the torso
# moves, the held object moves with it, the mesh does not) -- the per-view pose bug, which is someone else's.
HEAD_AIM_VIEW = "head_aim"
HEAD_AIM_MIN = 0.12  # rad: a smaller turn is not worth a ramp and a second render
TURNED_HEAD_VIEWS = (HEAD_AIM_VIEW,)  # head views beyond HEAD_VIEWS: same camera, same optics, computed turn
HEAD_VIEW_SETTLE_STEPS = 30  # after a yaw ramp: only the torso moved (the arms settle for LOOK_SETTLE_STEPS)
# The external capture sensors, one per optics: moved onto the robot camera's pose per view (_capture_obs). The
# robot's own cameras render rgb only (video, mirror); depth and segmentation come from these.
SHADOW_CAMS = {"head": CAMERA_NAME, "wrist": "tiptop_wrist_cam"}
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
CAMERA_MIN_MARGIN = 0.08  # added to where the bottom image edge meets an object's support: room to be whole.
# Raised to 0.15 m on 2026-09-12 on the theory that a battery kept coming out of the capture with an empty mask
# because it sat just past the frame's bottom edge, and put back: the wider margin moved the stance from 0.55 m
# to 0.60 m and the mask was still empty (runs/bench_batteries_6 against _5), so the object is hidden at those
# stances for another reason and the tighter margin only costs reach. What actually recovers the round is the
# retry from a different pose, which finds the battery every time (493 mask pixels at the stance that works).

# _footprint_free: what an AABB in the footprint means
# Furniture sent to the planner as static obstacles (see nearby_obstacles). Close enough to matter, big enough
# to be a fixture, and capped because every one of them costs a mask in the capture and a hull in perception.
OBSTACLE_REACH = 2.5  # m from the base; beyond this the arm cannot reach it anyway
OBSTACLE_LIMIT = 8
OBSTACLE_MIN_SIZE = 0.30  # m on its longest axis
FILLABLE_META_LINKS = ("fillable", "openfillable")  # what OmniGibson's Inside state needs to be satisfiable at all
INSIDE_MIN_SIDE = 0.02  # m: below 2 * cuTAMP's placement shrink the OBB raises and kills the whole round
SHELF_CASE_HEIGHT = 0.4  # m: a joint-less fillable volume taller than this is a case of shelves, not one bin
BOARD_MIN_AREA = 0.02  # m2 of level faces at one height that make a board something can rest on
BOARD_GAP = 0.025  # m between face heights that separates two boards (a convex piece's top is not quite flat)
PLACE_HEIGHT = 0.9  # m: the board nearest this is the one a standing robot places on comfortably (0.7-1.1)
PLACE_HEIGHT_MAX = 1.5  # m: a board above this is out of the arm's reach from any stance (a hallstand's 2.37 m top)
HEADROOM = 0.03  # m the item needs under the next board
BOARD_REACH = 1.0  # m from the base to the nearest point of a board for it to be placed on from here
STANDS_ON_TOL = 0.05  # m: a body whose top is within this of a task object's underside is that object's support
HAND_STACK = 0.21  # m above the fingertips the wrist's link7 sphere tops out (enclosed/hand_stack.out): a top-down
# hand needs this much clear above what it takes; a shelf or a cavity roof nearer than that means a side grasp
# What the base can drive over is decided by the base's own underside (see _footprint_free), not by a guess at
# how thin a thing is: the 8 cm rule that used to live here exempted the toys a task has to pick up.
# place_robot: the overview camera in the base's frame, (eye dx, eye dy, eye z, target dx, target z); "shoulder" looks
# over the left shoulder at the workspace, "front" looks back at the chest so both hands and what they hold are in view
# Regions of the placements made for their effect (inside_regions: stamp, cut, heat, aim, attached)
STAMP_MARGIN = 0.02  # m an adjacency remover reaches beyond its link's box (PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN)
VACUUM_HOVER = 0.012  # m a projection remover rests above the particles: its slab hangs 1-21 mm under it
STAMP_YAW_TOL = math.radians(5)  # the tool is set down as it is held: its footprint was fitted at that yaw
HEAT_TOP_ABOVE = 0.05  # m above the heat link the source's cooking surface may stand (a grate over a burner)
AIM_YAW_TOL = math.radians(15)
ATTACH_TOL = 0.05  # m the male meta link may be off the female one for AttachedTo to snap (attached_to.py:34)
ATTACH_LIFT = 0.01  # m above the aligned height the child is set: at or above, never below (spec 6.2 Attach)
ATTACH_YAW_TOL = math.radians(10)  # the snap allows 15 deg
# N-push (spec S05): a flat item under a shelf board is slid out until this much of it hangs past the board's edge,
# for a pinch to close on; the push is a press (cuTAMP Push) on a 1 cm "button" at its far face
PUSH_OVERHANG = 0.03
PUSH_RADIUS = 0.01
# N-rotate (spec S38): a pour is the last wrist joint turned this far from a top-down rim grasp, held, and back
POUR_TILT = math.radians(90)  # 80-110 deg spills a container; joint7 alone has the range
POUR_HOLD_STEPS = 60  # env steps the tilt is held for the contents to fall (2 s at 30 Hz)
# E-level (spec S22): a carrier with passengers (a plate under a pizza) may tip at most this far in the bridge's own
# held motions; rigid passengers slide at ~27 deg (default friction, inferred)
LEVEL_TILT = math.radians(20)
OVERVIEW_OFFSETS = {"shoulder": (-1.5, 1.1, 1.7, 0.7, 0.55), "front": (1.15, -0.75, 1.35, 0.3, 0.9)}
LOCKED_DRIFT = 0.015  # rad a locked joint may sit off the planner's model before it is driven back
FALL_DROP = 0.05  # m below the teleport height: the base is falling, not settling
TILT_LIMIT_DEG = 1.0  # a base that settles further off level than this is fighting something it was put in
SHIFT_LIMIT = 0.02  # m it may slide while settling before the same is true
# The robot DOES fold before a teleport: place_robot calls fold_for_travel, and the way back out is
# collision-checked (0dfdc9955). An earlier comment here said it no longer did, which was true only for the day
# between 39023d80f removing the fold and 0dfdc9955 restoring it -- do not act on that reading. Measured on
# 2026-09-13 by ramping to each candidate and reading back the overhang past the base's own rectangle
# (x -0.39..0.24, y -0.34..0.34) and whether the ramp finished: the working posture leaves 41.2 cm; a torso fold
# to -2.25 leaves 29.2 cm and never arrives; every planned arm joint to zero leaves 9.5 cm and does arrive. That
# last one was adopted. It was then measured at 7967 of an episode's 16946 steps and removed for a day, but the
# cost was ramping it at the CAPTURE cap of 0.6 rad/s; folding in over the robot's own base is the opposite
# motion, runs at travel speed and costs about a quarter as much. See place_robot for the full write-up.
FOLD_OVERHANG = 0.07  # m the folded upper body still reaches beyond the base: what the teleport actually lands as
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
DOOR_TRAVEL_MAX = math.pi / 2  # rad a side-hinged door is pulled at most: the arc beyond is outside any fixed stance
LID_FRACTION = 0.85  # of the range a lid (horizontal hinge) is opened at least: past every balance angle measured
# (car 78%, toolbox / bins / jar under 50%, close/lid_geometry.out); short of it the lid falls shut
JOINT_TOL = 0.05  # of the range: OmniGibson's own open threshold, and how close a push must bring the joint
EDGE_INSET = 0.015  # m below a panel's top edge the fingertips pinch it (the "edge" grip)
EDGE_THICKNESS = 0.01  # ponytail: a panel is assumed 2 cm thick; the jaw is centred 1 cm behind its face
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
# Free-object sticky grasps close BEFORE contact; only the final slow approach may touch the target.
# A teleport materialises the robot: nothing backs it off an obstacle it lands a few millimetres inside, and its
# physics colliders are not its collision spheres. book_6's stance left 5 cm to a wall, passed the zero-margin
# check, and PhysX flipped the robot onto its back (base 179 deg off level, boxing_books 2026-09-22).
TELEPORT_CLEARANCE = 0.03
UNFOLD_FRACTIONS = (1.0, 0.75, 0.5, 0.25)  # of the straight unfold after a teleport: the first that stays clear
UNFOLD_MIN = 0.5  # a stance is taken at once if the arm unfolds at least this far there (place_robot_for)
AVOID_YAW = np.pi / 6  # rad: a stance the landing check refused rules out its spot at headings this close only
GROUND_TOP = 0.05  # m: an object whose top is below this is ground (floor, paver, lawn, rug), no TELEPORT_CLEARANCE
STICKY_LIFT = 0.05  # m a sticky-taken object is lifted back out along its approach (off a table)
STICKY_LIFT_FRONT = 0.10  # ... and pulled back out of a shelf
STICKY_LIFT_VEL = 0.5  # rad/s: slow, with an object in the hand
STICKY_STANDOFF = 0.02  # m of free space in front of the physical contact surface
STICKY_CONTACT_STEP = 0.002  # m between Cartesian contact-seeking waypoints
STICKY_CONTACT_SPEED = 0.01  # m/s requested Cartesian advance, also capped at 0.1 rad/s per joint
# Places to try taking hold of one panel, spread up its face. A fridge door is 2 m tall and only a band of it
# is in the arm's reach, so one point per joint is one guess; these are offered nearest the hand's own height
# first, which is both the likeliest to solve and the least the arm has to travel.
# An object thinner than this on its smallest axis is flat: M2T2 has no side to propose a grasp on, so the
# hand is pressed onto its top face instead (press_grasp). A hardback is about 3 cm, a board game about 5.
FLAT_THICKNESS = 0.06
GRASP_COLUMN_SAMPLES = 5
GRASP_COLUMN_INSET = 0.08  # m kept clear of the panel's top and bottom edges, so the jaw lands on the face
BASE_MASS_KG = 250.0  # omnigibson/eval/evaluator.py sets this for r1/r1pro; keeps the robot upright


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


def boards(vertices, faces) -> list[tuple]:
    """The horizontal boards of a mesh (z up): (top z, xy lo, xy hi, ceiling z) per group of upward faces at one
    height with at least BOARD_MIN_AREA, lowest first. The ceiling is the lowest group of DOWNWARD faces over it
    (the underside of the next board), inf when there is none."""
    v, f = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) / 2.0
    up = n[:, 2] / np.maximum(2.0 * area, 1e-12)
    z = (a[:, 2] + b[:, 2] + c[:, 2]) / 3.0

    def levels(facing, pick):
        order = np.flatnonzero(facing)[np.argsort(z[facing])]
        out = []
        for group in np.split(order, np.flatnonzero(np.diff(z[order]) > BOARD_GAP) + 1):
            if len(group) and area[group].sum() >= BOARD_MIN_AREA:
                pts = v[f[group]].reshape(-1, 3)
                out.append((float(pick(z[group])), pts[:, :2].min(0), pts[:, :2].max(0)))
        return out

    undersides = levels(up < -0.95, np.min)
    result = []
    for zt, lo, hi in levels(up > 0.95, np.max):
        over = [
            zu for zu, ulo, uhi in undersides if zu > zt + 0.01 and (np.minimum(hi, uhi) > np.maximum(lo, ulo)).all()
        ]
        result.append((zt, lo, hi, min(over, default=float("inf"))))
    return result


def held_tilt(quat_from, quat_to) -> float:
    """How far (rad) a load held level at gripper orientation ``quat_from`` (xyzw) tips when the gripper turns to
    ``quat_to``: the angle its up axis leaves the vertical by. A turn about the vertical tips nothing."""
    a, b = (T.quat2mat(th.as_tensor(np.asarray(q, dtype=np.float32))).numpy().astype(np.float64) for q in (quat_from, quat_to))
    up = b @ (a.T @ np.array([0.0, 0.0, 1.0]))
    return float(math.acos(float(np.clip(up[2], -1.0, 1.0))))


def roof_over(mesh, centre_xy, half_xy, z_lo: float, z_hi: float) -> bool:
    """Whether ``mesh`` has geometry over the rectangle between heights ``z_lo`` and ``z_hi``: rays up from a 3 x 3
    grid over the rectangle's inner half, any hit within the band."""
    if mesh is None or not len(mesh.faces) or z_hi <= z_lo:
        return False
    c, h = np.asarray(centre_xy, dtype=np.float64), np.asarray(half_xy, dtype=np.float64)
    origins = np.array([[c[0] + sx * h[0] / 2, c[1] + sy * h[1] / 2, z_lo + 1e-3] for sx in (-1, 0, 1) for sy in (-1, 0, 1)])
    hits, _, _ = mesh.ray.intersects_location(origins, np.tile([0.0, 0.0, 1.0], (len(origins), 1)))
    return bool(len(hits)) and bool((np.asarray(hits, dtype=np.float64)[:, 2] <= z_hi).any())


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


class BasePlacementCollision(RuntimeError):
    """The measured robot or carried volume occupies physical geometry at a requested base destination, or (with
    ``unfold`` set) the landing is clear but the arm unfolds only that fraction of the way there."""

    def __init__(self, message: str, unfold: float | None = None, obstacle: str | None = None):
        super().__init__(message)
        self.unfold = unfold
        self.obstacle = obstacle  # what the landing posture met, for the stance search to clear from then on


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
        self.send_room = False  # privileged physical collision map, independent of masks and viewer (--room)
        self.send_inside = False  # plan inside() onto the compartment floor (--inside-region)
        self.level = set()  # labels carried level (a plate under a pizza; OracleKnowledge.describe): held ramps refuse a tilt
        self.overview = self.env.external_sensors.get(OVERVIEW_CAM)
        self.objects = {}
        self.context = {}  # furniture shown in the Rerun mirror (track_context)
        self.obstacles = {}  # furniture sent to the planner as a static obstacle this stance (nearby_obstacles)
        self.bddl_names = {}  # tiptop label -> BDDL instance name for tracked task objects
        self.posture = {}
        self.q_home = None
        self.stance_ready = None  # the posture a partway unfold left the arm in at this stance; captures return to it
        self.blocked_swings = 0  # capture swings stopped against something this instance (see capture)
        self._scene_meshes = {}  # object name -> (box stamp, world trimesh) for the arm's collision check
        self._held_boxes = {}  # (arm, label) -> the held object's corners in the gripper frame
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
        """BDDL instance name -> simulated object for the loaded BehaviorTask (no agent, no floors or lawns, no
        systems)."""
        task = self.env.task
        scope = task.object_scope if isinstance(task, BehaviorTask) else {}  # a DummyTask has no scope
        return {
            k: v
            for k, v in scope.items()
            if isinstance(v, USDObject) and bddl_category(k) not in ("agent", *FLOOR_CATEGORIES)  # systems have no pose
        }

    def floor_name(self) -> str:
        """The BDDL name of the loaded task's floor, or its lawn (``task_scope`` leaves both out; a strategy puts
        things down on it when nothing else will do)."""
        names = [name for name in self.env.task.object_scope if bddl_category(name) in FLOOR_CATEGORIES]
        if not names:
            raise KeyError(f"the task {self.config['task'].get('activity_name')!r} has no floor in its scope")
        return names[0]

    def track_task_objects(self, skip_categories=("table", *FLOOR_CATEGORIES, "agent")) -> dict:
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
        """``protocol.tiptop_goal`` over the BDDL names this sim tracks; the translation itself is the wire's."""
        return translate_goal(atoms, self.bddl_names, category_level)

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


    def button_world(self, bddl: str) -> tuple[np.ndarray, np.ndarray, float]:
        """A toggle button as the simulator knows it (privileged; reached only through OracleKnowledge's
        button_hints): world position ON the body's surface, the outward unit normal of the body face it sits on,
        and the radius within which ToggledOn counts a finger.

        The face is read off the physical mesh: the visual one carries the toggle marker itself, a sphere up to
        46 mm wide that pulled the box face to the wrong side of every wall switch and put the lighter's button on
        its underside (verify_press/marker_in_box.out, 2026-09-24). Of the faces the marker's sphere reaches (half
        the width ToggledOn's overlap sphere is given) the most upward wins, since a press down on it is braced
        by the support; else the nearest. The position is where a ray along the normal meets the physical
        surface, so the stroke's push_depth counts from the surface: the washer's centre floats 11.2 mm proud of
        it and the 5 mm stroke stopped 6.2 mm short (press_verify/washer_gap.out); the box face is no substitute,
        a lamp's or a microwave's stands 2-21 cm in front of the button (manip_atoms/T1-bridge/button_faces.out)."""
        from omnigibson.object_states import ToggledOn

        obj = self.scene_object(bddl)
        if ToggledOn not in obj.states:
            raise ValueError(f"{bddl} has no toggle button (no ToggledOn state)")
        mesh = self.collision_mesh_world(obj)
        if mesh is None:
            raise ValueError(f"{bddl} has no physical mesh to find its button's face on")
        state = obj.states[ToggledOn]
        pos_w = state.link.get_position_orientation()[0].cpu().numpy().astype(np.float64)
        obj_pos, obj_quat = obj.get_position_orientation()
        obj_pos = obj_pos.cpu().numpy().astype(np.float64)
        rot = T.quat2mat(obj_quat).cpu().numpy().astype(np.float64)
        vertices = (np.asarray(mesh.vertices, dtype=np.float64) - obj_pos) @ rot  # the object's own frame
        radius = float(th.min(state.visual_marker.extent * state.scale * state.link.scale))
        up = rot.T @ np.array([0.0, 0.0, 1.0])  # world up in the object's frame
        n_local = face_normal_local(vertices, rot.T @ (pos_w - obj_pos), within=radius / 2, up=up)
        n_world = rot @ n_local
        hits, _, _ = mesh.ray.intersects_location([pos_w + 0.05 * n_world], [-n_world], multiple_hits=True)
        if len(hits):
            pos_w = hits[np.argmin(np.linalg.norm(hits - (pos_w + 0.05 * n_world), axis=1))]
        else:
            log.warning(f"{bddl}: no physical surface under its button along {np.round(n_world, 2).tolist()}")
        return pos_w, n_world, radius

    def button_hints(self, atoms: list[dict], category_level: bool = False) -> dict:
        """The toggle button of every toggled_on goal object, for the request's gt_buttons (privileged): base-frame
        position, outward normal of the face it sits on, and the radius within which ToggledOn counts a finger."""
        out = {}
        for atom in atoms:
            # A goal asking for a switch to be OFF reaches here as not(toggled_on, x) -- a HEAD's flat tokens,
            # because bddl compiles the negation into the ground atom. It is still a press, and it still needs its
            # button described. strategies.press_targets has read both forms since it was written; this read only
            # the positive one, so turning_out_all_lights_before_sleep -- whose goal is five `not toggled_on`
            # atoms and nothing else -- produced ZERO button hints and every round died on "goal objects
            # ['switch_1_button'] were not found in any of the 3 view(s)". The button is named by pose, never
            # segmented, so no hint means no press (2026-09-15).
            if atom["predicate"] in ("toggled_on", "push") and atom["args"]:
                bddl = atom["args"][0]
            elif atom["predicate"] == "not" and atom["args"][:1] == ["toggled_on"] and len(atom["args"]) > 1:
                bddl = atom["args"][1]
            else:
                continue
            # a push (N-push) is a press on the item's own face, with the stroke's depth: from its points, not a mesh
            hint = self.push_face(bddl) if atom["predicate"] == "push" else (*self.button_world(bddl), None)
            if hint is None:
                continue
            pos_w, n_world, radius, depth = hint
            identity = th.tensor([0.0, 0.0, 0.0, 1.0])
            pos_b, _ = self.to_base(th.tensor(pos_w, dtype=th.float32), identity)
            tip_b, _ = self.to_base(th.tensor(pos_w + n_world, dtype=th.float32), identity)
            label = label_category(bddl) if category_level else self.label_of(bddl)
            out[button_label(label)] = {
                "position": [float(v) for v in pos_b],
                "normal": [float(v) for v in (tip_b - pos_b)],
                "radius": radius,
                **({"depth": float(depth)} if depth is not None else {}),
            }
            log.info(
                f"button of {bddl}: {button_label(label)} at {np.round(out[button_label(label)]['position'], 3).tolist()} "
                f"(base), normal {np.round(out[button_label(label)]['normal'], 2).tolist()}, "
                f"radius {out[button_label(label)]['radius']:.3f} m"
                + (f", pushed {depth:.3f} m past it" if depth is not None else "")
            )
        return out

    def push_face(self, item: str) -> tuple | None:
        """N-push (spec S05): the face of ``item`` a closed hand pushes to slide it out to its board's edge, as a
        button hint (world position, outward normal, PUSH_RADIUS, depth); None when no push is wanted. Only for a
        flat item (its thinnest seen-box axis vertical) lying on a board of FIXED furniture (the map's) under another
        board: nothing a top-down pinch can take (hardbacks were pushed before 1,200 of 1,200 demo picks,
        boxing_books). The face is the side turned away from the robot, into the compartment, at mid-height, so the
        stroke runs toward the open front; the depth is the travel that leaves the near edge PUSH_OVERHANG past the
        board's edge for a pinch to close on -- and None once it does, so a pushed item is not pushed again. The
        item's shape is its points (``own_box``); the board and its edge are the map's."""
        obj = self.scene_object(item)
        box = self.own_box(obj)
        if box is None:
            return None
        pos, rot, lo, hi = box
        thin = int(np.argmin(hi - lo))
        if abs(rot[2, thin]) < 0.7:
            return None  # standing on edge: a side pinch takes it as it is
        corners = pos + np.array(list(itertools.product(*zip(lo, hi)))) @ rot.T
        centre, bottom = corners.mean(axis=0), float(corners[:, 2].min())
        board = None
        for other, alo, ahi in self.scene_aabbs():
            if other is obj or not getattr(other, "fixed_base", False) or not alo[2] <= bottom <= ahi[2]:
                continue
            if not ((alo[:2] <= centre[:2]).all() and (centre[:2] <= ahi[:2]).all()):
                continue
            mesh = self.collision_mesh_world(other)
            for z, blo, bhi, ceiling in boards(mesh.vertices, mesh.faces) if mesh is not None else ():
                under = abs(z - bottom) <= STANDS_ON_TOL and (blo <= centre[:2]).all() and (centre[:2] <= bhi).all()
                if under and ceiling < float("inf"):
                    board = (z, blo, bhi, mesh)
            if board is not None:
                break
        if board is None:
            return None
        into = centre[:2] - self.base_pose()[0][:2].cpu().numpy().astype(np.float64)
        into = into / (np.linalg.norm(into) or 1.0)
        # the side face (an axis that is not the thin one) whose outward normal points furthest from the robot
        a, s = max(((a, s) for a in range(3) if a != thin for s in (-1.0, 1.0)), key=lambda t: t[1] * rot[:2, t[0]] @ into)
        n = s * rot[:, a]
        position = pos + rot @ ((lo + hi) / 2.0) + n * float(hi[a] - lo[a]) / 2.0
        u = -n[:2] / (np.linalg.norm(n[:2]) or 1.0)  # the stroke, along the board
        front = float((corners[:, :2] @ u).max())
        # the board's edge on the stroke's own line: a ray from just inside the board's top out along the stroke
        # leaves the board at its edge whatever the case's yaw (boards() keeps only a board's axis-aligned box, which
        # a case turned 45 deg to the robot inflates by half its width: the book would be pushed off it)
        z, blo, bhi, mesh = board
        hits, _, _ = mesh.ray.intersects_location([[*centre[:2], z - 0.005]], [[*u, 0.0]], multiple_hits=True)
        out = [float((h[:2] - centre[:2]) @ u) for h in hits]
        edge = float(centre[:2] @ u) + min([d for d in out if d > 0.0], default=float("inf"))
        if not np.isfinite(edge):
            edge = max(float(np.array([x, y]) @ u) for x in (blo[0], bhi[0]) for y in (blo[1], bhi[1]))
        depth = edge + PUSH_OVERHANG - front
        if depth <= 0.0:
            log.info(f"{item} already hangs {front - edge:.3f} m past its board's edge; no push")
            return None
        log.info(f"push {item}: its far face at {np.round(position, 3).tolist()}, {depth:.3f} m toward the board's edge")
        return position, n, PUSH_RADIUS, float(depth)

    def upright_rotation(self, item: str) -> list | None:
        """E-6dof (spec S21): the rotation (base frame, 3x3) that stands ``item`` on end -- its longest seen-box axis
        turned vertical the shorter way round -- for a rose or a toothbrush going into a narrow vessel; yaw is the
        planner's. None until a capture has seen it."""
        box = self.own_box(self.scene_object(item))
        if box is None:
            return None
        _, rot, lo, hi = box
        axis = rot[:, int(np.argmax(hi - lo))]
        up = np.array([0.0, 0.0, 1.0 if axis[2] >= 0.0 else -1.0])
        pivot, cos = np.cross(axis, up), float(axis @ up)
        R = np.eye(3)
        if np.linalg.norm(pivot) > 1e-9:
            R = trimesh.transformations.rotation_matrix(math.atan2(float(np.linalg.norm(pivot)), cos), pivot)[:3, :3]
        base = T.quat2mat(self.base_pose()[1]).cpu().numpy().astype(np.float64)  # world <- base
        return (base.T @ R @ base).tolist()

    def tracked_label(self, name: str) -> str:
        """The tracked label of an object named in a goal atom: BDDL name -> per-instance label; a label stays."""
        return self.label_of(name) if name in self.bddl_names.values() else name

    def inside_region(self, item: str, container: str) -> dict | None:
        """The placement surface for inside(item, container): the open compartment's floor, as Cuboid kwargs in the
        base frame whose TOP face IS that floor. None, with the reason logged, when it cannot be derived, which
        leaves the round exactly as it is today.

        Privileged, like ``button_hints`` and ``openable_joints``. The geometry source is the predicate's own
        definition rather than a task guess: OmniGibson's ``Inside`` is unsatisfiable without a fillable meta link.
        """
        rect = self.inside_rect(item, container)
        if rect is None:
            return None
        centre, half, floor, ceiling = rect
        log.info(
            f"inside({item}, {container}): placing at ({centre[0]:.2f}, {centre[1]:.2f}), floor z={floor:.3f} world, "
            f"{2 * half[0]:.2f} x {2 * half[1]:.2f} m, under a ceiling of {ceiling:.3f}"
        )
        region = self.region_box(centre, half, floor, ceiling - floor)
        # an item longer than the vessel is wide goes in on end (E-6dof): lying, its centre never enters the volume
        # (roses by flat drop 2/5, spec S21). The planner turns it about its own origin, yaw free.
        seen = self.own_box(self.scene_object(item))
        if seen is not None and float((seen[3] - seen[2]).max()) > 2 * float(half.min()):
            log.info(f"inside({item}, {container}): {item} is longer than the vessel is wide; stood on end")
            region = dict(region, rotation=self.upright_rotation(item), yaw_tolerance=None)
        return region

    def side_entry(self, item: str, container: str) -> bool:
        """Whether ``item`` goes into or comes out of ``container`` sideways: a FIXED container (its mesh is the
        scanned map's) with geometry over its compartment rectangle within ``HAND_STACK`` above the item's top
        (``item_height``: the captures' points), a shelf or a cavity roof a top-down hand would hit. A pulled-out
        drawer's exposed part and an open-top box have none."""
        obj = self.scene_object(container)
        if obj is None or not obj.fixed_base:
            return False
        rect = self.inside_rect(item, container)
        if rect is None:
            return False
        centre, half, floor, _ = rect
        top = floor + self.item_height(item)
        # HEADROOM over the stack: petcxr's right bays leave a 10 cm item 21.3 cm, 4 mm more than the stack, which no
        # planner margin lets through (side_entry_check.out)
        roofed = roof_over(self.collision_mesh_world(obj), centre, half, top, top + HAND_STACK + HEADROOM)
        log.info(f"{container} has {'a' if roofed else 'no'} roof within {HAND_STACK + HEADROOM:.2f} m over {item} (top z={top:.3f})")
        return roofed

    def bay(self, link, lo, hi, near=None, n: int = 16, levels: int = 8) -> tuple:
        """The compartment of the fillable ``link`` nearest ``near`` (xy; the AABB centre when None): the accepted point
        of an n x n x levels grid over the AABB (lo, hi) nearest ``near`` (the lowest of equals), the run of accepted
        points through it along each axis, and its floor, probed 1 cm at a time straight down from it -- petcxr's two
        columns start 26 cm apart, so the volume's own bottom is the OTHER column's floor, behind the shut door.
        (centre xy, half xy, floor z); the AABB's own when nothing is accepted."""
        xs, ys, zs = (np.linspace(lo[a], hi[a], m + 2)[1:-1] for a, m in ((0, n), (1, n), (2, levels)))
        accepted = lambda pts: np.asarray(link.check_points_in_volume(th.tensor(pts, dtype=th.float32)))
        ok = accepted([[x, y, z] for x in xs for y in ys for z in zs]).reshape(n, n, levels)
        centre, half = (lo[:2] + hi[:2]) / 2.0, (hi[:2] - lo[:2]) / 2.0
        if not ok.any():
            return centre, half, float(lo[2])
        aim = centre if near is None else np.asarray(near, dtype=np.float64)[:2]
        i, k, l = min(zip(*np.nonzero(ok)),
                      key=lambda p: (round(float(np.hypot(xs[p[0]] - aim[0], ys[p[1]] - aim[1])), 2), p[2]))  # fmt: skip

        def run(line, at):
            a = b = at
            while a > 0 and line[a - 1]:
                a -= 1
            while b < len(line) - 1 and line[b + 1]:
                b += 1
            return a, b

        i0, i1 = run(ok[:, k, l], i)
        k0, k1 = run(ok[i, :, l], k)
        column = np.linspace(zs[l], lo[2], int((zs[l] - lo[2]) / 0.01) + 2)
        down = accepted([[xs[i], ys[k], z] for z in column])
        floor = float(column[int(np.argmin(down)) - 1]) if not down.all() else float(lo[2])
        cell = np.array([xs[1] - xs[0], ys[1] - ys[0]])
        return (np.array([xs[i0] + xs[i1], ys[k0] + ys[k1]]) / 2.0,
                (np.array([xs[i1] - xs[i0], ys[k1] - ys[k0]]) + cell) / 2.0, floor)  # fmt: skip

    def inside_rect(self, item: str, container: str) -> tuple | None:
        """The compartment of ``container`` that ``item`` goes into: (centre xy, half xy, floor z, ceiling z), world
        frame, a rectangle the fillable volume itself accepts; None, with the reason logged, when there is none."""
        from omnigibson.tiptop.articulation import openable_joints

        from b1k.bridge.articulation import is_open

        obj = self.scene_object(container)
        if obj is None:
            return None
        fills = [
            link
            for link in obj.links.values()
            if getattr(link, "is_meta_link", False) and getattr(link, "meta_link_type", "") in FILLABLE_META_LINKS
        ]
        if not fills:
            log.info(f"inside({item}, {container}): no fillable meta link, so there is no interior to place on")
            return None
        joints = openable_joints(obj)
        opened = [j for j in joints if is_open(j["lower"], j["upper"], j["position"], closed=j.get("closed"))]
        if joints and not opened:
            log.info(f"inside({item}, {container}): every joint of {container} is shut; no region")
            return None
        near = None
        if opened:  # the compartment belonging to the joint that is furthest open
            j = max(opened, key=lambda j: abs(float(j["position"]) - float(j.get("closed", 0.0))))
            moving = obj.links[j["link"]]
            mid = sum(v.cpu().numpy().astype(np.float64) for v in moving.aabb) / 2.0
            link = min(fills, key=lambda l: float(np.abs(l.visual_aabb_center.cpu().numpy() - mid).sum()))
            near = mid[:2]
        else:  # open-topped: no joint to open, the largest fillable volume is the one
            j, link = None, max(fills, key=lambda l: float(np.prod(l.visual_aabb_extent.cpu().numpy())))
        lo, hi = (v.cpu().numpy().astype(np.float64) for v in link.visual_aabb)
        # Restrict placement sampling to the exposed part of an open drawer. The complete collision map still
        # contains the cabinet above the hidden part; this region describes placement semantics, not free space.
        if j is not None and j["kind"] == "prismatic":
            body = self.container_body(obj, j["link"])
            k = int(np.argmax(np.abs(np.asarray(j["axis"], dtype=np.float64))))
            if body is not None and k < 2:
                blo, bhi = body.bounds
                if float(j["axis"][k]) * float(j["position"]) < 0:
                    hi[k] = min(hi[k], float(blo[k]))
                else:
                    lo[k] = max(lo[k], float(bhi[k]))
        item_obj = self.scene_object(item)
        if item_obj is None:
            return None
        ilo, ihi = (v.cpu().numpy().astype(np.float64) for v in item_obj.aabb)
        if j is None and obj.fixed_base and hi[2] - lo[2] > SHELF_CASE_HEIGHT:
            # a case of shelves has ONE fillable volume: aim at a reachable compartment, not its bottom board
            # (32 of 32 bookcase regions went to the bottom shelf, IK 0/256, and the fallback landed on the lid).
            # Fixed furniture only: its boards are the scanned map's; a movable bin's mesh is not ours to read.
            board = self.shelf_of(obj, self.item_height(item), (lo, hi))
            if board is None:
                log.info(f"inside({item}, {container}): no shelf of {container} in reach has room for it; no region")
                return None
            lo, hi = np.array([*board[1], board[0]]), np.array([*board[2], board[3]])
        # fridge petcxr's one fillable volume is two columns with the AABB's centre between them, so every centred
        # rectangle was refused (spec 6.2); gjeoer / rkgjer span two door bays. The rectangle is centred on the
        # accepted point nearest the opened door, sized to the run of accepted points about it, on that bay's floor.
        centre, half, lo[2] = self.bay(link, lo, hi, near=near)
        # where the SCORER will look for the item's AABB centre once it rests on the floor, clamped into the
        # volume so a tall object is still aimed at the floor rather than refused back onto the lid
        z_rest = min(lo[2] + (ihi[2] - ilo[2]) / 2.0, (lo[2] + hi[2]) / 2.0)
        for _ in range(8):  # shrink until the fillable volume itself accepts all four corners: the scorer's gate
            corners = th.tensor(
                [[centre[0] + sx * half[0], centre[1] + sy * half[1], z_rest] for sx in (-1, 1) for sy in (-1, 1)],
                dtype=th.float32,
            )
            if bool(link.check_points_in_volume(corners).all()):
                break
            half = half * 0.85
        else:
            log.info(f"inside({item}, {container}): {link.name} accepts no rectangle at z={z_rest:.3f}; no region")
            return None
        if 2 * float(half.min()) <= INSIDE_MIN_SIDE:
            log.info(f"inside({item}, {container}): the emerged interior is {2 * half} m, too small to place on")
            return None
        return centre, half, float(lo[2]), float(hi[2])

    def region_box(self, centre_xy, half_xy, z_top: float, dz: float) -> dict:
        """Cuboid kwargs in the base frame for a placement surface at world ``z_top``: the box is sunk so its TOP
        face is the surface (cuTAMP places on a surface's highest face)."""
        pos_b, quat_b = self.to_base(
            th.tensor([float(centre_xy[0]), float(centre_xy[1]), z_top - dz / 2.0], dtype=th.float32),
            th.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        pos, quat = (v.cpu().numpy().astype(np.float64) for v in (pos_b, quat_b))
        return {
            "dims": [2 * float(half_xy[0]), 2 * float(half_xy[1]), float(dz)],
            "pose": [*(float(v) for v in pos), float(quat[3]), *(float(v) for v in quat[:3])],  # cuRobo wants wxyz
        }

    def shelf_of(self, obj, item_height: float, within=None) -> tuple | None:
        """The board of a piece of furniture an item should rest on: (top z, xy lo, xy hi, ceiling z), world
        frame. Of the boards of its physical mesh (``boards``), those inside ``within`` (a world AABB), under
        PLACE_HEIGHT_MAX, with headroom for the item under the next board; the one nearest PLACE_HEIGHT among those
        with a point within BOARD_REACH of the base, else among the rest (the stance search moves the robot to it:
        from 1 m off the bench, a hallstand's 2.37 m top was the only board "in reach" and 6 rounds IK-failed on it,
        putting_shoes_on_rack 2026-09-24). None when no board has room."""
        mesh = self.collision_mesh_world(obj)
        if mesh is None:
            return None
        base = self.base_pose()[0][:2].cpu().numpy().astype(np.float64)
        near, far = [], []
        for z, lo, hi, ceiling in boards(mesh.vertices, mesh.faces):
            if within is not None:
                lo, hi = np.maximum(lo, within[0][:2]), np.minimum(hi, within[1][:2])
                ceiling = min(ceiling, float(within[1][2]))
                if not within[0][2] - 0.02 <= z < within[1][2] or (hi <= lo).any():
                    continue
            if z > PLACE_HEIGHT_MAX or ceiling - z < item_height + HEADROOM:
                continue
            (near if np.linalg.norm(np.clip(base, lo, hi) - base) <= BOARD_REACH else far).append((z, lo, hi, ceiling))
        return min(near or far, key=lambda b: abs(b[0] - PLACE_HEIGHT), default=None)

    def rest_region(self, item: str, fixture: str) -> dict | None:
        """The placement surface for touching(item, fixture): the fixture's board nearest PLACE_HEIGHT with
        headroom for the item, in reach. Planned onto the hull's top instead, a hallstand's top shelf gave IK
        0/256 four times of five; the one shoe that landed rests on its bench (2026-09-23). Fixed furniture only
        (the scanned map); a movable rack keeps the hull's top."""
        stand = self.scene_object(fixture)
        if not stand.fixed_base:
            return None
        board = self.shelf_of(stand, self.item_height(item))
        if board is None:
            log.info(f"touching({item}, {fixture}): no board of {fixture} in reach has room for it; no region")
            return None
        z, lo, hi, ceiling = board
        log.info(
            f"touching({item}, {fixture}): placing on its board at z={z:.3f} world, "
            f"{hi[0] - lo[0]:.2f} x {hi[1] - lo[1]:.2f} m, under a ceiling of {ceiling:.3f}"
        )
        return self.region_box((lo + hi) / 2.0, (hi - lo) / 2.0, z, 0.02)

    def footprint_region(self, name: str, z: float | None = None) -> dict | None:
        """A placement slab over the named fixture's footprint at world ``z``, its own top by default: the box of
        fixed furniture is the scanned map's; a movable table's is the world box of the points the captures saw
        it by (``own_box``), None until one has. A table's top as the planner's support plane instead of the plane
        RANSAC fits: with the item in hand nothing rests on the table, and 4 of 5 executed ontop(x, table) rounds
        were planned onto the floor (2026-09-23); the floor inside a fixture's footprint for under(x, it)."""
        obj = self.scene_object(name)
        if obj.fixed_base:
            lo, hi = (v.cpu().numpy().astype(np.float64) for v in obj.aabb)
        elif (box := self.seen_aabb(obj)) is not None:
            lo, hi = box
        else:
            log.info(f"{name} is movable and no capture has seen it: its top is not the map's to read; no region")
            return None
        return self.region_box((lo[:2] + hi[:2]) / 2.0, (hi[:2] - lo[:2]) / 2.0, float(hi[2]) if z is None else z, 0.02)

    def inside_regions(self, atoms: list[dict], oracle=None) -> dict:
        """``place_surfaces`` for the goal: the interior of every inside(a, b) container and a board of every
        touching(a, b) fixture, keyed by b's request label; under the planner's own support label the floor for
        a goal that puts something on the floor, the floor inside b's footprint for under(a, b), or the named
        table's top for ontop(a, table). The placements made for their effect -- stamp, cut, heat, aim, attached
        -- get their region from the geometry below over a hint fetched through ``oracle`` (the describing
        OracleKnowledge: particles, heat link, frames); with no oracle they get none."""
        pairs = [(a["predicate"], *a["args"]) for a in atoms if len(a.get("args", ())) == 2]
        intents = [t for t in pairs if t[0] in (*INTENT_PREDICATES, "attached")]
        pairs = [t for t in pairs if t not in intents]
        supports = {c for p, _, c in pairs if p != "inside"}
        out = {}
        # a fixture's footprint is read off its box only when it is part of the scanned map (fixed base)
        tables = {s for s in supports if bddl_category(s) == "table"}
        if any(bddl_category(support) in FLOOR_CATEGORIES for support in supports):
            out[PLANNER_SUPPORT] = self.floor_surface()
        elif under := [c for p, _, c in pairs if p == "under" and self.scene_object(c).fixed_base]:
            out[PLANNER_SUPPORT] = self.floor_surface(within=under[0])
        elif len(tables) == 1 and (top := self.footprint_region(tables.pop())) is not None:
            out[PLANNER_SUPPORT] = top
        refused = getattr(self, "region_refused", set())
        self.region_sent = set()  # the pairs this request carries a region for (bench refuses only those)
        for predicate, item, container in pairs:
            if predicate == "touching" and (region := self.rest_region(item, container)) is not None:
                out[self.label_of(container)] = region
            if predicate != "inside":
                continue
            if (item, container) in refused:  # the region already failed to plan for it: the hull's top, as before
                log.info(f"inside({item}, {container}): its compartment floor found no placement before; no region")
                continue
            if container in supports:  # also an ontop target in this goal: it needs its own hull as the surface
                log.info(f"inside({item}, {container}): {container} is also a support here; no region")
                continue
            region = self.inside_region(item, container)
            if region is not None:
                out[self.label_of(container)] = region
                self.region_sent.add((item, container))
        for predicate, item, target in intents if oracle is not None else ():
            if predicate == "stamp":
                region = self.stamp_region(item, target, oracle.particles(target), oracle.projection_box(item))
            elif predicate == "cut":
                region = self.knife_region(item, target)
            elif predicate == "heat":
                region = self.heat_region(item, target, oracle.heat_link(target))
            elif predicate == "aim":
                region = self.aim_target(item, target, oracle.nozzle(item))
            elif predicate == "pour":
                region = None  # held over the target's own hull top (keep_holding stops the place above it)
            else:
                region = self.attach_target(item, target, oracle.attach_frames(item, target))
            if region is not None:  # the wire names the plane for aim and for a floor target (protocol.tiptop_goal)
                floor = predicate == "aim" or bddl_category(target) in FLOOR_CATEGORIES
                out[PLANNER_SUPPORT if floor else self.label_of(target)] = region
        return out

    def seen_aabb(self, obj) -> tuple | None:
        """The world box of the points the captures saw ``obj`` by, at its pose now (``own_box``): (lo, hi); None
        until a capture has seen it."""
        box = self.own_box(obj)
        if box is None:
            return None
        pos, rot, lo, hi = box
        corners = pos + np.array(list(itertools.product(*zip(lo, hi)))) @ rot.T
        return corners.min(axis=0), corners.max(axis=0)

    def stamp_region(self, tool, target, particles, projection=None) -> dict | None:
        """The placement surface for stamp(tool, target): over the densest cluster of ``particles`` ((n, 3) world)
        the tool's footprint covers, at the cluster's own height, so the tool set down there removes it. An
        adjacency remover takes every particle inside its link's world box grown by STAMP_MARGIN
        (particle_modifier.py:524-531), so the footprint is the tool's seen box as it is held plus the margin, and
        the region carries that yaw (rotation identity, STAMP_YAW_TOL) so the fit holds when it lands. A projection
        remover (``projection``: the vacuum's slab, (lo, hi) in its own frame) removes inside that slab instead,
        which hangs 1-21 mm under its bottom: the fit is the slab's, no margin, and the tool rests VACUUM_HOVER above
        the particles. The box is the tool centres that cover the cluster, grown by the tool's half extents, since
        the planner keeps every sphere of the object inside a surface (cutamp stable_placement_costs). None when the
        tool has not been seen or nothing is left."""
        box = self.own_box(self.scene_object(tool))
        particles = np.asarray(particles, dtype=np.float64).reshape(-1, 3)
        if box is None or not len(particles):
            why = "no capture has seen the tool" if box is None else "nothing to remove"
            log.info(f"stamp({tool}, {target}): {why}; no region")
            return None
        pos, rot, lo, hi = box
        offsets = (np.array(list(itertools.product(*zip(lo, hi)))) - (lo + hi) / 2.0) @ rot.T  # body about its centre
        half = offsets.max(axis=0)
        # what the tool removes, relative to its body centre (xy) and to its bottom (z), world axes
        if projection is None:
            reach_lo, reach_hi = -half - STAMP_MARGIN, half + STAMP_MARGIN
            reach_lo[2], reach_hi[2] = -STAMP_MARGIN, 2 * half[2] + STAMP_MARGIN
        else:
            slab = (np.array(list(itertools.product(*zip(*projection)))) - (lo + hi) / 2.0) @ rot.T
            reach_lo, reach_hi = slab.min(axis=0), slab.max(axis=0)
            reach_lo[2], reach_hi[2] = reach_lo[2] + half[2], reach_hi[2] + half[2]
        size = reach_hi - reach_lo
        # ponytail: greedy over every particle as a corner of the footprint (4 corners), 40 particles at most
        best = []
        for p in particles:
            for sx, sy in itertools.product((0, 1), (0, 1)):
                x0, y0 = p[0] - sx * size[0], p[1] - sy * size[1]
                inside = particles[
                    (particles[:, 0] >= x0 - 1e-9) & (particles[:, 0] <= x0 + size[0] + 1e-9)
                    & (particles[:, 1] >= y0 - 1e-9) & (particles[:, 1] <= y0 + size[1] + 1e-9)
                ]  # fmt: skip
                inside = inside[inside[:, 2] <= inside[:, 2].min() + size[2]]  # within the tool's reach upward too
                if len(inside) > len(best):
                    best = inside
        cluster = np.asarray(best)
        centres_lo, centres_hi = cluster.max(axis=0) - reach_hi, cluster.min(axis=0) - reach_lo  # covering centres
        # the tool's bottom: resting on the particles' surface, or hovering where the slab straddles them
        top = max(float(cluster[:, 2].min()), float((centres_lo[2] + centres_hi[2]) / 2.0))
        centre, extent = (centres_lo[:2] + centres_hi[:2]) / 2.0, (centres_hi[:2] - centres_lo[:2]) / 2.0 + half[:2]
        log.info(
            f"stamp({tool}, {target}): {len(cluster)} of {len(particles)} particles under one stamp at "
            f"({centre[0]:.2f}, {centre[1]:.2f}), z={top:.3f} world, {2 * extent[0]:.2f} x {2 * extent[1]:.2f} m"
        )
        region = self.region_box(centre, extent, top, 0.02)
        return dict(region, rotation=np.eye(3).tolist(), yaw_tolerance=STAMP_YAW_TOL)

    def knife_region(self, knife, food) -> dict | None:
        """The placement surface for cut(knife, food): the food's top (its seen box), a square of the knife's length
        about the food's centre so the knife set down anywhere on it rests on the food: the slicer fires on any
        knife-link contact while armed (slicer_active.py:72-135), so the place is the cut."""
        food_box, knife_box = self.seen_aabb(self.scene_object(food)), self.own_box(self.scene_object(knife))
        if food_box is None or knife_box is None:
            log.info(f"cut({knife}, {food}): no capture has seen {food if food_box is None else knife}; no region")
            return None
        lo, hi = food_box
        length = float((knife_box[3] - knife_box[2]).max())
        return self.region_box((lo[:2] + hi[:2]) / 2.0, (hi[:2] - lo[:2] + length) / 2.0, float(hi[2]), 0.02)

    def heat_region(self, item, source, heat_link) -> dict | None:
        """The placement surface for heat(item, source): on the source's top, the item placements whose centre lies
        in the heat sphere (``heat_link``: (world xyz, radius); any overlap with it heats the whole object,
        heat_source_or_sink.py:253-266), grown by the item's seen box so the whole item is accepted there. The top is
        the source's map mesh under the sphere for a fixture (a stove's grate), its seen box for a movable."""
        if heat_link is None:
            return None
        centre, radius = np.asarray(heat_link[0], dtype=np.float64), float(heat_link[1])
        obj = self.scene_object(source)
        if obj.fixed_base:  # rays down at the link and around it: the grate over a burner, not the panel behind it
            angles = np.arange(0, 6.28, 0.785)
            ring = [centre[:2]] + [centre[:2] + 0.04 * np.array([math.cos(a), math.sin(a)]) for a in angles]
            origins = [[x, y, centre[2] + HEAT_TOP_ABOVE] for x, y in ring]
            down = [[0.0, 0.0, -1.0]] * len(origins)
            hits, _, _ = self.collision_mesh_world(obj).ray.intersects_location(origins, down)
            top = float(hits[:, 2].max()) if len(hits) else float(centre[2])
        elif (box := self.seen_aabb(obj)) is not None:
            top = float(box[1][2])
        else:
            log.info(f"heat({item}, {source}): {source} is movable and no capture has seen it; no region")
            return None
        item_box = self.seen_aabb(self.scene_object(item))
        if item_box is None:
            log.info(f"heat({item}, {source}): no capture has seen {item}; no region")
            return None
        reach = math.sqrt(max(radius**2 - (top - centre[2]) ** 2, 0.0)) / math.sqrt(2)  # the square inside the sphere
        if reach <= 0.0:
            log.info(f"heat({item}, {source}): the surface at z={top:.3f} is out of the heat sphere; no region")
            return None
        log.info(f"heat({item}, {source}): within {radius:.2f} m of the heat link (z={centre[2]:.3f}) at z={top:.3f}")
        return self.region_box(centre[:2], (item_box[1][:2] - item_box[0][:2]) / 2.0 + reach, top, 0.02)

    def aim_target(self, tool, target, nozzle) -> dict | None:
        """The placement surface for aim(tool, target): the tool stood on the target's support on the robot's side
        of it, within the nozzle's reach (``nozzle``: (4x4 world frame, reach m); it sprays down the frame's -z),
        turned so the spray points at the target: the region carries the yaw that turns the nozzle's direction now
        onto the target, with AIM_YAW_TOL. Under the planner's support label (the wire says on(tool, table)): the
        target's bottom is that support's height."""
        if nozzle is None:
            return None
        frame, reach = np.asarray(nozzle[0], dtype=np.float64), float(nozzle[1])
        target_box, tool_box = self.seen_aabb(self.scene_object(target)), self.seen_aabb(self.scene_object(tool))
        if target_box is None or tool_box is None:
            log.info(f"aim({tool}, {target}): no capture has seen {target if target_box is None else tool}; no region")
            return None
        lo, hi = target_box
        centre = (lo[:2] + hi[:2]) / 2.0
        u = self.base_pose()[0][:2].cpu().numpy().astype(np.float64) - centre  # toward the robot
        u = u / (np.linalg.norm(u) or 1.0)
        face = float(np.abs(u) @ ((hi[:2] - lo[:2]) / 2.0))  # the target's box along u
        half = reach / 2.0 + (tool_box[1][:2] - tool_box[0][:2]) / 2.0
        direction = -frame[:3, 2]
        yaw = math.atan2(-u[1], -u[0]) - math.atan2(direction[1], direction[0])  # about z: the same in the base frame
        c, s = math.cos(yaw), math.sin(yaw)
        log.info(f"aim({tool}, {target}): within {reach:.2f} m on the robot's side, turned {math.degrees(yaw):.0f} deg")
        return dict(
            self.region_box(centre + u * (face + reach / 2.0), half, float(lo[2]), 0.02),
            rotation=[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            yaw_tolerance=AIM_YAW_TOL,
        )

    def attach_target(self, child, parent, frames) -> dict | None:
        """The placement surface for attached(child, parent): where the child's body must rest for its male meta link
        to land within ATTACH_TOL of the parent's female one (``frames``: (male 4x4, female 4x4), world), turned so
        the frames align: the region carries R = R_F R_M^T -- the child as held, turned onto the female frame; the
        planner's object frame is base-aligned at the capture, so this is the rotation it applies -- in the base
        frame, with ATTACH_YAW_TOL. Its top sets the child ATTACH_LIFT above the aligned height: at or above, since
        the poster has 3.2 cm of margin downward (spec 6.2 Attach) and the snap allows 5 cm."""
        if frames is None:
            return None
        male, female = (np.asarray(f, dtype=np.float64) for f in frames)
        box = self.own_box(self.scene_object(child))
        if box is None:
            log.info(f"attached({child}, {parent}): no capture has seen {child}; no region")
            return None
        pos, rot, lo, hi = box
        R = female[:3, :3] @ male[:3, :3].T
        centre_now = pos + rot @ ((lo + hi) / 2.0)
        centre = female[:3, 3] - R @ (male[:3, 3] - centre_now)  # the body's centre with the male frame on the female
        offsets = (np.array(list(itertools.product(*zip(lo, hi)))) - (lo + hi) / 2.0) @ (R @ rot).T
        half = offsets.max(axis=0)[:2] + ATTACH_TOL / math.sqrt(2)
        top = float(centre[2] + offsets[:, 2].min()) + ATTACH_LIFT
        base = T.quat2mat(self.base_pose()[1]).cpu().numpy().astype(np.float64)  # world <- base
        log.info(f"attached({child}, {parent}): body centre to {np.round(centre, 2).tolist()}, bottom z={top:.3f}")
        return dict(
            self.region_box(centre[:2], half, top, 0.02),
            rotation=(base.T @ R @ base).tolist(),
            yaw_tolerance=ATTACH_YAW_TOL,
        )

    def floor_surface(self, within: str | None = None) -> dict:
        """A placement slab on the floor (base frame, top face at the floor under the base): in front of the robot
        for goals that put something on the floor, or the footprint of the object ``within`` names for under(x,
        it). The planner otherwise places onto whatever plane its RANSAC fit picked: at the sink that was a shelf
        inside the sink stand, and every put-down failed (dishes 2026-09-22). The planner keeps it as a collider,
        as it keeps the plane it replaces."""
        bx, by, bz = (float(v) for v in self.base_pose()[0])
        tops = [float(hi[2]) for o, lo, hi in self.scene_aabbs()
                if o.category in ("floors", "lawn") and lo[0] <= bx <= hi[0] and lo[1] <= by <= hi[1]]
        top = min(tops, key=lambda t: abs(t - bz)) if tops else bz
        if within is None:
            return {"dims": [0.6, 0.8, 0.02], "pose": [0.6, 0.15, top - bz - 0.01, 1.0, 0.0, 0.0, 0.0]}
        return self.footprint_region(within, top)

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
    def scope_floor(self, name: str):
        """The simulated object a BDDL floor (or lawn) name stands for, or None.

        ``task_scope`` leaves floors out on purpose -- they are not things the episode poses or frames -- but a
        goal may still name one as the place an item has to end up, and then the runner needs to know where it
        is. bringing_in_wood asks for three sheets of plywood ontop floor.n.01_2 while they start on
        floor.n.01_1, so "the floor" is not one place.
        """
        if bddl_category(name) not in FLOOR_CATEGORIES:
            return None
        obj = getattr(self.env.task, "object_scope", {}).get(name)
        return obj if isinstance(obj, USDObject) else None

    def scene_object(self, name: str):
        obj = self.task_scope().get(name) or self.scope_floor(name) or self.env.scene.object_registry("name", name)
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

    def robot_height_cells(self, obj) -> set | None:
        """Which ``FOOTPRINT_CELL`` squares of floor this object occupies anywhere the ROBOT is, floor to head.

        A bounding box is not the object. A box drawn round an L-shaped desk covers the notch the robot can stand
        in, and refusing that notch is what made every task whose objects sit on a desk fail with "no base pose
        ... overlaps desk" before a single round ran (2026-09-14).

        The slab was once the base's own height, on the reasoning that a coffee table is all air where the wheels
        go, so the base could roll underneath it. The wheels can; the robot cannot. Above the base sit the torso
        and the arms, and a tabletop at 0.40 m is exactly where they are. Run that way, tidying_living_room parked
        the robot under the coffee table at 0.25-0.60 m: the arm could no longer fold for travel (18 blocked ramps
        against 0 before), never returned to the ready posture, came to rest in front of the head camera, and
        every look at the notebook returned an empty mask. 0.250 -> 0.000 (2026-09-15). The commit that made the
        slab the base's own height said in its own message that it was "a regression risk for tasks that currently
        work by standing close to furniture", and was merged without a run; this is that risk arriving.

        So the slab runs from the base's underside to the top of the robot. What that still buys over the box is
        every gap empty at EVERY height -- the notch of an L, the hollow inside a horseshoe of furniture -- which
        is the half of the original measurement that was about shape rather than about height.

        Returns None when the object has no mesh, and the caller then keeps the box's word.

        Computed once per object -- the mesh is cached and nothing moves during a stance search -- because doing
        it per candidate is what took an earlier version of this search from a median of 1.0 s to 48.8 s.
        """
        if obj.name in self._base_cells:
            return self._base_cells[obj.name]
        cells = None
        try:
            mesh = self.scene_mesh(obj)
            lo_b, _ = self.base_box()
            floor = float(self.base_pose()[0][2])
            cells = footprint_cells(mesh, floor + float(lo_b[2]), floor + ROBOT_HEIGHT)
        except Exception as why:  # noqa: BLE001 - no mesh, or an unreadable one: the box stands
            log.debug(f"no robot-height geometry for {obj.name} ({type(why).__name__}); keeping its box")
            cells = None
        self._base_cells[obj.name] = cells
        return cells

    def base_meets(self, obj, centre, yaw, rect_lo, rect_hi) -> bool:
        """Whether the robot standing here actually meets ``obj``, rather than merely meeting its bounding box.

        The rectangle is the base's, because that is the widest the robot gets; the heights tested are the
        whole robot's (``robot_height_cells``), because the torso and arms ride above the wheels.
        """
        cells = self.robot_height_cells(obj)
        if cells is None:
            return True  # nothing better to go on than the box, which the caller has already found overlapping
        if not cells:
            return False  # no geometry anywhere in the robot's height: this part of the box is empty air
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
        self, x: float, y: float, ignore, aabbs=None, yaw: float | None = None, arms: bool = True, reaching=()
    ) -> tuple[bool, str]:
        """Floor under the whole footprint, inside a room, and no object's box overlapping the base.

        The geometry of all three is in ``b1k.bridge.geometry`` (``footprint_corners``, ``corner_off_floor``,
        ``footprint_blockers``) and takes plain boxes. What is left here is what only the simulator can answer:
        where the base and its underside actually are, which room a point is in, whether an object the base's
        rectangle meets is solid there or is 96% air (``base_meets``), and where the arms would come to rest.

        ``arms``: also refuse a stance where an arm at its CURRENT posture would rest inside something (below).
        Off for an opening stance, whose arms stay folded over the base after the teleport and go from there
        to the handle by a checked path: with the working posture they would sit 0.41 m past the base, i.e.
        inside the very drawer front the stance is chosen to reach, and every stance near it was refused.

        ``aabbs``: a scene_aabbs() snapshot to test against (taken here otherwise; nothing moves during a search).
        """
        rect = (
            self.base_box()[:, :2] + np.array([[-FOLD_OVERHANG] * 2, [FOLD_OVERHANG] * 2]) if yaw is not None else None
        )
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        underside = float(self.base_pose()[0][2]) + float(self.base_box()[0][2])  # world z the base clears
        off = corner_off_floor(
            footprint_corners(x, y, yaw, rect), [(lo, hi) for o, lo, hi in aabbs if o.category == "floors"]
        )
        if off is not None:
            return False, f"no floor under ({off[0]:.2f}, {off[1]:.2f})", 0.0
        try:
            room = self.env.scene.seg_map.get_room_instance_by_point(th.tensor([x, y]))
        except Exception:  # noqa: BLE001 - a point off the map's raster raises inside the lookup; it is a filter only
            room = "unknown"
        if room is None:
            return False, "outside every room", 0.0
        # A partial-rooms scene loads the walls of every room but only the floors of the rooms it loads, so the
        # stance search could put the robot in a room with no floor and it fell out of the world: boxing_books
        # (bedroom_0) twice on 2026-09-22, and 117 of last week's 189 topples of 45 deg or more. The floor test
        # above uses the floor objects' boxes, and an L-shaped living-room floor's box covers the missing room.
        loaded = getattr(self.env.scene, "load_room_instances", None)
        if loaded is not None and room not in (None, "unknown") and room not in loaded:
            return False, f"in {room}, which this scene did not load (no floor there)", 0.0
        # a floor covering is ground only while it is flat: the landing check (base_placement_collision) reads
        # ground by height, and a 9.5 cm paver kerb free to this test was solid to that one -- every candidate
        # stance refused after the search accepted it (bringing_in_wood 301, 2026-09-23)
        near = [
            (obj, lo, hi)
            for obj, lo, hi in aabbs
            if not (obj is self.robot or obj in ignore or obj.category == "floors"
                    or obj.category in FLOOR_COVERINGS and float(hi[2]) < GROUND_TOP)
        ]  # fmt: skip
        blocked, clearance = footprint_blockers(x, y, yaw, rect, [(lo, hi) for _, lo, hi in near], underside)
        for i in blocked:
            obj = near[i][0]
            # The box says they meet; ask the object itself, which may be legs and air at the height the base
            # sweeps. With no yaw there is no rectangle to ask about and the square's word is final.
            if rect is None or self.base_meets(obj, (x, y), yaw, rect[0], rect[1]):
                return False, f"overlaps {obj.name}", 0.0
        # The base's rectangle is not the robot: with the working posture the arms reach 0.41 m past it, so a
        # stance whose base is clear can still leave the hand inside a box on the floor. The user watched exactly
        # that in putting_away_toys: after picking up a toy the robot teleported to the table and its arm came to
        # rest INSIDE the toy box (2026-09-13). The arms are tested in 3D, at the posture they will unfold to and
        # at the pose being judged, so a stance is refused for where the arm ENDS UP rather than only for where
        # the wheels are.
        if yaw is not None and self.q_home is not None and arms:
            # The arm may rest inside the thing it is reaching INTO -- that is what reaching into a bookcase or a
            # bin looks like -- but the BASE still may not stand inside it, which is why these are two sets and
            # not one. An earlier comment claimed objects being stood for were in ``ignore``; they never were.
            # Standing for a bookcase, the arm ends up inside the bookcase's box and the stance was refused for
            # it, so "no base pose reaches ['bookcase.n.01_2'] ... {'overlaps bookcase_zfpyqe_0': 2497}" was the
            # search refusing every stance from which the goal could be served (2026-09-15).
            spared = {o.name for o in ignore} | {o.name for o in reaching}
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
        drop = -float(pos[2])  # move_base puts the base at z 0; a fall through a missing floor stays level at first
        self.last_settle = {"tilt_deg": tilt, "drop_m": drop, "moved_m": moved}
        if drop > FALL_DROP:
            return False, f"the base dropped {drop:.2f} m below where it was put (no floor under it)"
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
            try:
                collision = self.ramp_collision([], [], [], self.grasp_contacts(arm, obj), self.CLOSE)
            except Exception as exc:
                log.warning("refusing unvalidated grasp closure on %s: %s", obj.name, exc)
                return seed, False
            if collision is not None:
                log.warning("refusing grasp closure on %s: %s", obj.name, collision)
                return seed, False
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
            # unleashed: this press exists to make contact -- the grasp assist fires on finger contact, so bounding
            # how hard it leans would be bounding the thing it is for
            stopped = self.ramp_to(
                targets, self.posture, self.CLOSE, OPEN_SETTLE_STEPS, note="press onto the panel", leashed=False,
                allowed_contacts=self.grasp_contacts(arm, obj),
            )
            if stopped is not None:
                log.warning("stopping the grasp press after rejected motion: %s", stopped[0])
                return seed, False
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
        # under the assisted weld a pressed face has no second finger and no ray between the pads (robot.py ~3150):
        # only what a jaw closes around (a bar) or pinches (a panel's top edge) can take hold
        pressable = getattr(self.robot, "grasping_mode", "sticky") == "sticky"
        for j in joints:
            vertical = j["kind"] == "revolute" and abs(float(j["axis"][2])) >= 0.7
            want = fraction if vertical or j["kind"] != "revolute" else max(fraction, LID_FRACTION)
            travel = opening_travel(j["kind"], j["lower"], j["upper"], j["position"], want, closed=j.get("closed"))
            if vertical:  # what a door's arc leaves beyond this is push_joint's to finish
                travel = float(np.clip(travel, -DOOR_TRAVEL_MAX, DOOR_TRAVEL_MAX))
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
            for z in heights if pressable else ():
                point = np.array([face_c[0], face_c[1], z], dtype=np.float64)
                on_surface = self.surface_point(link, point, into)
                out.append(
                    dict(joint=j, travel=travel, kind=h["kind"], tips=on_surface, into=into, jaws=uprights,
                         press=GRASP_PRESS, lead=lead, rank=(1, abs(float(on_surface[2] - hand_world[2]))),
                         handle=h, nudges=GRASP_NUDGES)
                )  # fmt: skip
            # the panel's top edge, pinched from above across its thickness: what a flat panel offers a jaw
            top = face_c + h["up"] * (h["face_extent"][1] / 2.0 - EDGE_INSET) - lead * EDGE_THICKNESS
            out.append(
                dict(joint=j, travel=travel, kind="edge", tips=top, into=-h["up"], jaws=[lead, -lead], press=0.0,
                     lead=lead, rank=(2, abs(float(top[2] - hand_world[2]))), handle=h, nudges=1)
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

    def stance_for_grasp(
        self, obj, grasps: list, arm: str = "left", limit: int = OPEN_STANCE_TRIES, with_torso: bool = True
    ) -> tuple:
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
        ik = self.arm_ik(arm, frame=f"{arm}_gripper_link", with_torso=with_torso)
        joints_of = self.ik_joint_names(arm, with_torso=with_torso)
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
        plan wins; an empty list means every candidate intersects geometry and no motion is authorized.
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
        log.warning("reach: no candidate clears the scene; closest candidate intersects %s", best[2] if best else [])
        return []

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
            # unleashed: pulling a drawer IS pushing on a joint, and follow_pull already fails on "the hand moved
            # but the joint did not", so it does not need the leash to notice it is stuck
            stopped = self.ramp_to(
                self._targets_from(joints_of, solution), self.posture, self.CLOSE, 0,
                note=f"pull {obj.name} waypoint {i + 1} of {len(pulls)}", max_vel=OPEN_MAX_JOINT_VEL, leashed=False,
                allowed_contacts=self.grasp_contacts(arm, obj),
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
            already = any(is_open(j["lower"], j["upper"], j["position"], closed=j.get("closed")) for j in joints)
            return {"opened": already, "why": "already open" if already else f"no joint of {name} can be taken hold of"}
        run = self._drive_joint(arm, obj, grasps, name, stand=stand)
        if "plan" not in run:
            return {"opened": False, **run}
        chosen, plan, done, blocked = run["grasp"], run["plan"], run["waypoints"], run["why"]
        j = chosen["joint"]
        now = next((k for k in openable_joints(obj) if k["name"] == j["name"]), j)
        opened = is_open(now["lower"], now["upper"], now["position"], closed=now.get("closed"))
        log.info(
            f"{name}.{j['name']} ({j['kind']}): asked for {chosen['travel']:+.3f}, pulled {done} of {plan['reached']} "
            f"solved waypoints ({OPEN_PATH_STEPS - 1} in the full path), joint now {now['position']:.3f} of "
            f"[{now['lower']:.2f}, {now['upper']:.2f}] -- {'OPEN' if opened else 'still closed'}"
            + (f"; {blocked}" if blocked else "")
        )
        out = {"opened": opened, "why": blocked, "joint": j["name"], "position": now["position"], "grip": chosen["kind"],
               "waypoints": done, "solved": plan["reached"], "held": run["held"], "stance": run["stance"]}  # fmt: skip
        # a door's arc runs out of any fixed stance (solve_pull keeps a pull reaching OPEN_MIN_FRACTION): what the
        # pull left is pushed, on the door's inner face
        target = float(j["position"] + chosen["travel"])
        if done and j["kind"] == "revolute" and abs(float(now["position"]) - target) > JOINT_TOL * abs(j["upper"] - j["lower"]):
            pushed = self.push_joint(arm, name, now, target)
            position = float(pushed.get("position", now["position"]))
            out.update(pushed=pushed, position=position,
                       opened=is_open(now["lower"], now["upper"], position, closed=now.get("closed")))  # fmt: skip
        return out

    def push_grasps(self, obj, j: dict, target: float, hand_world) -> list:
        """Where the closed hand pushes ``obj``'s moving link toward joint value ``target``: a column of points on
        the face that TRAILS the link's motion (``handle_on`` read along the motion reversed: a drawer's front as it
        closes, a door's outer face, its inner face when it is pushed wider), the hand coming in along the motion
        and its standoff back along it. Best first: nearest the hand's own height."""
        travel = float(target) - float(j["position"])
        link = obj.links.get(j["link"])
        if link is None or abs(travel) < 1e-4:
            return []
        mesh = self.link_trimesh_world(link)
        if mesh is None or not len(mesh.vertices):
            return []
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        motion = leading_direction(j["kind"], j["axis"], j["origin"], verts, travel)
        h = handle_on(verts, -motion)
        face_c = np.asarray(h["face_centre"], dtype=np.float64)
        band = max(0.0, h["face_extent"][1] / 2.0 - GRASP_COLUMN_INSET)
        out = []
        for s in np.linspace(-band, band, GRASP_COLUMN_SAMPLES):
            point = self.surface_point(link, face_c + h["up"] * float(s), motion)
            out.append(
                dict(joint=j, travel=travel, kind="push", tips=point, into=motion, jaws=[np.array([0.0, 0.0, 1.0]), h["side"]],
                     press=0.0, lead=-motion, rank=(0, abs(float(point[2] - hand_world[2]))), handle=h, nudges=0)
            )  # fmt: skip
        out.sort(key=lambda g: g["rank"])
        return out

    def push_joint(self, arm: str, name: str, joint: dict, target: float) -> dict:
        """Push ``name``'s moving link to joint value ``target`` with the closed hand: a close is a push (5 of
        13,362 demo closes grasp, and the simulator has no latch), and so is widening a door past where the pull
        reached. The hand comes in along the link's motion onto the face that trails it (``push_grasps``) and
        follows ``follow_joint`` waypoints there, contact allowed only with the moving link (``solve_pull``'s body
        check); the joint counts as there within ``JOINT_TOL`` of its range."""
        obj = self.scene_object(name)
        span = abs(float(joint["upper"] - joint["lower"]))
        hand_world = self.robot.eef_links[arm].get_position_orientation()[0].cpu().numpy().astype(np.float64)
        grasps = self.push_grasps(obj, joint, target, hand_world)
        if not grasps:
            there = abs(float(joint["position"]) - float(target)) <= JOINT_TOL * span
            return {"reached": there, "why": "" if there else f"nothing of {name}.{joint['name']} to push on"}
        run = self._drive_joint(arm, obj, grasps, name, take_hold=False)
        if "plan" not in run:
            return {"reached": False, **run}
        now = next((k for k in openable_joints(obj) if k["name"] == joint["name"]), joint)
        reached = abs(float(now["position"]) - float(target)) <= JOINT_TOL * span
        log.info(
            f"{name}.{joint['name']}: pushed {run['waypoints']} of {run['plan']['reached']} waypoints toward "
            f"{float(target):+.3f}, joint now {now['position']:.3f} -- {'THERE' if reached else 'short'}"
            + (f"; {run['why']}" if run["why"] else "")
        )
        return {"reached": reached, "why": run["why"], "joint": joint["name"], "position": now["position"],
                "target": float(target), "waypoints": run["waypoints"], "solved": run["plan"]["reached"],
                "stance": run["stance"]}  # fmt: skip

    def _drive_joint(self, arm: str, obj, grasps: list, name: str, stand: bool = True, take_hold: bool = True) -> dict:
        """Take the first of ``grasps`` whose whole motion solves and carry it along its joint: the stance
        (``stand``), the reach to the standoff, the approach, the hold (``take_hold``: close on the handle; else
        the hand comes closed and pushes), the pull (``follow_pull``) and the retreat. The torso is held while
        the other hand holds something: a carried item rides its lean. A dict with "why" alone when it stops
        short; else grasp, plan, waypoints, why, held, stance."""
        torso = self.other_arm not in self.hands().values()
        gripper = self.OPEN if take_hold else self.CLOSE
        ik = self.arm_ik(arm, frame=f"{arm}_gripper_link", with_torso=torso)
        joints_of = self.ik_joint_names(arm, with_torso=torso)
        chosen, jaw_world, plan, pose = None, None, None, None
        if stand:
            pose, chosen, jaw_world = self.stance_for_grasp(obj, grasps, arm=arm, with_torso=torso)
            if pose is None:
                return {"why": f"no stance in front of {name} lets the arm reach its handle and pull"}
            try:
                self.place_robot(*pose, note=f"stand to open {name}", unfold=False)
            except RuntimeError as exc:  # BasePlacementCollision or an unvalidated destination
                return {"why": f"opening stance rejected ({exc})"}
            self.hold(OPEN_SETTLE_STEPS, gripper)
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
            return {"why": f"no grasp on {name} solves from here"}
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
        if not legs and not self.planned_standoff(arm, ik, joints_of, plan["solutions"][0], name):
            return {"why": f"no collision-free approach to {name}"}
        for k, leg in enumerate(legs):
            stopped = self.ramp_to(self._targets_from(joints_of, leg), self.posture, gripper, OPEN_SETTLE_STEPS,
                                   note=f"reach the standoff of {name} (leg {k + 1} of {len(legs)})", max_vel=OPEN_MAX_JOINT_VEL)  # fmt: skip
            if stopped is not None and " intersects " in stopped[0]:
                # refused before it moved: the straight leg sweeps the cabinet (store_honey, 2026-09-22)
                if self.planned_standoff(arm, ik, joints_of, plan["solutions"][0], name):
                    break
            if stopped is not None:
                q_now = self.robot.get_joint_positions()
                measured = [float(q_now[self.joint_index[jn]]) for jn in joints_of]
                off = float(np.linalg.norm(ik.fk(measured, f"{arm}_gripper_link")[0] - ik.fk(leg, f"{arm}_gripper_link")[0]))
                if off > OPEN_REACH_SLACK or k + 1 < len(legs):
                    self.hold(OPEN_SETTLE_STEPS, gripper)
                    return {"joint": j["name"], "position": j["position"],
                            "why": f"{stopped[0]} stopped following on the way to the standoff (leg {k + 1} of {len(legs)}, "
                                   f"hand {off * 100:.1f} cm short)"}  # fmt: skip
                log.info(f"{stopped[0]} settled {stopped[2]:.2f} rad short at the end of the reach; the hand is {off * 100:.1f} cm off the standoff, going on")
        approach = lambda: self.ramp_to(self._targets_from(joints_of, plan["solutions"][1]), self.posture, gripper,
                                        OPEN_SETTLE_STEPS, note=f"approach the handle of {name}", max_vel=OPEN_MAX_JOINT_VEL,
                                        allowed_contacts=self.grasp_contacts(arm, obj))  # fmt: skip
        stopped = approach()
        # the idle hand rode the torso's lean into the cabinet (store_honey 302/303, 2026-09-23); a pair of the
        # robot's own links is named in the model's link order, so the idle arm may be on either side
        if (stopped is not None and any(s.startswith(f"{self.other_arm}_") for s in stopped[0].split(" intersects "))
                and self.tuck_idle_arm()):  # fmt: skip
            stopped = approach()
        if stopped is not None:
            return {"why": f"handle approach rejected: {stopped[0]}"}
        grabbed = False
        if take_hold:
            seed, grabbed = self.close_on(arm, ik, obj, j["link"], plan["grasp_pose"], -plan["lead"], plan["solutions"][1],
                                          joints_of, nudges=chosen["nudges"])  # fmt: skip
            if not grabbed:
                log.info(f"nothing to pull on: the assist never took hold of {name}.{j['name']}")
        # 4. the pull (or the push: the same waypoints with the hand closed)
        done, blocked = self.follow_pull(arm, joints_of, plan, obj, chosen)
        # 5. let go and back off along the pull, checked like everything else
        self.hold(OPEN_SETTLE_STEPS, gripper)
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
            self.ramp_to(self._targets_from(joints_of, back), self.posture, gripper, OPEN_SETTLE_STEPS,
                         note=f"back off from {name}", max_vel=OPEN_MAX_JOINT_VEL)  # fmt: skip
        return {"grasp": chosen, "plan": plan, "waypoints": done, "why": blocked, "held": bool(grabbed), "stance": pose}

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

    def best_base_pose(self, points_xy, ignore=(), reach: float = 0.9, aabbs=None, half_widths=None, support_z=None,
                       avoid=(), boxes=None, frame_strict: bool = True, footprint: dict | None = None, reaching=(),
                       refused=()):
        """``geometry.best_base_pose`` on this robot: the search itself is geometry and lives there.

        Four of the five things it used to read off the live scene are one reading each -- the head camera's K,
        image size and base-frame pose (``head_camera_in_base``), how far ahead of the base the bottom of the
        frame meets each object's support (``camera_floor_distance``), and the scene's AABBs, which was already
        an argument. The fifth, ``_footprint_free``, genuinely has to ask the scene for every candidate, so it
        goes in as a callback with ``ignore``, ``aabbs`` and ``reaching`` bound here.

        ``refused``: (x, y, yaw, obstacle or None) the landing check refused earlier in this search. A candidate
        within AVOID_RADIUS and AVOID_YAW of one is not proposed again, and every obstacle named is checked at
        every other candidate the way the landing check will (its measured posture, carried volume and
        TELEPORT_CLEARANCE) before the search offers it: the proposer's footprint model spares the target for the
        arm and has no torso, so the same obstacle refused 68% of the 2801 landing candidates it had already
        refused once in that search (2026-09-23).
        """
        aabbs = self.scene_aabbs() if aabbs is None else aabbs
        k, base_from_cam, base_z = self.head_camera_in_base()
        camera = HeadCamera(k, base_from_cam, base_z, int(self.robot_cam.image_width), int(self.robot_cam.image_height))
        min_dists = None
        if support_z is not None:
            heights = [support_z] * len(points_xy) if np.isscalar(support_z) else list(support_z)
            min_dists = [self.camera_floor_distance(float(z)) + CAMERA_MIN_MARGIN for z in heights]
        # by its mesh only when it is the map's (fixed base); a movable basket leaves its avoid disc and nothing else.
        # The name can be a robot link ("base_link" when a carried tile's volume met the base, laying_tile_floors
        # 2026-09-24): no scene object, never a blocker -- the registry answers None where scene_object raises
        hard = {o for *_, o in refused
                if o and getattr(self.env.scene.object_registry("name", o), "fixed_base", False)}

        def footprint_free(x, y, yaw):
            if any(
                np.hypot(x - rx, y - ry) < AVOID_RADIUS and abs((yaw - ryaw + np.pi) % (2 * np.pi) - np.pi) < AVOID_YAW
                for rx, ry, ryaw, _ in refused
            ):
                return False, "refused before", 0.0
            free, why, clearance = self._footprint_free(x, y, ignore, aabbs=aabbs, yaw=yaw, reaching=reaching)
            if free and hard:
                # ponytail: re-reads every scene AABB per candidate (~50 ms); hand it `aabbs` if a search gets slow
                hit = self.base_placement_collision(float(x), float(y), float(yaw), only=hard)
                if hit is not None:
                    return False, f"{hit[0]} would intersect {hit[1]}", 0.0
            return free, why, clearance

        return search_base_poses(
            points_xy,
            camera,
            footprint_free,
            reach=reach,
            half_widths=half_widths,
            min_dists=min_dists,
            avoid=avoid,
            boxes=boxes,
            frame_strict=frame_strict,
            footprint=footprint,
        )

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

    def place_robot_for(self, *names: str, ignore_names=(), reach: float = 0.9, avoid=(), refused=None) -> dict:
        """Stand where every item and the target (the last name) are in the left arm's reach ("navigation done").

        A single name is a target with no items (a one-object task such as turning_on_radio). ignore_names:
        objects that do not count as obstacles, resolved like place_robot_near's (unknown names raise). Objects
        the robot holds never count (they travel with it). ``avoid``: (x, y) poses not to stand at again.
        ``refused``: a list of (x, y, yaw, obstacle) the landing check refused, extended here with this call's
        so that a retry (the caller's wider search) can hand them back and not test them again: a refusal used
        to leave only an (x, y) disc behind, which also hid every other heading at that spot, and the wider
        search started from nothing and re-tested the identical 8 candidates in 152 of 377 failed searches
        (2026-09-23). The candidates are judged against the obstacles named from then on (``best_base_pose``).
        """
        if not names:
            raise ValueError("place_robot_for needs [ITEM,...,]TARGET")
        refused = [] if refused is None else refused
        objects = [self.scene_object(n) for n in names]
        points = [o.aabb_center.cpu().numpy()[:2] for o in objects]
        support_z = [float(o.aabb[0][2]) for o in objects]  # each object's bottom: the top of what it stands on
        ignore = [self.scene_object(n) for n in ignore_names] + [self.objects[l] for l in (self.grasped_labels() or {})]
        log.info(
            f"head camera at z {float(self.robot_cam.get_position_orientation()[0][2]):.2f} m sees a surface at "
            f"z {min(support_z):.2f} from {self.camera_floor_distance(min(support_z)):.2f} m ahead"
        )
        short = []  # (unfold fraction, -attempt, x, y, yaw): landing clear, the arm unfolds less than UNFOLD_MIN
        placed = False
        # Carrying, the arm is wherever the grasp left it (a fold with something in the hand is usually blocked), and
        # "how far does it unfold from here" says nothing about the stance: every stance at the toy box was refused
        # at 0% while the toy was in the hand, each toy went back on the floor, putting_away_toys 0.75 -> 0.0
        # (2026-09-23). The landing check already covers the arm and what it carries where they are.
        need = 0.0 if self.hands() else UNFOLD_MIN
        for attempt in range(8):
            best, rejected = self.best_base_pose(
                points,
                ignore=ignore,
                reaching=objects,  # the arm may rest inside what it is reaching for; the base still may not
                reach=reach,
                half_widths=[self.xy_radius(n) for n in names],
                support_z=support_z,
                avoid=avoid,
                boxes=[(o.aabb[0].cpu().numpy(), o.aabb[1].cpu().numpy()) for o in objects],
                refused=refused,
            )
            if best is None:
                if short:  # the search ran dry, but some stances let the arm part of the way out
                    break
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
            if attempt:
                try:
                    collision = self.base_placement_collision(float(x), float(y), float(yaw))
                except Exception as exc:
                    raise RuntimeError(f"cannot validate base destination: {exc}") from exc
                if collision is not None:
                    refused.append((float(x), float(y), float(yaw), collision[1]))
                    log.warning("rejecting destination before another fold: %s intersects %s", *collision[:2])
                    continue
            try:
                pose = self.place_robot(float(x), float(y), float(yaw), note=f"stand for {' + '.join(names)}",
                                        min_unfold=need)  # fmt: skip
            except BasePlacementCollision as exc:
                log.warning("%s; trying another stance", exc)
                refused.append((float(x), float(y), float(yaw), exc.obstacle))
                if exc.unfold:  # landing clear, the arm gets part of the way out
                    short.append((exc.unfold, -attempt, float(x), float(y), float(yaw)))
                continue
            placed = True
            break
        if not placed:
            if not short:
                raise RuntimeError(f"no collision-free base destination for {list(names)} after 8 candidates")
            fraction, _, x, y, yaw = max(short)  # the furthest unfold, the better-scored stance on a tie
            log.info(f"no stance for {list(names)} lets the arm unfold {UNFOLD_MIN:.0%} of the way; standing at "
                     f"({x:.2f}, {y:.2f}), where it unfolds {fraction:.0%}")  # fmt: skip
            try:
                pose = self.place_robot(x, y, yaw, note=f"stand for {' + '.join(names)}", min_unfold=fraction)
            except BasePlacementCollision as exc:
                raise RuntimeError(f"no collision-free base destination for {list(names)}: {exc}") from exc
        # Only a log line comes of this, so nothing it does is worth ending an episode for. It reads the mesh of
        # every object whose box the sight line crosses, and preparing a mesh can fail on geometry trimesh will
        # not subdivide: on 2026-09-15 a picking_up_toys episode died here with "max_iter exceeded!" at round 4,
        # having run two rounds of a possible six, because the stance search had put the robot somewhere new and
        # the ray crossed a scene object no previous run had asked about. The same rule as the missing-object
        # diagnostic in run.py: a diagnostic must never replace, or prevent, the thing it explains.
        try:
            hidden = self.hidden_from_here(names)
        except Exception as why:  # noqa: BLE001 - a diagnostic must never end a run
            log.warning(f"could not work out what stands in the head camera's way: {type(why).__name__}: {why}")
            hidden = {}
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
        folded = travel_fold_targets(here, self.planned_joints)
        if folded is None:
            return None
        # The torso comes home too, and the arm unfolds to the READY posture at the new stance, not back to wherever
        # it was: after a place on a high shelf the torso stood raised (head camera 1.53 m), the unfold from there
        # swept the wrist camera through the desk at every stance, and re_shelving_library_books spent 44 minutes
        # refusing 175 stances without attempting another book (2026-09-23). The capture drives to ready anyway.
        for i, joint in enumerate(self.planned_joints):
            if joint.startswith("torso"):
                folded[i] = float(self.q_home[i])
        blocked = self.ramp_to(
            folded,
            self.posture,
            self.last_gripper,
            TRAVEL_SETTLE_STEPS,
            note="fold for travel",
            max_vel=TRAVEL_MAX_JOINT_VEL,
        )
        self._fold_blocked = blocked is not None
        if blocked is not None:
            log.warning(f"the fold before the teleport was stopped by {blocked[0]}; the measured landing posture must pass validation")
        return [float(v) for v in self.q_home]

    def unfold_after_travel(self, targets) -> None:
        """Unfold through ramp_to's complete, measured whole-robot collision preflight."""
        if targets is None:
            return
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

    def unfold_reach(self, targets, x: float, y: float, yaw: float) -> tuple[float, list | None, str]:
        """How far toward ``targets`` the arm can unfold from its posture now at a base pose, checked before going
        there: (fraction of UNFOLD_FRACTIONS, the legs to ramp through in turn or None, what stops it further).

        Each fraction is tried straight and then elbow first (the hand rises before it travels): from the travel
        fold the hand hangs low beside the base, and the straight line out sweeps it forward at coffee-table height
        -- turning_on_radio's 0.65 m stance could not unfold even 25% straight (2026-09-22).

        A stance is not refused because the FULL unfold hits something: that refused exactly the stances the robot
        wanted, where the ready hand lands on the table the object stands on or in the object itself --
        turning_on_radio's 0.65 m stance for "left_gripper_finger_link1 intersects coffee_table", the next for
        "... intersects radio_89", and the 0.75 m one it took reached none of the grasps (2026-09-22). Nor is a
        stance taken where the arm cannot come out at all: with the arm left in the travel fold the capture's
        return-to-ready ramps are refused, two refusals switch the wrist looks off for the instance, the folded
        wrist cameras see nothing, and the planner plans from the fold (setup_a_bar, the same day)."""
        if targets is None:
            return 1.0, None, ""
        here = [float(v) for v in self.q_arm()]
        why = "nothing checked"
        for fraction in UNFOLD_FRACTIONS:
            partway = [a + fraction * (b - a) for a, b in zip(here, targets)]
            tried = []
            for name, legs in reach_candidates(self.planned_joints, here, partway, None, elbow=ELBOW):
                legs = [leg for leg in legs if np.max(np.abs(np.subtract(leg, here)), initial=0.0) > 1e-6]
                if name not in ("straight", "elbow first") or legs in tried:  # the torso does not move in an unfold
                    continue
                tried.append(legs)
                collision = self.base_placement_collision(x, y, yaw, then=legs)
                if collision is None:
                    return fraction, legs, why
                why = f"{collision[0]} intersects {collision[1]}"
        return 0.0, None, why

    def place_robot(self, x: float, y: float, yaw: float, note: str = "", unfold: bool = True,
                    min_unfold: float = 0.0) -> dict:
        """Teleport the base to a floor pose (the navigation stand-in). OmniGibson moves an object held by the
        grasp assist along with the robot, so a carried object stays in the gripper.

        The arms come in over the base for the teleport (``fold_for_travel``) and go back out as far as the way
        back is clear (``unfold_reach``); ``min_unfold`` refuses a pose where that is less than the fraction. A teleport does not sweep -- it materialises the robot wherever it
        lands -- so the landing posture is the whole of the question: with the working posture the arms sit 0.41 m
        past the base's own rectangle and 0.095 m folded.

        The fold was taken out for a day because it cost 7967 of an episode's 16946 steps, and that was the wrong
        half to remove. The cost was in ramping it at the CAPTURE speed cap of 0.6 rad/s, which is there because
        observation swings were knocking objects about -- a motion out through a scene nobody has planned. Bringing
        the arms in over the robot's own base is the opposite motion, so it runs at its own speed and costs about a
        quarter as much. Coming back out IS a motion into the room, so it is collision-checked first (2026-09-14).
        """
        unfold_to = self.fold_for_travel()
        try:
            collision = self.base_placement_collision(x, y, yaw)  # the landing posture
            fraction, partway, why = (self.unfold_reach(unfold_to, x, y, yaw) if unfold and collision is None
                                      else (1.0, None, ""))  # fmt: skip
        except Exception as exc:
            raise RuntimeError(f"cannot validate base destination: {exc}") from exc
        if collision is not None:
            raise BasePlacementCollision(
                f"base destination rejected: {collision[0]} intersects {collision[1]}", obstacle=collision[1]
            )
        if fraction < min_unfold:
            raise BasePlacementCollision(
                f"base destination rejected: the arm unfolds only {fraction:.0%} of the way there ({why} further out)",
                unfold=fraction,
            )
        self.move_base(x, y, yaw)
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
        # A fold stopped by furniture was stopped by furniture AT THE OLD STANCE. The robot has since moved, so
        # try again here before unfolding: otherwise it arrives with the arm wherever it jammed, which is how the
        # arm ends up inside the thing the new stance was chosen to reach and in front of the head camera.
        # tidying_living_room blocked the fold 12 times in one run and ran no placement round at all; its rounds
        # died on "the left arm starts inside coffee_table_osroux_0, which the planner refuses before it looks at
        # the goal" (2026-09-15). Costs one ramp at travel speed, and only on the runs that were already in
        # trouble.
        # ...but only when the obstacle was the SCENE. A third of blocked folds (209 of 657) happen with something
        # in the hand, and then the thing in the way travels with the arm: a 30 cm floor tile folded in over the
        # base meets the base. Retrying at the new stance cannot help, and measured over the first runs with the
        # retry in, 27 of its 36 attempts were blocked again. Skip it and keep the steps.
        carrying = bool(getattr(self, "held_objects", None))
        if carrying and getattr(self, "_fold_blocked", False):
            log.info(f"the fold is blocked and the hand holds {self.hands()}; it would be blocked here too, not retrying")
            self._fold_blocked = False
        if getattr(self, "_fold_blocked", False) and unfold_to is not None and self.planned_joints:
            folded = travel_fold_targets(self.q_arm(), self.planned_joints) or []
            again = self.ramp_to(folded, self.posture, self.last_gripper, TRAVEL_SETTLE_STEPS,
                                 note="fold again after the teleport", max_vel=TRAVEL_MAX_JOINT_VEL)  # fmt: skip
            log.info(
                "the fold was blocked at the old stance; "
                + (f"it is blocked here too ({again[0]})" if again is not None else "it folded here")
            )
            self._fold_blocked = False
        self.stance_ready = None
        if unfold and partway is None and unfold_to is not None:
            self.stance_ready = [float(v) for v in self.q_arm()]  # no clear unfold: captures leave the arm be
        if unfold and partway is not None:  # an opening stance stays folded: reach_plan chooses the way out
            if fraction < 1.0:
                log.info(f"unfolding {fraction:.0%} of the way to the ready posture ({why} further out)")
                self.stance_ready = list(partway[-1])
            for leg in partway:  # the legs unfold_reach found clear, straight or elbow first
                self.unfold_after_travel(leg)
        log.info(f"robot placed at ({x:.2f}, {y:.2f}) yaw {math.degrees(yaw):.0f} deg {note}")
        self.log_teleport_contacts()
        return {"x": float(x), "y": float(y), "yaw": float(yaw)}

    def right_robot(self, x: float, y: float, yaw: float) -> bool:
        """Stand a fallen or toppled robot back up at a known-good pose; whether it then settled level.

        The joints are SET (arms at the travel fold, locked joints at their posture, zero velocity), never ramped:
        after a fall the old re-stand went through fold_for_travel's collision-checked ramps on a tumbling
        articulation, and the next stance search then ran from a base frame on its back (2026-09-22)."""
        q = self.robot.get_joint_positions().clone()
        folded = travel_fold_targets([float(v) for v in self.q_arm()], self.planned_joints) or list(self.q_arm())
        if any(obj is not None for obj in self.robot._ag_obj_in_hand.values()):
            # SETting the joints leaves a grasp-assisted object where it was, and move_base then carries it at that
            # offset into the folded hand, over the base; with something held the arm stays as it is
            folded = list(self.q_arm())
        for joint, value in zip(self.planned_joints, folded):
            q[self.joint_index[joint]] = float(value)
        for joint, value in self.posture.items():
            q[self.joint_index[joint]] = float(value)
        self.robot.set_joint_positions(q, drive=False)
        self.robot.set_joint_velocities(th.zeros_like(q))
        self.move_base(x, y, yaw)
        self.hold(3, self.last_gripper)
        level, why = self.settled_level(x, y, tilt_deg=5.0, shift=0.1)
        log.warning(f"righted the robot at ({x:.2f}, {y:.2f}): " + ("level again" if level else why))
        return level

    def move_base(self, x: float, y: float, yaw: float) -> None:
        """Put the base at a floor pose. THE ONLY PLACE IN THE BRIDGE THAT MOVES THE BASE.

        Today this teleports, which is the navigation stand-in the whole bench rests on. Everything else in
        ``place_robot`` -- the travel fold, the overview camera, the blocked-fold retry, the unfold, the contact
        probe -- is what ARRIVING costs and a drive needs all of it too, so a navigation stack replaces this body
        and nothing else. The challenge's own policy contract cannot teleport at all: omnigibson/eval/r1pro.yaml
        gives the base a HolonomicBaseJointController at motor_type velocity, capped +-0.75 m/s in x and y and
        +-1.0 rad/s in yaw, so a stance can only ever be driven to. ``best_base_pose`` still chooses WHERE, and is
        pose-invariant, so it survives that change untouched.

        Three things a teleport gives callers for free that a drive does not, all of which are assumptions
        somewhere else in this file rather than here:
          * it is EXACT -- ``place_robot`` returns the requested pose, and the avoid list, ``settled_level`` and
            ``last_level`` all read it as where the robot is. A drive arrives near, not at; those four readers
            want the measured pose (``base_pose``) once that is true.
          * it is INSTANT -- no steps pass, so nothing in the scene moves while the robot travels.
          * place_robot validates the destination before calling this primitive, not a navigable route.
            Occupied endpoints raise BasePlacementCollision.
        """
        quat = T.euler2quat(th.tensor([0.0, 0.0, float(yaw)]))
        self.robot.set_position_orientation(position=th.tensor([x, y, 0.0]), orientation=quat)
        self.robot.keep_still()
        self.teleports += 1

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
        self.locked_nominal = {j: float(v) for j, v in locked.items() if "finger" not in j}
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
        self.stance_ready = None

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
        self.locked_nominal = {j: v for j, v in locked.items() if "finger" not in j}
        self.q_home = [float(v) for v in embodiment["q_home"]]
        self.stance_ready = None  # the other arm's joints: this stance's partway posture no longer applies
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
        # the orientation is loose: what matters is that the object is in the picture, not how it is held -- unless
        # the load must stay level (E-level: a plate under a pizza), which bounds the whole turn
        loose = LEVEL_TILT if self.level and self.level & {l for l, a in self.hands().items() if a == arm} else 1.2
        candidates, unreachable = [], 0
        for offset in PRESENT_OFFSETS:
            target = base + np.array([offset[0], side * offset[1], offset[2]], dtype=np.float64)
            solution = ik.solve(target, quat_xyzw, seed=seed, tolerance_pos=0.04, tolerance_rad=loose)
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
        # Lula knows no collisions and lands on one of two shoulder branches with the same hand pose. Seeded from
        # the ready posture the upper arm rolls across the head camera on the way out (bringing_in_wood 301/303:
        # left_arm_joint3 through 3.7 rad, the fingers on zed_link, 33 episodes); seeded abducted (LOOK_ARM) it
        # swings out to the side, the branch 302 reached cleanly.
        abducted = list(seed)
        abducted[1] = (1.0 if arm == "left" else -1.0) * LOOK_ARM["left_arm_joint2"]
        candidates = []
        for offset in LOOK_OFFSETS:
            eye, cam_quat = look_pose(target, shoulder.cpu().numpy(), side=1 if arm == "left" else -1, offset=offset)
            pose = link_pose_for_camera(eye, cam_quat, self.camera_in_link[arm])
            tried = []
            for branch in (seed, abducted):
                solution = ik.solve(*pose, seed=branch)
                if solution is None or any(np.allclose(solution, t, atol=1e-3) for t in tried):
                    continue
                tried.append(solution)
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
        # The swing against the robot itself does FK at every sample, about 1 s a candidate: asked of the roomiest
        # first and stopped at the first clean one, not of every candidate before ranking (3-4 s an arm, 2026-09-23).
        for touched, _, offset, solution, objects in sorted(candidates):
            struck = self.swing_collision(arm, here, solution)
            if struck:
                log.info(f"{arm} arm: look offset {offset} swings {struck[0]} into {struck[1]}; skipped")
                continue
            if touched:
                log.info(
                    f"{arm} arm: look offset {offset} is the roomiest of {len(candidates)}, and still takes the arm "
                    f"within {ARM_RADIUS} m of {objects} at {touched} point(s) on the way"
                )
            elif len(candidates) > 1:
                log.info(f"{arm} arm: look offset {offset} chosen, clear of the scene the whole way")
            return solution
        return None

    def swing_collision(self, arm: str, start, goal) -> tuple | None:
        """What ``arm`` meets of the robot itself (or what it carries) on ``ramp_arms``' elbow-first swing from
        ``start`` to ``goal``, the rest of the robot where it is: (link, obstacle), else None."""
        joints = list(self.robot.arm_joint_names[arm])
        via = list(start)
        via[ELBOW] = float(goal[ELBOW])
        try:
            for leg_start, leg_goal in ((start, via), (via, goal)):
                hit = self.ramp_collision(joints, leg_start, leg_goal, scene=False)
                if hit is not None:
                    return hit[:2]
        except Exception as exc:  # noqa: BLE001 - the ramp's own preflight refuses what it cannot check
            log.warning(f"cannot check the {arm} arm's swing against the robot: {exc}")
        return None

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
        points = R1ProSim._link_points(self, ik, q, names, at)
        if getattr(self, "send_room", False):  # what the hand holds is part of the arm here too
            points += [R1ProSim._to_world(self, p, at) for p in R1ProSim.held_points(self, arm, ik, q)]
        return points

    def held_corners(self, arm: str, label: str, ik: ArmIK) -> np.ndarray:
        """The 8 corners of what ``arm`` holds, in the gripper's frame. The grasp is rigid, so this is read once."""
        key = (arm, label)
        if key not in self._held_boxes:
            lo, hi = (v.cpu().numpy() for v in self.objects[label].aabb)
            identity = th.tensor([0.0, 0.0, 0.0, 1.0])
            corners = np.stack(
                [
                    self.to_base(th.tensor([x, y, z], dtype=th.float32), identity)[0].cpu().numpy()
                    for x in (float(lo[0]), float(hi[0]))
                    for y in (float(lo[1]), float(hi[1]))
                    for z in (float(lo[2]), float(hi[2]))
                ]
            )
            now = self.robot.get_joint_positions()
            q_now = [float(now[self.joint_index[j]]) for j in self.robot.arm_joint_names[arm]]
            pos, quat = ik.fk(q_now, f"{arm}_gripper_link")
            rot = T.quat2mat(th.tensor(np.asarray(quat), dtype=th.float32)).cpu().numpy()
            self._held_boxes[key] = (corners - np.asarray(pos, dtype=np.float64)) @ rot
        return self._held_boxes[key]

    def held_points(self, arm: str, ik: ArmIK, q) -> list[np.ndarray]:
        """Where the corners of what ``arm`` holds would be at joints ``q``, in the BASE frame.

        Five places drop the held object from the obstacle list, all correctly -- it is not scene furniture. The
        other half, putting it back onto the ROBOT, was never written, so every carrying ramp swung an invisible
        object through the room. Corners are appended to an ordered polyline, and the segments drawn between them
        lie inside the box's own hull, so the approximation errs towards saying "clear", never the reverse.
        """
        points = []
        for label, hand in self.hands().items():
            if hand != arm or label not in self.objects:
                continue
            try:
                corners = R1ProSim.held_corners(self, arm, label, ik)
                pos, quat = ik.fk(q, f"{arm}_gripper_link")
                rot = T.quat2mat(th.tensor(np.asarray(quat), dtype=th.float32)).cpu().numpy()
                points.extend(np.asarray(pos, dtype=np.float64) + corners @ rot.T)
            except Exception:  # noqa: BLE001 - no FK for this frame: fall back to the blind check
                continue
        return points

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
            points.append(R1ProSim._to_world(self, pos, at))
        return points

    def _to_world(self, p, at=None) -> np.ndarray:
        """A base-frame point in the world frame, at the base's current pose or at an (x, y, yaw) stance."""
        p = np.asarray(p, dtype=np.float64)
        if at is None:
            return np.asarray(self.base_to_world(p), dtype=np.float64)
        x, y, yaw = at
        c, sn = math.cos(float(yaw)), math.sin(float(yaw))
        return np.array([x + c * p[0] - sn * p[1], y + sn * p[0] + c * p[1], p[2]], dtype=np.float64)

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
                physical = self.collision_mesh_world(obj)
                if physical is None:  # visual-only geometry has no physical surface
                    continue
                # Exact cached triangle BVH queries handle long floor/wall triangles without subdivision.
                if bool((trimesh.proximity.closest_point(physical, samples)[1] <= clearance).any()):
                    hits.append(obj.name)
            except Exception:
                hits.append(obj.name)  # extraction/query failure: conservatively keep the box's word
        return hits

    def nearby_obstacles(
        self, exclude=(), reach: float = OBSTACLE_REACH, limit: int = OBSTACLE_LIMIT, *, collision_map: bool = False
    ) -> list[str]:
        """Register nearby scene bodies, using a complete physical map for ``--room``.

        The legacy ``--obstacles`` route reconstructs a small furniture set from masks; it is not a collision
        map. The physical map includes target containers, supports, small objects, merged walls and objects
        whose AABB encloses the robot. An AABB is only a broad-phase filter: being inside it does not justify
        erasing its mesh. Floors are excluded because wheel contact is expected; raised mats and rugs remain.
        """
        if collision_map:
            base = self.base_pose()[0]
            base = base.cpu().numpy() if hasattr(base, "cpu") else np.asarray(base, dtype=np.float64)
            excluded = set(exclude) | {self.robot.name}
            rows = []
            for obj, lo, hi in self.scene_aabbs():
                if obj is self.robot or obj.name in excluded or obj.category == "floors":
                    continue
                if lo[2] > base[2] + reach or hi[2] < base[2] - 0.1:
                    continue
                gap = float(np.linalg.norm(np.clip(base[:2], lo[:2], hi[:2]) - base[:2]))
                if gap <= reach:
                    rows.append((gap, obj.name, obj))
            rows.sort(key=lambda row: (row[0], row[1]))
            # No count cap: omitted obstacles would silently make an otherwise valid plan unsafe. The planner
            # must size its collision cache for this map or explicitly refuse the request.
            self.obstacles = {name: obj for _, name, obj in rows}
            return list(self.obstacles)
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
        rows_all = list(self.scene_aabbs())
        # TRIED AND REVERTED 2026-09-18: also shipping the task's own fixtures as ground-truth geometry, on the
        # theory that a body the goal never MOVES is one to plan around. It cost more than it bought and it never
        # bought anything: dispose_of_batteries improved with the room on, but everything it shipped (cubicles,
        # cabinets, chairs, walls) is plain scene furniture that was shipped before the rule existed. What the
        # rule added was the goal's own containers -- the wicker baskets, the toy box -- and each of those
        # refused the very approach the round needed. Spare everything tracked; the room is furniture.
        # What the task's objects STAND on is perception's job, not ours: the planner fits it as a slab and samples
        # every placement on that slab's top. Ship the real surface as a static too and every Place particle is
        # inside an obstacle, with no collision message anywhere -- the round just dies with no satisfying particle.
        stands_on = [(lo, hi) for obj, lo, hi in rows_all if id(obj) in tracked]
        base_span = float(np.abs(self.base_box()[:, :2]).max())  # circumscribes the base's own box at any yaw
        rows = []
        for obj, lo, hi in rows_all:
            if obj is self.robot or obj.name in spared or id(obj) in tracked or obj.category in FLOOR_COVERINGS:
                continue
            if (hi[0] - lo[0]) * (hi[1] - lo[1]) > HOUSE_AABB_AREA:
                continue  # merged walls, roofs, ceilings
            if lo[2] > ROBOT_HEIGHT:
                continue  # entirely overhead
            extent = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
            if float(np.max(extent)) < OBSTACLE_MIN_SIZE:
                continue  # small enough to be something to pick up, not a fixture to plan around
            if any(
                hi[2] <= o_lo[2] + STANDS_ON_TOL
                and hi[0] > o_lo[0]
                and lo[0] < o_hi[0]
                and hi[1] > o_lo[1]
                and lo[1] < o_hi[1]
                for o_lo, o_hi in stands_on
            ):
                continue  # a tracked object stands on it: perception models this as the support slab
            # Distance to the BOX, not to its centre. The wall and the sofa the robot is standing against have
            # their centres metres away: in store_honey the walls and the sofa ranked ninth and tenth by centre
            # distance, past ``limit``, while their surfaces were 0.00 m from the base. Anything nearer than the
            # base's own box is something the base is standing IN -- shipping it makes every start state invalid,
            # because cuRobo's world holds base_link, the wheels, the torso and both arms.
            gap = float(np.linalg.norm(np.clip(here, lo[:2], hi[:2]) - here))
            if base_span < gap <= reach:
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
                return False  # a settling hold would repeat the gripper command the preflight may have rejected
        return self.ramp_to(q_arm, posture, gripper, settle_steps, note) is None

    def _motion_collision_model(self):
        from omnigibson.tiptop.collision import JointPathCollision

        if not hasattr(self, "_ramp_collision_model"):
            urdf = Path(self.robot.urdf_path)
            config = urdf.parent.parent / "curobo" / "r1pro_description_curobo_arm_no_torso.yaml"
            joints = [name for name in self.joint_index if name in self.urdf_joints]
            self._ramp_collision_model = JointPathCollision(urdf, config, joints, self.robot.disabled_collision_pairs)
        return self._ramp_collision_model

    def _motion_finger_ranges(self, measured, gripper):
        """Actual-to-commanded finger travel, independent of arm interpolation timing."""
        ranges = {}
        for arm, command in ((self.arm, gripper), (self.other_arm, self.other_gripper)):
            for name in self.robot.finger_joint_names[arm]:
                joint = self.robot.joints[name]
                fraction = float(np.clip((command + 1.0) / 2.0, 0.0, 1.0))
                goal = float(joint.lower_limit) + fraction * float(joint.upper_limit - joint.lower_limit)
                start = float(measured[self.joint_index[name]])
                if abs(start - goal) > 1e-6:
                    ranges[name] = (start, goal)
        return ranges

    def _motion_obstacles(self, held_ids, allowed_contacts, base=None, only=None):
        """Physical world near the current or requested base, with floor contact only for chassis and wheels.
        ``only``: names to restrict it to (the obstacles a landing check already named, re-checked per stance)."""
        obstacles = []
        base = self.base_pose()[0].cpu().numpy() if base is None else np.asarray(base)
        for obj, lo, hi in self.scene_aabbs():
            if obj is self.robot or id(obj) in held_ids or (only is not None and obj.name not in only):
                continue
            if np.linalg.norm(np.clip(base, lo, hi) - base) > OBSTACLE_REACH:
                continue
            mesh = self.collision_mesh_world(obj)
            if mesh is not None:
                obstacles.append((obj.name, mesh))
                if obj.category == "floors":
                    allowed_contacts.setdefault(obj.name, set()).update(
                        {"base_link", "wheel_motor_link1", "wheel_motor_link2", "wheel_motor_link3"}
                    )
        return obstacles

    def own_box(self, obj):
        """The object's box in ITS OWN frame, from the depth points the captures saw it by (``seen_boxes``, never
        its simulator mesh): (world position, rotation matrix, lo, hi); None when no capture has seen it."""
        label = next((label for label, o in self.objects.items() if o is obj), None)
        box = self.seen_boxes.get(label)
        if box is None:
            return None
        pos, quat = obj.get_position_orientation()
        return pos.cpu().numpy().astype(np.float64), T.quat2mat(quat).cpu().numpy().astype(np.float64), *box

    def item_height(self, name: str) -> float:
        """How tall an item stands, from the capture's points (its own box projected on the vertical); 0 when no
        capture has seen it (its mesh is not ours to read), so a board then needs HEADROOM alone."""
        box = self.own_box(self.scene_object(name))
        if box is None:
            log.info(f"{name}: no capture has seen it; its height is unknown to the board choice")
            return 0.0
        _, rot, lo, hi = box
        return float(np.abs(rot[2]) @ (hi - lo))

    def carried_volume(self, obj, arm: str) -> tuple:
        """The held object's sphere cover riding on the arm's gripper link: (link, centres, radii), gripper frame.

        Built on its OWN box, not the gripper-frame AABB of the same points: that is the box of a box turned
        45 deg. A 0.45 x 0.42 m tile at 132 deg to the jaw became 0.61 x 0.62 m and "intersected" base_link it
        cleared by 3 cm, and every motion after the grasp was refused at sample 0 (laying_tile 301, 2026-09-23).
        An object no capture has seen falls back to the gripper-frame box of its simulator mesh, as before.
        """
        from omnigibson.tiptop.collision import box_spheres

        link = f"{arm}_gripper_link"
        gpos, gquat = self.robot.links[link].get_position_orientation()
        gpos, grip = gpos.cpu().numpy().astype(np.float64), T.quat2mat(gquat).cpu().numpy().astype(np.float64)
        box = self.own_box(obj)
        if box is None:
            mesh = self.collision_mesh_world(obj)
            if mesh is None:
                raise ValueError(f"held object {obj.name!r} has no physical mesh")
            local = (np.asarray(mesh.vertices) - gpos) @ grip
            box = gpos, grip, local.min(axis=0), local.max(axis=0)
        pos, rot, lo, hi = box
        centres, radii = box_spheres(np.stack([lo, hi]))
        return link, (centres @ rot.T + pos - gpos) @ grip, radii

    def base_placement_collision(self, x, y, yaw, then=None, only=None):
        """Check the measured landing posture and live carried volume before any base teleport, and with ``then``
        (planned-joint targets) the straight unfold from it to them as well; ``only`` restricts the room to the
        obstacles named (the stance search re-asking about the ones that refused an earlier candidate).

        ``then`` is how unfold_reach finds how far the arm may unfold at a stance:
        dispose_of_batteries landed 0.14 m from a desk, the unfold was refused (left_arm_link6 through the desk),
        and every round then planned from a folded arm jammed against it (2026-09-22) -- the landing check with
        TELEPORT_CLEARANCE is what keeps a folded arm off the furniture."""
        from omnigibson.tiptop.collision import CARRIED

        model = self._motion_collision_model()
        measured = self.robot.get_joint_positions()
        full = [float(measured[self.joint_index[name]]) for name in model.joint_names]
        destination = np.array([x, y, 0.0], dtype=np.float64)
        if not np.isfinite(destination).all() or not np.isfinite(yaw):
            raise ValueError("base destination must be finite")
        c, s = math.cos(yaw), math.sin(yaw)
        transform = np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        held_ids, attachments = set(), []
        for arm, obj in self.robot._ag_obj_in_hand.items():
            if obj is None:
                continue
            if obj.fixed_base:
                raise ValueError(f"cannot teleport while grasping fixed object {obj.name}")
            attachments.append(self.carried_volume(obj, arm))
            held_ids.add(id(obj))
        path = [full]
        waypoints = [] if then is None else (list(then) if np.ndim(then) == 2 else [then])  # one target or legs
        for waypoint in waypoints:
            unfolded = list(path[-1])
            for name, value in zip(self.planned_joints, waypoint):
                unfolded[model.joint_names.index(name)] = float(value)
            path.append(unfolded)
        allowed_contacts = {}
        obstacles = self._motion_obstacles(held_ids, allowed_contacts, base=destination, only=only)
        # TELEPORT_CLEARANCE is room from walls and furniture; the floor under the robot gets none extra: a hand
        # low after a floor pick was 2 cm from the floor at every destination, and every stance was refused
        # (laying_tile_floors, 2026-09-22)
        # ground by geometry, not by name: pavers and lawns are ground too, and a base 3 cm "into" the paver refused
        # every outdoor stance (bringing_in_wood 1.0 -> 0.33, 2026-09-23)
        floors = {obj.name: getattr(obj, "fixed_base", False) for obj, lo, hi in self.scene_aabbs()  # name -> the map's
                  if getattr(obj, "category", None) == "floors" or float(hi[2]) < GROUND_TOP}  # fmt: skip
        hit = model.check_polyline(
            path, transform, [o for o in obstacles if o[0] not in floors], allowed_contacts, attachments=attachments,
            clearance=TELEPORT_CLEARANCE,
        )
        ground = [o for o in obstacles if o[0] in floors]
        if hit is None and ground:
            hit = model.check_polyline(path, transform, ground, allowed_contacts, attachments=attachments)
        if hit is not None and any(hit[1] == name for name, _ in ground):
            # The hand, or what it carries, is at the floor HERE too: a plank picked off it leaves the fingers within
            # the sampling inflation of the floor, and a teleport of this same posture deepens nothing. Refusing
            # every destination for it froze the episode (48 corridor stances, bringing_in_wood 302, 2026-09-23).
            # ponytail: no depth comparison across floors; ground a kerb higher at the destination is not caught
            hand = {f"{self.arm}_gripper_link", *self.robot.finger_link_names[self.arm], CARRIED}
            here, allowed_here = T.pose2mat(self.base_pose()).cpu().numpy(), {}
            ground_here = [o for o in self._motion_obstacles(held_ids, allowed_here, only=only) if o[0] in floors]
            now = model.check_polyline(path[:1], here, ground_here, allowed_here, attachments=attachments)
            if now is not None and (now[0] in hand or now[0].startswith("attached_object")):
                for name, fixed in floors.items():
                    if fixed:  # a loose tile lying flat is ground for the base's clearance, not floor for the hand
                        allowed_contacts.setdefault(name, set()).update(hand)
                hit = model.check_polyline(path, transform, ground, allowed_contacts, attachments=attachments)
        return hit

    def validate_motion(self, positions, gripper, label):
        """Robot-only preflight for planned polylines and gripper events; never steps the simulator.

        TiPToP checks its carried-object model; this redundant check uses measured fingers and the complete
        current robot state. Live grasp-assist state, rather than the episode's delayed hand ledger, determines
        which physical bodies travel with the robot. Only active fingers may contact an explicit action target.
        """
        try:
            model = self._motion_collision_model()
            measured = self.robot.get_joint_positions()
            positions = np.asarray(positions, dtype=np.float64)
            if positions.ndim == 1:
                positions = positions[None, :]
            if positions.ndim != 2 or positions.shape[1] != len(self.planned_joints) or len(positions) == 0:
                raise ValueError("motion joint shape does not match the robot embodiment")
            full = np.tile([float(measured[self.joint_index[name]]) for name in model.joint_names], (len(positions), 1))
            columns = [model.joint_names.index(name) for name in self.planned_joints]
            full[:, columns] = positions
            held_ids = {id(obj) for obj in self.robot._ag_obj_in_hand.values() if obj is not None}
            allowed_contacts = {}
            action = re.match(r"^(Pick|Place|PlaceNear|Push)\(\s*([^,)]+)", label or "")  # PlaceNear: inside rounds
            if action:
                name = action[2].strip()
                obj = self.objects.get(name)
                if obj is None and action[1] == "Push":
                    obj = self.objects.get(name.removesuffix("_button"))
                if obj is not None:
                    fingers = set(self.robot.finger_link_names[self.arm])
                    allowed_contacts[obj.name] = fingers
                    if action[1] == "Pick":
                        # the planner lets the fingers meet what the object rests on at the grasp (cutamp-19: a thin
                        # eraser's desk), where the sphere model has them a few mm in; a finger at the tote refused
                        # Pick(paintbrush_1) at execution (organizing_art_supplies, sweep4 2026-09-24)
                        for name in self.rests_against(obj):
                            allowed_contacts.setdefault(name, set()).update(fingers)
            obstacles = self._motion_obstacles(held_ids, allowed_contacts)
            # The planner already inflated these same spheres by the model's buffer; counting it again here refused
            # a Place whose forearm sphere was 0.4 mm clear of a trash can (1.6 mm "inside" once re-buffered,
            # dispose_of_batteries 2026-09-22). Audit planned motion against the raw spheres, keeping the sampling
            # inflation; bridge-owned ramps, which nothing planned, keep the buffer.
            return model.check_polyline(
                full,
                T.pose2mat(self.base_pose()).cpu().numpy(),
                obstacles,
                allowed_contacts=allowed_contacts,
                joint_ranges=self._motion_finger_ranges(measured, gripper),
                clearance=-model.buffer,
                excuse_start=True,
            )
        except Exception as exc:
            log.warning("cannot validate planned motion %s: %s", label, exc)
            return f"motion validation unavailable: {type(exc).__name__}: {exc}"

    def ramp_collision(self, names, start, goal, allowed_contacts=None, gripper=None, scene=True):
        """Preflight the actual whole-robot path, including the torso, opposite arm, cameras and held objects.
        ``scene=False`` judges the robot against itself and what it carries only (a look configuration's swing,
        chosen before the scene is consulted; the ramp's own preflight still sees the room)."""
        model = self._motion_collision_model()
        measured = self.robot.get_joint_positions()
        here = {name: float(measured[self.joint_index[name]]) for name in model.joint_names}
        initial, target = dict(here), dict(here)
        initial.update(zip(names, start))
        target.update(zip(names, goal))
        finger_ranges = self._motion_finger_ranges(measured, gripper) if gripper is not None else {}
        held = self.hands()
        held_ids = {id(self.objects[label]) for label in held if label in self.objects}
        allowed_contacts = {name: set(links) for name, links in (allowed_contacts or {}).items()}
        obstacles = self._motion_obstacles(held_ids, allowed_contacts) if scene else []
        attachments = []
        for label, arm in held.items():
            if label not in self.objects:
                raise ValueError(f"no collision geometry for held object {label!r}")
            attachments.append(self.carried_volume(self.objects[label], arm))
        world_from_base = T.pose2mat(self.base_pose()).cpu().numpy()
        return model.check(
            [initial[name] for name in model.joint_names],
            [target[name] for name in model.joint_names],
            world_from_base,
            obstacles,
            allowed_contacts=allowed_contacts,
            attachments=attachments,
            joint_ranges=finger_ranges,
            excuse_start=True,
        )

    @staticmethod
    def grasp_contacts(arm, obj):
        """Only the grasping gripper/fingers may touch the named manipulation target."""
        return {obj.name: {f"{arm}_gripper_link", f"{arm}_gripper_finger_link1", f"{arm}_gripper_finger_link2"}}

    def ramp_to(
        self,
        q_arm,
        posture: dict,
        gripper: float,
        settle_steps: int,
        note: str = "",
        max_vel: float | None = None,
        leashed: bool = True,
        allowed_contacts: dict | None = None,
        stop_on_contact: str | None = None,
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
        joints followed, else (joint, step, lag); a preflight refusal is (what intersects what, 0, 0.0, path sample).

        The command is leashed to within ``EXEC_LEASH`` of where the joints actually are -- the same bound every
        planned segment already gets in ``b1k.bridge.executor`` -- so the drive's torque is bounded while the ramp
        runs on. A travel-speed ramp into furniture used to command 0.47 rad past the arm before the block fired.
        Detection is unchanged, and the leash is provably inert on a healthy ramp: it can only bite when
        ``|q - measured| > EXEC_LEASH``, and ``EXEC_LEASH == RAMP_BLOCK_TOL``, so every step it clips is a step that
        already counts toward ``RAMP_BLOCK_STEPS``. Five in a row and the ramp stops and holds at the measured
        posture, so it can throttle for at most four steps before the ramp reports the truth. ``leashed=False`` is
        for the two ramps whose purpose IS to load a joint (the grasp press, the drawer pull)."""
        # Stop at first contact, before the simulator's 0.3 s sticky-grasp window completes. This option is
        # confined to the short, closed-hand terminal approach; free-space transit has no target exemption.
        if stop_on_contact is not None:
            try:
                touching, _ = self.robot._find_gripper_contacts(arm=stop_on_contact)
                if touching or self.robot._ag_obj_in_hand.get(stop_on_contact) is not None:
                    return ("finger contact", 0, 0.0)
            except Exception as exc:
                log.warning("refusing contact-seeking ramp without contact observations: %s", exc)
                return ("contact observation unavailable", 0, 0.0)
        now = self.robot.get_joint_positions()
        names = list(self.planned_joints) + list(posture)
        start = [float(now[self.joint_index[j]]) for j in names]
        goal = [float(v) for v in q_arm] + [float(posture[j]) for j in posture]
        # This runs before the first control step. Lag remains an execution monitor, not collision planning.
        try:
            moving = np.max(np.abs(np.asarray(goal) - np.asarray(start)), initial=0.0) > 1e-4
            collision = None
            if moving or self._motion_finger_ranges(now, gripper):
                collision = self.ramp_collision(names, start, goal, allowed_contacts, gripper)
        except Exception as exc:
            log.warning("refusing unvalidated ramp %s: %s", note, exc)
            return (f"collision validation unavailable: {type(exc).__name__}", 0, 0.0)
        if collision is not None:
            link, obstacle, sample = collision
            log.warning("refusing ramp %s: %s intersects %s at path sample %d", note, link, obstacle, sample)
            return (f"{link} intersects {obstacle}", 0, 0.0, sample)  # the path sample too: at 0 the start itself
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
        lead = 0.0  # the furthest ahead of the arm this ramp ever commanded: what the leash bounds
        for i, q in enumerate(path):
            self.posture = {j: float(v) for j, v in zip(posture, q[k:])}
            cmd = leash(q[:k], last[:k]) if leashed else np.asarray(q[:k], dtype=np.float64)
            lead = max(lead, float(np.abs(np.asarray(cmd) - last[:k]).max()) if k else 0.0)
            self.step(cmd, gripper)
            measured = self.robot.get_joint_positions()[idx].cpu().numpy().astype(np.float64)
            rates = np.abs(measured - last) / self.dt
            if rates.max() > fastest:
                j = int(rates.argmax())
                fastest = float(rates[j])
                culprit = f" ({names[j]} at step {i + 1}: {last[j]:+.3f} -> {measured[j]:+.3f} rad, target {q[j]:+.3f})"
            lag = np.where(ramped, np.abs(measured - q), 0.0)
            last = measured
            if stop_on_contact is not None:
                try:
                    touching, _ = self.robot._find_gripper_contacts(arm=stop_on_contact)
                    if touching or self.robot._ag_obj_in_hand.get(stop_on_contact) is not None:
                        blocked = ("finger contact", i + 1, 0.0)
                        break
                except Exception as exc:
                    log.warning("stopping contact-seeking ramp without contact observations: %s", exc)
                    blocked = ("contact observation unavailable", i + 1, 0.0)
                    break
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
            if blocked[0] == "finger contact":
                log.info("stopping %s at first finger contact (step %d)", note, blocked[1])
            else:
                # WHAT it met, not just which joint lagged. Every block was unattributable until now, so 31 of them in
                # one episode said nothing about where to look. Probed at the posture the arm actually stopped in.
                try:
                    probe_arm = self.arm if self.arm in self.robot.arm_names else self.robot.arm_names[0]
                    stopped_at = self.robot.get_joint_positions()
                    struck = self.arm_hits_scene(
                        probe_arm,
                        self._stance_ik(probe_arm),
                        [float(stopped_at[self.joint_index[j]]) for j in self.robot.arm_joint_names[probe_arm]],
                    )
                except Exception:  # noqa: BLE001 - a diagnostic must never end a ramp
                    struck = []
                log.warning(
                    f"{blocked[0]} stopped following the ramp at step {blocked[1]} of {len(path)} ({blocked[2]:.2f} rad "
                    f"behind its target for {RAMP_BLOCK_STEPS} steps): the arm is pushing against something, so the "
                    f"ramp stopped there [motion: {note or 'unnamed'}, env step {self.n_steps}, "
                    f"in the way: {struck or 'nothing the box test sees'}]"
                )
            held = self.robot.get_joint_positions()
            self.posture = {j: float(held[self.joint_index[j]]) for j in posture}
            q_hold = [float(held[self.joint_index[j]]) for j in self.planned_joints]
            # Even a zero-settle waypoint must replace the last drive target after a tracking abort. Keep the
            # gripper command already sent during this ramp; preflight refusals return before any motor call.
            settle_steps = max(1, settle_steps)
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
            f"{fastest:.2f} rad/s measured{culprit}, worst command lead {lead:.3f} rad"
            f"{'' if leashed else ' (unleashed)'}, then {settle_steps} settle steps"
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

    def restore_locked_arm(self) -> None:
        """Drive an empty locked arm back to the posture the planner models it at.

        A blocked ramp keeps whatever the arm was pushed to (that is what stops it pressing into furniture), and
        nothing ever brought it back: the planner refuses a request whose locked joints differ from its model, so
        one bump failed every later round of the episode. A holding arm, and the torso, stay where they are."""
        nominal = getattr(self, "locked_nominal", None) or {}
        if not nominal:
            return
        holding = set(self.hands().values())
        joints = {j: v for j, v in nominal.items()
                  if not j.startswith("torso") and not any(j.startswith(f"{arm}_") for arm in holding)}
        measured = self.robot.get_joint_positions()
        drift = {j: abs(float(measured[self.joint_index[j]]) - v) for j, v in joints.items() if j in self.joint_index}
        if not drift or max(drift.values()) < LOCKED_DRIFT:
            return
        worst = max(drift, key=drift.get)
        # Straight, then the elbow alone first (the hand rises before it travels), then the elbow last: the straight
        # line back was refused by the shoe in the other hand (putting_shoes_on_rack 302) or stopped on the other
        # arm (assembling_gift_baskets 302), and nothing else was tried; 8 and 2 rounds refused (2026-09-23).
        goal = {**self.posture, **joints}
        arms = {arm: self.robot.arm_joint_names[arm] for arm in self.robot.arm_names}
        for elbow_first in (None, True, False):
            q = self.robot.get_joint_positions()
            now = {j: float(q[self.joint_index[j]]) for j in goal}
            legs = [goal]
            if elbow_first is not None:
                legs.insert(0, via_configuration(list(goal), now, goal, arms, ELBOW, elbow_first))
            for leg in legs:
                stopped = self.ramp_to(self.q_arm(), leg, self.last_gripper, TRAVEL_SETTLE_STEPS,
                                       note="locked arm back to its modelled posture", max_vel=TRAVEL_MAX_JOINT_VEL)
                if stopped is not None:
                    break
            if stopped is None:
                break
        log.info(f"{worst} was {drift[worst]:.3f} rad off the planner's model; "
                 + ("returned it" if stopped is None else f"could not return it ({stopped[0]})"))

    def tuck_idle_arm(self) -> bool:
        """The idle arm, when it holds nothing, into the posture its own planner works from (elbow bent, the hand
        up beside the torso): hanging at its locked posture it rides the torso's lean into what the working arm
        reaches for. False when it holds something or the ramp is refused; ``restore_locked_arm`` brings it back."""
        if self.other_arm in self.hands().values():
            return False
        meta = load_embodiment_meta(f"r1pro_{self.other_arm}")
        joints = set(self.robot.arm_joint_names[self.other_arm])
        tucked = {j: float(v) for j, v in zip(meta["joint_names"], meta["q_home"]) if j in joints}
        stopped = self.ramp_to(self.q_arm(), {**self.posture, **tucked}, self.last_gripper, TRAVEL_SETTLE_STEPS,
                               note=f"tuck the idle {self.other_arm} arm")  # fmt: skip
        return stopped is None

    def capture(self, task: str) -> tuple[dict, dict]:
        """Capture task perception, then attach the measured start and complete collision map on every exit."""
        self.restore_locked_arm()
        request, extras = self._capture_with_motion(task)
        request["q_init"] = np.asarray(self.q_arm(), dtype=np.float32)
        measured = self.robot.get_joint_positions()
        request["locked_joints"] = {
            name: float(measured[self.joint_index[name]])
            for name in self.posture
            if name not in self.planned_joints
        }
        if self.send_room:
            self.nearby_obstacles(collision_map=True)
            request["room"] = self.room_collision_scene()
        return request, extras

    def _capture_with_motion(self, task: str) -> tuple[dict, dict]:
        """Every view in one posture: each free arm whose wrist camera is a view points it at the look target
        (``wrist_look``: what the base pose was chosen for, at the hand that holds it once it has been picked up;
        a held object otherwise, when nothing was stood for), which also takes the arm out of the head camera's
        frame. An arm that holds something stays where it is:
        the held object is what the next plan is about and must be seen, and the gripper keeps its command. The
        planned arm swings out of view (``look_arm``) when no look configuration exists; with ``look_arm`` None
        nothing moves. The head views come last, the torso turned to their yaw with the arms as they are
        (``_capture_views``). The plan starts from the ready posture the arms return to."""
        # at a stance where the arm unfolded only partway, q_home is exactly what it could not reach: returning to
        # it after the looks was refused every capture, and two refusals switched the wrist looks off for the rest
        # of the instance (setup_a_bar, 2026-09-22)
        if self.stance_ready is not None:
            ready = list(self.stance_ready)
        else:
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
            log.warning("capture motion was refused or interrupted; capturing from the measured posture")
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
        self.restore_locked_arm()  # now, not a round later: the request's locked joints must be the model's
        extras["q_look"] = moved
        log.info(
            f"captured with {sorted(moved)} arm(s) posed; plan starts from the ready posture (max error {lag:.4f} rad)"
        )
        return request, extras

    def press_grasp(self, arm: str, name: str, spare=()) -> bool:
        """Reach a collision-free standoff, close, then make slow finger contact for a sticky grasp.

        M2T2 proposes grasps from the point cloud, and a book lying flat on a table gives it nothing usable --
        there is no side a parallel jaw can get under. The round then fails before the arm moves, which reads as a
        planning failure rather than as "this cannot be grasped that way". The user's instruction (2026-09-14):
        close the gripper on the book and let the assisted grasp take it.

        The hand closes first, in free space, and the closed-hand transit permits no target contact. Only the
        short terminal approach permits the active fingers to touch the named target; its arm advance stops at
        first contact while the assist establishes the hold. Physical supports, the palm, cameras and both arms
        remain collision checked throughout.
        """
        obj = self.scene_object(name)
        lo, hi = (v.cpu().numpy().astype(np.float64) for v in obj.aabb)
        extent = hi - lo
        # No thickness gate. It used to return here for anything thicker than FLAT_THICKNESS, and refused 366
        # attempts across the corpus -- pillows 48, a tissue dispenser 21, soda cans 37, the fax machine 14. By
        # the time this runs the PLANNER HAS ALREADY FAILED on this object, so "a fallback for the shape M2T2
        # cannot serve, not a replacement for it" is a distinction without a difference: there is nothing else
        # left to try. And sticky grasping needs no flatness at all -- OmniGibson skips both the antipodal
        # raycast and the two-finger requirement in that mode, so one finger on a pillow welds exactly as one
        # finger on a book (2026-09-15).
        flat = float(np.min(extent)) <= FLAT_THICKNESS
        if not flat:
            log.info(f"{name} is {np.round(extent, 3).tolist()} m, thicker than the shape this was written for; "
                     f"pressing anyway because the planner has already failed on it")  # fmt: skip
        top = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, float(hi[2])], dtype=np.float64)
        unit = th.tensor([0.0, 0.0, 0.0, 1.0])
        try:
            mesh = self.collision_mesh_world(obj)
            if mesh is None:
                return False
        except Exception as exc:
            log.warning("cannot read sticky contact geometry for %s: %s", name, exc)
            return False

        def surface(origin, direction):
            hits = mesh.ray.intersects_location([origin], [direction])[0]
            distances = (hits - origin) @ direction
            hits, distances = hits[distances > 0], distances[distances > 0]
            return hits[int(np.argmin(distances))] if len(hits) else None

        # A hollow bowl's AABB top is empty. Seek a real surface, not an invented lid above its cavity.
        try:
            top_hit = surface(top + [0.0, 0.0, STICKY_STANDOFF], np.array([0.0, 0.0, -1.0]))
        except Exception as exc:
            log.warning("cannot find a physical top contact for %s: %s", name, exc)
            return False
        ik = self.arm_ik(arm, frame=f"{arm}_gripper_link", with_torso=True)
        joints_of = self.ik_joint_names(arm, with_torso=True)
        q = self.robot.get_joint_positions()
        seed = [float(q[self.joint_index[j]]) for j in joints_of]
        aabbs = self.scene_aabbs()
        base_pos, base_quat = self.base_pose()
        world_to_base = T.quat2mat(base_quat).cpu().numpy().astype(np.float64).T
        down = world_to_base @ np.array([0.0, 0.0, -1.0])
        # The AABB axes are world-frame directions; rotate both jaw options into the IK's base frame.
        across_world = np.array([1.0, 0.0, 0.0]) if extent[0] <= extent[1] else np.array([0.0, 1.0, 0.0])
        jaws_above = (world_to_base @ across_world, world_to_base @ np.array([across_world[1], across_world[0], 0.0]))

        # Two ways in, tried in order. FROM ABOVE is the original, and the right one for anything lying on a
        # surface. FROM THE FRONT is for a book standing in a shelf, where above is exactly where the next shelf
        # is: boxing_books_up_for_storage refused every attempt with "the way down to it sweeps through
        # bookcase_otwukr_2", and with the mesh check in place that refusal is CORRECT -- the arm really would go
        # through the shelf. So come in horizontally at the object's mid-height, onto the face nearest the robot,
        # which is how a person takes a book off a shelf. Sticky grasping needs one finger in contact, not a jaw
        # around the whole width (2026-09-15).
        middle = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0], dtype=np.float64)
        ways = []
        if top_hit is not None:
            top_base = self.to_base(th.tensor(top_hit, dtype=th.float32), unit)[0].cpu().numpy()
            ways.append(("from above", top_base, down, jaws_above))
        origin = base_pos.cpu().numpy().astype(np.float64).copy()
        origin[2] = middle[2]
        toward = middle - origin
        span = float(np.linalg.norm(toward))
        if span > 1e-6:
            into_world = toward / span
            # A projected AABB radius is not the ray's near surface, and cannot be combined with a base-frame
            # direction. Intersect the actual physical mesh in world coordinates before transforming the hit.
            try:
                contact = surface(origin, into_world)
            except Exception as exc:
                log.warning("cannot find a physical front contact for %s: %s", name, exc)
                contact = None
            if contact is not None:
                point_base = self.to_base(th.tensor(contact, dtype=th.float32), unit)[0].cpu().numpy()
                ways.append(
                    (
                        "from the front",
                        point_base,
                        world_to_base @ into_world,
                        (world_to_base @ np.array([0.0, 0.0, 1.0]),
                         world_to_base @ np.array([-into_world[1], into_world[0], 0.0])),
                    )
                )

        # The coarse line model omits only the target; the authoritative full-sphere transit check retains it
        # with NO contact exception. A terminal contact exception starts only once the hand is at the standoff.
        ignore = {obj.name}
        # Close FIRST, here in free space, then travel closed (the user's instruction for sticky grasps: close the
        # gripper, then touch the object). A closed hand is half the width of an open one: it fits between the
        # shelf boards around a book and past the neighbours of a battery that the open fingers brushed.
        if ways and self.ramp_to(
            self.q_arm(), self.posture, self.CLOSE, OPEN_GRASP_STEPS, note=f"sticky close before approaching {name}"
        ) is not None:
            return False
        for how, point_base, into_dir, jaws in ways:
            for jaw in jaws:
                solution = self._press_solution(arm, ik, seed, point_base, into_dir, jaw, press=-STICKY_STANDOFF)
                if solution is None:
                    continue
                where, rot = self.grasp_target(arm, point_base, into_dir, jaw, press=-STICKY_STANDOFF)
                quat = T.mat2quat(th.tensor(rot, dtype=th.float32)).cpu().numpy()
                # Candidate selection is only a coarse prefilter. ramp_to verifies the complete robot,
                # including the opposite arm and the torso, against the complete physical map before moving.
                legs = self.reach_plan(arm, ik, joints_of, solution, exclude=ignore, aabbs=aabbs)
                swept, start = [], seed
                for leg in legs or []:
                    swept += [n for n in self.path_hits_scene(arm, ik, start, leg, aabbs=aabbs) if n not in ignore]
                    start = leg
                if not legs or swept:
                    # No straight way in (a book between a bookcase's boards): let the planner find one.
                    log.info(f"{name}: no straight way in {how} ({swept[0] if swept else 'no clear candidate'})")
                    if self.planned_approach(where, quat, note=f"sticky precontact {name} {how}"):
                        measured = self.robot.get_joint_positions()
                        held = self._sticky_close_on(
                            arm, ik, obj, pose_matrix(where, quat), into_dir,
                            [float(measured[self.joint_index[j]]) for j in joints_of], joints_of,
                        )
                        if held:
                            self.lift_held(ik, obj, into_dir, joints_of, how)
                            held = self.robot._ag_obj_in_hand.get(arm) is obj  # the lift may have let go
                        return held
                    continue
                log.info(f"approaching the sticky precontact pose for {name} {how}")
                executed = False
                for leg in legs:
                    stopped = self.ramp_to(
                        self._targets_from(joints_of, leg), self.posture, self.CLOSE, OPEN_SETTLE_STEPS,
                        note=f"sticky precontact {name} {how}",
                    )
                    if stopped is not None:
                        log.warning("grasp approach rejected before closing: %s", stopped[0])
                        if executed or stopped[1] != 0:
                            return False
                        break  # preflight refused without movement: another jaw/approach may be clear
                    executed = True
                if stopped is not None:
                    continue
                held = self._sticky_close_on(
                    arm, ik, obj, pose_matrix(where, quat), into_dir, [float(v) for v in solution], joints_of,
                )
                if held:
                    self.lift_held(ik, obj, into_dir, joints_of, how)
                    held = self.robot._ag_obj_in_hand.get(arm) is obj
                return held
        log.info(f"{name}: the pressed grasp found no way onto it")
        return False

    def lift_held(self, ik, obj, into, joints_of, how: str) -> bool:
        """Back a just-taken object straight out the way the hand came in: up off a table, out of a shelf.

        A sticky grasp leaves the object resting where it was, so the fold for travel was refused (the carried
        object intersects its support at sample 0) and the robot teleported with the hand down at table height.
        Only the carried object may touch what it was resting on during this motion; the robot's own links stay
        checked against everything. Best effort: False leaves the hand where it is."""
        from omnigibson.tiptop.collision import CARRIED

        measured = self.robot.get_joint_positions()
        seed = [float(measured[self.joint_index[j]]) for j in joints_of]
        position, quat = ik.fk(seed)
        distance = STICKY_LIFT_FRONT if how == "from the front" else STICKY_LIFT
        goal = np.asarray(position, dtype=np.float64) - np.asarray(into, dtype=np.float64) * distance
        solution = ik.solve(goal, quat, seed=seed, tolerance_pos=0.005, tolerance_rad=0.05)
        if solution is None:
            log.info(f"{obj.name}: no local IK to lift it {distance:.2f} m back the way the hand came in")
            return False
        resting = {name: {CARRIED} for name in self.rests_against(obj)}
        for name, links in self.retreat_contacts().items():  # a hand that closed on something lying on the floor
            resting.setdefault(name, set()).update(links)
        # The hand record is written only after press_grasp returns (bench.note_pressed_grasp), and ramp_collision
        # reads what is carried from it: without this the object was an obstacle the closed fingers touch, with no
        # carried spheres for CARRIED to exempt, and every lift was refused at sample 0 (clean_up_your_desk,
        # 2026-09-22). Counted as held for this one motion only.
        label = next((name for name, o in self.objects.items() if o is obj), None)
        arm = "right" if any(j.startswith("right_arm") for j in joints_of) else "left"
        carrying = label is not None and label not in self.held_objects
        if carrying:
            self.held_objects[label] = arm
        try:
            stopped = self.ramp_to(
                self._targets_from(joints_of, solution), self.posture, self.CLOSE, OPEN_SETTLE_STEPS,
                note=f"lift {obj.name} off what it rested on", max_vel=STICKY_LIFT_VEL, allowed_contacts=resting,
            )
        finally:
            if carrying:
                self.held_objects.pop(label, None)
        if stopped is not None:
            log.info(f"{obj.name}: the lift after the sticky grasp was stopped ({stopped[0]})")
            # Refused before any step by the robot's own body or the floor: no motion out of here exists (every
            # later one starts in the same "collision", the planner's included), and kept in the hand the object
            # froze the episode -- 13 of 114 ended "retained in the hand" this way (laying_tile x3,
            # clean_up_your_desk x2, 2026-09-23). It still rests where it was taken: let go, and the next round
            # tries from another stance (stand_for avoids this one).
            obstacle = stopped[0].rsplit(" intersects ", 1)[-1]
            own = obstacle in set(self._motion_collision_model().links)
            # at path sample 0 only: a refusal partway along the lift (the far end of the tile meeting the base) is
            # not a start nothing can leave, and letting go there lost grasps the planner could still have carried
            if stopped[3:] == (0,) and (own or obstacle.startswith("floors_")):
                from omnigibson.tiptop.run import DROP_STEPS

                log.info(f"{obj.name}: letting go where it rests rather than holding what nothing can carry")
                self.hold(DROP_STEPS, self.OPEN)
        return stopped is None

    def rests_against(self, obj) -> set[str]:
        """The scene bodies whose boxes meet ``obj``'s (within 1 cm): what it stands on, and the container and
        neighbours packed round it."""
        lo, hi = (v.cpu().numpy().astype(np.float64) for v in obj.aabb)
        return {
            other.name
            for other, olo, ohi in self.scene_aabbs()
            if other is not obj and other is not self.robot and np.all(olo <= hi + 0.01) and np.all(lo - 0.01 <= ohi)
        }

    def retreat_contacts(self, bddl: str | None = None) -> dict:
        """What a hand backing out of a grasp may still be touching as it leaves: the target and the floor, for the
        gripper and finger links only (the rest of the robot stays checked). Without it the way back was refused
        at sample 0 by the very contact it leaves -- a finger in a detergent bottle, the gripper at the floor --
        and the arm stayed down for the rest of the episode (sorting_household_items, 2026-09-22)."""
        from omnigibson.tiptop.collision import CARRIED

        links = {f"{self.arm}_gripper_link", *self.robot.finger_link_names[self.arm]}
        # CARRIED too: what the hand lifts off the floor starts on it (a knocked-over bottle), and its box can sit a
        # little clear of the floor's so the resting test in lift_held missed it (sorting_household_items, 2026-09-22)
        allowed = {obj.name: {*links, CARRIED} for obj, _, _ in self.scene_aabbs()
                   if getattr(obj, "category", None) == "floors"}  # fmt: skip
        if bddl is not None:
            allowed.setdefault(self.scene_object(bddl).name, set()).update(links)
        return allowed

    def return_to_ready(self, note: str = "back to the ready posture", allowed_contacts=None) -> bool:
        """The planned arm back at its ready posture, where a planned Pick's GoToInitial leaves it and where the
        other arm's planner locks it (adopt_embodiment); False when neither a checked straight ramp nor the
        planner's path gets it there. After a sticky grasp the arm is wherever the grasp ended, and the right-arm
        press was refused with "r1pro_right locks left_arm_joint5 at -0.051 rad but the simulator has it at
        -0.510" (turning_on_radio, 2026-09-22)."""
        if self.q_home is None:
            return False
        if np.max(np.abs(np.subtract(self.q_arm(), self.q_home)), initial=0.0) < 0.02:
            return True
        if self.level and self.level & {l for l, a in self.hands().items() if a == self.arm}:
            # E-level: the ready posture turns the hand; a load that must stay level stays where it is instead
            ik = self.arm_ik(self.arm, frame=f"{self.arm}_gripper_link", with_torso=True)
            now = self.robot.get_joint_positions()
            q_now = [float(now[self.joint_index[j]]) for j in ik.arm_joints]
            home = dict(zip(self.planned_joints, self.q_home))
            tilt = held_tilt(ik.fk(q_now)[1], ik.fk([home.get(j, v) for j, v in zip(ik.arm_joints, q_now)])[1])
            if tilt > LEVEL_TILT:
                log.info(f"{note}: the ready posture would tip the level load {math.degrees(tilt):.0f} deg; staying here")
                return False
        stopped = self.ramp_to(list(self.q_home), self.posture, self.last_gripper, OPEN_SETTLE_STEPS, note=note,
                               max_vel=STICKY_LIFT_VEL, allowed_contacts=allowed_contacts)  # fmt: skip
        if stopped is None:
            return True
        if self.robot._ag_obj_in_hand.get(self.arm) is not None:
            # a move request carries no attachment: the held object would be a room static the fingers start in
            log.info(f"{note}: the straight ramp was stopped ({stopped[0]}) and the hand holds something the "
                     "planner's move request cannot carry; staying here")
            return False
        log.info(f"{note}: the straight ramp was stopped ({stopped[0]}); asking the planner")
        return self.planned_approach(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), note=note, goal_q=list(self.q_home))

    def tilt_wrist(self, arm: str, angle: float = POUR_TILT, hold_steps: int = POUR_HOLD_STEPS) -> bool:
        """N-rotate (spec S38): tip what ``arm`` holds by turning its last wrist joint ``angle`` rad, the way its
        limit allows, hold ``hold_steps`` for the contents to fall, and turn back. Both ramps are checked like every
        ramp (the carried volume against the scene). False when either was stopped: the hand is then wherever it
        stopped, still holding."""
        name = f"{arm}_arm_joint7"
        if name not in self.planned_joints:
            log.warning(f"{name} is not a planned joint; no wrist tilt")
            return False
        i = self.planned_joints.index(name)
        q = [float(v) for v in self.q_arm()]
        joint = self.robot.joints[name]
        room_up, room_down = float(joint.upper_limit) - q[i], q[i] - float(joint.lower_limit)
        tipped = list(q)
        tipped[i] += angle if room_up >= angle or room_up >= room_down else -angle
        stopped = self.ramp_to(tipped, self.posture, self.last_gripper, hold_steps, note=f"tip the {arm} wrist to pour",
                               max_vel=STICKY_LIFT_VEL)  # fmt: skip
        back = self.ramp_to(q, self.posture, self.last_gripper, OPEN_SETTLE_STEPS, note=f"{arm} wrist back level",
                            max_vel=STICKY_LIFT_VEL)  # fmt: skip
        return stopped is None and back is None

    def planned_standoff(self, arm: str, ik, joints_of, q_standoff, name: str) -> bool:
        """The planner's path to a handle's pre-solved standoff where no straight reach is clear; False without one.

        The goal is the standoff CONFIGURATION, not its pose: the approach and the pull were solved from it, and
        another IK branch at the same pose would swing the arm on the straight approach that follows."""
        if arm != self.arm or any(j not in joints_of for j in self.planned_joints):
            return False
        self.hold(OPEN_SETTLE_STEPS, self.OPEN)  # the planner plans with the fingers as they are
        where, quat = ik.fk(q_standoff, f"{arm}_gripper_link")
        goal_q = [float(q_standoff[joints_of.index(j)]) for j in self.planned_joints]
        if self.planned_approach(where, quat, note=f"reach the standoff of {name}", goal_q=goal_q):
            return True
        # cuRobo found the pre-solved configuration itself in collision (store_honey: "Start or End state in
        # collision" with a valid start; Lula's IK never checks self-collision). Let its own IK pick a branch at
        # the same pose: the straight approach from there is still preflighted, so a swinging branch is refused.
        return self.planned_approach(where, quat, note=f"reach the standoff of {name} (any configuration)")

    def planned_approach(self, where, quat, note: str, goal_q=None) -> bool:
        """Put the planned hand's gripper link at ``where``/``quat`` (base frame) along a path the planner checked
        through the room, with the fingers as they are now; False when there is no planner, no path, or the path
        did not complete. Executed like any plan, so the bridge audits it against the measured robot first.
        ``goal_q`` (planned joints) asks for that exact configuration instead of any one reaching the pose."""
        from b1k.bridge.client import move
        from b1k.bridge.executor import PlanExecutor
        from b1k.bridge.protocol import parse_plan

        client = getattr(self, "move_planner", None)
        if client is None or not self.send_room:
            return False
        measured = self.robot.get_joint_positions()
        self.nearby_obstacles(collision_map=True)
        request = {
            "type": "move",
            "q_init": np.asarray(self.q_arm(), dtype=np.float32),
            "locked_joints": {j: float(measured[self.joint_index[j]]) for j in self.posture if j not in self.planned_joints},
            "room": self.room_collision_scene(),
            "goal_link": f"{self.arm}_gripper_link",
            "goal_pose": np.asarray(pose_matrix(where, quat), dtype=np.float64),
        }
        if goal_q is not None:
            request["goal_q"] = np.asarray(goal_q, dtype=np.float32)
        try:
            response = move(client, request)
        except Exception as exc:  # noqa: BLE001 - a planner that cannot answer leaves the old refusal standing
            log.warning(f"{note}: the planner could not be asked for a path ({type(exc).__name__}: {exc})")
            return False
        if not response.get("success"):
            log.info(f"{note}: the planner has no path either ({response.get('error')})")
            return False
        if list(response["joint_names"]) != list(self.planned_joints):
            log.warning(f"{note}: the planner's joints {response['joint_names']} are not {self.planned_joints}")
            return False
        plan = parse_plan({"steps": [{"type": "trajectory", "positions": response["positions"], "dt": response["dt"],
                                      "label": note}], "q_init": request["q_init"]})
        stats = PlanExecutor(self).execute(plan)
        log.info(f"{note}: planned path {'completed' if stats.get('completed') else 'stopped: ' + str(stats.get('error'))}")
        return bool(stats.get("completed")) and not stats.get("error")

    def _sticky_close_on(self, arm, ik, obj, pose, into, seed, joints_of) -> bool:
        """Close in free space, then seek first finger contact slowly and wait without advancing the arm.

        Sticky grasping needs 0.3 s of continuous contact while CLOSE is commanded. The terminal search is
        bounded to the physical surface plus GRASP_PRESS; it never retries deeper after touching an object.
        """
        try:
            self.robot._find_gripper_contacts(arm=arm)
        except Exception as exc:
            log.warning("refusing sticky grasp without contact observations: %s", exc)
            return False
        stopped = self.ramp_to(
            self.q_arm(), self.posture, self.CLOSE, OPEN_GRASP_STEPS,
            note=f"sticky close before contact {obj.name}",
        )
        if stopped is not None:
            return False
        allowed = {obj.name: set(self.robot.finger_link_names[arm])}
        direction = np.asarray(into, dtype=np.float64)
        quat = T.mat2quat(th.tensor(pose[:3, :3], dtype=th.float32)).cpu().numpy()
        distance = STICKY_STANDOFF + GRASP_PRESS
        intervals = int(np.ceil(distance / STICKY_CONTACT_STEP))
        for advance in np.linspace(0.0, distance, intervals + 1):
            held = self.robot._ag_obj_in_hand.get(arm)
            touching, _ = self.robot._find_gripper_contacts(arm=arm)
            if held is not None or touching:
                break
            if advance == 0:
                continue
            measured = self.robot.get_joint_positions()
            seed = [float(measured[self.joint_index[j]]) for j in joints_of]
            goal = np.asarray(pose[:3, 3]) + direction * advance
            solution = ik.solve(goal, quat, seed=seed, tolerance_pos=0.0005, tolerance_rad=0.05)
            if solution is None:
                log.info("sticky contact seek for %s has no local IK", obj.name)
                return False
            delta = float(np.max(np.abs(np.asarray(solution) - seed), initial=0.0))
            # Include the measured IK residual: a loose precontact solution must not turn the first tiny
            # waypoint into a fast correction. Every waypoint gets at least its requested Cartesian time.
            travel = max(float(np.linalg.norm(goal - ik.fk(seed)[0])), distance / intervals)
            speed = min(0.1, max(delta, 1e-6) * STICKY_CONTACT_SPEED / travel)
            stopped = self.ramp_to(
                self._targets_from(joints_of, solution), self.posture, self.CLOSE, 0,
                note=f"sticky contact seek {obj.name}", max_vel=speed,
                allowed_contacts=allowed, stop_on_contact=arm,
            )
            if stopped is not None:
                if stopped[0] == "finger contact":
                    break
                return False
        # Cancel the advance target and preserve the measured full posture throughout the grasp window.
        # A failed window ends this attempt; moving farther would push an unattached object.
        measured = self.robot.get_joint_positions()
        self.posture = {j: float(measured[self.joint_index[j]]) for j in self.posture}
        q_hold = self.q_arm().copy()
        self.step(q_hold, self.CLOSE)
        for _ in range(OPEN_GRASP_STEPS + 1):
            held = self.robot._ag_obj_in_hand.get(arm)
            if held is not None:
                if held is not obj:
                    log.warning("sticky grasp expected %s but assist holds %s; stopping", obj.name, held.name)
                return held is obj
            touching, _ = self.robot._find_gripper_contacts(arm=arm)
            target = {link.prim_path for link in obj.links.values()}
            if not touching or not set(touching).issubset(target):
                log.info("sticky contact seek for %s ended without exclusive target contact", obj.name)
                return False
            self.step(q_hold, self.CLOSE)
        log.info("sticky contact with %s did not establish a hold; not pressing deeper", obj.name)
        return False

    def _press_solution(self, arm: str, ik, seed, point_base, into_dir, jaw, press=GRASP_PRESS):
        """Arm joints that put the open hand on ``point_base`` coming in along ``into_dir``, or None.

        The two tolerances are the pattern the drawer pull uses: ask for the tight one, settle for 2 cm.
        """
        where, rot = self.grasp_target(arm, point_base, into_dir, jaw, press=press)
        quat = T.mat2quat(th.tensor(rot, dtype=th.float32)).cpu().numpy()
        for tolerance in (OPEN_GRASP_TOLERANCE, 0.02):
            solution = ik.solve(where, quat, seed=seed, tolerance_pos=tolerance, tolerance_rad=0.5)
            if solution is not None:
                return solution
        return None

    def clear_start_posture(self, arm: str, q_ready):
        """Report a potentially obstructed start without moving the arm or hiding its obstacle.

        A joint-space lift selected solely for a clear endpoint is not collision-free recovery. Send the measured
        configuration and complete geometry so the planner can reject an invalid start instead of planning in
        an artificially emptied world. The bridge mesh check is diagnostic; cuRobo checks the robot's spheres.
        """
        if arm not in self.robot.arm_names:
            return q_ready
        try:
            ik = self._stance_ik(arm)
            by_name = dict(zip(self.planned_joints, [float(v) for v in q_ready]))
            q = [by_name.get(j, 0.0) for j in self.robot.arm_joint_names[arm]]
            inside = self.arm_hits_scene(arm, ik, q)
        except Exception as exc:  # diagnostic failure must not erase geometry or change the measured posture
            log.warning("could not check the planning start against scene meshes: %s", exc)
            return q_ready
        if inside:
            log.warning(
                "the %s arm starts near collision geometry %s; retaining all obstacles and asking the planner "
                "to validate the measured start state",
                arm,
                inside,
            )
        return q_ready

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
