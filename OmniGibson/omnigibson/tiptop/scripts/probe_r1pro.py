"""Probe the live R1Pro in an empty scene: joint order, limits, eef/camera poses at sampled configs, intrinsics.

Writes probe.json for tiptop/scripts/check_r1pro_embodiment.py (validates the planner model against the simulator).
Run from OmniGibson/ in the behavior env: OMNIGIBSON_HEADLESS=1 python omnigibson/tiptop/scripts/probe_r1pro.py probe.json
"""

import json
import sys

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T

OUT = sys.argv[1]
rng = np.random.default_rng(0)
jc = lambda: {
    "name": "JointController",
    "motor_type": "position",
    "use_delta_commands": False,
    "use_impedances": False,
    "command_input_limits": None,
    "command_output_limits": None,
}
cfg = {
    "env": {"action_frequency": 30, "rendering_frequency": 30, "physics_frequency": 120},
    "scene": {"type": "Scene"},
    "robots": [
        {
            "type": "R1Pro",
            "name": "robot_r1",
            "position": [0, 0, 0],
            "orientation": [0, 0, 0, 1],
            "obs_modalities": ["rgb", "depth_linear", "seg_instance"],
            "action_normalize": False,
            "self_collisions": True,
            "grasping_mode": "sticky",
            "controller_config": {
                "base": {
                    "name": "HolonomicBaseJointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "command_output_limits": None,
                },
                "trunk": jc(),
                "arm_left": jc(),
                "arm_right": jc(),
                "gripper_left": {"name": "MultiFingerGripperController", "mode": "binary"},
                "gripper_right": {"name": "MultiFingerGripperController", "mode": "binary"},
            },
            "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_height": 720, "image_width": 720}}},
        }
    ],
}
env = og.Environment(configs=cfg)
robot = env.robots[0]
for _ in range(3):
    og.sim.step()
info = {}
names = list(robot.joints.keys())
info["joint_names"] = names
info["control_idx"] = {
    "base": robot.base_control_idx.tolist(),
    "trunk": robot.trunk_control_idx.tolist(),
    "arm_left": robot.arm_control_idx["left"].tolist(),
    "arm_right": robot.arm_control_idx["right"].tolist(),
    "gripper_left": robot.gripper_control_idx["left"].tolist(),
    "gripper_right": robot.gripper_control_idx["right"].tolist(),
}
lo, hi = robot.joint_lower_limits.tolist(), robot.joint_upper_limits.tolist()
info["limits"] = {n: [lo[i], hi[i]] for i, n in enumerate(names)}
info["default_joint_pos"] = robot.reset_joint_pos.tolist() if hasattr(robot, "reset_joint_pos") else None
info["links"] = list(robot.links.keys())
info["eef_link_names"] = dict(robot.eef_link_names)
info["finger_link_names"] = {k: list(v) for k, v in robot.finger_link_names.items()}
info["finger_joint_names"] = {k: list(v) for k, v in robot.finger_joint_names.items()}
info["arm_joint_names"] = {k: list(v) for k, v in robot.arm_joint_names.items()}
info["trunk_joint_names"] = list(robot.trunk_joint_names) if hasattr(robot, "trunk_joint_names") else None
info["root_link_name"] = robot.root_link_name
info["default_arm"] = robot.default_arm
info["eef_to_fingertip_lengths"] = str(getattr(robot, "eef_to_fingertip_lengths", None))
info["sensors"] = list(robot.sensors.keys())
info["action_dim"] = int(robot.action_dim)
info["controller_order"] = list(robot.controller_order)
info["controller_action_idx"] = {k: v.tolist() for k, v in robot.controller_action_idx.items()}


def rel(pos, quat):
    bp, bq = robot.get_position_orientation()
    p, q = T.relative_pose_transform(th.as_tensor(pos, dtype=th.float32), th.as_tensor(quat, dtype=th.float32), bp, bq)
    return p.tolist(), q.tolist()


def link_rel(name):
    return rel(*robot.links[name].get_position_orientation())


# base vs root frame
info["base_pose_world"] = [x.tolist() for x in robot.get_position_orientation()]
info["root_pose_world"] = [x.tolist() for x in robot.root_link.get_position_orientation()]
info["base_link_pose_world"] = [x.tolist() for x in robot.links["base_link"].get_position_orientation()]

# cameras
cams = {}
for sname, s in robot.sensors.items():
    if "Camera" not in sname:
        continue
    cams[sname] = {
        "intrinsics_default_aperture": s.intrinsic_matrix.tolist(),
        "horizontal_aperture": float(s.horizontal_aperture),
        "focal_length": float(s.focal_length),
        "image_wh": [int(s.image_width), int(s.image_height)],
        "pose_base_usd_q0": rel(*s.get_position_orientation()),
    }
info["cameras_q0"] = cams


def set_q(qd):
    q = robot.get_joint_positions().clone()
    for n, v in qd.items():
        q[names.index(n)] = float(v)
    robot.set_joint_positions(q, drive=False)
    robot.set_joint_positions(q, drive=True)
    robot.keep_still()
    og.sim.step_physics()
    for _ in range(2):
        og.sim.render()


samples = []
arm_l = info["arm_joint_names"]["left"]
arm_r = info["arm_joint_names"]["right"]
trunk = info["trunk_joint_names"]
fing = info["finger_joint_names"]["left"] + info["finger_joint_names"]["right"]
for k in range(25):
    qd = {}
    if k == 0:  # all zeros, fingers open
        qd = {n: 0.0 for n in arm_l + arm_r + trunk} | {n: 0.05 for n in fing}
    elif k == 1:  # challenge torso posture, arms at zero
        qd = {n: 0.0 for n in arm_l + arm_r} | dict(zip(trunk, [1.025, -1.45, -0.47, 0.0])) | {n: 0.05 for n in fing}
    else:  # random configuration inside 80% of the joint ranges
        for n in arm_l + arm_r + trunk:
            lo_, hi_ = info["limits"][n]
            qd[n] = float(rng.uniform(lo_ + 0.1 * (hi_ - lo_), hi_ - 0.1 * (hi_ - lo_)))
        for n in fing:
            qd[n] = float(rng.uniform(0.0, 0.05))
    set_q(qd)
    qnow = robot.get_joint_positions()
    rec = {
        "q_cmd": qd,
        "q_actual": {n: float(qnow[names.index(n)]) for n in qd},
        "left_eef_link": link_rel("left_eef_link"),
        "right_eef_link": link_rel("right_eef_link"),
        "left_gripper_link": link_rel("left_gripper_link"),
        "torso_link4": link_rel("torso_link4"),
        "zed_link": link_rel("zed_link"),
        "left_realsense_link": link_rel("left_realsense_link"),
        "left_gripper_finger_link1": link_rel("left_gripper_finger_link1"),
        "left_gripper_finger_link2": link_rel("left_gripper_finger_link2"),
        "cams": {sname: rel(*s.get_position_orientation()) for sname, s in robot.sensors.items() if "Camera" in sname},
        "base_pose_world": [x.tolist() for x in robot.get_position_orientation()],
    }
    samples.append(rec)
info["samples"] = samples
# finger geometry: aabb of finger links at open/closed
set_q({n: 0.05 for n in fing})
info["finger_open_link_pos"] = {n: link_rel(n)[0] for n in ["left_gripper_finger_link1", "left_gripper_finger_link2"]}
set_q({n: 0.0 for n in fing})
info["finger_closed_link_pos"] = {n: link_rel(n)[0] for n in ["left_gripper_finger_link1", "left_gripper_finger_link2"]}
with open(OUT, "w") as f:
    json.dump(info, f, indent=1)
print("PROBE DONE", OUT)
og.shutdown()
