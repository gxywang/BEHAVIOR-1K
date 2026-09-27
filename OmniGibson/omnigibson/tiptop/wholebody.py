"""Whole-body planner metadata at the simulator boundary; virtual coordinates are never robot joint indices."""

import copy

import numpy as np

BASE_JOINTS = ("base_x", "base_y", "base_yaw")
ROBOT_TYPES = {arm: f"r1pro_{arm}_wholebody" for arm in ("left", "right")}


def physical_embodiment(embodiment: dict) -> dict:
    """A copy for capture/setup routines that address real torso/arm joints; retain full metadata on the client."""
    names = tuple(embodiment["joint_names"])
    mobile = embodiment.get("robot_type") in ROBOT_TYPES.values()
    if not mobile:
        if set(names) & set(BASE_JOINTS):
            raise ValueError("virtual base joints require a whole-body embodiment")
        return embodiment
    arm = embodiment.get("arm")
    if arm not in ROBOT_TYPES or embodiment["robot_type"] != ROBOT_TYPES[arm]:
        raise ValueError("whole-body robot_type and arm disagree")
    if tuple(embodiment.get("virtual_base_joints", ())) != BASE_JOINTS:
        raise ValueError(f"whole-body metadata must declare virtual_base_joints={BASE_JOINTS}")
    if names[:3] != BASE_JOINTS or len(names) != len(set(names)):
        raise ValueError(f"whole-body joint order must start with {BASE_JOINTS}, without duplicates")
    physical = tuple(f"torso_joint{i}" for i in range(1, 5)) + tuple(f"{arm}_arm_joint{i}" for i in range(1, 8))
    if names[3:] != physical:
        raise ValueError(f"whole-body physical joint order is not {physical}")
    home = np.asarray(embodiment["q_home"], dtype=float)
    if home.shape != (len(names),) or not np.isfinite(home).all() or not np.allclose(home[:3], 0.0):
        raise ValueError("whole-body home must have finite joints and zero virtual base coordinates")
    if set(embodiment.get("locked_joints", {})) & set(BASE_JOINTS):
        raise ValueError("whole-body virtual base coordinates cannot be locked physical joints")
    result = copy.deepcopy(embodiment)
    result["joint_names"], result["q_home"] = list(physical), home[3:].tolist()
    result["robot_type"] = f"r1pro_{arm}"
    result["base_link"] = "base_link"
    result.pop("virtual_base_joints", None)
    return result


def validate_planner_metadata(metadata: dict, arm: str) -> None:
    """Refuse absent or incompatible mobile capability before launching the simulator."""
    if metadata.get("robot_type") != ROBOT_TYPES[arm]:
        raise ValueError(f"whole-body {arm} server must explicitly identify its robot_type")
    if (metadata.get("capabilities") or {}).get("whole_body") is not True:
        raise ValueError("planner does not advertise capabilities.whole_body=true")
    embodiment = metadata["embodiment"]
    if embodiment.get("robot_type") != ROBOT_TYPES[arm] or embodiment.get("arm") != arm:
        raise ValueError("server and embodiment whole-body metadata disagree")
    physical_embodiment(embodiment)
    if metadata.get("dof") != len(embodiment["joint_names"]):
        raise ValueError("whole-body server must declare the complete virtual + physical dof")


def base_controller(whole_body: bool) -> dict:
    """The evaluator's normalized body-frame velocity contract, or the existing fixed-base bench controller."""
    if whole_body:
        return {
            "name": "HolonomicBaseJointController",
            "motor_type": "velocity",
            "vel_kp": 150,
            "command_input_limits": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            "command_output_limits": [[-0.75, -0.75, -1.0], [0.75, 0.75, 1.0]],
            "use_impedances": False,
        }
    return {
        "name": "HolonomicBaseJointController",
        "motor_type": "position",
        "command_input_limits": None,
        "command_output_limits": None,
    }
