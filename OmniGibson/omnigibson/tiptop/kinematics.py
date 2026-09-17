"""Arm inverse kinematics for a camera link: a free arm points its wrist camera at the workspace for a capture
(``R1ProSim.capture``).

Lula (shipped with Isaac Sim) solves the arm's joints with every other joint fixed where it is; the robot
description it needs is written in memory from the URDF and the joint names, so only the URDF is read from disk.
This module is what is left here after the move: finding Lula means locating the isaacsim package, which is the
simulator. The pose arithmetic it is used with -- ``look_at_quat_xyzw``, ``pose_matrix``, ``matrix_pose``,
``link_from_camera``, ``link_pose_for_camera``, ``look_pose`` -- moved to ``b1k.bridge.kinematics`` and is
re-exported below, so callers of this module are unchanged.

Frames: the URDF's root link is the robot base frame, which the bridge uses as the planner's world frame (Lula's
link poses match the simulator's base-frame link poses). Quaternions are (x, y, z, w). Cameras follow the USD
convention (-z forward, +y up), as the simulator's sensors do.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from b1k.bridge.kinematics import (  # noqa: F401
    link_from_camera,
    link_pose_for_camera,
    look_at_quat_xyzw,
    look_pose,
    matrix_pose,
    pose_matrix,
)

LULA_PIP_PREBUNDLE = "exts/isaacsim.robot_motion.lula/pip_prebundle"  # under the isaacsim package, outside the app


def lula_module():
    """The lula module: on the path inside the Kit app, in Isaac Sim's pip_prebundle outside it (located without
    importing the isaacsim package, which would start the app)."""
    try:
        import lula
    except ImportError:
        spec = importlib.util.find_spec("isaacsim")
        if spec is None or not spec.submodule_search_locations:
            raise
        sys.path.append(str(Path(list(spec.submodule_search_locations)[0]) / LULA_PIP_PREBUNDLE))
        import lula
    return lula


class ArmIK:
    """Lula IK for one arm of a URDF robot, every other joint fixed at a given value."""

    def __init__(
        self, urdf_path, arm_joints: list[str], fixed: dict[str, float], frame: str, root_link: str = "base_link"
    ):
        """``arm_joints``: the configuration space, in URDF order; ``fixed``: the values of the robot's other
        movable joints; ``frame``: the URDF link the targets are for (a camera link)."""
        self.arm_joints = list(arm_joints)
        self.frame = frame
        rules = "\n".join(f"  - {{name: {name}, rule: fixed, value: {float(value)}}}" for name, value in fixed.items())
        description = (
            "api_version: 1.0\n"
            f"root_link: {root_link}\n"
            "cspace:\n" + "\n".join(f"  - {name}" for name in self.arm_joints) + "\n"
            f"default_q: [{', '.join('0.0' for _ in self.arm_joints)}]\n"
            "cspace_to_urdf_rules:\n" + rules + "\n"
        )
        self._lula = lula_module()
        self.robot = self._lula.load_robot_from_memory(description, Path(urdf_path).read_text())
        self.kinematics = self.robot.kinematics()

    def fk(self, q, frame: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Position and (x, y, z, w) quaternion of ``frame`` (default: the IK frame) at arm joints ``q``, base frame."""
        pose = self.kinematics.pose(np.asarray(q, float), frame or self.frame)
        return np.asarray(pose.translation, float).reshape(3), Rotation.from_matrix(pose.rotation.matrix()).as_quat()

    def solve(self, pos, quat_xyzw, seed, tolerance_pos: float = 0.01, tolerance_rad: float = 0.1, descents: int = 20):
        """Arm joints that put the frame at ``pos`` / ``quat_xyzw`` (base frame) within the tolerances, from the
        ``seed`` configuration; None when Lula finds none."""
        lula = self._lula
        rot = Rotation.from_quat(np.asarray(quat_xyzw, float)).as_matrix()
        target = lula.Pose3(lula.Rotation3(rot), np.asarray(pos, float).reshape(3, 1))
        config = lula.CyclicCoordDescentIkConfig()
        config.cspace_seeds = [np.asarray(seed, float)]
        config.position_tolerance = tolerance_pos
        config.orientation_tolerance = tolerance_rad
        config.max_num_descents = descents
        result = lula.compute_ik_ccd(self.kinematics, target, self.frame, config)
        return np.asarray(result.cspace_position, float) if result.success else None
