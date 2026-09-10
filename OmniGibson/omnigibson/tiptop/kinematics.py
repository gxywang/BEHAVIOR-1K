"""Arm inverse kinematics for a camera link: a free arm points its wrist camera at the workspace for a capture
(``R1ProSim.capture``).

Lula (shipped with Isaac Sim) solves the arm's joints with every other joint fixed where it is; the robot
description it needs is written in memory from the URDF and the joint names, so only the URDF is read from disk.
Frames: the URDF's root link is the robot base frame, which the bridge uses as the planner's world frame (Lula's
link poses match the simulator's base-frame link poses). Quaternions are (x, y, z, w). Cameras follow the USD
convention (-z forward, +y up), as the simulator's sensors do. numpy and scipy only, apart from lula itself.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

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


def look_at_quat_xyzw(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Orientation (x, y, z, w) of a USD/OpenGL camera at ``eye`` looking at ``target`` (camera -z = view direction,
    +y up)."""
    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight along `up`: pick any horizontal axis as the image x axis
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    return Rotation.from_matrix(np.stack([right, cam_up, -forward], axis=1)).as_quat()


def pose_matrix(pos, quat_xyzw) -> np.ndarray:
    """(4, 4) homogeneous transform from a position and an (x, y, z, w) quaternion."""
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, float)).as_matrix()
    mat[:3, 3] = np.asarray(pos, float)
    return mat


def matrix_pose(mat) -> tuple[np.ndarray, np.ndarray]:
    """Position and (x, y, z, w) quaternion of a (4, 4) transform."""
    mat = np.asarray(mat, float)
    return mat[:3, 3].copy(), Rotation.from_matrix(mat[:3, :3]).as_quat()


def link_from_camera(link_pos, link_quat, cam_pos, cam_quat) -> np.ndarray:
    """The constant pose of a camera in its link's frame, from both poses read in the same frame at one instant."""
    return np.linalg.inv(pose_matrix(link_pos, link_quat)) @ pose_matrix(cam_pos, cam_quat)


def link_pose_for_camera(cam_pos, cam_quat, link_from_cam) -> tuple[np.ndarray, np.ndarray]:
    """Where the link must be for its camera to have the given pose."""
    return matrix_pose(pose_matrix(cam_pos, cam_quat) @ np.linalg.inv(link_from_cam))


def look_pose(target, shoulder, side: int, offset=(0.2, 0.3, -0.05)) -> tuple[np.ndarray, np.ndarray]:
    """A camera pose by an arm's shoulder that looks at ``target`` (base frame): the camera sits ``offset`` (ahead,
    aside on the arm's side, up) from ``shoulder`` (``side`` +1 for the robot's left arm, -1 for its right), so
    it is within the arm's reach whatever the target, and turns toward the target. Returns the camera position and
    its look-at orientation."""
    shoulder = np.asarray(shoulder, float)
    eye = shoulder + np.array([offset[0], side * offset[1], offset[2]])
    return eye, look_at_quat_xyzw(eye, np.asarray(target, float))


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
