"""The bench gripper is the evaluator's: MultiFingerGripperController in smooth mode, where the command in [-1, 1]
IS the finger position (+1 open, -1 closed), so a ramped close is a ramped finger target. No simulator: the controller
is built on the torch compute backend the simulator would bind."""

import pytest
import torch as th

import omnigibson.utils.backend_utils as backend
from omnigibson.controllers.multi_finger_gripper_controller import MultiFingerGripperController
from omnigibson.tiptop.r1pro import make_r1pro_env_config
from omnigibson.tiptop.scene import TiptopSim


def test_plus_and_minus_one_are_the_open_and_closed_finger_positions_in_smooth_mode():
    backend._compute_backend.set_methods_from_backend(backend._ComputeTorchBackend)
    config = make_r1pro_env_config()["robots"][0]["controller_config"]
    assert config["gripper_left"]["mode"] == config["gripper_right"]["mode"] == "smooth"
    gripper = {k: v for k, v in config["gripper_left"].items() if k != "name"}
    limits = {  # r1pro's two finger joints: 0 (closed) .. 0.05 m (open)
        "position": [th.zeros(2), th.full((2,), 0.05)],
        "velocity": [th.full((2,), -1.0), th.ones(2)],
        "effort": [th.full((2,), -100.0), th.full((2,), 100.0)],
        "has_limit": [True, True],
    }
    controller = MultiFingerGripperController(
        control_freq=30, motor_type="position", control_limits=limits, dof_idx=[0, 1], **gripper
    )

    def target(command):
        return float(controller._preprocess_command(command)[0])

    assert target(TiptopSim.OPEN) == pytest.approx(0.05) and target(TiptopSim.CLOSE) == pytest.approx(0.0)
    assert target(0.0) == pytest.approx(0.025), "linear between them: a ramped command is a ramped finger target"
    assert target(-3.0) == pytest.approx(0.0), "clipped to the command range"
