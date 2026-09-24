"""The simulator mirror identifies its episode before any planning request exists."""

from types import SimpleNamespace

import numpy as np
import pytest

from b1k.bridge.client import SimStateStream
from b1k.bridge.protocol import packb, unpackb


class Socket:
    def __init__(self):
        self.messages = []

    def recv(self, timeout=None):
        return packb({"sim_state_supported": True})

    def send(self, payload):
        self.messages.append(unpackb(payload))

    def close(self):
        pass


def simulated_state():
    return SimpleNamespace(
        stream_scene=lambda: {"dish": {"pose": np.eye(4, dtype=np.float32)}},
        mirror_q=lambda: np.arange(11, dtype=np.float32),
        mirror_fingers=lambda: [0.05, 0.05],
        object_poses_base_mats=lambda: {"dish": np.eye(4, dtype=np.float32)},
        stream_images=lambda: {"head": b"jpeg"},
    )


@pytest.mark.parametrize("initial_episode", ["", "dishes_301_0"])
def test_scene_and_state_preserve_episode_step_and_original_payload(monkeypatch, initial_episode):
    socket = Socket()
    monkeypatch.setattr("b1k.bridge.client.connect", lambda *args, **kwargs: socket)
    sim = simulated_state()
    sim.episode_name, sim.n_steps = initial_episode, 0
    stream = SimStateStream("localhost")
    assert stream.attach(sim)
    scene = socket.messages[0]
    assert scene["type"] == "sim_scene" and scene["sim_step"] == 0
    assert scene.get("episode", "") == initial_episode
    np.testing.assert_array_equal(scene["objects"]["dish"]["pose"], np.eye(4))

    # Bench calls begin_episode after connecting; initial stance messages must carry its new name.
    sim.episode_name, sim.n_steps = "dishes_301_0", 7
    stream.on_step(sim)
    state = socket.messages[1]
    assert state["type"] == "sim_state" and state["episode"] == "dishes_301_0" and state["sim_step"] == 7
    np.testing.assert_array_equal(state["q"], np.arange(11, dtype=np.float32))
    assert state["q_gripper"] == 0.05 and state["images"] == {"head": b"jpeg"}
    np.testing.assert_array_equal(state["objects"]["dish"], np.eye(4))


def test_older_simulators_without_episode_metadata_can_still_stream(monkeypatch):
    socket = Socket()
    monkeypatch.setattr("b1k.bridge.client.connect", lambda *args, **kwargs: socket)
    sim = simulated_state()
    stream = SimStateStream("localhost")
    assert stream.attach(sim)
    stream.on_step(sim)
    assert len(socket.messages) == 2
    assert all("episode" not in msg and "sim_step" not in msg for msg in socket.messages)
