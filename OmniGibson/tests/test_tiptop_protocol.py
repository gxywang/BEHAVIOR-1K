"""Pure-python tests for the TiPToP wire/file formats (no Isaac Sim needed)."""

import json

import numpy as np
import pytest

from omnigibson.tiptop.protocol import (
    build_request,
    depth_to_points,
    load_observation_h5,
    match_objects,
    packb,
    parse_plan,
    resample_trajectory,
    save_observation_h5,
    unpackb,
    points_to_pixels,
)


def _request(gt=True):
    h, w = 12, 16
    rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    depth = np.random.rand(h, w).astype(np.float32) + 0.5
    K = np.array([[20.0, 0, 8], [0, 20.0, 6], [0, 0, 1]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [0.3, 0.0, 0.5]
    masks = np.zeros((2, h, w), bool)
    masks[0, 2:5, 3:6] = True
    masks[1, 6:10, 8:14] = True
    g = (
        {"labels": ["mug", "bowl"], "masks": masks, "atoms": [{"predicate": "on", "args": ["mug", "bowl"]}]}
        if gt
        else None
    )
    return build_request(rgb, depth, K, T, "put the mug in the bowl", np.zeros(7), gt=g)


def test_msgpack_numpy_wire_format_roundtrip():
    req = _request()
    raw = packb(req)
    back = unpackb(raw)
    for key in ("rgb", "depth", "intrinsics", "world_from_cam", "q_init", "gt_masks"):
        assert back[key].dtype == req[key].dtype and back[key].shape == req[key].shape
        assert np.array_equal(back[key], req[key])
    assert back["task"] == req["task"] and back["gt_labels"] == ["mug", "bowl"] and back["gt_atoms"] == req["gt_atoms"]


def test_wire_format_matches_msgpack_numpy_keys():
    raw = packb({"a": np.arange(3, dtype=np.float32)})
    import msgpack

    plain = msgpack.unpackb(raw, raw=False, strict_map_key=False)["a"]
    assert plain[b"nd"] is True and plain[b"type"] == "<f4" and plain[b"shape"] == [3] and plain[b"kind"] == b""
    assert np.frombuffer(plain[b"data"], dtype="<f4").tolist() == [0.0, 1.0, 2.0]


def test_build_request_validation():
    req = _request(gt=False)
    with pytest.raises(ValueError):
        build_request(req["rgb"], req["depth"][:-1], req["intrinsics"], req["world_from_cam"], "x", req["q_init"])
    bad_depth = req["depth"].copy()
    bad_depth[0, 0] = np.inf
    with pytest.raises(ValueError):
        build_request(req["rgb"], bad_depth, req["intrinsics"], req["world_from_cam"], "x", req["q_init"])


def test_parse_plan_and_resample():
    plan = {
        "version": "1.0.0",
        "q_init": [0] * 7,
        "steps": [
            {
                "type": "trajectory",
                "label": "Pick(a)",
                "positions": np.linspace(0, 1, 11)[:, None].repeat(7, 1).tolist(),
                "velocities": None,
                "dt": 0.02,
            },
            {"type": "gripper", "label": "Pick(a)", "action": "close"},
        ],
    }
    parsed = parse_plan(plan)
    assert parsed["steps"][0]["positions"].shape == (11, 7) and parsed["steps"][1]["action"] == "close"
    traj = resample_trajectory(parsed["steps"][0]["positions"], 0.02, 1 / 30)
    assert traj.shape[1] == 7 and np.allclose(traj[-1], 1.0) and np.allclose(traj[0], 0.0)
    assert len(traj) == 7  # 0.2 s at 30 Hz -> 6 samples + end point
    with pytest.raises(ValueError):
        parse_plan(dict(plan, version="2.0.0"))
    with pytest.raises(ValueError):
        parse_plan({"steps": [{"type": "gripper", "action": "squeeze"}]})


def test_h5_roundtrip(tmp_path):
    req = _request()
    req["robot_mask"] = np.zeros((12, 16), dtype=bool)
    req["robot_mask"][2:4, 3:6] = True
    quat = [1.0, 0.0, 0.0, 0.0]
    save_observation_h5(tmp_path / "obs.h5", req, [0.3, 0.0, 0.5], quat, extra={"note": {"a": 1}})
    back = load_observation_h5(tmp_path / "obs.h5")
    assert np.array_equal(back["rgb"], req["rgb"]) and np.allclose(back["depth"], req["depth"])
    assert back["gt_labels"] == ["mug", "bowl"] and back["gt_atoms"] == req["gt_atoms"]
    assert np.array_equal(back["gt_masks"].astype(bool), req["gt_masks"].astype(bool))
    assert back["robot_mask"].dtype == bool and np.array_equal(back["robot_mask"], req["robot_mask"])
    import h5py

    with h5py.File(tmp_path / "obs.h5") as f:
        assert f.attrs["pos_w_z_offset_m"] == 0.0 and f["depth"].shape == (12, 16, 1)
        assert json.loads(f.attrs["note"]) == {"a": 1}


def test_depth_to_points_pinhole():
    depth = np.full((4, 6), 2.0, dtype=np.float32)
    K = np.array([[10.0, 0, 3], [0, 10.0, 2], [0, 0, 1]])
    pts = depth_to_points(depth, K)
    assert np.allclose(pts[2, 3], [0, 0, 2.0])  # principal point maps to the optical axis
    assert np.allclose(pts[2, 5], [0.4, 0, 2.0])
    depth[0, 0] = 0
    assert np.isnan(depth_to_points(depth, K)[0, 0]).all()


def test_points_to_pixels_inverts_depth_to_points():
    """points_to_pixels is the inverse of depth_to_points, including the world_from_cam transform."""
    K = np.array([[10.0, 0, 3], [0, 10.0, 2], [0, 0, 1]])
    depth = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    # a non-trivial rigid transform: 90 deg about z, then a translation
    world_from_cam = np.array(
        [[0.0, -1.0, 0.0, 0.5], [1.0, 0.0, 0.0, -0.2], [0.0, 0.0, 1.0, 1.3], [0.0, 0.0, 0.0, 1.0]]
    )
    pts = depth_to_points(depth, K, world_from_cam).reshape(-1, 3)
    px, z = points_to_pixels(pts, K, world_from_cam)
    v, u = np.meshgrid(np.arange(2), np.arange(2), indexing="ij")
    assert np.allclose(px[:, 0], u.ravel(), atol=1e-9)
    assert np.allclose(px[:, 1], v.ravel(), atol=1e-9)
    assert np.allclose(z, depth.ravel(), atol=1e-9)


def test_points_to_pixels_flags_points_outside_the_frame():
    """A point beyond the image edge projects outside [0, w) - what the capture clipping check relies on."""
    K = np.array([[306.0, 0, 360.0], [0, 306.0, 360.0], [0, 0, 1]])
    eye = np.eye(4)
    px, z = points_to_pixels([[0.0, 0.0, 1.0], [-2.0, 0.0, 1.0], [0.0, 2.0, 1.0]], K, eye)
    assert np.allclose(px[0], [360.0, 360.0]) and z[0] == 1.0
    assert px[1][0] < 0  # off the left edge
    assert px[2][1] > 720  # off the bottom edge


def test_match_objects_pairs_by_position_not_name():
    # perception's "candle_2" is the simulator's candle_1, one candle over (2026-09-04, round_01 of runs/demo)
    perceived = {"candle": [0.594, 0.572, 0.471], "candle_2": [0.714, 0.572, 0.472]}
    simulated = {"candle_1": [0.732, 0.596, 0.472], "candle_2": [0.612, 0.595, 0.472]}
    match = match_objects(perceived, simulated)
    assert match["candle"]["sim"] == "candle_2" and match["candle_2"]["sim"] == "candle_1"
    assert all(m["dist"] < 0.04 for m in match.values())


def test_match_objects_flags_false_detections_and_uses_each_simulated_object_once():
    perceived = {"candle": [0.607, 0.169, 0.41], "cookie": [0.653, -0.149, 0.436], "cookie_2": [0.66, -0.14, 0.44]}
    simulated = {"cookie_1": [0.668, -0.153, 0.428]}
    match = match_objects(perceived, simulated)
    assert match["candle"]["sim"] is None and match["candle"]["dist"] > 0.3  # nothing near: a phantom
    assert match["cookie"]["sim"] == "cookie_1"
    assert match["cookie_2"]["sim"] is None and match["cookie_2"]["dist"] < 0.02  # the partner was taken


def test_sim_state_message_roundtrips_matrices_and_jpeg_bytes():
    msg = {
        "type": "sim_state",
        "t": 1.5,
        "q": np.zeros(11, np.float32),
        "objects": {"candle_2": np.eye(4, dtype=np.float32)},
        "images": {"head_cam": b"\xff\xd8jpeg"},
    }
    back = unpackb(packb(msg))
    assert back["images"]["head_cam"] == b"\xff\xd8jpeg"
    assert back["objects"]["candle_2"].shape == (4, 4) and back["q"].dtype == np.float32
