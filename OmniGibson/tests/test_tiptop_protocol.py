"""Pure-python tests for the TiPToP wire/file formats (no Isaac Sim needed)."""

import json

import numpy as np
import pytest

from omnigibson.tiptop.protocol import (
    build_request,
    canonical_object_name,
    depth_to_points,
    load_observation_h5,
    match_objects,
    packb,
    rerun_name,
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
        "q": np.zeros(11, np.float32),
        "objects": {"candle_2": np.eye(4, dtype=np.float32)},
        "images": {"head_cam": b"\xff\xd8jpeg"},
    }
    back = unpackb(packb(msg))
    assert back["images"]["head_cam"] == b"\xff\xd8jpeg"
    assert back["objects"]["candle_2"].shape == (4, 4) and back["q"].dtype == np.float32


def test_match_objects_edge_cases():
    assert match_objects({}, {"a": [0, 0, 0]}) == {}
    assert match_objects({"p": [0, 0, 0]}, {}) == {"p": {"sim": None, "dist": None}}
    assert match_objects({"p": [0, 0, 0]}, {"a": [0.08, 0, 0]})["p"]["sim"] == "a"  # max_dist is inclusive
    assert match_objects({"p": [0, 0, 0]}, {"a": [0.08 + 1e-6, 0, 0]})["p"]["sim"] is None
    tie = match_objects({"p": [0, 0, 0]}, {"b": [0.01, 0, 0], "a": [-0.01, 0, 0]})
    assert tie["p"]["sim"] == "a"  # equal distance: name order, deterministic
    m = match_objects({"z": [0, 0, 0], "a": [0.05, 0, 0]}, {"s": [0.02, 0, 0]})
    assert m["z"]["sim"] == "s" and m["a"]["sim"] is None and abs(m["a"]["dist"] - 0.03) < 1e-9  # nearest pair first


def test_resample_trajectory_edges():
    one = np.full((1, 7), 0.3, np.float32)
    out = resample_trajectory(one, 0.02, 1 / 30)
    assert out.shape == (1, 7) and out is not one
    two = np.array([[0.0] * 7, [1.0] * 7])
    assert np.array_equal(resample_trajectory(two, 0.01, 1 / 30), two)  # coarser than the trajectory: start + end
    p = np.linspace(0, 1, 11)[:, None].repeat(7, 1)
    assert np.allclose(resample_trajectory(p, 0.02, 0.02), p)  # identity
    for n in (4, 8, 14, 31):
        r = resample_trajectory(np.linspace(0, 1, n)[:, None], 0.1, 0.1)
        assert len(r) == n and not np.allclose(r[-1], r[-2])  # no duplicated end point


def test_parse_plan_skips_metadata_and_rejects_empty_trajectory():
    steps = parse_plan({"steps": [{"type": "metadata", "x": 1}, {"type": "gripper", "action": "open"}]})["steps"]
    assert [s["type"] for s in steps] == ["gripper"]
    with pytest.raises(ValueError):
        parse_plan({"steps": [{"type": "trajectory", "positions": []}]})


def test_load_observation_h5_droid_layout(tmp_path):
    import h5py

    req = _request(gt=False)
    with h5py.File(tmp_path / "droid.h5", "w") as f:
        f.create_dataset("rgb", data=req["rgb"])
        f.create_dataset("depth", data=req["depth"])
        f.create_dataset("intrinsic_matrix", data=req["intrinsics"])
        f.create_dataset("q_init", data=req["q_init"])
        f.create_dataset("pos_w", data=[0.3, 0.0, 0.5])
        f.create_dataset("quat_w_ros", data=[1.0, 0.0, 0.0, 0.0])
    back = load_observation_h5(tmp_path / "droid.h5")
    # the stock loader's 1.5 cm DROID calibration offset applies when the file does not say otherwise
    assert np.allclose(back["world_from_cam"][:3, 3], [0.3, 0.0, 0.515])
    assert np.allclose(back["world_from_cam"][:3, :3], np.eye(3))


def test_canonical_object_name_and_rerun_name():
    for raw in ("candle.n.01_2", "candle_2"):
        assert canonical_object_name(raw) == ("candle", "2")
    assert canonical_object_name("can__of__soda.n.01_1") == ("can_of_soda", "1")
    assert canonical_object_name("candle") == ("candle", "")
    assert canonical_object_name("wicker_basket.n.01") == ("wicker_basket", "")
    assert canonical_object_name("breakfast_table_skczfi_0") == ("breakfast_table_skczfi", "0")
    assert rerun_name("table.n.02_1") == "table_n_02_1" and rerun_name("can of soda/1") == "can_of_soda_1"


def test_face_normal_local_picks_the_nearest_face():
    from omnigibson.tiptop.protocol import face_normal_local

    box = np.array([[-0.069, -0.16, -0.118], [0.069, 0.16, 0.118]])  # the radio's local bounding box
    # the toggle button: 2 mm inside the +x face, far from the others
    assert face_normal_local(box, [0.0447 + 0.0022, 0.0421, -0.0125]).tolist() == [1.0, 0.0, 0.0]
    assert face_normal_local(box, [0.0447, 0.0421, -0.0125]).tolist() == [1.0, 0.0, 0.0]
    assert face_normal_local(box, [0.0, 0.0, 0.117]).tolist() == [0.0, 0.0, 1.0]  # on the top face
    assert face_normal_local(box, [-0.068, 0.1, 0.0]).tolist() == [-1.0, 0.0, 0.0]  # on the -x face
    assert face_normal_local(box, [0.0, -0.159, 0.0]).tolist() == [0.0, -1.0, 0.0]


def test_match_objects_per_object_tolerance():
    from omnigibson.tiptop.protocol import match_objects

    perceived = {"radio_1": [0.70, 0.20, 0.60]}  # a hull centred 7 cm above a 24 cm radio
    simulated = {"radio_receiver_1": [0.77, 0.22, 0.53], "candle_1": [0.30, 0.10, 0.45]}
    assert match_objects(perceived, simulated, 0.08)["radio_1"]["sim"] is None
    tolerance = {"radio_receiver_1": 0.15, "candle_1": 0.08}
    assert match_objects(perceived, simulated, tolerance)["radio_1"]["sim"] == "radio_receiver_1"


def test_compose_views_layout():
    import numpy as np

    from omnigibson.tiptop.executor import compose_views

    head = np.full((720, 720, 3), 10, np.uint8)
    overview = np.full((360, 640, 4), 200, np.uint8)  # rgba from the sensor is cut to rgb
    wrist = np.full((480, 480, 3), 90, np.uint8)
    frame = compose_views({"head_cam": head, "overview": overview, "wrist_cam": wrist}, column_width=560, caption="x")
    assert frame.shape == (720, 1280, 3)
    assert frame[700, 100].tolist() == [10, 10, 10]  # the capture camera fills the left
    assert frame[300, 1000].tolist() == [200, 200, 200]  # overview scaled to 560x315 at the top right
    assert frame[500, 1000].tolist() == [90, 90, 90]  # wrist under it, 405x405 centred in the column
    assert frame[719, 745].tolist() == [0, 0, 0]  # padding beside the centred wrist tile
    only = compose_views({"cam": head})
    assert only.shape == (720, 720, 3)


class _FakeSim:
    """Joint-space stand-in for the executor: the arm reaches every target at once; toggling is scripted."""

    OPEN, CLOSE, dt = 1.0, -1.0, 1 / 30

    def __init__(self, flip_at):
        import numpy as np

        self.q = np.zeros(2, np.float32)
        self.steps, self.flip_at, self.toggled, self.gripper_log = 0, flip_at, False, []
        self.robot = type("R", (), {"is_grasping": staticmethod(lambda: 0)})()

    def step(self, q, gripper):
        import numpy as np

        self.q = np.asarray(q, np.float32)
        self.steps += 1
        self.gripper_log.append(gripper)
        if float(self.q[0]) >= self.flip_at:  # the finger reaches the button part-way through the press segment
            self.toggled = True

    def q_arm(self):
        return self.q

    def q_fingers(self):
        import numpy as np

        return np.zeros(2)

    def hold(self, n, gripper, q_arm=None):
        for _ in range(n):
            self.step(self.q, gripper)


def test_press_stops_only_the_press_segment_and_the_back_off_runs():
    from omnigibson.tiptop.executor import PlanExecutor

    sim = _FakeSim(flip_at=0.15)
    seg = lambda label, a, b: {"type": "trajectory", "label": label, "positions": [[a, a], [b, b]], "dt": 5 * sim.dt}
    plan = {
        "q_init": [0.0, 0.0],
        "steps": [
            {"type": "gripper", "action": "close", "label": "Push(b)"},
            seg("Push(b)", 0.0, 0.1),  # approach: ends before the button
            seg("Push(b)", 0.1, 0.2),  # press: the button flips at 0.15 -> stopped early
            seg("Push(b)", 0.2, 0.1),  # back-off: same label, must run to its end
            {"type": "gripper", "action": "open", "label": "Push(b)"},
        ],
    }
    ex = PlanExecutor(sim, gripper_hold_steps=2, press_done=lambda: sim.toggled)
    stats = ex.execute(plan)
    early = [t["stopped_early"] for t in stats["trajectories"]]
    assert early == [False, True, False]
    assert stats["trajectories"][2]["executed"] == stats["trajectories"][2]["resampled"]  # the back-off completed
    assert abs(float(sim.q[0]) - 0.1) < 1e-6  # and the arm ended at the hover pose
    assert sim.gripper_log[-1] == sim.OPEN and sim.gripper_log[2] == sim.CLOSE


def test_button_tracker_carries_a_detected_button_through_a_grasp():
    import numpy as np

    from omnigibson.tiptop.knowledge import ButtonTracker

    def pose(yaw, xyz):
        c, s = np.cos(yaw), np.sin(yaw)
        mat = np.eye(4)
        mat[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        mat[:3, 3] = xyz
        return mat

    tracker = ButtonTracker()
    detected = {
        "radio_button": {"position": [0.5, 0.1, 0.8], "normal": [-1.0, 0.0, 0.0], "radius": 0.01, "source": "detected"}
    }
    tracker.update(detected, held={})
    assert tracker.current(lambda arm: None)["radio_button"]["position"] == [0.5, 0.1, 0.8]  # free object: unchanged
    close = pose(0.0, [0.5, 0.1, 0.9])  # the gripper closes on the object, 10 cm above the button
    tracker.grasped("radio", "left", close)
    now = pose(np.pi / 2, [0.3, 0.4, 1.0])  # then lifts, moves and turns a quarter turn about z
    current = tracker.current(lambda arm: now if arm == "left" else None)["radio_button"]
    assert np.allclose(current["position"], [0.3, 0.4, 0.9], atol=1e-9)  # still 10 cm below the gripper
    assert np.allclose(current["normal"], [0.0, -1.0, 0.0], atol=1e-9)  # the face turned with it
    given = {"radio_button": {"position": [0, 0, 0], "normal": [1, 0, 0], "radius": 0.01, "source": "given"}}
    tracker.update(given, held={"radio": ("left", now)})
    assert np.allclose(
        tracker.current(lambda arm: now)["radio_button"]["position"], [0.3, 0.4, 0.9]
    )  # a prior echoed back changes nothing


def test_block_grasping_wraps_the_robot_for_both_call_styles():
    from omnigibson.tiptop.scene import TiptopSim

    class Robot:
        def _calculate_in_hand_object(self, arm="default"):
            return ("radio", "link", arm)

    sim = TiptopSim.__new__(TiptopSim)
    sim.robot = Robot()
    sim.block_grasping("right")
    assert sim.robot._calculate_in_hand_object(arm="right") is None  # keyword call, as OmniGibson does
    assert sim.robot._calculate_in_hand_object("right") is None
    assert sim.robot._calculate_in_hand_object(arm="left") == ("radio", "link", "left")
    sim.unblock_grasping()
    assert sim.robot._calculate_in_hand_object(arm="right") == ("radio", "link", "right")
    sim.block_grasping(None)  # every arm
    assert sim.robot._calculate_in_hand_object(arm="left") is None
    sim.unblock_grasping()


def test_executor_keeps_a_closed_gripper_at_the_start_of_a_plan():
    from omnigibson.tiptop.executor import PlanExecutor

    sim = _FakeSim(flip_at=10**9)
    sim.last_gripper = sim.CLOSE  # an object is in the hand from an earlier plan
    executor = PlanExecutor(sim, gripper_hold_steps=1)
    plan = {
        "q_init": np.zeros(1, dtype=np.float32),
        "steps": [
            {
                "type": "trajectory",
                "label": "Place(a, grasp0, p1, table, q1)",
                "positions": np.zeros((3, 1), np.float32),
                "velocities": None,
                "dt": sim.dt,
            },
            {"type": "gripper", "label": "Place(a, grasp0, p1, table, q1)", "action": "open"},
        ],
    }
    executor.execute(plan)
    assert sim.gripper_log[0] == sim.CLOSE and sim.gripper_log[-1] == sim.OPEN
