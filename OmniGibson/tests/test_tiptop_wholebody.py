"""Whole-body simulator boundary with a fake host: no Isaac application or planner server."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th
import yaml
from scipy.spatial.transform import Rotation

from b1k.observation import PROPRIO_SLICES
from b1k.connector.types import Provided
from omnigibson.tiptop.host.bench_host import BenchHost
from omnigibson.tiptop.host.skillbench import parse_args
from omnigibson.tiptop.r1pro import R1ProSim, make_r1pro_env_config
from omnigibson.tiptop.wholebody import BASE_JOINTS, physical_embodiment, validate_planner_metadata


def metadata(arm="left"):
    names = list(BASE_JOINTS) + [f"torso_joint{i}" for i in range(1, 5)]
    names += [f"{arm}_arm_joint{i}" for i in range(1, 8)]
    embodiment = {"robot_type": f"r1pro_{arm}_wholebody", "arm": arm, "joint_names": names,
                  "virtual_base_joints": list(BASE_JOINTS), "q_home": [0.0] * len(names), "locked_joints": {}}
    return {"robot_type": embodiment["robot_type"], "dof": len(names), "embodiment": embodiment,
            "capabilities": {"whole_body": True}}


@pytest.mark.parametrize("arm", ["left", "right"])
def test_virtual_coordinates_do_not_enter_robot_joint_indices_or_mutate_planner_metadata(arm):
    meta = metadata(arm)
    before = copy.deepcopy(meta)
    validate_planner_metadata(meta, arm)
    physical = physical_embodiment(meta["embodiment"])
    names = physical["joint_names"]
    assert len(names) == 11 and names[0] == "torso_joint1" and len(physical["q_home"]) == 11
    assert physical["base_link"] == "base_link" and "virtual_base_joints" not in physical
    sim = SimpleNamespace(joint_index={name: i for i, name in enumerate(names)}, OPEN=1.0,
                          robot=SimpleNamespace(gripper_control_idx={arm: [27, 28]}))
    R1ProSim.reset_embodiment(sim, meta["embodiment"])
    assert sim.planned_joints == names and sim.arm_idx.tolist() == list(range(11))
    assert meta == before


@pytest.mark.parametrize("kind", ["capability", "virtual", "order", "unknown", "dof", "nonzero_home", "arm"])
def test_incompatible_whole_body_metadata_is_refused(kind):
    meta = metadata()
    if kind == "capability":
        meta["capabilities"] = {}
    elif kind == "virtual":
        meta["embodiment"].pop("virtual_base_joints")
    elif kind == "order":
        meta["embodiment"]["joint_names"][:2] = ["base_y", "base_x"]
    elif kind == "unknown":
        meta["embodiment"]["joint_names"][-1] = "not_a_robot_joint"
    elif kind == "dof":
        meta["dof"] = 11
    elif kind == "nonzero_home":
        meta["embodiment"]["q_home"][0] = 0.1
    else:
        meta["embodiment"]["arm"] = "right"
    with pytest.raises(ValueError):
        validate_planner_metadata(meta, "left")


def test_mobile_base_controller_matches_the_actual_evaluator_config():
    path = Path(__file__).resolve().parents[1] / "omnigibson/eval/r1pro.yaml"
    expected = yaml.safe_load(path.read_text())["controller_config"]["base"]
    config = make_r1pro_env_config(whole_body=True)
    assert config["robots"][0]["controller_config"]["base"] == expected
    assert make_r1pro_env_config()["robots"][0]["controller_config"]["base"]["motor_type"] == "position"


def test_measured_pose_velocity_tilt_and_capture_frame_are_source_tagged_and_not_commands():
    proprio = {k: th.zeros(s.stop - s.start) for k, s in PROPRIO_SLICES.items()}
    proprio["base_qvel"][:] = th.tensor([0.1, -0.2, 0.3])
    pos, quat = np.array([1.0, 2.0, 0.017]), Rotation.from_euler("xyz", [0.02, -0.03, 1.2]).as_quat()
    sent = []
    sim = SimpleNamespace(n_steps=7, robot=SimpleNamespace(_get_proprioception_dict=lambda: proprio),
                          base_pose=lambda: (pos, quat), step_action=lambda action: sent.append(action.copy()))
    host = BenchHost(sim, whole_body=True)
    obs = host.observe_now()
    assert sim.n_steps == 7 and sent == [], "reading localization consumes no simulation steps"
    state, transform = obs.base_state, obs.raw["map_from_base"]
    assert state.source == transform.source == "oracle" and state.step == transform.step == 7
    assert (state.value.pose.x, state.value.pose.y, state.value.pose.z) == pytest.approx(pos)
    assert (state.value.roll, state.value.pitch, state.value.pose.yaw) == pytest.approx([0.02, -0.03, 1.2])
    assert state.value.velocity == pytest.approx([0.1, -0.2, 0.3]), "the evaluator reports body-frame velocity"
    assert np.allclose(transform.value[:3, :3], Rotation.from_quat(quat).as_matrix())
    assert not transform.value.flags.writeable
    logical = host.parse(obs.raw, 3)
    assert logical.base_state.step == logical.raw["map_from_base"].step == logical.step == 3
    assert obs.base_state.step == 7, "clock translation leaves the original measurement unchanged"
    synced = host.observe_at(0)
    assert synced.step == synced.base_state.step == synced.raw["map_from_base"].step == 0
    assert synced.raw["sim_step"] == 7 and sim.n_steps == 7, "capture sync reads fresh without another sim step"
    action = np.zeros(23, dtype=np.float32)
    action[:3] = [0.4, 0.2, -0.1]
    action[6:13] = np.arange(7) / 10
    raw = host.env_step(action)
    assert np.array_equal(sent[0], action) and raw["base_state"] == state
    assert host.base_trace[0]["base_action"].tolist() == pytest.approx(action[:3])
    assert host.base_trace[0]["action23"].shape == (23,)
    assert host.base_trace[0]["action23"] == pytest.approx(action)
    assert host.base_trace[0]["proprio"].shape == (61,)
    assert host.base_trace[0]["proprio"] == pytest.approx(raw["proprio"])
    assert json.loads(json.dumps(host.base_trace, default=lambda x: x.tolist()))[0]["base_state"]["source"] == "oracle"
    action[6] = 99
    raw["proprio"][0] = 99
    assert host.base_trace[0]["action23"][6] == 0
    assert host.base_trace[0]["proprio"][0] == pytest.approx(0.1)
    pos[0] = 3.0
    assert transform.value[0, 3] == 1.0 and host.observe_now().base_state.value.pose.x == 3.0
    sim.n_steps = 8
    with pytest.raises(ValueError, match="stale"):
        host.parse(raw, 4)


def test_whole_body_cli_selects_native_backend_and_rejects_legacy():
    args = parse_args(["--case", "cases.yaml", "--out-dir", "out", "--whole-body"])
    assert args.whole_body and args.backend == "tiptop" and args.no_state_stream
    with pytest.raises(SystemExit):
        parse_args(["--case", "cases.yaml", "--out-dir", "out", "--whole-body", "--backend", "legacy"])


def test_each_trial_trace_excludes_setup_and_previous_trial_steps(monkeypatch):
    from omnigibson.tiptop.host import skillbench

    proprio = {k: th.zeros(s.stop - s.start) for k, s in PROPRIO_SLICES.items()}
    sim = SimpleNamespace(n_steps=0, robot=SimpleNamespace(_get_proprioception_dict=lambda: proprio),
                          base_pose=lambda: (np.array([1., 2., .01]), np.array([0., 0., 0., 1.])))
    sim.step_action = lambda action: setattr(sim, "n_steps", sim.n_steps + 1)
    host = BenchHost(sim, whole_body=True)
    monkeypatch.setattr(skillbench, "Runtime", lambda *a, **kw: object())
    monkeypatch.setattr(skillbench, "SkillRegistry", lambda *a, **kw: object())
    monkeypatch.setattr(skillbench, "DirectConnector", lambda *a, **kw: object())
    action = np.zeros(23, dtype=np.float32)
    action[:3] = [0.2, 0.1, -0.1]

    def execute(*args):
        assert host.base_trace == [], "no setup or previous-trial trace may enter the trial"
        host.env_step(action)

    monkeypatch.setattr(skillbench, "one_call", execute)
    trials = []
    for _ in range(2):
        host.env_step(np.zeros(23))  # setup/restore action outside the scored skill
        skillbench.run_trial(host, None, {}, {}, None, None)
        trials.append(copy.deepcopy(host.base_trace))  # the harness serializes a trial immediately after it ends
    assert trials[0] is not trials[1]
    assert [len(trace) for trace in trials] == [1, 1]
    assert [trace[0]["step"] for trace in trials] == [2, 4]
    for trace in trials:
        assert trace[0]["base_state"]["source"] == trace[0]["map_from_base"]["source"] == "oracle"
        assert trace[0]["base_state"]["step"] == trace[0]["step"]
        assert trace[0]["base_action"] == pytest.approx(action[:3])


def test_capture_rebases_distinct_render_times_to_one_after_base_and_preserves_map_cloud():
    from b1k.bridge.protocol import depth_to_points
    from b1k.connector.observe import ObserveRequest
    from b1k.connector.types import ObjRef
    from b1k.observation import pose_to_matrix
    from omnigibson.tiptop.host.capture_observer import CaptureObserver

    def pose(xyz, angles):
        return pose_to_matrix(xyz, Rotation.from_euler("xyz", angles).as_quat())

    render_bases = [pose([1., 2., .01], [0., 0., .1]), pose([1.1, 2.2, .03], [.01, -.02, .25])]
    cameras = [pose([1.4, 2.2, 1.2], [.1, -.2, .3]), pose([1.7, 2.5, .8], [-.2, .3, -.4])]
    after_base = pose([1.3, 2.4, .05], [.03, -.04, .45])
    views, camera_extras = [], {}
    for name, base, camera in zip(("head", "left_wrist"), render_bases, cameras):
        views.append({"name": name, "rgb": np.zeros((2, 2, 3), np.uint8), "depth": np.ones((2, 2)),
                      "intrinsics": np.eye(3), "world_from_cam": np.linalg.inv(base) @ camera})
        camera_extras[name] = {"cam_pos_world": camera[:3, 3].tolist(),
                              "cam_quat_xyzw_world_cv": Rotation.from_matrix(camera[:3, :3]).as_quat().tolist()}
    request = {**views[0], "view_name": "head", "views": [views[1]]}
    extras = {**camera_extras["head"], "views": {"left_wrist": camera_extras["left_wrist"]}}
    proprio = {k: th.zeros(s.stop - s.start) for k, s in PROPRIO_SLICES.items()}
    sim = SimpleNamespace(n_steps=0, primary_view="head", extra_views=("left_wrist",),
                          robot=SimpleNamespace(_get_proprioception_dict=lambda: proprio), look_at=lambda *a: None,
                          tracked_label=lambda obj: obj,
                          base_pose=lambda: (after_base[:3, 3], Rotation.from_matrix(after_base[:3, :3]).as_quat()))

    def capture(task):
        sim.n_steps += 9  # render at distinct bases, then step while returning the arm before observe_now
        return request, extras

    sim.capture = capture
    masks = {name: {"apple_1": np.ones((2, 2), bool)} for name in ("head", "left_wrist")}
    segmenter = SimpleNamespace(masks=lambda *a: Provided(masks, "oracle", sim.n_steps))
    host = BenchHost(sim, whole_body=True)
    observer = CaptureObserver(sim, host, segmenter, "pick apple")
    req = ObserveRequest((ObjRef("apple_1", "apple"),), views=observer.views)
    with pytest.raises(StopIteration) as done:
        next(observer.observe(req, host.observe_now()))
    percept, after = done.value.value
    assert percept.map_from_base.source == "oracle"
    assert percept.map_from_base.step == percept.info.step == after.step == 9
    for raw, camera, view in zip(views, cameras, percept.views.values()):
        reconstructed = percept.map_from_base.value @ view.base_from_cam
        assert np.allclose(reconstructed, camera, atol=1e-6)
        expected_cloud = depth_to_points(view.depth, view.intrinsics, camera)
        actual_cloud = depth_to_points(view.depth, view.intrinsics, reconstructed)
        assert np.allclose(actual_cloud, expected_cloud, atol=1e-6)
        assert not np.allclose(raw["world_from_cam"], view.base_from_cam), "after-base must not label an old frame"
        assert not view.base_from_cam.flags.writeable
    extras["views"]["left_wrist"].pop("cam_pos_world")
    with pytest.raises(ValueError, match="left_wrist lacks render-time"):
        next(observer.observe(req, after))


def test_frame_transform_is_captured_with_the_images_and_stale_lazy_carriers_are_refused():
    proprio = {k: th.zeros(s.stop - s.start) for k, s in PROPRIO_SLICES.items()}
    rendered = []
    frame = {"rgb": np.zeros((2, 2, 3), np.uint8), "depth": np.ones((2, 2)),
             "intrinsics": np.eye(3), "world_from_cam": np.eye(4)}
    sim = SimpleNamespace(n_steps=10, robot=SimpleNamespace(_get_proprioception_dict=lambda: proprio),
                          base_pose=lambda: (np.array([1., 2., .01]), np.array([0., 0., 0., 1.])),
                          primary_view="head", extra_views=(), robot_cam_names={"head": "h"},
                          view_frame=lambda name: (rendered.append(name) or dict(frame), {}),
                          objects={"apple_1": 1}, bddl_names={"apple_1": "apple.n.01_1"})
    segmenter = SimpleNamespace(masks=lambda *a: Provided({"head": {"apple_1": np.ones((2, 2), bool)}}, "oracle", 10))
    host = BenchHost(sim, segmenter, whole_body=True)
    obs = host.parse(host.raw(), 3)
    assert rendered == []
    transform = obs.sensors.map_from_base
    assert rendered == ["head"] and transform.step == 3 and transform.source == "oracle"
    assert transform.value[:3, 3] == pytest.approx([1., 2., .01])
    stale = host.parse(host.raw(), 3).sensors
    sim.n_steps += 1
    with pytest.raises(ValueError, match="observation step"):
        stale.map_from_base
    assert obs.sensors.map_from_base is transform, "already captured image/transform pairs remain immutable"
