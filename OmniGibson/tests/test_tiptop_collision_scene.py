"""Collision maps must preserve physical geometry independently of task perception and the viewer."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch as th
import trimesh

from omnigibson.tiptop.r1pro import R1ProSim
from omnigibson.tiptop.scene import TiptopSim
import omnigibson.tiptop.scene as scene_module


def _pose(x=0.0, y=0.0, z=0.0):
    return th.tensor([x, y, z]), th.tensor([0.0, 0.0, 0.0, 1.0])


def test_collision_map_keeps_targets_supports_merged_walls_and_near_base_objects_without_a_count_cap():
    def body(name, lo, hi, category="furniture"):
        return SimpleNamespace(name=name, category=category), np.array(lo), np.array(hi)

    rows = [
        body("merged_walls", [-8, -8, 0], [8, 8, 2.5]),
        body("near_base_sofa", [-0.3, -0.3, 0], [0.6, 0.6, 0.5]),
        body("support_table", [0.5, -0.5, 0], [1.2, 0.5, 0.7]),
        body("target_box", [0.5, 0, 0.7], [1, 0.5, 1.0]),
        body("small_book", [0.6, 0, 0.7], [0.8, 0.1, 0.72], "book"),
        body("raised_mat", [0.3, 0.3, 0], [0.5, 0.5, 0.08], "mat"),
        body("floor", [-8, -8, -0.1], [8, 8, 0], "floors"),
        body("far", [3, 3, 0], [4, 4, 1]),
        *[body(f"near_{i}", [1, 1, 0], [1.1, 1.1, 0.2]) for i in range(30)],
    ]
    sim = SimpleNamespace(
        robot=SimpleNamespace(name="robot"),
        objects={"box_1": rows[3][0], "book_1": rows[4][0]},
        obstacles={"old_stance": object()},
        base_pose=_pose,
        scene_aabbs=lambda: rows,
    )
    names = R1ProSim.nearby_obstacles(sim, collision_map=True, limit=1)
    assert set(names) == {obj.name for obj, _, _ in rows} - {"floor", "far"}
    assert len(names) == 36
    assert "old_stance" not in sim.obstacles


def test_physical_map_preserves_container_opening_and_updates_cached_articulated_links(monkeypatch):
    floor = trimesh.creation.box([1, 1, 0.04])
    wall = trimesh.creation.box([0.04, 1, 0.5])
    wall.apply_translation([0.48, 0, 0.25])
    physical = trimesh.util.concatenate([floor, wall])
    visual = trimesh.creation.box([1, 1, 1])  # deliberately fills the opening
    link_pose = [_pose()]
    meshes = {"physical": physical, "visual": visual}
    reads = []

    def read_mesh(prim, **kwargs):
        reads.append(prim)
        return meshes[prim].copy()

    monkeypatch.setattr(scene_module, "mesh_prim_to_trimesh_mesh", read_mesh)
    link = SimpleNamespace(
        prim_path="/container/door",
        visual_only=False,
        visual_meshes={"visual": SimpleNamespace(prim="visual", purpose="default")},
        collision_meshes={"physical": SimpleNamespace(prim="physical")},
        get_position_orientation=lambda: link_pose[0],
    )
    obj = SimpleNamespace(links={"door": link}, get_position_orientation=_pose, fixed_base=True)
    sim = SimpleNamespace(objects={"box_1": obj}, obstacles={"scene_box": obj}, _link_meshes={})
    sim.link_trimesh_world = MethodType(TiptopSim.link_trimesh_world, sim)
    sim.collision_mesh_world = MethodType(TiptopSim.collision_mesh_world, sim)
    sim.to_base = lambda pos, quat: (pos - th.tensor([2.0, 0.0, 0.0]), quat)
    first = TiptopSim.room_collision_scene(sim)["scene_box"]
    assert reads == ["physical"]
    assert first["task_label"] == "box_1"
    assert first["fixed_base"] is True
    assert len(first["faces"]) == len(physical.faces)
    assert np.allclose(first["vertices"], physical.vertices)
    assert np.allclose(first["pose"][:3, 3], [-2, 0, 0])
    # The cavity centre is empty: no convex hull/OBB, perception hull or visual mesh may replace this mesh.
    mesh = trimesh.Trimesh(first["vertices"], first["faces"], process=False)
    _, distance, _ = trimesh.proximity.closest_point_naive(mesh, [[0, 0, 0.3]])
    assert distance[0] > 0.2
    link_pose[0] = _pose(y=0.4)
    second = TiptopSim.room_collision_scene(sim)["scene_box"]
    assert reads == ["physical"]  # only rigid link geometry is cached
    assert np.allclose(second["vertices"], first["vertices"] + [0, 0.4, 0])


def test_visual_only_geometry_does_not_turn_a_semantic_fillable_volume_into_a_solid_obstacle():
    link = SimpleNamespace(visual_only=True)
    sim = SimpleNamespace()
    assert TiptopSim.link_trimesh_world(sim, link, collision_only=True) is None


def test_collision_map_extraction_failure_does_not_silently_remove_obstacle():
    obj = SimpleNamespace(links={"body": object()})
    sim = SimpleNamespace(objects={}, obstacles={"wall": obj})

    def unavailable(*args, **kwargs):
        raise ValueError("malformed physical mesh")

    sim.collision_mesh_world = unavailable
    with pytest.raises(ValueError, match="malformed physical mesh"):
        TiptopSim.room_collision_scene(sim)


def test_start_collision_is_reported_without_erasing_obstacles_or_commanding_a_lift():
    obstacle = object()
    sim = SimpleNamespace(
        robot=SimpleNamespace(arm_names=["left"], arm_joint_names={"left": ["left_arm_joint1"]}),
        planned_joints=["left_arm_joint1"],
        _stance_ik=lambda arm: object(),
        arm_hits_scene=lambda *args, **kwargs: ["target_box"],
        obstacles={"target_box": obstacle},
    )
    measured = [0.7]
    assert R1ProSim.clear_start_posture(sim, "left", measured) is measured
    assert sim.obstacles == {"target_box": obstacle}


def test_unfold_refuses_known_collision_before_commanding_motion():
    sim = SimpleNamespace(
        robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(1)),
        planned_joints=["left_arm_joint1"],
        joint_index={"left_arm_joint1": 0},
        ramp_collision=lambda *args: ("right_realsense_link", "bookcase", 4),
        posture={},
        last_gripper=1.0,
    )
    sim.ramp_to = MethodType(R1ProSim.ramp_to, sim)
    R1ProSim.unfold_after_travel(sim, [1.0])  # no motor step/hold methods: no command may be issued


def _path_model():
    from omnigibson.tiptop.collision import JointPathCollision

    model = JointPathCollision.__new__(JointPathCollision)
    model.links = np.array(["torso_link4", "right_realsense_link", "left_gripper_link"])
    model.local_centres = np.array([[0.0, 0.0, 1.0], [0.0, -0.5, 1.0], [0.0, 0.5, 1.0]])
    model.radii = np.array([0.05, 0.02, 0.02])
    model.self_ignore = set()
    model.buffer, model.self_buffer = 0.0, {}
    model.attachment_ignore = {"left_gripper_link": {"left_gripper_link"}, "right_gripper_link": set()}
    model.bounds = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=float)

    def fk(q, link):
        side = q[1] if link == "right_realsense_link" else q[2] if link == "left_gripper_link" else 0
        return np.array([q[0] + side, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])

    model.fk = SimpleNamespace(fk=fk)
    return model


def _box(extents, centre):
    box = trimesh.creation.box(extents)
    box.apply_translation(centre)
    return box


def test_ramp_checks_moving_torso_opposite_arm_camera_and_between_clear_endpoints():
    model = _path_model()
    shelf = _box([0.01, 0.03, 0.03], [0.5, -0.5, 1.0])
    # Nominal endpoints are clear; the camera on the opposite arm crosses the shelf during either motion.
    assert model.check([0, 0, 0], [0, 0, 0], np.eye(4), [("shelf", shelf)]) is None
    assert model.check([1, 0, 0], [1, 0, 0], np.eye(4), [("shelf", shelf)]) is None
    for target in ([1, 0, 0], [0, 1, 0]):
        link, obj, sample = model.check([0, 0, 0], target, np.eye(4), [("shelf", shelf)])
        assert (link, obj) == ("right_realsense_link", "shelf")
        assert sample > 0
    assert model.check([0, 0, 0], [0, 0, 1], np.eye(4), [("shelf", shelf)]) is None


def test_ramp_only_allows_named_gripper_target_contact_and_never_support_or_camera_contact():
    model = _path_model()
    target = _box([0.05, 0.05, 0.05], [0, 0.5, 1])
    allowed = {"book": {"left_gripper_link"}}
    assert model.check([0, 0, 0], [0, 0, 0], np.eye(4), [("book", target)], allowed) is None
    assert model.check([0, 0, 0], [0, 0, 0], np.eye(4), [("support", target)], allowed)[:2] == (
        "left_gripper_link", "support"
    )
    camera_target = _box([0.05, 0.05, 0.05], [0, -0.5, 1])
    assert model.check([0, 0, 0], [0, 0, 0], np.eye(4), [("book", camera_target)], allowed)[:2] == (
        "right_realsense_link", "book"
    )


def test_ramp_rejects_deep_start_penetration_and_keeps_open_container_cavity_free():
    model = _path_model()
    solid = _box([3, 3, 3], [0, 0, 1])
    assert model.check([0, 0, 0], [0, 0, 0], np.eye(4), [("solid", solid)])[2] == 0
    floor = _box([3, 3, 0.04], [0, 0, 0])
    wall = _box([0.04, 3, 2], [1.5, 0, 1])
    cavity = trimesh.util.concatenate([floor, wall])
    assert model.check([0, 0, 0], [0.1, 0, 0], np.eye(4), [("container", cavity)]) is None


def test_ramp_preflight_failure_commands_no_motor_step():
    sim = SimpleNamespace(
        robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(1)),
        planned_joints=["left_arm_joint1"],
        joint_index={"left_arm_joint1": 0},
        ramp_collision=lambda *args: ("right_realsense_link", "bookcase", 4),
    )
    # No step/hold methods: geometric refusal must happen before either can execute.
    result = R1ProSim.ramp_to(sim, [0.5], {}, 1.0, 10)
    assert result == ("right_realsense_link intersects bookcase", 0, 0.0, 4)  # ..., the path sample refused at


def test_descendant_prismatic_extension_is_included_in_rotational_sweep_bound(tmp_path):
    from omnigibson.tiptop.collision import JointPathCollision

    urdf = tmp_path / "robot.urdf"
    urdf.write_text('''<robot name="test">
      <link name="base_link"/><link name="rotor"/><link name="slide"/><link name="camera"/>
      <joint name="torso" type="revolute"><parent link="base_link"/><child link="rotor"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
      <joint name="finger" type="prismatic"><parent link="rotor"/><child link="slide"/>
        <origin xyz="1 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
      <joint name="camera_mount" type="fixed"><parent link="slide"/><child link="camera"/>
        <origin xyz="1 0 0"/></joint></robot>''')
    config = tmp_path / "spheres.yml"
    config.write_text("robot_cfg:\n  kinematics:\n    collision_spheres:\n      camera:\n"
                      "        - center: [0.1, 0, 0]\n          radius: 0.01\n")
    model = JointPathCollision(urdf, config, ["torso", "finger"])
    for arm in ("left", "right"):
        ignored = model.attachment_ignore[f"{arm}_gripper_link"]
        assert ignored == {f"{arm}_{suffix}" for suffix in (
            "gripper_link", "gripper_finger_link1", "gripper_finger_link2", "realsense_link", "arm_link6", "arm_link7"
        )}
    extended = model.centres([0, 1], model.links, model.local_centres)[0]
    assert np.allclose(extended, [3.1, 0, 0])
    assert model.bounds[0, 0] >= np.linalg.norm(extended) - 1e-12


def test_held_object_cover_collides_even_when_the_gripper_clears_the_obstacle():
    from omnigibson.tiptop.collision import box_spheres

    model = _path_model()
    model.motion_bounds = lambda links, centres: np.ones((len(links), 3))
    centres, radii = box_spheres([[0.2, 0.48, 0.98], [0.4, 0.52, 1.02]])
    shelf = _box([0.01, 0.1, 0.1], [0.3, 0.5, 1.0])
    assert model.check([0, 0, 0], [0, 0, 0], np.eye(4), [("shelf", shelf)]) is None
    hit = model.check(
        [0, 0, 0], [0, 0, 0], np.eye(4), [("shelf", shelf)],
        attachments=[("left_gripper_link", centres, radii)],
    )
    assert hit[:2] == ("attached_object_left", "shelf")  # the carried volume, not the gripper link it rides on


def test_capture_always_attaches_physical_map_even_when_camera_motion_returns_early():
    requests = []
    physical = {"bookcase": {"fixed_base": True}}
    sim = SimpleNamespace(
        _capture_with_motion=lambda task: ({"q_init": [99.0]}, {"views": {}}),
        restore_locked_arm=lambda: None,
        q_arm=lambda: np.array([0.2]),
        robot=SimpleNamespace(get_joint_positions=lambda: th.tensor([0.2, 0.3])),
        planned_joints=["left_arm_joint1"],
        posture={"right_arm_joint1": 0.0},
        joint_index={"left_arm_joint1": 0, "right_arm_joint1": 1},
        send_room=True,
        nearby_obstacles=lambda **kwargs: requests.append(kwargs),
        room_collision_scene=lambda: physical,
    )
    request, _ = R1ProSim.capture(sim, "pick")
    assert np.allclose(request["q_init"], [0.2])
    assert request["locked_joints"]["right_arm_joint1"] == pytest.approx(0.3)
    assert request["room"] is physical
    assert requests == [{"collision_map": True}]


def test_planned_polyline_preserves_corner_and_checks_each_edge():
    from omnigibson.tiptop.collision import JointPathCollision

    model = JointPathCollision.__new__(JointPathCollision)
    model.links = np.array(["right_realsense_link"])
    model.local_centres = np.zeros((1, 3))
    model.radii = np.array([0.02])
    model.self_ignore = set()
    model.buffer, model.self_buffer = 0.0, {}
    model.bounds = np.ones((1,2))
    model.fk = SimpleNamespace(fk=lambda q, link: (np.array([q[0], q[1], 1.0]), np.array([0, 0, 0, 1])))
    path = [[-1, -1], [-1, 1], [1, 1]]
    middle = _box([0.1, 0.1, 0.1], [0, 0, 1])
    assert model.check_polyline(path, np.eye(4), [("middle", middle)]) is None
    assert model.check(path[0], path[-1], np.eye(4), [("middle", middle)]) is not None
    on_edge = _box([0.1, 0.1, 0.1], [-1, 0, 1])
    assert model.check_polyline(path, np.eye(4), [("edge", on_edge)])[:2] == ("right_realsense_link", "edge")


def test_gripper_event_checks_the_whole_open_close_sweep_with_no_arm_motion():
    from omnigibson.tiptop.collision import JointPathCollision

    model = JointPathCollision.__new__(JointPathCollision)
    model.joint_names = ["finger_joint"]
    model.links = np.array(["left_gripper_finger_link1"])
    model.local_centres = np.zeros((1, 3))
    model.radii = np.array([0.003])
    model.self_ignore = set()
    model.buffer, model.self_buffer = 0.0, {}
    model.bounds = np.ones((1,1))
    model.parents = {"left_gripper_finger_link1": ("base_link", "finger_joint", "prismatic", 0.05)}
    model.joint_axes = {"finger_joint": np.array([1, 0, 0])}
    model.fk = SimpleNamespace(fk=lambda q, link: (np.array([q[0], 0, 0]), np.array([0, 0, 0, 1])))
    wall = _box([0.002, 0.1, 0.1], [0.025, 0, 0])
    for start, end in ((0, 0.05), (0.05, 0)):
        assert model.check_polyline([[start]], np.eye(4), [("wall", wall)]) is None
        hit = model.check_polyline([[start]], np.eye(4), [("wall", wall)], joint_ranges={"finger_joint": (start, end)})
        assert hit[:2] == ("left_gripper_finger_link1", "wall")


def test_planned_validation_uses_live_grasp_state_measured_opposite_arm_and_narrow_contacts():
    held = SimpleNamespace(name="held_book", category="book")
    target = SimpleNamespace(name="pick_book", category="book", aabb=(th.tensor([-1.0, -1, -1]), th.tensor([1.0, 1, 1])))
    support = SimpleNamespace(name="bookcase", category="bookcase")
    floor = SimpleNamespace(name="floor", category="floors")
    lamp = SimpleNamespace(name="lamp", category="lamp")  # in reach, but 20 cm clear of the book's box
    calls = []
    names = ["torso", "left_arm", "right_arm", "left_finger", "right_finger"]
    model = SimpleNamespace(joint_names=names, buffer=0.002, check_polyline=lambda *args, **kwargs: calls.append((args, kwargs)))
    measured = th.tensor([0.1, 0.2, 0.3, 0.04, 0.05])
    robot = SimpleNamespace(
        get_joint_positions=lambda: measured,
        _ag_obj_in_hand={"left": held},
        finger_joint_names={"left": ["left_finger"], "right": ["right_finger"]},
        finger_link_names={"left": ["left_gripper_finger_link1", "left_gripper_finger_link2"]},
        joints={name: SimpleNamespace(lower_limit=0.0, upper_limit=0.05) for name in names[-2:]},
    )
    sim = SimpleNamespace(
        robot=robot,
        _motion_collision_model=lambda: model,
        joint_index={name: i for i, name in enumerate(names)},
        planned_joints=names[:2],
        arm="left", other_arm="right", other_gripper=1.0,
        objects={"book_1": target, "carried_1": held},
        base_pose=_pose,
        scene_aabbs=lambda: [(obj, np.array([-1, -1, -1]), np.array([1, 1, 1])) for obj in (held, target, support, floor)]
        + [(lamp, np.array([1.2, 1.2, 0]), np.array([1.4, 1.4, 0.5]))],
        collision_mesh_world=lambda obj: _box([0.1, 0.1, 0.1], [0, 0, 1]),
    )
    sim._motion_obstacles = MethodType(R1ProSim._motion_obstacles, sim)
    sim._motion_finger_ranges = MethodType(R1ProSim._motion_finger_ranges, sim)
    sim.rests_against = MethodType(R1ProSim.rests_against, sim)
    assert R1ProSim.validate_motion(sim, [[0.1, 0.2], [0.4, 0.5]], -1.0, "Pick(book_1, grasp0)") is None
    (positions, _, obstacles), options = calls[-1]
    assert np.allclose(positions[:, 2], [0.3, 0.3])  # actual opposite arm, never nominal zero
    assert np.allclose(positions[:, :2], [[0.1, 0.2], [0.4, 0.5]])
    assert {name for name, _ in obstacles} == {"pick_book", "bookcase", "floor", "lamp"}
    assert options["joint_ranges"]["left_finger"] == pytest.approx((0.04, 0.0))
    fingers = set(robot.finger_link_names["left"])
    assert options["allowed_contacts"]["pick_book"] == fingers
    # at the grasp the fingertips may meet what the book rests against (the planner lets them, cutamp-19), no more
    assert options["allowed_contacts"]["bookcase"] == fingers
    assert options["allowed_contacts"]["floor"] == {
        "base_link", "wheel_motor_link1", "wheel_motor_link2", "wheel_motor_link3", *fingers
    }
    assert "lamp" not in options["allowed_contacts"]
    assert "attachments" not in options  # planner owns its attachment collision model
    assert options["clearance"] == -model.buffer  # the planner already applied the sphere buffer
    robot._ag_obj_in_hand["left"] = None
    R1ProSim.validate_motion(sim, [[0.1, 0.2]], 1.0, "GoToInitial(q0)")
    assert "held_book" in {name for name, _ in calls[-1][0][2]}  # released object is physical world again
    assert not {"pick_book", "bookcase"} & set(calls[-1][1]["allowed_contacts"])


def test_planned_motion_validation_unavailability_is_an_error_without_steps():
    sim = SimpleNamespace(_motion_collision_model=lambda: 1 / 0)
    error = R1ProSim.validate_motion(sim, [[0]], 1.0, "Pick(book_1)")
    assert "validation unavailable" in error and "ZeroDivisionError" in error


def _press_candidates(mesh, base_position, base_rotation):
    """Record proposed fingertip poses without running IK or moving the simulator."""
    from scipy.spatial.transform import Rotation

    book = SimpleNamespace(name="book", aabb=tuple(th.tensor(v) for v in mesh.bounds))
    candidates = []

    def record(arm, ik, seed, point, into, jaw, **kwargs):
        candidates.append((point, into, jaw))
        return None

    sim = SimpleNamespace(
        scene_object=lambda name: book,
        to_base=lambda point, quat: (th.tensor(base_rotation.T @ (point.numpy() - base_position)), quat),
        base_pose=lambda: (th.tensor(base_position), th.tensor(Rotation.from_matrix(base_rotation).as_quat())),
        collision_mesh_world=lambda obj: mesh,
        arm_ik=lambda *args, **kwargs: object(),
        ik_joint_names=lambda *args, **kwargs: [],
        robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(0)),
        joint_index={},
        scene_aabbs=lambda: [],
        _press_solution=record,
        q_arm=lambda: np.zeros(0), ramp_to=lambda *args, **kwargs: None, posture={}, CLOSE=-1.0,
    )
    assert R1ProSim.press_grasp(sim, "left", "book") is False
    return candidates


@pytest.mark.parametrize("scene_yaw, base_yaw, direction", [(0, 90, [0, 1, 0]), (0, 0, [1, 1, 0]),
                                                            (37, 127, [0, 1, 0])])
def test_pressed_front_contact_and_jaws_stay_in_frame_in_rotated_scenes(scene_yaw, base_yaw, direction):
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler("z", scene_yaw, degrees=True).as_matrix()
    base_rotation = Rotation.from_euler("z", base_yaw, degrees=True).as_matrix()
    centre = np.array([1.0, 2.0, 1.0])
    mesh = trimesh.creation.box([0.04, 0.30, 0.22])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, centre
    mesh.apply_transform(transform)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    into_world = rotation @ direction
    base_position = centre - into_world
    base_position[2] = 0.0
    candidates = _press_candidates(mesh, base_position, base_rotation)
    assert len(candidates) == 4
    # Analytic near face of the box: the oblique ray meets its thin X face, not its support plane.
    distance = 0.15 if direction[0] == 0 else 0.02 / direction[0]
    expected_world = centre - into_world * distance
    for point, into, jaw in candidates[2:]:
        assert np.allclose(base_rotation @ point + base_position, expected_world, atol=1e-6)
        assert np.allclose(base_rotation @ into, into_world, atol=1e-6)
        assert abs(float(jaw @ into)) < 1e-6
    extent = np.ptp(mesh.vertices, axis=0)
    narrow_axis_world = [1, 0, 0] if extent[0] <= extent[1] else [0, 1, 0]
    assert np.allclose(base_rotation @ candidates[0][2], narrow_axis_world, atol=1e-6)
    assert np.allclose(base_rotation @ candidates[0][1], [0, 0, -1], atol=1e-6)


def test_pressed_front_contact_skips_a_ray_through_a_physical_opening():
    left = _box([0.04, 0.30, 0.22], [-0.1, 1.0, 1.0])
    right = _box([0.04, 0.30, 0.22], [0.1, 1.0, 1.0])
    candidates = _press_candidates(trimesh.util.concatenate([left, right]), np.zeros(3), np.eye(3))
    assert len(candidates) == 0  # neither ray hits: no invented surface at the centre or convex bounding box


@pytest.mark.parametrize("command", [-1.0, 1.0])
def test_gripper_only_ramp_is_checked_before_any_motor_command(command):
    checks = []

    def collision(*args):
        checks.append(args)
        return "left_gripper_finger_link1", "bookcase", 3

    sim = SimpleNamespace(
        robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(1)),
        planned_joints=["left_arm_joint1"],
        joint_index={"left_arm_joint1": 0},
        _motion_finger_ranges=lambda measured, gripper: {"left_finger": (0.025, 0.0 if gripper < 0 else 0.05)},
        ramp_collision=collision,
    )
    assert R1ProSim.ramp_to(sim, [0.0], {}, command, 10) == (
        "left_gripper_finger_link1 intersects bookcase", 0, 0.0, 3
    )
    assert checks[0][-1] == command


@pytest.mark.parametrize("failure", ["collision", "unavailable", None])
def test_sticky_grasp_closure_checks_finger_sweep_before_hold(failure):
    book = SimpleNamespace(name="book", links={"body": object()})
    checks, holds = [], []

    def collision(*args):
        checks.append(args)
        if failure == "unavailable":
            raise ValueError("physical geometry unavailable")
        if failure == "collision":
            return "left_gripper_finger_link1", "bookcase", 0
        return None

    sim = SimpleNamespace(
        link_trimesh_world=lambda link: None,
        ramp_collision=collision,
        grasp_contacts=R1ProSim.grasp_contacts,
        hold=lambda *args: holds.append(args),
        robot=SimpleNamespace(_ag_obj_in_hand={"left": book}),
        CLOSE=-1.0,
    )
    seed = [0.2]
    assert R1ProSim.close_on(sim, "left", None, book, "body", np.eye(4), [0, 1, 0], seed, []) == (
        seed, failure is None
    )
    assert len(holds) == (1 if failure is None else 0)
    assert checks[0][3] == R1ProSim.grasp_contacts("left", book)
    assert checks[0][4] == -1.0


def _arm_scene_query(physical, points, *, failure=None):
    class Body:
        name = "furniture"

    body = Body()

    def read(obj):
        if failure is not None:
            raise failure
        return physical

    sim = SimpleNamespace(
        objects={}, hands=lambda: [], arm_points=lambda *args, **kwargs: np.asarray(points),
        collision_mesh_world=read,
        scene_aabbs=lambda: [(body, np.array([-2000.0, -2000.0, -2]), np.array([2000.0, 2000.0, 2]))],
    )
    return R1ProSim.arm_hits_scene(sim, "left", None, [], clearance=0.025)


def test_arm_scene_query_uses_physical_bvh_for_long_thin_triangles(monkeypatch):
    mesh = trimesh.Trimesh([[0, 0, 0], [1000, 0, 0], [0, 0.001, 0]], [[0, 1, 2]], process=False)

    def subdivision_must_not_run(*args, **kwargs):
        raise AssertionError("long triangles must not require subdivision")

    monkeypatch.setattr(trimesh.Trimesh, "subdivide_to_size", subdivision_must_not_run)
    assert _arm_scene_query(mesh, [[10, 0.0001, 0.02], [11, 0.0001, 0.02]]) == ["furniture"]
    assert _arm_scene_query(mesh, [[10, 0.0001, 0.04], [11, 0.0001, 0.04]]) == []
    assert len(mesh.faces) == 1


def test_arm_scene_physical_query_preserves_cavities_and_handles_failures_conservatively():
    physical = trimesh.util.concatenate([
        _box([1, 1, 0.04], [0, 0, 0]),
        _box([0.04, 1, 0.6], [0.48, 0, 0.3]),
    ])
    points = [[-0.1, 0, 0.3], [0.1, 0, 0.3]]
    assert _arm_scene_query(physical, points) == []
    assert _arm_scene_query(physical, [[0.44, -0.1, 0.3], [0.44, 0.1, 0.3]]) == ["furniture"]
    assert _arm_scene_query(None, points) == []  # explicitly visual-only
    assert _arm_scene_query(None, points, failure=ValueError("bad physical mesh")) == ["furniture"]


def test_capture_ramp_does_not_repeat_a_rejected_finger_command_while_settling():
    names = [f"left_arm_joint{i}" for i in range(1, 5)]
    checks = []

    def collision(*args):
        checks.append(args)
        return "left_gripper_finger_link1", "shelf", 0

    sim = SimpleNamespace(
        robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(4), arm_joint_names={"left": names}),
        planned_joints=names,
        joint_index={name: i for i, name in enumerate(names)},
        ramp_collision=collision,
    )
    sim.ramp_to = MethodType(R1ProSim.ramp_to, sim)
    # No step or hold methods: neither the arm path nor a settling gripper command may run after rejection.
    assert R1ProSim.ramp_arms(sim, [0, 0, 0, 0.5], {}, ["left"], -1.0, 5, True) is False
    assert len(checks) == 1
    assert checks[0][-1] == -1.0


def _destination_sim(obstacle, held=None):
    model = _path_model()
    model.joint_names = ["torso", "right", "left"]
    model.motion_bounds = lambda links, centres: np.ones((len(links), 3))
    furniture = SimpleNamespace(name="trash_can", category="furniture")
    meshes = {"trash_can": obstacle}
    bodies = [furniture]
    if held is not None:
        bodies.append(held)
        meshes[held.name] = _box([0.1, 0.1, 0.1], [0.3, 0.5, 1.0])
    sim = SimpleNamespace(
        robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(3), _ag_obj_in_hand={"left": held},
                              links={"left_gripper_link": SimpleNamespace(get_position_orientation=_pose)}),
        joint_index={name: i for i, name in enumerate(model.joint_names)},
        _motion_collision_model=lambda: model,
        base_pose=_pose,
        collision_mesh_world=lambda obj: meshes[obj.name],
        scene_aabbs=lambda: [(obj, *meshes[obj.name].bounds) for obj in bodies],
        objects={}, seen_boxes={},  # no capture has seen the held object: its mesh box, as before
        arm="left", level_held=lambda arm: set(),
    )
    for name in ("_motion_obstacles", "base_placement_collision", "own_box", "carried_volume"):
        setattr(sim, name, MethodType(getattr(R1ProSim, name), sim))
    return sim


def test_teleport_checks_actual_camera_at_rotated_destination_beyond_old_map_crop():
    sim = _destination_sim(_box([0.04, 0.04, 0.04], [5.5, 0.0, 1.0]))
    assert sim._motion_obstacles(set(), {}) == []  # the target is far from the old base
    assert sim.base_placement_collision(5, 0, np.pi / 2)[:2] == ("right_realsense_link", "trash_can")
    assert sim.base_placement_collision(5, 1, np.pi / 2) is None
    sim.fold_for_travel = lambda: setattr(sim, "_fold_blocked", True)
    # Neither teleport nor camera/hold methods exist: a failed fold cannot bypass destination validation.
    from omnigibson.tiptop.r1pro import BasePlacementCollision
    with pytest.raises(BasePlacementCollision, match="right_realsense_link.*trash_can"):
        R1ProSim.place_robot(sim, 5, 0, np.pi / 2)
    assert sim._fold_blocked


def test_teleport_demands_clearance_that_an_ordinary_ramp_does_not():
    from omnigibson.tiptop.r1pro import TELEPORT_CLEARANCE

    model = _path_model()
    model.motion_bounds = lambda links, centres: np.ones((len(links), 3))
    # the torso sphere (radius 0.05 at x=0) stands 2 cm off a wall
    wall = _box([0.02, 1.0, 1.0], [0.08, 0.0, 1.0])
    assert model.check_polyline([[0, 0, 0]], np.eye(4), [("wall", wall)]) is None
    hit = model.check_polyline([[0, 0, 0]], np.eye(4), [("wall", wall)], clearance=TELEPORT_CLEARANCE)
    assert hit[:2] == ("torso_link4", "wall")


def test_a_blocked_unfold_limits_how_far_the_arm_goes_not_whether_the_stance_is_taken():
    sim = _destination_sim(_box([0.04, 0.04, 0.04], [5.5, 0.5, 1.0]))
    sim.planned_joints = ["left"]
    assert sim.base_placement_collision(5, 0, 0.0) is None  # the folded arm lands clear...
    # ...but unfolding the hand forward sweeps it through the obstacle: the unfold stops short of it
    assert sim.base_placement_collision(5, 0, 0.0, then=[1.0])[:2] == ("left_gripper_link", "trash_can")
    sim.q_arm = lambda: [0.0]
    fraction, legs, why = R1ProSim.unfold_reach(sim, [1.0], 5, 0, 0.0)
    assert fraction < 1.0 and "trash_can" in why
    assert legs == ([[fraction]] if fraction else None)


def test_teleport_checks_live_held_volume_and_rejects_unavailable_geometry():
    sim = _destination_sim(_box([0.04, 0.04, 0.04], [4.5, 0.3, 1.0]))
    assert sim.base_placement_collision(5, 0, np.pi / 2) is None
    held = SimpleNamespace(name="carried_can", category="can", fixed_base=False, get_position_orientation=_pose)
    sim = _destination_sim(_box([0.04, 0.04, 0.04], [4.5, 0.3, 1.0]), held)
    assert sim.base_placement_collision(5, 0, np.pi / 2)[:2] == ("attached_object_left", "trash_can")
    sim.fold_for_travel = lambda: None
    sim.collision_mesh_world = lambda obj: None
    with pytest.raises(RuntimeError, match="cannot validate base destination.*no physical mesh"):
        R1ProSim.place_robot(sim, 5, 0, np.pi / 2)


def test_carried_volume_self_collision_covers_edges_and_preserves_grasp_contacts():
    model = _path_model()
    model.motion_bounds = lambda links, centres: np.ones((len(links), 3))
    # The held volume moves with the left hand, crossing the torso between two clear endpoints.
    attachment = ("left_gripper_link", np.array([[0.0, 0.0, 1.0]]), np.array([0.02]))
    for q in ([-0.0, 0, -0.5], [0, 0, 0.5]):
        assert model.check(q, q, np.eye(4), [], attachments=[attachment]) is None
    hit = model.check([0, 0, -0.5], [0, 0, 0.5], np.eye(4), [], attachments=[attachment])
    assert hit[:2] == ("attached_object_left", "torso_link4") and hit[2] > 0
    grasped = ("left_gripper_link", np.array([[0.0, 0.5, 1.0]]), np.array([0.02]))
    assert model.check([0, 0, 0], [0, 0, 0], np.eye(4), [], attachments=[grasped]) is None
    opposite_camera = ("left_gripper_link", np.array([[0.0, -0.5, 1.0]]), np.array([0.02]))
    assert model.check([0, 0, 0], [0, 0, 0], np.eye(4), [], attachments=[opposite_camera])[:2] == (
        "attached_object_left", "right_realsense_link"
    )


def test_carried_objects_in_opposite_hands_are_not_exempt_from_each_other():
    model = _path_model()
    model.motion_bounds = lambda links, centres: np.ones((len(links), 3))
    left = ("left_gripper_link", np.array([[0.0, 0.0, 2.0]]), np.array([0.02]))
    right = ("right_gripper_link", np.array([[0.0, 0.0, 2.0]]), np.array([0.02]))
    assert model.check([0, 0, 0], [0, 0, 0], np.eye(4), [], attachments=[left, right])[:2] == (
        "attached_object_left", "attached_object_right"
    )


def test_stance_retry_avoids_rejected_landings_without_repeating_motor_attempts():
    from omnigibson.tiptop.r1pro import BasePlacementCollision

    target = SimpleNamespace(aabb_center=th.tensor([0.0, 0.0, 0.5]), aabb=(th.zeros(3), th.ones(3)))
    searches, placements, endpoint_checks = [], [], []

    def best(*args, **kwargs):
        searches.append((list(kwargs["avoid"]), list(kwargs["refused"])))
        return (0, float(len(searches)), 0.0, 0.0, [0.5], [0.0], 0.1), {}

    def place(x, y, yaw, **kwargs):
        placements.append(x)
        if x == 1.0:
            raise BasePlacementCollision("camera intersects bin after blocked fold", obstacle="bin")
        return {"x": x, "y": y, "yaw": yaw}

    def check(x, y, yaw):
        endpoint_checks.append(x)
        return ("camera", "bin", 0) if x == 2.0 else None

    sim = SimpleNamespace(
        scene_object=lambda name: target, grasped_labels=lambda: {}, objects={},
        robot_cam=SimpleNamespace(get_position_orientation=_pose), camera_floor_distance=lambda z: 0.4,
        best_base_pose=best, hands=lambda: {}, xy_radius=lambda name: 0.1, place_robot=place, base_placement_collision=check,
        hidden_from_here=lambda names: {}, to_base=lambda *args: args, look_at=lambda *names: None,
    )
    refused = []
    result = R1ProSim.place_robot_for(sim, "target", refused=refused)
    assert result["x"] == 3.0
    assert placements == [1.0, 3.0] and endpoint_checks == [2.0, 3.0]
    # each refusal goes into the next search with its heading and what it met, and out to the caller for a retry;
    # the accepted-stance avoid list is not where they go (a disc there hid every heading at that spot)
    assert searches == [([], []), ([], [(1.0, 0.0, 0.0, "bin")]), ([], [(1.0, 0.0, 0.0, "bin"), (2.0, 0.0, 0.0, "bin")])]
    assert refused == [(1.0, 0.0, 0.0, "bin"), (2.0, 0.0, 0.0, "bin")]


def test_the_landing_checks_verdicts_are_fed_back_into_the_next_search(monkeypatch):
    """68% of the sweep's 2801 landing refusals named an obstacle the same search had already been refused for, and a
    refusal left only an (x, y) disc behind, which hid every other heading at that spot (2026-09-23). A refusal may
    also name a ROBOT link -- "base_link" when a carried tile's volume met the base -- which is no scene object: the
    search crashed looking it up (laying_tile_floors, 2026-09-24); it keeps its disc and blocks nothing else."""
    import omnigibson.tiptop.r1pro as r1pro

    asked, captured = [], {}
    scene = {"kerb": SimpleNamespace(fixed_base=True), "hamper_225": SimpleNamespace(fixed_base=False)}  # a basket
    sim = SimpleNamespace(
        scene_aabbs=lambda: [], head_camera_in_base=lambda: (np.eye(3), np.eye(4), 0.0),
        robot_cam=SimpleNamespace(image_width=720, image_height=720),
        _footprint_free=lambda x, y, ignore, aabbs=None, yaw=None, reaching=(): (True, "free", 0.5),
        base_placement_collision=lambda x, y, yaw, only=None: asked.append((x, only))
        or (("base_link", "kerb", 0) if x < 1.0 else None),
        env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda _key, name: scene.get(name))),
    )
    monkeypatch.setattr(r1pro, "search_base_poses", lambda pts, cam, fp, **kw: captured.setdefault("fp", fp) and (None, {}))
    # a landing refused by the kerb; a short unfold; a landing refused by a movable basket (its mesh is not ours);
    # a landing refused by the robot's own base link (not in the scene at all)
    refused = [(0.0, 0.0, 0.0, "kerb"), (5.0, 5.0, 0.0, None), (7.0, 0.0, 0.0, "hamper_225"),
               (9.0, 0.0, 0.0, "base_link")]
    R1ProSim.best_base_pose(sim, [np.zeros(2)], refused=refused)  # must not raise on "base_link"
    fp = captured["fp"]
    assert fp(0.1, 0.0, np.radians(15)) == (False, "refused before", 0.0)  # the same spot, nearly the same heading
    assert fp(5.1, 5.0, np.radians(-15)) == (False, "refused before", 0.0)
    assert fp(7.1, 0.0, np.radians(-15)) == (False, "refused before", 0.0)  # the basket keeps its avoid disc
    assert fp(9.1, 0.0, np.radians(-15)) == (False, "refused before", 0.0)  # so does the robot's own link
    assert fp(0.1, 0.0, np.radians(45)) == (False, "base_link would intersect kerb", 0.0)  # turned: the kerb is asked
    assert fp(2.0, 0.0, 0.0) == (True, "free", 0.5)
    assert fp(7.1, 0.0, np.radians(45)) == (True, "free", 0.5)
    assert fp(9.1, 0.0, np.radians(45)) == (True, "free", 0.5)  # a robot link is nobody's hard blocker
    # about the kerb only: never the basket, never the robot's link
    assert asked == [(0.1, {"kerb"}), (2.0, {"kerb"}), (7.1, {"kerb"}), (9.1, {"kerb"})]
    captured.clear()
    R1ProSim.best_base_pose(sim, [np.zeros(2)])
    assert captured["fp"](0.1, 0.0, 0.0) == (True, "free", 0.5) and len(asked) == 4  # nothing refused: no check


def test_base_teleport_cannot_detach_fixed_articulated_furniture_as_a_carried_object():
    fixed = SimpleNamespace(name="cabinet", category="cabinet", fixed_base=True)
    sim = _destination_sim(_box([0.04, 0.04, 0.04], [10, 0, 1]), fixed)
    sim.fold_for_travel = lambda: None
    with pytest.raises(RuntimeError, match="cannot teleport while grasping fixed object cabinet"):
        R1ProSim.place_robot(sim, 5, 0, 0)


def _linear_sphere_model(radius):
    from omnigibson.tiptop.collision import JointPathCollision

    model = JointPathCollision.__new__(JointPathCollision)
    model.links = np.array(["left_arm_link6"])
    model.local_centres = np.zeros((1, 3))
    model.radii = np.array([radius])
    model.self_ignore = set()
    model.buffer, model.self_buffer = 0.0, {}
    model.bounds = np.ones((1,1))
    model.fk = SimpleNamespace(fk=lambda q, link: (np.array([q[0], 0, 0]), np.array([0, 0, 0, 1])))
    return model


def test_finer_sweep_certifies_measured_narrow_clearance_without_shrinking_physical_spheres(monkeypatch):
    import omnigibson.tiptop.collision as collision

    # Reduced geometry with the actual trash Pick's arm6 radius and sample232 clearance. The archived exact
    # path replay also passes at 5 mm: cpu_sweep_resolution_probe.py / cpu_sweep_resolution_diagnostic.json.
    radius, clearance = 0.042, 0.0048825638807474606
    model = _linear_sphere_model(radius)
    cabinet = _box([2, 0.01, 1], [0.5, radius + clearance + 0.005, 0])
    resolution = collision.MAX_SWEEP_STEP
    assert resolution == 0.005
    monkeypatch.setattr(collision, "MAX_SWEEP_STEP", 0.01)
    assert model.check([0], [1], np.eye(4), [("cabinet", cabinet)]) is not None
    monkeypatch.setattr(collision, "MAX_SWEEP_STEP", resolution)
    assert model.check([0], [1], np.eye(4), [("cabinet", cabinet)]) is None
    assert model.radii[0] == radius


def test_finer_sweep_still_rejects_a_collision_between_two_clear_sample_endpoints():
    from omnigibson.tiptop.collision import MAX_SWEEP_STEP

    model = _linear_sphere_model(0.0001)
    thin_wall = _box([0.00005, 0.01, 0.01], [MAX_SWEEP_STEP / 2, 0, 0])
    for endpoint in (0, MAX_SWEEP_STEP):
        assert model.check([endpoint], [endpoint], np.eye(4), [("wall", thin_wall)]) is None
    # This edge gets one interval: both sampled centres are clear, but the unchanged Lipschitz inflation
    # covers their entire intervening motion and therefore still refuses to cross the thin wall.
    assert model.check([0], [MAX_SWEEP_STEP], np.eye(4), [("wall", thin_wall)]) is not None


def test_a_contact_the_motion_starts_in_is_excused_while_it_comes_no_closer():
    """bringing_in_wood 302 (2026-09-23): after a floor pick the finger sat 4.6 mm over the floor, inside the sampling
    inflation, and every later motion was refused at sample 0 -- the episode froze with the plank in hand."""
    model = _path_model()
    model.motion_bounds = lambda links, centres: np.ones((len(links), 3))
    wall = _box([0.02, 0.2, 0.2], [0.3, 0.5, 1.0])  # its near face at x = 0.29; the gripper sphere (r 0.02) starts 5 mm in
    start, away, deeper = [0, 0, 0.275], [0, 0, 0.0], [0, 0, 0.285]
    assert model.check(start, away, np.eye(4), [("wall", wall)]) == ("left_gripper_link", "wall", 0)
    assert model.check(start, away, np.eye(4), [("wall", wall)], excuse_start=True) is None
    hit = model.check(start, deeper, np.eye(4), [("wall", wall)], excuse_start=True)
    assert hit[:2] == ("left_gripper_link", "wall") and hit[2] > 0  # deeper is still a hit
    # a lone gripper event has no motion to excuse: its finger sweep is the motion, and it may still hit the wall
    assert model.check_polyline([start], np.eye(4), [("wall", wall)], excuse_start=True) == ("left_gripper_link", "wall", 0)
    # what the start does NOT touch is judged as before: a clear start that runs into the wall is a hit
    hit = model.check(away, deeper, np.eye(4), [("wall", wall)], excuse_start=True)
    assert hit[:2] == ("left_gripper_link", "wall") and hit[2] > 0
    # the carried volume against the robot's own body: the same rule (laying_tile's tile at the base)
    carried = ("left_gripper_link", np.array([[0.0, 0.0, 1.0]]), np.array([0.02]))  # 6 cm from the torso sphere at q2 = 0.06
    assert model.check([0, 0, 0.06], [0, 0, 0.5], np.eye(4), [], attachments=[carried])[:2] == ("attached_object_left", "torso_link4")
    assert model.check([0, 0, 0.06], [0, 0, 0.5], np.eye(4), [], attachments=[carried], excuse_start=True) is None
    assert model.check([0, 0, 0.06], [0, 0, 0.02], np.eye(4), [], attachments=[carried], excuse_start=True)[:2] == (
        "attached_object_left", "torso_link4"
    )
    hit = model.check([0, 0, 0.5], [0, 0, 0.02], np.eye(4), [], attachments=[carried], excuse_start=True)
    assert hit[:2] == ("attached_object_left", "torso_link4") and hit[2] > 0  # from clear into the torso: still a hit


def test_the_carried_volume_is_the_objects_own_box_not_its_gripper_frame_aabb():
    """laying_tile 301 (2026-09-23): a 0.45 x 0.42 m tile at 132 deg to the jaw became a 0.61 x 0.62 m gripper-frame
    AABB and 'intersected' base_link it cleared by 3 cm. The box is the capture's depth points in the tile's own
    frame (never its mesh); an object no capture has seen keeps the gripper-frame box of its mesh, as before."""
    from scipy.spatial.transform import Rotation

    turned = Rotation.from_euler("z", 45, degrees=True)
    mesh = trimesh.creation.box([0.4, 0.2, 0.02])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1]))
    mesh.apply_translation([1.0, 2.0, 0.5])
    pose = (th.tensor([1.0, 2.0, 0.5]), th.tensor(turned.as_quat(), dtype=th.float32))
    tile = SimpleNamespace(name="tile", get_position_orientation=lambda: pose)
    sim = SimpleNamespace(
        objects={"tile_1": tile},
        seen_boxes={"tile_1": (np.array([-0.2, -0.1, -0.01]), np.array([0.2, 0.1, 0.01]))},  # what captures saw, own frame
        collision_mesh_world=lambda obj: mesh,
        robot=SimpleNamespace(links={"left_gripper_link": SimpleNamespace(get_position_orientation=lambda: (
            th.tensor([1.0, 2.0, 0.5]), th.tensor([0.0, 0.0, 0.0, 1.0])))}),
    )
    sim.own_box = MethodType(R1ProSim.own_box, sim)
    link, centres, radii = R1ProSim.carried_volume(sim, tile, "left")
    assert link == "left_gripper_link" and len(centres) < 11 * 11  # 4 cm cells over 0.4 x 0.2, not over the 0.42 x 0.42 AABB
    own = turned.inv().apply(centres)  # gripper frame == world frame here; back into the tile's own frame
    assert np.all(np.abs(own) <= np.array([0.2, 0.1, 0.01]) + 1e-3)  # every sphere centre inside the tile's own box
    assert np.abs(own[:, 0]).max() > 0.15  # ...and spread along its length, not clustered at the centre
    sim.seen_boxes = {}
    _, centres, _ = R1ProSim.carried_volume(sim, tile, "left")
    assert len(centres) == 11 * 11  # unseen: 4 cm cells over the mesh's 0.42 x 0.42 gripper-frame AABB, as before


def test_a_capture_grows_what_it_saw_of_each_object_in_its_own_frame():
    """What the hands carry is the capture's depth under the object's mask (in the competition an object is its
    point cloud, not a mesh): each view's points, brought back by the pose that view rendered the object at, as a
    box in the object's frame that every later capture grows (one capture sees one side: laying_tile 303's wrist
    view of the tile in the hand covered 0.31 x 0.28 m of 0.47 x 0.44)."""
    k = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    depth = np.ones((64, 64), dtype=np.float32)  # the floor, 1 m under a camera looking straight down
    depth[27:37, 22:42] = 0.9  # the top face of a 10 cm box: 20 x 10 pixels = 0.18 x 0.09 m at that range
    depth[30, 30] = 0.0  # a pixel the robot's self-mask zeroed
    mask = np.zeros((64, 64), dtype=bool)
    mask[27:37, 22:42] = True
    box_1 = np.array([[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.05], [0.0, 0.0, 0.0, 1.0]])
    view, extras = {"depth": depth, "intrinsics": k}, {
        "cam_pos_world": [0.0, 0.0, 1.0], "cam_quat_xyzw_world_cv": [1.0, 0.0, 0.0, 0.0],  # OpenCV z down
        "object_pose_mats_at_render": {"box_1": box_1.tolist()},  # turned 90 deg, its origin 5 cm up
    }
    sim = TiptopSim.__new__(TiptopSim)
    sim.seen_boxes = {}
    sim.remember_seen([("head", view, extras)], ["box_1", "unseen_1"], {"head": np.stack([mask, np.zeros_like(mask)])})
    lo, hi = sim.seen_boxes["box_1"]
    assert "unseen_1" not in sim.seen_boxes
    assert lo == pytest.approx([-0.036, -0.081, 0.05], abs=1e-6)  # world x became the box's -y; the zeroed pixel dropped
    assert hi == pytest.approx([0.045, 0.09, 0.05], abs=1e-6)
    raised = box_1.copy()
    raised[2, 3] = 0.06  # the same face seen with the box's origin 1 cm higher: the face is 1 cm lower in its frame
    sim.remember_seen([("head", view, {**extras, "object_pose_mats_at_render": {"box_1": raised.tolist()}})],
                      ["box_1"], {"head": mask[None]})
    assert sim.seen_boxes["box_1"][0] == pytest.approx([-0.036, -0.081, 0.04], abs=1e-6)  # the box grew downward
    assert sim.seen_boxes["box_1"][1] == pytest.approx([0.045, 0.09, 0.05], abs=1e-6)
    sim.remember_seen([("head", view, extras)], ["box_1"], {"head": np.zeros((1, 64, 64), dtype=bool)})
    assert sim.seen_boxes["box_1"][0] == pytest.approx([-0.036, -0.081, 0.04], abs=1e-6)  # unseen: nothing changes


def test_the_landing_check_excuses_a_hand_already_at_the_floor_here():
    """bringing_in_wood 302 (2026-09-23): 48 corridor stances refused for a finger 'in' the floor that it was already
    'in' at the current stance; a teleport of the same posture deepens nothing."""
    from omnigibson.tiptop.r1pro import GROUND_TOP

    model = _path_model()
    model.joint_names = ["torso", "right", "left"]
    model.motion_bounds = lambda links, centres: np.ones((len(links), 3))
    here = SimpleNamespace(name="floors_here", category="floors", fixed_base=True)
    there = SimpleNamespace(name="floors_there", category="floors", fixed_base=True)
    # strips under the gripper sphere (y 0.5) only: 1.5 cm into both, and neither under the torso or the camera
    meshes = {"floors_here": _box([2, 0.2, 0.1], [0.0, 0.5, 0.945]), "floors_there": _box([2, 0.2, 0.1], [5.0, 0.5, 0.945])}
    assert meshes["floors_here"].bounds[1][2] > GROUND_TOP  # the test's floor is up at the gripper; ground by category
    sim = SimpleNamespace(
        arm="left", robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(3), _ag_obj_in_hand={},
                                          finger_link_names={"left": ["left_gripper_finger_link1"]}),
        joint_index={name: i for i, name in enumerate(model.joint_names)}, _motion_collision_model=lambda: model,
        base_pose=_pose, collision_mesh_world=lambda obj: meshes[obj.name],
        scene_aabbs=lambda: [(obj, *meshes[obj.name].bounds) for obj in (here, there)],
    )
    sim._motion_obstacles = MethodType(R1ProSim._motion_obstacles, sim)
    assert R1ProSim.base_placement_collision(sim, 5.0, 0.0, 0.0) is None  # in the floor here too: the stance stands
    meshes["floors_here"].apply_translation([0.0, 0.0, -0.1])  # clear of the floor here: the destination is refused
    assert R1ProSim.base_placement_collision(sim, 5.0, 0.0, 0.0) == ("left_gripper_link", "floors_there", 0)


def test_a_loose_tile_at_the_destination_is_ground_for_the_base_but_not_floor_for_the_hand():
    """laying_tile_floors (2026-09-23): a tile lying flat is ground by height (top 3 cm) but not the map's, and the
    hand-at-the-floor excuse let a carried tile be stood into one. Only fixed-base ground is excused; a paver is."""
    from omnigibson.tiptop.r1pro import GROUND_TOP

    model = _path_model()
    model.joint_names = ["torso", "right", "left"]
    model.motion_bounds = lambda links, centres: np.ones((len(links), 3))
    model.local_centres = np.array([[0.0, 0.0, 1.0], [0.0, -0.5, 1.0], [0.0, 0.5, 0.02]])  # the hand down at the floor
    floor = SimpleNamespace(name="floors_here", category="floors", fixed_base=True)
    tile = SimpleNamespace(name="ceramic_tile_186", category="ceramic_tile", fixed_base=False)
    meshes = {"floors_here": _box([0.3, 0.3, 0.02], [0.0, 0.5, 0.0]),  # top 1 cm: the hand is in it here
              "ceramic_tile_186": _box([0.45, 0.42, 0.03], [5.0, 0.5, 0.015])}  # top 3 cm, lying at the destination
    assert meshes["ceramic_tile_186"].bounds[1][2] < GROUND_TOP
    sim = SimpleNamespace(
        arm="left", robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(3), _ag_obj_in_hand={},
                                          finger_link_names={"left": ["left_gripper_finger_link1"]}),
        joint_index={name: i for i, name in enumerate(model.joint_names)}, _motion_collision_model=lambda: model,
        base_pose=_pose, collision_mesh_world=lambda obj: meshes[obj.name],
        scene_aabbs=lambda: [(obj, *meshes[obj.name].bounds) for obj in (floor, tile)],
    )
    sim._motion_obstacles = MethodType(R1ProSim._motion_obstacles, sim)
    assert R1ProSim.base_placement_collision(sim, 5.0, 0.0, 0.0) == ("left_gripper_link", "ceramic_tile_186", 0)
    tile.fixed_base = True  # a paver: the map's own ground, excused like the floor
    assert R1ProSim.base_placement_collision(sim, 5.0, 0.0, 0.0) is None


def test_a_ramp_that_swings_a_hand_into_the_robots_own_head_is_refused():
    """bringing_in_wood 301 (2026-09-23): the capture swing rolled left_arm_joint3 through 3.7 rad and the fingers
    stopped on zed_link, logged 'in the way: nothing the box test sees'; 43 such swings in 30 episodes. The real
    r1pro spheres and the simulator's own disabled pairs, no scene."""
    import yaml
    from pathlib import Path

    from omnigibson.tiptop.collision import JointPathCollision

    r1pro = Path(__file__).resolve().parents[2] / "datasets/omnigibson-robot-assets/models/r1pro"
    joints = [f"torso_joint{i}" for i in range(1, 5)] + [f"{s}_arm_joint{i}" for i in range(1, 8) for s in ("left", "right")]
    joints += [f"{s}_gripper_finger_joint{i}" for s in ("left", "right") for i in (1, 2)]
    disabled = yaml.safe_load((r1pro / "r1pro.yaml").read_text())["disabled_collision_pairs"]
    model = JointPathCollision(r1pro / "urdf/r1pro.urdf", r1pro / "curobo/r1pro_description_curobo_arm_no_torso.yaml",
                               joints, disabled)
    ready = np.array([1.025, -1.45, -0.47, 0.0] + [0.0] * 14 + [0.05] * 4)
    left = [model.joint_names.index(f"left_arm_joint{i}") for i in range(1, 8)]

    def with_left(arm):
        q = ready.copy()
        q[left] = arm
        return q

    ready = with_left([-1.6312, 0.2636, -1.812, -1.4576, -0.0508, -0.3727, -1.3193])  # r1pro_left's q_home
    look = [-3.076, 0.7612, 1.8993, -1.9185, 1.7689, -0.9253, -0.838]  # wood 301's commanded look posture

    def elbow_first(arm):  # ramp_arms' swing: the elbow alone, then the rest (swing_collision checks the same)
        via = ready.copy()
        via[left[3]] = arm[3]
        return [ready, via, with_left(arm)]

    hit = model.check_polyline(elbow_first(look), np.eye(4), [], excuse_start=True)
    assert hit[:2] == ("left_gripper_finger_link1", "zed_link") and hit[2] > 0
    # the other shoulder branch of the same look (302) swings out to the side, clear of the head
    assert model.check_polyline(elbow_first([0.01, 2.36, -1.19, -1.86, 1.69, -0.85, -0.89]), np.eye(4), []) is None
    # from the stop, the fingers on the head at sample 0, the way back is the robot's own business, not the ramp's
    stuck = with_left([-2.11, 0.43, -0.76, -1.92, 0.41, -0.57, -1.2])
    assert model.check(stuck, ready, np.eye(4), []) is None
    # what the simulator lets pass is not judged: left_arm_link2 sits 8 mm off torso_link4 at ready and closes on it
    # on most clean swings
    assert ("left_arm_link2", "torso_link4") in model.self_ignore and ("left_arm_link1", "left_arm_link2") in model.self_ignore
