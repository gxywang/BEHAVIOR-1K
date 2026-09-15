"""Pure-python tests for the geometric ground-truth masks (no Isaac Sim needed).

A depth image is synthesized by ray-casting trimesh boxes from a pinhole camera; the first hit per pixel defines both
the depth and the reference mask of each box.
"""

import numpy as np
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector

from omnigibson.tiptop.gt_masks import (
    masks_from_geometry,
    meshes_at_view_poses,
    points_within_tol,
    surface_distances,
)

H = W = 96
K = np.array([[120.0, 0.0, 48.0], [0.0, 120.0, 48.0], [0.0, 0.0, 1.0]])
# OpenCV camera 1 m above the origin looking straight down: +x_cam = +x, +y_cam = -y, +z_cam (view) = -z
WORLD_FROM_CAM = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 1.0], [0, 0, 0, 1.0]])


def _box(extents, center):
    return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))


def _render(boxes: dict):
    """Ray-cast the boxes: (depth (H, W) float32 z-depth, {name: first-hit mask})."""
    names = list(boxes)
    scene = trimesh.util.concatenate([boxes[n] for n in names])
    owner = np.repeat(np.arange(len(names)), [len(boxes[n].faces) for n in names])
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    dirs_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1).reshape(-1, 3)
    rot, origin = WORLD_FROM_CAM[:3, :3], WORLD_FROM_CAM[:3, 3]
    dirs = dirs_cam @ rot.T
    origins = np.broadcast_to(origin, dirs.shape)
    tri, ray, loc = RayMeshIntersector(scene).intersects_id(origins, dirs, return_locations=True, multiple_hits=True)
    z = (loc - origin) @ rot[:, 2]  # z-depth along the view axis
    order = np.lexsort((z, ray))  # per ray, nearest hit first
    ray, tri, z = ray[order], tri[order], z[order]
    ray, first = np.unique(ray, return_index=True)
    depth = np.zeros(H * W)
    depth[ray] = z[first]
    hit_owner = np.full(H * W, -1)
    hit_owner[ray] = owner[tri[first]]
    return depth.reshape(H, W).astype(np.float32), {n: (hit_owner == i).reshape(H, W) for i, n in enumerate(names)}


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_masks_from_geometry_two_boxes_one_occluding():
    boxes = {
        "far": _box((0.30, 0.30, 0.10), (0.0, 0.0, 0.30)),  # top at z=0.35: 0.65 m from the camera
        "near": _box((0.10, 0.10, 0.10), (0.06, 0.0, 0.60)),  # floats above the far box's right half, occluding it
        "outside": _box((0.10, 0.10, 0.10), (3.0, 3.0, 0.50)),  # far outside the field of view
        "ground": _box((2.0, 2.0, 0.02), (0.0, 0.0, -0.01)),  # 0.25 m below the far box: no contact halo
    }
    depth, expected = _render(boxes)
    assert expected["far"].sum() > 500 and expected["near"].sum() > 500 and expected["outside"].sum() == 0
    assert (expected["ground"] & ~expected["far"] & ~expected["near"]).sum() > 2000
    # the occluder hides part of the far box: its projected footprint is larger than its visible mask
    assert 0.6 * 2900 < expected["far"].sum() < 2900

    masks = masks_from_geometry(depth, K, WORLD_FROM_CAM, {n: boxes[n] for n in ("far", "near", "outside")})
    assert set(masks) == {"far", "near", "outside"}
    for m in masks.values():
        assert m.shape == (H, W) and m.dtype == bool
    assert _iou(masks["far"], expected["far"]) > 0.98
    assert _iou(masks["near"], expected["near"]) > 0.98
    assert not (masks["far"] & expected["near"]).any()  # occluder pixels are not attributed to the occluded box
    assert not (masks["near"] & expected["far"]).any()
    assert not (masks["far"] & expected["ground"]).any() and not (masks["near"] & expected["ground"]).any()
    assert not masks["outside"].any()


def test_masks_from_geometry_contact_halo_is_thin():
    boxes = {
        "box": _box((0.20, 0.20, 0.10), (0.0, 0.0, 0.05)),  # resting on the ground: contact line at z=0
        "ground": _box((2.0, 2.0, 0.02), (0.0, 0.0, -0.01)),
    }
    depth, expected = _render(boxes)
    mask = masks_from_geometry(depth, K, WORLD_FROM_CAM, {"box": boxes["box"]})["box"]
    assert (expected["box"] & ~mask).sum() == 0  # every visible box pixel is found
    dilated = expected["box"].copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        dilated |= np.roll(np.roll(expected["box"], dy, axis=0), dx, axis=1)
    dilated2 = dilated.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        dilated2 |= np.roll(np.roll(dilated, dy, axis=0), dx, axis=1)
    assert (mask & ~dilated2).sum() == 0  # ground pixels within tol of the bottom edge: at most a 2 px halo
    assert _iou(mask, expected["box"]) > 0.9


def test_masks_from_geometry_invalid_depth_and_empty_mesh():
    box = _box((0.20, 0.20, 0.10), (0.0, 0.0, 0.30))
    depth, expected = _render({"box": box})
    depth[:, : W // 2] = 0.0  # invalid pixels never belong to an object
    masks = masks_from_geometry(depth, K, WORLD_FROM_CAM, {"box": box, "empty": trimesh.Trimesh()})
    assert not masks["box"][:, : W // 2].any() and masks["box"][:, W // 2 :].sum() == expected["box"][:, W // 2 :].sum()
    assert not masks["empty"].any()
    assert masks_from_geometry(np.zeros((H, W)), K, WORLD_FROM_CAM, {"box": box})["box"].sum() == 0


def test_points_within_tol_exact_distance():
    box = _box((0.20, 0.20, 0.20), (0.0, 0.0, 0.0))
    tol = 0.008
    pts = np.array(
        [
            [0.0, 0.0, 0.1 + 0.005],  # 5 mm above the top face -> hit (no vertex nearby: triangle path)
            [0.0, 0.0, 0.1 + 0.012],  # 12 mm above -> miss
            [0.1 + 0.004, 0.1 + 0.004, 0.1 + 0.004],  # 6.9 mm from a corner -> hit (vertex path)
            [0.0, 0.0, 0.0],  # inside the box, 10 cm from every face -> miss (surface distance, not containment)
            [0.05, 0.05, 0.1],  # on the face -> hit
        ]
    )
    assert points_within_tol(box, pts, tol).tolist() == [True, False, True, False, True]
    # a mesh with huge triangles is subdivided, not slowed down: same answers
    big = _box((2.0, 2.0, 0.02), (0.0, 0.0, -0.11))  # top face at z=-0.10
    assert points_within_tol(big, np.array([[0.7, -0.4, -0.095], [0.7, -0.4, -0.08]]), tol).tolist() == [True, False]


def test_masks_from_geometry_objects_in_contact_are_disjoint():
    # 'top' rests on 'bottom': the contact band of each lies within tol of the other's surface
    boxes = {
        "bottom": _box((0.30, 0.30, 0.10), (0.0, 0.0, 0.05)),
        "top": _box((0.12, 0.12, 0.08), (0.02, 0.0, 0.14)),
        "ground": _box((2.0, 2.0, 0.02), (0.0, 0.0, -0.01)),  # 'bottom' rests on it too
    }
    depth, expected = _render(boxes)
    assert expected["top"].sum() > 200 and (expected["bottom"] & ~expected["top"]).sum() > 1000
    masks = masks_from_geometry(depth, K, WORLD_FROM_CAM, {n: boxes[n] for n in ("bottom", "top", "ground")})
    assert not (masks["bottom"] & masks["top"]).any()  # each contested pixel goes to the closest surface
    assert not (masks["bottom"] & masks["ground"]).any() and not (masks["top"] & masks["ground"]).any()
    for name in ("bottom", "top", "ground"):
        assert _iou(masks[name], expected[name]) > 0.98, name
    assert (masks["bottom"] & expected["ground"]).sum() == 0
    # the closest-surface rule absorbs the contact halo of a labelled support surface: at tol 3 cm (from this camera
    # the first ground pixel beyond the footprint is 17-25 mm from the side face, so a 1-2 px ring lies within tol)
    # the unlabelled ground leaks into 'bottom' ...
    alone = masks_from_geometry(depth, K, WORLD_FROM_CAM, {"bottom": boxes["bottom"]}, tol=0.03)["bottom"]
    assert (alone & expected["ground"]).sum() > 0
    # ... while the labelled ground keeps every one of its pixels
    both = masks_from_geometry(depth, K, WORLD_FROM_CAM, {n: boxes[n] for n in ("bottom", "ground")}, tol=0.03)
    assert (both["bottom"] & expected["ground"]).sum() == 0 and not (both["bottom"] & both["ground"]).any()


def test_surface_distances_exact():
    box = _box((0.20, 0.20, 0.20), (0.0, 0.0, 0.0))
    tol = 0.008
    pts = np.array(
        [
            [0.0, 0.0, 0.1 + 0.005],  # 5 mm above the top face (triangle path)
            [0.1 + 0.003, 0.0, 0.0],  # 3 mm outside the +x face
            [0.1 + 0.004, 0.1 + 0.004, 0.1 + 0.004],  # 6.93 mm from a corner
            [0.0, 0.0, 0.1 + 0.012],  # 12 mm above -> beyond tol
            [0.0, 0.0, 0.0],  # inside, 10 cm from every face -> beyond tol
        ]
    )
    d = surface_distances(box, pts, tol)
    assert np.allclose(d[:3], [0.005, 0.003, 0.004 * np.sqrt(3)], atol=1e-9)
    assert np.isinf(d[3:]).all()
    assert (
        surface_distances(trimesh.Trimesh(), pts, tol).shape == (5,) and surface_distances(box, pts[:0], tol).size == 0
    )


def test_preparing_a_mesh_is_cached_so_a_repeated_query_does_not_subdivide_it_again():
    """The stance and look-pose ranking asks about the same desk once per candidate pose.

    Preparation subdivides oversized triangles -- a table top is two big triangles that become thousands -- and
    doing that per query took a putting_away_toys round from about 150 s to 880 s (2026-09-13).
    """
    import time

    import trimesh

    from omnigibson.tiptop.gt_masks import _prepare, points_within_tol

    # one big quad: two triangles with 4 m edges, which subdivision turns into a great many
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], faces=[[0, 1, 2], [0, 2, 3]], process=False
    )
    points = np.array([[1.0, 1.0, 0.5], [2.0, 2.0, 0.001]])
    first = time.perf_counter()
    points_within_tol(mesh, points, 0.05)
    cold = time.perf_counter() - first
    second = time.perf_counter()
    for _ in range(5):
        points_within_tol(mesh, points, 0.05)
    warm = (time.perf_counter() - second) / 5
    assert _prepare(mesh) is _prepare(mesh), "the same prepared mesh comes back"
    assert warm < cold, f"a repeated query should reuse the preparation (cold {cold:.3f}s, warm {warm:.3f}s)"


def _at(center):
    return trimesh.transformations.translation_matrix(center)


def test_a_view_that_saw_the_object_elsewhere_masks_it_where_it_was():
    """The held-object bug: a capture builds one mesh per object, at the pose it has after every view has been
    rendered, but the torso turns between head views and whatever is in the gripper travels with it. Masking a
    turned view with the later mesh finds nothing; moving the mesh back to where that view saw it finds the object.
    """
    seen_at, built_at = (0.0, 0.0, 0.50), (0.28, 0.0, 0.50)  # the view saw it here; the capture's mesh is 28 cm away
    boxes = {"held": _box((0.10, 0.10, 0.10), seen_at), "ground": _box((2.0, 2.0, 0.02), (0.0, 0.0, -0.01))}
    depth, expected = _render(boxes)
    assert expected["held"].sum() > 300

    stale = {"held": _box((0.10, 0.10, 0.10), built_at)}
    stale["held"].metadata = {"world_from_obj": _at(built_at)}
    assert masks_from_geometry(depth, K, WORLD_FROM_CAM, stale)["held"].sum() == 0  # the bug: an empty mask

    fixed = meshes_at_view_poses(stale, {"held": _at(seen_at)})
    assert _iou(masks_from_geometry(depth, K, WORLD_FROM_CAM, fixed)["held"], expected["held"]) > 0.95


def test_meshes_at_view_poses_leaves_alone_what_did_not_move():
    """Identity for everything the robot is not carrying, and the mesh object itself is passed straight through."""
    mesh = _box((0.1, 0.1, 0.1), (0.2, 0.0, 0.5))
    mesh.metadata = {"world_from_obj": _at((0.2, 0.0, 0.5))}
    out = meshes_at_view_poses({"still": mesh}, {"still": _at((0.2, 0.0, 0.5))})
    assert out["still"] is mesh


def test_meshes_at_view_poses_passes_through_what_it_cannot_correct():
    """A label this view recorded no pose for, and a mesh with no build pose (an obstacle), are both left alone."""
    tagged = _box((0.1, 0.1, 0.1), (0.0, 0.0, 0.5))
    tagged.metadata = {"world_from_obj": _at((0.0, 0.0, 0.5))}
    untagged = _box((0.1, 0.1, 0.1), (1.0, 0.0, 0.5))
    out = meshes_at_view_poses({"a": tagged, "b": untagged}, {"b": _at((2.0, 0.0, 0.5))})
    assert out["a"] is tagged and out["b"] is untagged


def test_meshes_at_view_poses_applies_a_rotation_not_just_a_shift():
    """The correction is a rigid motion: an object turned in the gripper is masked turned, not merely moved."""
    built = _at((0.0, 0.0, 0.5))
    seen = trimesh.transformations.rotation_matrix(np.pi / 2, (0, 0, 1), (0.0, 0.0, 0.5)) @ built
    mesh = _box((0.40, 0.05, 0.05), (0.0, 0.0, 0.5))
    mesh.metadata = {"world_from_obj": built}
    moved = meshes_at_view_poses({"bar": mesh}, {"bar": seen})["bar"]
    assert moved.extents[0] < 0.1 < moved.extents[1]  # the long axis is now y, and the mesh itself is untouched
    assert mesh.extents[0] > 0.3


def test_meshes_at_view_poses_rejects_a_pos_quat_pair_instead_of_a_matrix():
    """The capture keeps two different things called a pose: 4x4 matrices per view, and the (pos, quat) pairs of
    extras['object_poses_world']. Feeding the pair in here used to die inside numpy, several frames away."""
    import pytest

    mesh = _box((0.1, 0.1, 0.1), (0.0, 0.0, 0.5))
    mesh.metadata = {"world_from_obj": _at((0.0, 0.0, 0.5))}
    pair = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="must be 4x4 matrices"):
        meshes_at_view_poses({"held": mesh}, {"held": pair})


def test_a_turn_far_from_the_world_origin_is_not_reported_as_a_huge_move():
    """motion[:3, 3] is the motion's translation about the world origin, so a pure rotation of an object standing
    9 m out has a metres-long translation component. What decides 'it moved' has to be the object's own travel."""
    built = _at((9.0, 0.0, 0.5))
    seen = trimesh.transformations.rotation_matrix(np.pi / 4, (0, 0, 1), (9.0, 0.0, 0.5)) @ built
    mesh = _box((0.30, 0.05, 0.05), (9.0, 0.0, 0.5))
    mesh.metadata = {"world_from_obj": built}
    moved = meshes_at_view_poses({"bar": mesh}, {"bar": seen})["bar"]
    motion = np.asarray(seen) @ np.linalg.inv(np.asarray(built))
    assert np.linalg.norm(motion[:3, 3]) > 3.0  # the misleading number
    assert np.linalg.norm(moved.bounds.mean(axis=0) - np.array([9.0, 0.0, 0.5])) < 0.02  # it stayed put and turned
