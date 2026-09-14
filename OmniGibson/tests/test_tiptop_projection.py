"""The projection against two real captures: what the bridge says an object's pose is, and what the renderer drew.

Every mask the planner is given, every grasp it plans and every placement the runner checks rests on these two
agreeing. They were compared by hand on 2026-09-13 after two annotated images suggested they disagreed by 22 cm
(they do not; the small objects land within 5 px). The fixture makes that a test instead of an afternoon:
``tiptop_capture_fixture.json`` carries the camera, the poses and the rendered mask centroids from two captures of
runs/bench_batteries_8, so no simulator is needed.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from omnigibson.tiptop.protocol import points_to_pixels
from omnigibson.tiptop.r1pro import frame_objects

FIXTURE = Path(__file__).with_name("tiptop_capture_fixture.json")
SMALL_OBJECT_TOLERANCE_PX = 8  # a battery is ~20 px across at these ranges; the pose must land on it


def captures():
    return json.loads(FIXTURE.read_text())["captures"]


def project(capture, label):
    obj = capture["objects"][label]
    px, z = points_to_pixels(
        [obj["aabb_center_base"]], np.asarray(capture["intrinsics"]), np.asarray(capture["world_from_cam"])
    )
    return px[0], float(z[0]), obj


@pytest.mark.parametrize("capture", captures(), ids=[c["round"] for c in captures()])
def test_a_small_objects_pose_projects_onto_the_pixels_the_renderer_drew(capture):
    small = {k: v for k, v in capture["objects"].items() if v["mask_pixels"] < 5000}
    assert small, "the fixture should carry at least one small object per capture"
    for label in small:
        (u, v), ahead, obj = project(capture, label)
        cu, cv = obj["mask_centroid"]
        gap = float(np.hypot(cu - u, cv - v))
        assert ahead > 0, f"{label} projects behind the camera"
        assert gap <= SMALL_OBJECT_TOLERANCE_PX, (
            f"{label}: the pose projects to ({u:.0f}, {v:.0f}) but the renderer drew it at ({cu:.0f}, {cv:.0f}), "
            f"{gap:.0f} px apart -- the poses and the picture have come apart"
        )


@pytest.mark.parametrize("capture", captures(), ids=[c["round"] for c in captures()])
def test_every_object_projects_inside_the_image_when_the_renderer_drew_it_whole(capture):
    h, w = capture["image"]
    for label, obj in capture["objects"].items():
        if obj["mask_pixels"] >= 5000:
            continue  # a large object's box centre and its visible pixels are different points
        (u, v), _, _ = project(capture, label)
        assert 0 <= u < w and 0 <= v < h, f"{label} projects to ({u:.0f}, {v:.0f}), outside a {w}x{h} image"


def test_the_fixture_says_which_run_it_came_from():
    data = json.loads(FIXTURE.read_text())
    assert "bench_batteries_8" in data["_comment"]
    assert len(data["captures"]) >= 2


def test_a_box_partly_behind_the_camera_is_a_cut_not_a_rejection():
    """A tall container the robot must stand close to always has corners behind the camera plane.

    Requiring every corner in front is unsatisfiable there: standing to open a fridge, 4310 of 5832 candidate
    stances were thrown out as "behind the head camera" and no stance was found at all (2026-09-13). Only a box
    entirely behind the camera is a rejection; partly behind is a cut, which the strict pass refuses and the
    fallback pass charges for, exactly as it treats a box cut by the image edge.
    """
    k = np.array([[300.0, 0.0, 360.0], [0.0, 300.0, 360.0], [0.0, 0.0, 1.0]])
    base_from_cam = np.eye(4)  # camera at the base origin, looking down +x with OpenCV axes
    base_from_cam[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    # a tall box straddling the camera plane: it reaches from just ahead of the robot to behind it
    straddling = [np.array([[x, y, z] for x in (-0.3, 0.4) for y in (-0.3, 0.3) for z in (0.0, 1.9)])]
    why, _ = frame_objects(straddling, k, base_from_cam, 0.0, 720, 720, 0.0, 0.0, 0.0, strict=True)
    assert why == "cut by the head camera's near plane"
    why, penalty = frame_objects(straddling, k, base_from_cam, 0.0, 720, 720, 0.0, 0.0, 0.0, strict=False)
    assert why is None, "the fallback pass must accept it, with a penalty"
    assert penalty > 0.0
    # a box entirely behind the camera is still a rejection
    behind = [np.array([[x, y, z] for x in (-1.4, -0.8) for y in (-0.3, 0.3) for z in (0.0, 1.0)])]
    why, _ = frame_objects(behind, k, base_from_cam, 0.0, 720, 720, 0.0, 0.0, 0.0, strict=False)
    assert why == "behind the head camera"
