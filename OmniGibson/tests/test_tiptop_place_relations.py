"""W3-L3, the bench side of place in and next_to (SPEC §6.2, §4.1): the oracle GeometryProvider's next_to box from the
map's voxels (a static reference) and its cavity from inside_rect, with the voxel compartment that accepts it logged
beside it; the next_to demo cases load."""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

from b1k.connector.types import ObjRef, Provided
from b1k.connector.world import FurniturePiece, VoxelGrid

BENCH = Path(__file__).resolve().parents[2] / "tiptop/b1k/skills/bench"


def _bookcase():
    occupied = np.zeros((15, 40, 50), dtype=bool)  # 0.3 x 0.8 x 1.0 m: boards at 0, 0.3, 0.6, 0.96 m, a back, two sides
    for z in (0, 15, 30, 48):
        occupied[:, :, z:z + 2] = True
    occupied[-1], occupied[:, 0], occupied[:, -1] = True, True, True
    return FurniturePiece(ObjRef("bookcase.n.01_1", "bookcase", True), tuple(map(tuple, np.eye(4))),
                          {"body": VoxelGrid(0.02, (0.0, 0.0, 0.0), occupied)}, ())


def _geometry(inside, asked=None):
    from omnigibson.tiptop.oracle.geometry import OracleGeometry

    obj = SimpleNamespace(aabb=(th.tensor([-0.01, -0.01, 0.0]), th.tensor([0.31, 0.81, 1.2])))  # the visual box: taller

    def rect(item, container, height=None):
        return inside if asked is None else asked.append(height) or inside

    sim = SimpleNamespace(scene_object=lambda n: obj, inside_rect=rect, n_steps=7)
    pieces = {"bookcase.n.01_1": _bookcase()}
    return OracleGeometry(sim, SimpleNamespace(piece=lambda o: Provided(pieces.get(o.id), "map", 7)))


def test_a_static_next_to_reference_is_its_map_voxels_box_and_a_movable_one_its_aabb():
    geometry = _geometry(None)
    box = geometry.extent(ObjRef("bookcase.n.01_1", "bookcase", True))
    assert box.source == "map" and np.allclose(box.value.lo, (0.0, 0.0, 0.0)) and np.allclose(box.value.hi, (0.3, 0.8, 1.0))
    sandal = geometry.extent(ObjRef("sandal.n.01_1", "sandal"))
    assert sandal.source == "oracle" and np.allclose(sandal.value.hi, (0.31, 0.81, 1.2))


def test_the_cavity_is_inside_rects_under_the_roof_of_the_voxel_compartment_accepting_it(caplog):
    """inside_rect's ceiling is the fillable volume's top, no roof (clean_up_your_desk: 0.955-0.962 m under a roof at
    1.308 m: no folder fits, and the side entry turned for a 7 mm 'roof'); the roof is the accepting compartment's."""
    shelf, book, logged = ObjRef("bookcase.n.01_1", "bookcase", True), ObjRef("book.n.02_1", "book"), "accepted by"
    with caplog.at_level(logging.INFO, logger="omnigibson.tiptop.oracle.geometry"):
        got = _geometry(((0.15, 0.4), (0.1, 0.3), 0.35, 0.36)).cavity(shelf, book)
    assert got.source == "oracle" and np.isclose(got.value.floor.z, 0.35) and np.isclose(got.value.top_z, 0.36)
    assert np.isclose(got.value.floor.ceiling, 0.6), "the middle compartment's roof, not the fillable top"
    line = next(r.getMessage() for r in caplog.records if logged in r.getMessage())
    assert "voxel compartments 3" in line and "None" not in line.split(logged)[1], line
    assert _geometry(((0.15, 0.4), (0.1, 0.3), 0.5, 0.52)).cavity(shelf, book).value.floor.ceiling is None, \
        "no compartment has its floor there: no roof known"
    assert _geometry(None).cavity(shelf, book).value is None
    asked = []
    _geometry(None, asked).cavity(shelf, book, 0.3)
    assert asked == [0.3], "the caller's perceived height reaches inside_rect's board choice (0 chose a 7 mm slot)"


def test_the_next_to_demo_cases_load_on_the_floor_next_to_their_reference():
    from b1k.connector.skills import Rel
    from omnigibson.tiptop.host import skillbench

    cases = skillbench.load_cases(BENCH / "demo_place_next_to.yaml")
    assert len(cases) == 6 and all((BENCH / c["demo"]["snapshot"]).exists() for c in cases)
    assert [c["setup"].get("ready") for c in cases].count("left") == 2, "the tidying crouch reset for S1"
    for case in cases:
        rels = case["call"].args.relations
        assert [r.rel for r in rels] == [Rel.ON, Rel.NEXT_TO] and rels[0].target.id == "floor.n.01_1"
        assert {f["pred"] for f in case["success"]} == {"ontop", "nextto"}


def test_the_place_in_cases_load_in_the_right_hand_into_their_containers():
    """SPEC §9's first place-in set as demo situations: the honey drawer, the fancyy cabinet, the tote."""
    from b1k.connector.skills import Rel
    from omnigibson.tiptop.host import skillbench

    cases = skillbench.load_cases(BENCH / "place_in.yaml")
    assert [c["task"] for c in cases] == ["store_honey", "storing_food", "organizing_art_supplies"]
    assert all((BENCH / c["demo"]["snapshot"]).exists() for c in cases)
    for case in cases:
        (r,) = case["call"].args.relations
        assert r.rel is Rel.IN and [f["pred"] for f in case["success"]] == ["inside"]
        held = case["setup"]["held"]["right"]  # jar_of_honey_72 for jar__of__honey.n.01_1
        assert held.startswith(case["call"].args.obj.category.replace("__", "_")), "the right hand holds the placed one"


def test_inside_rect_sizes_the_compartment_by_the_callers_height_over_the_captures():
    """W3-L3: a native place's item was never in the sim's captures, so item_height read 0 and the desk's board choice
    took a 7 mm slot inside the bookcase's board; the caller's perceived height is used when it gives one."""
    from omnigibson.tiptop.r1pro import R1ProSim

    checked = []
    top = th.tensor([0.4, 0.4, 0.3])
    link = SimpleNamespace(is_meta_link=True, meta_link_type="fillable", visual_aabb=(th.zeros(3), top),
                           visual_aabb_extent=top,
                           check_points_in_volume=lambda pts: checked.append(pts) or th.ones(len(pts), dtype=th.bool))
    box = SimpleNamespace(links={"fill": link}, joints={}, fixed_base=False)

    def captured(name):
        raise AssertionError("the captures were asked though the caller knew the height")

    sim = SimpleNamespace(scene_object=lambda n: box, item_height=captured,
                          bay=lambda link, lo, hi, near=None: (np.array([0.2, 0.2]), np.array([0.2, 0.2]), 0.0))
    got = R1ProSim.inside_rect(sim, "cup.n.01_1", "bin.n.01_1", 0.1)
    assert got is not None and np.allclose(checked[0][:, 2], 0.05), "the corners are checked at the item's mid-height"
