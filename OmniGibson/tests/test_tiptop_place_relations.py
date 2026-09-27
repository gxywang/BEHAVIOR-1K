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


def _geometry(inside):
    from omnigibson.tiptop.oracle.geometry import OracleGeometry

    obj = SimpleNamespace(aabb=(th.tensor([-0.01, -0.01, 0.0]), th.tensor([0.31, 0.81, 1.2])))  # the visual box: taller
    sim = SimpleNamespace(scene_object=lambda n: obj, inside_rect=lambda item, container: inside, n_steps=7)
    pieces = {"bookcase.n.01_1": _bookcase()}
    return OracleGeometry(sim, SimpleNamespace(piece=lambda o: Provided(pieces.get(o.id), "map", 7)))


def test_a_static_next_to_reference_is_its_map_voxels_box_and_a_movable_one_its_aabb():
    geometry = _geometry(None)
    box = geometry.extent(ObjRef("bookcase.n.01_1", "bookcase", True))
    assert box.source == "map" and np.allclose(box.value.lo, (0.0, 0.0, 0.0)) and np.allclose(box.value.hi, (0.3, 0.8, 1.0))
    sandal = geometry.extent(ObjRef("sandal.n.01_1", "sandal"))
    assert sandal.source == "oracle" and np.allclose(sandal.value.hi, (0.31, 0.81, 1.2))


def test_the_cavity_is_inside_rects_and_the_voxel_compartment_accepting_it_is_logged(caplog):
    shelf, book, logged = ObjRef("bookcase.n.01_1", "bookcase", True), ObjRef("book.n.02_1", "book"), "accepted by"
    with caplog.at_level(logging.INFO, logger="omnigibson.tiptop.oracle.geometry"):
        got = _geometry(((0.15, 0.4), (0.1, 0.3), 0.35, 0.6)).cavity(shelf, book)
    assert got.source == "oracle" and np.isclose(got.value.floor.z, 0.35) and np.isclose(got.value.top_z, 0.6)
    line = next(r.getMessage() for r in caplog.records if logged in r.getMessage())
    assert "voxel compartments 3" in line and "None" not in line.split(logged)[1], line
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="omnigibson.tiptop.oracle.geometry"):
        assert _geometry(None).cavity(shelf, book).value is None
    assert any("inside_rect None" in r.getMessage() for r in caplog.records)


def test_the_next_to_demo_cases_load_on_the_floor_next_to_their_reference():
    from b1k.connector.skills import Rel
    from omnigibson.tiptop.host import skillbench

    cases = skillbench.load_cases(BENCH / "demo_place_next_to.yaml")
    assert len(cases) == 4 and all((BENCH / c["demo"]["snapshot"]).exists() for c in cases)
    for case in cases:
        rels = case["call"].args.relations
        assert [r.rel for r in rels] == [Rel.ON, Rel.NEXT_TO] and rels[0].target.id == "floor.n.01_1"
        assert {f["pred"] for f in case["success"]} == {"ontop", "nextto"}
