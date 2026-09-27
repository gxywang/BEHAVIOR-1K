"""W3-L4, the bench side of place under and touching (SPEC §6.2, §4.1): the oracle GeometryProvider's overhang and
boards from the map's voxels, tagged map; the under and touching demo cases load."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from b1k.connector.types import ObjRef, Provided
from b1k.connector.world import FurniturePiece, VoxelGrid

BENCH = Path(__file__).resolve().parents[2] / "tiptop/b1k/skills/bench"
SINK = ObjRef("sink.n.01_1", "sink", True)


def test_the_overhang_and_the_boards_are_the_maps_and_a_target_it_has_no_piece_for_has_none():
    """setting_mousetraps' wall-hung sink as a block of voxels 0.348 m over the floor: the floor under it with its
    underside as the ceiling, its top its one board; a movable (no map piece) has neither."""
    from omnigibson.tiptop.oracle.geometry import OracleGeometry

    sink = FurniturePiece(SINK, tuple(map(tuple, np.eye(4))),
                          {"body": VoxelGrid(0.02, (0.0, 0.0, 0.348), np.ones((27, 62, 37), dtype=bool))}, ())
    pieces = {SINK.id: sink}
    world_map = SimpleNamespace(piece=lambda o: Provided(pieces.get(o.id), "map", 7))
    geometry = OracleGeometry(SimpleNamespace(n_steps=7), world_map)
    under = geometry.overhang(SINK, 0.0)
    assert under.source == "map" and under.value.z == 0.0 and np.isclose(under.value.ceiling, 0.348)
    boards = geometry.boards(SINK)
    assert boards.source == "map" and [(round(b.z, 3), b.ceiling) for b in boards.value] == [(1.088, None)]
    toy = ObjRef("teddy_bear.n.01_1", "teddy_bear")
    assert geometry.overhang(toy, 0.0).value is None and geometry.boards(toy).value == ()


def test_the_under_and_touching_demo_cases_load_as_named():
    """The under set: the mousetrap under the wall-hung sink ojjqku (reset to ready: S1 refuses the crouch), the mouse
    under the desk uqcmzf (the right hand, as the humans hold it), the detergent under the multi-station sink (the
    'place under' demos, also reset: S1 refuses the human's idle right arm); the touching set: a shoe in the left hand
    at the hall tree, each with its ready reset."""
    from b1k.connector.skills import Rel
    from omnigibson.tiptop.host import skillbench

    under = skillbench.load_cases(BENCH / "demo_place_under.yaml")
    touching = skillbench.load_cases(BENCH / "demo_place_touching.yaml")
    assert [c["task"] for c in under] == ["setting_mousetraps", "getting_organized_for_work",
                                          "getting_organized_for_work", "sorting_household_items", "sorting_household_items"]
    assert [c["task"] for c in touching] == ["putting_shoes_on_rack"] * 4
    for cases, rel, target in ((under, Rel.UNDER, None), (touching, Rel.TOUCHING, "hallstand.n.01_1")):
        for case in cases:
            (r,) = case["call"].args.relations
            assert r.rel is rel and (target is None or r.target.id == target) and Path(case["demo"]["snapshot"]).exists()
            assert case["success"][0]["pred"] == rel.value
    assert [c["setup"].get("ready") for c in under + touching].count("left") == 4
    assert [next(iter(c["setup"]["held"])) for c in under] == ["left", "right", "right", "left", "left"]
