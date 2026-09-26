"""The oracle handle feature (SPEC §4.1, §6.6, §8 Track C): handle_on over the moving link's mesh, as a HandleFeature
in the link's frame of the pseudo map, which the articulate RequestBuilder poses at the estimated joint value."""

import numpy as np


def box(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def test_a_bar_on_a_drawer_front_is_a_frame_on_its_front_middle_in_the_links_map_frame():
    from omnigibson.tiptop.oracle.articulation import handle_feature

    # a drawer front (x 0..0.02) with a horizontal bar 2.7 cm proud and 1.2 cm tall on two posts; it slides out +x
    verts = np.concatenate([box((0.0, -0.2, 0.0), (0.02, 0.2, 0.3)), box((0.035, -0.1, 0.24), (0.047, 0.1, 0.252)),
                            box((0.02, -0.095, 0.24), (0.035, -0.085, 0.252)),
                            box((0.02, 0.085, 0.24), (0.035, 0.095, 0.252))])
    joint = {"kind": "prismatic", "axis": (1.0, 0.0, 0.0), "origin": (0.0, 0.0, 0.0), "lower": 0.0, "upper": 0.4,
             "closed": 0.0, "position": 0.1}
    map_from_link = np.eye(4)
    map_from_link[:3, 3] = (0.1, 0.0, 0.0)  # the link's map frame, carried 0.1 m out by the joint
    f = handle_feature(verts, joint, map_from_link)
    frame = np.asarray(f.frame)
    assert f.kind == "bar" and np.isclose(f.length, 0.2)
    assert np.allclose(f.cross_section, (0.012, 0.027)), "(width across the jaw, the front's standoff off the panel)"
    assert np.allclose(frame[:3, 2], [1.0, 0.0, 0.0]), "z: out of the face, the way the drawer opens"
    assert np.allclose(np.abs(frame[:3, 1]), [0.0, 0.0, 1.0]), "y: the jaw closes across the bar's height"
    assert np.isclose(np.linalg.det(frame[:3, :3]), 1.0)
    assert np.allclose(frame[:3, 3], [0.047 - 0.1, 0.0, 0.246]), "the bar's front middle, in the link's frame"


def test_each_joint_gets_its_handle_posed_where_the_drawer_stands_now(monkeypatch):
    """OracleArticulation.joints: the feature is handle_on over the link's mesh at the joint's live position, in the
    link's pseudo-map frame, so link_pose at that position puts it back where the mesh has it."""
    from types import SimpleNamespace

    import omnigibson.object_states.open_state as open_state
    import torch as th
    from b1k.connector.types import ObjRef, Provided
    from b1k.connector.world import FurniturePiece, JointFrame, VoxelGrid, link_pose

    from omnigibson.tiptop.oracle import articulation

    out = 0.1  # the drawer stands 0.1 m out
    verts = np.concatenate([box((0.0, -0.2, 0.0), (0.02, 0.2, 0.3)), box((0.035, -0.1, 0.24), (0.047, 0.1, 0.252)),
                            box((0.02, -0.095, 0.24), (0.035, -0.085, 0.252)),
                            box((0.02, 0.085, 0.24), (0.035, 0.095, 0.252))]) + (1.0 + out, 2.0, 0.3)
    j = {"name": "j_link_1", "kind": "prismatic", "axis": (1.0, 0.0, 0.0), "origin": (1.0, 2.0, 0.3), "lower": 0.0,
         "upper": 0.4, "closed": 0.0, "position": out, "link": "link_1"}
    cab = ObjRef("cabinet.n.01_1", "cabinet", True)
    grid = VoxelGrid(0.02, (0.0, 0.0, 0.0), np.ones((1, 1, 1), dtype=bool))
    piece = FurniturePiece(cab, tuple(map(tuple, np.eye(4).tolist())), {"body": grid, "link_1": grid},
                           (JointFrame("j_link_1", "prismatic", "link_1", (1.0, 2.0, 0.3), (1.0, 0.0, 0.0), 0.0, 0.4,
                                       "lower"),))
    joint = object()
    link = SimpleNamespace(aabb=(th.tensor([1.0, 1.8, 0.3]), th.tensor([1.2, 2.2, 0.6])))
    obj = SimpleNamespace(links={"link_1": link}, joints={"j_link_1": joint})
    sim = SimpleNamespace(scene_object=lambda name: obj, n_steps=7,
                          link_trimesh_world=lambda link: SimpleNamespace(vertices=verts))
    monkeypatch.setattr(open_state, "_get_relevant_joints", lambda o: (None, [joint]))
    monkeypatch.setattr(articulation, "openable_joints", lambda o: [j])
    monkeypatch.setattr(articulation, "pseudo_map",
                        lambda s: SimpleNamespace(piece=lambda o: Provided(piece, "map", 0)))
    (spec,) = articulation.OracleArticulation(sim).joints(cab).value
    (feature,) = spec.features
    placed = np.asarray(link_pose(piece, "link_1", {"j_link_1": out})) @ np.asarray(feature.frame)
    assert np.allclose(placed, np.asarray(articulation.handle_feature(verts, j, np.eye(4)).frame)), \
        "the handle, posed by the map at the live value, is where the link's mesh has it"
    assert spec.relevant and spec.moving_link == "link_1" and spec.closed_end == "lower"
