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
