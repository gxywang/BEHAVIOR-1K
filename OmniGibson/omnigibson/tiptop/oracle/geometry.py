"""The oracle GeometryProvider (SPEC §4.1): thin wrappers over today's privileged queries. The map-voxel versions
(b1k/map/geometry.py) replace them; these stay as the logged cross-check."""

import numpy as np

from b1k.connector.types import AABB, Provided
from b1k.connector.world import Cavity, Region
from b1k.map.geometry import top_support as voxel_top_support


def rect(lo, hi) -> tuple:
    """The xy rectangle of a box, counter-clockwise."""
    (x0, y0), (x1, y1) = (float(v) for v in lo[:2]), (float(v) for v in hi[:2])
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


class OracleGeometry:
    """top_support: the map's open top (b1k.map.geometry over the piece's voxels: the modal top, never the hull's
    maximum, SPEC 6.2), tagged map; the target's AABB top, tagged oracle, only for a target the map has no piece for.
    cavity: the compartment the fillable meta-link accepts (inside_rect, the scorer's own Inside volume); extent: the
    AABB. World frame. overhang and boards come with their week-3 skills."""

    def __init__(self, sim, map=None):
        self.sim, self.map = sim, map

    def _aabb(self, o):
        return (v.cpu().numpy().astype(np.float64) for v in self.sim.scene_object(o.id).aabb)

    def top_support(self, target) -> Provided:
        piece = self.map.piece(target).value if self.map is not None else None
        region = voxel_top_support(piece) if piece is not None else None
        if region is not None:
            return Provided(region, "map", self.sim.n_steps)
        lo, hi = self._aabb(target)  # no voxels for it: the AABB (bed_1's headboard top, 6.5 cm over the mattress)
        return Provided(Region(rect(lo, hi), float(hi[2])), "oracle", self.sim.n_steps)

    def cavity(self, target, item) -> Provided:
        got = self.sim.inside_rect(item.id, target.id)
        if got is None:
            return Provided(None, "oracle", self.sim.n_steps)
        centre, half, floor, ceiling = got
        c, h = np.asarray(centre, float), np.asarray(half, float)
        return Provided(Cavity(Region(rect(c - h, c + h), floor, ceiling), ceiling), "oracle", self.sim.n_steps)

    def extent(self, o) -> Provided:
        lo, hi = self._aabb(o)
        return Provided(AABB(tuple(map(float, lo)), tuple(map(float, hi))), "oracle", self.sim.n_steps)
