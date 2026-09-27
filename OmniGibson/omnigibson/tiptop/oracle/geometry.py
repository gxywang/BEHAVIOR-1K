"""The oracle GeometryProvider (SPEC §4.1): thin wrappers over today's privileged queries. The map-voxel versions
(b1k/map/geometry.py) replace them; these stay as the logged cross-check."""

import logging

import numpy as np

from b1k.connector.types import AABB, Provided
from b1k.connector.world import Cavity, Region
from b1k.map.geometry import aabb as voxel_aabb
from b1k.map.geometry import accepting, cavities
from b1k.map.geometry import top_support as voxel_top_support

log = logging.getLogger(__name__)


def rect(lo, hi) -> tuple:
    """The xy rectangle of a box, counter-clockwise."""
    (x0, y0), (x1, y1) = (float(v) for v in lo[:2]), (float(v) for v in hi[:2])
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


class OracleGeometry:
    """top_support: the map's open top (b1k.map.geometry over the piece's voxels: the modal top, never the hull's
    maximum, SPEC 6.2), tagged map; the target's AABB top, tagged oracle, only for a target the map has no piece for.
    cavity: the compartment the fillable meta-link accepts (inside_rect, the scorer's own Inside volume), tagged
    oracle, with the map's voxel compartments logged beside it (SPEC §8 week 3: the voxel cavity replaces it once it
    accepts what inside_rect accepts); extent: a fixed piece's map voxel box (next_to's static reference), else the
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
        found = None
        if got is not None:
            centre, half, floor, ceiling = got
            c, h = np.asarray(centre, float), np.asarray(half, float)
            found = Cavity(Region(rect(c - h, c + h), floor, ceiling), ceiling)
        piece = self.map.piece(target).value if self.map is not None else None
        if piece is not None:  # the voxel cavity's gate: does a compartment accept what inside_rect accepts
            voxels = cavities(piece)
            match = accepting(voxels, found.floor) if found is not None else None
            said = lambda c: None if c is None else (c.floor.polygon, round(c.floor.z, 3), round(c.top_z, 3))  # noqa
            log.info(f"cavity {target.id}: inside_rect {said(found)}; voxel compartments {len(voxels)}; "
                     f"accepted by {said(match)}")
        return Provided(found, "oracle", self.sim.n_steps)

    def extent(self, o) -> Provided:
        lo, hi = self._aabb(o)
        piece = self.map.piece(o).value if self.map is not None else None
        box = voxel_aabb(piece) if piece is not None else None
        if box is not None:  # a static next_to reference (SPEC 6.2): its posed map voxels, the sim's AABB logged beside
            log.info(f"extent {o.id}: map voxels {box.lo} {box.hi}; sim AABB {lo.round(3).tolist()} "
                     f"{hi.round(3).tolist()}")
            return Provided(box, "map", self.sim.n_steps)
        return Provided(AABB(tuple(map(float, lo)), tuple(map(float, hi))), "oracle", self.sim.n_steps)
