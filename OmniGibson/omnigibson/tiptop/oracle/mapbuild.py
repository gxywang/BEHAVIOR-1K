"""The pseudo map (SPEC D27, §4.1): the competition map's form, pose + one voxel grid per link + joint frames, built
from the sim's per-link collision meshes. Every moving link is captured at its joint's CLOSED value, never the live
one, so the map carries no joint state. Privileged code (it reads sim meshes); what it serves is tagged "map"."""

import numpy as np
from scipy.spatial import ConvexHull

from b1k.connector.types import ObjRef, Provided
from b1k.connector.world import FurniturePiece, JointFrame, VoxelGrid, joint_motion
from omnigibson.tiptop.articulation import openable_joints

RESOLUTION = 0.02  # m, the voxel edge: the working value until the competition map's own is known (SPEC Q11)
NOT_FURNITURE = ("floors", "lawn", "driveway", "ceilings")  # the scene's categories of the ground and what is overhead
EPS = 1e-9  # m and voxels: float dust, never geometry
# voxel x hull-face tests evaluated at once (x-slabs of a hull's box): each float64 temporary stays near 128 MB. The
# whole box at once cost voxels x faces x 8 bytes several times over: a house's large hulls at 2 cm took the sim to
# 200 GB in the first native place's room() (week-4 fix pass, the S1 re-run on bringing_in_wood)
CHUNK = 1 << 24


def voxelize(hulls: list, resolution: float) -> tuple:
    """(origin, occupied) of the union of convex pieces (one vertex array per collision hull, as the physics sees
    them: OmniGibson gives every collision mesh a convexHull approximation), on the grid through the origin. A voxel
    is occupied when it overlaps a hull: no hull face separates it, the face plane pushed out by the voxel's own half
    extent along its normal. Conservative, so a panel thinner than a voxel still occupies one; a voxel that only
    touches a hull (a shared face) is free, so grid-aligned geometry is not dilated."""
    hulls = [np.asarray(v, float) for v in hulls if len(v) >= 4]
    origin = np.floor(np.min([v.min(0) for v in hulls], 0) / resolution + EPS) * resolution
    cell = lambda x, rnd, dust: rnd((x - origin) / resolution + dust).astype(int)  # dust: a hair past a grid line
    spans = [(np.maximum(cell(v.min(0), np.floor, EPS), 0), cell(v.max(0), np.ceil, -EPS)) for v in hulls]
    spans = [(i0, np.maximum(i1, i0 + 1)) for i0, i1 in spans]
    occupied = np.zeros(np.max([i1 for _, i1 in spans], 0), dtype=bool)
    for v, (i0, i1) in zip(hulls, spans):
        eq = ConvexHull(v, qhull_options="QJ").equations  # n . x + d <= 0 inside; QJ: a flat piece still has a hull
        lim = 0.5 * resolution * np.abs(eq[:, :3]).sum(1) - EPS
        ys, zs = np.arange(i0[1], i1[1]), np.arange(i0[2], i1[2])
        step = max(1, CHUNK // max(1, len(ys) * len(zs) * len(eq)))  # x layers per slab (one at least)
        for x0 in range(int(i0[0]), int(i1[0]), step):
            x1 = min(x0 + step, int(i1[0]))
            idx = np.stack(np.meshgrid(np.arange(x0, x1), ys, zs, indexing="ij"), -1)
            centres = origin + (idx + 0.5) * resolution
            inside = (centres @ eq[:, :3].T + eq[:, 3] < lim).all(-1)
            occupied[x0:x1, i0[1]:i1[1], i0[2]:i1[2]] |= inside
    return tuple(map(float, origin)), occupied


def to_frame(T: np.ndarray, points) -> np.ndarray:
    p = np.asarray(points, float).reshape(-1, 3)
    return p @ T[:3, :3].T + T[:3, 3]


def build_piece(obj, ref: ObjRef, resolution: float = RESOLUTION) -> FurniturePiece:
    """One piece: its pose, a "body" grid of every link that does not move and one grid per moving link, each in
    the body frame with the moving link brought back to its closed value along its JointFrame."""
    import omnigibson.utils.transform_utils as T
    from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh

    pose = T.pose2mat(obj.get_position_orientation()).cpu().numpy().astype(np.float64)
    body = np.linalg.inv(pose)
    joints, back = [], {}
    for j in openable_joints(obj):  # ponytail: one level, moving links hang from the body (a handle link welded to
        #                             a drawer would stay in "body"); the map's JointFrame has the same limit
        frame = JointFrame(j["name"], j["kind"], j["link"], tuple(to_frame(body, j["origin"])[0]),
                           tuple(body[:3, :3] @ np.asarray(j["axis"], float)), j["lower"], j["upper"],
                           "lower" if j["closed"] == j["lower"] else "upper")  # fmt: skip
        joints.append(frame)
        back[j["link"]] = np.linalg.inv(joint_motion(frame, j["position"]))  # live value -> closed
    hulls = {}
    for key, link in obj.links.items():
        if getattr(link, "visual_only", False):
            continue
        to_body = back.get(key, np.eye(4)) @ body
        for geom in link.collision_meshes.values():
            mesh = mesh_prim_to_trimesh_mesh(geom.prim, include_normals=False, include_texcoord=False, world_frame=True)
            hulls.setdefault(key if key in back else "body", []).append(to_frame(to_body, mesh.vertices))
    links = {name: VoxelGrid(resolution, *voxelize(vs, resolution)) for name, vs in hulls.items()}
    return FurniturePiece(ref, tuple(map(tuple, pose)), links, tuple(joints))


class PseudoMap:
    """MapProvider over the live scene: one piece per fixed furniture object, built on first use and kept for the
    scene (pseudo_map() gives one per sim). Named by BDDL name when the task scope names it, else by scene name."""

    def __init__(self, sim, resolution: float = RESOLUTION):
        self.sim, self.resolution, self.pieces = sim, resolution, {}

    def _ref(self, obj) -> ObjRef | None:
        if not getattr(obj, "fixed_base", False) or obj.category in NOT_FURNITURE or obj is self.sim.robot:
            return None
        scope = {id(o): name for name, o in self.sim.task_scope().items()}
        return ObjRef(scope.get(id(obj), obj.name), obj.category, True)

    def furniture(self) -> tuple:
        return tuple(ref for obj in self.sim.env.scene.objects if (ref := self._ref(obj)) is not None)

    def piece(self, o) -> Provided:
        obj = self.sim.scene_object(o.id)
        ref = self._ref(obj)
        if ref is not None and obj.name not in self.pieces:
            self.pieces[obj.name] = build_piece(obj, ref, self.resolution)
        return Provided(None if ref is None else self.pieces[obj.name], "map", self.sim.n_steps)


_MAPS = {}  # id(sim) -> its scene's PseudoMap: one sim per process on the bench


def pseudo_map(sim) -> PseudoMap:
    if id(sim) not in _MAPS:
        _MAPS[id(sim)] = PseudoMap(sim)
    return _MAPS[id(sim)]
