"""--collision mesh (SPEC §8 Track A2): today's physical meshes as the CollisionWorld, for the A/B against the map."""

import numpy as np

import omnigibson.utils.transform_utils as T
from b1k.connector.types import Provided
from omnigibson.tiptop.oracle.mapbuild import pseudo_map


class MeshRoom:
    """The simulator's physical meshes of the furniture the map holds, what --room sends a legacy round
    (R1ProSim.room_collision_scene), each posed in the map frame as MapCollisionWorld's entries are. Oracle. The
    same bodies as the map room (fixed-base furniture, not the ground: PseudoMap._ref) within ``radius`` of ``near``,
    so a mesh-vs-voxel A/B changes the form of the room and nothing else: --room also ships the task's movables and
    the lawn, which the native request has as perceived hulls or not at all."""

    def __init__(self, sim):
        self.sim = sim

    def room(self, near, radius) -> Provided:
        self.sim.nearby_obstacles(collision_map=True)  # registers them as sim.obstacles, which the scene reads
        scene, bodies, furniture = self.sim.room_collision_scene(), self.sim.obstacles, pseudo_map(self.sim)
        entries = [{**e, "name": n, "pose": T.pose2mat(bodies[n].get_position_orientation()).cpu().numpy()}
                   for n, e in scene.items()
                   if furniture._ref(bodies[n]) is not None and within(bodies[n], near, radius)]  # fmt: skip
        return Provided(entries, "oracle", self.sim.n_steps)


def within(body, near, radius: float) -> bool:
    """Whether any of the body's box is within ``radius`` of ``near``."""
    lo, hi = (np.asarray(v.cpu(), dtype=np.float64) for v in body.aabb)
    near = np.asarray(near, dtype=np.float64)
    return float(np.linalg.norm(np.clip(near, lo, hi) - near)) <= radius
