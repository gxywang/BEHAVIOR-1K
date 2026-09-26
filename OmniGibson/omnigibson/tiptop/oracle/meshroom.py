"""--collision mesh (SPEC §8 Track A2): today's physical meshes as the CollisionWorld, for the A/B against the map."""

import omnigibson.utils.transform_utils as T
from b1k.connector.types import Provided


class MeshRoom:
    """The simulator's physical meshes near the robot, what --room sends a legacy round (R1ProSim.room_collision_scene),
    each posed in the map frame as MapCollisionWorld's entries are. Oracle. The planner server takes these until
    tiptop/skills/voxels.py merges the map's voxels.
    ponytail: today's obstacle reach around the robot (nearby_obstacles), not the caller's near/radius."""

    def __init__(self, sim):
        self.sim = sim

    def room(self, near, radius) -> Provided:
        self.sim.nearby_obstacles(collision_map=True)  # registers them as sim.obstacles, which the scene reads
        scene, bodies = self.sim.room_collision_scene(), self.sim.obstacles
        entries = [{**e, "name": n, "pose": T.pose2mat(bodies[n].get_position_orientation()).cpu().numpy()}
                   for n, e in scene.items()]  # fmt: skip
        return Provided(entries, "oracle", self.sim.n_steps)
