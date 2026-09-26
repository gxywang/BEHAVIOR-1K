"""The bench's Navigator (SPEC D5, D20): candidate stances for what a skill must reach, and a teleport to one.

propose: candidates on rings around the reach points' centroid (STANDOFFS out, ANGLES around, facing it with a yaw
offset), kept where the footprint is free (R1ProSim._footprint_free over the scene's boxes: oracle, so the Stance says
so), the preferred standoff first and, within a ring, the nearest to where the base stands. Which of them the arm
reaches from is not decided here: the planner's check_stances (the IK service) ranks them, and reach() drives to the
best. go_to: R1ProSim.place_robot, the teleport with its fold and unfold ramps, which steps the sim itself
(requires_sim_clock: 0 Runtime steps, the Runtime re-seeds its latch after) and is charged MOVE_TO_STEPS shadow steps
(D20); a landing the check refuses is a NavResult that is not ok. ``steps`` counts the sim steps the teleports took,
for the bench's U0 check.
"""

import math

import numpy as np

from b1k.connector.skills import NavResult, Stance
from b1k.connector.types import Belief, Pose2

MOVE_TO_STEPS = 559  # the human move-to mean (SPEC D20)
STANDOFFS = (0.6, 0.75, 0.9, 1.05, 0.45)  # m from the reach points' centroid, the preferred first; an object 0.3 m into
#                                            a counter needs the base a further 0.5 m off its edge (the arm's resting
#                                            posture is tested too), so the rings reach past the arm's 0.9 m
ANGLES = np.arange(0.0, 2 * np.pi, np.pi / 6)  # around the centroid
YAW_OFFSETS = (0.0, np.pi / 6, -np.pi / 6)  # facing the centroid, and the object to the left or right of straight ahead


class TeleportNavigator:
    requires_sim_clock = True

    def __init__(self, sim):
        self.sim, self.steps = sim, 0

    def base_pose(self) -> Belief:
        import omnigibson.utils.transform_utils as T

        pos, quat = self.sim.robot.get_position_orientation()
        return Belief(Pose2(float(pos[0]), float(pos[1]), float(T.quat2euler(quat)[2]), float(pos[2])), "oracle", 0)

    def propose(self, req, k: int = 8) -> list:
        pts = np.asarray(req.reach_points, dtype=np.float64)[:, :2]
        mid = pts.mean(axis=0)
        here = self.base_pose().value
        aabbs = self.sim.scene_aabbs()
        reaching = [self.sim.scene_object(o.id) for o in req.targets]
        out = []
        for r in STANDOFFS:
            ring = []
            for a in ANGLES:
                x, y = mid + r * np.array([math.cos(a), math.sin(a)])
                for off in YAW_OFFSETS:
                    yaw = math.atan2(mid[1] - y, mid[0] - x) + off
                    free = self.sim._footprint_free(x, y, [], aabbs=aabbs, yaw=yaw, reaching=reaching)
                    if free[0]:
                        ring.append((math.hypot(x - here.x, y - here.y), float(x), float(y), float(yaw)))
                        break  # one heading per spot: the next offsets are the same stance turned
            for d, x, y, yaw in sorted(ring):
                out.append(Stance(f"ring:{r:.2f}:{len(out)}", Pose2(x, y, yaw, here.z), 1.0,
                                  f"{r:.2f} m from the target, {d:.2f} m from here, footprint free", "oracle"))
                if len(out) == k:
                    return out
        return out

    def go_to(self, stance: Stance, obs):
        from omnigibson.tiptop.r1pro import BasePlacementCollision

        n0 = self.sim.n_steps
        try:
            self.sim.place_robot(stance.pose.x, stance.pose.y, stance.pose.yaw, note=f"go_to {stance.key}")
        except BasePlacementCollision as e:
            self.steps += self.sim.n_steps - n0
            return NavResult(False, stance, 0, 0, str(e)), obs
        self.steps += self.sim.n_steps - n0
        return NavResult(True, stance, 0, MOVE_TO_STEPS), obs
        yield  # a generator that yields nothing: the teleport stepped the sim itself (requires_sim_clock)

    def apply(self, update) -> None:
        pass  # ponytail: occupancy updates (an opened door's footprint) shape no candidate yet; the footprint test is live
