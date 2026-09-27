"""The bench's Navigator (SPEC D5, D20): candidate stances for what a skill must reach, and a teleport to one.

propose: first the legacy stance search's best (R1ProSim.best_base_pose, as place_robot_for asks it: the head camera
frames every target and its support, and the left arm is within reach; SPEC §7's stand_for), then candidates on rings
around the reach points' centroid (STANDOFFS out, ANGLES around, facing it with a yaw offset), kept where the footprint
is free (R1ProSim._footprint_free over the scene's boxes: oracle, so the Stance says so), the preferred standoff first
and, within a ring, the nearest to where the base stands. The rings frame nothing: the store_honey jar's first free
ring stance stood 1.05 m off, facing an oven with the jar at the frame's edge, and the pick's perception found no
support plane in every trial. Which of them the arm reaches from is not decided here: the planner's check_stances
(the IK service) ranks them, and reach() drives to the best. go_to: R1ProSim.place_robot, the teleport with its fold
and unfold ramps, which steps the sim itself (requires_sim_clock: 0 Runtime steps, the Runtime re-seeds its latch
after) and is charged MOVE_TO_STEPS shadow steps (D20); a landing the check refuses is a NavResult that is not ok.
``steps`` counts the sim steps the teleports took, for the bench's U0 check.
"""

import logging
import math
from collections import Counter

import numpy as np

from b1k.connector.skills import NavResult, Stance
from b1k.connector.types import Belief, Pose2

log = logging.getLogger(__name__)
MOVE_TO_STEPS = 559  # the human move-to mean (SPEC D20)
STANDOFFS = (0.6, 0.75, 0.9, 1.05, 0.45)  # m from the reach points' centroid, the preferred first; an object 0.3 m into
#                                            a counter needs the base 0.75 m off, and the perception that follows wants
#                                            it close (from 1.05 m the jar's support plane was not found)
ANGLES = np.arange(0.0, 2 * np.pi, np.pi / 6)  # around the centroid
YAW_OFFSETS = (0.0, np.pi / 6, -np.pi / 6)  # facing the centroid, and the object to the left or right of straight ahead


class TeleportNavigator:
    requires_sim_clock = True

    def __init__(self, sim):
        self.sim, self.steps, self.looking = sim, 0, ()  # looking: the last proposal's targets

    def base_pose(self) -> Belief:
        import omnigibson.utils.transform_utils as T

        pos, quat = self.sim.robot.get_position_orientation()
        return Belief(Pose2(float(pos[0]), float(pos[1]), float(T.quat2euler(quat)[2]), float(pos[2])), "oracle", 0)

    def propose(self, req, k: int = 8) -> list:
        self.looking = tuple(o.id for o in req.targets)
        pts = np.asarray(req.reach_points, dtype=np.float64)[:, :2]
        mid = pts.mean(axis=0)
        here = self.base_pose().value
        aabbs = self.sim.scene_aabbs()
        reaching = [self.sim.scene_object(o.id) for o in req.targets]
        out = [Stance("search:0", Pose2(x, y, yaw, here.z), 1.0, "the legacy stance search's best: the head frames "
                      "the targets, the left arm reaches them", "oracle") for x, y, yaw in self.searched(req, reaching)]
        refused = Counter()
        for r in STANDOFFS:
            if len(out) >= k:
                break
            ring = []
            for a in ANGLES:
                x, y = mid + r * np.array([math.cos(a), math.sin(a)])
                for off in YAW_OFFSETS:
                    yaw = math.atan2(mid[1] - y, mid[0] - x) + off
                    # arms=False: the teleport folds the arms and place_robot tests the landing and the unfold itself;
                    # tested at their working posture, every ring within 0.9 m of a jar on a counter was refused
                    free = self.sim._footprint_free(x, y, [], aabbs=aabbs, yaw=yaw, arms=False, reaching=reaching)
                    if free[0]:
                        ring.append((math.hypot(x - here.x, y - here.y), float(x), float(y), float(yaw)))
                        break  # one heading per spot: the next offsets are the same stance turned
                    refused[f"{r:.2f} m: {free[1]}"] += 1
            for d, x, y, yaw in sorted(ring):
                out.append(Stance(f"ring:{r:.2f}:{len(out)}", Pose2(x, y, yaw, here.z), 1.0,
                                  f"{r:.2f} m from the target, {d:.2f} m from here, footprint free", "oracle"))
        log.info(f"propose: {[s.key for s in out[:k]]}; ring headings refused: {dict(refused)}")
        return out[:k]

    def searched(self, req, objs) -> list:
        """[(x, y, yaw)] of the legacy stance search's best for ``req``'s targets, or [] where it finds none."""
        best, _ = self.sim.best_base_pose(
            [o.aabb_center.cpu().numpy()[:2] for o in objs], reaching=objs,
            half_widths=[self.sim.xy_radius(o.id) for o in req.targets], support_z=[float(o.aabb[0][2]) for o in objs],
            boxes=[(o.aabb[0].cpu().numpy(), o.aabb[1].cpu().numpy()) for o in objs])
        return [] if best is None else [tuple(float(v) for v in best[1:4])]

    def go_to(self, stance: Stance, obs):
        n0 = self.sim.n_steps
        try:  # the arm unfolds as far as it can: UNFOLD_MIN refused every stance the IK service reached from at the
            #   store_honey jar (25%, the countertop) and the wall switch (0%, a cabinet), where the press succeeded
            #   with no minimum; place_robot_for itself falls back to the furthest unfold when no stance makes it
            self.sim.place_robot(stance.pose.x, stance.pose.y, stance.pose.yaw, note=f"go_to {stance.key}")
        except Exception as e:  # noqa: BLE001 - BasePlacementCollision, or the landing's or the unfold's own check
            #                     (RuntimeError 'cannot validate base destination'): a teleport that did not land
            return NavResult(False, stance, 0, 0, str(e)), obs
        finally:  # the fold's ramps stepped the sim whether or not it landed: the U0 check counts them
            self.steps += self.sim.n_steps - n0
        if self.looking:  # place_robot cleared what the captures look at; a capture that does not aim (the legacy
            self.sim.look_at(*self.looking)  # round's own) looked at DEFAULT_LOOK_TARGET after a reach
        return NavResult(True, stance, 0, MOVE_TO_STEPS), obs
        yield  # a generator that yields nothing: the teleport stepped the sim itself (requires_sim_clock)

    def apply(self, update) -> None:
        pass  # ponytail: occupancy updates (an opened door's footprint) shape no candidate yet; the footprint test is live
