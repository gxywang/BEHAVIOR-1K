"""The video's overview camera (harness-only, never on the Runtime's path: named in the boundary test's HARNESS_SETUP,
so it may read the simulator). The 'shoulder' eye place_robot puts the camera at sat behind a pillar at the store_honey
stance and looked at a window at the drawer stances (VIDEO_FINDINGS 9, 13): aim_overview raycasts a few candidate eyes
around the robot at its chest and at the call's target, and keeps the first with a clear line of sight."""

import logging
import math
from typing import Optional

import numpy as np
import torch as th

from omnigibson.tiptop.r1pro import OVERVIEW_OFFSETS
from omnigibson.utils.sampling_utils import raytest

log = logging.getLogger("omnigibson.tiptop")

# Candidate eyes in the base frame (dx, dy, z), tried after the --overview view's own: over the right shoulder,
# ahead-left and ahead-right looking back, behind and higher, overhead
OVERVIEW_EYES = ((-1.5, -1.1, 1.7), (1.15, 0.75, 1.35), (1.15, -0.75, 1.35), (-2.4, 0.0, 2.2), (0.0, 0.8, 2.6))
SIGHT_MARGIN = 0.3  # m: a raycast hit this close to the point looked at is that point's own body
CHEST_Z = 1.1  # m: where the eye must see the robot (its chest, over the base)


def clear_sight(eye, point, allow=()) -> bool:
    """Nothing between ``eye`` and ``point`` (world) but a body ``allow`` names (prim path prefixes) or the point's own
    surroundings (a hit within SIGHT_MARGIN of it: the target's own body, the fingers around it)."""
    hit = raytest(th.tensor(eye, dtype=th.float32), th.tensor(point, dtype=th.float32))
    return (not hit["hit"] or hit["distance"] >= float(np.linalg.norm(np.subtract(point, eye))) - SIGHT_MARGIN
            or any(str(hit["rigidBody"]).startswith(p) for p in allow))


def aim_overview(sim, x: float, y: float, yaw: float, target: Optional[str] = None) -> None:
    """The overview camera for a base at (x, y, yaw): the first eye, the --overview view's own (OVERVIEW_OFFSETS, as
    place_robot aims it) then OVERVIEW_EYES, that sees the robot's chest and ``target`` (the call's target object, by
    name; else the view's own workspace point) with nothing in the way. Every eye blocked: the view's own, as before."""
    dx, dy, z, tx, tz = OVERVIEW_OFFSETS[sim.overview_view]
    c, s = math.cos(yaw), math.sin(yaw)
    world = lambda ox, oy, oz: (x + ox * c - oy * s, y + ox * s + oy * c, oz)
    chest, robot = world(0.0, 0.0, CHEST_Z), (sim.robot.prim_path,)
    if target is not None:
        obj = sim.scene_object(target)
        look, own = tuple(float(v) for v in obj.aabb_center), (obj.prim_path,)
    else:
        look, own = world(tx, 0.0, tz), ()
    eyes = [world(dx, dy, z)] + [world(*e) for e in OVERVIEW_EYES]
    eye = next((e for e in eyes if clear_sight(e, chest, robot) and clear_sight(e, look, own)), eyes[0])
    log.info(f"overview camera: eye {eyes.index(eye)} at {np.round(eye, 2).tolist()} looking at {np.round(look, 2).tolist()}")
    sim.aim_overview(eye, look)
