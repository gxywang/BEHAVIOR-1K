"""The oracle ArticulationSource (SPEC §4.1): an object's joints as JointSpecs, read off the live scene."""

import numpy as np

from b1k.connector.types import Provided
from b1k.connector.world import JointSpec
from omnigibson.tiptop.articulation import openable_joints


def gravity(kind: str, axis, origin, centroid, opens_up: bool) -> str:
    """Which way gravity turns a joint near shut: the torque of the moving link's weight about the hinge, signed by
    the joint's opening direction. A slide or a vertical hinge feels none. Not "any non-vertical hinge is a lid": a
    link hanging below its hinge falls shut."""
    if kind != "revolute":
        return "neutral"
    torque = float(np.dot(np.asarray(axis, float), np.cross(np.asarray(centroid, float) - origin, (0.0, 0.0, -1.0))))
    if abs(torque) < 1e-3:
        return "neutral"
    return "falls_open" if (torque > 0) == opens_up else "falls_shut"


class OracleArticulation:
    """Frames and limits from openable_joints (world frame, the map's frame on the bench), which end is shut, and
    whether the Open state's metadata counts the joint. Handle features stay () here: the extractor that
    generalizes handle_on is Track C's."""

    def __init__(self, sim):
        self.sim = sim

    def joints(self, o) -> Provided:
        from omnigibson.object_states.open_state import _get_relevant_joints

        obj = self.sim.scene_object(o.id)
        relevant = {id(j) for j in _get_relevant_joints(obj)[1]}
        specs = []
        for j in openable_joints(obj):
            lo, hi = (v.cpu().numpy() for v in obj.links[j["link"]].aabb)
            closed_end = "lower" if j["closed"] == j["lower"] else "upper"
            specs.append(JointSpec(
                j["name"], j["kind"], tuple(map(float, j["axis"])), tuple(map(float, j["origin"])), j["lower"],
                j["upper"], closed_end, id(obj.joints[j["name"]]) in relevant,
                gravity(j["kind"], j["axis"], j["origin"], (lo + hi) / 2.0, closed_end == "lower"), j["link"], ()))
        return Provided(tuple(specs), "oracle", self.sim.n_steps)
