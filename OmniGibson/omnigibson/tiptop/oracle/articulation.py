"""The oracle ArticulationSource (SPEC §4.1): an object's joints as JointSpecs, read off the live scene."""

import numpy as np

from b1k.bridge.articulation import handle_on, leading_direction
from b1k.connector.types import Provided
from b1k.connector.world import HandleFeature, JointSpec, link_pose
from omnigibson.tiptop.articulation import openable_joints
from omnigibson.tiptop.oracle.mapbuild import pseudo_map


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


def handle_feature(vertices, j: dict, map_from_link) -> HandleFeature:
    """What handle_on reads off the moving link's mesh (world frame, the joint at j["position"]), in the link's map
    frame (``map_from_link``: link_pose at that value). The frame's z is the face's outward normal, y the jaw
    direction across a bar (up the face on a panel), x = y cross z; its origin is the middle of a bar's front, or
    the panel's centre. cross_section is (the bar's width across the jaw, how far its front stands off the panel)."""
    lead = leading_direction(j["kind"], j["axis"], j["origin"], vertices, 1.0 if j["closed"] == j["lower"] else -1.0)
    h = handle_on(vertices, lead)
    bar = h["kind"] == "bar"
    y = h["jaw"] if bar else h["up"]
    point = np.asarray(h["point"], dtype=np.float64)
    if bar:  # the bar's middle along it, on its front
        point = point + h["along"] * (sum(h["ends"]) / 2.0 - point @ h["along"]) + lead * (h["front"] - point @ lead)
    world = np.eye(4)
    world[:3, :3] = np.stack([np.cross(y, lead), y, lead], axis=1)
    world[:3, 3] = point
    frame = np.linalg.inv(np.asarray(map_from_link, dtype=np.float64)) @ world
    width = float(min(h["extent"][:2])) if bar else 0.0
    return HandleFeature("bar" if bar else "face", tuple(map(tuple, frame.tolist())),
                         float(h["span"] if bar else h["face_extent"][0]), (width, float(h["proud"]) if bar else 0.0),
                         float(h["proud"] - h["bar_depth"]) if bar else 0.0)


class OracleArticulation:
    """Frames and limits from openable_joints (world frame, the map's frame on the bench), which end is shut, and
    whether the Open state's metadata counts the joint. The handle feature is handle_on over the moving link's mesh
    (the oracle half of a feature perception gives later), in the link's frame of the pseudo map."""

    def __init__(self, sim):
        self.sim = sim

    def joints(self, o) -> Provided:
        from omnigibson.object_states.open_state import _get_relevant_joints

        obj = self.sim.scene_object(o.id)
        relevant = {id(j) for j in _get_relevant_joints(obj)[1]}
        piece = pseudo_map(self.sim).piece(o).value
        specs = []
        for j in openable_joints(obj):
            lo, hi = (v.cpu().numpy() for v in obj.links[j["link"]].aabb)
            closed_end = "lower" if j["closed"] == j["lower"] else "upper"
            mesh = self.sim.link_trimesh_world(obj.links[j["link"]])
            features = ()
            if piece is not None and mesh is not None and len(mesh.vertices):
                at = link_pose(piece, j["link"], {j["name"]: j["position"]})
                features = (handle_feature(np.asarray(mesh.vertices, dtype=np.float64), j, at),)
            specs.append(JointSpec(
                j["name"], j["kind"], tuple(map(float, j["axis"])), tuple(map(float, j["origin"])), j["lower"],
                j["upper"], closed_end, id(obj.joints[j["name"]]) in relevant,
                gravity(j["kind"], j["axis"], j["origin"], (lo + hi) / 2.0, closed_end == "lower"), j["link"],
                features))
        return Provided(tuple(specs), "oracle", self.sim.n_steps)
