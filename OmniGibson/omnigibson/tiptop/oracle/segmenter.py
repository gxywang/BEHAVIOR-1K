"""OracleSegmenter (SPEC §4.1): instance masks from the simulator's geometry, where a detector would give them."""

from b1k.bridge.protocol import capture_views
from b1k.connector.types import Provided


class OracleSegmenter:
    """{view: {label: (H, W) bool}} for every view of one capture, from the tracked objects' meshes at their true
    poses (scene.oracle_masks, the oracle knowledge source's masks). Oracle-tagged."""

    def __init__(self, sim):
        self.sim = sim

    def masks(self, labels: list, request: dict, extras: dict) -> Provided:
        meshes = self.sim.object_meshes(labels)  # one set of meshes for every view, as OracleKnowledge.describe
        out = {
            name: dict(zip(labels, self.sim.oracle_masks(view, view_extras, labels, meshes=meshes)))
            for name, view, view_extras in capture_views(request, extras)
        }
        return Provided(out, "oracle", self.sim.n_steps)
