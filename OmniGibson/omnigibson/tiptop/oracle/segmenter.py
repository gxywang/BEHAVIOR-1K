"""OracleSegmenter (SPEC §4.1): instance masks from the simulator's geometry, where a detector would give them."""

import numpy as np

from b1k.bridge.protocol import capture_views
from b1k.connector.types import Provided


class OracleSegmenter:
    """{view: {label: (H, W) bool}} for every view of one capture, from the tracked objects' meshes at their true
    poses (scene.oracle_masks, the oracle knowledge source's masks). Oracle-tagged."""

    def __init__(self, sim):
        self.sim = sim

    def masks(self, labels: list, request: dict, extras: dict) -> Provided:
        # a place's target may be furniture no tracked object stands for (the floor, an untracked table): an empty
        # mask; its region is the map's support (SPEC 6.2), never a segmented object
        tracked = [label for label in labels if label in self.sim.objects or self.sim.tracked_object(label) is not None]
        meshes = self.sim.object_meshes(tracked)  # one set of meshes for every view, as OracleKnowledge.describe
        out = {}
        for name, view, view_extras in capture_views(request, extras):
            masks = self.sim.oracle_masks(view, view_extras, tracked, meshes=meshes) if tracked else ()
            out[name] = {**{label: np.zeros(view["depth"].shape, bool) for label in labels},
                         **dict(zip(tracked, masks))}
        return Provided(out, "oracle", self.sim.n_steps)
