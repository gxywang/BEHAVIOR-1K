"""OracleSegmenter (SPEC §4.1): instance masks from the simulator's geometry, where a detector would give them."""

import numpy as np

from b1k.bridge.protocol import FLOOR_CATEGORIES, capture_views, label_category
from b1k.connector.types import Provided

TOUCH = 0.01  # m: gt_masks' 8 mm surface tolerance, with margin


class OracleSegmenter:
    """{view: {label: (H, W) bool}} for every view of one capture, from the tracked objects' meshes at their true
    poses (scene.oracle_masks, the oracle knowledge source's masks). Oracle-tagged. A pixel within the tolerance of
    two masked surfaces goes to the nearer one (gt_masks), so every tracked object whose box comes within TOUCH of a
    label's is masked with it and left out of the answer: alone, apple_1 took 18 pixels of the apple_2 it leans on
    (w2s2 freeze_fruit), which made its cloud 2 cm taller and gave the pick side grasps through the bowl."""

    def __init__(self, sim):
        self.sim = sim

    def masks(self, labels: list, request: dict, extras: dict) -> Provided:
        # a place's target may be furniture no tracked object stands for (the floor, an untracked table, a burner):
        # an empty mask; its region is the map's support (SPEC 6.2), never a segmented object. Any other label no
        # tracked object stands for (a movable the bench did not track, a misnamed object) is the harness's error,
        # never an empty mask a skill would answer NOT_VISIBLE to
        tracked = [label for label in labels if label in self.sim.objects or self.sim.tracked_object(label) is not None]
        stray = [label for label in labels if label not in tracked and not self.furniture(label)]
        if stray:
            raise ValueError(f"no tracked object for labels {stray}")
        every = [*tracked, *self.touching(tracked)]
        meshes = self.sim.object_meshes(every)  # one set of meshes for every view, as OracleKnowledge.describe
        out = {}
        for name, view, view_extras in capture_views(request, extras):
            masks = self.sim.oracle_masks(view, view_extras, every, meshes=meshes) if every else ()
            out[name] = {**{label: np.zeros(view["depth"].shape, bool) for label in labels},
                         **dict(zip(tracked, masks))}
        return Provided(out, "oracle", self.sim.n_steps)

    def furniture(self, label: str) -> bool:
        """Whether an untracked label is what the bench leaves untracked on purpose, a support: a fixed piece of the
        scene (a floor, a burner) or a task table (track_task_objects skips tables and floors)."""
        if label_category(label) in ("table", *FLOOR_CATEGORIES):
            return True
        try:
            return bool(getattr(self.sim.scene_object(label), "fixed_base", False))
        except ValueError:  # no such object in the scene
            return False

    def touching(self, labels: list) -> list:
        """The other tracked objects whose world box comes within TOUCH of a label's."""
        boxes = {label: [v.cpu().numpy() for v in obj.aabb] for label, obj in self.sim.objects.items()}
        mine = [boxes[label] for label in labels if label in boxes]
        return [label for label, (lo, hi) in boxes.items() if label not in labels
                and any((lo <= b + TOUCH).all() and (hi >= a - TOUCH).all() for a, b in mine)]
