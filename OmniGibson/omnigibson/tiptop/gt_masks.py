"""Moved to ``b1k.bridge.gt_masks``: the masks are computed from a depth image and trimesh surfaces, with no
Isaac annotator and no prim, so they live with the policy.

Kept here as a re-export; new code should import ``b1k.bridge.gt_masks`` directly.
"""

from b1k.bridge.gt_masks import *  # noqa: F401,F403
from b1k.bridge.gt_masks import _prepare, _triangle_distances  # noqa: F401
