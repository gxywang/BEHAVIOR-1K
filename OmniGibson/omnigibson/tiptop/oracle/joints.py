"""The oracle JointStateEstimator (SPEC D27): the live joint value, which the map never carries."""

import numpy as np

from b1k.connector.types import Provided


class OracleJoints:
    """The simulator's joint position: oracle (counted in pseudo, refused in competition). A depth estimator (the
    moving link's map voxels fitted to the head cloud) or a proprio one (the hand's arc while it holds the handle)
    replaces it."""

    def __init__(self, sim):
        self.sim = sim

    def value(self, o, joint: str) -> Provided:
        q = self.sim.scene_object(o.id).joints[joint].get_state()[0]
        return Provided(float(np.asarray(q).reshape(-1)[0]), "oracle", self.sim.n_steps)
