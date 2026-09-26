"""The oracle ButtonLocator (SPEC §4.1): a toggle button by pose, off the simulator's ToggledOn marker and the physical
face it sits on (R1ProSim.button_world, privileged), in the map frame and tagged oracle. The ButtonTracker (perception)
replaces it: a switch is below the map's voxel size."""

from b1k.connector.types import Provided
from b1k.connector.world import ButtonSpec

# m past the face the stroke aims: ToggledOn flips after 5 steps with a finger inside its marker sphere, which sits
# behind the knob's face on the radio; today's push_depth. A wall stops the hand earlier: the server clamps the stroke
# there (press_diagnosis.md).
STROKE = 0.015


class OracleButtons:
    def __init__(self, sim):
        self.sim = sim

    def button(self, o) -> Provided:
        try:
            point, normal, radius = self.sim.button_world(o.id)
        except ValueError:  # no ToggledOn state, or no physical mesh to find its face on
            return Provided(None, "oracle", self.sim.n_steps)
        return Provided(ButtonSpec(tuple(map(float, point)), tuple(map(float, normal)), STROKE, o, float(radius)),
                        "oracle", self.sim.n_steps)
