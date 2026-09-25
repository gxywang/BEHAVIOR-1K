"""R1ProSim as the skill bench's host (SPEC §8 Track A2): DirectConnector's env and adapter, and the HostHooks a
self-stepping run hands back through."""

import time

import numpy as np

from b1k.connector.observe import StepObs
from b1k.observation import PROPRIO_SLICES


class BenchHost:
    """env_step / parse for DirectConnector; commanded_targets / observe_now for the Runtime (HostHooks). A StepObs
    carries the evaluator's 61 proprio numbers, read off the robot's own proprioception in the eval layout, and no
    frames: observation is the planner's observe(). Counts the wall seconds spent inside env steps, so a trial can
    tell the planning wall time from the execution's."""

    def __init__(self, sim):
        self.sim, self.env_wall_s = sim, 0.0

    def proprio(self) -> np.ndarray:
        d = self.sim.robot._get_proprioception_dict()
        return np.concatenate([d[k].cpu().numpy().reshape(-1) for k in PROPRIO_SLICES]).astype(np.float32)

    def raw(self) -> dict:
        return {"proprio": self.proprio()}

    def env_step(self, a23) -> dict:
        t0 = time.monotonic()
        self.sim.step_action(a23)
        self.env_wall_s += time.monotonic() - t0
        return self.raw()

    def parse(self, raw: dict, step: int) -> StepObs:
        return StepObs(step, raw["proprio"], raw)

    def commanded_targets(self) -> dict:
        return self.sim.commanded_targets()

    def observe_now(self) -> StepObs:
        raw = self.raw()
        return StepObs(self.sim.n_steps, raw["proprio"], raw)
