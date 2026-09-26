"""R1ProSim as the skill bench's host (SPEC §8 Track A2): DirectConnector's env and adapter, and the HostHooks a
self-stepping run hands back through."""

import logging
import time
from functools import cached_property

import numpy as np

from b1k.connector.observe import StepObs
from b1k.connector.types import Provided
from b1k.observation import PROPRIO_SLICES, CameraView

log = logging.getLogger("omnigibson.tiptop")


class Frames:
    """The robot cameras' CURRENT images for the GoalPanel (SPEC §4.1, gate item 5): the head and both wrists rendered
    from where they stand, with the oracle segmenter's masks of every tracked object in every view, keyed by BDDL name
    and source-tagged. Percept-shaped (views, masks, view_masks), so the PerceptionVerifier reads a StepObs's sensors
    as it reads a Percept. Reading sensors is not a capture (the planner owns those): no sim step, no aim, no arm
    swing. A StepObs is made every step and a checker reads one per run, so nothing renders until it is read."""

    def __init__(self, host):
        self.host = host

    @cached_property
    def _frames(self) -> tuple:
        sim, t0 = self.host.sim, time.monotonic()
        names = [v for v in (sim.primary_view, *sim.extra_views) if v in sim.robot_cam_names]  # its own camera: no turn
        rendered = {name: sim.view_frame(name) for name in names}  # the shadow camera onto the robot camera, one render
        head, extras = rendered[names[0]]
        request = {**head, "view_name": names[0], "views": [{**rendered[n][0], "name": n} for n in names[1:]]}
        extras = {**extras, "views": {n: rendered[n][1] for n in names[1:]}}
        labels = list(sim.objects)  # ponytail: every tracked object, not the goal's alone (the carrier never sees the goal)
        masks = self.host.segmenter.masks(labels, request, extras)
        ids = [sim.bddl_names.get(label, label) for label in labels]
        by_view = {n: dict(zip(ids, (masks.value[n][label] for label in labels))) for n in names}
        views = {n: CameraView(n, v["rgb"], v["depth"], v["intrinsics"], v["world_from_cam"]) for n, (v, _) in rendered.items()}
        self.host.frames_wall_s += time.monotonic() - t0
        log.info(f"frames for the GoalPanel at sim step {sim.n_steps}: {names}, {len(labels)} objects, "
                 f"{time.monotonic() - t0:.1f} s, no sim step")
        return views, Provided(by_view[names[0]], masks.source, masks.step), Provided(by_view, masks.source, masks.step)

    views = property(lambda self: self._frames[0])
    masks = property(lambda self: self._frames[1])
    view_masks = property(lambda self: self._frames[2])


class BenchHost:
    """env_step / parse for DirectConnector; commanded_targets / observe_now for the Runtime (HostHooks). A StepObs
    carries the evaluator's 61 proprio numbers, read off the robot's own proprioception in the eval layout, and, once
    the trial's segmenter is set, the cameras' current Frames (rendered only when a checker reads them: the planner
    owns observation). Counts the wall seconds spent inside env steps and inside frame renders, so a trial can tell
    the planning wall time from the execution's."""

    def __init__(self, sim, segmenter=None):
        self.sim, self.segmenter, self.env_wall_s, self.frames_wall_s = sim, segmenter, 0.0, 0.0

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

    def frames(self):
        return Frames(self) if self.segmenter is not None else None

    def parse(self, raw: dict, step: int) -> StepObs:
        return StepObs(step, raw["proprio"], raw, sensors=self.frames())

    def commanded_targets(self) -> dict:
        return self.sim.commanded_targets()

    def observe_now(self) -> StepObs:
        raw = self.raw()
        return StepObs(self.sim.n_steps, raw["proprio"], raw, sensors=self.frames())
