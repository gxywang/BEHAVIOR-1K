"""R1ProSim as the skill bench's host (SPEC §8 Track A2): DirectConnector's env and adapter, and the HostHooks a
self-stepping run hands back through."""

import logging
import math
from dataclasses import asdict, replace
import time
from functools import cached_property

import numpy as np

from b1k.connector.observe import BaseState, StepObs
from b1k.connector.types import Pose2, Provided
from b1k.observation import PROPRIO_SLICES, CameraView

log = logging.getLogger("omnigibson.tiptop")


class Frames:
    """The robot cameras' CURRENT images for the GoalPanel (SPEC §4.1, gate item 5): the head and both wrists rendered
    from where they stand, with the oracle segmenter's masks of every tracked object in every view, keyed by BDDL name
    and source-tagged. Percept-shaped (views, masks, view_masks), so the PerceptionVerifier reads a StepObs's sensors
    as it reads a Percept. Reading sensors is not a capture (the planner owns those): no sim step, no aim, no arm
    swing. A StepObs is made every step and a checker reads one per run, so nothing renders until it is read."""

    def __init__(self, host, step=None):
        self.host = host
        self.sim_step = host.sim.n_steps
        self.step = self.sim_step if step is None else step

    @cached_property
    def _frames(self) -> tuple:
        sim, t0 = self.host.sim, time.monotonic()
        if self.host.whole_body and sim.n_steps != self.sim_step:
            raise ValueError("whole-body frames must be read at their observation step")
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
        _, map_from_base = self.host.base_measurement()
        if map_from_base is not None:
            map_from_base = replace(map_from_base, step=self.step)
        return (views, Provided(by_view[names[0]], masks.source, masks.step),
                Provided(by_view, masks.source, masks.step), map_from_base)

    views = property(lambda self: self._frames[0])
    masks = property(lambda self: self._frames[1])
    view_masks = property(lambda self: self._frames[2])
    map_from_base = property(lambda self: self._frames[3])


class BenchHost:
    """env_step / parse for DirectConnector; commanded_targets / observe_now for the Runtime (HostHooks). A StepObs
    carries the evaluator's 61 proprio numbers, read off the robot's own proprioception in the eval layout, and, once
    the trial's segmenter is set, the cameras' current Frames (rendered only when a checker reads them: the planner
    owns observation). Counts the wall seconds spent inside env steps and inside frame renders, so a trial can tell
    the planning wall time from the execution's."""

    def __init__(self, sim, segmenter=None, whole_body: bool = False):
        self.sim, self.segmenter, self.env_wall_s, self.frames_wall_s = sim, segmenter, 0.0, 0.0
        self.whole_body, self.base_trace = whole_body, []

    def base_measurement(self, proprio=None) -> tuple:
        """An explicit oracle localization provider for development; no command or simulation step is involved."""
        if not self.whole_body:
            return None, None
        from scipy.spatial.transform import Rotation

        pos, quat = self.sim.base_pose()  # base_link, identical to the frame used by view_frame / to_base
        pos, quat = (np.asarray(v.cpu() if hasattr(v, "cpu") else v, dtype=np.float64) for v in (pos, quat))
        rotation = Rotation.from_quat(quat).as_matrix()
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        p = self.proprio() if proprio is None else proprio
        state = BaseState(Pose2(float(pos[0]), float(pos[1]), yaw, float(pos[2])),
                          tuple(float(v) for v in p[PROPRIO_SLICES["base_qvel"]]), roll, pitch)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3], transform[:3, 3] = rotation, pos
        transform.setflags(write=False)
        return Provided(state, "oracle", self.sim.n_steps), Provided(transform, "oracle", self.sim.n_steps)

    def proprio(self) -> np.ndarray:
        d = self.sim.robot._get_proprioception_dict()
        return np.concatenate([d[k].cpu().numpy().reshape(-1) for k in PROPRIO_SLICES]).astype(np.float32)

    def raw(self) -> dict:
        p = self.proprio()
        state, transform = self.base_measurement(p)
        return {"proprio": p, "base_state": state, "map_from_base": transform, "sim_step": self.sim.n_steps}

    def env_step(self, a23) -> dict:
        t0 = time.monotonic()
        self.sim.step_action(a23)
        self.env_wall_s += time.monotonic() - t0
        raw = self.raw()
        if self.whole_body:
            self.base_trace.append({"step": self.sim.n_steps, "base_state": asdict(raw["base_state"]),
                                    "map_from_base": asdict(raw["map_from_base"]),
                                    "base_action": np.asarray(a23, dtype=np.float32)[:3].copy(),
                                    "action23": np.asarray(a23, dtype=np.float32).copy(),
                                    "proprio": raw["proprio"].copy()})
        return raw

    def frames(self, step=None):
        return Frames(self, step) if self.segmenter is not None else None

    def parse(self, raw: dict, step: int) -> StepObs:
        if self.whole_body:
            # Only this adapter's freshly measured sample may move from the sim clock (which includes captures)
            # to DirectConnector's execution clock. A stale/external provider is never made fresh by re-stamping.
            if raw.get("sim_step") != self.sim.n_steps:
                raise ValueError("cannot re-stamp a stale whole-body host observation")
            raw = dict(raw)
            for key in ("base_state", "map_from_base"):
                provided = raw.get(key)
                if provided is None or provided.step != raw["sim_step"]:
                    raise ValueError(f"whole-body host {key} is missing or stale")
                raw[key] = replace(provided, step=step)
        return StepObs(step, raw["proprio"], raw, sensors=self.frames(step), base_state=raw.get("base_state"))

    def commanded_targets(self) -> dict:
        return self.sim.commanded_targets()

    def observe_now(self) -> StepObs:
        raw = self.raw()
        return self.parse(raw, self.sim.n_steps)

    def observe_at(self, step: int) -> StepObs:
        """Fresh measurement on the Runtime clock after a self-stepping capture or navigation service."""
        return self.parse(self.raw(), step)
