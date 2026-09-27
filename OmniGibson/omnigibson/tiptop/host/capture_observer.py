"""The bench Observer (SPEC D4): the planner's observe() on R1ProSim."""

import time

import numpy as np

from b1k.bridge.protocol import capture_views
from b1k.connector.observe import Percept, PerceptInfo
from b1k.connector.types import Provided
from b1k.observation import CameraView, pose_to_matrix
from b1k.runtime.compose import PROPRIO_Q
from omnigibson.tiptop.scene import TiptopSim


class CaptureObserver:
    """R1ProSim.capture aims the head through the trunk, swings the free arms out of view and steps the sim itself,
    so this is requires_sim_clock: it ends before its first yield (0 Runtime steps) and the Runtime re-seeds its
    latch from the host afterwards. With aim it frames the request's targets first (R1ProSim.look_at: the head aim
    and the wrist looks), as a round that stood for them did. Without aim (trunk-read: allowed beside a lease) it
    renders the views from wherever the cameras stand, TiptopSim.capture: no look, no swing, no head turn, 0 sim
    steps. It captures the views the sim was built with (``views``) and refuses a request for others, or for a
    look_at point; its settle is R1ProSim's own, not settle_steps.
    Views are CameraViews (the Percept's type), the primary first; masks are the primary view's, keyed by ObjRef.id,
    from the injected segmenter with its source tag, and ``visible`` is read off the same masks: a target only a
    wrist view sees is not visible in the Percept the builder sends. ``view_masks`` carries every view's masks the
    same way, so a builder can send the wrist views too. ``wall_s`` sums the seconds spent capturing, which are not
    planning, and ``steps`` the sim steps the captures took (the bench's U0 check adds them)."""

    requires_sim_clock = True

    def __init__(self, sim, host, segmenter, task: str):
        self.sim, self.host, self.segmenter, self.task = sim, host, segmenter, task
        self.wall_s, self.steps = 0.0, 0

    @property
    def views(self) -> tuple:
        return (self.sim.primary_view, *self.sim.extra_views)

    def observe(self, req, obs):
        if tuple(req.views) != self.views or req.look_at is not None:
            raise NotImplementedError(f"the bench captures {self.views}, framed on the targets; asked for "
                                      f"{tuple(req.views)} at {req.look_at}")
        t0, n0 = time.monotonic(), self.sim.n_steps
        if req.aim:
            self.sim.look_at(*(o.id for o in req.targets))
            request, extras = self.sim.capture(self.task)
        else:
            request, extras = TiptopSim.capture(self.sim, self.task)  # the cameras where they are: nothing moves
        seen = tuple(dict.fromkeys((*req.targets, *req.context)))  # the context is masked, never aimed at
        labels = [self.sim.tracked_label(o.id) for o in seen]
        masks = self.segmenter.masks(labels, request, extras)
        self.wall_s += time.monotonic() - t0
        self.steps += self.sim.n_steps - n0
        after = self.host.observe_now()
        map_from_base = after.raw.get("map_from_base")
        mobile = bool(getattr(self.host, "whole_body", False))
        if mobile:
            if map_from_base is None or map_from_base.source != "oracle" or map_from_base.step != after.step:
                raise ValueError("whole-body oracle capture requires a fresh oracle map_from_base")
            base_from_map = np.linalg.inv(map_from_base.value)
        views = {}
        for name, view, camera in capture_views(request, extras):
            base_from_cam = view["world_from_cam"]
            if mobile:
                # A capture can step between views and while returning the arm. Each camera's render-time
                # world pose is authoritative; express every saved view in the SAME measured after-base.
                if "cam_pos_world" not in camera or "cam_quat_xyzw_world_cv" not in camera:
                    raise ValueError(f"whole-body capture {name} lacks render-time world camera pose")
                map_from_cam = pose_to_matrix(camera["cam_pos_world"], camera["cam_quat_xyzw_world_cv"])
                if not np.isfinite(map_from_cam).all():
                    raise ValueError(f"whole-body capture {name} has invalid render-time camera pose")
                base_from_cam = np.asarray(base_from_map @ map_from_cam, dtype=np.float32)
                base_from_cam.setflags(write=False)
            views[name] = CameraView(name, view["rgb"], view["depth"], view["intrinsics"], base_from_cam)
        by_view = {name: {o.id: masks.value[name][label] for o, label in zip(seen, labels)} for name in views}
        primary = by_view[next(iter(views))]
        visible = {o.id: float(primary[o.id].any()) for o in seen}
        info = PerceptInfo("", after.step, 0, 0, visible, masks.source)  # the Runtime stamps id and epochs
        q = {g: after.proprio[s].copy() for g, s in PROPRIO_Q.items()}
        return Percept(info, views, q, Provided(primary, masks.source, masks.step),
                       Provided(by_view, masks.source, masks.step), map_from_base), after
        yield {}  # unreachable: a generator that ends before its first yield
