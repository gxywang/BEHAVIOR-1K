"""The bench Observer (SPEC D4): the planner's observe() on R1ProSim."""

import time

from b1k.bridge.protocol import capture_views
from b1k.connector.observe import Percept, PerceptInfo
from b1k.connector.types import Provided
from b1k.observation import CameraView
from b1k.runtime.compose import PROPRIO_Q


class CaptureObserver:
    """R1ProSim.capture aims the head through the trunk, swings the free arms out of view and steps the sim itself,
    so this is requires_sim_clock: it ends before its first yield (0 Runtime steps) and the Runtime re-seeds its
    latch from the host afterwards. It frames the request's targets first (R1ProSim.look_at: the head aim and the
    wrist looks), as a round that stood for them did. It captures the views the sim was built with (``views``) and
    refuses a request for others, or for a look_at point; its settle is R1ProSim's own, not settle_steps.
    Views are CameraViews (the Percept's type), the primary first; masks are the primary view's, keyed by ObjRef.id,
    from the injected segmenter with its source tag, and ``visible`` is read off the same masks: a target only a
    wrist view sees is not visible in the Percept the builder sends. ``wall_s`` sums the seconds spent capturing,
    which are not planning, and ``steps`` the sim steps the captures took (the bench's U0 check adds them)."""

    requires_sim_clock = True

    def __init__(self, sim, host, segmenter, task: str):
        self.sim, self.host, self.segmenter, self.task = sim, host, segmenter, task
        self.wall_s, self.steps = 0.0, 0

    @property
    def views(self) -> tuple:
        return (self.sim.primary_view, *self.sim.extra_views)

    def observe(self, req, obs):
        if not req.aim:  # R1ProSim.capture always re-aims; a capture from where the head is lands with hold (week 2)
            raise NotImplementedError("observe(aim=False) is not on the bench yet")
        if tuple(req.views) != self.views or req.look_at is not None:
            raise NotImplementedError(f"the bench captures {self.views}, framed on the targets; asked for "
                                      f"{tuple(req.views)} at {req.look_at}")
        t0, n0 = time.monotonic(), self.sim.n_steps
        self.sim.look_at(*(o.id for o in req.targets))
        request, extras = self.sim.capture(self.task)
        labels = [self.sim.tracked_label(o.id) for o in req.targets]
        masks = self.segmenter.masks(labels, request, extras)
        self.wall_s += time.monotonic() - t0
        self.steps += self.sim.n_steps - n0
        after = self.host.observe_now()
        views = {name: CameraView(name, v["rgb"], v["depth"], v["intrinsics"], v["world_from_cam"])
                 for name, v, _ in capture_views(request, extras)}  # fmt: skip
        primary = masks.value[next(iter(views))]
        visible = {o.id: float(primary[label].any()) for o, label in zip(req.targets, labels)}
        info = PerceptInfo("", after.step, 0, 0, visible, masks.source)  # the Runtime stamps id and epochs
        on_primary = Provided({o.id: primary[label] for o, label in zip(req.targets, labels)}, masks.source, masks.step)
        return Percept(info, views, {g: after.proprio[s].copy() for g, s in PROPRIO_Q.items()}, on_primary), after
        yield {}  # unreachable: a generator that ends before its first yield
