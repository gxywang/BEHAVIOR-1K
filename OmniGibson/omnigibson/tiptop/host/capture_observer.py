"""The bench Observer (SPEC D4): the planner's observe() on R1ProSim."""

from b1k.bridge.protocol import capture_views
from b1k.connector.observe import Percept, PerceptInfo
from b1k.runtime.compose import PROPRIO_Q


class CaptureObserver:
    """R1ProSim.capture aims the head through the trunk, swings the free arms out of view and steps the sim itself,
    so this is requires_sim_clock: it ends before its first yield (0 Runtime steps) and the Runtime re-seeds its
    latch from the host afterwards. Views are the capture's own per-view dicts (the server's request form); masks
    come from the injected segmenter, with its source tag. ``visible`` is 1.0 for a target any view has pixels of."""

    requires_sim_clock = True

    def __init__(self, sim, host, segmenter, task: str):
        self.sim, self.host, self.segmenter, self.task = sim, host, segmenter, task

    def observe(self, req, obs):
        if not req.aim:  # R1ProSim.capture always re-aims; a capture from where the head is lands with hold (week 2)
            raise NotImplementedError("observe(aim=False) is not on the bench yet")
        request, extras = self.sim.capture(self.task)
        labels = [self.sim.tracked_label(o.id) for o in req.targets]
        masks = self.segmenter.masks(labels, request, extras)
        after = self.host.observe_now()
        seen = lambda label: float(any(m[label].any() for m in masks.value.values()))
        visible = {o.id: seen(label) for o, label in zip(req.targets, labels)}
        info = PerceptInfo("", after.step, 0, 0, visible, masks.source)  # the Runtime stamps id and epochs
        views = {name: view for name, view, _ in capture_views(request, extras)}
        return Percept(info, views, {g: after.proprio[s].copy() for g, s in PROPRIO_Q.items()}, masks), after
        yield {}  # unreachable: a generator that ends before its first yield
