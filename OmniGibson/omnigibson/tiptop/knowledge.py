"""The privileged knowledge source, and the old address of the rest.

``OracleKnowledge`` is what the simulator knows and an agent does not: per-instance labels rather than categories,
object masks computed from the objects' own meshes at their own poses (``gt_masks``), the true pose of every
toggle button the task presses, and ``localize`` straight off each object's AABB. That is exactly the information
the challenge forbids at evaluation time. It exists so planning and execution can be developed and measured
without a detector in the way, and a run that used it says so (``report``).

It stays here because of that, not because of what it imports -- a producer of ground truth is not policy code,
however clean its import list. Everything else that was in this module is policy and moved to
``b1k.bridge.knowledge``: ``SceneKnowledge``, ``ButtonTracker``, the ``KnowledgeSource`` base, ``OnboardKnowledge``
(which sends what an agent in the challenge would) and ``make_knowledge``. Importing this module registers the
oracle into that registry, so ``make_knowledge("oracle", sim, goal)`` keeps working for every caller here, and the
PRIVILEGED warning still fires from ``make_knowledge`` because the class declares ``privileged = True``.

The names below are re-exported, so ``from omnigibson.tiptop.knowledge import ...`` is unchanged.
"""

import logging

import numpy as np

from b1k.bridge.knowledge import *  # noqa: F401,F403
from b1k.bridge.knowledge import KnowledgeSource, SceneKnowledge, register_source
from b1k.bridge.protocol import PLANNER_SUPPORT, capture_views

log = logging.getLogger(__name__)


@register_source
class OracleKnowledge(KnowledgeSource):
    """The simulator's truth: per-instance labels, masks from geometry, the true pose of every button the task
    presses (sent in every round so the pick round can choose a grasp that presents it). Privileged."""

    name = "oracle"
    privileged = True

    def describe(self, atoms, request, extras, floor=False) -> SceneKnowledge:
        labels, tiptop_atoms = self.translate(atoms)
        # The furniture standing around the robot, so the planner has a world to plan in. cuTAMP's collision world
        # otherwise holds the task's own objects and one fitted plane, and it plans straight through everything
        # else in the room -- which is what the bridge has been compensating for, in the wrong place. These reach
        # cuTAMP through held_labels, which it takes as statics: obstacles to plan around, never things to pick
        # up. They have to be segmented here too, since a label with no hull is dropped by the planner as "not an
        # obstacle then: perception did not reconstruct it".
        # nearby_obstacles registers what it returns in sim.obstacles, which is what object_meshes resolves a
        # furniture label through (tracked_object). Obstacles are kept OUT of sim.objects on purpose: they are
        # geometry for the planner to avoid, not things the episode poses, checks or frames.
        obstacles = self.sim.nearby_obstacles(exclude=labels) if getattr(self.sim, "send_obstacles", False) else []
        labels = list(labels) + [o for o in obstacles if o not in labels]
        views = capture_views(request, extras)
        meshes = self.sim.object_meshes(labels)  # one mesh per label for every view's masks
        masks = {
            name: self.sim.oracle_masks(view, view_extras, labels, meshes=meshes) for name, view, view_extras in views
        }
        counts = {
            name: {label: int(m.sum()) for label, m in zip(labels, view_masks)} for name, view_masks in masks.items()
        }
        visible = [label for label in labels if any(counts[name][label] for name in counts)]
        hidden = [label for label in labels if label not in visible]
        # Every object a goal atom names has to be one the planner is given, and the planner is given the visible
        # ones PLUS two kinds of label that never carry a mask. Testing against `hidden` missed an atom naming
        # something that never became a label at all -- the round went out and the planner rejected it with "Goal
        # predicate holding(toy_figure_3) references unknown object 'toy_figure_3'" -- but testing against
        # `visible` alone was worse: it refuses every goal that names the support plane or a button.
        #
        # PLANNER_SUPPORT ("table") is the plane tiptop fits by RANSAC, a surface and never a detected object, so
        # it is never segmented and never visible; every `ontop(item, table)` and every put-down onto the floor,
        # which is rewritten to the same label, was being refused. A button is named by pose through button_hints
        # rather than found as an object, so a press was refused the same way. The planner exempts both itself.

        # An object IN THE ROBOT'S OWN HAND is the third kind of label that carries no mask, and for the same
        # reason: it is given to the planner by the `in_hand` carry feature rather than found in the picture. The
        # head camera cannot see into the gripper and the wrist camera is behind it, so a placement round for
        # something the robot is already holding was refused before it was ever planned -- "goal objects
        # ['tile_3'] are not visible in any view ['head', 'left_wrist', 'right_wrist'] (empty masks)" while
        # tile_3 was in the gripper. It cost every placement of a carried object, which is exactly what the
        # pressed grasp has just started producing more of (2026-09-15).
        carried = set(sum(self.hands(), []))
        rescued = sorted((({a for atom in tiptop_atoms for a in atom["args"]}) & carried) - set(visible))
        if rescued:
            log.info(f"{rescued} carry no mask because the robot is holding them; the planner is told so")
        exempt = (
            {PLANNER_SUPPORT}
            | {a for atom in tiptop_atoms if atom.get("predicate") == "pressed" for a in atom["args"]}
            | carried
        )
        needed = sorted({a for atom in tiptop_atoms for a in atom["args"] if a not in visible and a not in exempt})
        if needed:
            untracked = [a for a in needed if a not in labels]
            raise GoalNotVisible(
                f"goal objects {needed} are not visible in any view {list(counts)} (empty masks"
                + (f"; {untracked} are not tracked at all" if untracked else "")
                + ")"
            )
        for name in counts:
            log.info(f"oracle masks in {name}: pixels per label { {label: counts[name][label] for label in visible} }")
        log.info(f"out of every view: {hidden or 'none'}")
        keep = [labels.index(label) for label in visible]
        primary = views[0][0]
        held, in_hand = self.hands()
        seen_obstacles = [o for o in obstacles if o in visible]
        if obstacles:
            log.info(
                f"furniture sent to the planner as obstacles: {seen_obstacles or 'none'}"
                + (
                    f" ({len(obstacles) - len(seen_obstacles)} had no pixels)"
                    if len(seen_obstacles) < len(obstacles)
                    else ""
                )
            )
        return SceneKnowledge(
            labels=visible,
            atoms=tiptop_atoms,
            masks=masks[primary][keep],
            view_masks={name: view_masks[keep] for name, view_masks in masks.items() if name != primary},
            buttons=self.sim.button_hints(self.goal, category_level=False),
            held_labels=sorted(set(held) | set(seen_obstacles)),
            in_hand=in_hand,
            workspace=self.sim.workspace(floor),
        )

    def localize(self, *bddl_names):
        out = {}
        for name in bddl_names:
            obj = self.sim.scene_object(name)
            lo, hi = [v.cpu().numpy().astype(np.float64) for v in obj.aabb]
            out[name] = {"center": obj.aabb_center.cpu().numpy().astype(np.float64), "lo": lo, "hi": hi}
        return out
