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
import omnigibson.utils.transform_utils as T

from b1k.bridge.geometry import box_corners
from b1k.bridge.judgement import highest_support
from b1k.bridge.knowledge import *  # noqa: F401,F403
from b1k.bridge.knowledge import KnowledgeSource, SceneKnowledge, register_source
from b1k.bridge.protocol import PLANNER_SUPPORT, bddl_label, capture_views, label_category

log = logging.getLogger(__name__)


def _frame(link) -> np.ndarray:
    """A link's world pose as a 4x4."""
    return T.pose2mat(link.get_position_orientation()).cpu().numpy().astype(np.float64)


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
        held, in_hand = self.hands()
        # E-level: a carrier with passengers (localized resting on it: a plate under a pizza) is kept level by the
        # planner (`level`, for the pick's lift and every carry), and once in the hand its passengers ride under its
        # label: part of what the hand carries, not bodies standing at the hand for the planner to avoid
        lifted = set(in_hand) | {self.sim.tracked_label(a["args"][0]) for a in atoms if a["predicate"] == "holding"}
        level = {label: riders for label in lifted if (riders := self.passengers(label))}
        if level:
            request["level"] = sorted(level)
            log.info(f"carried level: {level}")
            for carrier in set(level) & set(in_hand):
                for view_masks in masks.values():
                    for rider in level[carrier]:
                        view_masks[labels.index(carrier)] |= view_masks[labels.index(rider)]
                        view_masks[labels.index(rider)] = False
        self.sim.level = set(level)
        self.sim.remember_seen(views, labels, masks)  # the objects' shapes, as the depth under these masks sees them
        # The interior of a container an inside() goal names, as its placement surface. The planner has no
        # containment predicate, so inside(a, b) arrives as on(a, b) and is answered against b's own convex hull,
        # i.e. 2 mm above its LID. The same channel carries a fixture's board for touching(a, b), a named table's
        # top and the floor under a fixture for under(a, b) (r1pro.inside_regions). Privileged, like the button
        # poses; written on the request as `room` is, since attach_knowledge only carries the fixed set of keys.
        # After remember_seen: the board is chosen by the item's height as this capture's points see it too. The
        # regions of a stamp, cut, heat, aim or attach round need this source's hints (particles, heat link, frames):
        # fetched through `oracle`, so each privileged read has one name (section 1.6 of the build design).
        regions = self.sim.inside_regions(atoms, oracle=self) if getattr(self.sim, "send_inside", False) else {}
        workspace = self.sim.workspace(floor)
        if regions:
            request["place_surfaces"] = regions
            # a region above the planner's box (a nail at 1.71 m, installing_smoke_detectors) is cropped out of the
            # views with its target: the box reaches over the highest region's top, so the target has points
            top = max(box["pose"][2] + box["dims"][2] / 2.0 for box in regions.values())
            if top + 0.3 > workspace[1][2]:
                workspace[1][2] = top + 0.3
                log.info(f"the planner's workspace is raised to z={workspace[1][2]:.2f} for a region at {top:.2f}")
        side = getattr(self.sim, "side_grasp", None)  # Episode.pick marked this round: a roof within the hand stack
        if side:
            request["side_grasp"] = sorted(self.sim.label_of(name) for name in side)
        # an inside() container that got no region reaches the planner as on(a, b), which it cannot tell from a
        # book to stack on (E-stack judges those by the centre alone): named, so the whole footprint stays inside
        inside = {self.sim.tracked_label(a["args"][1]) for a in atoms if a["predicate"] == "inside"} - set(regions)
        if inside:
            request["inside"] = sorted(inside)
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
        carried = set(held) | set(in_hand)
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
            # the task's buttons every round, and this round's own: an instrumental press of a fixture the goal
            # never names (the stove that cooks, tracked by Episode.fixture_for) has its button described too
            buttons=self.sim.button_hints(self.goal + list(atoms), category_level=False),
            held_labels=sorted(set(held) | set(seen_obstacles)),
            in_hand=in_hand,
            workspace=workspace,
        )

    def passengers(self, label: str) -> list[str]:
        """Labels of the tracked objects resting on ``label``'s box by localization (``judgement.highest_support``:
        bottom within -2..+15 cm of its top, centre over it): what a plate or a sheet carries (E-level)."""
        names = getattr(self.sim, "bddl_names", {})  # label -> BDDL name (a tabletop sim tracks none)
        if label not in names:
            return []
        boxes = self.localize(*names.values())
        carrier = boxes.get(names[label])
        if carrier is None:
            return []
        return sorted(
            other
            for other, bddl in names.items()
            if other != label and bddl in boxes and highest_support(boxes[bddl], {label: carrier}) == label
        )

    def localize(self, *bddl_names):
        out = {}
        for name in bddl_names:
            try:
                obj = self.sim.scene_object(name)
            except ValueError:  # a transition product that has not appeared, or a whole a cut removed: left out
                continue
            lo, hi = [v.cpu().numpy().astype(np.float64) for v in obj.aabb]
            out[name] = {"center": obj.aabb_center.cpu().numpy().astype(np.float64), "lo": lo, "hi": hi}
        return out

    # ---------------------------------------------------------------- privileged hints, one named read each
    # Each is what a perception module would later supply (spec 6.2 names the replacement); the regions that use
    # them (r1pro.stamp_region and the others) take the values and stay pure geometry.
    def particles(self, target: str) -> np.ndarray:
        """(n, 3) world positions of the visual particles attached to ``target``, every system's: what a stamp must
        cover. Later: instance segmentation labels particles by system."""
        from omnigibson.systems.system_base import VisualParticleSystem

        group = VisualParticleSystem.get_group_name(obj=self.sim.scene_object(target))
        found = [
            s.get_group_particles_position_orientation(group)[0].cpu().numpy().astype(np.float64)
            for s in self.sim.env.scene.active_systems.values()
            if isinstance(s, VisualParticleSystem) and group in s.groups
        ]
        return np.concatenate(found) if found else np.zeros((0, 3))

    def heat_link(self, source: str):
        """(world xyz, radius m): the source's live heat link and how far it heats (HeatSourceOrSink.link, the first
        heatsource meta link, and its distance_threshold); None for a source that heats what is inside it (an oven, a
        microwave: a placement inside, not near). Later: burner detection."""
        from omnigibson.object_states import HeatSourceOrSink

        state = self.sim.scene_object(source).states.get(HeatSourceOrSink)
        if state is None or state.requires_inside:
            return None
        return _frame(state.link)[:3, 3], float(state.distance_threshold)

    def attach_frames(self, child: str, parent: str):
        """(male 4x4, female 4x4) world frames of the first free pair of attachment meta links between ``child`` and
        ``parent`` (AttachedTo's own candidates); None when they have none. Later: part detection."""
        from omnigibson.object_states import AttachedTo

        child_obj, parent_obj = self.sim.scene_object(child), self.sim.scene_object(parent)
        if AttachedTo not in child_obj.states or AttachedTo not in parent_obj.states:
            return None
        state, parent_state = child_obj.states[AttachedTo], parent_obj.states[AttachedTo]
        for male, females in (state._get_parent_candidates(parent_obj) or {}).items():
            for female in sorted(females):
                if parent_state.children.get(female) is None:
                    return _frame(state.links[male]), _frame(parent_state.links[female])
        return None

    def nozzle(self, tool: str):
        """(4x4 world frame, reach m) of the tool's particle applier: it sprays down the frame's -z for the
        projection mesh's height (particle_modifier.py:141-148, 434-444); None for a tool that applies by contact.
        Later: part detection."""
        from omnigibson.object_states import ParticleApplier

        state = self.sim.scene_object(tool).states.get(ParticleApplier)
        params = getattr(state, "_projection_mesh_params", None)
        if params is None:
            return None
        return _frame(state.link), float(params["extents"][2]) * float(state.link.scale[2])

    def projection_box(self, tool: str):
        """(lo, hi) in the tool's own frame: the box its particle remover's projection volume fills, hung below the
        meta link (tip at the link origin, down its -z: particle_modifier.py:434-444) -- the vacuum's removal slab,
        1-21 mm under its bottom (surface_verify_vacuum_usd.out); None for an adjacency remover, which takes its
        whole link. Later: part detection."""
        from omnigibson.object_states import ParticleRemover
        from omnigibson.utils.constants import ParticleModifyMethod

        obj = self.sim.scene_object(tool)
        state = obj.states.get(ParticleRemover)
        if state is None or state.method != ParticleModifyMethod.PROJECTION:
            return None
        ex = np.asarray(state._projection_mesh_params["extents"], dtype=np.float64) * state.link.scale.cpu().numpy()
        obj_from_link = np.linalg.inv(T.pose2mat(obj.get_position_orientation()).cpu().numpy()) @ _frame(state.link)
        corners = box_corners((-ex[0] / 2, -ex[1] / 2, -ex[2]), (ex[0] / 2, ex[1] / 2, 0.0))
        corners = corners @ obj_from_link[:3, :3].T + obj_from_link[:3, 3]
        return corners.min(axis=0), corners.max(axis=0)

    def appeared(self) -> list[str]:
        """BDDL names of task objects that now exist and were not tracked -- what a transition created (the halves of
        a cut) -- tracked here one by one under their labels (track_task_objects would wipe what fixture_for tracked);
        an object a transition removed (the whole) is dropped. Later: detection by category."""
        scope = self.sim.env.task.object_scope
        new = []
        for bddl, obj in self.sim.task_scope().items():
            if bddl not in self.sim.bddl_names.values() and label_category(bddl) != "table":
                label = bddl_label(bddl)
                self.sim.objects[label], self.sim.bddl_names[label] = obj, bddl
                new.append(bddl)
        for label, bddl in list(self.sim.bddl_names.items()):
            if bddl in scope and scope[bddl] is None:
                self.sim.objects.pop(label, None)
                self.sim.bddl_names.pop(label)
                log.info(f"{bddl} is gone (a transition removed it); no longer tracked")
        if new:
            log.info(f"appeared: {new}, now tracked")
        return sorted(new)
