"""What the client tells the planner about the scene beyond the RGB-D image, and where that knowledge comes from.

The planner needs the names of the objects the goal talks about and the goal itself; everything else it works out
from the image. ``OnboardKnowledge`` sends just that, the way an agent in the challenge would: category names, the
goal atoms, and what its own hands hold (its gripper state). A toggle button is named for the detector to find on
its object, and a button seen in an earlier round is carried through a grasp by the arm's kinematics
(``ButtonTracker``). ``OracleKnowledge`` reads the simulator instead: per-instance labels, masks from the objects'
geometry (or Isaac's instance segmentation), and the true pose of every toggle button the task presses. That is
privileged information the challenge forbids at evaluation time; it exists so planning and execution can be
developed and measured without a detector. The oracle gives exactly two kinds of thing, perception (masks, button
poses) and localization (``localize``: where the task objects are, as boxes); it never says whether a round worked.
Whether a pick or a place succeeded is judged by the episode from the robot's own readings and from localization
(bench.py, ``Episode.satisfied``), and a press runs its planned stroke with no signal from the switch. A run that
uses the oracle says so (``report``). Both sources produce the same ``SceneKnowledge``; the rest of the pipeline
never asks which one it got.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from omnigibson.tiptop.protocol import attach_knowledge, canonical_object_name, capture_views

log = logging.getLogger(__name__)

SEEN_HALF_EXTENT = 0.10  # m: the box the onboard source draws around a perceived position it has no extent for


class GoalNotVisible(ValueError):
    """A goal object has no pixels in any view of the capture: the planner would not see it."""


@dataclass
class SceneKnowledge:
    """One request's worth of knowledge in the planner's terms: labels, TiPToP predicates, the robot base frame."""

    labels: list
    atoms: list
    masks: np.ndarray | None = None  # (N, H, W) bool aligned with labels, the primary view; None: the planner detects
    view_masks: dict = field(default_factory=dict)  # view name -> (N, H_v, W_v) bool, the further views
    buttons: dict = field(default_factory=dict)  # label -> {position, normal, radius}
    held_labels: list = field(default_factory=list)  # in a hand the plan does not move
    in_hand: list = field(default_factory=list)  # in the planned hand: the plan starts holding them
    workspace: list | None = None  # base-frame box for this request (the embodiment's, TiptopSim.workspace)

    def attach(self, request: dict) -> dict:
        return attach_knowledge(
            request,
            self.labels,
            self.atoms,
            masks=self.masks,
            buttons=self.buttons,
            held=self.held_labels,
            in_hand=self.in_hand,
            workspace=self.workspace,
            view_masks=self.view_masks,
        )

    def summary(self) -> dict:
        """What went into the request, for capture.json and the log (masks by pixel count, per view)."""
        return {
            "labels": list(self.labels),
            "atoms": list(self.atoms),
            "mask_pixels": {label: int(m.sum()) for label, m in zip(self.labels, self.masks)}
            if self.masks is not None
            else None,
            "view_mask_pixels": {
                name: {label: int(m.sum()) for label, m in zip(self.labels, masks)}
                for name, masks in self.view_masks.items()
            },
            "buttons": dict(self.buttons),
            "held_labels": list(self.held_labels),
            "in_hand": list(self.in_hand),
            "workspace": self.workspace,
        }


class ButtonTracker:
    """Buttons the planner detected, kept across rounds. A button belongs to the object its label names
    (``<object>_button``); once that object is grasped it moves rigidly with the gripper, so the pose detected
    earlier is carried along with the arm's own kinematics: p_now = T_eef_now @ inv(T_eef_at_grasp) @ p_then."""

    def __init__(self):
        self.specs = {}  # label -> {position, normal, radius} (base frame) as detected, and the eef pose then if held

    def update(self, buttons: dict, held: dict) -> None:
        """Record the planner's detected buttons; ``held`` maps an object label to (arm, 4x4 eef pose now) for
        objects in hand at the time of the capture (their detection is relative to that gripper pose)."""
        for label, spec in (buttons or {}).items():
            if spec.get("source") != "detected":
                continue
            parent = label[: -len("_button")]
            entry = {k: [float(v) for v in spec[k]] for k in ("position", "normal")}
            entry["radius"] = float(spec["radius"])
            entry["arm"], entry["eef"] = held[parent] if parent in held else (None, None)
            self.specs[label] = entry
            log.info(
                f"tracking button {label} at {np.round(entry['position'], 3).tolist()}"
                + (f" (in the {entry['arm']} hand)" if entry["arm"] else "")
            )

    def grasped(self, parent: str, arm: str, eef: np.ndarray) -> None:
        """The object ``parent`` was just grasped by ``arm`` whose eef pose is ``eef``: its buttons now move with it."""
        for label, entry in self.specs.items():
            if label[: -len("_button")] == parent and entry["arm"] is None:
                entry["arm"], entry["eef"] = arm, np.asarray(eef, dtype=np.float64)
                log.info(f"button {label}: its object is now in the {arm} hand; its pose follows the gripper")

    def current(self, eef_pose_base) -> dict:
        """gt_buttons for the next request: every tracked button at its pose now (``eef_pose_base(arm)`` -> 4x4)."""
        out = {}
        for label, entry in self.specs.items():
            position, normal = np.asarray(entry["position"]), np.asarray(entry["normal"])
            if entry["arm"] is not None:
                motion = eef_pose_base(entry["arm"]) @ np.linalg.inv(entry["eef"])
                position = motion[:3, :3] @ position + motion[:3, 3]
                normal = motion[:3, :3] @ normal
            out[label] = {"position": position.tolist(), "normal": normal.tolist(), "radius": entry["radius"]}
        return out


class KnowledgeSource:
    """What every source shares: the task goal translated into the planner's labels and predicates, and the
    robot's own record of what its hands hold (``sim.hands()``: tracked label -> arm)."""

    name = ""
    privileged = False
    category_level = False  # labels name categories ("candle") rather than instances ("candle_4")

    def __init__(self, sim, goal: list[dict]):
        self.sim = sim
        self.goal = list(goal)  # the whole task goal, what an agent knows from the task definition

    def translate(self, atoms: list[dict]) -> tuple[list[str], list[dict]]:
        """(request labels, TiPToP atoms) for goal atoms over the simulator's object names."""
        return self.sim.tiptop_goal(atoms, category_level=self.category_level)

    def label(self, tracked: str) -> str:
        """A tracked object's label at this source's level ('candle_4' stays, or becomes 'candle')."""
        return canonical_object_name(tracked)[0] if self.category_level else tracked

    def hands(self) -> tuple[list[str], list[str]]:
        """(labels in a hand the plan does not move, labels in the planned hand)."""
        arm = self.sim.arm
        held = self.sim.hands()
        return sorted(self.label(l) for l, a in held.items() if a != arm), sorted(
            self.label(l) for l, a in held.items() if a == arm
        )

    def describe(self, atoms: list[dict], request: dict, extras: dict, floor: bool = False) -> SceneKnowledge:
        """Everything this source knows for one request about the round's ``atoms``; ``floor`` says the round works
        at a container on the floor, so the planner's workspace must include it."""
        raise NotImplementedError

    def learned(self, response: dict) -> None:
        """The planner's report on a request (what it detected), kept for later rounds where it applies."""

    def picked(self, tracked: str, arm: str, eef: np.ndarray) -> None:
        """A plan just closed ``arm`` (eef pose ``eef``, base frame) on the tracked object; what moves with it now."""

    def localize(self, *bddl_names: str) -> dict:
        """Where the named task objects are, world frame: name -> {"center": (3,), "lo": (3,), "hi": (3,)} (an
        axis-aligned box). The episode's geometric checks (is the item over the basket, which support is it on,
        how far is the container) run on this and on nothing else."""
        raise NotImplementedError

    def report(self) -> dict:
        return {"source": self.name, "privileged": self.privileged}


class OracleKnowledge(KnowledgeSource):
    """The simulator's truth: per-instance labels, masks from geometry, the true pose of every button the task
    presses (sent in every round so the pick round can choose a grasp that presents it). Privileged."""

    name = "oracle"
    privileged = True

    def describe(self, atoms, request, extras, floor=False) -> SceneKnowledge:
        labels, tiptop_atoms = self.translate(atoms)
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
        # Every object a goal atom names has to be one the planner is given, and the planner is given exactly the
        # visible ones. Testing against `hidden` missed the case where an atom names something that never became a
        # label at all: the round then went out and the planner rejected it with "Goal predicate holding(
        # toy_figure_3) references unknown object 'toy_figure_3'. Known objects: ['table', 'toy_box_1']", which
        # cost a round and read as a planning failure rather than as the visibility failure it is (2026-09-13).
        needed = sorted({a for atom in tiptop_atoms for a in atom["args"] if a not in visible})
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
        return SceneKnowledge(
            labels=visible,
            atoms=tiptop_atoms,
            masks=masks[primary][keep],
            view_masks={name: view_masks[keep] for name, view_masks in masks.items() if name != primary},
            buttons=self.sim.button_hints(self.goal, category_level=False),
            held_labels=held,
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


class OnboardKnowledge(KnowledgeSource):
    """What the robot itself knows: the task's category names and goal, its gripper state, and the buttons it
    saw in earlier rounds carried through grasps. The planner's detector finds the objects; a ``<object>_button``
    label asks it to find the button on that object."""

    name = "onboard"
    category_level = True

    def __init__(self, sim, goal):
        super().__init__(sim, goal)
        self.buttons = ButtonTracker()
        self.seen = {}  # label -> world position of the object the planner last reported (a point, no extent yet)

    def localize(self, *bddl_names):
        """The planner's last reported position of each object, as a box of ``SEEN_HALF_EXTENT`` around it (the
        response carries positions, not hulls, so the extent is nominal); an object never reported is unknown."""
        out = {}
        for name in bddl_names:
            label = self.label(self.sim.tracked_label(name))
            if label not in self.seen:
                raise KeyError(f"{name} ({label}) has not been perceived yet; nothing to localize it from")
            c = np.asarray(self.seen[label], dtype=np.float64)
            out[name] = {"center": c, "lo": c - SEEN_HALF_EXTENT, "hi": c + SEEN_HALF_EXTENT}
        return out

    def describe(self, atoms, request, extras, floor=False) -> SceneKnowledge:
        labels, tiptop_atoms = self.translate(atoms)
        _, goal_atoms = self.translate(self.goal)
        # every button the task presses is asked for in every round: seen on the table before the pick, its pose is
        # carried through the grasp and sent as a prior a fresh detection may override
        button_labels = {a for atom in goal_atoms if atom["predicate"] == "pressed" for a in atom["args"]}
        held, in_hand = self.hands()
        tracked = self.buttons.current(self.sim.eef_pose_base)
        if tracked:
            log.info(f"button poses carried from earlier rounds: {tracked}")
        return SceneKnowledge(
            labels=sorted(set(labels) | button_labels),
            atoms=tiptop_atoms,
            buttons=tracked,
            held_labels=held,
            in_hand=in_hand,
            workspace=self.sim.workspace(floor),
        )

    def learned(self, response: dict) -> None:
        for label, info in (response.get("objects") or {}).items():
            if isinstance(info, dict) and info.get("position") is not None:
                pos_b = np.asarray(info["position"], dtype=np.float64)
                self.seen[label] = self.sim.base_to_world(pos_b)
        if response.get("buttons"):  # an object in hand moves with its gripper: remember the pose relative to it
            held = {self.label(l): (arm, self.sim.eef_pose_base(arm)) for l, arm in self.sim.hands().items()}
            self.buttons.update(response["buttons"], held)

    def picked(self, tracked: str, arm: str, eef: np.ndarray) -> None:
        self.buttons.grasped(self.label(tracked), arm, eef)


SOURCES = {cls.name: cls for cls in (OracleKnowledge, OnboardKnowledge)}


def make_knowledge(name: str, sim, goal: list[dict]) -> KnowledgeSource:
    if name not in SOURCES:
        raise ValueError(f"unknown knowledge source {name!r}; known: {sorted(SOURCES)}")
    source = SOURCES[name](sim, goal)
    if source.privileged:
        log.warning(f"PRIVILEGED knowledge source {name!r}: simulator masks and button poses go to the planner")
    return source
