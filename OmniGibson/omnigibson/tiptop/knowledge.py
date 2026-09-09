"""What the client tells the planner about the scene beyond the RGB-D image, and where that knowledge comes from.

The planner needs the names of the objects the goal talks about and the goal itself; everything else it works out
from the image. ``OnboardKnowledge`` sends just that, the way an agent in the challenge would: category names, the
goal atoms, and what its own hands hold (its gripper state). A toggle button is named for the detector to find on
its object, and a button seen in an earlier round is carried through a grasp by the arm's kinematics
(``ButtonTracker``). ``OracleKnowledge`` reads the simulator instead: per-instance labels, masks from the objects'
geometry (or Isaac's instance segmentation), and the true pose of every toggle button the task presses. That is
privileged information the challenge forbids at evaluation time; it exists so planning and execution can be
developed and measured without a detector. It lives in this module and in ``R1ProSim``'s ``oracle_masks`` /
``button_hints`` / ``toggled``; the benchmark's ``Episode`` (bench.py) reads the simulator too, for the decisions a
strategy makes between rounds, and says so in its own docstring. A run that uses this source says so (``report``).
Both sources produce
the same ``SceneKnowledge``; the rest of the pipeline never asks which one it got. The one thing a source tells the
executor is ``press_done``: the oracle knows the instant a switch flips, the onboard source has no such signal.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from omnigibson.tiptop.protocol import attach_knowledge, canonical_object_name

log = logging.getLogger(__name__)

# The planner works inside a base-frame box (its configured crop is the tabletop ahead of the robot, z from 0.25 m).
# For a container standing on the floor the box must reach the floor; the client asks for this one then.
FLOOR_WORKSPACE = [[0.05, -0.80, -0.05], [1.30, 0.80, 1.60]]


class GoalNotVisible(ValueError):
    """A goal object has no pixels in the capture: the frame the planner would work from does not show it."""


@dataclass
class SceneKnowledge:
    """One request's worth of knowledge in the planner's terms: labels, TiPToP predicates, the robot base frame."""

    labels: list
    atoms: list
    masks: np.ndarray | None = None  # (N, H, W) bool aligned with labels; None: the planner detects
    buttons: dict = field(default_factory=dict)  # label -> {position, normal, radius}
    held_labels: list = field(default_factory=list)  # in a hand the plan does not move
    in_hand: list = field(default_factory=list)  # in the planned hand: the plan starts holding them
    workspace: list | None = None  # base-frame box for this request, None: the planner's default

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
        )

    def summary(self) -> dict:
        """What went into the request, for capture.json and the log (masks by pixel count)."""
        return {
            "labels": list(self.labels),
            "atoms": list(self.atoms),
            "mask_pixels": {label: int(m.sum()) for label, m in zip(self.labels, self.masks)}
            if self.masks is not None
            else None,
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

    def press_done(self, bddl_targets: list[str]) -> Callable[[], bool] | None:
        """A signal that the press of these switches has landed, for the executor to end the push on; None when the
        source has no such signal and the push runs to its planned depth."""
        return None

    def report(self) -> dict:
        return {"source": self.name, "privileged": self.privileged}


class OracleKnowledge(KnowledgeSource):
    """The simulator's truth: per-instance labels, masks from geometry, the true pose of every button the task
    presses (sent in every round so the pick round can choose a grasp that presents it). Privileged."""

    name = "oracle"
    privileged = True

    def describe(self, atoms, request, extras, floor=False) -> SceneKnowledge:
        labels, tiptop_atoms = self.translate(atoms)
        masks = self.sim.oracle_masks(request, extras, labels)
        counts = {label: int(m.sum()) for label, m in zip(labels, masks)}
        visible = [label for label in labels if counts[label]]
        hidden = [label for label in labels if not counts[label]]
        needed = sorted({a for atom in tiptop_atoms for a in atom["args"] if a in hidden})
        if needed:
            raise GoalNotVisible(f"goal objects {needed} are not visible in the capture (empty masks)")
        log.info(
            f"oracle masks: pixels per label { {label: counts[label] for label in visible} }; out of view: {hidden or 'none'}"
        )
        held, in_hand = self.hands()
        return SceneKnowledge(
            labels=visible,
            atoms=tiptop_atoms,
            masks=masks[[labels.index(label) for label in visible]],
            buttons=self.sim.button_hints(self.goal, category_level=False),
            held_labels=held,
            in_hand=in_hand,
            workspace=FLOOR_WORKSPACE if floor else None,
        )

    def press_done(self, bddl_targets):
        return lambda: all(self.sim.toggled(name) for name in bddl_targets)


class OnboardKnowledge(KnowledgeSource):
    """What the robot itself knows: the task's category names and goal, its gripper state, and the buttons it
    saw in earlier rounds carried through grasps. The planner's detector finds the objects; a ``<object>_button``
    label asks it to find the button on that object."""

    name = "onboard"
    category_level = True

    def __init__(self, sim, goal):
        super().__init__(sim, goal)
        self.buttons = ButtonTracker()

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
            workspace=FLOOR_WORKSPACE if floor else None,
        )

    def learned(self, response: dict) -> None:
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
