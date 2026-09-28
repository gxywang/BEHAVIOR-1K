"""The oracle WorldView (SPEC §8 Track A2): what the week-2 preconditions read, over today's Episode queries."""

from b1k.bridge.articulation import is_open
from b1k.bridge.protocol import bddl_category
from b1k.connector.types import AABB, Belief, ObjRef, Pose2
from b1k.connector.world import derived_source
from omnigibson.tiptop.articulation import openable_joints

ARMS = ("left", "right")
HANDS = ("sensor", "episode")


def ref(sim, name: str) -> ObjRef:
    """The planner's name for a task object: its BDDL instance, category head, and whether it is fixed furniture."""
    return ObjRef(name, bddl_category(name), bool(getattr(sim.scene_object(name), "fixed_base", False)))


class OracleWorld:
    """held / holding come from the GraspSensor plus the skills' WorldUpdates, never from localization or the grasp
    assist: a hand the sensor says is empty holds nothing, and a closed hand no skill named holds something unknown.
    Everything else is a thin wrapper over the Episode's own query of the same name (oracle: the simulator's truth or
    the oracle knowledge source's localization). Q1 reuses it.

    ``hands="episode"`` (the pseudo planner's host, WEEK4_PLAN 3.4): the hands are the robot's own record instead,
    ``sim.hands()`` (run.note_hands writes it; Episode.holding and held_names read it), so the Runner sees through
    the shim what it sees today. A skill's held / released update writes that record; the GraspSensor's reading is
    only compared with it (``disagreements``), never obeyed. ``joints``: the JointStateEstimator (OracleJoints) the
    live joint value of is_open and open_fraction comes from (rule 6, SPEC 5.5); the limits and the closed frame
    stay openable_joints'. None: the position openable_joints read, the old answer."""

    def __init__(self, ep, grasp, joints=None, hands: str = "sensor"):
        if hands not in HANDS:
            raise ValueError(f"hands={hands!r} is not one of {HANDS}")
        self.ep, self.sim, self.grasp, self.joints, self.mode = ep, ep.sim, grasp, joints, hands
        self.obs, self.hands = None, {a: set() for a in ARMS}  # hands: the sensor mode's ledger
        self.disagreements = 0  # episode mode: readings of the GraspSensor that contradicted the record

    @staticmethod
    def _b(value, source="oracle") -> Belief:
        return Belief(value, source, 0)

    def ref(self, name: str) -> ObjRef:
        return ref(self.sim, name)

    # -- written by the Runtime ----------------------------------------------------------------------------------------
    def tick(self, obs) -> None:
        self.obs = obs

    def apply(self, u) -> None:
        if u.arm is None:
            return
        if self.mode == "episode":
            return self._apply_record(u)
        if u.kind == "held" and u.obj is not None:
            self.hands[u.arm].add(u.obj)
        elif u.kind == "released":
            self.hands[u.arm] -= {u.obj} if u.obj is not None else self.hands[u.arm]

    def _apply_record(self, u) -> None:
        """The robot's own record, by tracked label, as note_hands writes it: idempotent. A release names the arm
        that let go: a label the record has in the OTHER hand stays (a hand-to-hand move is released(old arm) and
        held(new arm), in either order), as legacy's release pops its own arm's entries only (bench.py release)."""
        record = self.sim.held_objects
        if u.kind == "held" and u.obj is not None:
            record[self.sim.tracked_label(u.obj.id)] = u.arm
        elif u.kind == "released" and u.obj is not None:
            label = self.sim.tracked_label(u.obj.id)
            if record.get(label) == u.arm:
                record.pop(label)
        elif u.kind == "released":
            for label in [label for label, holder in record.items() if holder == u.arm]:
                record.pop(label)

    # -- the hands ----------------------------------------------------------------------------------------------------
    def _record(self, arm) -> tuple:
        """Episode mode: what the record says ``arm`` holds, as BDDL names, in the record's insertion order; a label
        with no BDDL name (an object a transition removed) is left out, as Episode.held_names leaves it out."""
        sim = self.sim
        names = getattr(sim, "bddl_names", {})
        return tuple(
            self.ref(names[label]) for label, holder in sim.hands().items() if holder == arm and label in names
        )

    def held(self, arm) -> Belief:
        if self.mode == "episode":
            record = self._record(arm)
            sensed = None if self.grasp is None else self.grasp.held(arm, self.obs).value
            if sensed is not None and sensed != bool(record):  # compared, counted, never obeyed
                self.disagreements += 1
            return self._b(record)
        sensed = self.grasp.held(arm, self.obs).value
        if sensed is False:
            self.hands[arm].clear()
            return self._b((), "proprio")
        # held, or unknown (the fingers still move, or the sensor's window is not full yet at a trial's first step):
        # what the skills and the setup recorded stands, as an unknown reading never opens the latch's closed hand
        return self._b(tuple(sorted(self.hands[arm], key=lambda o: o.id)) or None, "belief")

    def holding(self, o) -> Belief:
        if self.mode == "episode":  # Episode.holding: tracked_label(bddl) in sim.hands()
            label = self.sim.tracked_label(o.id)
            return self._b(tuple(a for a in ARMS if self.sim.hands().get(label) == a))
        return self._b(tuple(a for a in ARMS if o in (self.held(a).value or ())), "belief")

    def _refresh_hands(self, after=None) -> list:
        """Pop the record's entries that localization says have left the hand: run.note_hands with no atoms
        (run.py:812-817), which only pops and never steps; the episode host calls it after a native pick, place,
        release, hold or press (WEEK4_PLAN 3.3 HandRefresh), through oracle.refresh_hands. ``after``: the step the
        run started at, as legacy passes its plan's start step: a remembered look from before it (the onboard source
        keeps its last) is unknown, never evidence that the hand let go. The popped labels. RuntimeError if the sim
        stepped. Private: not a WorldView member, so neither the planner's world() nor a skill reaches it."""
        from types import SimpleNamespace

        from omnigibson.tiptop.run import note_hands

        sim = self.sim
        before, n0 = list(sim.hands()), sim.n_steps
        note_hands(sim, [], SimpleNamespace(close_eef=None, gripper=None), self.ep.knowledge, after=after)
        if sim.n_steps != n0:
            raise RuntimeError(f"refresh_hands stepped the sim: {n0} -> {sim.n_steps}")
        return [label for label in before if label not in sim.hands()]

    # -- thin wrappers over the Episode -------------------------------------------------------------------------------
    def objects(self) -> list:
        return [self.ref(n) for n in self.sim.task_scope()]

    def box(self, o) -> Belief:
        b = self.ep.boxes(o.id).get(o.id)
        return self._b(None if b is None else AABB(tuple(map(float, b["lo"])), tuple(map(float, b["hi"]))))

    def _joints(self, o, joint) -> tuple:
        """(the object's openable joints, the source of the answer): openable_joints' frames and limits, the position
        the JointStateEstimator's when one is given. The limits and the closed frame are openable_joints' (the
        simulator's, oracle data) whatever the estimator is, so the answer is as privileged as its most privileged
        input (derived_source): oracle, even over a proprio or depth estimator."""
        js = [j for j in openable_joints(self.sim.scene_object(o.id)) if joint is None or j["name"] == joint]
        if self.joints is None:
            return js, "oracle"
        read = [self.joints.value(o, j["name"]) for j in js]
        return [dict(j, position=p.value) for j, p in zip(js, read)], derived_source(
            "oracle", *(p.source for p in read)
        )

    def is_open(self, o, joint=None) -> Belief:
        js, source = self._joints(o, joint)
        if any(j["position"] is None for j in js):  # an estimator that cannot see the joint: unknown, not an error
            return self._b(None, source)
        shut = [not is_open(j["lower"], j["upper"], j["position"], closed=j["closed"]) for j in js]
        return self._b(not all(shut) if js else None, source)

    def open_fraction(self, o, joint=None) -> Belief:
        js, source = self._joints(o, joint)
        if any(j["position"] is None for j in js):
            return self._b(None, source)
        return self._b(
            max(abs(j["position"] - j["closed"]) / (j["upper"] - j["lower"]) for j in js) if js else None, source
        )

    def switched_on(self, o) -> Belief:
        return self._b(self.ep.switched_on(o.id))

    def support_of(self, o) -> Belief:
        s = self.ep.support_of(o.id)
        return self._b(None if s is None else self.ref(s))

    def enclosed_by(self, o) -> Belief:
        """The shut containers it is inside, outermost first, up its support chain (strategies.reach_into)."""
        name, support, shut = o.id, self.ep.support_of(o.id), []
        for _ in range(3):
            if not support or self.ep.is_floor(support):
                break
            if self.ep.is_shut(support) and self.ep.goal_already_holds("inside", name, support):
                shut.append(support)
            name, support = support, self.ep.support_of(support)
        return self._b(tuple(self.ref(s) for s in reversed(shut)))

    def distance(self, a, b) -> Belief:
        try:
            return self._b(self.ep.distance(a.id, b.id))
        except KeyError:  # never perceived
            return self._b(None)

    def edge_gap(self, item, support) -> Belief:
        return self._b(self.ep.edge_gap(item.id, None if support is None else support.id))

    def fixture_for(self, ability, near=None) -> Belief:
        n = self.ep.fixture_for(ability, None if near is None else near.id)
        return self._b(None if n is None else self.ref(n))

    def appeared(self) -> list:
        return [self.ref(n) for n in self.ep.after_transition()]

    def base_pose(self) -> Belief:
        import omnigibson.utils.transform_utils as T

        pos, quat = self.sim.robot.get_position_orientation()
        return self._b(Pose2(float(pos[0]), float(pos[1]), float(T.quat2euler(quat)[2]), float(pos[2])))
