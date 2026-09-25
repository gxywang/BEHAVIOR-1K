"""The oracle WorldView (SPEC §8 Track A2): what the week-2 preconditions read, over today's Episode queries."""

from b1k.bridge.articulation import is_open
from b1k.bridge.protocol import bddl_category
from b1k.connector.types import AABB, Belief, ObjRef, Pose2
from omnigibson.tiptop.articulation import openable_joints

ARMS = ("left", "right")


class OracleWorld:
    """held / holding come from the GraspSensor plus the skills' WorldUpdates, never from localization or the grasp
    assist: a hand the sensor says is empty holds nothing, and a closed hand no skill named holds something unknown.
    Everything else is a thin wrapper over the Episode's own query of the same name (oracle: the simulator's truth or
    the oracle knowledge source's localization). Q1 reuses it."""

    def __init__(self, ep, grasp):
        self.ep, self.sim, self.grasp = ep, ep.sim, grasp
        self.obs, self.hands = None, {a: set() for a in ARMS}

    @staticmethod
    def _b(value, source="oracle") -> Belief:
        return Belief(value, source, 0)

    def ref(self, name: str) -> ObjRef:
        return ObjRef(name, bddl_category(name), bool(getattr(self.sim.scene_object(name), "fixed_base", False)))

    # -- written by the Runtime ----------------------------------------------------------------------------------------
    def tick(self, obs) -> None:
        self.obs = obs

    def apply(self, u) -> None:
        if u.arm is None:
            return
        if u.kind == "held" and u.obj is not None:
            self.hands[u.arm].add(u.obj)
        elif u.kind == "released":
            self.hands[u.arm] -= {u.obj} if u.obj is not None else self.hands[u.arm]

    # -- the hands ----------------------------------------------------------------------------------------------------
    def held(self, arm) -> Belief:
        sensed = self.grasp.held(arm, self.obs).value
        if sensed is None:
            return self._b(None, "proprio")
        if not sensed:
            self.hands[arm].clear()
            return self._b((), "proprio")
        return self._b(tuple(sorted(self.hands[arm], key=lambda o: o.id)) or None, "belief")

    def holding(self, o) -> Belief:
        return self._b(tuple(a for a in ARMS if o in (self.held(a).value or ())), "belief")

    # -- thin wrappers over the Episode -------------------------------------------------------------------------------
    def objects(self) -> list:
        return [self.ref(n) for n in self.sim.task_scope()]

    def box(self, o) -> Belief:
        b = self.ep.boxes(o.id).get(o.id)
        return self._b(None if b is None else AABB(tuple(map(float, b["lo"])), tuple(map(float, b["hi"]))))

    def _joints(self, o, joint) -> list:
        return [j for j in openable_joints(self.sim.scene_object(o.id)) if joint is None or j["name"] == joint]

    def is_open(self, o, joint=None) -> Belief:
        js = self._joints(o, joint)
        shut = [not is_open(j["lower"], j["upper"], j["position"], closed=j["closed"]) for j in js]
        return self._b(not all(shut) if js else None)

    def open_fraction(self, o, joint=None) -> Belief:
        js = self._joints(o, joint)
        return self._b(max(abs(j["position"] - j["closed"]) / (j["upper"] - j["lower"]) for j in js) if js else None)

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
        return self._b(Pose2(float(pos[0]), float(pos[1]), float(T.quat2euler(quat)[2])))
