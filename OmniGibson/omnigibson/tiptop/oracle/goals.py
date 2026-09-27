"""The scorer behind `goal_checker: scorer` (SPEC §4.1, D19): the sim's BDDL evaluator, pseudo stack only."""


class EpisodeScorer:
    """ScorerChecker's Scorer: the task's own predicates (env.task._evaluate_predicate, through R1ProSim.holds), plus
    the two skill-level atoms BDDL has no predicate for, read off the robot's grasp: holding(obj, arm) and
    hand_empty(arm), and lifted(obj): clear of everything that stands, so a pick_up (holding and lifted) is not a
    grasp alone: the knife held on its board and the basket closed on at the floor both scored succeeded in week 2
    (VIDEO_FINDINGS 2, 11). Clear: the object's contact cluster, the robot left out, reaches no fixed_base piece (a
    floor, a wall, furniture) and nothing outside the scene's objects; what it carries (a basket's decorations or
    vegetables, W3 g2) touches it and is carried too. None when it cannot judge (an unknown predicate or object, an
    object with no rigid contact rows: a cloth). Oracle; the GoalPanel counts it.

    ``scope_only`` (the pseudo planner's host, WEEK4_PLAN 3.4): the task's evaluator alone, as Episode.goal_already_holds
    reads it. An ontop, nextto or inside of a name the task does not scope then falls to sim.holds, which raises
    KeyError, which is None here: legacy's False through the shim."""

    def __init__(self, sim, scope_only: bool = False):
        self.sim, self.scope_only = sim, scope_only

    def holds(self, fact):
        from omnigibson.controllers import IsGraspingState
        from omnigibson.utils.usd_utils import RigidContactAPI

        robot = self.sim.robot
        try:
            if fact.pred == "holding":
                obj, arm = fact.args
                return robot.is_grasping(arm, self.sim.scene_object(obj)) == IsGraspingState.TRUE
            if fact.pred == "lifted":  # ponytail: contact only, so a hover of a few mm over the support counts as
                #                        clear; add a height margin over support_of's top when a case shows one
                o = self.sim.scene_object(fact.args[0])
                return lifted(o, robot, RigidContactAPI)
            if fact.pred == "hand_empty":
                return robot.is_grasping(fact.args[0]) != IsGraspingState.TRUE
            if (not self.scope_only and fact.pred in ("ontop", "nextto", "inside")
                    and not set(fact.args) <= set(self.sim.env.task.object_scope)):
                from omnigibson.object_states import Inside, NextTo, OnTop  # a target the task does not name (the
                #    brisket's burner, the toilet a mousetrap goes beside): the state BDDL's predicate reads, directly
                obj, target = (self.sim.scene_object(n) for n in fact.args)
                state = {"ontop": OnTop, "nextto": NextTo, "inside": Inside}[fact.pred]
                return bool(obj.states[state].get_value(target))
            return self.sim.holds(fact.pred, *fact.args)
        except Exception:  # noqa: BLE001 - not a predicate or an object the evaluator knows: cannot judge
            return None


def lifted(o, robot, contacts) -> bool | None:
    """Whether ``o``'s contact cluster (``contacts``: RigidContactAPI), the robot left out, reaches nothing that stands:
    no fixed_base object and no prim outside the scene's objects (the ground plane). None for an object with no rigid
    contact rows (a cloth: RigidContactAPI rows are RIGID links only, so "touches nothing" would be vacuous)."""
    idx = o.scene.idx
    if not len(contacts.get_contact_row_indices(idx, [o])):
        return None
    cluster, frontier = {id(o)}, [o]
    while frontier:
        found = []
        for _, path in contacts.get_contact_pairs(idx, frontier, None, True):
            other = o.scene.object_registry("prim_path", "/".join(path.split("/")[:-1]), None)
            if other is robot or id(other) in cluster:
                continue
            if other is None or other.fixed_base:
                return False
            cluster.add(id(other))
            found.append(other)
        frontier = found
    return True
