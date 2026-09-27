"""The scorer behind `goal_checker: scorer` (SPEC §4.1, D19): the sim's BDDL evaluator, pseudo stack only."""


class EpisodeScorer:
    """ScorerChecker's Scorer: the task's own predicates (env.task._evaluate_predicate, through R1ProSim.holds), plus
    the two skill-level atoms BDDL has no predicate for, read off the robot's grasp: holding(obj, arm) and
    hand_empty(arm), and lifted(obj): the object touches nothing but the robot, so a pick_up (holding and lifted) is
    not a grasp alone: the knife held on its board and the basket closed on at the floor both scored succeeded in
    week 2 (VIDEO_FINDINGS 2, 11). None when it cannot judge (an unknown predicate or object). Oracle; the GoalPanel
    counts it."""

    def __init__(self, sim):
        self.sim = sim

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
                return not RigidContactAPI.is_in_contact(scene_idx=o.scene.idx, query_set=[o], with_set=None,
                                                         ignore_set=[robot], current_only=True)
            if fact.pred == "hand_empty":
                return robot.is_grasping(fact.args[0]) != IsGraspingState.TRUE
            if fact.pred in ("ontop", "nextto", "inside") and not set(fact.args) <= set(self.sim.env.task.object_scope):
                from omnigibson.object_states import Inside, NextTo, OnTop  # a target the task does not name (the
                #    brisket's burner, the toilet a mousetrap goes beside): the state BDDL's predicate reads, directly
                obj, target = (self.sim.scene_object(n) for n in fact.args)
                state = {"ontop": OnTop, "nextto": NextTo, "inside": Inside}[fact.pred]
                return bool(obj.states[state].get_value(target))
            return self.sim.holds(fact.pred, *fact.args)
        except Exception:  # noqa: BLE001 - not a predicate or an object the evaluator knows: cannot judge
            return None
