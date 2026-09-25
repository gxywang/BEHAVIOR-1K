"""The scorer behind `goal_checker: scorer` (SPEC §4.1, D19): the sim's BDDL evaluator, pseudo stack only."""


class EpisodeScorer:
    """ScorerChecker's Scorer: the task's own predicates (env.task._evaluate_predicate, through R1ProSim.holds), plus
    the two skill-level atoms BDDL has no predicate for, read off the robot's grasp: holding(obj, arm) and
    hand_empty(arm). None when it cannot judge (an unknown predicate or object). Oracle; the GoalPanel counts it."""

    def __init__(self, sim):
        self.sim = sim

    def holds(self, fact):
        from omnigibson.controllers import IsGraspingState

        robot = self.sim.robot
        try:
            if fact.pred == "holding":
                obj, arm = fact.args
                return robot.is_grasping(arm, self.sim.scene_object(obj)) == IsGraspingState.TRUE
            if fact.pred == "hand_empty":
                return robot.is_grasping(fact.args[0]) != IsGraspingState.TRUE
            return self.sim.holds(fact.pred, *fact.args)
        except Exception:  # noqa: BLE001 - not a predicate or an object the evaluator knows: cannot judge
            return None
