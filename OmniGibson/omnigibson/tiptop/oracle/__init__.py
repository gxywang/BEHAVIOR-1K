"""The pseudo stack's providers (SPEC D18, §4.1): the one package that reads the simulator's truth. Every value leaves
tagged with its source, and reaches a skill only through guarded(), which counts it (pseudo) or refuses it
(competition). The skill bench names this package as its --providers; a perceived stack swaps it whole."""

from b1k.connector.goals import Clock, goal_panel
from b1k.connector.world import ProvenancePolicy, guarded
from b1k.map.collision import MapCollisionWorld
from b1k.perception.grasp_sensor import ProprioGraspSensor
from b1k.perception.verifier import PerceptionVerifier
from b1k.runtime.skillrun import Services
from omnigibson.tiptop.host.teleport_nav import MOVE_TO_STEPS
from omnigibson.tiptop.oracle.articulation import OracleArticulation
from omnigibson.tiptop.oracle.buttons import OracleButtons
from omnigibson.tiptop.oracle.geometry import OracleGeometry
from omnigibson.tiptop.oracle.goals import EpisodeScorer
from omnigibson.tiptop.oracle.joints import OracleJoints
from omnigibson.tiptop.oracle.mapbuild import pseudo_map
from omnigibson.tiptop.oracle.meshroom import MeshRoom
from omnigibson.tiptop.oracle.segmenter import OracleSegmenter
from omnigibson.tiptop.oracle.taskinfo import task_info  # noqa: F401 - a host loads this package as --providers
from omnigibson.tiptop.oracle.world import OracleWorld


def pseudo_services(ep, planner, routing: dict, collision: str = "map", hands: str = "sensor",
                    scorer_scope_only: bool = False, shadows=None) -> tuple:
    """(Services, segmenter) for one Episode: the oracle WorldView, the pseudo map (one per scene) and the room from
    it (collision="mesh": today's physical meshes instead, the A/B), the oracle geometry, articulation and joint
    values, the proprio GraspSensor, and the GoalPanel from routing.yaml's goal_checker line with the scorer behind
    it. The segmenter is the bench Observer's. The clock's shadow is the teleports so far at the human move-to cost
    (D20). ``hands``, ``scorer_scope_only``: the episode host's OracleWorld and EpisodeScorer (WEEK4_PLAN 3.4);
    ``shadows``, when given, replaces routing's goal_checkers_shadow list."""
    sim, policy = ep.sim, ProvenancePolicy("pseudo")
    grasp, joints, map_ = ProprioGraspSensor(), OracleJoints(sim), pseudo_map(sim)
    # the task's own scope (task_scope() leaves the floors out, so none was ever found). One floor: the base plane
    # answers its atoms; several (bringing_in_wood's garden and corridor): none of them
    floors = [n for n in sim.env.task.object_scope if ep.is_floor(n)]
    room = MeshRoom(sim) if collision == "mesh" else MapCollisionWorld(map_, joints, lambda: sim.n_steps)
    if shadows is not None:
        routing = dict(routing, goal_checkers_shadow=list(shadows))
    svc = Services(
        world=guarded(OracleWorld(ep, grasp, joints=joints, hands=hands), policy, "world"),
        map=guarded(map_, policy, "map"),
        geometry=guarded(OracleGeometry(sim, map_), policy, "geometry"),
        articulation=guarded(OracleArticulation(sim), policy, "articulation"),
        joints=guarded(joints, policy, "joints"),
        collision=guarded(room, policy, "collision"),
        grasp=grasp,
        buttons=guarded(OracleButtons(sim), policy, "buttons"),
        goals=goal_panel(routing, "pseudo", scorer=EpisodeScorer(sim, scope_only=scorer_scope_only),
                         perception=PerceptionVerifier(floor=floors[0] if len(floors) == 1 else None)),
        planner=planner,
        percepts={},
        provenance=policy,
        clock=lambda: Clock(sim.n_steps, sim.max_steps, MOVE_TO_STEPS * sim.teleports),
        epochs=lambda: (0, 0),
    )
    return svc, OracleSegmenter(sim)


def refresh_hands(world, after=None) -> list:
    """OracleWorld._refresh_hands through the guarded world a host holds (Services.world): the popped labels.
    ``after``: the step the native run started at (a look from before it is unknown)."""
    return world._refresh_hands(after=after)
