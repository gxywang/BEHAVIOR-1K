"""The pseudo stack's providers (SPEC D18, §4.1): the one package that reads the simulator's truth. Every value leaves
tagged with its source, and reaches a skill only through guarded(), which counts it (pseudo) or refuses it
(competition). The skill bench names this package as its --providers; a perceived stack swaps it whole."""

from b1k.connector.goals import Clock, goal_panel
from b1k.connector.world import ProvenancePolicy, guarded
from b1k.map.collision import MapCollisionWorld
from b1k.perception.grasp_sensor import ProprioGraspSensor
from b1k.perception.verifier import PerceptionVerifier
from b1k.runtime.skillrun import Services
from omnigibson.tiptop.oracle.articulation import OracleArticulation
from omnigibson.tiptop.oracle.buttons import OracleButtons
from omnigibson.tiptop.oracle.geometry import OracleGeometry
from omnigibson.tiptop.oracle.goals import EpisodeScorer
from omnigibson.tiptop.oracle.joints import OracleJoints
from omnigibson.tiptop.oracle.mapbuild import pseudo_map
from omnigibson.tiptop.oracle.meshroom import MeshRoom
from omnigibson.tiptop.oracle.segmenter import OracleSegmenter
from omnigibson.tiptop.oracle.world import OracleWorld


def pseudo_services(ep, planner, routing: dict, collision: str = "map") -> tuple:
    """(Services, segmenter) for one Episode: the oracle WorldView, the pseudo map (one per scene) and the room from
    it (collision="mesh": today's physical meshes instead, the A/B), the oracle geometry, articulation and joint
    values, the proprio GraspSensor, and the GoalPanel from routing.yaml's goal_checker line with the scorer behind
    it. The segmenter is the bench Observer's."""
    sim, policy = ep.sim, ProvenancePolicy("pseudo")
    grasp, joints, map_ = ProprioGraspSensor(), OracleJoints(sim), pseudo_map(sim)
    # the task's own scope (task_scope() leaves the floors out, so none was ever found). One floor: the base plane
    # answers its atoms; several (bringing_in_wood's garden and corridor): none of them
    floors = [n for n in sim.env.task.object_scope if ep.is_floor(n)]
    room = MeshRoom(sim) if collision == "mesh" else MapCollisionWorld(map_, joints, lambda: sim.n_steps)
    svc = Services(
        world=guarded(OracleWorld(ep, grasp), policy, "world"),
        map=guarded(map_, policy, "map"),
        geometry=guarded(OracleGeometry(sim, map_), policy, "geometry"),
        articulation=guarded(OracleArticulation(sim), policy, "articulation"),
        joints=guarded(joints, policy, "joints"),
        collision=guarded(room, policy, "collision"),
        grasp=grasp,
        buttons=guarded(OracleButtons(sim), policy, "buttons"),
        goals=goal_panel(routing, "pseudo", scorer=EpisodeScorer(sim),
                         perception=PerceptionVerifier(floor=floors[0] if len(floors) == 1 else None)),
        planner=planner,
        percepts={},
        provenance=policy,
        clock=lambda: Clock(sim.n_steps, sim.max_steps, 0),
        epochs=lambda: (0, 0),
    )
    return svc, OracleSegmenter(sim)
