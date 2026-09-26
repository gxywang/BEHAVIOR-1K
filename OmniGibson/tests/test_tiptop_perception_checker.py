"""routing.yaml names perception in goal_checkers_shadow, and goal_panel() refuses a checker the host does not provide,
so the pseudo stack registers the PerceptionVerifier (SPEC §4.1, D19): every bench trial logs it beside the scorer."""

from types import SimpleNamespace

from b1k.perception.verifier import PerceptionVerifier
from b1k.skills.registry import load_routing
from omnigibson.tiptop.oracle import pseudo_services


def test_the_pseudo_stack_registers_the_perception_checker_routing_names():
    sim = SimpleNamespace(n_steps=0, max_steps=None, scene_object=lambda n: None, robot=None, task_scope=lambda: {},
                          env=SimpleNamespace(scene=SimpleNamespace(objects=[])))
    routing = load_routing()
    assert routing["goal_checkers_shadow"] == ["perception"]
    svc, _ = pseudo_services(SimpleNamespace(sim=sim), "planner", routing)
    assert svc.goals.primary == "scorer" and isinstance(svc.goals.checkers["perception"], PerceptionVerifier)
