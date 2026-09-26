"""The oracle ButtonLocator (SPEC §4.1, week 2 press): the sim's toggle button by pose, tagged oracle, on the bench's
Services."""

from types import SimpleNamespace

import numpy as np

from b1k.connector.types import ObjRef
from b1k.connector.world import ButtonSpec
from omnigibson.tiptop.oracle.buttons import STROKE, OracleButtons

radio = ObjRef("radio_receiver.n.01_1", "radio_receiver")


def _sim(**buttons):
    def button_world(bddl):
        if bddl not in buttons:
            raise ValueError(f"{bddl} has no toggle button (no ToggledOn state)")
        return buttons[bddl]

    return SimpleNamespace(n_steps=7, button_world=button_world)


def test_the_oracle_locator_gives_the_buttons_face_normal_and_radius_tagged_oracle():
    sim = _sim(**{radio.id: (np.array([2.0, 1.0, 0.9]), np.array([0.0, -1.0, 0.0]), 0.046)})
    got = OracleButtons(sim).button(radio)
    assert got.source == "oracle" and got.step == 7
    assert got.value == ButtonSpec((2.0, 1.0, 0.9), (0.0, -1.0, 0.0), STROKE, radio, 0.046)
    assert all(type(v) is float for v in (*got.value.point, *got.value.normal, got.value.radius))
    assert OracleButtons(sim).button(ObjRef("apple.n.01_1", "apple")).value is None, "nothing to press: NO_FEATURE"


def test_the_pseudo_stack_serves_the_oracle_buttons_counted(monkeypatch):
    from omnigibson.tiptop.oracle import pseudo_services

    sim = _sim(**{radio.id: (np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.02)})
    sim.max_steps, sim.robot, sim.task_scope = None, None, lambda: {}
    sim.env, sim.scene_object = SimpleNamespace(scene=SimpleNamespace(objects=[])), lambda n: None
    svc, _ = pseudo_services(SimpleNamespace(sim=sim), "planner", {"goal_checker": "scorer", "goal_checkers_shadow": []})
    assert svc.buttons.button(radio).value.radius == 0.02 and svc.provenance.take() == {"buttons.button": 1}
