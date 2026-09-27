"""The oracle ButtonLocator (SPEC §4.1, week 2 press): the sim's toggle button by pose, tagged oracle, on the bench's
Services."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from b1k.connector.skills import PressArgs, SkillCall, Status
from b1k.connector.types import ObjRef
from b1k.connector.world import ButtonSpec
from omnigibson.tiptop.host import skillbench
from omnigibson.tiptop.host.legacy_skills import LegacyBackend
from omnigibson.tiptop.oracle.buttons import STROKE, OracleButtons

ROOT = Path(__file__).resolve().parents[2]
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
    sim.env = SimpleNamespace(scene=SimpleNamespace(objects=[]), task=SimpleNamespace(object_scope={}))
    sim.scene_object = lambda n: None
    routing = {"goal_checker": "scorer", "goal_checkers_shadow": []}
    svc, _ = pseudo_services(SimpleNamespace(sim=sim), "planner", routing)
    assert svc.buttons.button(radio).value.radius == 0.02 and svc.provenance.take() == {"buttons.button": 1}


def test_the_legacy_baseline_presses_through_todays_toggled_on_round_with_the_calls_arm():
    """The press gate's legacy lane (SPEC §8): Episode.achieve on the toggled_on atom, one round from the case's stance,
    judged by the GoalPanel after it ran like every legacy call."""
    ep, seen = SimpleNamespace(sim=SimpleNamespace(n_steps=0), records=[]), {}

    def achieve(atoms, arm="left"):
        seen.update(atoms=atoms, arm=arm)
        ep.sim.n_steps += 30
        return True

    ep.achieve = achieve
    goals = SimpleNamespace(judge=lambda goal, obs, who: (SimpleNamespace(value=None), {}), primary="scorer")
    lb = LegacyBackend(ep, lambda: None, single_round=True)
    call = SkillCall("press", PressArgs(ObjRef("switch.n.01_1", "switch", True), want_on=False), arm="left")
    assert lb.supports(call)
    with pytest.raises(StopIteration) as done:
        next(lb.run(call, SimpleNamespace(goals=goals), None))
    r = done.value.value
    assert seen == {"atoms": [{"predicate": "toggled_on", "args": ["switch.n.01_1"]}], "arm": "left"}
    assert (r.status, r.steps, r.requires_sim_clock, r.world_updates) == (Status.SUCCEEDED, 30, True, ())


def test_the_press_cases_load_the_switches_off_the_stove_on_and_the_radio_from_the_humans_hand():
    cases = {c["id"]: c for c in skillbench.load_cases(ROOT / "tiptop/b1k/skills/bench/press.yaml")}
    switches = [c for c in cases.values() if c["task"] == "turning_out_all_lights_before_sleep"]
    assert len(switches) == 3
    assert all(c["call"].args.want_on is False and c["call"].args.target.fixed for c in switches), "the lights go off"
    assert {c["call"].args.target.id for c in switches} == {"switch.n.01_1", "switch.n.01_2", "switch.n.01_3"}
    assert all(c["instance"] == 301 and c["setup"]["torso"] == [1.025, -1.45, -0.47, 0.0] for c in switches)
    stove = cases["press_cook_bacon_stove_knob"]["call"]
    assert isinstance(stove.args, PressArgs) and stove.args.want_on is True and stove.arm == "left"
    held = cases["press_turning_on_radio_held"]  # the demo situation: the right hand holds the radio, the bench picks the free left hand
    assert held["mode"] == "train" and held["setup"]["held"] == {"right": "radio_89"} and held["call"].arm is None
    assert Path(held["demo"]["snapshot"]).is_file() and held["call"].args.want_on is True, "the state is checked here"
    assert held["call"].variant is None
    raw = {c["id"]: c for c in yaml.safe_load((ROOT / "tiptop/b1k/skills/bench/press.yaml").read_text())}
    ab = raw["press_turning_on_radio_fingertip"]
    assert "press_turning_on_radio_fingertip" not in cases and "not implemented" in ab["skip"], \
        "the server refuses hand != closed: the case could only return UNSUPPORTED in 0 steps, so it is off the bench"
    assert ab["call"]["variant"] == "press.fingertip"
