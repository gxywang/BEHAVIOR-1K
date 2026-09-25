"""The demo-case source (no simulator): dead reckoning, the human's arm and grip, name resolution, and the SkillCall
dicts the bench reads back through the connector codec. The dataset-backed checks skip when the demos are absent."""

import math

import numpy as np
import pytest
from b1k.connector import codec, skills

from omnigibson.tiptop.host import demo_cases as D

OBJS = {
    "radio_89": ("radio_receiver.n.01_1", "radio", False),
    "coffee_table_koagbh_0": ("table.n.02_1", "coffee_table", False),
    "fridge_dszchb_0": ("electric_refrigerator.n.01_1", "fridge", True),
    "apple_1": ("apple.n.01_1", "apple", False),
    "apple_2": ("apple.n.01_2", "apple", False),
    "top_cabinet_a_0": (None, "top_cabinet", True),
    "top_cabinet_b_0": (None, "top_cabinet", True),
}


@pytest.fixture
def fake_task(monkeypatch):
    monkeypatch.setattr(D, "task_objects", lambda task: OBJS)


def test_dead_reckon_straight_and_turn():
    state = np.zeros((31, 61))
    state[:, 0] = 0.3  # 0.3 m/s forward for 1 s, facing +y
    pose = D.dead_reckon(state, 1.0, 2.0, math.pi / 2)
    assert np.allclose(pose[-1], [1.0, 2.3, math.pi / 2], atol=1e-9)
    state[:, 0], state[:, 2] = 0.0, 0.5  # turn in place
    assert np.isclose(D.dead_reckon(state, 0, 0, 0)[-1, 2], 0.5)


def test_closing_arm_and_held():
    action = np.ones((100, 23))
    action[60:, 22] = -1.0  # right closes at 60
    action[80:, 14] = -1.0  # left later
    assert D.closing_arm(action, 0, 100) == "right"
    assert D.closing_arm(action, 0, 50) is None
    assert D.closing_arm(action, 70, 100) == "left"  # right was already closed: holding, not closing
    assert D.held_at(action, 61) == {"left": False, "right": True}


def test_resolve(fake_task):
    assert D.resolve("t", "radio_89") == "radio_89"
    assert D.resolve("t", "electric_refrigerator") == "fridge_dszchb_0"  # the BDDL synset head
    assert D.resolve("t", "fridge") == "fridge_dszchb_0"  # the category
    assert D.resolve("t", "apple") is None  # two apples: ambiguous
    assert D.resolve("t", "top_cabinet") is None


@pytest.mark.parametrize(
    "human, objs, skill, facts",
    [
        ("pick up from", ["radio_89", "coffee_table_koagbh_0"], "pick_up", [("holding", ("radio_receiver.n.01_1",), True)]),
        ("place on", ["radio_89", "coffee_table_koagbh_0"], "place",
         [("ontop", ("radio_receiver.n.01_1", "table.n.02_1"), True)]),
        ("open door", ["fridge_dszchb_0"], "open", [("open", ("electric_refrigerator.n.01_1",), True)]),
        ("close door", ["fridge_dszchb_0"], "close", [("open", ("electric_refrigerator.n.01_1",), False)]),
        ("turn off switch", ["radio_89"], "press", [("toggled_on", ("radio_receiver.n.01_1",), False)]),
    ],
)
def test_skill_call_decodes(fake_task, human, objs, skill, facts):
    call, atoms = D.skill_call("t", human, objs)
    decoded = codec.from_dict(call)
    assert isinstance(decoded, skills.SkillCall) and decoded.skill == skill
    ref = getattr(decoded.args, "obj", None) or decoded.args.target
    assert ref.id == OBJS[objs[0]][0] and ref.fixed == OBJS[objs[0]][2]
    if skill == "place":
        assert decoded.args.relations[0].rel is skills.Rel.ON
    assert [(f.pred, f.args, f.value) for f in map(codec.from_dict, atoms)] == facts


needs_demos = pytest.mark.skipif(not (D.DATASET / "manipulation_ranges.jsonl").exists(), reason="no B1K demos here")


@needs_demos
def test_episode_zero_is_training_instance_one():
    ep = D.episodes().loc[0]
    assert ep["task_instance_id"] == 1 == ep["raw_episode_id"] % 10000 // 10
    state, action = D.episode_arrays(0, stop=300)
    assert state.shape == (300, 61) and action.shape == (300, 23)
    assert np.allclose(state[0, D.PROPRIO_SLICES["trunk_qpos"]], [1.025, -1.45, -0.47, 0.0], atol=0.01)  # reset pose
