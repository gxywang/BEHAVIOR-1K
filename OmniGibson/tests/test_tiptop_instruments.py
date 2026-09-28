"""W4-A2: the bench's week-4 flags default to today's run and install nothing; the StepLedger attributes every env
step to its innermost owner (a nested ep.stand_for inside ep.pick is ep.stand_for's), records each top-level ep.*
call's whole delta, counts the write-calls per owner, keeps the video tail apart, and restores every wrapper; the
GripperWatch sees each open-to-closed command per arm and reads the assist at settle; the state digest moves when the
state does. No Isaac Sim: fakes throughout."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from b1k.planner.pseudo import tape as tp
from omnigibson.tiptop import bench
from omnigibson.tiptop.host.instruments import EP_MEMBERS, UNOWNED, StepLedger
from omnigibson.tiptop.oracle import watch
from omnigibson.tiptop.oracle.watch import GripperWatch, state_digest

MINIMAL = ["--out-dir", "o", "--task-name", "store_honey"]


# ------------------------------------------------------------------------------------------------- the flags
def test_the_defaults_are_todays_run_and_install_nothing():
    args = bench.parse_args(MINIMAL)
    assert (args.runner, args.routing_profile, args.route, args.audit) == ("legacy", "parity", [], False)
    assert (args.runner_tape, args.wstape, args.wstape_path, args.wstape_live_at) == (None, "off", None, None)
    assert (args.replicate, args.seed, args.shadow, args.providers) == (0, None, False, "omnigibson.tiptop.oracle")
    assert bench.instrumented(args) is False


@pytest.mark.parametrize(
    "flags", [["--runner-tape"], ["--runner-tape", "/tmp/x"], ["--wstape", "log"], ["--wstape", "record"], ["--audit"]]
)
def test_each_instrument_flag_installs_the_ledger(flags):
    assert bench.instrumented(bench.parse_args(MINIMAL + flags)) is True


def test_flags_that_only_the_connector_reads_install_nothing_on_the_legacy_runner():
    args = bench.parse_args(
        MINIMAL
        + [
            "--routing-profile",
            "native",
            "--route",
            "place.on=tiptop",
            "--shadow",
            "--seed",
            "7",
            "--providers",
            "x.y",
        ]
    )
    assert bench.instrumented(args) is False
    assert args.route == ["place.on=tiptop"] and args.replicate == 0 and args.seed == 7


def test_a_replicate_without_the_websocket_tape_is_refused():
    """The seed stamp rides the websocket tape: --replicate R > 0 with --wstape off would record R in the Runner
    tape's header while the planner seeded every request 2300 + k, as replicate 0's."""
    with pytest.raises(SystemExit):
        bench.parse_args(MINIMAL + ["--replicate", "2"])
    assert bench.parse_args(MINIMAL + ["--replicate", "2", "--wstape", "log"]).replicate == 2
    assert bench.parse_args(MINIMAL + ["--replicate", "0"]).replicate == 0


def test_runner_tape_without_a_path_means_the_out_dirs_tapes():
    assert bench.parse_args(MINIMAL + ["--runner-tape"]).runner_tape == ""
    assert bench.parse_args(MINIMAL + ["--runner-tape", "--wstape", "log"]).runner_tape == ""


def test_the_connector_runner_is_accepted_now_that_its_host_has_landed():
    """W4-E: host/episode_host.py is in this checkout; the flag was refused with a W4-E notice until then."""
    args = bench.parse_args(MINIMAL + ["--runner", "connector"])
    assert args.runner == "connector" and bench.instrumented(args)


@pytest.mark.parametrize(
    "flags, why",
    [
        (["--route", "place=on=tiptop"], "--route takes SKILL[.QUAL]=BACKEND"),
        (["--route", "Place.on=tiptop"], "--route takes SKILL[.QUAL]=BACKEND"),
        (["--wstape-live-at", "2"], "--wstape-live-at goes with --wstape replay-live"),
        (["--wstape", "replay", "--wstape-live-at", "2"], "--wstape-live-at goes with --wstape replay-live"),
        (["--wstape", "replay-live", "--wstape-live-at", "-1"], "N is a frame index"),
        (["--replicate", "-1"], "R is 0 or more"),
    ],
)
def test_a_malformed_week4_flag_is_refused_with_its_reason(flags, why, capsys):
    with pytest.raises(SystemExit):
        bench.parse_args(MINIMAL + flags)
    assert why in capsys.readouterr().err


def test_seed_everything_seeds_random_and_numpy():
    import random

    bench.seed_everything(11)
    a = (random.random(), float(np.random.rand()))
    bench.seed_everything(11)
    assert a == (random.random(), float(np.random.rand()))


# ------------------------------------------------------------------------------------------------- the fakes
class Body:
    """An object class with a class-level set_position_orientation, as the scene's objects have."""

    def __init__(self, name):
        self.name, self.moves = name, 0

    def set_position_orientation(self, *a, **k):
        self.moves += 1

    def get_position_orientation(self):
        return np.array([self.moves, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])


class Robot(Body):
    def __init__(self):
        super().__init__("robot")
        self.q = np.zeros(4)
        self.controller_action_idx = {"gripper_left": [0], "gripper_right": [1]}
        self.grasp = {"left": 1, "right": -1}  # what is_grasping answers (IsGraspingState: TRUE 1, FALSE -1)

    def set_joint_positions(self, q, **k):
        self.q = np.asarray(q, dtype=float)

    def get_joint_positions(self):
        return self.q

    def is_grasping(self, arm="left"):
        g = self.grasp[arm]
        return g() if callable(g) else g


class EpisodeOver(Exception):
    pass


class Sim:
    """R1ProSim as the ledger sees it: step_env counts n_steps while the episode is open, held_objects is the hand
    record, place_robot/capture/look_at are called by the Episode, objects have a class-level pose setter."""

    def __init__(self):
        self.n_steps, self.episode_open, self.teleports = 0, True, 0
        self.held_objects = {}
        self.robot = Robot()
        self.objects = {"jar": Body("jar"), "lid": Body("lid")}
        self.stop_at = None
        self.seen_boxes = {}

    def step_env(self, action):
        self.n_steps += 1 if self.episode_open else 0
        if self.stop_at is not None and self.n_steps >= self.stop_at:
            raise EpisodeOver("over")
        return {}

    def action(self, left, right=1.0):
        return {"robot": np.array([left, right], dtype=np.float32)}

    def hold(self, n, gripper=1.0):
        for _ in range(n):
            self.step_env(self.action(gripper))

    def place_robot(self, x, y, yaw):
        if (x, y) == (9.0, 9.0):
            raise RuntimeError("base destination rejected")  # R1ProSim: BasePlacementCollision before move_base
        self.move_base(x, y, yaw)
        self.robot.set_joint_positions(np.ones(4))
        self.robot.set_joint_positions(np.zeros(4))

    def move_base(self, x, y, yaw):  # the one place R1ProSim counts a teleport (right_robot calls it too)
        self.robot.set_position_orientation()
        self.teleports += 1

    # what PlanExecutor drives
    OPEN, CLOSE, arm, last_gripper = 1.0, -1.0, "left", 1.0

    def q_arm(self):
        return np.zeros(4)

    def step(self, q_arm, gripper):
        self.last_gripper = float(gripper)
        return self.step_env(self.action(gripper))

    def finger_travel(self, arm=None):
        return 0.04

    def eef_pose_base(self, arm=None):
        return np.eye(4)

    def capture(self, task):
        self.hold(2)
        return {}, {}

    def look_at(self, *names):
        pass


class Renderer:
    def __init__(self):
        self.n = 0

    def render(self):
        self.n += 1


class Episode:
    """bench.Episode's shape: pick stands for the object (self.stand_for, an instance-attribute call once the ledger
    is on), then steps; put_down is an achieve; walk_to_floor is a stand_for."""

    def __init__(self, sim):
        self.sim = sim
        self.floor = "floor.n.01_1"

    def stand_for(self, *names):
        self.sim.place_robot(0.0, 0.0, 0.0)
        self.sim.hold(3)
        return {"x": 0.0, "y": 0.0, "yaw": 0.0}

    def pick(self, name, into=None):
        self.stand_for(name)
        self.sim.capture("task")
        self.sim.hold(3, -1.0)
        self.sim.held_objects[name] = "left"
        return True

    def achieve(self, atoms, arm="left", floor=None, done=None):
        self.sim.hold(4)
        self.sim.held_objects.pop("jar", None)
        return True

    def put_down(self, name, support, floor=None):
        return self.achieve([])

    def walk_to_floor(self, name):
        self.stand_for(name)
        return True

    def open_up(self, name, fraction=None):
        self.sim.hold(2)
        return True

    def release(self, steps=2):
        self.sim.hold(steps)

    def pour(self, item, target):
        return self.achieve([])

    def dwell(self, steps):
        self.sim.hold(steps)
        return steps

    def holding(self, name):
        return name in self.sim.held_objects


LEDGERS = []


@pytest.fixture(autouse=True)
def _finish_ledgers():
    """Class-level wrappers outlive a test unless finished: every ledger a test made is finished after it."""
    yield
    for led in LEDGERS:
        led.finish()
    LEDGERS.clear()


def make_ledger(sim, **kw):
    led = StepLedger(sim, **kw)
    LEDGERS.append(led)
    return led


def ledger_on(sim=None, renderer=None):
    sim = sim or Sim()
    ep = Episode(sim)
    led = make_ledger(sim, renderer=renderer)
    led.install_episode(ep)
    return sim, ep, led


# ------------------------------------------------------------------------------------------------- the ledger
def test_nested_ep_steps_go_to_the_innermost_owner_and_the_top_level_call_keeps_its_whole_delta():
    sim, ep, led = ledger_on()
    assert ep.pick("jar") is True
    owners = led.owners
    # pick: stand_for (3 hold steps) nested, then capture (2) and hold (3) in pick itself
    assert owners["ep.stand_for"]["steps"] == 3 and owners["ep.stand_for"]["env_step_calls"] == 3
    assert owners["ep.pick"]["steps"] == 5 and owners["ep.pick"]["env_step_calls"] == 5
    assert UNOWNED not in owners
    [call] = led.calls
    assert (call.owner, call.step_before, call.step_after, call.steps, call.env_step_calls) == ("ep.pick", 0, 8, 8, 8)
    assert call.error is None and call.call_id is None
    assert led.totals()["sum_matches_sim"] and led.totals()["owned_steps"] == sim.n_steps == 8


def test_every_ep_member_is_owned_and_a_walk_to_floor_nests_a_stand_for():
    sim, ep, led = ledger_on()
    assert set(EP_MEMBERS) == {
        "pick",
        "achieve",
        "put_down",
        "open_up",
        "release",
        "pour",
        "dwell",
        "stand_for",
        "walk_to_floor",
    }
    ep.walk_to_floor("floor")
    ep.put_down("jar", "table")
    ep.open_up("cabinet", 0.8)
    ep.release()
    ep.pour("cup", "bowl")
    ep.dwell(5)
    o = led.owners
    assert o["ep.stand_for"]["steps"] == 3 and "ep.walk_to_floor" in o and o["ep.walk_to_floor"]["steps"] == 0
    assert o["ep.achieve"]["steps"] == 8 and o["ep.put_down"]["steps"] == 0 and o["ep.pour"]["steps"] == 0
    assert o["ep.open_up"]["steps"] == 2 and o["ep.release"]["steps"] == 2 and o["ep.dwell"]["steps"] == 5
    assert [c.owner for c in led.calls] == [
        "ep.walk_to_floor",
        "ep.put_down",
        "ep.open_up",
        "ep.release",
        "ep.pour",
        "ep.dwell",
    ]
    assert [c.steps for c in led.calls] == [3, 4, 2, 2, 4, 5]
    assert led.totals()["owned_steps"] == sim.n_steps == 20


def test_write_calls_are_counted_per_owner():
    r = Renderer()
    sim, ep, led = ledger_on(renderer=r)
    ep.pick("jar")  # stand_for: place_robot, robot pose, 2 joint writes; pick: capture, one held write
    with led.owner("rt"):
        r.render()
        sim.objects["jar"].set_position_orientation()
        sim.look_at("jar")
    ep.achieve([])  # pops jar: one held write
    o = led.owners
    assert (
        o["ep.stand_for"]["place_robot"],
        o["ep.stand_for"]["set_position_orientation"],
        o["ep.stand_for"]["set_joint_positions"],
        o["ep.stand_for"]["writes"],
    ) == (1, 1, 2, 3)
    assert (o["ep.pick"]["capture"], o["ep.pick"]["held_writes"], o["ep.pick"]["place_robot"]) == (1, 1, 0)
    assert (o["rt"]["renders"], o["rt"]["set_position_orientation"], o["rt"]["look_at"], o["rt"]["steps"]) == (
        1,
        1,
        1,
        0,
    )
    assert o["ep.achieve"]["held_writes"] == 1 and o["ep.achieve"]["writes"] == 1
    assert led.totals()["unowned_writes"] == 0


def test_a_pose_write_through_a_super_chain_counts_once_and_a_subclass_shares_the_base_wrapper():
    class Fancy(Body):
        def set_position_orientation(self, *a, **k):
            super().set_position_orientation(*a, **k)

    sim = Sim()
    sim.objects["cup"] = Fancy("cup")
    led = make_ledger(sim)
    with led.owner("rt"):
        sim.objects["cup"].set_position_orientation()  # Fancy's, then Body's through super(): one write
        sim.robot.set_position_orientation()  # Robot defines none: Body's wrapper, one write
        sim.objects["jar"].set_position_orientation()
    assert led.owners["rt"]["set_position_orientation"] == 3 and sim.objects["cup"].moves == 1
    led.finish()
    assert "set_position_orientation" not in Robot.__dict__ and not hasattr(
        Fancy.set_position_orientation, "__wrapped__"
    )


def test_steps_outside_every_owner_are_unowned_and_the_video_tail_is_closed():
    sim, ep, led = ledger_on()
    sim.hold(2)
    ep.dwell(3)
    sim.episode_open = False
    sim.hold(4)  # the epilogue: n_steps does not move, the calls are counted apart
    o = led.owners
    assert o[UNOWNED]["steps"] == 2 and o[UNOWNED]["env_step_calls"] == 2
    assert o["ep.dwell"]["steps"] == 3
    t = led.totals()
    assert (t["unowned_steps"], t["closed_env_step_calls"], t["owned_steps"], t["n_steps"]) == (2, 4, 5, 5)
    assert led.closed.env_step_calls == 4 and led.closed.steps == 0 and led.step_env.calls == 9


def test_an_episode_over_raised_by_the_step_is_still_counted_and_named_on_the_call():
    sim, ep, led = ledger_on()
    sim.stop_at = 2
    with pytest.raises(EpisodeOver):
        ep.dwell(5)
    assert led.owners["ep.dwell"]["steps"] == 2 and led.owners["ep.dwell"]["env_step_calls"] == 2
    [call] = led.calls
    assert call.error == "EpisodeOver" and call.steps == 2


def test_a_replaced_hand_record_is_rewrapped_and_counted_as_one_write():
    sim, ep, led = ledger_on()
    with led.owner("restore"):
        sim.held_objects = {"jar": "left"}  # skillbench's restore does this
        sim.hold(1)  # the next step re-wraps it
        sim.held_objects["lid"] = "right"
    assert led.owners["restore"]["held_writes"] == 2
    assert dict(sim.held_objects) == {"jar": "left", "lid": "right"}


def test_the_ledger_never_changes_a_value():
    sim, ep, led = ledger_on()
    assert ep.pick("jar") is True and dict(sim.held_objects) == {"jar": "left"}
    assert ep.stand_for("jar") == {"x": 0.0, "y": 0.0, "yaw": 0.0}
    assert sim.capture("t") == ({}, {})
    assert dict(sim.held_objects) == {"jar": "left"} and sim.held_objects.get("jar") == "left"
    assert sim.n_steps == 3 + 2 + 3 + 3 + 2


def test_finish_restores_every_wrapper_and_returns_the_summary():
    r = Renderer()
    sim, ep, led = ledger_on(renderer=r)
    ep.pick("jar")
    summary = led.finish()
    assert "step_env" not in sim.__dict__ and "place_robot" not in sim.__dict__ and "render" not in r.__dict__
    assert "set_joint_positions" not in sim.robot.__dict__
    assert "set_position_orientation" not in Robot.__dict__, "the wrapper goes on the class that defines it"
    assert Body.__dict__["set_position_orientation"] is Body.set_position_orientation and not hasattr(
        Body.set_position_orientation, "__wrapped__"
    )
    assert type(sim.held_objects) is dict and sim.held_objects == {"jar": "left"}
    assert "pick" not in ep.__dict__
    ep.pick("lid")  # unwrapped: the sim moves on (8 more steps), the ledger counts nothing more
    again = led.finish()
    assert (again["owners"], again["calls"], again["closed"]) == (
        summary["owners"],
        summary["calls"],
        summary["closed"],
    )
    assert again["totals"]["owned_steps"] == 8 and again["totals"]["n_steps"] == 16 and sim.n_steps == 16
    assert set(summary) == {"owners", "closed", "calls", "totals"}
    json.dumps(summary)


def test_the_call_id_a_host_sets_is_on_the_top_level_call_and_the_gripper_rows():
    sim, ep, led = ledger_on()
    g = GripperWatch(sim, led, settle_steps=2)
    led.call_id = "q1-3"
    ep.pick("jar")
    assert led.calls[0].call_id == "q1-3"
    [close] = g.closes
    assert (close["owner"], close["call_id"]) == ("ep.pick", "q1-3")


# ------------------------------------------------------------------------------------------------- the watch
def test_the_watch_records_each_open_to_closed_command_per_arm_and_reads_the_assist_at_settle():
    sim = Sim()
    led = make_ledger(sim)
    g = GripperWatch(sim, led, settle_steps=3)
    sim.hold(2, 1.0)  # open: nothing
    with led.owner("ep.pick"):
        sim.hold(1, -1.0)  # the close is seen at this step
        [row] = g.events
        assert (row["event"], row["arm"], row["step"], row["owner"], row["settled_at"]) == (
            "close",
            "left",
            3,
            "ep.pick",
            None,
        )
        sim.hold(2, -1.0)  # settle_steps steps of the same closed command: final
        assert row["is_grasping"] == 1 and row["settled_at"] == 5
    sim.robot.grasp["left"] = -1
    sim.hold(1, -1.0)
    assert len(g.events) == 1 and row["is_grasping"] == 1, "a closed command that stays closed is one close, settled"
    sim.hold(1, 1.0)  # open again
    assert [e["event"] for e in g.events] == ["close", "open"] and g.events[1]["owner"] == UNOWNED
    sim.hold(1, -1.0)  # a second close, opened before it settles: its last closed step's reading, -1 = on air
    sim.hold(1, 1.0)
    assert [e["event"] for e in g.events] == ["close", "open", "close", "open"]
    assert g.events[2]["is_grasping"] == -1 and g.events[2]["settled_at"] == 8
    assert g.finish() is g.events and len(g.closes) == 2


def test_a_creeping_close_settles_on_the_step_the_executor_reads_is_grasping():
    """PlanExecutor.set_gripper(creep=True): the command ramps down (below 0 halfway), then holds CLOSE for
    gripper_hold_steps before the executor logs is_grasping. The weld here lands late in the hold."""
    sim = Sim()
    led = make_ledger(sim)
    g = GripperWatch(sim, led)  # SETTLE_STEPS, the executor's 25
    sim.robot.grasp["left"] = lambda: 1 if sim.n_steps >= 30 else -1
    sim.hold(1, 1.0)
    ramp = [0.75, 0.5, 0.25, -0.25, -0.5, -0.75]  # crosses 0 at the 4th
    for c in ramp:
        sim.hold(1, c)
    assert len(g.closes) == 1 and g.closes[0]["step"] == 1 + 4 and g.closes[0]["settled_at"] is None
    sim.hold(25, -1.0)  # the hold: steps 8..32, read by the executor after the last (25 closed steps from the
    # crossing would read at step 29, before the weld)
    close = g.closes[0]
    assert close["settled_at"] == 1 + len(ramp) + 25 == sim.n_steps and close["is_grasping"] == 1
    sim.robot.grasp["left"] = -1
    sim.hold(3, -1.0)
    assert close["is_grasping"] == 1, "settled: later readings are not the executor's"


def test_an_open_before_settle_keeps_the_last_closed_reading_not_the_one_after_letting_go():
    sim = Sim()
    commanded = []
    sim.robot.grasp["left"] = lambda: 1 if commanded and commanded[-1] < 0 else -1  # the weld lets go on an open
    real = sim.step_env

    def step_env(action):  # the command reaches the assist before the observer reads it, as in the sim
        commanded.append(float(action["robot"][0]))
        return real(action)

    sim.step_env = step_env
    led = make_ledger(sim)
    g = GripperWatch(sim, led, settle_steps=25)
    sim.hold(1, 1.0)
    sim.hold(4, -1.0)
    sim.hold(1, 1.0)
    close, opened = g.events
    assert (close["is_grasping"], close["settled_at"]) == (1, 5) and (opened["event"], opened["is_grasping"]) == (
        "open",
        -1,
    )


def test_a_command_closed_from_the_first_step_is_no_transition_and_the_right_arm_is_its_own():
    sim = Sim()
    led = make_ledger(sim)
    g = GripperWatch(sim, led, settle_steps=1)
    sim.step_env(sim.action(-1.0, 1.0))  # left already closed at the first step seen: no close event
    sim.step_env(sim.action(-1.0, -1.0))  # right closes
    sim.step_env(sim.action(-1.0, -1.0))
    assert [(e["arm"], e["event"], e["is_grasping"]) for e in g.events] == [("right", "close", -1)]
    sim.stop_at = 4
    with pytest.raises(EpisodeOver):
        sim.step_env(sim.action(1.0, -1.0))  # the left opens on the step that ends the episode: still seen
    assert [(e["arm"], e["event"]) for e in g.events] == [("right", "close"), ("left", "open")]


def test_a_pending_close_is_final_when_the_watch_is_finished_and_an_unreadable_assist_is_unknown():
    sim = Sim()
    led = make_ledger(sim)
    g = GripperWatch(sim, led, settle_steps=50)
    sim.hold(1, 1.0)
    sim.hold(1, -1.0)
    assert g.events[0]["is_grasping"] == 1 and g.events[0]["settled_at"] is None
    g.finish()
    assert g.events[0]["is_grasping"] == 1 and g.events[0]["settled_at"] == 2
    sim.robot.grasp["left"] = lambda: 1 / 0
    sim.hold(1, 1.0)
    sim.hold(1, -1.0)
    assert g.events[-1]["is_grasping"] == 0, "counters.py reads every close's is_grasping as an int"


# ------------------------------------------------------------------------------------------------- the digest
def test_the_state_digest_moves_with_the_state_and_round_trips_the_tape_codec():
    sim = Sim()
    led = make_ledger(sim)
    knowledge = SimpleNamespace(seen={"jar": np.array([1.0, 2.0])})
    d0 = state_digest(sim, knowledge)
    assert set(d0) == {"n_steps", "env_steps", "teleports", "held", "robot", "objects", "joints", "memory"}
    assert (d0["n_steps"], d0["env_steps"], d0["teleports"], d0["held"], d0["memory"][0]) == (0, 0, 0, (), 1)
    assert tp.loads(tp.dumps(d0)) == d0
    sim.hold(1)
    d1 = state_digest(sim, knowledge)
    assert (d1["n_steps"], d1["env_steps"]) == (1, 1) and d1["robot"] == d0["robot"] and d1["objects"] == d0["objects"]
    sim.robot.set_joint_positions(np.ones(4))
    d2 = state_digest(sim, knowledge)
    assert d2["robot"] != d1["robot"] and d2["objects"] == d1["objects"]
    sim.objects["lid"].set_position_orientation()
    d3 = state_digest(sim, knowledge)
    assert d3["objects"] != d2["objects"]
    sim.held_objects["jar"] = "left"
    assert state_digest(sim, knowledge)["held"] == (("jar", "left"),)
    knowledge.seen["lid"] = np.array([0.0])
    d4 = state_digest(sim, knowledge)
    assert d4["memory"][0] == 2 and d4["memory"][1] != d3["memory"][1]
    led.finish()


def test_the_digest_survives_a_fake_without_a_robot_or_a_knowledge_source():
    d = state_digest(SimpleNamespace(n_steps=3), None)
    assert (d["n_steps"], d["env_steps"], d["robot"], d["objects"], d["memory"], d["held"]) == (
        3,
        None,
        None,
        None,
        None,
        (),
    )
    assert watch.SETTLE_STEPS == 25, "the executor's gripper_hold_steps"


def test_renders_inside_an_env_step_are_apart_from_the_explicit_ones():
    r = Renderer()
    sim = Sim()
    real = sim.step_env

    def step_env(action):  # og.sim.step renders once of itself (simulator.py step)
        r.render()
        return real(action)

    sim.step_env = step_env
    led = make_ledger(sim, renderer=r)
    with led.owner("ep.pick"):
        sim.hold(3)
        r.render()  # a capture's own render
    row = led.owners["ep.pick"]
    assert (row["renders"], row["renders_in_step"], row["steps"]) == (1, 3, 3)


def test_a_hand_record_replaced_inside_an_owner_is_that_owners_write():
    sim, ep, led = ledger_on()
    sim.held_objects = {"cup": "right"}  # replaced while unowned, and no step before the next owner begins
    with led.owner("restore"):
        sim.held_objects = {"jar": "left"}  # replaced, and no step before the owner ends
    sim.hold(1)
    assert led.owners["restore"]["held_writes"] == 1 and led.owners[UNOWNED]["held_writes"] == 1
    led.finish()
    with led.owner("after"):
        sim.held_objects = {"lid": "right"}
    assert type(sim.held_objects) is dict, "a finished ledger wraps nothing again"


# ------------------------------------------------------------------------------------------------- bench.main
class MainSim(Sim):
    """What bench.main touches on R1ProSim between build_r1pro_sim and the result JSON."""

    def __init__(self):
        super().__init__()
        self.env = SimpleNamespace(reset=lambda: None, task=SimpleNamespace(success=True))
        self.recorders, self.send_room, self.video_caption, self.max_steps = [], False, "", 10**6
        self.last_gripper = 1.0

    def task_scope(self):
        return {"jar.n.01_1", "cabinet.n.01_1"}

    def reset_embodiment(self, embodiment):
        pass

    def begin_episode(self, metrics, stop_when_done=True, max_steps=None, name=""):
        self.n_steps, self.episode_open = 0, True

    def end_episode(self):
        self.episode_open = False
        return self.n_steps

    def mark_goal_initial(self):
        pass

    def goal_status(self):
        return {"satisfied": [], "unsatisfied": [], "total": 1}

    def hands(self):
        return dict(self.held_objects)


class MainEpisode(Episode):
    def __init__(self, sim, args, planners, knowledge, out_dir, spec=None):
        super().__init__(sim)
        self.records = [{"round": 1, "env_steps": 3}]


class MainStrategy:
    """strategy_for's Runner as bench.main uses it: its inputs, and a run that picks and places, noting what it saw."""

    def __init__(self, seen):
        self.goal = [{"predicate": "inside", "args": ["jar.n.01_1", "cabinet.n.01_1"]}]
        self.options = [list(self.goal)]
        self.scope, self.attempts = ["cabinet.n.01_1", "jar.n.01_1"], 2
        self.spec = SimpleNamespace(task="store_honey", instruction="put the jar in the cabinet")
        self.seen = seen

    def run(self, ep):
        from b1k.bridge import client as bridge_client

        sim = ep.sim if isinstance(ep, (Episode,)) else ep._inner.sim
        self.seen.append(
            {
                "ep": type(ep).__name__,
                "step_env_wrapped": "step_env" in vars(sim),
                "pick_wrapped": "pick" in vars(ep._inner if hasattr(ep, "_inner") else ep),
                "held_type": type(sim.held_objects).__name__,
                "body_class": Body.__dict__["set_position_orientation"] is BODY_SET_POSE,
                "connect": bridge_client.connect is REAL_CONNECT,
            }
        )
        ep.pick("jar.n.01_1")
        ep.achieve(self.goal)


BODY_SET_POSE = Body.__dict__["set_position_orientation"]
REAL_CONNECT = None
TODAY_KEYS = {"task", "instance_id", "rollout_id", "steps", "success", "q_score", "bench"}
TODAY_BENCH = {
    "reason",
    "max_steps",
    "wall_time_s",
    "knowledge",
    "collision_map",
    "teleports",
    "goal",
    "video",
    "rounds",
}


def run_main(tmp_path, monkeypatch, extra, strategy=None):
    """bench.main end to end on fakes: every simulator and planner piece faked at its seam, nothing else."""
    import omnigibson.eval.evaluator as evaluator
    import omnigibson.eval.utils.score_utils as score_utils
    import omnigibson.metrics as metrics
    import omnigibson.tiptop.knowledge as knowledge_mod
    from b1k.bridge import client as bridge_client
    from b1k.bridge import strategies

    global REAL_CONNECT
    REAL_CONNECT = bridge_client.connect
    seen, sims = [], []

    class Metric:
        def __init__(self, human):
            pass

        def step(self, *a):
            pass

        def aggregate(self, env):
            return {"q_score": {"final": 1.0}}

    def build(args, embodiment, max_steps):
        sims.append(MainSim())
        return sims[-1]

    def posture(sim, args, embodiment):  # the embodiment posture steps before the Runner starts
        sim.hold(3)

    monkeypatch.setattr(bench, "setup_logging", lambda: None)
    monkeypatch.setattr(evaluator, "resolve_instance_ids", lambda task, instances, mode: [301 + i for i in instances])
    monkeypatch.setattr(evaluator, "load_task_instance", lambda *a, **k: None)
    monkeypatch.setattr(score_utils, "load_human_stats", lambda task: {"length": 1000})
    monkeypatch.setattr(metrics, "AgentMetric", Metric)
    monkeypatch.setattr(metrics, "TaskMetric", Metric)
    monkeypatch.setattr(knowledge_mod, "make_knowledge", lambda *a, **k: SimpleNamespace(report=lambda: {}, seen={}))
    monkeypatch.setattr(strategies, "strategy_for", lambda *a, **k: (strategy or MainStrategy)(seen))
    monkeypatch.setattr(strategies, "task_goal_atoms", lambda sim: [])
    monkeypatch.setattr(strategies, "task_goal_options", lambda sim: [[{"predicate": "inside", "args": ["a", "b"]}]])
    monkeypatch.setattr(bench, "connect_planners", lambda args: (None, {"embodiment": {}}, None, None))
    monkeypatch.setattr(bench, "check_imports", lambda *m: "imports: fake")
    monkeypatch.setattr(bench, "build_r1pro_sim", build)
    monkeypatch.setattr(bench, "apply_embodiment_posture", posture)
    monkeypatch.setattr(bench, "wants_home_torso", lambda spec: False)
    monkeypatch.setattr(bench, "Episode", MainEpisode)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as code:
        bench.main(
            [
                "--out-dir",
                str(out),
                "--task-name",
                "store_honey",
                "--instances",
                "0",
                "--no-video",
                "--no-state-stream",
                *extra,
            ]
        )
    assert code.value.code == 0
    assert bridge_client.connect is REAL_CONNECT, "nothing left patched"
    result = json.loads((out / "json" / "store_honey_301_0.json").read_text())
    return result, seen, sims[0], out


def test_bench_main_with_no_week4_flag_wraps_nothing_and_writes_todays_keys(tmp_path, monkeypatch):
    result, seen, sim, out = run_main(tmp_path, monkeypatch, [])
    assert seen == [
        {
            "ep": "MainEpisode",
            "step_env_wrapped": False,
            "pick_wrapped": False,
            "held_type": "dict",
            "body_class": True,
            "connect": True,
        }
    ]
    assert set(result) == TODAY_KEYS and set(result["bench"]) == TODAY_BENCH
    assert not (out / "tapes").exists() and not (out / "wstape").exists()
    assert not (out / "store_honey_301_0" / "ledger.jsonl").exists()


def test_bench_main_under_week4_flags_writes_the_instruments_and_the_runner_tape(tmp_path, monkeypatch):
    result, seen, sim, out = run_main(tmp_path, monkeypatch, ["--runner-tape", "--wstape", "log", "--seed", "3"])
    assert seen == [
        {
            "ep": "TapeRecorder",
            "step_env_wrapped": True,
            "pick_wrapped": True,
            "held_type": "_HeldRecord",
            "body_class": False,
            "connect": False,
        }
    ]
    assert set(result) == TODAY_KEYS and set(result["bench"]) == TODAY_BENCH | {"instruments"}
    block = result["bench"]["instruments"]
    t = block["ledger"]["totals"]
    assert t["n0"] == 3, "the posture's steps come before the Runner's first call"
    assert (t["owned_steps"], t["unowned_steps"], t["n_steps"], t["sum_matches_sim"]) == (8 + 4, 0, 3 + 8 + 4, True)
    assert [c["owner"] for c in block["ledger"]["calls"]] == ["ep.pick", "ep.achieve"]
    assert block["wstape"]["mode"] == "log" and block["runner_tape"]["writes"] == 2
    tape = tp.Tape.load(out / "tapes" / "store_honey_301_0.json")
    assert [w["member"] for w in tape.writes] == ["pick", "achieve"]
    assert tape.writes[0]["step"] == [3, 3 + 8] and tape.writes[0]["digest"][1]["held"] == (("jar.n.01_1", "left"),)
    assert tape.header["ending"] == {"reason": "strategy finished", "raised": None}
    assert tape.header["options"] == [[{"predicate": "inside", "args": ["a", "b"]}]]
    assert tape.header["planner_modules"] == {"left": {}}, "the planner's own provenance, off its metadata"
    rows = [json.loads(line) for line in (out / "store_honey_301_0" / "ledger.jsonl").read_text().splitlines()]
    assert {r["owner"] for r in rows} == {"ep.pick", "ep.stand_for", "ep.achieve"}
    assert type(sim.held_objects) is dict and "step_env" not in vars(sim), "finished at the instance's end"


class DivergingStrategy(MainStrategy):
    """A strict --wstape replay's divergence, raised out of the Runner's run: a BaseException bench.main names."""

    def run(self, ep):
        from omnigibson.tiptop.host.wstape import TapeDiverged

        ep.pick("jar.n.01_1")
        raise TapeDiverged("q_init", 1, "3 of 8 elements differ", "plan")


def test_bench_main_names_a_tape_divergence_as_the_reason_and_still_writes_the_instruments(tmp_path, monkeypatch):
    result, seen, sim, out = run_main(
        tmp_path, monkeypatch, ["--runner-tape", "--wstape", "log"], strategy=DivergingStrategy
    )
    assert result["bench"]["reason"] == "tape diverged: q_init"
    assert result["bench"]["instruments"]["runner_tape"]["writes"] == 1
    tape = tp.Tape.load(out / "tapes" / "store_honey_301_0.json")
    assert tape.header["ending"]["raised"].type == "TapeDiverged"
    assert (out / "store_honey_301_0" / "gripper.jsonl").exists()


def test_bench_main_settles_the_gripper_watch_on_the_executors_own_hold(tmp_path, monkeypatch):
    made = []
    real = watch.GripperWatch

    def recording(sim, ledger, settle_steps=watch.SETTLE_STEPS):
        made.append(settle_steps)
        return real(sim, ledger, settle_steps=settle_steps)

    monkeypatch.setattr(watch, "GripperWatch", recording)
    run_main(tmp_path, monkeypatch, ["--runner-tape", "--gripper-hold-steps", "40"])
    assert made == [40]


def test_the_digest_sees_a_joint_that_moves_no_root_pose():
    """An open or a close slides a drawer and moves no object's root: the tracked objects' joint positions are
    their own digest key (store_honey's cabinet, j_link_4)."""

    class Cabinet(Body):
        n_dof = 1

        def __init__(self, name):
            super().__init__(name)
            self.q = np.zeros(1)

        def get_joint_positions(self):
            return self.q

    sim = Sim()
    sim.objects["cabinet"] = Cabinet("cabinet")
    d0 = state_digest(sim, None)
    sim.objects["cabinet"].q = np.array([0.317])
    d1 = state_digest(sim, None)
    assert d0["objects"] == d1["objects"] and d0["joints"] != d1["joints"] and d1["joints"][0] == 1


def test_tape_diff_compares_two_digests_on_the_keys_both_carry():
    td = tape_diff()
    old = {"n_steps": 3, "robot": "a", "objects": "b"}
    assert td._digests_equal(old, {**old, "joints": (1, "c")}), "a key only the newer digest has is not a divergence"
    assert not td._digests_equal(old, {**old, "robot": "x", "joints": (1, "c")})
    assert td._one_sided([old, old], [{**old, "joints": (1, "c")}, old]) == ["joints"]


# ------------------------------------------------------------------------------------------------- scripts/tape_diff.py
def tape_diff():
    import importlib.util
    from pathlib import Path

    script = Path(bench.__file__).parent / "scripts" / "tape_diff.py"
    spec = importlib.util.spec_from_file_location("tiptop_tape_diff", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recorded_runner(path, pick_ok, step=1, ending=None):
    """A real Runner on the strategies suite's fake, recorded as bench.main records it (header, options, scope,
    step probe and digest around every write, the ending)."""
    import itertools

    from test_tiptop_strategies import FakeEpisode, basket_world

    from b1k.bridge.strategies import strategy_for

    boxes, goal = basket_world()
    ep = FakeEpisode(boxes, pick_ok=pick_ok, place_ok={"candle.n.01_1", "candle.n.01_2", "cookie.n.01_1"})
    runner = strategy_for("assembling_gift_baskets", goal, options=[goal])
    tape = tp.Tape(
        tp.header(
            "assembling_gift_baskets",
            0,
            "legacy",
            "parity",
            0,
            strategy=runner,
            arms=("left",),
            floor=ep.floor,
            max_steps=None,
        )
    )
    tape.header["options"], tape.header["scope"] = [goal], list(runner.scope)
    clock = itertools.count(0, step)
    runner.run(tp.TapeRecorder(ep, tape, step_probe=lambda: next(clock), digest=lambda: {"calls": len(ep.calls)}))
    tape.header["ending"] = ending or {"reason": "strategy finished", "raised": None}
    tape.save(path)
    return path


ALL = {"candle.n.01_1", "candle.n.01_2", "cookie.n.01_1"}


def test_e0_replays_the_runner_from_its_tape_and_fails_on_another_ending(tmp_path):
    td = tape_diff()
    a = recorded_runner(tmp_path / "a.json", ALL)
    report = td.e0(a)
    assert report["pass"] and report["writes_consumed"][0] == report["writes_consumed"][1] > 0
    assert report["unanswerable"] == [] and report["ending"] is None and report["bench_reason"] == "strategy finished"
    over = tp.Exc("EpisodeOver", "omnigibson.tiptop.scene", "timeout", ("timeout", 99))
    b = recorded_runner(tmp_path / "b.json", ALL, ending={"reason": "timeout", "raised": over})
    assert td.e0(b)["pass"] is False, "the Runner finished; the bench saw an EpisodeOver"
    assert td.main(["e0", str(a)]) == 0 and td.main(["e0", str(b)]) == 1


def test_the_runner_tape_diff_names_the_first_divergence_and_the_per_write_deltas(tmp_path):
    td = tape_diff()
    a = recorded_runner(tmp_path / "a.json", ALL)
    same = recorded_runner(tmp_path / "same.json", ALL, step=2)  # the same run, every write twice as long
    other = recorded_runner(tmp_path / "other.json", ALL - {"candle.n.01_1"})  # the first pick fails
    r = td.runner_report(a, same)
    assert r["headers_equal"] and r["first_divergence"]["kind"] == "diagnostic", "only the step probe differs"
    w = r["per_write"][1]
    assert (w["delta_a"], w["delta_b"], w["digest_before_equal"], w["outcome_equal"]) == (1, 2, True, True)
    r = td.runner_report(a, other)
    d = r["first_divergence"]
    assert d["kind"] == "answer" and d["a"]["member"] == "pick" and d["a"]["ret"] is True and d["b"]["ret"] is False
    first = next(w for w in r["per_write"] if not (w["same_call"] and w["outcome_equal"]))
    assert r["prefix"] == d["index"] and first["member"] == "pick" and first["same_call"] and not first["outcome_equal"]


def test_teleports_are_counted_where_they_happen_and_a_refused_placement_is_a_call_only():
    sim, ep, led = ledger_on()
    with led.owner("go_to"):
        sim.place_robot(1.0, 2.0, 0.0)
        with pytest.raises(RuntimeError):
            sim.place_robot(9.0, 9.0, 0.0)  # refused: no teleport
    with led.owner("ep.stand_for"):
        sim.move_base(0.0, 0.0, 0.0)  # bench's right_robot: a teleport outside place_robot
    o = led.owners
    assert (o["go_to"]["place_robot"], o["go_to"]["place_robot_calls"]) == (1, 2)
    assert (o["ep.stand_for"]["place_robot"], o["ep.stand_for"]["place_robot_calls"]) == (1, 0)
    assert sum(r["place_robot"] for r in o.values()) == sim.teleports == 2, "the column sums to bench.teleports"


def test_a_close_names_who_issued_it_the_plan_executor_or_the_sim():
    from b1k.bridge.executor import PlanExecutor

    sim = Sim()
    led = make_ledger(sim)
    g = GripperWatch(sim, led)
    ex = PlanExecutor(sim)
    ex.set_gripper("close", creep=True)  # a plan's Pick close: the one the executor logs with is_grasping=
    ex.set_gripper("open")
    assert ex.start_gripper("closed") is True  # closed before the plan: not logged with is_grasping=
    ex.set_gripper("open")
    sim.hold(3, -1.0)  # a closed fist the sim itself commands (a drawer pull)
    assert [c["via"] for c in g.closes] == ["executor", "executor.start", "sim"]
    assert g.closes[0]["settled_at"] is not None and g.closes[0]["is_grasping"] == 1
