"""The connector host on the bench (WEEK4_PLAN §3.1, §3.3 episode_host.py, §5.6 U0; W4-E): what bench.main runs
under ``--runner connector`` in the place of ``strategy.run(ep)``.

``build(...)`` wires, under the StepLedger owner ``host.build`` (0 steps, the state digest unchanged):
- the providers' pseudo_services over the Episode, hands="episode", collision pinned to "map" whatever --room says,
  the scorer scope-only, no shadow unless --shadow;
- EpisodeRegistry over EPISODE_SPECS with the routing profile (parity | native) and the --route overlays, whose
  backends are legacy = EpisodeLegacyBackend (the Runner's literal Episode call, taken from the LegacyChannel) and
  tiptop / scripted from skillbench.make_backends (its own legacy entry dropped);
- EpisodeNavigator over a TeleportNavigator (owner ``go_to``), the CaptureObserver (owner ``observe``), BenchHost
  with the sim clock stamp;
- the Runtime with the providers' task_info, skillrun.PASSTHROUGH = (EpisodeOver,), DirectConnector with env_step
  under owner ``rt`` (an EpisodeOver raised inside it is counted: U0-a's X);
- ConnectorAudit (U0-d) and, with --audit, PurityAudit around the connector and DualEpisode around the shim;
- PseudoPlanner(factory=strategy_for, ...) whose Runner's construction inputs are compared with the strategy the
  bench built (the Runner-input equality; a mismatch is logged and flagged, never raised).

bench.py passes the providers module, strategies.strategy_for, its strategy, the ledger and the oracle package's
watch module in; this module never names that package (the privileged scan).

``close(reason, raised)`` never raises: it judges U0-a..d and the G3 items, streams skill_calls, goal_checks, ledger
and audit rows to jsonl in the instance dir, and returns the ``connector`` block for the result JSON.
"""

from __future__ import annotations

import functools
import json
import logging
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import b1k.runtime.skillrun as skillrun
from b1k.planner.pseudo.planner import PseudoPlanner
from b1k.planner.pseudo.shim import ClockView, counts as shim_counts
from b1k.planner.pseudo.tape import runner_inputs
from b1k.runtime.core import Runtime
from b1k.runtime.direct import DirectConnector
from omnigibson.tiptop.host.bench_host import BenchHost
from omnigibson.tiptop.host.capture_observer import CaptureObserver
from omnigibson.tiptop.host.legacy_channel import LegacyChannel
from omnigibson.tiptop.host.legacy_episode import EPISODE_SPECS, EpisodeLegacyBackend, EpisodeNavigator, EpisodeRegistry
from omnigibson.tiptop.host.q1_audit import AuditedConnector, DualEpisode, PurityAudit
from omnigibson.tiptop.host.routing_profiles import PARITY, native, overlay
from omnigibson.tiptop.host.teleport_nav import TeleportNavigator
from omnigibson.tiptop.scene import EpisodeOver

log = logging.getLogger("omnigibson.tiptop")
COLLISION = "map"  # pinned: the connector's native calls plan in the map, whatever --room says (WEEK4_PLAN §5)
EP_PREFIX = "ep."
REQUEST_OPS_ALLOWED = ("plan", "move")  # G3: a parity run's planner requests are the legacy Episode's alone
PLANNER_FRAMES_IGNORED = ("metadata", "stream")  # the connect frame and the state stream are not requests
HAND_SKILLS = ("pick_up", "place", "release", "hold", "press")  # W4-I's HandRefresh (not wired here: refresh == 0)


def _plain(x: Any) -> Any:
    """JSON-plain: what json.dumps cannot carry becomes its str."""
    return json.loads(json.dumps(x, default=_default))


def _default(x: Any):
    if isinstance(x, (set, frozenset)):
        return sorted(x, key=repr)
    if isinstance(x, BaseException):
        return f"{type(x).__name__}: {x}"
    if hasattr(x, "value") and isinstance(getattr(x, "value"), (str, int)):  # an enum
        return x.value
    return str(x)


def owned_generator(ledger, name: str, fn: Callable) -> Callable:
    """``fn`` (a generator function: observe, go_to, a backend's run) with its whole body, first yield to return,
    under the ledger owner ``name``: the steps a self-stepping run takes inside land on that owner (an ``ep.*``
    owner nested inside is the innermost and takes its own)."""

    @functools.wraps(fn)
    def call(*args, **kwargs):
        with ledger.owner(name):
            return (yield from fn(*args, **kwargs))

    return call


def _owned_call(ledger, name: str, fn: Callable, *args, **kwargs):
    with ledger.owner(name):
        return fn(*args, **kwargs)


def call_id_generator(ledger, fn: Callable) -> Callable:
    """A backend's run with the connector call id in flight on the ledger (the gripper rows and the top-level ep.*
    calls carry it)."""

    @functools.wraps(fn)
    def run(call, svc, obs):
        ledger.call_id = call.call_id
        try:
            return (yield from fn(call, svc, obs))
        finally:
            ledger.call_id = None

    return run


def routing_for(args) -> dict:
    """The profile (parity | native, the shadow only with --shadow) with the --route overlays."""
    profile = PARITY if getattr(args, "routing_profile", "parity") == "parity" else native(shadow=bool(getattr(args, "shadow", False)))
    return overlay(profile, list(getattr(args, "route", ()) or ()))


def planner_client_of(planners: dict):
    """The skills' planner: the left arm's client, or ArmPlanners over both when a right-arm planner is connected
    (skillbench.main's rule)."""
    left = planners.get("left")
    client = left[0] if isinstance(left, tuple) else left
    right = planners.get("right")
    if right is None:
        return client
    from b1k.bridge.client import ArmPlanners

    return ArmPlanners(client, right[0] if isinstance(right, tuple) else right)


def u0a(rt, live_steps: int, over_in_env_step: int, episode_over: bool) -> dict:
    """U0-a, the amended identity (WEEK4_PLAN §5.6): rt.step - idle == Σ charged + L + X with idle 0; L and X 0
    unless the reason is EpisodeOver."""
    charged = sum(int(v) for v in rt.charged.values())
    identity = rt.step - rt.idle_steps == charged + live_steps + over_in_env_step
    ok = identity and rt.idle_steps == 0 and (episode_over or (live_steps == 0 and over_in_env_step == 0))
    return {"ok": bool(ok), "identity": bool(identity), "step": int(rt.step), "idle_steps": int(rt.idle_steps),
            "charged": dict(rt.charged), "charged_sum": charged, "L": int(live_steps), "X": int(over_in_env_step),
            "episode_over": bool(episode_over)}


class EpisodeHost:
    """What build() returns; see the module docstring. ``run()`` is the planner over the connector, ``close()`` the
    verdicts and the block."""

    def __init__(self, **parts):
        self.__dict__.update(parts)
        self.over_in_env_step = 0
        self.closed = False
        self.block: Optional[dict] = None
        self.runner_inputs_check: Optional[dict] = None
        self.digest_wall_s = 0.0

    # -- the run ----------------------------------------------------------------------------------------------------
    def run(self):
        return self.planner.run(self.conn)

    def env_step(self, a):
        """DirectConnector's env_step under owner ``rt``; an EpisodeOver raised inside is U0-a's X."""
        with self.ledger.owner("rt"):
            try:
                return self.bench_host.env_step(a)
            except EpisodeOver:
                self.over_in_env_step += 1
                raise

    def digest(self) -> dict:
        t0 = time.monotonic()
        try:
            return self.state_digest()
        finally:
            self.digest_wall_s += time.monotonic() - t0

    def writes(self) -> int:
        return sum(int(r.writes) for r in self.ledger.rows.values())

    def factory(self, name: str, goal, **kwargs):
        """strategy_for, with the Runner's construction inputs compared against the bench's strategy at once."""
        runner = self.strategy_for(name, goal, **kwargs)
        got, want = runner_inputs(runner), runner_inputs(self.host_strategy)
        equal = got == want
        self.runner_inputs_check = {"equal": equal, "planner": _plain(got), "host": _plain(want)}
        if not equal:
            log.error(f"Runner inputs differ between the pseudo planner and the host's strategy: {got} != {want}")
        return runner

    # -- the verdicts -------------------------------------------------------------------------------------------------
    def _live(self) -> dict:
        return {cid: int(run.steps) for cid, run in self.rt.runs.items()}

    def u0b(self, rows: dict, totals: dict) -> dict:
        def steps(owner: str) -> int:
            return int(rows.get(owner, {}).get("steps", 0))

        checks = {
            "unowned_steps": totals["unowned_steps"] == 0,
            "unowned_env_step_calls": totals["unowned_env_step_calls"] == 0,
            "unowned_writes": totals["unowned_writes"] == 0,
            "rt_equals_rt_step": steps("rt") == int(self.rt.step),
            "observe_equals_observer_steps": steps("observe") == int(getattr(self.observer, "steps", 0)),
            "go_to_equals_teleport_steps": steps("go_to") == int(getattr(self.teleport, "steps", 0)),
            "host_build_zero": steps("host.build") == 0 and int(rows.get("host.build", {}).get("env_step_calls", 0)) == 0,
            "refresh_zero": steps("refresh") == 0,
            "sum_owners_equals_sim": totals["owned_steps"] == int(self.sim.n_steps) - self.n0,
            "ledger_n0_equals_n0": int(self.ledger.n0) == self.n0,
            "env_step_calls_equal_deltas": all(int(r.get("env_step_calls", 0)) == int(r.get("steps", 0))
                                               for r in rows.values()),
            "closed_zero": int(totals["closed_env_step_calls"]) == 0,
        }
        return {"ok": all(checks.values()), "checks": checks, "totals": dict(totals),
                "observer_steps": int(getattr(self.observer, "steps", 0)),
                "teleport_steps": int(getattr(self.teleport, "steps", 0)),
                "navigator_steps": int(getattr(self.navigator, "steps", 0)),
                "legacy_backend_steps": int(getattr(self.legacy, "steps", 0)), "n0": self.n0,
                "n_steps": int(self.sim.n_steps)}

    def u0c(self) -> dict:
        """The Runner holds a ClockView, never R1ProSim: the object it holds unwraps (TapeRecorder._inner,
        DualEpisode._shim) to the shim, whose ``sim`` is the ClockView."""
        shim = getattr(self.planner, "shim", None)
        view = getattr(shim, "sim", None) if shim is not None else None
        chain, held = [], self.runner_holds
        while held is not None and len(chain) < 8:
            chain.append(type(held).__name__)
            inner = held.__dict__ if hasattr(held, "__dict__") else {}
            held = inner.get("_inner", inner.get("_shim"))
        ok = shim is not None and isinstance(view, ClockView) and view is not getattr(self.episode, "sim", None)
        return {"ok": bool(ok), "clock_view": type(view).__name__ if view is not None else None, "chain": chain}

    def rule2(self, rows: dict) -> dict:
        rt = rows.get("rt", {})
        during_native = {k: int(rt.get(k, 0)) for k in ("place_robot", "capture", "look_at")}
        debt = Counter()
        for owner, r in rows.items():
            if owner.startswith(EP_PREFIX):
                for k in ("place_robot", "capture", "look_at"):
                    debt[k] += int(r.get(k, 0))
        return {"ok": sum(during_native.values()) == 0, "rt": during_native, "d21_debt": dict(debt)}

    def requests(self) -> Optional[dict]:
        ws = self.wstape
        if ws is None:
            return None
        by = Counter()
        for row in getattr(ws, "frames", ()):
            op, owner = row.get("op", "?"), row.get("owner", "unowned")
            by[f"{op}/{owner}"] += 1
        counted = [(k.split("/", 1)[0], k.split("/", 1)[1], n) for k, n in by.items()]
        requests = [(op, owner, n) for op, owner, n in counted if op not in PLANNER_FRAMES_IGNORED]
        skill = sum(n for op, _, n in requests if op == "skill")
        reach = sum(n for op, _, n in requests if op == "reach")
        outside = sum(n for op, owner, n in requests if not owner.startswith(EP_PREFIX))
        other = sum(n for op, _, n in requests if op not in REQUEST_OPS_ALLOWED)
        return {"by_type_and_owner": dict(by), "requests": sum(n for _, _, n in requests), "skill": skill,
                "reach": reach, "not_owned_by_ep": outside, "not_plan_or_move": other,
                "ok": skill == 0 and reach == 0 and outside == 0 and other == 0, "mode": getattr(ws, "mode", None)}

    # -- close --------------------------------------------------------------------------------------------------------
    def close(self, reason: Optional[str] = None, raised: Optional[BaseException] = None) -> dict:
        """The verdicts and the block; the jsonl streams in the instance dir. Never raises."""
        if self.closed and self.block is not None:
            return self.block
        self.closed = True
        try:
            block = self._close(reason, raised)
        except BaseException as e:  # noqa: BLE001 - a broken measurement must not cost the instance its result
            log.exception("episode_host.close failed")
            block = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                     "reason": reason, "collision": COLLISION}
        finally:
            try:
                skillrun.PASSTHROUGH = self.saved_passthrough
            except Exception:  # noqa: BLE001
                pass
        self.block = block
        try:
            self._stream(block)
        except Exception as e:  # noqa: BLE001
            log.exception("episode_host: streaming the jsonl files failed")
            block["stream_error"] = f"{type(e).__name__}: {e}"
        return block

    def _close(self, reason, raised) -> dict:
        episode_over = isinstance(raised, EpisodeOver)
        rows, totals = self.ledger.owners, self.ledger.totals()
        live = self._live()
        a = u0a(self.rt, sum(live.values()), self.over_in_env_step, episode_over)
        b = self.u0b(rows, totals)
        c = self.u0c()
        audit_summary, audit_violations = self.audit.verdict(self.n0)
        d = {"ok": not audit_violations, "summary": audit_summary, "violations": audit_violations[:50]}
        r2 = self.rule2(rows)
        build = {"steps": self.build_steps, "digest_changed": self.build_digest_changed,
                 "digest_diff": self.build_digest_diff, "ok": self.build_steps == 0 and not self.build_digest_changed}
        purity = dual = None
        if self.purity is not None:
            ps, pv = self.purity.verdict(self.ledger)
            purity = {"ok": not pv, "summary": ps, "violations": pv[:50]}
        if self.dual is not None:
            ds = self.dual.summary()
            dual = {"ok": ds["mismatches"] == 0, "summary": ds, "mismatches": _plain(self.dual.mismatches[:50])}
        renders = {owner: int(r.get("renders", 0)) for owner, r in rows.items() if r.get("renders")}
        renders_outside = sum(n for owner, n in renders.items() if not owner.startswith(EP_PREFIX))
        req = self.requests()
        inputs = self.runner_inputs_check or {"equal": False, "planner": None, "host": _plain(runner_inputs(self.host_strategy)),
                                              "note": "the planner never built its Runner"}
        if getattr(self.planner, "runner_inputs", None) is not None:
            want = runner_inputs(self.host_strategy)
            want_t = (want["goal"], want["options_sha256"], want["n_options"], want["scope"], want["attempts"], want["spec"])
            inputs["planner_run"] = _plain(self.planner.runner_inputs)
            inputs["equal_at_run"] = tuple(self.planner.runner_inputs) == want_t
            inputs["equal"] = bool(inputs["equal"] and inputs["equal_at_run"])
        calls = Counter(f"{r.get('skill')}/{r.get('backend')}/{r.get('status')}/{r.get('code')}" for r in self.rt.log)
        live_native = [(cid, n) for cid, n in live.items() if n > 0]
        g3 = {
            "dual_mismatches": None if dual is None else dual["summary"]["mismatches"],
            "purity_violations": None if purity is None else purity["summary"]["violations"],
            "typed_literal_mismatches": len(self.channel.mismatches),
            "renders_outside_ep": renders_outside,
            "requests_ok": None if req is None else req["ok"],
            "runner_inputs_equal": bool(inputs["equal"]),
            "u0a": a["ok"], "u0b": b["ok"], "u0c": c["ok"], "u0d": d["ok"], "rule2": r2["ok"], "build": build["ok"],
        }
        hard = [g3["u0a"], g3["u0b"], g3["u0c"], g3["u0d"], g3["rule2"], g3["build"], g3["runner_inputs_equal"],
                g3["typed_literal_mismatches"] == 0, g3["renders_outside_ep"] == 0]
        if dual is not None:
            hard.append(dual["ok"])
        if purity is not None:
            hard.append(purity["ok"])
        if req is not None:
            hard.append(req["ok"])
        g3["pass"] = all(hard)
        shim = getattr(self.planner, "shim", None)
        return {
            "ok": g3["pass"],
            "reason": reason,
            "raised": None if raised is None else f"{type(raised).__name__}: {raised}",
            "collision": COLLISION,
            "routing_profile": getattr(self.args, "routing_profile", "parity"),
            "routes": list(getattr(self.args, "route", ()) or ()),
            "hands": "episode", "scorer_scope_only": True, "shadow": bool(getattr(self.args, "shadow", False)),
            "audit_enabled": self.purity is not None,
            "step": int(self.rt.step), "idle_steps": int(self.rt.idle_steps), "charged": dict(self.rt.charged),
            "live_runs": live,
            "live_at_end": {"call_id": live_native[0][0], "steps": live_native[0][1]} if live_native else None,
            "over_in_env_step": self.over_in_env_step,
            "planner_oracle_reads": dict(self.rt.planner_oracle_reads),
            "epochs": {"base": self.rt.base_epoch, "trunk": self.rt.trunk_epoch},
            "ledger": _plain(rows),
            "ledger_totals": _plain(totals),
            "ledger_calls": [c.to_dict() for c in self.ledger.calls],
            "u0a": a, "u0b": b, "u0c": c, "u0d": d, "rule2": r2, "build": build,
            "purity": purity, "dual": dual,
            "mismatches": _plain(self.channel.mismatches),
            "channel_exceptions": dict(self.channel.exc_counts),
            "calls": dict(calls),
            "advisory": dict(self.registry.advisory),
            "requests": req,
            "renders": renders,
            "hand_refresh": {"count": 0, "popped": []},
            "runner_inputs": inputs,
            "shim": None if shim is None else shim_counts(shim),
            "world_disagreements": int(getattr(self.svc.world, "disagreements", 0) or 0),
            "wall": {"digest_s": round(self.digest_wall_s, 2), "env_s": round(getattr(self.bench_host, "env_wall_s", 0.0), 2),
                     "frames_s": round(getattr(self.bench_host, "frames_wall_s", 0.0), 2),
                     "observe_s": round(getattr(self.observer, "wall_s", 0.0), 2)},
            "g3": g3,
        }

    def _stream(self, block: dict) -> None:
        out = self.inst_dir
        if out is None:
            return
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "skill_calls.jsonl", "w") as f:
            for row in self.rt.log:
                f.write(json.dumps(row, default=_default) + "\n")
        with open(out / "goal_checks.jsonl", "w") as f:
            for row in getattr(self.svc.goals, "log", ()):
                f.write(json.dumps(row, default=_default) + "\n")
        with open(out / "ledger.jsonl", "w") as f:
            for row in self.ledger.owners.values():
                f.write(json.dumps(row) + "\n")
        with open(out / "ledger_calls.jsonl", "w") as f:
            for c in self.ledger.calls:
                f.write(json.dumps(c.to_dict()) + "\n")
        with open(out / "audit.jsonl", "w") as f:
            for op, n_in, n_out in self.audit.rows:
                f.write(json.dumps({"kind": "op", "op": op, "sim_in": n_in, "sim_out": n_out}) + "\n")
            if self.purity is not None:
                for row in self.purity.rows:
                    f.write(json.dumps(row, default=_default) + "\n")
            if self.dual is not None:
                for row in self.dual.rows:
                    f.write(json.dumps(row, default=_default) + "\n")
        with open(out / "connector.json", "w") as f:
            json.dump(block, f, indent=1, default=_default)


def build(episode, sim, planners: dict, args, providers, strategy_for: Callable, host_strategy, ledger, watch, *,
          tape: Optional[Callable] = None, inst_dir=None, max_steps: Optional[int] = None, wstape=None,
          planner_client=None, bench_host=None, observer=None, teleport=None, make_backends: Optional[Callable] = None,
          knowledge=None) -> EpisodeHost:
    """See the module docstring. ``tape``: a factory (what the Runner holds -> the TapeRecorder around it) when
    --runner-tape is given; ``wstape``: the websocket tape (its frames give the planner requests by type and owner);
    the keyword seats (``planner_client``, ``bench_host``, ``observer``, ``teleport``, ``make_backends``) default to
    the bench's pieces and let a test put fakes in them."""
    knowledge = episode.knowledge if knowledge is None else knowledge
    state_digest = lambda: watch.state_digest(sim, knowledge)  # noqa: E731
    n0, digest0 = int(sim.n_steps), state_digest()
    max_steps = int(getattr(sim, "max_steps", None) or 0) if max_steps is None else max_steps
    saved = skillrun.PASSTHROUGH
    with ledger.owner("host.build"):
        routing = routing_for(args)
        client = planner_client_of(planners) if planner_client is None else planner_client
        svc, segmenter = providers.pseudo_services(
            episode, client, routing, collision=COLLISION, hands="episode", scorer_scope_only=True,
            shadows=None if getattr(args, "shadow", False) else [],
        )
        # the Runtime applies a result's WorldUpdates (_finish) with no owner in flight; OracleWorld(hands="episode")
        # writes the hand record there, idempotently, so the write-call is owned by ``apply`` and never unowned
        world_apply = svc.world.apply
        svc.world.apply = lambda u: _owned_call(ledger, "apply", world_apply, u)
        bench_host = BenchHost(sim, segmenter, sim_clock=True) if bench_host is None else bench_host
        if observer is None:
            observer = CaptureObserver(sim, bench_host, segmenter, getattr(args, "task", getattr(args, "task_name", "")))
        teleport = TeleportNavigator(sim) if teleport is None else teleport
        channel = LegacyChannel()
        legacy = EpisodeLegacyBackend(episode, channel, bench_host.observe_now)
        if make_backends is None:
            from omnigibson.tiptop.host.skillbench import make_backends as skillbench_backends

            make_backends = skillbench_backends
        backends = dict(make_backends(episode, bench_host, svc))
        backends["legacy"] = legacy
        for backend in backends.values():
            backend.run = call_id_generator(ledger, backend.run)
        registry = EpisodeRegistry(EPISODE_SPECS, backends, routing)
        navigator = EpisodeNavigator(episode, channel, teleport, base_pose=svc.world.base_pose)
        navigator.go_to = owned_generator(ledger, "go_to", navigator.go_to)
        observer.observe = owned_generator(ledger, "observe", observer.observe)
        calls: list = []
        rt = Runtime(registry, svc, observer=observer, navigator=navigator,
                     task_info=lambda: providers.task_info(sim, planners, max_steps, args.task_name),
                     host=bench_host, log=calls)
        skillrun.PASSTHROUGH = (EpisodeOver,)
        host = EpisodeHost(
            episode=episode, sim=sim, planners=planners, args=args, providers=providers, strategy_for=strategy_for,
            host_strategy=host_strategy, ledger=ledger, watch=watch, state_digest=state_digest, n0=n0,
            digest0=digest0, svc=svc, segmenter=segmenter, bench_host=bench_host, observer=observer,
            teleport=teleport, channel=channel, legacy=legacy, backends=backends, registry=registry,
            navigator=navigator, rt=rt, calls=calls, saved_passthrough=saved, inst_dir=inst_dir, wstape=wstape,
            max_steps=max_steps, purity=None, dual=None, runner_holds=None,
        )
        direct = DirectConnector(rt, host.env_step, bench_host, bench_host.raw())
        audit = AuditedConnector(direct, lambda: int(sim.n_steps))
        conn: Any = audit
        if getattr(args, "audit", False):
            host.purity = PurityAudit(audit, host.digest, host.writes)
            conn = host.purity

        def runner_episode(shim):  # what the Runner holds: [TapeRecorder(] [DualEpisode(] shim [)] [)]
            ep = shim
            if getattr(args, "audit", False):
                host.dual = ep = DualEpisode(shim, episode, host.purity)
            if tape is not None:
                ep = tape(ep)
            host.runner_holds = ep
            return ep

        host.direct, host.audit, host.conn = direct, audit, conn
        host.planner = PseudoPlanner(factory=host.factory, channel=channel, attempts=args.attempts_per_item,
                                     tape=runner_episode, views=tuple(getattr(args, "views", ("head",)) or ("head",)))
    digest1 = state_digest()
    host.build_steps = int(sim.n_steps) - n0
    host.build_digest_changed = digest1 != digest0
    host.build_digest_diff = sorted(k for k in set(digest0) | set(digest1) if digest0.get(k) != digest1.get(k))
    log.info(
        f"episode host built: profile {getattr(args, 'routing_profile', 'parity')} routes "
        f"{list(getattr(args, 'route', ()) or ())}, collision {COLLISION}, hands episode, audit "
        f"{bool(getattr(args, 'audit', False))}, backends {sorted(backends)}; build steps {host.build_steps}, digest "
        f"{'changed at ' + str(host.build_digest_diff) if host.build_digest_changed else 'unchanged'}"
    )
    return host
