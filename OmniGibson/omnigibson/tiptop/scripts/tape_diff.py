"""Two runs' tapes side by side (WEEK4_PLAN 5.5, W4-A2): where they stopped being the same episode.

    python OmniGibson/omnigibson/tiptop/scripts/tape_diff.py A B [--json]
    python OmniGibson/omnigibson/tiptop/scripts/tape_diff.py --runner-tape A.json B.json --wstape A_dir B_dir
    python OmniGibson/omnigibson/tiptop/scripts/tape_diff.py e0 TAPE.json [--json]

``A`` and ``B`` are bench out dirs (``--out-dir``): the Runner tapes are found under ``tapes/*.json``, the request
stream under ``wstape/`` (a record or a log) or ``wstape_replay.jsonl`` (a replay's own log, which names its diffs
against the tape it was served from). Reports, for the Runner tapes: the record counts, the common prefix length,
the first divergence (index, kind: decision | answer | diagnostic, the two records) and, write by write, the step
delta and whether the digest before and after each write is the same in both; for the request stream: the frame
counts, the common prefix (frames whose op and non-VOLATILE request digests agree), and the first differing frame
with its fields. A replay log is compared as served: its ``matched`` frames are the prefix.

``e0`` replays a Runner tape offline (WEEK4_PLAN 5.2 E0): the same Runner, built from the tape's header (the task,
the goal, the options and the scope the bench recorded), run against a TapeEpisode over the tape. Pass: every write
consumed in order, no read the tape could not answer, and the run ends the way the bench's did (the header's
``ending``: finished, or raised the recorded exception's class).

Needs no simulator; b1k.planner.pseudo.tape for the codec and the diff, b1k.bridge.strategies for e0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for root in (HERE.parents[4] / "tiptop", HERE.parents[3]):  # the checkout's tiptop and OmniGibson, if not on the path
    if root.is_dir() and str(root) not in sys.path:
        sys.path.append(str(root))

from b1k.planner.pseudo import tape as tp  # noqa: E402

VOLATILE = ("rgb", "views[*].rgb")


# ------------------------------------------------------------------------------------------------- the Runner tape
def find_runner_tape(run: Path) -> Path | None:
    if run.is_file():
        return run
    tapes = sorted((run / "tapes").glob("*.json")) if (run / "tapes").is_dir() else sorted(run.glob("*.tape.json"))
    return tapes[0] if tapes else None


def runner_report(a: Path, b: Path) -> dict:
    ta, tb = tp.Tape.load(a), tp.Tape.load(b)
    d = tp.diff(ta, tb)
    prefix = len(ta.records) if d is None else d.index
    writes = []
    for i, (wa, wb) in enumerate(zip(ta.writes, tb.writes)):
        same_call = (wa["member"], wa["args"], wa["kwargs"]) == (wb["member"], wb["args"], wb["kwargs"])
        sa, sb = wa.get("step") or [None, None], wb.get("step") or [None, None]
        da, db = wa.get("digest") or [None, None], wb.get("digest") or [None, None]
        writes.append(
            {
                "i": i,
                "member": wa["member"] if same_call else f"{wa['member']} | {wb['member']}",
                "same_call": same_call,
                "step_a": sa,
                "step_b": sb,
                "delta_a": None if None in sa else sa[1] - sa[0],
                "delta_b": None if None in sb else sb[1] - sb[0],
                "digest_before_equal": _digests_equal(da[0], db[0]),
                "digest_after_equal": _digests_equal(da[1], db[1]),
                "digest_keys_one_side": _one_sided(da, db),
                "outcome_equal": tp.dumps({k: wa.get(k) for k in ("ret", "exc")})
                == tp.dumps({k: wb.get(k) for k in ("ret", "exc")}),
            }
        )
    return {
        "a": str(a),
        "b": str(b),
        "records": [len(ta.records), len(tb.records)],
        "writes": [len(ta.writes), len(tb.writes)],
        "prefix": prefix,
        "identical": d is None,
        "first_divergence": None if d is None else {"index": d.index, "kind": d.kind, "a": d.a, "b": d.b},
        "per_write": writes,
        "headers_equal": tp.dumps({k: v for k, v in ta.header.items() if k != "provenance"})
        == tp.dumps({k: v for k, v in tb.header.items() if k != "provenance"}),
    }


def _digests_equal(a, b) -> bool:
    """Two state digests on the keys both carry: a tape recorded before a key was added (the joints, the fix pass)
    compares with a newer one on what both measured; a key one side lacks is reported, not a divergence."""
    if isinstance(a, dict) and isinstance(b, dict):
        return all(tp.dumps(a[k]) == tp.dumps(b[k]) for k in set(a) & set(b))
    return tp.dumps(a) == tp.dumps(b)


def _one_sided(da, db) -> list:
    keys = set()
    for a, b in zip(da, db):
        if isinstance(a, dict) and isinstance(b, dict):
            keys |= set(a) ^ set(b)
    return sorted(keys)


# ------------------------------------------------------------------------------------------------- the request stream
def _rows(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_stream(run: Path) -> tuple[str, Path] | None:
    """(kind, path): a record or log's index.jsonl, or a replay's own log."""
    if run.is_file():
        return ("replay" if run.name.startswith("wstape_replay") else "index", run)
    if (run / "wstape" / "index.jsonl").exists():
        return "index", run / "wstape" / "index.jsonl"
    if (run / "wstape_replay.jsonl").exists():
        return "replay", run / "wstape_replay.jsonl"
    if (run / "index.jsonl").exists():
        return "index", run / "index.jsonl"
    return None


def _volatile(path: str) -> bool:
    """``rgb`` and ``views[i].rgb`` (VOLATILE, as wstape.is_volatile reads it): the digests never compared."""
    return re.sub(r"\[\d+\]", "[*]", path) in VOLATILE


def _fields(row: dict) -> dict:
    return {k: v for k, v in (row.get("fields") or {}).items() if not _volatile(k)}


def _on_tape(replay_log: Path) -> int | None:
    """How many frames the tape a replay was served from holds: the run's result JSON (bench.instruments.wstape
    .on_tape, written by WsTape.summary), else None."""
    for js in sorted((replay_log.parent / "json").glob("*.json")) if (replay_log.parent / "json").is_dir() else ():
        try:
            ws = ((json.loads(js.read_text()).get("bench") or {}).get("instruments") or {}).get("wstape") or {}
        except (OSError, ValueError):
            continue
        if ws.get("on_tape") is not None:
            return int(ws["on_tape"])
    return None


def _replay_side(rows: list, on_tape: int | None) -> dict:
    """One replay log as served: its matched prefix, its first frame that did not match (or, all matched, the end
    of a stream shorter than its tape), the mismatched frames and the frame it went live at."""
    matched = 0
    for row in rows:
        if not row.get("matched", False) or row.get("live"):
            break
        matched += 1
    first = next((r for r in rows if not r.get("matched", False) or r.get("live")), None)
    div = None
    if first is not None:
        div = {
            "frame": first["i"],
            "op": first["op"],
            "tape_op": first.get("tape_op"),
            "fields": [d["path"] for d in first.get("diffs", [])] or (["<forced live>"] if first.get("live") else []),
            "diffs": first.get("diffs", []),
        }
    elif on_tape is not None and matched < on_tape:
        div = {"frame": matched, "op": "<end>", "tape_op": None, "fields": ["<end of stream>"], "diffs": []}
    return {
        "prefix": matched,
        "first_divergence": div,
        "on_tape": on_tape,
        "mismatched_frames": [r["i"] for r in rows if not r.get("matched", False)],
        "live_from": next((r["i"] for r in rows if r.get("live")), None),
    }


def stream_report(a: tuple, b: tuple) -> dict:
    (ka, pa), (kb, pb) = a, b
    ra, rb = _rows(pa), _rows(pb)
    out = {"a": str(pa), "b": str(pb), "kind": [ka, kb], "frames": [len(ra), len(rb)]}
    if "replay" in (ka, kb):  # a replay log names its own diffs against the tape it was served from
        sides = {}
        for key, kind, rows, path, other in (("a", ka, ra, pa, rb), ("b", kb, rb, pb, ra)):
            if kind == "replay":  # served from a tape: the other side, when it is that tape's index, or its JSON
                n = len(other) if (ka, kb).count("replay") == 1 else _on_tape(path)
                sides[key] = _replay_side(rows, n)
        rep = [sides[k] for k in ("a", "b") if k in sides]
        first = min(
            (s["first_divergence"] for s in rep if s["first_divergence"] is not None),
            key=lambda d: d["frame"],
            default=None,
        )
        out.update(
            {
                "prefix": min(s["prefix"] for s in rep),  # every side matched its tape this far
                "first_divergence": first,
                "mismatched_frames": sorted({i for s in rep for i in s["mismatched_frames"]}),
                "live_from": min((s["live_from"] for s in rep if s["live_from"] is not None), default=None),
                "replays": sides,
            }
        )
        return out
    prefix, first = 0, None
    for i, (x, y) in enumerate(zip(ra, rb)):
        fx, fy = _fields(x), _fields(y)
        differing = [k for k in fx if k not in fy or fx[k] != fy[k]] + [k for k in fy if k not in fx]
        if x["op"] != y["op"]:
            differing = ["op"] + differing
        if x.get("seed") != y.get("seed"):
            differing.append("seed")
        if differing:
            first = {
                "frame": i,
                "op": [x["op"], y["op"]],
                "fields": differing,
                "seed": [x.get("seed"), y.get("seed")],
                "owner": [x.get("owner"), y.get("owner")],
            }
            break
        prefix += 1
    if first is None and len(ra) != len(rb):
        first = {
            "frame": min(len(ra), len(rb)),
            "op": [
                "<end>" if len(ra) <= len(rb) else ra[len(rb)]["op"],
                "<end>" if len(rb) <= len(ra) else rb[len(ra)]["op"],
            ],
            "fields": ["<end of stream>"],
        }
    out.update(
        {
            "prefix": prefix,
            "first_divergence": first,
            "seeds": [[r.get("seed") for r in ra if r.get("seed")], [r.get("seed") for r in rb if r.get("seed")]],
            "responses_equal_in_prefix": [
                x.get("response_sha") == y.get("response_sha") for x, y in zip(ra[:prefix], rb[:prefix])
            ],
        }
    )
    return out


# ------------------------------------------------------------------------------------------------- e0
def e0(tape_path: Path) -> dict:
    """Replay the Runner from the tape's header against a TapeEpisode over the tape."""
    from b1k.bridge import strategies

    tape = tp.Tape.load(tape_path)
    h = tape.header
    inputs = h.get("inputs") or {}
    options = h.get("options")
    if options is None:
        raise SystemExit(
            f"{tape_path}: the header carries no options (the bench writes them); e0 cannot build the Runner"
        )
    runner = strategies.strategy_for(
        h["task"],
        inputs["goal"],
        options=options,
        attempts=inputs.get("attempts"),
        scope=h.get("scope") or inputs.get("scope") or (),
    )
    exc_classes = [strategies.Unreachable, strategies.TransferBlocked]
    try:
        from omnigibson.tiptop.scene import EpisodeOver

        exc_classes.append(EpisodeOver)
    except Exception:  # noqa: BLE001 - no OmniGibson on the path: EpisodeOver is imported by name at replay
        pass
    try:
        from omnigibson.tiptop.host.wstape import TapeDiverged

        exc_classes.append(TapeDiverged)
    except Exception:  # noqa: BLE001
        pass
    offline = tp.TapeEpisode(tape, exc_classes=tuple(exc_classes))
    ending = None
    try:
        runner.run(offline)
    except BaseException as e:  # noqa: BLE001 - the recorded ending is expected to come back
        ending = f"{type(e).__name__}: {e}"
    if "ending" in h:  # what the bench saw strategy.run do
        raised = h["ending"].get("raised")
        recorded_ending = None if raised is None else f"{raised.type}: {raised.message}"
    else:  # an older tape: the last write's exception, if it raised one
        last = tape.writes[-1] if tape.writes else None
        recorded_ending = None if last is None or "exc" not in last else f"{last['exc'].type}: {last['exc'].message}"
    consumed, total = offline.segment, len(tape.writes)
    same_ending = (ending is None and recorded_ending is None) or (
        ending is not None and recorded_ending is not None and ending.split(":")[0] == recorded_ending.split(":")[0]
    )
    return {
        "tape": str(tape_path),
        "task": h.get("task"),
        "instance": h.get("instance"),
        "records": len(tape.records),
        "writes_consumed": [consumed, total],
        "unanswerable": [list(u) for u in offline.unanswerable],
        "ending": ending,
        "recorded_ending": recorded_ending,
        "bench_reason": (h.get("ending") or {}).get("reason"),
        "pass": consumed == total and not offline.unanswerable and same_ending,
    }


# ------------------------------------------------------------------------------------------------- main
def _print_runner(r: dict) -> None:
    print(
        f"runner tape: {r['records'][0]} vs {r['records'][1]} records, {r['writes'][0]} vs {r['writes'][1]} writes; "
        f"common prefix {r['prefix']} records; headers {'equal' if r['headers_equal'] else 'DIFFER'}"
    )
    d = r["first_divergence"]
    if d is None:
        print("  identical to the end")
    else:
        print(f"  first divergence at record {d['index']} ({d['kind']}):")
        print(f"    a: {tp.dumps(d['a'])[:300]}")
        print(f"    b: {tp.dumps(d['b'])[:300]}")
    print("  per write: i member delta_a delta_b digest_before digest_after outcome")
    for w in r["per_write"]:
        print(
            f"    {w['i']:3d} {w['member']:14s} {str(w['delta_a']):>6s} {str(w['delta_b']):>6s} "
            f"{'=' if w['digest_before_equal'] else 'X':>13s} {'=' if w['digest_after_equal'] else 'X':>12s} "
            f"{'=' if w['outcome_equal'] else 'X':>7s}"
        )


def _print_stream(s: dict) -> None:
    print(
        f"request stream ({s['kind'][0]} vs {s['kind'][1]}): {s['frames'][0]} vs {s['frames'][1]} frames; "
        f"common prefix {s['prefix']}"
    )
    d = s["first_divergence"]
    if d is None:
        print("  identical to the end")
    else:
        print(f"  first divergence at frame {d['frame']} ({d['op']}): {d['fields']}")
        for diff in d.get("diffs", [])[:20]:
            print(f"    {diff['path']}: {diff['detail']}")
    if "seeds" in s:
        print(f"  seeds a {s['seeds'][0]}\n  seeds b {s['seeds'][1]}")
    if "mismatched_frames" in s:
        print(f"  mismatched frames {s['mismatched_frames']}; live from {s['live_from']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="A B (bench out dirs), or: e0 TAPE.json")
    p.add_argument("--runner-tape", nargs=2, metavar=("A", "B"))
    p.add_argument("--wstape", nargs=2, metavar=("A", "B"))
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    if args.runs and args.runs[0] == "e0":
        report = e0(Path(args.runs[1]))
        if args.json:
            print(json.dumps(report, indent=1, default=str))
        else:
            c, t = report["writes_consumed"]
            print(
                f"e0 {report['tape']}: {report['task']} instance {report['instance']}; {report['records']} records; "
                f"writes consumed {c}/{t}; unanswerable {len(report['unanswerable'])}; ending {report['ending']!r} "
                f"(recorded {report['recorded_ending']!r}); {'PASS' if report['pass'] else 'FAIL'}"
            )
            for u in report["unanswerable"][:20]:
                print(f"  unanswerable: {u}")
        return 0 if report["pass"] else 1
    out = {}
    a, b = (Path(x) for x in args.runs) if len(args.runs) == 2 else (None, None)
    rt = args.runner_tape or ([find_runner_tape(a), find_runner_tape(b)] if a else None)
    if rt and all(rt):
        out["runner"] = runner_report(Path(rt[0]), Path(rt[1]))
    ws = [(None, Path(x)) for x in args.wstape] if args.wstape else ([find_stream(a), find_stream(b)] if a else None)
    if ws and all(ws):
        ws = [find_stream(Path(pth)) if kind is None else (kind, pth) for kind, pth in ws]
        if all(ws):
            out["stream"] = stream_report(ws[0], ws[1])
    if not out:
        p.error("nothing to compare: give A B out dirs, or --runner-tape / --wstape pairs")
    if args.json:
        print(json.dumps(out, indent=1, default=str))
    else:
        if "runner" in out:
            _print_runner(out["runner"])
        if "stream" in out:
            _print_stream(out["stream"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
