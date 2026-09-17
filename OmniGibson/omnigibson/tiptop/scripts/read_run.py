"""Read a bench run: the score per instance, why rounds were lost, and the counts the capture reports itself.

    python OmniGibson/omnigibson/tiptop/scripts/read_run.py runs/bench_batteries_8 [the run's log]

Needs no simulator and no GPU: it reads ``summary.json`` and, when given the log, counts the lines the pipeline
writes about its own trouble (blocked capture swings, an arm on the head camera's line of sight, a view that was
all robot, a stance that framed nothing whole) and prints where each object the capture could not see actually
was. Written on 2026-09-13 so every run is read the same way instead of by grepping afresh.

Every round's saved request and response are read too (``<instance>/rNN_*/capture.json`` and
``server_response.json``): which labels the request asked for and which the planner's perception reported back.
Under the oracle the two agree by construction; under ``--knowledge onboard`` the difference is the detector's
recall on the goal objects, which is the number that says whether onboard perception can carry a task at all."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
log = Path(sys.argv[2]) if len(sys.argv) > 2 else None
BASELINES = Path(__file__).resolve().parent.parent / "baselines.json"


def detected(run: Path) -> None:
    """Per round: the labels the request asked the planner for, the goal's own objects among them, and what the
    planner's perception reported back (its ``objects``, or the label its detector could not find)."""
    rows = []
    for capture in sorted(run.glob("*/r*/capture.json")):
        meta = json.loads(capture.read_text())
        asked = list(meta.get("knowledge", {}).get("labels") or [])
        goal = sorted({a for atom in meta.get("goal_atoms") or [] for a in atom.get("args", [])})
        goal = [g for g in goal if g in asked]  # the support plane and a button are named, never detected
        response = capture.with_name("server_response.json")
        if not response.exists():
            rows.append((capture.parent.name, asked, goal, None, "no response saved"))
            continue
        resp = json.loads(response.read_text())
        found = sorted((resp.get("objects") or {}).keys())
        error = resp.get("error") or ""
        note = ""
        if "did not find" in error:
            note = re.search(r"did not find '([^']+)'", error).group(1) + " not found"
            # what the detector DID find is only in the error text: "(found: ['a', 'b'])"
            m = re.search(r"\(found: \[([^\]]*)\]\)", error)
            if m and not found:
                found = sorted(x.strip().strip("'") for x in m.group(1).split(",") if x.strip())
        rows.append((capture.parent.name, asked, goal, found, note))
    if not rows:
        return
    print("\n  what the planner's perception reported, per round (asked -> found; goal objects marked *):")
    hits = misses = 0
    for name, asked, goal, found, note in rows:
        if found is None:
            print(f"    {name:22s} asked {asked}: {note}")
            continue
        seen = {f.rsplit("_", 1)[0] if f.rsplit("_", 1)[-1].isdigit() else f for f in found}  # can_2 -> can
        got = [g for g in goal if g in seen or g in found]
        lost = [g for g in goal if g not in got]
        hits += not lost
        misses += bool(lost)
        shown = ", ".join(("*" if a in goal else "") + a for a in asked)
        print(f"    {name:22s} asked [{shown}] -> found {found}" + (f"   MISSING {lost}" if lost else "") + (f"  ({note})" if note else ""))
    if hits + misses:
        print(f"    goal objects all found in {hits} of {hits + misses} rounds with a response ({100 * hits / (hits + misses):.0f}%)")

summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
if summary:
    print(
        f"{summary['task']}: mean {summary['mean_q_score']:.4f} over {summary['instances']} instance(s), "
        f"{summary['successes']} complete"
    )
    known = json.loads(BASELINES.read_text()).get(summary["task"]) if BASELINES.exists() else None
    if known:
        gap = summary["mean_q_score"] - float(known["mean"])
        verdict = "BETTER than" if gap > 1e-9 else ("WORSE than" if gap < -1e-9 else "level with")
        same = int(known.get("instance_count", -1)) == int(summary["instances"])
        if known.get("stale"):
            print(f"    NOTE: that baseline is marked stale -- {known['note'][:150]}")
        print(
            f"  {verdict} the best on record ({float(known['mean']):.4f}, {known['run']}, "
            f"instances {known['instances'] or '-'})"
            + (f": {gap:+.4f}" if abs(gap) > 1e-9 else "")
            + ("" if same else "   [different instances: not like for like]")
        )
    for p in summary["per_instance"]:
        print(f"  {p['instance_id']}: {p['q_score']:.4f}  teleports {p['teleports']}  {p['wall_time_s']:.0f}s")
        print(f"      {p['what_failed']}")
else:
    print(f"{run}: no summary.json (the run did not finish)")
detected(run)

if log and log.exists():
    text = log.read_text(encoding="utf-8", errors="replace")
    counts = Counter()
    for name, pattern in {
        "rounds executed": r"round \d+ .*: executed",
        "rounds lost to empty masks": r"\[\w+\]: GoalNotVisible",
        "rounds lost to planning": r"\[\w+\]: TiptopPlanningError",
        # a goal object the planner's own detector could not find (onboard knowledge sends no masks): the same
        # loss as empty masks, reported by the other side of the wire. Counted inside "rounds lost to planning"
        # too, since the client sees it as a planning error
        "  of which the detector found no goal object": r"\[\w+\]: TiptopPlanningError: .*did not find",
        "segments abandoned (arm not following)": r"so the rest of this segment is abandoned",
        "capture swings blocked": r"capture swing stopped against something",
        "arms blocked against something": r"stopped following the ramp",
        "arm on the camera's line of sight": r"stands between the head camera",
        "views dropped (all robot)": r"nothing but the robot in this view",
        "stances with nothing framed whole": r"no stance frames every object whole",
        "look poses skipped (scene)": r"look offset .* puts \[",
        "look poses skipped (sight line)": r"between the head camera and the target",
    }.items():
        counts[name] = len(re.findall(pattern, text))
    print("\n  from the log:")
    for name, n in counts.items():
        print(f"    {n:4d}  {name}")
    capped = re.findall(
        r"held a target for the whole budget \((\d+) steps, ([\d.]+) rad short\); it last got "
        r"closer (\d+) step\(s\) before the end",
        text,
    )
    if capped:
        flat = [int(c[2]) for c in capped]
        print(f"\n  converges that spent the whole budget: {len(capped)}")
        print(
            f"    of those, steps spent no longer getting closer: min {min(flat)}, median "
            f"{sorted(flat)[len(flat) // 2]}, max {max(flat)}"
        )
        print(
            "    (a large number means real pushing; a small one means it was still settling when the budget ran"
            " out -- this is what an exit rule for converge() needs, see executor.converge)"
        )

    motions = re.findall(r"stopped following the ramp .*?\[motion: ([^,\]]+)", text)
    steps = re.findall(r"stopped following the ramp .*?env step (\d+)\]", text)
    if steps:
        print(
            f"\n  env steps to watch in the video (the arm met something): {', '.join(steps[:20])}"
            + (" ..." if len(steps) > 20 else "")
        )
    if motions:
        print("\n  which motion met the obstacle:")
        for motion, n in Counter(motions).most_common():
            print(f"    {n:4d}  {motion}")
    elif counts.get("arms blocked against something"):
        print("\n  (the ramps in this run predate the motion labels; rerun to attribute them)")

    missing = re.findall(
        r"(\w+) in the (\w+) view: ([\d.-]+) m ahead at pixel \(([\d.-]+), ([\d.-]+)\) of (\d+)x(\d+), "
        r"(OUTSIDE the image|inside the image), depth there ([\d.]+ m|-)",
        text,
    )
    if missing:
        print("\n  where a missed object actually was:")
        for label, view, ahead, u, v, w, h, where, depth in missing:
            if view.startswith("head"):
                print(
                    f"    {label:12s} {view:10s} {ahead:>6s} m ahead, pixel ({u},{v}) of {w}x{h}, {where}, depth {depth}"
                )
