"""Read a bench run: the score per instance, why rounds were lost, and the counts the capture reports itself.

    python OmniGibson/omnigibson/tiptop/scripts/read_run.py runs/bench_batteries_8 [the run's log]

Needs no simulator and no GPU: it reads ``summary.json`` and, when given the log, counts the lines the pipeline
writes about its own trouble (blocked capture swings, an arm on the head camera's line of sight, a view that was
all robot, a stance that framed nothing whole) and prints where each object the capture could not see actually
was. Written on 2026-09-13 so every run is read the same way instead of by grepping afresh."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
log = Path(sys.argv[2]) if len(sys.argv) > 2 else None
BASELINES = Path(__file__).resolve().parent.parent / "baselines.json"

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

if log and log.exists():
    text = log.read_text(encoding="utf-8", errors="replace")
    counts = Counter()
    for name, pattern in {
        "rounds executed": r"round \d+ .*: executed",
        "rounds lost to empty masks": r"\[\w+\]: GoalNotVisible",
        "rounds lost to planning": r"\[\w+\]: TiptopPlanningError",
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
