"""Re-send a saved round's request to a running planner and report whether it plans.

A round directory (``runs/<...>/rNN_<arm>_<predicate>/``) holds the exact request the planner got in ``obs.h5``:
the frame, the goal, and what the client knew (masks, button poses, what the hands held). Replaying it separates
a planner failure that reproduces from one that does not, without the simulator:

  python -m omnigibson.tiptop.replay runs/bench_radio/turning_on_radio_301_0/r02_right_toggled_on --port 8766 --repeats 3
"""

import argparse
import logging
from pathlib import Path

from omnigibson.tiptop.client import TiptopClient, TiptopPlanningError
from omnigibson.tiptop.protocol import load_observation_h5, plan_summary, request_from_observation

log = logging.getLogger("omnigibson.tiptop")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rounds", nargs="+", type=Path, help="round directories with an obs.h5")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, required=True, help="the planner of the arm that planned the round")
    p.add_argument("--repeats", type=int, default=1, help="how many times to plan each round (the planner samples)")
    p.add_argument("--timeout", type=float, default=300.0, help="seconds to wait for one plan")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    client = TiptopClient(args.host, args.port, expected_robot_type=None, expected_dof=None)
    for round_dir in args.rounds:
        request = request_from_observation(load_observation_h5(round_dir / "obs.h5"))
        for i in range(args.repeats):
            try:
                response = client.plan(request, timeout_s=args.timeout)
            except TiptopPlanningError as e:
                log.info(f"{round_dir.name} try {i}: no plan: {str(e)[:200]}")
            else:
                log.info(
                    f"{round_dir.name} try {i}: {plan_summary(response['plan'])} (save_dir {response.get('save_dir')})"
                )


if __name__ == "__main__":
    main()
