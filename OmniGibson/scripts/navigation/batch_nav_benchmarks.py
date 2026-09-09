"""Check every task benchmark in a directory, or run them in one simulator session."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_task_episode_selectors(values, parser):
    selectors = {}
    for value in values:
        if ":" not in value:
            parser.error("--task-episode values must use task_name:index[,index...]")
        task_name, episode_indices = value.split(":", 1)
        task_name = task_name.strip()
        if not task_name:
            parser.error("--task-episode requires a non-empty task name")
        try:
            indices = [int(index.strip()) for index in episode_indices.split(",") if index.strip()]
        except ValueError:
            parser.error("--task-episode episode indices must be integers")
        if not indices:
            parser.error("--task-episode requires at least one episode index")
        if any(index < 0 for index in indices):
            parser.error("--task-episode episode indices must be non-negative")
        selectors.setdefault(task_name, set()).update(indices)
    return selectors


def episode_index(episode):
    return int(episode["episode_id"].rsplit("_", 1)[1])


def selected_episode_ids_for_benchmark(benchmark, selectors):
    with open(benchmark, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected_episode_ids = []
    found = set()
    for episode in data.get("episodes", []):
        task_name = episode.get("task_name")
        if task_name not in selectors:
            continue
        index = episode_index(episode)
        if index not in selectors[task_name]:
            continue
        selected_episode_ids.append(episode["episode_id"])
        found.add((task_name, index))
    return selected_episode_ids, found


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Pass checker or nav2py options after --, e.g. -- --trace-failures.",
    )
    parser.add_argument("mode", choices=("check", "run"))
    parser.add_argument("--benchmark-dir", default="outputs/navigation/task_benchmarks")
    parser.add_argument(
        "--output-dir",
        default="outputs/navigation/nav2py_results",
        help="Result directory for run mode",
    )
    parser.add_argument(
        "--task-episode",
        action="append",
        default=[],
        help="Run only selected task episodes, formatted as task_name:index[,index...]. Repeat as needed.",
    )
    argv = sys.argv[1:]
    separator = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:separator])
    extra_args = argv[separator + 1 :]
    selectors = parse_task_episode_selectors(args.task_episode, parser)
    if selectors and args.mode != "run":
        parser.error("--task-episode is only supported in run mode")
    for option in extra_args:
        if option.split("=", 1)[0] in {"--input", "--benchmark", "--output", "--episode-ids"}:
            parser.error("Input and output files are managed by the wrapper; use --benchmark-dir and --output-dir.")

    benchmark_dir = Path(args.benchmark_dir).expanduser()
    benchmarks = sorted(path for path in benchmark_dir.glob("*.json") if path.is_file())
    if not benchmarks:
        parser.error(f"No benchmark JSON files found in {benchmark_dir}")
    run_plan = [(benchmark, None) for benchmark in benchmarks]
    if selectors:
        run_plan = []
        found = set()
        for benchmark in benchmarks:
            selected_episode_ids, benchmark_found = selected_episode_ids_for_benchmark(benchmark, selectors)
            found.update(benchmark_found)
            if selected_episode_ids:
                run_plan.append((benchmark, selected_episode_ids))
        requested = {(task_name, index) for task_name, indices in selectors.items() for index in indices}
        missing = sorted(requested - found)
        if missing:
            formatted = ", ".join(f"{task}:{index:03d}" for task, index in missing)
            parser.error(f"Requested task episodes were not found: {formatted}")
        if not run_plan:
            parser.error("No benchmark files matched --task-episode")

    output_dir = Path(args.output_dir).expanduser()
    if args.mode == "run":
        output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMNIGIBSON_HEADLESS", "1")

    if args.mode == "run":
        os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
        from run_nav2py_benchmark import keep_viewer_open
        from run_nav2py_benchmark import main as run_benchmark
        from run_nav2py_benchmark import parse_args as parse_run_args
        import omnigibson as og

        run_args = parse_run_args(extra_args)
        failures = []
        try:
            for index, (benchmark, selected_episode_ids) in enumerate(run_plan, 1):
                print(f"\n[{index}/{len(run_plan)}] run: {benchmark.name}", flush=True)
                if selected_episode_ids is not None:
                    print(f"  selected episodes: {', '.join(selected_episode_ids)}", flush=True)
                    run_args.episode_ids = selected_episode_ids
                run_args.benchmark = str(benchmark)
                run_args.output = str(output_dir / f"{benchmark.stem}_results.json")
                try:
                    run_benchmark(run_args, shutdown=False)
                except Exception as exc:
                    failures.append((benchmark.name, str(exc)))
                    print(f"  FAILED: {benchmark.name}: {exc}", flush=True)
            print(
                f"\nCompleted {len(run_plan)} files: "
                f"{len(run_plan) - len(failures)} passed, {len(failures)} failed.",
                flush=True,
            )
            for name, reason in failures:
                print(f"  FAILED: {name}: {reason}", flush=True)
            return 1 if failures else 0
        finally:
            if og.app is not None:
                if run_args.keep_open_on_complete:
                    keep_viewer_open(run_args.keep_open_seconds)
                og.shutdown()

    script = Path(__file__).resolve().with_name("check_nav_benchmark.py")
    failures = []
    for index, (benchmark, _) in enumerate(run_plan, 1):
        print(f"\n[{index}/{len(run_plan)}] {args.mode}: {benchmark.name}", flush=True)
        command = [sys.executable, str(script)]
        command.extend(["--input", str(benchmark)])
        result = subprocess.run(command + extra_args, env=env, check=False)
        if result.returncode != 0:
            failures.append((benchmark.name, result.returncode))

    print(f"\nCompleted {len(run_plan)} files: {len(run_plan) - len(failures)} passed, {len(failures)} failed.")
    for name, returncode in failures:
        print(f"  FAILED: {name} (exit code {returncode})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
