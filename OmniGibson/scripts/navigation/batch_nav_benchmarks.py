"""Check every task benchmark in a directory, or run them in one simulator session."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Pass checker or nav2py options after --, e.g. -- --trace-failures.",
    )
    parser.add_argument("mode", choices=("check", "run"))
    parser.add_argument("--benchmark-dir", default="outputs/navigation/task_benchmarks")
    parser.add_argument("--output-dir", default="outputs/navigation/nav2py_results", help="Result directory for run mode")
    argv = sys.argv[1:]
    separator = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:separator])
    extra_args = argv[separator + 1 :]
    for option in extra_args:
        if option.split("=", 1)[0] in {"--input", "--benchmark", "--output"}:
            parser.error("Input and output files are managed by the wrapper; use --benchmark-dir and --output-dir.")

    benchmark_dir = Path(args.benchmark_dir).expanduser()
    benchmarks = sorted(path for path in benchmark_dir.glob("*.json") if path.is_file())
    if not benchmarks:
        parser.error(f"No benchmark JSON files found in {benchmark_dir}")

    output_dir = Path(args.output_dir).expanduser()
    if args.mode == "run":
        output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMNIGIBSON_HEADLESS", "1")

    if args.mode == "run":
        os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
        from run_nav2py_benchmark import main as run_benchmark
        from run_nav2py_benchmark import parse_args as parse_run_args
        import omnigibson as og

        run_args = parse_run_args(extra_args)
        failures = []
        try:
            for index, benchmark in enumerate(benchmarks, 1):
                print(f"\n[{index}/{len(benchmarks)}] run: {benchmark.name}", flush=True)
                run_args.benchmark = str(benchmark)
                run_args.output = str(output_dir / f"{benchmark.stem}_results.json")
                try:
                    run_benchmark(run_args, shutdown=False)
                except Exception as exc:
                    failures.append((benchmark.name, str(exc)))
                    print(f"  FAILED: {benchmark.name}: {exc}", flush=True)
            print(
                f"\nCompleted {len(benchmarks)} files: "
                f"{len(benchmarks) - len(failures)} passed, {len(failures)} failed.",
                flush=True,
            )
            for name, reason in failures:
                print(f"  FAILED: {name}: {reason}", flush=True)
            return 1 if failures else 0
        finally:
            if og.app is not None:
                og.shutdown()

    script = Path(__file__).resolve().with_name("check_nav_benchmark.py")
    failures = []
    for index, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{index}/{len(benchmarks)}] {args.mode}: {benchmark.name}", flush=True)
        command = [sys.executable, str(script)]
        command.extend(["--input", str(benchmark)])
        result = subprocess.run(command + extra_args, env=env, check=False)
        if result.returncode != 0:
            failures.append((benchmark.name, result.returncode))

    print(f"\nCompleted {len(benchmarks)} files: {len(benchmarks) - len(failures)} passed, {len(failures)} failed.")
    for name, returncode in failures:
        print(f"  FAILED: {name} (exit code {returncode})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
