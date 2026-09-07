"""Check or run every task benchmark JSON in a directory, one process per file."""

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

    script_name = "check_nav_benchmark.py" if args.mode == "check" else "run_nav2py_benchmark.py"
    script = Path(__file__).resolve().with_name(script_name)
    output_dir = Path(args.output_dir).expanduser()
    if args.mode == "run":
        output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMNIGIBSON_HEADLESS", "1")

    failures = []
    for index, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{index}/{len(benchmarks)}] {args.mode}: {benchmark.name}", flush=True)
        command = [sys.executable, str(script)]
        if args.mode == "check":
            command.extend(["--input", str(benchmark)])
        else:
            output = output_dir / f"{benchmark.stem}_results.json"
            command.extend(["--benchmark", str(benchmark), "--output", str(output)])
        result = subprocess.run(command + extra_args, env=env, check=False)
        if result.returncode != 0:
            failures.append((benchmark.name, result.returncode))

    print(f"\nCompleted {len(benchmarks)} files: {len(benchmarks) - len(failures)} passed, {len(failures)} failed.")
    for name, returncode in failures:
        print(f"  FAILED: {name} (exit code {returncode})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
