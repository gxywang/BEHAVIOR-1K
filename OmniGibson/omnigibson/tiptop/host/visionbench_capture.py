"""Capture all manifest tasks using one or two explicitly selected GPUs, one simulator per worker.

Run as a file with the installed simulation Python. This launcher never installs, selects a GPU automatically,
restarts a failed task, or stops an existing process. Inspect GPU occupants before selecting --gpus.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from visionbench import ROOT, read_cases


def command(task: str, manifest: Path, out: Path, source_corpus: Path | None = None) -> list[str]:
    script = "visionbench_v2.py" if source_corpus is not None else "visionbench.py"
    args = [sys.executable, str(Path(__file__).with_name(script)), "--manifest", str(manifest),
            "--task", task, "--out", str(out)]
    if source_corpus is not None:
        args.extend(["--source-corpus", str(source_corpus)])
    return args


def environment(gpu: int) -> dict:
    env = dict(os.environ)
    env.pop("OMNIGIBSON_GPU_ID", None)
    env.update(OMNIGIBSON_HEADLESS="1", CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / p) for p in ("OmniGibson", "tiptop", "bddl3")]
                                      + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("visionbench_cases.json"))
    parser.add_argument("--out", type=Path, required=True, help="fresh corpus output directory")
    parser.add_argument("--source-corpus", type=Path, help="v2: relabel existing inputs and capture new test cases")
    parser.add_argument("--gpus", type=int, nargs="+", choices=(1, 2, 3), default=[3],
                        help="one or two explicit available GPU IDs; default one worker on GPU3")
    parser.add_argument("--validate", action="store_true", help="validate manifest and print commands; launch nothing")
    args = parser.parse_args(argv)
    if not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus):
        parser.error("select one or two distinct GPUs from 1, 2, 3")
    manifest, out = args.manifest.resolve(), args.out.resolve()
    if args.source_corpus is not None:
        args.source_corpus = args.source_corpus.resolve()
    cases = read_cases(manifest)
    if args.source_corpus is not None and any(c.get("split") not in ("dev", "test") for c in cases):
        parser.error("v2 requires an explicit dev/test manifest, e.g. --manifest visionbench_cases_v2.json")
    tasks = list(dict.fromkeys(c["task"] for c in cases))
    if args.validate:
        print(json.dumps({"cases": len(cases), "tasks": tasks, "gpus": args.gpus,
                          "commands": {t: command(t, manifest, out, args.source_corpus) for t in tasks}}, indent=2))
        return
    if out.exists():
        parser.error(f"output already exists; use a fresh directory: {out}")
    out.mkdir(parents=True)
    logs = out / "logs"
    logs.mkdir()
    pending = queue.Queue()
    for task in tasks:
        pending.put(task)
    lock = threading.Lock()
    results = []
    (out / "capture_run.json").write_text(json.dumps({"manifest": str(manifest), "gpus": args.gpus,
        "python": sys.executable, "case_ids": [c["id"] for c in cases], "tasks": tasks,
        "omnigibson_data_path": os.environ.get("OMNIGIBSON_DATA_PATH")}, indent=2) + "\n")

    def record(row):
        with lock:
            with (out / "launcher_results.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    def worker(gpu):
        env = environment(gpu)
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            selected = [c for c in cases if c["task"] == task]
            logfile = logs / f"{task}.log"
            start = time.monotonic()
            code, error = None, None
            try:
                with logfile.open("wb") as stream:
                    process = subprocess.Popen(command(task, manifest, out, args.source_corpus), cwd=ROOT, env=env,
                                               stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    record({"task": task, "gpu": gpu, "phase": "started", "pid": process.pid,
                            "log": str(logfile)})
                    code = process.wait()
            except Exception as exc:
                error = str(exc)
            complete = 0
            for case in selected:
                directory = out / case["id"]
                if (directory / "input.json").exists() and (directory / "labels.json").exists():
                    complete += 1
                elif not (directory / "failure.json").exists():
                    directory.mkdir(parents=True, exist_ok=True)
                    failure = {"id": case["id"], "ok": False, "phase": "initialization_or_process_exit",
                               "returncode": code, "error": error, "log": str(logfile)}
                    (directory / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            row = {"task": task, "gpu": gpu, "phase": "finished", "returncode": code,
                   "complete": complete, "expected": len(selected), "wall_s": round(time.monotonic() - start, 2),
                   "ok": code == 0 and complete == len(selected), "error": error}
            with lock:
                results.append(row)
            record(row)
            pending.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    summary = {"ok": all(r["ok"] for r in results), "complete": sum(r["complete"] for r in results),
               "expected": len(cases), "tasks": results}
    (out / "capture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
