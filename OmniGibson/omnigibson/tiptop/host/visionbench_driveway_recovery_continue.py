"""Bounded detached continuation: original queue -> exact recovery -> verify -> seal -> portable companion.

No model inference, substitution, retuning, source edits, or concurrent extra simulators.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from visionbench_driveway_recovery import read, binding, verify_bindings, write_new


def process_present(pid):
    path = Path("/proc") / str(pid) / "stat"
    try:
        state = path.read_text()
    except FileNotFoundError:
        return False
    # Zombie supervisor cannot launch further workers.
    return state.split(") ", 1)[1].split()[0] != "Z"


def gpu_readiness(gpus, minimum_mib):
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], text=True
    )
    by_index = {}
    for line in rows.strip().splitlines():
        index, uuid, free = [p.strip() for p in line.split(",")]
        by_index[int(index)] = {"uuid": uuid, "free_mib": int(free)}
    used = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True
    )
    processes = {}
    for line in used.strip().splitlines():
        if line.strip():
            uuid, pid = [p.strip() for p in line.split(",")]
            processes.setdefault(uuid, []).append(int(pid))
    info = {gpu: dict(by_index[gpu], compute_pids=processes.get(by_index[gpu]["uuid"], [])) for gpu in gpus}
    return all(r["free_mib"] >= minimum_mib and not r["compute_pids"] for r in info.values()), info


def capture_environment(gpu, data):
    from visionbench_capture import environment

    return environment(gpu) | {
        "OMNIGIBSON_DATA_PATH": str(data),
        "OMNIGIBSON_HEADLESS": "1",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
    }


def continue_run(config_path):
    config = read(config_path)
    host, run, recovery = (Path(config[k]) for k in ("host", "run", "recovery"))
    status_path = recovery / "status.json"

    def status(phase, **fields):
        value = {
            "schema": "driveway-capture-recovery-continuation/1",
            "phase": phase,
            "time": time.time(),
            "predictions_read": False,
            "source_edits": False,
            **fields,
        }
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(status_path)
        print(json.dumps(value), flush=True)

    def command(args, *, gpu=None, logname=None):
        verify_bindings(config["bindings"])
        args = [str(x) for x in args]
        status("running_command", command=args, gpu=gpu)
        environment = os.environ.copy() if gpu is None else capture_environment(gpu, config["sim_data"])
        if logname is not None:
            logpath = recovery / "logs" / logname
            logpath.parent.mkdir(exist_ok=True)
            with logpath.open("xb") as stream:
                subprocess.run(
                    args,
                    cwd=host.parents[3],
                    env=environment,
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
        else:
            subprocess.run(args, cwd=host.parents[3], env=environment, check=True, stdin=subprocess.DEVNULL)

    try:
        verify_bindings(config["bindings"])
        if config["gpus"] != [1, 3] or config["minimum_free_mib"] < 40000:
            raise ValueError("Only explicitly assigned GPUs1/3 with40GB free are allowed.")
        if (run / "sealed/seal_receipt.json").exists() or (recovery / "preparation.json").exists():
            raise ValueError("Continuation requires a fresh unprepared recovery and no final seal.")
        deadline = config["created_at"] + config["maximum_wait_seconds"]
        while process_present(config["supervisor_pid"]):
            verify_bindings(config["bindings"])
            if time.time() > deadline:
                raise TimeoutError("Original supervisor exceeded bounded continuation wait.")
            status("waiting_for_original_supervisor", supervisor_pid=config["supervisor_pid"])
            time.sleep(60)
        execution = read(run / "captures/execution.json")
        if execution.get("status") != "incomplete":
            raise ValueError("Expected the original queue to finish with the preserved driveway failures.")
        # Original observer/finalizer stop naturally on incomplete status. Do not race their exit.
        while any(process_present(pid) for pid in config["observer_pids"]):
            if time.time() > deadline:
                raise TimeoutError("Original observers did not finish after incomplete supervisor.")
            status("waiting_for_original_observers", observer_pids=config["observer_pids"])
            time.sleep(10)
        for name in ("monitor_completion_v4.json", "capture_provenance_finalization_v3.json"):
            path = run / name
            if path.exists():
                dst = recovery / "original_observers" / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open("xb") as stream:
                    stream.write(path.read_bytes())
        py, selection, captures = config["python"], run / "selection_fullcoverage.json", run / "captures"
        command(
            [
                py,
                host / "visionbench_driveway_recovery.py",
                "prepare",
                "--selection",
                selection,
                "--captures",
                captures,
                "--recovery",
                recovery,
            ]
        )
        preparation = read(recovery / "preparation.json")
        cases = {c["id"]: c for c in read(selection)["cases"]}
        tasks = list(dict.fromkeys(cases[c]["task"] for c in preparation["eligible_case_ids"]))
        for task in tasks:
            while True:
                verify_bindings(config["bindings"])
                ready, gpu_info = gpu_readiness(config["gpus"], config["minimum_free_mib"])
                if ready:
                    break
                if time.time() > deadline:
                    raise TimeoutError("Assigned GPUs did not become exclusively available.")
                status("waiting_for_assigned_gpus", gpu_info=gpu_info)
                time.sleep(60)
            write_new(recovery / "gpu_checks" / (task + ".json"), {"task": task, "time": time.time(), "gpus": gpu_info})
            command(
                [
                    py,
                    host / "visionbench_driveway_recovery.py",
                    "worker",
                    "--selection",
                    selection,
                    "--captures",
                    captures,
                    "--recovery",
                    recovery,
                    "--task",
                    task,
                ],
                gpu=config["gpus"][0],
                logname=task + ".log",
            )
        verify_bindings(config["bindings"])
        command(
            [
                py,
                host / "visionbench_driveway_recovery_verify.py",
                "promote",
                "--recovery",
                recovery,
                "--captures",
                captures,
            ]
        )
        command(
            [
                py,
                host / "visionbench_all_tasks_audit_v3.py",
                "--selection",
                selection,
                "--captures",
                captures,
                "--out",
                run / "sealed",
                "--seal",
            ]
        )
        command(
            [
                py,
                host / "visionbench_all_tasks_fidelity.py",
                "--selection",
                selection,
                "--captures",
                captures,
                "--output",
                run / "replay_fidelity_summary.json",
                "--fresh",
            ]
        )
        stage = run / "capture_provenance_stage_driveway_recovery"
        command(
            [
                py,
                host / "visionbench_driveway_recovery_verify.py",
                "stage",
                "--selection",
                selection,
                "--recovery",
                recovery,
                "--stage",
                stage,
                "--sim-data",
                config["sim_data"],
            ]
        )
        archive = run / "vision-bench-comprehensive-capture-provenance.tar"
        command([py, host / "visionbench_all_tasks_package_v3.py", "finalize", "--stage", stage, "--archive", archive])
        extracted = run / "capture_provenance_extracted_recovery"
        extracted.mkdir()
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                    raise ValueError("Unsafe companion archive member.")
            stream.extractall(extracted, filter="data")
        command([py, host / "visionbench_all_tasks_package_v3.py", "verify", "--provenance", extracted])
        command(
            [
                py,
                host / "visionbench_driveway_recovery_verify.py",
                "verify-portable",
                "--provenance",
                extracted,
                "--captures",
                captures,
                "--output",
                run / "capture_provenance_recovery_audit.json",
            ]
        )
        command(
            [
                py,
                host / "visionbench_all_tasks_package_v3.py",
                "audit-overlay",
                "--provenance",
                extracted,
                "--main-captures",
                captures,
                "--mount",
                run / "capture_provenance_audit_mount_recovery",
                "--output",
                run / "capture_provenance_overlay_audit_recovery.json",
            ]
        )
        status(
            "complete",
            archive=binding(archive),
            recovery_verification=binding(recovery / "verification_published.json"),
            portable_recovery_audit=binding(run / "capture_provenance_recovery_audit.json"),
            capture_overlay_audit=binding(run / "capture_provenance_overlay_audit_recovery.json"),
        )
    except Exception as error:
        status("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    continue_run(args.config.resolve())


if __name__ == "__main__":
    main()
