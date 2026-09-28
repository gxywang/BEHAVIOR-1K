"""Capture a frozen all-task manifest using established zero-settle independent replay.

One subprocess per task, at most two simultaneous simulator workers on explicitly
chosen GPUs. Failed cases remain failed; this launcher never selects replacements.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate(path):
    selection = json.loads(path.read_text())
    if selection.get('selection_frozen_before_capture_and_inference') is not True:
        raise ValueError('Selection must be frozen before captures/predictions.')
    cases = selection['cases']
    count = selection['snapshots_per_task']
    if len({c['task'] for c in cases}) != 100 or any(sum(c['task'] == t['task'] for c in cases) < count for t in selection['tasks']):
        raise ValueError('Expected at least the declared minimum cases for every one of100 tasks.')
    if len(cases) != selection['counts']['cases']:
        raise ValueError('Case count differs from frozen declaration.')
    if len({c['id'] for c in cases}) != len(cases) or len({c['episode_index'] for c in cases}) != len(cases):
        raise ValueError('Case IDs and episodes must be unique.')
    if {c['episode_index'] for c in cases} & set(selection['excluded_episodes']):
        raise ValueError('Query episode overlaps prior query or reference source.')
    for case in cases:
        if not case['target_ids'] or case['split'] != 'test' or case['cohort'] != 'comprehensive_test':
            raise ValueError(f'Invalid test case: {case["id"]}')
        if case['source_materialization_hold_steps'] != 0:
            raise ValueError('Only zero-settle replay is permitted.')
        if case['frame'] != case['materialization']['catalog_row']['start_frame']:
            raise ValueError('Requested replay frame differs from frozen selection.')
        if sha(case['annotation_path']) != case['annotation_sha256']:
            raise ValueError(f'Annotation changed: {case["id"]}')
    for source, expected in selection['source_receipts'].items():
        if sha(source) != expected:
            raise ValueError(f'Capture source changed: {source}')
    return selection


def worker(args, selection):
    import visionbench_expand as capture
    adapted = dict(selection, backup_cases=[dict(c, replaces=c['id']) for c in selection['cases']])
    by_id = {c['id']: c for c in selection['cases']}
    original_relabel, original_write = capture.relabel_case, capture.write_json

    def relabel_with_identity(og, sim, case, source, out, **kwargs):
        result = original_relabel(og, sim, case, source, out, **kwargs)
        labels_path = out / case['id'] / 'labels.json'
        labels = json.loads(labels_path.read_text())
        for row in labels['objects']:
            obj = sim.env.scene.object_registry('name', row['scene_name'])
            if obj is None:
                raise ValueError(f'Missing runtime evaluation object: {row["scene_name"]}')
            model = getattr(obj, 'model', None)
            row['asset_model'] = str(model) if model is not None else None
            row['asset_id'] = f'{obj.category}.{model}' if model is not None else None
            row['asset_identity_source'] = 'Runtime scene category/model, evaluation labels only.'
        labels.update(asset_identity_is_evaluation_only=True,
                      target_resolution=case['target_resolution'],
                      all_initial_task_object_ids=case['all_initial_task_object_ids'],
                      mask_provenance={'method': 'Geometry proximity to captured linear depth',
                                       'main_tolerance_m': .008, 'sensitivity_tolerance_m': .004,
                                       'renderer_instance_ground_truth': False,
                                       'limitations': 'Approximate geometry labels; contact halos/depth-mesh errors possible.'})
        original_write(labels_path, labels)
        return result

    def write_cohort(path, value):
        if path.name == 'labels.json' and path.parent.name in by_id:
            case = by_id[path.parent.name]
            value = dict(value, cohort=case['cohort'], split=case['split'], human_skill=case['human_skill'])
        original_write(path, value)

    capture.relabel_case, capture.write_json = relabel_with_identity, write_cohort
    capture.process_task(SimpleNamespace(selection=args.selection, out=args.out, data=args.data,
                                         task=args.task, allow_backups=False), adapted)
    receipt = json.loads((args.out / 'task_receipts' / f'{args.task}.json').read_text())
    if not receipt.get('ok'):
        raise RuntimeError(f'Task capture contains failures: {args.task}')


def supervise(args, selection):
    from visionbench_capture import environment
    tasks = [t['task'] for t in selection['tasks']]
    if args.tasks:
        if set(args.tasks) - set(tasks):
            raise ValueError('Unknown task subset.')
        tasks = [t for t in tasks if t in args.tasks]
    args.out.mkdir(parents=True, exist_ok=True)
    immutable = {'selection': str(args.selection), 'selection_sha256': sha(args.selection),
                 'adapter': str(Path(__file__).resolve()), 'adapter_sha256': sha(__file__),
                 'python': sys.executable, 'max_own_simulators': len(args.gpus), 'gpus': args.gpus,
                 'data': str(args.data), 'policy': 'Independent replay from frame zero; no replacements.'}
    receipt = args.out / 'selection_receipt.json'
    if receipt.exists():
        if not args.resume or json.loads(receipt.read_text()) != immutable:
            raise ValueError('Existing capture requires --resume and identical launch provenance.')
    else:
        write(receipt, immutable)
    pending = queue.Queue()
    for task in tasks:
        pending.put(task)
    lock, results = threading.Lock(), []

    def event(row):
        with lock:
            with (args.out / 'launcher_results.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)

    def slot(gpu):
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            completed = args.out / 'task_receipts' / f'{task}.json'
            if completed.exists() and json.loads(completed.read_text()).get('ok'):
                event({'phase': 'already_complete', 'task': task, 'gpu': gpu})
                continue
            validate(args.selection)
            logfile = args.out / 'logs' / f'{task}.{time.time_ns()}.log'
            logfile.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), '--selection', str(args.selection),
                       '--out', str(args.out), '--data', str(args.data), '--task', task]
            start = time.time()
            with logfile.open('wb') as stream:
                process = subprocess.Popen(command, cwd=ROOT,
                                           env=environment(gpu) | {'OMNIGIBSON_DATA_PATH': str(args.data)},
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                record = {'task': task, 'gpu': gpu, 'pid': process.pid, 'log': str(logfile), 'start_time': start}
                write(args.out / f'active_gpu{gpu}.json', record)
                event(dict(record, phase='started'))
                code = process.wait()
            result = dict(record, phase='finished', returncode=code, seconds=round(time.time()-start, 2),
                          ok=completed.exists() and json.loads(completed.read_text()).get('ok', False))
            with lock:
                results.append(result)
                write(args.out / 'execution.json', dict(immutable, status='running', tasks=list(results)))
            event(result)
            pending.task_done()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        for future in [pool.submit(slot, gpu) for gpu in args.gpus]:
            future.result()
    receipts = {t: json.loads((args.out/'task_receipts'/f'{t}.json').read_text()).get('ok', False)
                if (args.out/'task_receipts'/f'{t}.json').exists() else False for t in tasks}
    write(args.out / 'execution.json', dict(immutable, status='complete' if all(receipts.values()) else 'incomplete',
                                           task_status=receipts, tasks=results))
    if not all(receipts.values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[1, 3])
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--task')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    args.selection, args.out, args.data = args.selection.resolve(), args.out.resolve(), args.data.resolve()
    os.environ['OMNIGIBSON_DATA_PATH'] = str(args.data)
    if not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus):
        parser.error('Choose one or two distinct GPU IDs after checking ownership and headroom.')
    selection = validate(args.selection)
    if args.validate:
        print(json.dumps({'ok': True, 'selection_sha256': sha(args.selection), 'cases': len(selection['cases']),
                          'tasks': len(selection['tasks']), 'source_files': len(selection['source_receipts'])}))
    elif args.task:
        worker(args, selection)
    else:
        supervise(args, selection)


if __name__ == '__main__':
    main()
