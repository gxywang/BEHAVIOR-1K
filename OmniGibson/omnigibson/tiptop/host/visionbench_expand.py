"""Materialize and capture a predeclared expansion, at most two task workers on explicit GPUs.

Selection stays immutable. Snapshot hashes, completed-case mappings and failures live in separate receipts.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

from visionbench import ROOT, VIEWS, _assert_frozen, _stamp, capture_case
from visionbench_capture import environment
from visionbench_coverage import sha256
from visionbench_v2 import expanded_objects, relabel_case


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_selection(path):
    selection = json.loads(path.read_text())
    if selection.get('selection_locked_before_v3_predictions') is not True:
        raise ValueError('selection must be locked before expanded predictions')
    cases = selection['cases']
    if len(cases) != 60 or len({c['task'] for c in cases}) != 20:
        raise ValueError('expected60 cases across20 tasks')
    if set(c['task'] for c in cases) & set(selection['excluded_previous_tasks']):
        raise ValueError('new task overlaps previous benchmark')
    if len({c['id'] for c in cases}) != 60 or len({c['episode_index'] for c in cases}) != 60:
        raise ValueError('expected distinct case IDs and episodes')
    for case in cases + selection['backup_cases']:
        if Path(case['id']).name != case['id'] or not case['target_ids'] or case['split'] != 'test':
            raise ValueError(f'invalid selected case: {case["id"]}')
        if not case['materialization']['required'] and sha256(ROOT / case['snapshot']) != case['snapshot_sha256']:
            raise ValueError(f'cached snapshot hash changed: {case["id"]}')
    return selection


def configure_demo_replay(config, evaluator_robot, task_metadata):
    """Apply the complete recorded-action controller/reset contract while retaining capture-only cameras."""
    robot = config['robots'][0]
    for key in ('controller_config', 'reset_joint_pos'):
        robot[key] = copy.deepcopy(evaluator_robot[key])
    robot['position'] = copy.deepcopy(task_metadata['robot_start_position'])
    robot['orientation'] = copy.deepcopy(task_metadata['robot_start_orientation'])
    return config


def process_task(args, selection):
    import omnigibson as og
    from omnigibson.eval.evaluator import DISABLED_TRANSITION_RULES, DEFAULT_ROBOT_CONFIG_PATH, EVAL_BASE_LINK_MASS
    from omnigibson.eval.utils.eval_utils import generate_basic_environment_config
    from omegaconf import OmegaConf
    from omnigibson.tiptop.host import demo_cases
    from omnigibson.tiptop.r1pro import R1ProSim, challenge_task_info, make_r1pro_env_config
    from omnigibson.tiptop.run import setup_logging
    from omnigibson.utils.config_utils import TorchEncoder

    setup_logging()
    for rule in DISABLED_TRANSITION_RULES:
        rule.ENABLED = False
    scene, rooms = challenge_task_info(args.task)
    config = make_r1pro_env_config(scene_model=scene, load_room_instances=rooms, activity=args.task,
                                  grasping_mode='assisted', camera='head', views=VIEWS[1:], segmentation=False)
    # Dataset actions require the evaluator velocity base, gains and JoyLo reset joints.
    # The manipulation bridge defaults to position-delta base control and a different reset posture.
    evaluator_robot = OmegaConf.to_container(OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH))
    task_metadata = demo_cases.available_tasks()[args.task][0]
    configure_demo_replay(config, evaluator_robot, task_metadata)
    evaluator_environment = generate_basic_environment_config(args.task, task_metadata)
    evaluator_environment['scene']['load_room_instances'] = rooms
    config['scene'], config['task'] = evaluator_environment['scene'], evaluator_environment['task']
    config['env'] = evaluator_environment['env'] | {'external_sensors': config['env']['external_sensors']}
    selection_hash = sha256(args.selection)
    rows = [case for case in selection['cases'] if case['task'] == args.task]
    source = args.out / '_source_captures'
    results = []
    try:
        sim = R1ProSim(config, camera='head', views=VIEWS[1:], look_arm=None)
        # Match demo_cases.make_env exactly: apply the base mass while the simulator is stopped.
        og.sim.stop()
        sim.robot.base_footprint_link.mass = EVAL_BASE_LINK_MASS
        og.sim.play()

        def depth_audit(env, episode, frame):
            sim._link_meshes.clear()
            before = _stamp(og, sim)
            # Physics-only replay leaves robot-mounted camera transforms stale until a render flush.
            # Flush before view_frame reads that pose to place the external shadow camera.
            og.sim.render()
            og.sim.render()
            view, _ = sim.view_frame('head')
            _assert_frozen(before, _stamp(og, sim))
            depth = view['depth']
            recorded = demo_cases.recorded_head_depth(episode, frame)
            valid = np.isfinite(depth) & (depth > 0)
            difference = np.abs(np.clip(depth, .01, 10) - recorded)[valid]
            if not difference.size:
                return {'median_m': None, 'within_5cm': None, 'valid_fraction': 0.0,
                        'source': 'shadow_camera', 'warning': 'no valid depth; preserve selected case'}
            return {'median_m': float(np.median(difference)), 'within_5cm': float((difference < .05).mean()),
                    'valid_fraction': float(valid.mean()), 'source': 'shadow_camera; excludes self-filtered depth',
                    'camera_pose_refresh': {'renders': 2, 'physics_steps': 0, 'frozen_stamp_verified': True}}

        def one_case(chosen):
            case = json.loads(json.dumps(chosen))
            start = time.monotonic()
            needs_replay = case['materialization']['required']
            print(json.dumps({'phase': 'materialize' if needs_replay else 'cached_snapshot', 'id': case['id'],
                              'frame': case['frame'], 'episode': case['episode_index']}), flush=True)
            if needs_replay:
                setup = demo_cases.materialize(case['materialization']['catalog_row'], sim.env,
                                               method='replay', settle=0, depth_audit=depth_audit)
                snapshot = ROOT / case['snapshot']
                if snapshot.exists():
                    raise FileExistsError(f'never overwrite a materialized snapshot: {snapshot}')
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_text(json.dumps(setup['snapshot'], cls=TorchEncoder) + '\n')
                case['snapshot_sha256'] = sha256(snapshot)
                case['fidelity'] = setup['fidelity']
                case['fidelity']['controller_config_source'] = DEFAULT_ROBOT_CONFIG_PATH
                case['fidelity']['controller_config_sha256'] = sha256(Path(DEFAULT_ROBOT_CONFIG_PATH))
                print(json.dumps({'phase': 'materialized', 'id': case['id'],
                                  'replay_seconds': setup['fidelity']['seconds'],
                                  'depth': setup['fidelity']['depth']}), flush=True)
            write_json(args.out / 'materialization' / f'{case["id"]}.json',
                       {'selection_sha256': selection_hash, 'case': case,
                        'replay_configuration': {
                            'evaluator_yaml': DEFAULT_ROBOT_CONFIG_PATH,
                            'evaluator_yaml_sha256': sha256(Path(DEFAULT_ROBOT_CONFIG_PATH)),
                            'controller_config': config['robots'][0]['controller_config'],
                            'reset_joint_pos': config['robots'][0]['reset_joint_pos'],
                            'base_mass_kg': float(sim.robot.base_footprint_link.mass),
                            'mass_initialization': 'set while simulation stopped, then play; demo_cases.make_env sequence',
                            'scene_config': config['scene'],
                            'scene_model': scene, 'action_frequency_hz': config['env']['action_frequency'],
                            'physics_frequency_hz': config['env']['physics_frequency']}})
            capture_case(og, sim, case, source, labeler=lambda host, item: expanded_objects(host, item)[0],
                         restore_snapshot=not needs_replay)
            result = relabel_case(og, sim, case, source, args.out, restore_snapshot=False)
            labels_path = args.out / case['id'] / 'labels.json'
            labels = json.loads(labels_path.read_text())
            labels.update(cohort='expanded_test', stratum=case['stratum'], stage=case['stage'],
                          selection_sha256=selection_hash, materialization_receipt=f'materialization/{case["id"]}.json')
            write_json(labels_path, labels)
            result.update(selection_sha256=selection_hash, wall_s=round(time.monotonic() - start, 2),
                          replay_frames=case['frame'] if needs_replay else 0)
            return result

        for primary in rows:
            receipt = args.out / 'case_receipts' / f'{primary["id"]}.json'
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved.get('selection_sha256') != selection_hash or not saved.get('ok'):
                    raise ValueError(f'invalid completed receipt: {receipt}')
                effective = saved['effective_id']
                if not (args.out / effective / 'labels_strict4mm.npz').is_file():
                    raise ValueError(f'incomplete saved result: {effective}')
                results.append(saved)
                continue
            backup = next(case for case in selection['backup_cases'] if case['replaces'] == primary['id'])
            attempts = []
            for chosen in ((primary, backup) if args.allow_backups else (primary,)):
                try:
                    result = one_case(chosen)
                    result.update(primary_id=primary['id'], effective_id=chosen['id'], attempts=attempts)
                    write_json(receipt, result)
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    break
                except Exception as error:
                    failure = {'id': chosen['id'], 'primary_id': primary['id'], 'ok': False,
                               'error': str(error), 'traceback': traceback.format_exc(),
                               'selection_sha256': selection_hash}
                    failure_dir = args.out / '_attempt_failures' / chosen['id']
                    failure_dir.parent.mkdir(parents=True, exist_ok=True)
                    if (args.out / chosen['id']).exists():
                        shutil.move(str(args.out / chosen['id']), str(failure_dir))
                    write_json(failure_dir / 'failure.json', failure)
                    attempts.append(failure)
                    print(json.dumps(failure), flush=True)
            else:
                results.append({'primary_id': primary['id'], 'ok': False, 'attempts': attempts})
        write_json(args.out / 'task_receipts' / f'{args.task}.json',
                   {'task': args.task, 'ok': all(row['ok'] for row in results), 'cases': results,
                    'selection_sha256': selection_hash})
    finally:
        if og.app is not None:
            og.shutdown()


def launch(args, selection):
    tasks = list(dict.fromkeys(case['task'] for case in selection['cases']))
    if args.tasks:
        if set(args.tasks) - set(tasks):
            raise ValueError('unknown task in --tasks')
        tasks = [task for task in tasks if task in args.tasks]
    record = {'selection': str(args.selection), 'sha256': sha256(args.selection), 'python': sys.executable}
    selection_receipt = args.out / 'selection_receipt.json'
    if args.out.exists():
        if not args.resume or json.loads(selection_receipt.read_text()) != record:
            raise ValueError('existing output requires --resume with the exact same selection and Python')
    else:
        args.out.mkdir(parents=True)
        write_json(selection_receipt, record)
    pending, lock = queue.Queue(), threading.Lock()
    for task in tasks:
        pending.put(task)

    def worker(gpu):
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            existing = args.out / 'task_receipts' / f'{task}.json'
            if args.resume and existing.exists() and json.loads(existing.read_text()).get('ok'):
                continue
            logs = args.out / 'logs'
            logs.mkdir(exist_ok=True)
            attempt = len(list(logs.glob(f'{task}.*.log'))) + 1
            logfile = logs / f'{task}.{attempt:03d}.log'
            command = [sys.executable, str(Path(__file__).resolve()), '--selection', str(args.selection),
                       '--out', str(args.out), '--data', str(args.data), '--task', task]
            if args.allow_backups:
                command.append('--allow-backups')
            with logfile.open('wb') as stream:
                process = subprocess.Popen(command, cwd=ROOT,
                                           env=environment(gpu) | {'OMNIGIBSON_DATA_PATH': str(args.data)},
                                           stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT)
                row = {'phase': 'started', 'task': task, 'gpu': gpu, 'pid': process.pid, 'log': str(logfile)}
                with lock:
                    with (args.out / 'launcher_results.jsonl').open('a') as output:
                        output.write(json.dumps(row) + '\n')
                    print(json.dumps(row), flush=True)
                code = process.wait()
            receipt = json.loads(existing.read_text()) if existing.exists() else {}
            row = {'phase': 'finished', 'task': task, 'gpu': gpu, 'returncode': code, 'ok': receipt.get('ok', False)}
            with lock:
                with (args.out / 'launcher_results.jsonl').open('a') as output:
                    output.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
            pending.task_done()

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        for future in [pool.submit(worker, gpu) for gpu in args.gpus]:
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=Path(__file__).with_name('visionbench_expanded_selection.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path(os.environ.get(
        'OMNIGIBSON_DATA_PATH', '/home/wding8/projects/BEHAVIOR-1K/datasets')))
    parser.add_argument('--task', help='child worker: exactly one task')
    parser.add_argument('--tasks', nargs='+', help='launcher: optional task subset for smoke runs')
    parser.add_argument('--gpus', type=int, nargs='+', choices=(1, 2, 3), default=[3])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-backups', action='store_true',
                        help='explicitly enable predeclared substitutions; requires a new effective transfer lock')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    args.selection, args.out, args.data = args.selection.resolve(), args.out.resolve(), args.data.resolve()
    os.environ['OMNIGIBSON_DATA_PATH'] = str(args.data)
    if not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus):
        parser.error('one or two distinct explicit GPUs required')
    selection = read_selection(args.selection)
    if args.validate:
        print(json.dumps({'ok': True, 'primary_cases': len(selection['cases']),
                          'backups': len(selection['backup_cases']), 'selection_sha256': sha256(args.selection)}))
    elif args.task:
        process_task(args, selection)
    else:
        launch(args, selection)


if __name__ == '__main__':
    main()
