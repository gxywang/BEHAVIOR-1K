"""Taxonomy-complete approximate geometry labels on immutable vision benchmark inputs.

Original cases restore their snapshot and relabel SAVED depth and calibration without rendering. New cases first
capture once with the existing frozen-state exporter. In both paths source labels and input bytes are preserved.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from pathlib import Path

import numpy as np

from visionbench import ROOT, VIEWS, _aabb_base, _assert_frozen, _stamp, capture_case, read_cases, validate_arrays
from visionbench_coverage import precision_coverage, sha256

log = logging.getLogger(__name__)


def expanded_objects(sim, case):
    """One row per physical scene instance; broad/narrow task prompts become accepted categories on that row."""
    from b1k.bridge.protocol import bddl_category

    coverage = precision_coverage(case)
    evidence = coverage['precision_coverage']['categories']
    sim.objects = {}
    sim.track_task_objects(skip_categories=())
    scope = sim.task_scope()
    task_by_name = {}
    for bddl, obj in sorted(scope.items()):
        task_by_name.setdefault(obj.name, []).append((bddl, obj))
    objects = []
    for obj in sorted(sim.env.scene.objects, key=lambda o: o.name):
        if obj is sim.robot:
            continue
        categories = {phrase for phrase, row in evidence.items() if obj.category in row['asset_categories']}
        assigned = task_by_name.get(obj.name, [])
        if assigned:
            categories.update(case['category_prompts'][bddl_category(bddl)] for bddl, _ in assigned)
        if not categories:
            continue
        if assigned:
            bddl = next((b for b, _ in assigned if b in case['target_ids']), assigned[0][0])
            label = sim.tracked_label(bddl)
            canonical = case['category_prompts'][bddl_category(bddl)]
            identifier = bddl
        else:
            label = f'scene_{obj.name}'
            sim.objects[label] = obj
            canonical = min(categories, key=lambda p: (len(evidence[p]['asset_categories']), p))
            identifier = f'scene:{obj.name}'
        objects.append({'id': identifier, 'category': canonical, 'categories': sorted(categories),
                        'asset_category': obj.category, 'scene_name': obj.name,
                        'task_ids': [b for b, _ in assigned],
                        'target': any(b in case['target_ids'] for b, _ in assigned),
                        'task_object': bool(assigned), 'fixed_base': bool(obj.fixed_base), 'label': label})
    missing = set(case['target_ids']) - {b for row in objects for b in row['task_ids']}
    unsupported = sorted(b for b in missing if not b.startswith(('floor.', 'lawn.')))
    if unsupported:
        raise ValueError(f'task targets missing from complete scene labels: {unsupported}')
    for phrase, row in evidence.items():
        row['precision_graded'] = bool(row['asset_categories'])
        row['scene_instance_count'] = sum(phrase in obj['categories'] for obj in objects)
    coverage['precision_categories'] = sorted(p for p, r in evidence.items() if r['precision_graded'])
    coverage['precision_excluded_categories'] = sorted(p for p, r in evidence.items() if not r['precision_graded'])
    coverage['precision_coverage'].update(
        rule='all loaded scene objects in every descendant asset category of each supplied task synset',
        broad_unmatched_predictions='false positives for covered supplied categories',
        recall_scope='all loaded scene instances matching supplied task prompts; one row per physical instance',
        loaded_scene_objects_scanned=len(sim.env.scene.objects),
        overlap_policy='category is canonical; categories lists every accepted supplied prompt; one-to-one matching',
    )
    return objects, coverage


def assert_restored(saved, current, tolerance=1e-5):
    """Check every archived tracked pose/joint against the new restoration before using saved depth."""
    errors = {}

    def compare(name, old, new, quaternion=False):
        a, b = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
        if a.shape != b.shape:
            raise ValueError(f'restore shape mismatch: {name}')
        if quaternion and np.linalg.norm(a + b) < np.linalg.norm(a - b):
            b = -b
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        errors[name] = error
        if error > tolerance:
            raise ValueError(f'archived capture pose mismatch: {name}: {error} > {tolerance}')

    for i, part in enumerate(('position', 'quaternion')):
        compare(f'robot_{part}', saved['robot_pose'][i], current['robot_pose'][i], i == 1)
    compare('robot_qpos', saved['robot_qpos'], current['robot_qpos'])
    for label, pose in saved['object_poses'].items():
        if label not in current['object_poses']:
            raise ValueError(f'previously tracked object absent after restore: {label}')
        for i, part in enumerate(('position', 'quaternion')):
            compare(f'{label}_{part}', pose[i], current['object_poses'][label][i], i == 1)
    for label, qpos in saved['object_qpos'].items():
        compare(f'{label}_qpos', qpos, current['object_qpos'][label])
    return {'tolerance': tolerance, 'maximum_absolute_error': max(errors.values()), 'components_checked': len(errors)}


def restore_archived_object_poses(sim, saved):
    """Reconstruct recorded object poses after snapshot restoration, without taking a physics step."""
    import torch

    adjustments = []
    current = {label: [value.detach().cpu().numpy() for value in obj.get_position_orientation()]
               for label, obj in sim.objects.items()}
    for label, pose in saved['object_poses'].items():
        obj = sim.objects[label]
        position_error = float(np.max(np.abs(current[label][0] - pose[0])))
        quat = np.asarray(pose[1])
        orientation_error = float(min(np.max(np.abs(current[label][1] - quat)),
                                      np.max(np.abs(current[label][1] + quat))))
        qpos = saved['object_qpos'].get(label)
        joint_error = 0.0 if qpos is None else float(np.max(np.abs(
            obj.get_joint_positions().detach().cpu().numpy() - qpos)))
        if max(position_error, orientation_error, joint_error) > 0:
            obj.set_position_orientation(torch.tensor(pose[0], dtype=torch.float32),
                                         torch.tensor(pose[1], dtype=torch.float32))
            if qpos is not None:
                obj.set_joint_positions(torch.tensor(qpos, dtype=torch.float32))
            adjustments.append({'label': label, 'position_max_abs_before_m': position_error,
                                'quaternion_max_abs_before': orientation_error,
                                'joint_max_abs_before': joint_error})
    return {'method': 'set recorded capture object world poses and joints after snapshot restore; no physics/render',
            'adjusted_objects': adjustments, 'tolerance_relaxed': False}


def relabel_case(og, sim, case, source, out, restore_snapshot=True, reconstruct_poses=False):
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from omnigibson.tiptop.host.demo_cases import restore
    from omnigibson.tiptop.gt_masks import masks_from_geometry

    start = time.monotonic()
    directory = out / case['id']
    if directory.exists():
        raise FileExistsError(directory)
    origin = source / case['id']
    protected = {name: sha256(origin / name) for name in ('input.json', 'input.npz', 'labels.json', 'labels.npz')}
    old = json.loads((origin / 'labels.json').read_text())
    metadata = json.loads((origin / 'input.json').read_text())
    if metadata['source_snapshot_sha256'] != case['snapshot_sha256']:
        raise ValueError('source input snapshot does not match manifest')
    with np.load(origin / 'input.npz', allow_pickle=False) as archive:
        inputs = dict(archive)
    if restore_snapshot:
        restore(sim.env, json.loads((ROOT / case['snapshot']).read_text()), case['instance'], case['mode'])
    objects, coverage = expanded_objects(sim, case)
    reconstruction = restore_archived_object_poses(sim, old['capture_stamp']) if reconstruct_poses else None
    sim._link_meshes.clear()
    before = _stamp(og, sim)
    pose_audit = assert_restored(old['capture_stamp'], before)
    world_from_base = np.eye(4)
    world_from_base[:3, :3] = Rotation.from_quat(old['capture_stamp']['robot_pose'][1]).as_matrix()
    world_from_base[:3, 3] = old['capture_stamp']['robot_pose'][0]
    meshes = sim.object_meshes([row['label'] for row in objects])
    empty = [label for label, mesh in meshes.items() if mesh is None or not len(mesh.faces)]
    if empty:
        raise ValueError(f'label geometry missing: {empty}')
    labels, strict = {}, {}
    for view in VIEWS:
        transform = world_from_base @ inputs[f'{view}_base_from_cam']
        for tolerance, target in ((0.008, labels), (0.004, strict)):
            masks = masks_from_geometry(inputs[f'{view}_depth'], inputs[f'{view}_K'], transform, meshes, tolerance)
            target[f'{view}_masks'] = np.stack([masks[row['label']] for row in objects]).astype(bool)
        _assert_frozen(before, _stamp(og, sim))
    validate_arrays(inputs, labels, len(objects))
    validate_arrays(inputs, strict, len(objects))
    for i, row in enumerate(objects):
        row['pose_evidence'] = ('archived_capture_pose_matched'
                                if row['label'] in old['capture_stamp']['object_poses']
                                else 'same_restored_scene_preset_unarchived_distractor')
        row['aabb_base'] = _aabb_base(sim, sim.objects[row['label']])
        row['visible_pixels'] = {view: int(labels[f'{view}_masks'][i].sum()) for view in VIEWS}
        row['mask_quality'] = {}
        for view in VIEWS:
            main, tight = labels[f'{view}_masks'][i], strict[f'{view}_masks'][i]
            union = int((main | tight).sum())
            pixels = row['visible_pixels'][view]
            row['mask_quality'][view] = {
                'visible_pixels_8mm': pixels, 'visible_pixels_4mm': int(tight.sum()),
                'tolerance_iou_4mm_8mm': float((main & tight).sum() / union) if union else None,
                'changed_pixels_4mm_8mm': int((main ^ tight).sum()),
                'visibility': 'visible' if pixels >= 25 else ('tiny' if pixels else 'invisible'),
                'interpretation': 'tolerance sensitivity only; neither mask is renderer instance ground truth',
            }
        del row['label']
    after = _stamp(og, sim)
    _assert_frozen(before, after)
    truth = dict(old, objects=objects, label_revision='taxonomy_v2', split=case['split'],
                 label_scope='all loaded scene instances in taxonomy descendants of fixed task prompts',
                 label_source='oracle_geometry', geometry_tolerance_m=0.008,
                 strict_geometry_tolerance_m=0.004, strict_masks_file='labels_strict4mm.npz',
                 relabel_source=str(origin), source_file_sha256=protected,
                 relabel_restore_stamp=before, relabel_finish_stamp=after, archived_pose_match=pose_audit,
                 archived_pose_reconstruction=reconstruction,
                 renders_during_relabel=0, physics_steps_during_relabel=0,
                 label_state_origin='snapshot_restore' if restore_snapshot else 'same_frozen_capture',
                 pose_validation_scope='all archived tracked instances; newly added pilot distractors use the '
                                       'same restored scene preset without an archived capture pose',
                 geometry_limitations='Approximate proximity labels, contact halos and mesh/depth errors possible; '
                                      '4mm-vs-8mm sensitivity is diagnostic, not a calibrated accuracy bound.')
    truth.update(coverage)
    directory.mkdir(parents=True)
    for name in ('input.json', 'input.npz', *[f'{view}.png' for view in VIEWS]):
        os.link(origin / name, directory / name)
    np.savez_compressed(directory / 'labels.npz', **labels)
    np.savez_compressed(directory / 'labels_strict4mm.npz', **strict)
    (directory / 'labels.json').write_text(json.dumps(truth, indent=2) + '\n')
    for view in VIEWS:
        overlay = inputs[f'{view}_rgb'].copy()
        for i, mask in enumerate(labels[f'{view}_masks']):
            color = np.array([(67 * i + 71) % 255, (131 * i + 139) % 255, (193 * i + 211) % 255])
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)
        Image.fromarray(overlay).save(directory / f'{view}_oracle.png')
    if protected != {name: sha256(origin / name) for name in protected}:
        raise RuntimeError('source inputs or labels changed during relabel')
    return {'id': case['id'], 'ok': True, 'split': case['split'], 'objects': len(objects),
            'old_objects': len(old['objects']), 'wall_s': round(time.monotonic() - start, 2),
            'archived_pose_match': pose_audit, 'precision_categories': truth['precision_categories']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('visionbench_cases_v2.json'))
    parser.add_argument('--source-corpus', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task')
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--restore-archived-poses', action='store_true',
                        help='test-case recovery: reconstruct archived expanded object poses before strict validation')
    args = parser.parse_args(argv)
    cases = read_cases(args.manifest, args.task, args.ids)
    if any(case.get('split') not in ('dev', 'test') for case in cases):
        parser.error('v2 requires an explicit dev/test manifest')
    if args.restore_archived_poses and any(case['split'] != 'test' for case in cases):
        parser.error('explicit archived-pose reconstruction is restricted to new test captures')
    if args.validate:
        print(json.dumps({'ok': True, 'cases': len(cases), 'task_count': len({c['task'] for c in cases}),
                          'splits': {s: sum(c['split'] == s for c in cases) for s in ('dev', 'test')}}))
        return
    if len({case['task'] for case in cases}) != 1:
        parser.error('one task per process required')
    args.out.mkdir(parents=True, exist_ok=True)
    import omnigibson as og
    from omnigibson.eval.evaluator import DISABLED_TRANSITION_RULES
    from omnigibson.tiptop.r1pro import R1ProSim, challenge_task_info, make_r1pro_env_config
    from omnigibson.tiptop.run import setup_logging

    setup_logging()
    for rule in DISABLED_TRANSITION_RULES:
        rule.ENABLED = False
    scene, rooms = challenge_task_info(cases[0]['task'])
    config = make_r1pro_env_config(scene_model=scene, load_room_instances=rooms, activity=cases[0]['task'],
                                  grasping_mode='assisted', camera='head', views=VIEWS[1:], segmentation=False)
    failed = False
    try:
        sim = R1ProSim(config, camera='head', views=VIEWS[1:], look_arm=None)
        for case in cases:
            try:
                source = args.source_corpus
                restore_snapshot = True
                if not (source / case['id'] / 'input.json').exists():
                    if case['split'] != 'test':
                        raise FileNotFoundError(f'development source input missing: {case["id"]}')
                    source = args.out / '_source_captures'
                    capture_case(og, sim, case, source, labeler=lambda host, item: expanded_objects(host, item)[0])
                    restore_snapshot = False
                result = relabel_case(og, sim, case, source, args.out, restore_snapshot=restore_snapshot,
                                      reconstruct_poses=args.restore_archived_poses)
            except Exception as error:
                failed = True
                result = {'id': case['id'], 'ok': False, 'error': str(error), 'traceback': traceback.format_exc()}
                directory = args.out / case['id']
                directory.mkdir(parents=True, exist_ok=True)
                (directory / 'failure.json').write_text(json.dumps(result, indent=2) + '\n')
                log.exception('v2 labeling failed for %s', case['id'])
            with (args.out / 'capture_results.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            print(json.dumps(result), flush=True)
    finally:
        if og.app is not None:
            og.shutdown()
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
