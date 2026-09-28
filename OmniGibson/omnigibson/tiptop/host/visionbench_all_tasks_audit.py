"""Audit and optionally seal comprehensive captures without reading predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

VIEWS = ('head', 'left_wrist', 'right_wrist')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2) + '\n'
    if immutable and path.exists():
        if path.read_text() != text:
            raise ValueError(f'Sealed file already differs: {path}')
        return
    temporary = path.with_suffix('.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def distribution(values):
    values = [v for v in values if v is not None]
    return {'minimum': min(values), 'median': statistics.median(values), 'maximum': max(values)} if values else None


def audit(selection_path, captures):
    selection, selection_hash = read(selection_path), sha(selection_path)
    cases, missing, errors, files, quality = [], [], [], [], []
    task_coverage = defaultdict(lambda: {'primary_categories': set(), 'visible_primary_categories': set(),
                                       'scoped_categories': set(), 'visible_scoped_categories': set(),
                                       'models': set(), 'visible_models': set(), 'cases': 0})
    for case in selection['cases']:
        folder = captures / case['id']
        receipt_path = captures / 'case_receipts' / f"{case['id']}.json"
        if not receipt_path.is_file():
            missing.append({'case': case['id'], 'task': case['task'], 'status': 'no_success_receipt'})
            continue
        try:
            receipt = read(receipt_path)
            assert receipt['ok'] and receipt['effective_id'] == case['id'] and not receipt['attempts']
            assert receipt['selection_sha256'] == selection_hash
            inputs, labels = read(folder/'input.json'), read(folder/'labels.json')
            assert inputs['physics_steps_during_capture'] == labels['physics_steps_during_relabel'] == 0
            assert labels['cohort'] == case['cohort'] and labels['split'] == case['split']
            assert labels['asset_identity_is_evaluation_only'] is True
            assert labels['mask_provenance']['renderer_instance_ground_truth'] is False
            assert not {'asset_id', 'asset_model', 'target_ids', 'objects', 'expected_target_models'} & set(inputs)
            assert set(inputs['categories']) == set(case['category_prompts'].values())
            target_rows = [o for o in labels['objects'] if o['target']]
            assert set(case['target_ids']) <= {b for o in target_rows for b in o['task_ids']}
            materialization_path = captures/'materialization'/f"{case['id']}.json"
            materialized = read(materialization_path)['case']
            assert sha(case['snapshot']) == materialized['snapshot_sha256'] == inputs['source_snapshot_sha256']
            group = task_coverage[case['task']]
            with np.load(folder/'input.npz', allow_pickle=False) as images, \
                 np.load(folder/'labels.npz', allow_pickle=False) as masks, \
                 np.load(folder/'labels_strict4mm.npz', allow_pickle=False) as strict:
                for view in VIEWS:
                    shape = images[f'{view}_rgb'].shape[:2]
                    broad, narrow = masks[f'{view}_masks'], strict[f'{view}_masks']
                    assert broad.dtype == narrow.dtype == bool
                    assert broad.shape == narrow.shape == (len(labels['objects']), *shape)
                    assert not np.any(narrow & ~broad)
                    for index, obj in enumerate(labels['objects']):
                        assert int(broad[index].sum()) == obj['visible_pixels'][view]
                for obj in labels['objects']:
                    visible = any(obj['visible_pixels'][v] >= 25 for v in VIEWS)
                    if obj['task_object']:
                        group['scoped_categories'].add(obj['category'])
                        group['models'].add(obj['asset_id'])
                        if visible:
                            group['visible_scoped_categories'].add(obj['category'])
                            group['visible_models'].add(obj['asset_id'])
                    if obj['target']:
                        group['primary_categories'].add(obj['category'])
                        if visible:
                            group['visible_primary_categories'].add(obj['category'])
            notes = labels['render_convergence']
            assert set(notes) == set(VIEWS)
            quality.append({'case': case['id'], 'task': case['task'], 'split': case['split'],
                            'episode': case['episode_index'], 'render_convergence': notes,
                            'depth_recording_comparison': materialized['fidelity']['depth'],
                            'joint_max_rad': materialized['fidelity']['at_start']['joint_max_rad'],
                            'frozen_pose_maximum_error': receipt['archived_pose_match']['maximum_absolute_error']})
            for name in ('input.json', 'input.npz', 'labels.json', 'labels.npz', 'labels_strict4mm.npz',
                         *(v+'.png' for v in VIEWS)):
                files.append({'path': str(folder/name), 'sha256': sha(folder/name)})
            files.extend({'path': str(p), 'sha256': sha(p)} for p in (receipt_path, materialization_path))
            cases.append(materialized)
            group['cases'] += 1
        except Exception as exc:
            errors.append({'case': case['id'], 'task': case['task'], 'type': type(exc).__name__, 'error': str(exc)})
    coverage = {task: {key: sorted(value) if isinstance(value, set) else value for key, value in row.items()}
                for task, row in task_coverage.items()}
    bad_views = [{'case': row['case'], 'task': row['task'], 'split': row['split'], 'view': view,
                  'renders': note['renders']} for row in quality for view, note in row['render_convergence'].items()
                 if not note['converged']]
    complete = len(cases) == len(selection['cases']) and not errors
    result = {'schema': 'all-tasks-capture-audit/1', 'selection': str(selection_path),
              'selection_sha256': selection_hash, 'complete': complete, 'ok': not errors,
              'cases_expected': len(selection['cases']), 'cases_completed': len(cases),
              'cases_missing': missing, 'errors': errors, 'input_file_receipts': files,
              'visibility_threshold_pixels': 25, 'coverage': coverage, 'predictions_read': False,
              'replacement_cases': 0, 'selection_unchanged': True,
              'scope': 'Rigid-instance masks only; systems/future entities and floor/lawn exclusions retained in selection task records.'}
    quality_report = {'schema': 'all-tasks-capture-quality/1', 'selection_sha256': selection_hash,
                      'cases': len(cases), 'saved_views': len(cases)*3,
                      'saved_nonconverged_views': bad_views, 'saved_nonconverged_view_count': len(bad_views),
                      'saved_cases_with_nonconverged_views': len({r['case'] for r in bad_views}),
                      'replay_head_depth_median_m_distribution': distribution([q['depth_recording_comparison']['median_m'] for q in quality]),
                      'replay_head_depth_within_5cm_fraction_distribution': distribution([q['depth_recording_comparison']['within_5cm'] for q in quality]),
                      'maximum_frozen_pose_error': max((q['frozen_pose_maximum_error'] for q in quality), default=None),
                      'per_case': quality, 'predictions_read': False,
                      'limitations': ['Geometry proximity8mm masks, strict4mm subset; neither is renderer ground truth.',
                                      'Render convergence and recorded-depth fidelity are reported separately from integrity.',
                                      'All selected cases and nonconverged views retained; no visibility/performance substitution.',
                                      'At least three unseen episodes per known task; not unseen-task/asset generalization.']}
    return selection, result, quality_report, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    selection, result, quality, cases = audit(args.selection.resolve(), args.captures.resolve())
    if args.seal and not result['complete']:
        raise ValueError(f'Cannot seal incomplete capture: {len(cases)}/{len(selection["cases"])}; {result["errors"]}')
    write(args.out/'capture_audit.json', result, args.seal)
    write(args.out/'capture_quality_summary.json', quality, args.seal)
    if args.seal:
        manifest = {'schema': 'visionbench-cases/1', 'cases': cases,
                    'source_selection': str(args.selection.resolve()), 'source_selection_sha256': sha(args.selection),
                    'capture_audit': str((args.out/'capture_audit.json').resolve()),
                    'capture_audit_sha256': sha(args.out/'capture_audit.json'),
                    'capture_quality': str((args.out/'capture_quality_summary.json').resolve()),
                    'capture_quality_sha256': sha(args.out/'capture_quality_summary.json'),
                    'note': 'All predeclared primary and metadata-category supplements; no replacements or exclusions.'}
        write(args.out/'query_manifest.json', manifest, True)
        write(args.out/'seal_receipt.json', {'schema': 'all-tasks-capture-seal/1',
              'files': {str((args.out/name).resolve()): sha(args.out/name) for name in
                        ('capture_audit.json', 'capture_quality_summary.json', 'query_manifest.json')},
              'cases': len(cases), 'tasks': len({c['task'] for c in cases}),
              'auditor': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}}, True)
    print(json.dumps({'complete': result['complete'], 'ok': result['ok'], 'cases': len(cases),
                      'missing': len(result['cases_missing']), 'errors': result['errors'],
                      'nonconverged_views': quality['saved_nonconverged_view_count']}))
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
