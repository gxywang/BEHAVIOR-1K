"""Add metadata-only category coverage queries to a frozen 300-case base selection.

All additions are declared before simulator captures or model inference. They are
selected from annotations, never based on visibility, model scores or query images.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
import visionbench_all_tasks_select as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-extra-per-task', type=int, default=3)
    parser.add_argument('--extra-cap-override', action='append', default=[], metavar='TASK=N')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Final selection exists; never overwrite.')
    selection = base.read(args.base)
    inventory = base.read(selection['inventory']['path'])
    task_info = {t['task_name']: t for t in inventory['tasks']}
    caps = {name: args.max_extra_per_task for name in task_info}
    for value in args.extra_cap_override:
        name, value = value.split('=', 1)
        if name not in caps:
            parser.error(f'Unknown task override: {name}')
        caps[name] = int(value)
    catalog = pd.read_parquet(selection['catalog']['path'])
    excluded = set(selection['excluded_episodes']) | {c['episode_index'] for c in selection['cases']}
    catalog = catalog[~catalog.episode_index.isin(excluded)]
    covered = {t.partition('.n.')[0] for c in selection['cases'] for t in c['target_ids']}
    initial_covered = sorted(covered)
    candidates, possible = [], set()
    for task in inventory['tasks']:
        prompts, lookup, objects = base.scope(task, None)
        for raw in catalog[catalog.task == task['task_name']].to_dict('records'):
            row = json.loads(json.dumps(raw, default=base.plain))
            targets = base.target_ids(row, lookup)
            if not targets or not 150 <= row['start_frame'] < row['end_frame']:
                continue
            categories = {t.partition('.n.')[0] for t in targets}
            possible.update(categories)
            if categories - covered:
                candidates.append((row, targets, categories))
    extras, per_task = [], Counter()
    while possible - covered:
        eligible = [(row, targets, categories) for row, targets, categories in candidates
                    if per_task[row['task']] < caps[row['task']] and row['episode_index'] not in excluded
                    and categories - covered]
        if not eligible:
            break
        def rank(item):
            row, targets, categories = item
            token = f"{selection['selection_seed']}:coverage:{row['task']}:{row['episode_index']}:{row['start_frame']}"
            return (-len(categories - covered), row['start_frame'], hashlib.sha256(token.encode()).hexdigest())
        row, targets, categories = min(eligible, key=rank)
        task = task_info[row['task']]
        prompts, _, objects = base.scope(task, None)
        start, end = int(row['start_frame']), int(row['end_frame'])
        # Mid-manipulation for supplements, not a visibility search.
        frame = min(end-1, start + min(150, (end-start)//2))
        identifier = f"comprehensive.{row['task']}.e{row['episode_index']}.f{frame}"
        annotation = Path(selection['dataset']) / f"annotations/task-{task['task_index']:04d}" / f"episode_{row['raw_episode_id']:08d}.json"
        row['start_frame'] = frame
        extras.append({'id': identifier, 'task': row['task'], 'task_index': task['task_index'],
                       'instance': row['instance'], 'mode': 'train', 'episode_index': row['episode_index'], 'frame': frame,
                       'snapshot': str(args.out.resolve().parent/'snapshots'/f'{identifier}.json'),
                       'snapshot_sha256': None, 'source_materialization_hold_steps': 0,
                       'category_prompts': prompts, 'target_ids': targets,
                       'all_initial_task_object_ids': sorted(o['bddl_variable'] for o in objects),
                       'split': 'test', 'cohort': 'comprehensive_test', 'stratum': 'all_eval_tasks',
                       'tags': ['all_eval_tasks', 'metadata_category_coverage', row['human_skill'], 'middle'],
                       'stage': 'middle', 'human_skill': row['human_skill'],
                       'task_slot': selection['snapshots_per_task'] + per_task[row['task']],
                       'annotation_start_frame': start, 'annotation_end_frame': end,
                       'annotation_path': str(annotation), 'annotation_sha256': base.sha(annotation),
                       'annotation_instance_resolved': bool(row['resolved']),
                       'target_resolution': 'All matching task instances for every annotated object; generic categories may map to multiple instances.',
                       'selection_rank': 'metadata_category_coverage', 'new_target_categories': sorted(categories-covered),
                       'materialization': {'required': True, 'method': 'replay', 'settle_steps': 0, 'catalog_row': row}})
        covered.update(categories)
        excluded.add(row['episode_index'])
        per_task[row['task']] += 1
    cases = selection['cases'] + extras
    assert len({c['id'] for c in cases}) == len(cases) == len({c['episode_index'] for c in cases})
    selection.update(base_selection={'path': str(args.base.resolve()), 'sha256': base.sha(args.base)},
                     cases=cases,
                     supplementary_selection={'policy': 'Greedy maximum new global annotated target category coverage; then shortest annotation start frame, then seeded hash. Distinct episodes only. Mid-segment snapshot at min150frames,50% offset. No query pixels, visibility or model predictions read.',
                         'max_extra_per_task': args.max_extra_per_task,
                         'extra_cap_overrides': {k: v for k, v in caps.items() if v != args.max_extra_per_task},
                         'initial_target_categories': initial_covered,
                         'all_annotation_eligible_target_categories': sorted(possible),
                         'final_primary_target_categories': sorted(covered),
                         'still_unselected_target_categories': sorted(possible-covered),
                         'supplemental_case_count': len(extras), 'per_task_additions': dict(per_task)},
                     finalizer={'path': str(Path(__file__).resolve()), 'sha256': base.sha(__file__)},
                     source_receipts={str(base.HOST/name): base.sha(base.HOST/name) for name in base.SOURCES})
    for task in selection['tasks']:
        task['case_ids'] = [c['id'] for c in cases if c['task'] == task['task']]
    selection['counts'].update(cases=len(cases), views=3*len(cases), episodes=len(cases),
                              replay_frames=sum(c['frame'] for c in cases), max_frame=max(c['frame'] for c in cases),
                              base_cases=len(cases)-len(extras), supplemental_cases=len(extras),
                              actions=dict(Counter(c['human_skill'] for c in cases)))
    args.out.write_text(json.dumps(selection, indent=2)+'\n')
    print(json.dumps({'path': str(args.out), 'sha256': base.sha(args.out), **selection['counts'],
                      'target_categories_covered': len(covered), 'eligible': len(possible),
                      'unselected': sorted(possible-covered)}, indent=2))


if __name__ == '__main__':
    main()
