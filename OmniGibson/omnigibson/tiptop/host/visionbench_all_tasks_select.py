"""Freeze three independent, annotation-selected episodes for each official evaluation task.

This is a metadata-only selector. No RGB, geometry visibility, labels or predictions are
read. The output is immutable and contains no substitutions based on model outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import pandas as pd
from bddl.object_taxonomy import ObjectTaxonomy
from visionbench_expand_select import ALIASES, plain

ROOT = Path(__file__).resolve().parents[4]
HOST = Path(__file__).resolve().parent
SOURCES = ('visionbench_expand.py', 'visionbench_capture.py', 'visionbench.py', 'visionbench_v2.py',
           'visionbench_coverage.py', 'demo_cases.py', 'visionbench_all_tasks_capture.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def references(value):
    if isinstance(value, dict):
        for key in ('reference_episode', 'source_episode', 'episode_index'):
            if key in value and isinstance(value[key], int):
                yield value[key]
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def scope(task, taxonomy):
    objects = [o for o in task['objects'] if not o['bddl_variable'].startswith(('agent.', 'floor.', 'lawn.'))]
    # Include future rigid categories in prompts but do not pretend their instances already exist.
    stems = {o['bddl_variable'].partition('.n.')[0] for o in objects}
    for entity in task.get('future_entities', []):
        if entity.get('categories'):
            stems.add(entity['bddl_variable'].partition('.n.')[0])
    prompts = {s: ALIASES.get(s, s.replace('__', '_').replace('_', ' ')) for s in sorted(stems)}
    lookup = defaultdict(set)
    for obj in objects:
        for key in (obj['object_name'], obj['bddl_variable'], obj['bddl_variable'].partition('.n.')[0],
                    obj['asset_id'].partition('.')[0]):
            lookup[key].add(obj['bddl_variable'])
    return prompts, lookup, objects


def target_ids(row, lookup):
    return sorted(set().union(*(lookup.get(name, set()) for name in row['objects'] + row['human_objects'])))


def select_rows(rows, lookup, task, seed, count):
    eligible = []
    for raw in rows.to_dict('records'):
        row = json.loads(json.dumps(raw, default=plain))
        row['_targets'] = target_ids(row, lookup)
        if row['_targets'] and 150 <= row['start_frame'] < row['end_frame']:
            eligible.append(row)
    used_episodes, used_actions, used_target_categories = set(), set(), set()
    chosen = []
    for slot in range(count):
        pool = [r for r in eligible if r['episode_index'] not in used_episodes]
        bounded = [r for r in pool if r['start_frame'] <= 2000]
        pool = bounded or pool
        # Preserve a bounded, early replay pool for every annotated action, then prioritize
        # metadata object/action diversity. This never ranks visibility or detection quality.
        candidates = []
        for action in sorted({r['human_skill'] for r in pool}):
            seen = set()
            for row in sorted((r for r in pool if r['human_skill'] == action),
                              key=lambda r: (r['start_frame'], r['episode_index'], r['end_frame'])):
                if row['episode_index'] in seen:
                    continue
                seen.add(row['episode_index'])
                candidates.append(row)
                if len(seen) == 32:
                    break
        if not candidates:
            raise ValueError(f'Fewer than {count} eligible distinct episodes: {task}')
        def rank(row):
            categories = {t.partition('.n.')[0] for t in row['_targets']}
            token = f"{seed}:{task}:{slot}:{row['episode_index']}:{row['start_frame']}:{row['human_skill']}"
            return (-len(categories - used_target_categories), row['human_skill'] in used_actions,
                    hashlib.sha256(token.encode()).hexdigest())
        picked = min(candidates, key=rank)
        chosen.append(picked)
        used_episodes.add(picked['episode_index'])
        used_actions.add(picked['human_skill'])
        used_target_categories.update(t.partition('.n.')[0] for t in picked['_targets'])
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--exclude', type=Path, action='append', required=True,
                        help='Prior query selection or reference bank; may be repeated.')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--snapshots-per-task', type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Selection already exists; use a new versioned output.')
    if args.snapshots_per_task < 3:
        parser.error('At least three independent snapshots per task required.')
    inventory = read(args.inventory)
    tasks = sorted(inventory['tasks'], key=lambda t: t['task_index'])
    if len(tasks) != 100 or len({t['task_name'] for t in tasks}) != 100:
        raise ValueError('Official evaluation inventory must contain exactly 100 tasks')
    excluded = set()
    for path in args.exclude:
        value = read(path)
        excluded.update(references(value))
        excluded.update(value.get('excluded_episodes', []))
        if isinstance(value.get('episodes'), list) and all(isinstance(x, int) for x in value['episodes']):
            excluded.update(value['episodes'])
    catalog = pd.read_parquet(args.catalog)
    catalog = catalog[~catalog.episode_index.isin(excluded)]
    taxonomy = ObjectTaxonomy()
    cases, task_records = [], []
    for task in tasks:
        name = task['task_name']
        prompts, lookup, objects = scope(task, taxonomy)
        rows = select_rows(catalog[catalog.task == name], lookup, name, args.seed, args.snapshots_per_task)
        for slot, row in enumerate(rows):
            start, end = int(row['start_frame']), int(row['end_frame'])
            phase = slot % 3
            offset = (0, min(150, (end - start) // 2), min(300, 85 * (end - start) // 100))[phase]
            frame = min(end - 1, start + offset)
            episode = int(row['episode_index'])
            identifier = f'comprehensive.{name}.e{episode}.f{frame}'
            annotation = args.dataset / f"annotations/task-{task['task_index']:04d}" / f"episode_{int(row['raw_episode_id']):08d}.json"
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            targets = row.pop('_targets')
            row['start_frame'] = frame
            cases.append({'id': identifier, 'task': name, 'task_index': task['task_index'],
                          'instance': int(row['instance']), 'mode': 'train', 'episode_index': episode, 'frame': frame,
                          'snapshot': str(args.out.resolve().parent / 'snapshots' / f'{identifier}.json'),
                          'snapshot_sha256': None, 'source_materialization_hold_steps': 0,
                          'category_prompts': prompts, 'target_ids': targets,
                          'all_initial_task_object_ids': sorted(o['bddl_variable'] for o in objects),
                          'split': 'test', 'cohort': 'comprehensive_test', 'stratum': 'all_eval_tasks',
                          'tags': ['all_eval_tasks', row['human_skill'], ('start', 'middle', 'late')[phase]],
                          'stage': ('start', 'middle', 'late')[phase], 'human_skill': row['human_skill'],
                          'task_slot': slot, 'annotation_start_frame': start, 'annotation_end_frame': end,
                          'annotation_path': str(annotation), 'annotation_sha256': sha(annotation),
                          'annotation_instance_resolved': bool(row['resolved']),
                          'target_resolution': 'All matching task instances for every annotated object; generic categories may map to multiple instances.',
                          'selection_rank': 'primary',
                          'materialization': {'required': True, 'method': 'replay', 'settle_steps': 0, 'catalog_row': row}})
        task_records.append({'task': name, 'task_index': task['task_index'], 'scene_model': task['scene_model'],
                             'case_ids': [c['id'] for c in cases if c['task'] == name],
                             'initial_task_objects': len(objects), 'supplied_categories': len(prompts),
                             'systems_outside_rigid_mask_scope': task.get('systems', []),
                             'future_entities': task.get('future_entities', [])})
    assert len(cases) == 100 * args.snapshots_per_task
    assert len({c['episode_index'] for c in cases}) == len(cases)
    assert not {c['episode_index'] for c in cases} & excluded
    result = {'schema': 'visionbench-all-tasks-selection/1', 'selection_seed': args.seed,
              'selection_frozen_before_capture_and_inference': True, 'query_labels_or_predictions_used_for_selection': False,
              'inventory': {'path': str(args.inventory.resolve()), 'sha256': sha(args.inventory)},
              'catalog': {'path': str(args.catalog.resolve()), 'sha256': sha(args.catalog)},
              'dataset': str(args.dataset.resolve()), 'snapshots_per_task': args.snapshots_per_task,
              'exclusion_source_receipts': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in args.exclude],
              'excluded_episodes': sorted(excluded), 'tasks': task_records, 'cases': cases, 'backup_cases': [],
              'source_receipts': {str(HOST / name): sha(HOST / name) for name in SOURCES},
              'selector': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
              'selection_policy': 'All100 official tasks. Three distinct episode queries/task; exclude every prior query and all reference-source episodes. Within each action retain the earliest32 eligible distinct episodes starting150..2000 (unbounded fallback only if none). Prefer new annotated target categories then unused actions, hash rank remaining ties. Start/middle/late are stages within separate manipulation segments, not whole-episode progress; offsets0/min150,50%/min300,85%.',
              'failure_policy': 'No substitution. Infrastructure failures stay visible and may retry exactly the same case; low visibility, replay quality and model outcomes never remove or replace cases.',
              'target_policy': 'Primary recall covers annotated manipulation targets including all plausible generic-category instances. All loaded task objects and supplied-category scene objects support diagnostic recall/precision. Particles/systems are outside rigid-instance geometry scoring.',
              'interpretation': 'Prospective episode holdout on all100 known tasks/assets; three episodes per task. No unseen-task or unseen-asset generalization claim.',
              'counts': {'tasks': 100, 'cases': len(cases), 'views': 3 * len(cases),
                         'episodes': len({c['episode_index'] for c in cases}), 'excluded_episodes': len(excluded),
                         'replay_frames': sum(c['frame'] for c in cases), 'max_frame': max(c['frame'] for c in cases),
                         'actions': dict(Counter(c['human_skill'] for c in cases))}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'path': str(args.out), 'sha256': sha(args.out), **result['counts']}, indent=2))


if __name__ == '__main__':
    main()
