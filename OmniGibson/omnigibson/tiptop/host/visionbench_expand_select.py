"""Lock a deterministic, task-disjoint 60-case expansion from existing snapshots and the local demo catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from bddl.object_taxonomy import ObjectTaxonomy

ROOT = Path(__file__).resolve().parents[4]
TASKS = {
    'small_thin': ['dispose_of_batteries', 'clean_a_keyboard', 'installing_smoke_detectors', 'setting_mousetraps'],
    'tools_long': ['bringing_in_wood', 'attach_a_camera_to_a_tripod', 'put_together_a_basic_pruning_kit',
                   'spraying_for_bugs'],
    'containers': ['assembling_gift_baskets', 'boxing_books_up_for_storage', 'collecting_childrens_toys',
                   'preparing_lunch_box'],
    'articulation_storage': ['can_meat', 'cook_a_frozen_pie', 'make_gift_bags_for_baby_showers', 'storing_food'],
    'mixed_household': ['installing_a_modem', 'putting_shoes_on_rack', 'picking_up_trash', 'vacuuming_floors'],
}
ALIASES = {'ashcan': 'trash can', 'camera_tripod': 'camera tripod', 'digital_camera': 'digital camera',
           'electric_refrigerator': 'refrigerator', 'fire_alarm': 'smoke detector', 'hallstand': 'shoe rack',
           'insectifuge__atomizer': 'insect spray bottle', 'pot_plant': 'potted plant', 'teddy': 'teddy bear',
           'television_receiver': 'television', 'vacuum': 'vacuum cleaner', 'pruner': 'pruning shears'}
ACTIONS = ['place in', 'place on next to', 'place on', 'turn on switch', 'press', 'open drawer', 'close drawer',
           'open door', 'close door', 'hold', 'pick up from', 'push to', 'turn off switch']
SEED = 20260927


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_ids(value):
    if isinstance(value, dict):
        if value.get('__type__') == 'ObjRef':
            yield value['id']
        for child in value.values():
            yield from reference_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from reference_ids(child)


def plain(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


def task_info(task, taxonomy, data):
    text = (ROOT / 'bddl3/bddl/activity_definitions' / task / 'problem0.bddl').read_text()
    objects = text.split('(:objects', 1)[1].split('(:init', 1)[0]
    synsets = sorted(set(re.findall(r'-\s+([^\s()]+\.n\.\d+)', objects)))
    categories = [s.partition('.n.')[0] for s in synsets if taxonomy.get_subtree_categories(s)
                  and not s.startswith(('agent.', 'floor.', 'lawn.'))]
    prompts = {category: ALIASES.get(category, category.replace('__', '_').replace('_', ' '))
               for category in categories}
    available = yaml.safe_load((data / '2026-challenge-task-instances/metadata/available_tasks.yaml').read_text())
    scene = available[task][0]['scene_model']
    template = data / '2026-challenge-task-instances/scenes' / scene / 'json' / f'{scene}_task_{task}_0_0_template.json'
    names = json.loads(template.read_text())['metadata']['task']['inst_to_name']
    scoped = {name: bddl for bddl, name in names.items() if bddl.partition('.n.')[0] in prompts}
    return prompts, scoped


def cached_inventory():
    records = {}
    for path in sorted((ROOT / 'tiptop/b1k/skills/bench').glob('*.yaml')):
        docs = yaml.safe_load(path.read_text())
        if not isinstance(docs, list):
            continue
        for row in docs:
            snapshot = row.get('demo', {}).get('snapshot')
            if snapshot and (path.parent / snapshot).is_file() and 'ready' not in row['id']:
                if row.get('demo', {}).get('fidelity', {}).get('depth', {}).get('within_5cm', 0) < .95:
                    continue
                records.setdefault(row['id'], (row, path, path.parent / snapshot))
    return records.values()


def make_cached(doc, source, snapshot, prompts, stratum):
    demo = doc['demo']
    targets = sorted({key for key in reference_ids(doc['call']) if key.partition('.n.')[0] in prompts})
    return {'id': doc['id'], 'task': doc['task'], 'instance': doc['instance'], 'mode': 'train',
            'episode_index': demo['episode_index'], 'frame': int(re.search(r'\.f(\d+)', doc['id']).group(1)),
            'source_case': str(source.relative_to(ROOT)), 'snapshot': str(snapshot.relative_to(ROOT)),
            'snapshot_sha256': sha(snapshot), 'source_materialization_hold_steps': 10,
            'category_prompts': prompts, 'target_ids': targets, 'fidelity': demo['fidelity'],
            'split': 'test', 'cohort': 'expanded_test', 'stratum': stratum,
            'tags': [stratum, 'cached_human_snapshot', demo['human_skill']],
            'stage': 'annotated_start_cached', 'human_skill': demo['human_skill'],
            'materialization': {'required': False, 'settle_steps': 10}}


def make_replay(row, slot, prompts, scoped, stratum):
    row = json.loads(json.dumps(row, default=plain))
    start, end = int(row['start_frame']), int(row['end_frame'])
    offsets = (0, min(150, max(1, (end - start) // 2)), min(300, max(1, (end - start) * 3 // 4)))
    frame = min(end - 1, start + offsets[slot])
    if frame < start:
        raise ValueError('empty manipulation segment')
    identifier = f"expanded.{row['task']}.e{row['episode_index']}.f{frame}"
    targets = sorted({scoped[name] for name in row['objects'] if name in scoped})
    original_start = row['start_frame']
    row['start_frame'] = frame  # existing materialize() replays exactly to this selected timestep
    return {'id': identifier, 'task': row['task'], 'instance': int(row['instance']), 'mode': 'train',
            'episode_index': int(row['episode_index']), 'frame': frame,
            'snapshot': f'runs/vision/expanded_snapshots/{identifier}.json', 'snapshot_sha256': None,
            'source_materialization_hold_steps': 0, 'category_prompts': prompts, 'target_ids': targets,
            'fidelity': None, 'split': 'test', 'cohort': 'expanded_test', 'stratum': stratum,
            'tags': [stratum, row['human_skill'], ('start', 'middle', 'late')[slot]],
            'stage': ('start', 'middle', 'late')[slot], 'human_skill': row['human_skill'],
            'annotation_start_frame': original_start, 'annotation_end_frame': end,
            'excluded_annotation_objects': [name for name in row['objects'] if name not in scoped],
            'materialization': {'required': True, 'method': 'replay', 'settle_steps': 0, 'catalog_row': row}}


def choose(rows, used_episodes, used_instances, used_actions, task, slot, rank):
    pool = rows[~rows.episode_index.isin(used_episodes) & ~rows.instance.isin(used_instances)].copy()
    if pool.empty:
        raise ValueError(f'not enough distinct episodes/instances: {task}')
    actions = sorted(set(pool.human_skill), key=lambda a: (a in used_actions, ACTIONS.index(a) if a in ACTIONS else len(ACTIONS)))
    pool = pool[pool.human_skill == actions[0]].sort_values(['start_frame', 'n_before', 'episode_index'])
    pool = pool.drop_duplicates('episode_index').head(32)
    def order(row):
        key = f"{SEED}:{task}:{slot}:{rank}:{row['episode_index']}:{row['start_frame']}:{row['human_skill']}"
        return hashlib.sha256(key.encode()).hexdigest()
    return min(pool.to_dict('records'), key=order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path,
                        default=Path('/home/wding8/projects/BEHAVIOR-1K/runs/skill_arch_20260925/demo_cases/catalog.parquet'))
    parser.add_argument('--data', type=Path, default=Path('/home/wding8/projects/BEHAVIOR-1K/datasets'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('selection file already exists; preserve it and use a new versioned path')
    previous = json.loads(Path(__file__).with_name('visionbench_cases_v2.json').read_text())
    excluded = sorted({case['task'] for case in previous['cases']})
    tasks = [task for group in TASKS.values() for task in group]
    assert len(tasks) == len(set(tasks)) == 20 and not set(tasks) & set(excluded)
    catalog = pd.read_parquet(args.catalog)
    catalog = catalog[catalog.resolved & (catalog.start_frame >= 150) & (catalog.start_frame <= 2000)]
    taxonomy, inventory = ObjectTaxonomy(), list(cached_inventory())
    cases, backups = [], []
    for stratum, task_names in TASKS.items():
        for task in task_names:
            prompts, scoped = task_info(task, taxonomy, args.data)
            rows = catalog[catalog.task == task]
            rows = rows[[len(objects) > 0 and objects[0] in scoped for objects in rows.objects]]
            cached = [make_cached(doc, path, snap, prompts, stratum) for doc, path, snap in inventory
                      if doc['task'] == task]
            cached = [case for case in cached if case['target_ids']]
            cached.sort(key=lambda case: (case['human_skill'] == 'pick up from', case['frame'], case['id']))
            picked, used_episodes, used_instances, used_actions = [], set(), set(), set()
            if cached:
                picked.append(cached[0])
                used_episodes.add(cached[0]['episode_index'])
                used_instances.add(cached[0]['instance'])
                used_actions.add(cached[0]['human_skill'])
            for slot in range(len(picked), 3):
                row = choose(rows, used_episodes, used_instances, used_actions, task, slot, 'primary')
                picked.append(make_replay(row, slot, prompts, scoped, stratum))
                used_episodes.add(row['episode_index'])
                used_instances.add(row['instance'])
                used_actions.add(row['human_skill'])
            for slot, case in enumerate(picked):
                case['task_slot'] = slot
                case['selection_rank'] = 'primary'
                row = choose(rows, used_episodes, used_instances, {case['human_skill']}, task, slot, 'backup')
                backup = make_replay(row, slot, prompts, scoped, stratum)
                backup.update(task_slot=slot, selection_rank='backup', replaces=case['id'])
                used_episodes.add(row['episode_index'])
                used_instances.add(row['instance'])
                backups.append(backup)
            cases.extend(picked)
    result = {'schema': 'visionbench-cases/1', 'benchmark_version': 3,
              'selection_locked_before_v3_predictions': True, 'cohort': 'expanded_test',
              'selection_seed': SEED, 'excluded_previous_tasks': excluded,
              'reference_manifest_sha256': sha(Path(__file__).with_name('visionbench_cases_v2.json')),
              'catalog_sha256': sha(args.catalog), 'cases': cases, 'backup_cases': backups,
              'selection_policy': '20 fixed coverage tasks x 3 distinct episodes and instances. Prefer one existing '
                                  'human snapshot with original depth fidelity >=95%; choose remaining resolved '
                                  'task-object segments from earliest32 distinct-episode candidates per action, '
                                  'hash ranked with seed20260927. Prefer unused action classes. New segments start '
                                  'at150..2000; slots use start/middle/late offsets capped at0/150/300 frames.',
              'backup_policy': 'One predeclared backup per primary; only infrastructure/materialization/capture '
                               'failure permits substitution. Invisible/tiny targets and poor model predictions '
                               'never trigger replacement. Preserve failure records and seal a resolved manifest '
                               'before any expanded model inference.',
              'materialization_policy': 'New cases replay recorded actions and settle0; cached cases retain '
                                        'their documented10 hold steps. Capture and label without physics steps.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    pending = [case for case in cases if case['materialization']['required']]
    print(json.dumps({'manifest': str(args.out), 'sha256': sha(args.out), 'cases': len(cases), 'tasks': len(tasks),
                      'cached': len(cases) - len(pending), 'new_replays': len(pending),
                      'replay_frames': sum(case['frame'] for case in pending),
                      'max_new_frame': max(case['frame'] for case in pending),
                      'distinct_episodes': len({case['episode_index'] for case in cases}),
                      'actions': dict(pd.Series(case['human_skill'] for case in cases).value_counts())}, default=plain))


if __name__ == '__main__':
    main()
