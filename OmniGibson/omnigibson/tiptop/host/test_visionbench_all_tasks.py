"""Metadata-only regressions for comprehensive selection; no Isaac Sim launch."""
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visionbench_all_tasks_select as selector


def test_generic_annotation_retains_all_matching_instances():
    task = {'objects': [
        {'bddl_variable': 'toy.n.01_1', 'object_name': 'toy_a_0', 'asset_id': 'toy.a'},
        {'bddl_variable': 'toy.n.01_2', 'object_name': 'toy_b_0', 'asset_id': 'toy.b'},
        {'bddl_variable': 'floor.n.01_1', 'object_name': 'floor_0', 'asset_id': 'floors.f'},
    ]}
    prompts, lookup, objects = selector.scope(task, None)
    assert prompts == {'toy': 'toy'}
    assert selector.target_ids({'objects': ['toy'], 'human_objects': ['toy']}, lookup) == ['toy.n.01_1', 'toy.n.01_2']
    assert len(objects) == 2


def test_selection_deterministic_episode_disjoint_and_diverse():
    rows = pd.DataFrame([{'objects': [obj], 'human_objects': [obj], 'episode_index': episode,
                          'human_skill': action, 'start_frame': 150 + episode, 'end_frame': 1000}
                         for episode in range(10) for obj in (('a' if episode % 2 else 'b'),) for action in ('pick up from', 'place on')])
    lookup = {'a': {'a.n.01_1'}, 'b': {'b.n.01_1'}}
    left = selector.select_rows(rows, lookup, 'task', 123, 3)
    right = selector.select_rows(rows.sample(frac=1, random_state=4), lookup, 'task', 123, 3)
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
    assert len({r['episode_index'] for r in left}) == 3
    assert len({r['human_skill'] for r in left}) == 2
    assert {t for r in left for t in r['_targets']} == {'a.n.01_1', 'b.n.01_1'}


def test_later_frames_are_not_silently_dropped():
    rows = pd.DataFrame([{'objects': ['a'], 'human_objects': ['a'], 'episode_index': episode,
                          'human_skill': 'pick up from', 'start_frame': 4000 + episode, 'end_frame': 6000}
                         for episode in range(4)])
    picked = selector.select_rows(rows, {'a': {'a.n.01_1'}}, 'late_task', 123, 3)
    assert len(picked) == 3
    assert min(r['start_frame'] for r in picked) >= 4000


def test_reference_sources_include_every_nested_episode():
    document = {'by_category': {'toy': [{'source_episode': 15}, {'reference_episode': 16},
                                      {'source': {'episode_index': 17}}]}}
    assert set(selector.references(document)) == {15, 16, 17}
