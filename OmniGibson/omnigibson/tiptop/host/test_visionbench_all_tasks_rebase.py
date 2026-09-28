"""Portability and provenance protection; tests use no simulator or external artifacts."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visionbench_all_tasks_rebase import HOST_RELATIVE, rebase_selection, remap_path


def file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def deployment(tmp_path):
    old = tmp_path/'old host'/'behavior'
    new = tmp_path/'coworker host'/'behavior'
    old_dataset, dataset = tmp_path/'old demos', tmp_path/'shared demos'
    sim_data = tmp_path/'sim assets'
    sim_data.mkdir()
    old_other, other = tmp_path/'old vision', tmp_path/'new vision'
    source_name = HOST_RELATIVE/'visionbench_all_tasks_capture.py'
    source_hash = file(new/source_name, b'capture source identical\n')
    annotation_hash = file(dataset/'annotations/task-0001/episode_00000010.json', b'{"event": "pick up"}\n')
    catalog_hash = file(other/'catalog.parquet', b'unchanged catalog bytes')
    inventory_hash = file(other/'inventory.json', b'{"tasks": [1]}')
    python = new/'venv/bin/python'
    file(python, b'existing interpreter stub')
    parent = tmp_path/'archived_selection.json'
    original = {'source_receipts': {str(old/source_name): source_hash},
                'dataset': str(old_dataset),
                'catalog': {'path': str(old_other/'catalog.parquet'), 'sha256': catalog_hash},
                'inventory': {'path': str(old_other/'inventory.json'), 'sha256': inventory_hash},
                'cases': [{'id': 'case1', 'episode_index': 19, 'frame': 150, 'target_ids': ['toy.n.01_1'],
                           'category_prompts': {'toy': 'toy figure'},
                           'snapshot': str(old/'runs/old_snapshots/case1.json'),
                           'annotation_path': str(old_dataset/'annotations/task-0001/episode_00000010.json'),
                           'annotation_sha256': annotation_hash,
                           'materialization': {'required': True, 'settle_steps': 0}}]}
    file(parent, json.dumps(original).encode())
    kwargs = dict(parent_path=parent, parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                  checkout=new, dataset=dataset, sim_data=sim_data, snapshot_root=tmp_path/'new snapshots',
                  capture_root=tmp_path/'new capture', output=tmp_path/'replayed_selection.json', python=python,
                  mappings=[(old_other, other)])
    return original, kwargs


def test_relocated_replay_preserves_selection_and_parent_bytes(deployment):
    original, kwargs = deployment
    before = copy.deepcopy(original)
    parent_bytes = kwargs['parent_path'].read_bytes()
    result, receipt = rebase_selection(original, **kwargs)
    assert original == before
    assert kwargs['parent_path'].read_bytes() == parent_bytes
    assert result['cases'][0]['episode_index'] == original['cases'][0]['episode_index']
    assert result['cases'][0]['frame'] == original['cases'][0]['frame']
    assert result['cases'][0]['target_ids'] == original['cases'][0]['target_ids']
    assert result['cases'][0]['category_prompts'] == original['cases'][0]['category_prompts']
    assert result['cases'][0]['materialization'] == original['cases'][0]['materialization']
    assert result['cases'][0]['snapshot'] == str(kwargs['snapshot_root']/'case1.json')
    assert result['cases'][0]['annotation_path'].startswith(str(kwargs['dataset']))
    assert set(result['source_receipts']) == {str(kwargs['checkout']/HOST_RELATIVE/'visionbench_all_tasks_capture.py')}
    assert result['replay_deployment']['parent_selection']['sha256'] == kwargs['parent_sha256']
    assert len(receipt['verified_dependencies']) == 4
    assert not kwargs['output'].exists()  # Pure helper does not write any artifact.


@pytest.mark.parametrize('changed', ['annotation', 'source', 'catalog', 'inventory'])
def test_changed_active_dependency_is_rejected(deployment, changed):
    original, kwargs = deployment
    path = {'annotation': kwargs['dataset']/'annotations/task-0001/episode_00000010.json',
            'source': kwargs['checkout']/HOST_RELATIVE/'visionbench_all_tasks_capture.py',
            'catalog': kwargs['mappings'][0][1]/'catalog.parquet',
            'inventory': kwargs['mappings'][0][1]/'inventory.json'}[changed]
    path.write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='hash differs'):
        rebase_selection(original, **kwargs)


def test_existing_snapshot_and_capture_outputs_are_protected(deployment):
    original, kwargs = deployment
    file(kwargs['snapshot_root']/'case1.json', b'previous snapshot')
    with pytest.raises(ValueError, match='snapshot path already exists'):
        rebase_selection(original, **kwargs)
    (kwargs['snapshot_root']/'case1.json').unlink()
    file(kwargs['capture_root']/'existing.txt', b'previous results')
    with pytest.raises(ValueError, match='Capture root must be fresh'):
        rebase_selection(original, **kwargs)


def test_mapping_uses_path_boundaries_and_longest_prefix(tmp_path):
    assert remap_path('/data/demo/file', [(Path('/data'), Path('/new')), (Path('/data/demo'), Path('/special'))]) == '/special/file'
    assert remap_path('/database/file', [(Path('/data'), Path('/new'))]) == '/database/file'
