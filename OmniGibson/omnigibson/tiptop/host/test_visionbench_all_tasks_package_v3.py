"""Portable artifact integrity and real mounted capture audit without Isaac Sim."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visionbench_all_tasks_package_v3 as package


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_stage_copies_original_bytes_and_rejects_mismatched_source(tmp_path):
    original = tmp_path/'original.json'
    original.write_bytes(b'{"unchanged":true}\n')
    files = []
    package.add_file(tmp_path/'stage', files, original, 'metadata/file.json', 'metadata', package.sha(original))
    assert (tmp_path/'stage/metadata/file.json').read_bytes() == original.read_bytes()
    original.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        package.add_file(tmp_path/'stage', files, original, 'metadata/other.json', 'metadata', files[0]['sha256'])


def test_artifact_checksum_and_path_boundary_protection(tmp_path):
    source = tmp_path/'original.json'
    source.write_bytes(b'{}')
    files = []
    package.add_file(tmp_path/'stage', files, source, 'file.json', 'test')
    assert package.verify_files(tmp_path/'stage', {'files': files})['ok']
    (tmp_path/'stage/file.json').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum differs'):
        package.verify_files(tmp_path/'stage', {'files': files})
    for path in ('../secret', '/absolute', 'metadata/../../escape'):
        with pytest.raises(ValueError, match='Unsafe artifact path'):
            package.safe_relative(path)
    source_mesh = tmp_path/'mesh.usd'
    source_mesh.write_bytes(b'not included')
    with pytest.raises(ValueError, match='excluded'):
        package.add_file(tmp_path/'stage', files, source_mesh, 'mesh.usd', 'forbidden')


def test_mounted_audit_uses_companion_snapshots_without_rewriting_selection(tmp_path):
    provenance, main = tmp_path/'provenance', tmp_path/'main_captures'
    case_id = 'case1'
    folder = main/case_id
    folder.mkdir(parents=True)
    snapshot = provenance/'snapshots'/f'{case_id}.json'
    dump(snapshot, {'state': [1, 2, 3]})
    snapshot_hash = package.sha(snapshot)
    case = {'id': case_id, 'task': 'task1', 'episode_index': 1,
            'snapshot': '/unavailable/original/machine/snapshots/case1.json',
            'target_ids': ['toy.n.01_1'], 'category_prompts': {'toy': 'toy figure'},
            'cohort': 'comprehensive_test', 'split': 'test'}
    selection_path = provenance/'selection/selection_fullcoverage.json'
    dump(selection_path, {'cases': [case]})
    original_selection = selection_path.read_bytes()
    selection_hash = package.sha(selection_path)
    dump(folder/'input.json', {'physics_steps_during_capture': 0, 'categories': ['toy figure'],
                               'source_snapshot_sha256': snapshot_hash})
    views = ('head', 'left_wrist', 'right_wrist')
    labels = {'physics_steps_during_relabel': 0, 'cohort': 'comprehensive_test', 'split': 'test',
              'asset_identity_is_evaluation_only': True,
              'mask_provenance': {'renderer_instance_ground_truth': False},
              'objects': [{'id': 'toy.n.01_1', 'target': True, 'task_ids': ['toy.n.01_1'],
                           'task_object': True, 'category': 'toy figure', 'asset_id': 'toy.a',
                           'visible_pixels': {view: 4 for view in views},
                           'mask_quality': {view: {'visible_pixels_8mm': 4, 'visible_pixels_4mm': 4,
                                                  'changed_pixels_4mm_8mm': 0} for view in views}}],
              'render_convergence': {view: {'converged': True, 'renders': 2} for view in views}}
    dump(folder/'labels.json', labels)
    np.savez(folder/'input.npz', **{view+'_rgb': np.zeros((2, 2, 3), dtype=np.uint8) for view in views})
    masks = {view+'_masks': np.ones((1, 2, 2), dtype=bool) for view in views}
    np.savez(folder/'labels.npz', **masks)
    np.savez(folder/'labels_strict4mm.npz', **masks)
    for view in views:
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(folder/(view+'.png'))
    receipt = {'ok': True, 'effective_id': case_id, 'attempts': [], 'selection_sha256': selection_hash,
               'archived_pose_match': {'maximum_absolute_error': 0}}
    dump(provenance/'provenance/case_receipts/case1.json', receipt)
    materialized = {**case, 'snapshot_sha256': snapshot_hash,
                    'fidelity': {'depth': {'median_m': .001, 'within_5cm': .99},
                                 'at_start': {'joint_max_rad': .001}}}
    dump(provenance/'provenance/materialization/case1.json', {'case': materialized})
    dump(provenance/'provenance/task_receipts/task1.json', {'ok': True})
    captures = [{'path': '/old/captures/case1/'+path.name, 'sha256': package.sha(path)}
                for path in sorted(folder.iterdir())]
    assert len(captures) == 8
    dump(provenance/'sealed/capture_audit.json', {'selection_sha256': selection_hash,
                                                'input_file_receipts': captures})
    files = [{'path': path.relative_to(provenance).as_posix(), 'sha256': package.sha(path),
              'bytes': path.stat().st_size} for path in sorted(provenance.rglob('*')) if path.is_file()]
    dump(provenance/'ARTIFACT_MANIFEST.json', {'schema': package.SCHEMA, 'status': 'complete',
                                             'selection_sha256': selection_hash, 'files': files})
    result = package.audit_overlay(provenance, main, tmp_path/'audit_mount', tmp_path/'audit.json')
    assert result['ok'] and result['cases'] == 1 and result['main_files_verified'] == 8
    assert selection_path.read_bytes() == original_selection
    assert read_audit(tmp_path/'audit.json')['original_selection_bytes_preserved']
    # The companion cannot authorize modified bulk input bytes merely by retaining a valid snapshot.
    (folder/'head.png').write_bytes(b'changed raw image')
    with pytest.raises(ValueError, match='Main data capture checksum differs'):
        package.audit_overlay(provenance, main, tmp_path/'another_mount', tmp_path/'another_audit.json')


def read_audit(path):
    return json.loads(path.read_text())


def test_selected_instance_metadata_matches_evaluator_layout(tmp_path):
    selection = {'tasks': [{'task': 'task_one', 'scene_model': 'house'}],
                 'cases': [{'task': 'task_one', 'instance': 17, 'mode': 'train'},
                           {'task': 'task_one', 'instance': 23, 'mode': 'public_test'}]}
    sources = package.selected_instance_sources(selection, tmp_path)
    assert len(sources) == 2
    assert Path('2026-challenge-task-instances/scenes/house/json/house_task_task_one_instances/'
                'house_task_task_one_0_17_template-tro_state.json') in sources.values()
    assert Path('2026-challenge-task-instances/scene_test/public/house/json/house_task_task_one_instances/'
                'house_task_task_one_0_23_template-tro_state.json') in sources.values()
    for source, relative in sources.items():
        assert source == tmp_path / relative
