"""Stage, finalize, verify, and audit the capture provenance companion artifact.

Only explicitly named metadata/provenance files are included. No meshes, video shards,
model weights, reference images, or unrelated archived experiments are traversed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

SCHEMA = 'visionbench-capture-provenance/2'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe artifact path: {value}')
    return Path(*path.parts)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def add_file(stage, files, source, relative, role, expected=None):
    source = Path(source).resolve()
    relative = safe_relative(relative)
    destination = stage/relative
    if source.suffix.lower() in {'.usd', '.usda', '.usdc', '.obj', '.mp4', '.pt', '.pth', '.safetensors'}:
        raise ValueError(f'Binary asset/model/video excluded from provenance artifact: {source}')
    actual = sha(source)
    if expected and actual != expected:
        raise ValueError(f'Frozen dependency hash differs: {source}')
    if any(row['path'] == relative.as_posix() for row in files):
        raise ValueError(f'Duplicate artifact destination: {relative}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(source, destination)
    if sha(source) != actual or sha(destination) != actual:
        raise ValueError(f'File changed during staging: {source}')
    files.append({'path': relative.as_posix(), 'sha256': actual, 'bytes': destination.stat().st_size,
                  'original_path': str(source), 'role': role})


def verify_files(root, manifest):
    seen = set()
    for row in manifest['files']:
        path = root/safe_relative(row['path'])
        if row['path'] in seen or not path.is_file() or path.is_symlink():
            raise ValueError(f'Invalid or duplicate artifact file: {row["path"]}')
        seen.add(row['path'])
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError(f'Artifact checksum differs: {row["path"]}')
    return {'ok': True, 'files': len(seen), 'bytes': sum(row['bytes'] for row in manifest['files'])}


def selected_instance_sources(selection, sim_data):
    """Resolve evaluator train/public instance metadata without importing Isaac Sim."""
    mode_dirs = {'train': 'scenes', 'public_test': 'scene_test/public', 'hidden_test': 'scene_test/private'}
    tasks = {task['task']: task for task in selection['tasks']}
    sources = {}
    for case in selection['cases']:
        scene = tasks[case['task']]['scene_model']
        instance_name = f"{scene}_task_{case['task']}_0_{case['instance']}_template-tro_state.json"
        relative = (Path('2026-challenge-task-instances') / mode_dirs[case['mode']] / scene / 'json'
                    / f"{scene}_task_{case['task']}_instances" / instance_name)
        sources[sim_data / relative] = relative
    return sources


def stage_metadata(selection_path, stage, sim_data):
    if stage.exists():
        raise FileExistsError(f'Use a fresh staging directory: {stage}')
    selection_path = selection_path.resolve()
    selection = read(selection_path)
    run = selection_path.parent
    inventory = read(selection['inventory']['path'])
    stage.mkdir(parents=True)
    files = []
    add_file(stage, files, selection_path, 'selection/selection_fullcoverage.json', 'frozen_final_selection')
    add_file(stage, files, selection['base_selection']['path'], 'selection/selection_initial300.json',
             'archived_base_selection', selection['base_selection']['sha256'])
    for name in ('catalog', 'inventory'):
        source = selection[name]
        add_file(stage, files, source['path'], 'metadata/'+name+Path(source['path']).suffix,
                 name, source['sha256'])
    for record in inventory['metadata_sources']:
        add_file(stage, files, record['path'], 'metadata/dataset_metadata/'+Path(record['path']).name,
                 'dataset_metadata', record['sha256'])
    for task in inventory['tasks']:
        add_file(stage, files, task['template']['path'], 'metadata/task_templates/'+task['task_name']+'.json',
                 'task_template_metadata', task['template']['sha256'])
        add_file(stage, files, task['bddl_definition']['path'], 'metadata/bddl/'+task['task_name']+'.bddl',
                 'task_bddl', task['bddl_definition']['sha256'])
    for source, relative in sorted(selected_instance_sources(selection, sim_data).items()):
        add_file(stage, files, source, 'metadata/sim_data/' + relative.as_posix(),
                 'selected_task_instance_configuration')
    annotations = {}
    for case in selection['cases']:
        annotation = Path(case['annotation_path'])
        annotations[annotation] = case['annotation_sha256']
    dataset = Path(selection['dataset'])
    for annotation, expected in sorted(annotations.items()):
        relative = annotation.relative_to(dataset)
        add_file(stage, files, annotation, relative.as_posix(), 'query_annotation', expected)
    for source, expected in selection['source_receipts'].items():
        add_file(stage, files, source, 'source_capture/'+Path(source).name, 'frozen_capture_source', expected)
    original_sources = {row['original_path'] for row in files}
    archives = [*selection['exclusion_source_receipts'], selection['selector'], selection['finalizer']]
    for record in archives:
        source = Path(record['path']).resolve()
        if str(source) not in original_sources:
            add_file(stage, files, source, 'archival_metadata/'+record['sha256'][:16]+'_'+source.name,
                     'frozen_selection_ancestry', record['sha256'])
            original_sources.add(str(source))
    capture_source = next(Path(path) for path in selection['source_receipts']
                          if Path(path).name == 'visionbench_all_tasks_capture.py')
    robot = capture_source.parents[4]/'OmniGibson/omnigibson/eval/r1pro.yaml'
    add_file(stage, files, robot, 'metadata/eval_r1pro.yaml', 'evaluator_robot_config')
    for name in ('scope_inventory.json', 'launch_receipt.json', 'first_capture_visual_review.json',
                 'portability_validation.json', 'capture_diagnostic_amendment_v2.json', 'geometry_source_receipts.json'):
        source = run/name
        if source.is_file():
            add_file(stage, files, source, 'run_receipts/'+name, 'run_provenance')
    for name in ('failed_50case_capture_audit.json', 'failed_50case_capture_quality_summary.json',
                 'saved_mask_comparison_v1.json', 'diagnose_mask_ownership.py'):
        source = run/'fidelity_diagnostics/strict4mm_ownership'/name
        if source.is_file():
            add_file(stage, files, source, 'diagnostic_amendment/'+name, 'preserved_pre_amendment_evidence')
    geometry_receipts = read(run/'geometry_source_receipts.json')
    for record in geometry_receipts['sources']:
        add_file(stage, files, record['path'], 'source_geometry/'+record['artifact_name'],
                 'geometry_dependency_source', record['sha256'])
    for name in ('visionbench_all_tasks_audit_v2.py', 'visionbench_all_tasks_package_v2.py',
                 'visionbench_all_tasks_fidelity.py'):
        add_file(stage, files, Path(__file__).parent/name, 'source_diagnostics/'+name,
                 'post_capture_diagnostic_source')
    draft = {'schema': SCHEMA, 'status': 'metadata_staged_not_publishable',
             'selection_sha256': sha(selection_path), 'source_selection': str(selection_path),
             'source_run': str(run), 'files': files,
             'exclusions': ['licensed meshes/textures', 'dataset video/action shards', 'model checkpoints',
                            'reference images', 'unrelated archives', 'active logs/progress files'],
             'tool': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}}
    write_new(stage/'STAGING_MANIFEST.json', draft)
    return {'stage': str(stage), **verify_files(stage, draft)}


def finalize(stage, archive):
    draft = read(stage/'STAGING_MANIFEST.json')
    verify_files(stage, draft)
    if (stage/'ARTIFACT_MANIFEST.json').exists() or archive.exists():
        raise FileExistsError('Final artifact already exists; never overwrite.')
    run = Path(draft['source_run'])
    selection = read(stage/'selection/selection_fullcoverage.json')
    for record in draft['files']:
        if record['role'] == 'selected_task_instance_configuration' and sha(record['original_path']) != record['sha256']:
            raise ValueError('Selected task instance configuration changed after metadata staging.')
    if sha(run/'selection_fullcoverage.json') != draft['selection_sha256']:
        raise ValueError('Original frozen selection changed.')
    execution = read(run/'captures/execution.json')
    if execution.get('status') != 'complete' or not all(execution['task_status'].values()):
        raise ValueError('Cannot package active/incomplete capture execution.')
    sealed_audit = read(run/'sealed/capture_audit.json')
    if not sealed_audit.get('complete') or not sealed_audit.get('ok') or sealed_audit['errors']:
        raise ValueError('Completed independent capture seal required.')
    if sealed_audit.get('schema') != 'all-tasks-capture-audit/2':
        raise ValueError('Ownership-aware audit v2 required.')
    diagnostics = sealed_audit['ownership_diagnostics']
    if any(diagnostics[key] for key in ('strict_union_additions', 'primary_overlap_pixels', 'strict_overlap_pixels')):
        raise ValueError('Tolerance union/disjoint invariants failed.')
    if not all(sealed_audit['mask_invariants'].values()):
        raise ValueError('Mask integrity invariant failed.')
    seal = read(run/'sealed/seal_receipt.json')
    for path, expected in seal['files'].items():
        if sha(path) != expected:
            raise ValueError(f'Sealed capture metadata differs: {path}')
    fidelity = read(run/'replay_fidelity_summary.json')
    if not fidelity.get('complete') or fidelity['cases_completed'] != len(selection['cases']):
        raise ValueError('Final saved-query fidelity summary for every case required.')
    if fidelity.get('cache_rows_reused') != 0:
        raise ValueError('Run the final fidelity command with --fresh before packaging.')
    files = list(draft['files'])
    for name in ('capture_audit.json', 'capture_quality_summary.json', 'query_manifest.json', 'seal_receipt.json'):
        add_file(stage, files, run/'sealed'/name, 'sealed/'+name, 'sealed_capture_metadata')
    add_file(stage, files, run/'replay_fidelity_summary.json', 'replay_fidelity_summary.json', 'saved_query_fidelity')
    for case in selection['cases']:
        identifier = case['id']
        add_file(stage, files, case['snapshot'], 'snapshots/'+identifier+'.json', 'original_snapshot')
        for kind in ('case_receipts', 'materialization'):
            add_file(stage, files, run/'captures'/kind/(identifier+'.json'),
                     'provenance/'+kind+'/'+identifier+'.json', kind)
    for task in selection['tasks']:
        add_file(stage, files, run/'captures/task_receipts'/(task['task']+'.json'),
                 'provenance/task_receipts/'+task['task']+'.json', 'task_receipt')
    for name in ('execution.json', 'selection_receipt.json'):
        add_file(stage, files, run/'captures'/name, 'provenance/'+name, 'capture_execution_provenance')
    # The robot controller configuration must match every captured materialization receipt.
    expected_robot = next(row['sha256'] for row in files if row['path'] == 'metadata/eval_r1pro.yaml')
    for case in selection['cases']:
        materialization = read(stage/'provenance/materialization'/(case['id']+'.json'))
        if materialization['replay_configuration']['evaluator_yaml_sha256'] != expected_robot:
            raise ValueError('Captured evaluator robot config differs from packaged controller config.')
    manifest = {**draft, 'status': 'complete', 'files': sorted(files, key=lambda row: row['path']),
                'cases': len(selection['cases']), 'tasks': len(selection['tasks']),
                'main_data_capture_location': '<VISION_RUN>/captures/<case_id>',
                'audit_recipe': 'visionbench_all_tasks_package_v2.py audit-overlay --provenance <THIS_ROOT> --main-captures <VISION_RUN>/captures --mount <FRESH_AUDIT_MOUNT> --output <FRESH_AUDIT_JSON>',
                'replay_recipe': 'Use visionbench_all_tasks_rebase.py with selection/selection_fullcoverage.json, metadata/catalog.parquet and metadata/inventory.json; point --dataset/--sim-data to installed complete external datasets.',
                'artifact_bytes': sum(row['bytes'] for row in files)}
    manifest.pop('source_run', None)
    write_new(stage/'ARTIFACT_MANIFEST.json', manifest)
    verify_files(stage, manifest)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'x') as stream:
        for name in ['ARTIFACT_MANIFEST.json', *[row['path'] for row in manifest['files']]]:
            path = stage/safe_relative(name)
            info = stream.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            info.mode = 0o644
            with path.open('rb') as source:
                stream.addfile(info, source)
    return {'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
            'files': len(manifest['files']), 'manifest_sha256': sha(stage/'ARTIFACT_MANIFEST.json')}


def audit_overlay(provenance, main_captures, mount, output):
    manifest = read(provenance/'ARTIFACT_MANIFEST.json')
    if manifest.get('status') != 'complete':
        raise ValueError('Only a complete companion artifact can be audited.')
    verify_files(provenance, manifest)
    selection_path = provenance/'selection/selection_fullcoverage.json'
    selection = read(selection_path)
    sealed = read(provenance/'sealed/capture_audit.json')
    if sha(selection_path) != manifest['selection_sha256'] or sealed['selection_sha256'] != manifest['selection_sha256']:
        raise ValueError('Selection provenance mismatch.')
    # Compare raw main data bytes with original sealed receipts before any mounted audit.
    checked = 0
    case_ids = {case['id'] for case in selection['cases']}
    for record in sealed['input_file_receipts']:
        original = Path(record['path'])
        if original.parent.name not in case_ids:
            continue
        current = main_captures/original.parent.name/original.name
        if sha(current) != record['sha256']:
            raise ValueError(f'Main data capture checksum differs: {current}')
        checked += 1
    if checked != 8*len(case_ids):
        raise ValueError(f'Expected8 native capture/image files per case; found {checked}')
    if mount.exists() or output.exists():
        raise FileExistsError('Audit mount and output must be fresh.')
    mount.mkdir(parents=True)
    for case in selection['cases']:
        source = main_captures/case['id']
        if not source.is_dir():
            raise ValueError(f'Missing main capture directory: {source}')
        (mount/case['id']).symlink_to(source.resolve(), target_is_directory=True)
    for kind in ('case_receipts', 'materialization', 'task_receipts'):
        (mount/kind).symlink_to((provenance/'provenance'/kind).resolve(), target_is_directory=True)
    import visionbench_all_tasks_audit_v2 as auditor
    original_sha = auditor.sha
    snapshot_mapping = {Path(case['snapshot']): provenance/'snapshots'/(case['id']+'.json')
                        for case in selection['cases']}
    def snapshot_aware_sha(path):
        return original_sha(snapshot_mapping.get(Path(path), path))
    auditor.sha = snapshot_aware_sha
    try:
        _, result, quality, _ = auditor.audit(selection_path, mount)
    finally:
        auditor.sha = original_sha
    if not result['complete'] or not result['ok'] or result['errors']:
        raise ValueError(f'Independent mounted capture audit failed: {result["errors"]}')
    proof = {'schema': 'visionbench-capture-overlay-audit/1', 'ok': True,
             'companion_manifest_sha256': sha(provenance/'ARTIFACT_MANIFEST.json'),
             'selection_sha256': sha(selection_path), 'raw_main_capture_files_verified': checked,
             'snapshot_read_mapping': {str(old): str(new) for old, new in snapshot_mapping.items()},
             'original_selection_bytes_preserved': True, 'case_count': len(case_ids),
             'independent_audit': result, 'capture_quality': quality,
             'auditor_source_sha256': sha(Path(auditor.__file__)),
             'package_tool_sha256': sha(__file__)}
    write_new(output, proof)
    return {'ok': True, 'cases': len(case_ids), 'main_files_verified': checked,
            'output': str(output), 'output_sha256': sha(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    stage = sub.add_parser('stage')
    stage.add_argument('--selection', type=Path, required=True)
    stage.add_argument('--stage', type=Path, required=True)
    stage.add_argument('--sim-data', type=Path, required=True)
    final = sub.add_parser('finalize')
    final.add_argument('--stage', type=Path, required=True)
    final.add_argument('--archive', type=Path, required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--provenance', type=Path, required=True)
    audit = sub.add_parser('audit-overlay')
    audit.add_argument('--provenance', type=Path, required=True)
    audit.add_argument('--main-captures', type=Path, required=True)
    audit.add_argument('--mount', type=Path, required=True)
    audit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'stage':
        result = stage_metadata(args.selection.resolve(), args.stage.resolve(), args.sim_data.resolve())
    elif args.command == 'finalize':
        result = finalize(args.stage.resolve(), args.archive.resolve())
    elif args.command == 'verify':
        manifest = read(args.provenance/'ARTIFACT_MANIFEST.json')
        if manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete':
            raise ValueError('Expected finalized companion artifact manifest.')
        result = verify_files(args.provenance.resolve(), manifest)
    else:
        result = audit_overlay(args.provenance.resolve(), args.main_captures.resolve(),
                               args.mount.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
