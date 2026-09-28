"""V3 capture companion: preserve both independent tolerance labels and all prior evidence."""
from pathlib import Path
import argparse
import json
import tarfile

import visionbench_all_tasks_package_v2 as previous

SCHEMA = 'visionbench-capture-provenance/3'
sha, read, safe_relative = previous.sha, previous.read, previous.safe_relative
write_new, add_file, verify_files = previous.write_new, previous.add_file, previous.verify_files
selected_instance_sources = previous.selected_instance_sources


def stage_metadata(selection_path, stage, sim_data):
    previous.stage_metadata(selection_path, stage, sim_data)
    draft_path = stage/'STAGING_MANIFEST.json'
    draft = read(draft_path)
    files, run = draft['files'], selection_path.parent
    for name in ('visionbench_all_tasks_audit_v3.py', 'visionbench_all_tasks_package_v3.py'):
        add_file(stage, files, Path(__file__).parent/name, 'source_diagnostics/'+name,
                 'independent_tolerance_diagnostic_source')
    add_file(stage, files, run/'capture_diagnostic_amendment_v3.json',
             'run_receipts/capture_diagnostic_amendment_v3.json', 'independent_tolerance_amendment')
    for name in ('capture_audit.json', 'capture_quality_summary.json', 'pixel_counts.json',
                 'all_completed_mask_comparison.json'):
        add_file(stage, files, run/'fidelity_diagnostics/strict4mm_union_failure'/name,
                 'diagnostic_amendment_v3/'+name, 'preserved_union_failure_evidence')
    draft.update(schema=SCHEMA, tool={'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
                 prior_staging_tool=draft['tool'], files=files)
    draft_path.rename(stage/'STAGING_BASE_V2_MANIFEST.json')
    write_new(draft_path, draft)
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
    if sealed_audit.get('schema') != 'all-tasks-capture-audit/3':
        raise ValueError('Independent-tolerance audit v3 required.')
    diagnostics = sealed_audit['ownership_diagnostics']
    if any(diagnostics[key] for key in ('primary_overlap_pixels', 'strict_overlap_pixels')):
        raise ValueError('Per-tolerance disjoint invariants failed.')
    if not diagnostics['union_conservation_verified'] or not diagnostics['assignment_conservation_verified']:
        raise ValueError('Independent-mask conservation checks failed.')
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
                'audit_recipe': 'visionbench_all_tasks_package_v3.py audit-overlay --provenance <THIS_ROOT> --main-captures <VISION_RUN>/captures --mount <FRESH_AUDIT_MOUNT> --output <FRESH_AUDIT_JSON>',
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
    import visionbench_all_tasks_audit_v3 as auditor
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
    proof = {'schema': 'visionbench-capture-overlay-audit/3', 'ok': True,
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
