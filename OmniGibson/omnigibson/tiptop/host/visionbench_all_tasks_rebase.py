"""Create a portable replay selection without modifying the frozen source manifest.

Only known deployment path fields are rewritten. Input/model/selection semantics and
all source, annotation, catalog, and inventory hashes remain unchanged. The output
records its parent hash and an explicit mapping receipt; original provenance documents
can remain archived at their original paths because capture never reads them.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex


HOST_RELATIVE = Path('OmniGibson/omnigibson/tiptop/host')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remap_path(value, mappings):
    path = Path(value)
    if not path.is_absolute():
        return value
    for old, new in sorted(mappings, key=lambda row: len(row[0].parts), reverse=True):
        if path.is_relative_to(old):
            return str(new / path.relative_to(old))
    return value


def verify(path, expected, role):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Missing {role}: {path}')
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'{role} hash differs: {path}; expected {expected}, got {actual}')
    return {'role': role, 'path': str(path), 'sha256': actual}


def rebase_selection(original, *, parent_path, parent_sha256, checkout, dataset, sim_data,
                     snapshot_root, capture_root, output, python, mappings=(), catalog=None, inventory=None):
    """Pure manifest conversion plus read-only dependency verification; never writes input files."""
    required = ('source_receipts', 'cases', 'inventory', 'catalog', 'dataset')
    if any(key not in original for key in required):
        raise ValueError('Expected an all-tasks capture selection with complete provenance.')
    old_capture = [Path(path) for path in original['source_receipts']
                   if Path(path).name == 'visionbench_all_tasks_capture.py']
    if len(old_capture) != 1:
        raise ValueError('Cannot unambiguously infer the original simulator checkout.')
    old_checkout = old_capture[0].parents[4]
    mapping = [(old_checkout, checkout), (Path(original['dataset']), dataset), *mappings]
    # Duplicate old roots with different destinations are ambiguous and must be explicit errors.
    by_old = {}
    for old, new in mapping:
        old, new = old.resolve(), new.resolve()
        if old in by_old and by_old[old] != new:
            raise ValueError(f'Conflicting path mappings for {old}')
        by_old[old] = new
    mapping = list(by_old.items())
    result = copy.deepcopy(original)
    rewritten, verified = [], []

    def change(container, key, new, role):
        old = container[key]
        container[key] = str(new)
        if old != str(new):
            rewritten.append({'role': role, 'old': old, 'new': str(new)})

    change(result, 'dataset', dataset, 'demo_dataset')
    for name, override in (('catalog', catalog), ('inventory', inventory)):
        change(result[name], 'path', override or remap_path(result[name]['path'], mapping), name)
        verified.append(verify(result[name]['path'], result[name]['sha256'], name))
    result['source_receipts'] = {}
    for source, expected in original['source_receipts'].items():
        destination = remap_path(source, mapping)
        if destination == source and not Path(source).is_relative_to(checkout):
            raise ValueError(f'Unmapped active capture source: {source}')
        verified.append(verify(destination, expected, 'capture_source'))
        result['source_receipts'][destination] = expected
        if destination != source:
            rewritten.append({'role': 'capture_source', 'old': source, 'new': destination})
    annotation_cache = {}
    for case in result['cases']:
        old_snapshot = case['snapshot']
        change(case, 'snapshot', snapshot_root / f"{case['id']}.json", 'new_snapshot')
        if Path(case['snapshot']).exists():
            raise ValueError(f'New snapshot path already exists; use a fresh root: {case["snapshot"]}')
        change(case, 'annotation_path', remap_path(case['annotation_path'], mapping), 'annotation')
        annotation_key = (case['annotation_path'], case['annotation_sha256'])
        if annotation_key not in annotation_cache:
            annotation_cache[annotation_key] = verify(*annotation_key, 'annotation')
        if Path(old_snapshot) == Path(case['snapshot']):
            raise ValueError('Replay must use a new snapshot root; preserve archived snapshots.')
    verified.extend(annotation_cache.values())
    # Provenance metadata paths are never used to change case selection. If present locally,
    # verify them too; if omitted from the deployment bundle, retain their original hashes.
    optional = []
    for name in ('selector', 'finalizer', 'base_selection'):
        if name in result:
            change(result[name], 'path', remap_path(result[name]['path'], mapping), 'provenance_'+name)
            optional.append((result[name]['path'], result[name]['sha256'], name))
    for receipt in result.get('exclusion_source_receipts', []):
        change(receipt, 'path', remap_path(receipt['path'], mapping), 'exclusion_provenance')
        optional.append((receipt['path'], receipt['sha256'], 'exclusion_provenance'))
    missing_optional = []
    for path, expected, role in optional:
        if Path(path).is_file():
            verified.append(verify(path, expected, role))
        else:
            missing_optional.append({'role': role, 'path': path, 'sha256': expected})
    if capture_root.exists() and any(capture_root.iterdir()):
        raise ValueError(f'Capture root must be fresh or empty: {capture_root}')
    if not dataset.is_dir() or not sim_data.is_dir():
        raise ValueError('Both demo dataset and simulator data directories must exist.')
    if not python.is_file():
        raise ValueError(f'Simulation interpreter does not exist: {python}')
    capture_command = [str(python), str(checkout/HOST_RELATIVE/'visionbench_all_tasks_capture.py'),
                       '--selection', str(output), '--out', str(capture_root), '--data', str(sim_data)]
    audit_command = [str(python), str(checkout/HOST_RELATIVE/'visionbench_all_tasks_audit.py'),
                     '--selection', str(output), '--captures', str(capture_root),
                     '--out', str(capture_root.parent/'sealed'), '--seal']
    result['replay_deployment'] = {
        'schema': 'visionbench-capture-replay-deployment/1',
        'parent_selection': {'path': str(parent_path), 'sha256': parent_sha256},
        'semantic_selection_changed': False,
        'capture_source_bytes_changed': False,
        'mapping': [{'old': str(old), 'new': str(new)} for old, new in mapping],
        'simulator_checkout': str(checkout), 'simulator_data': str(sim_data),
        'demo_dataset': str(dataset), 'snapshot_root': str(snapshot_root), 'capture_root': str(capture_root),
        'python': str(python), 'validate_command': capture_command + ['--validate'],
        'capture_command_without_gpu_selection': capture_command, 'audit_command': audit_command,
        'environment': {'OMNIGIBSON_HEADLESS': '1', 'OMNIGIBSON_DATA_PATH': str(sim_data),
                        'B1K_DEMOS': str(dataset), 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
                        'OPENBLAS_NUM_THREADS': '8',
                        'PYTHONPATH': ':'.join(str(checkout/name) for name in ('OmniGibson', 'bddl3', 'tiptop'))},
        'missing_archival_provenance_files': missing_optional,
        'note': 'Use explicit available GPU IDs. Hashes verify inputs/code, not identical simulator rendering across hardware/runtime versions.'}
    receipt = {'schema': 'visionbench-capture-path-rebase/1', 'parent_selection': str(parent_path),
               'parent_sha256': parent_sha256, 'output_selection': str(output),
               'rewritten_path_fields': rewritten, 'verified_dependencies': verified,
               'missing_archival_provenance_files': missing_optional,
               'case_count': len(result['cases']), 'original_selection_unchanged': True}
    return result, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sim-data', type=Path, required=True)
    parser.add_argument('--snapshot-root', type=Path, required=True)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW')
    args = parser.parse_args()
    selection, output = args.selection.resolve(), args.output.resolve()
    receipt_path = output.with_name(output.stem+'.rebase.json')
    if output.exists() or receipt_path.exists() or selection == output:
        parser.error('Output selection/receipt already exists; choose fresh paths.')
    mappings = []
    for value in args.path_map:
        if '=' not in value:
            parser.error('--path-map expects OLD=NEW')
        old, new = value.split('=', 1)
        if not Path(old).is_absolute() or not Path(new).is_absolute():
            parser.error('Path mapping roots must be absolute.')
        mappings.append((Path(old), Path(new)))
    # Preserve the venv interpreter invocation: resolving its symlink changes environment semantics.
    python = args.python.absolute()
    original_bytes = selection.read_bytes()
    parent_hash = hashlib.sha256(original_bytes).hexdigest()
    result, receipt = rebase_selection(json.loads(original_bytes), parent_path=selection, parent_sha256=parent_hash,
        checkout=args.checkout.resolve(), dataset=args.dataset.resolve(), sim_data=args.sim_data.resolve(),
        snapshot_root=args.snapshot_root.resolve(), capture_root=args.capture_root.resolve(), output=output,
        python=python, mappings=mappings, catalog=args.catalog.resolve() if args.catalog else None,
        inventory=args.inventory.resolve() if args.inventory else None)
    if selection.read_bytes() != original_bytes:
        raise ValueError('Parent selection changed during rebasing.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    receipt['output_sha256'] = sha(output)
    receipt['tool'] = {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}
    with receipt_path.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2)+'\n')
    deployment = result['replay_deployment']
    print(json.dumps({'selection': str(output), 'sha256': receipt['output_sha256'],
                      'receipt': str(receipt_path), 'dependencies_verified': len(receipt['verified_dependencies']),
                      'archival_files_not_deployed': len(receipt['missing_archival_provenance_files']),
                      'validate': shlex.join(deployment['validate_command']),
                      'capture': shlex.join(deployment['capture_command_without_gpu_selection'])+' --gpus GPU_ID [GPU_ID]',
                      'audit': shlex.join(deployment['audit_command'])}, indent=2))


if __name__ == '__main__':
    main()
