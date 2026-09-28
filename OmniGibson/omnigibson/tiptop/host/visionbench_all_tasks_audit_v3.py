"""Audit unchanged, independently computed 4mm/8mm labels without nesting assumptions.

The original capture checks are inherited from the preserved v2 auditor. Foreground
support changes and instance ownership transfers are measured with conservation checks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import visionbench_all_tasks_audit_v2 as previous

sha, read, write = previous.sha, previous.read, previous.write


def validate_tolerance_masks(broad, strict, objects, view):
    assert broad.dtype == strict.dtype == bool, 'Expected native Boolean tolerance masks'
    assert broad.shape == strict.shape and broad.shape[0] == len(objects), 'Tolerance mask shapes differ'
    b_union, s_union = broad.any(axis=0), strict.any(axis=0)
    b_overlap, s_overlap = int((broad.sum(axis=0) > 1).sum()), int((strict.sum(axis=0) > 1).sum())
    assert b_overlap == 0, '8mm instance masks overlap'
    assert s_overlap == 0, '4mm instance masks overlap'
    added, removed = s_union & ~b_union, b_union & ~s_union
    transfers, per_object = [], []
    incoming, outgoing = np.zeros(len(objects), dtype=np.int64), np.zeros(len(objects), dtype=np.int64)
    for new in np.flatnonzero((strict & ~broad & b_union).reshape(len(objects), -1).any(axis=1)):
        counts = (broad & strict[new] & ~broad[new]).sum(axis=(1, 2))
        for old in np.flatnonzero(counts):
            count = int(counts[old])
            incoming[new] += count
            outgoing[old] += count
            transfers.append({'from': objects[old]['id'], 'to': objects[new]['id'],
                              'from_primary': objects[old]['target'], 'to_primary': objects[new]['target'],
                              'pixels': count})
    for index, obj in enumerate(objects):
        b_area, s_area = int(broad[index].sum()), int(strict[index].sum())
        gained, lost = int((strict[index] & added).sum()), int((broad[index] & removed).sum())
        expected = {'visible_pixels_8mm': b_area, 'visible_pixels_4mm': s_area,
                    'changed_pixels_4mm_8mm': int((broad[index] ^ strict[index]).sum())}
        assert obj['visible_pixels'][view] == b_area, 'Stored 8mm pixel count differs'
        assert all(obj['mask_quality'][view][key] == value for key, value in expected.items()), 'Stored tolerance pixel counts differ'
        assert b_area - s_area == lost - gained + int(outgoing[index]) - int(incoming[index]), 'Per-object assignment conservation failed'
        per_object.append({'object': obj['id'], 'primary': obj['target'], 'primary_pixels': b_area,
                           'strict_pixels': s_area, 'added_from_background': gained, 'removed_to_background': lost,
                           'transferred_in': int(incoming[index]), 'transferred_out': int(outgoing[index]),
                           'assignment_conservation_verified': True})
    additions, removals = int(added.sum()), int(removed.sum())
    transfer_pixels = sum(row['pixels'] for row in transfers)
    same = int((broad & strict).sum())
    b_pixels, s_pixels = int(b_union.sum()), int(s_union.sum())
    assert b_pixels - s_pixels == removals - additions, 'Foreground union conservation failed'
    assert same + transfer_pixels + removals == b_pixels, 'Primary assignment partition failed'
    assert same + transfer_pixels + additions == s_pixels, 'Strict assignment partition failed'
    xor = int((broad ^ strict).sum())
    assert xor == additions + removals + 2 * transfer_pixels, 'Boolean mask symmetric difference conservation failed'
    return {'view': view, 'strict_union_additions': additions, 'strict_union_removals': removals,
            'primary_overlap_pixels': b_overlap, 'strict_overlap_pixels': s_overlap,
            'ownership_transfer_pixels': transfer_pixels, 'same_instance_retained_pixels': same,
            'instance_assignment_changes': additions + removals + transfer_pixels,
            'instance_mask_symmetric_difference_pixels': xor,
            'primary_union_pixels': b_pixels, 'strict_union_pixels': s_pixels,
            'union_conservation_verified': True, 'assignment_conservation_verified': True,
            'transfers': transfers, 'per_object': per_object}


def audit(selection_path, captures):
    # Reuse every original capture check; substitute only the explicitly amended tolerance diagnostic.
    original_validator, original_sha = previous.validate_tolerance_masks, previous.sha
    previous.validate_tolerance_masks, previous.sha = validate_tolerance_masks, sha
    try:
        selection, result, quality, cases = previous.audit(selection_path, captures)
    finally:
        previous.validate_tolerance_masks, previous.sha = original_validator, original_sha
    result['schema'] = 'all-tasks-capture-audit/3'
    result['mask_invariants'] = {key: not result['errors'] for key in
        ('native_boolean', 'matching_shapes', 'per_tolerance_disjoint', 'stored_pixel_counts_match',
         'union_conservation', 'assignment_conservation')}
    diagnostics = result['ownership_diagnostics']
    for key in ('strict_union_removals', 'same_instance_retained_pixels', 'instance_assignment_changes',
                'instance_mask_symmetric_difference_pixels', 'primary_union_pixels', 'strict_union_pixels'):
        diagnostics[key] = sum(row[key] for row in diagnostics['per_view'])
    diagnostics.update(
        union_conservation_verified=not result['errors'], assignment_conservation_verified=not result['errors'],
        cases_with_union_additions=sorted({row['case'] for row in diagnostics['per_view'] if row['strict_union_additions']}),
        cases_with_union_removals=sorted({row['case'] for row in diagnostics['per_view'] if row['strict_union_removals']}),
        instance_assignment_changes_definition='Changed pixels counted once: foreground additions + foreground removals + ownership transfers.',
        instance_mask_symmetric_difference_definition='Boolean instance-stack XOR: additions + removals + twice ownership transfers.')
    amendment = result['diagnostic_amendment']
    amendment.update(revision='independent-tolerance-v3',
        prior_auditor_v2_sha256=sha(Path(previous.__file__)),
        reason='Observed independent tolerance labels can change both contact ownership and foreground support; neither per-instance nor foreground-union nesting is certified.',
        actual_transfer_geometry_cause='Unresolved for the real coffee/laptop cases; separate unchanged-source synthetic numerical probes demonstrate possible mechanisms.',
        interpretation='Both masks are independent approximate geometry labels; support changes and transfers are diagnostics, never corrected or clipped.',
        required_invariants=list(result['mask_invariants']))
    quality['schema'] = 'all-tasks-capture-quality/3'
    quality['ownership_diagnostics'], quality['diagnostic_amendment'] = diagnostics, amendment
    return selection, result, quality, cases


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
                    'note': 'All predeclared cases; unchanged independent tolerance labels, no replacements or exclusions.'}
        write(args.out/'query_manifest.json', manifest, True)
        write(args.out/'seal_receipt.json', {'schema': 'all-tasks-capture-seal/3',
              'files': {str((args.out/name).resolve()): sha(args.out/name) for name in
                        ('capture_audit.json', 'capture_quality_summary.json', 'query_manifest.json')},
              'cases': len(cases), 'tasks': len({c['task'] for c in cases}),
              'auditor': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
              'auditor_dependencies': {str(Path(previous.__file__).resolve()): sha(previous.__file__)}}, True)
    print(__import__('json').dumps({'complete': result['complete'], 'ok': result['ok'], 'cases': len(cases),
          'missing': len(result['cases_missing']), 'errors': result['errors'],
          'nonconverged_views': quality['saved_nonconverged_view_count'],
          'strict_union_additions': result['ownership_diagnostics']['strict_union_additions'],
          'ownership_transfer_pixels': result['ownership_diagnostics']['ownership_transfer_pixels']}))
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
