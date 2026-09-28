"""Conservation checks for independent approximate tolerance masks."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_visionbench_all_tasks_audit_v2 import rows
from visionbench_all_tasks_audit_v3 import validate_tolerance_masks


def test_foreground_changes_and_transfer_conserve_every_instance_without_mutation():
    broad = np.zeros((2, 3, 3), dtype=bool)
    broad[0, 0, :2] = True
    broad[1, 1, 0] = True
    strict = np.zeros_like(broad)
    strict[0, 0, 0] = strict[0, 2, 2] = strict[1, 0, 1] = True
    before = broad.copy(), strict.copy()
    report = validate_tolerance_masks(broad, strict, rows(broad, strict), 'head')
    assert report['strict_union_additions'] == report['strict_union_removals'] == report['ownership_transfer_pixels'] == 1
    assert report['instance_assignment_changes'] == 3
    assert report['instance_mask_symmetric_difference_pixels'] == 4
    assert report['union_conservation_verified'] and report['assignment_conservation_verified']
    assert all(row['assignment_conservation_verified'] for row in report['per_object'])
    assert report['per_object'][0]['added_from_background'] == 1
    assert report['per_object'][1]['removed_to_background'] == 1
    assert np.array_equal(broad, before[0]) and np.array_equal(strict, before[1])


@pytest.mark.parametrize('which', ['primary', 'strict'])
def test_overlapping_instances_are_still_rejected(which):
    broad = np.zeros((2, 2, 2), dtype=bool)
    broad[0, 0, 0] = True
    strict = broad.copy()
    (broad if which == 'primary' else strict)[1, 0, 0] = True
    with pytest.raises(AssertionError, match='overlap'):
        validate_tolerance_masks(broad, strict, rows(broad, strict), 'head')


def test_count_tampering_and_nonboolean_masks_remain_failures():
    broad = np.ones((1, 2, 2), dtype=bool)
    strict = broad.copy()
    objects = rows(broad, strict)
    objects[0]['mask_quality']['head']['visible_pixels_4mm'] = 3
    with pytest.raises(AssertionError, match='Stored tolerance pixel counts'):
        validate_tolerance_masks(broad, strict, objects, 'head')
    with pytest.raises(AssertionError, match='Boolean'):
        validate_tolerance_masks(broad.astype('uint8'), strict, rows(broad, strict), 'head')
