"""Independent tolerance-mask validation for the additive audit amendment."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visionbench_all_tasks_audit_v2 import validate_tolerance_masks


def rows(broad, strict):
    return [{'id': f'object{i}', 'target': i == 0, 'visible_pixels': {'head': int(broad[i].sum())},
             'mask_quality': {'head': {'visible_pixels_8mm': int(broad[i].sum()),
                                      'visible_pixels_4mm': int(strict[i].sum()),
                                      'changed_pixels_4mm_8mm': int((broad[i] ^ strict[i]).sum())}}}
            for i in range(len(broad))]


def test_contact_ownership_transfer_is_reported_without_changing_arrays():
    broad = np.zeros((2, 2, 2), dtype=bool)
    broad[0, 0, :] = True
    strict = broad.copy()
    strict[0, 0, 0] = False
    strict[1, 0, 0] = True
    original = broad.copy(), strict.copy()
    result = validate_tolerance_masks(broad, strict, rows(broad, strict), 'head')
    assert result['ownership_transfer_pixels'] == 1
    assert result['strict_union_additions'] == result['primary_overlap_pixels'] == result['strict_overlap_pixels'] == 0
    assert result['transfers'] == [{'from': 'object0', 'to': 'object1', 'from_primary': True, 'to_primary': False, 'pixels': 1}]
    assert np.array_equal(broad, original[0]) and np.array_equal(strict, original[1])


@pytest.mark.parametrize('failure', ['new_foreground', 'primary_overlap', 'strict_overlap'])
def test_spatial_support_and_exclusivity_remain_hard_failures(failure):
    broad = np.zeros((2, 2, 2), dtype=bool)
    broad[0, 0, 0] = True
    strict = broad.copy()
    if failure == 'new_foreground':
        strict[1, 1, 1] = True
    elif failure == 'primary_overlap':
        broad[1, 0, 0] = True
    else:
        strict[1, 0, 0] = True
    with pytest.raises(AssertionError):
        validate_tolerance_masks(broad, strict, rows(broad, strict), 'head')


def test_stored_count_tampering_is_rejected():
    broad = np.ones((1, 2, 2), dtype=bool)
    strict = broad.copy()
    objects = rows(broad, strict)
    objects[0]['mask_quality']['head']['visible_pixels_4mm'] = 3
    with pytest.raises(AssertionError, match='Stored tolerance pixel counts'):
        validate_tolerance_masks(broad, strict, objects, 'head')
