"""Saved-query fidelity semantics and recorded-frame alignment, without simulator/GPU."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visionbench_all_tasks_fidelity as fidelity


class Recording:
    def __init__(self, pts, pixels):
        self.frame = SimpleNamespace(pts=pts, to_ndarray=lambda format: pixels)
        self.streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 30))])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def seek(self, *args, **kwargs):
        pass

    def decode(self, stream):
        yield self.frame


def metadata():
    return {'episode_index': 1, '_metadata_path': '/test/meta.parquet', '_metadata_sha256': 'a'*64,
            f'videos/{fidelity.DEPTH_KEY}/chunk_index': 0,
            f'videos/{fidelity.DEPTH_KEY}/file_index': 0,
            f'videos/{fidelity.DEPTH_KEY}/from_timestamp': 0.0}


def test_exact_frame_decodes_log_depth_and_records_proof(monkeypatch, tmp_path):
    pixels = np.array([[0, 4095]], dtype=np.uint16)
    monkeypatch.setattr(fidelity.av, 'open', lambda path: Recording(30, pixels))
    depth, proof = fidelity.recorded_depth(tmp_path, metadata(), 30)
    assert np.allclose(depth, [[.01, 10]])
    assert proof['timestamp_error_s'] == 0
    assert proof['quantized_frame_sha256'] == hashlib.sha256(pixels.tobytes()).hexdigest()


def test_wrong_recorded_frame_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(fidelity.av, 'open', lambda path: Recording(32, np.zeros((2, 2), dtype=np.uint16)))
    with pytest.raises(ValueError, match='misaligned'):
        fidelity.recorded_depth(tmp_path, metadata(), 30)


def test_final_saved_depth_cannot_be_replaced_by_bad_earlier_probe(monkeypatch, tmp_path):
    capture = tmp_path/'captures'
    for directory in ('case_receipts', 'materialization', 'case1'):
        (capture/directory).mkdir(parents=True)
    (capture/'case_receipts/case1.json').write_text(json.dumps({'ok': True}))
    raw = {'depth': {'median_m': .851, 'within_5cm': .018, 'valid_fraction': 1.0},
           'at_start': {'joint_max_rad': .02, 'eef_m': {'left': .01, 'right': .02}},
           'frame0': {'joint_max_rad': .004}, 'base_vs_reckoned': {'xy_m': .15, 'yaw_deg': 6}}
    (capture/'materialization/case1.json').write_text(json.dumps({'case': {'fidelity': raw}}))
    (capture/'case1/labels.json').write_text(json.dumps({'render_convergence': {'head': {'converged': True}}}))
    depth = np.array([[.5, .8], [1.2, 2.3]], dtype=np.float32)
    np.savez(capture/'case1/input.npz', head_depth=depth)
    original_bytes = (capture/'case1/input.npz').read_bytes()
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps({'dataset': str(tmp_path), 'cases': [
        {'id': 'case1', 'task': 'task1', 'episode_index': 1, 'frame': 30,
         'stage': 'middle', 'selection_rank': 'primary'}]}))
    output = tmp_path/'fidelity.json'
    monkeypatch.setattr(fidelity, 'episode_video_metadata', lambda dataset, episodes: {1: metadata()})
    monkeypatch.setattr(fidelity, 'recorded_depth', lambda *args: (depth, {'timestamp_error_s': 0.0}))
    monkeypatch.setattr(sys, 'argv', ['fidelity', '--selection', str(selection), '--captures', str(capture),
                                    '--output', str(output), '--fresh'])
    fidelity.main()
    report = json.loads(output.read_text())
    assert report['distributions']['saved_depth_median_m']['median'] == 0
    assert report['distributions']['saved_depth_within_5cm']['median'] == 1
    assert report['distributions']['materialization_depth_median_m']['median'] == .851
    assert report['per_case'][0]['materialization_depth_within_5cm'] == .018
    assert report['fidelity_exclusions'] == []
    assert report['selection_or_primary_evaluation_changed'] is False
    assert report['per_case'][0]['input']['sha256'] == hashlib.sha256(original_bytes).hexdigest()
    assert (capture/'case1/input.npz').read_bytes() == original_bytes
