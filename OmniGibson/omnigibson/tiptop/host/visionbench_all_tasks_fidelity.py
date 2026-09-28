"""Compare final saved head depth with its recorded frame; never filters selected cases.

The earlier materialization depth probe is preserved separately. Its camera/render
settling can differ from the final saved query, so it is not a substitute for this check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

DEPTH_KEY = 'observation.depth_linear.zed_link_camera_0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {'count': 0}
    return {'count': int(values.size), 'minimum': float(values.min()), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9)), 'p95': float(np.quantile(values, .95)),
            'maximum': float(values.max())}


def episode_video_metadata(dataset, episodes):
    columns = ['episode_index'] + [f'videos/{DEPTH_KEY}/{field}' for field in
                                  ('chunk_index', 'file_index', 'from_timestamp')]
    result = {}
    for path in sorted((dataset/'meta/episodes').glob('*/*.parquet')):
        table = pq.read_table(path, columns=columns, filters=[('episode_index', 'in', sorted(episodes))])
        for row in table.to_pylist():
            row['_metadata_path'] = str(path)
            row['_metadata_sha256'] = sha(path)
            result[int(row['episode_index'])] = row
    missing = episodes - set(result)
    if missing:
        raise ValueError(f'No recorded video metadata for episodes: {sorted(missing)}')
    return result


def recorded_depth(dataset, metadata, frame):
    chunk, file_index, start = (metadata[f'videos/{DEPTH_KEY}/{field}'] for field in
                               ('chunk_index', 'file_index', 'from_timestamp'))
    video = dataset/f'videos/{DEPTH_KEY}/chunk-{chunk:03d}/file-{file_index:03d}.mp4'
    requested = float(start) + frame/30
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        container.seek(int(requested/stream.time_base), stream=stream, backward=True)
        for decoded in container.decode(stream):
            actual = float(decoded.pts * stream.time_base)
            if actual >= requested - .5/30:
                quantized = decoded.to_ndarray(format='gray12le')
                proof = {'video': str(video), 'metadata': metadata['_metadata_path'],
                         'metadata_sha256': metadata['_metadata_sha256'], 'episode_offset_s': float(start),
                         'episode_frame': frame, 'requested_timestamp_s': requested,
                         'decoded_timestamp_s': actual, 'timestamp_error_s': actual-requested,
                         'decoded_pts': int(decoded.pts), 'time_base': str(stream.time_base),
                         'quantized_frame_sha256': hashlib.sha256(quantized.tobytes()).hexdigest(),
                         'pixel_format': 'gray12le', 'source_quantized_min': int(quantized.min()),
                         'source_quantized_max': int(quantized.max())}
                break
        else:
            raise ValueError(f'No depth frame at timestamp {requested}: {video}')
    if abs(proof['timestamp_error_s']) > .5/30 + 1e-6:
        raise ValueError(f'Recorded depth is misaligned: {proof}')
    # Matches OmniGibson eval/utils/obs_utils.py dequantize_depth defaults:
    # 12-bit log encoding, min=.01m, max=10m, shift=3.5.
    depth = np.clip(np.exp(quantized.astype(float)/4095 * (math.log(13.5)-math.log(3.51))
                           + math.log(3.51)) - 3.5, .01, 10)
    return depth, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, help='Default: dataset path in selection.')
    parser.add_argument('--fresh', action='store_true', help='Recompute all saved-depth comparisons and input hashes.')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    dataset = (args.dataset or Path(selection['dataset'])).resolve()
    selection_hash, tool_hash = sha(args.selection), sha(__file__)
    cache = {}
    if args.output.exists() and not args.fresh:
        previous = json.loads(args.output.read_text())
        if previous.get('selection_sha256') == selection_hash and previous.get('tool', {}).get('sha256') == tool_hash:
            cache = {row['case']: row for row in previous['per_case']}
    complete_cases = []
    missing = []
    for case in selection['cases']:
        receipt = args.captures/'case_receipts'/f"{case['id']}.json"
        if receipt.is_file() and json.loads(receipt.read_text()).get('ok'):
            complete_cases.append(case)
        else:
            missing.append(case['id'])
    metadata = episode_video_metadata(dataset, {case['episode_index'] for case in complete_cases
                                                if case['id'] not in cache})
    rows, reused = [], 0
    for case in complete_cases:
        materialization = args.captures/'materialization'/f"{case['id']}.json"
        labels_path = args.captures/case['id']/'labels.json'
        input_path = args.captures/case['id']/'input.npz'
        mat_hash, label_hash = sha(materialization), sha(labels_path)
        stat = input_path.stat()
        cached = cache.get(case['id'])
        if cached and cached['materialization']['sha256'] == mat_hash and cached['labels']['sha256'] == label_hash \
                and cached['input']['size'] == stat.st_size and cached['input']['mtime_ns'] == stat.st_mtime_ns:
            rows.append(cached)
            reused += 1
            continue
        if case['episode_index'] not in metadata:
            metadata.update(episode_video_metadata(dataset, {case['episode_index']}))
        fidelity = json.loads(materialization.read_text())['case']['fidelity']
        labels = json.loads(labels_path.read_text())
        recorded, frame_proof = recorded_depth(dataset, metadata[case['episode_index']], case['frame'])
        with np.load(input_path, allow_pickle=False) as images:
            saved = images['head_depth']
            if recorded.shape != saved.shape:
                raise ValueError(f'Recorded and saved head depth shapes differ: {case["id"]}')
            valid = np.isfinite(saved) & (saved > 0)
            difference = np.abs(np.clip(saved, .01, 10) - recorded)[valid]
            median = float(np.median(difference)) if difference.size else None
            within = float((difference < .05).mean()) if difference.size else None
        row = {'case': case['id'], 'task': case['task'], 'episode': case['episode_index'],
               'frame': case['frame'], 'stage': case['stage'], 'selection_rank': case['selection_rank'],
               'saved_depth_median_m': median, 'saved_depth_within_5cm': within,
               'saved_depth_valid_fraction': float(valid.mean()),
               'materialization_depth_median_m': fidelity['depth']['median_m'],
               'materialization_depth_within_5cm': fidelity['depth']['within_5cm'],
               'materialization_depth_valid_fraction': fidelity['depth']['valid_fraction'],
               'joint_max_rad': fidelity['at_start']['joint_max_rad'],
               'eef_left_m': fidelity['at_start']['eef_m']['left'],
               'eef_right_m': fidelity['at_start']['eef_m']['right'],
               'eef_max_m': max(fidelity['at_start']['eef_m'].values()),
               'initial_joint_max_rad': fidelity['frame0']['joint_max_rad'],
               'base_vs_dead_reckoned': fidelity['base_vs_reckoned'],
               'nonconverged_saved_views': [view for view, value in labels['render_convergence'].items()
                                            if value.get('converged') is False],
               'materialization': {'path': str(materialization.resolve()), 'sha256': mat_hash},
               'labels': {'path': str(labels_path.resolve()), 'sha256': label_hash},
               'input': {'path': str(input_path.resolve()), 'sha256': sha(input_path),
                         'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
               'recorded_frame_proof': frame_proof}
        rows.append(row)
    metrics = ('saved_depth_median_m', 'saved_depth_within_5cm', 'saved_depth_valid_fraction',
               'materialization_depth_median_m', 'materialization_depth_within_5cm',
               'joint_max_rad', 'eef_left_m', 'eef_right_m', 'eef_max_m', 'initial_joint_max_rad')
    # Rankings are descriptive, not thresholds defining a new evaluation subset.
    worst_median = sorted(rows, key=lambda row: row['saved_depth_median_m'] or -1, reverse=True)[:20]
    worst_fraction = sorted(rows, key=lambda row: row['saved_depth_within_5cm'] if row['saved_depth_within_5cm'] is not None else -1)[:20]
    result = {'schema': 'all-tasks-replay-fidelity/2', 'selection_sha256': selection_hash,
              'cases_expected': len(selection['cases']), 'cases_completed': len(rows), 'missing_cases': missing,
              'complete': len(rows) == len(selection['cases']), 'predictions_read': False,
              'selection_or_primary_evaluation_changed': False, 'fidelity_exclusions': [],
              'dataset': str(dataset), 'cache_rows_reused': reused,
              'distributions': {metric: summary(row[metric] for row in rows) for metric in metrics},
              'comparison_roles': {'saved_depth': 'Final model-input head depth versus recorded dataset frame.',
                                   'materialization_depth': 'Earlier pre-capture probe; transient camera/render settling may make it unrepresentative of final saved query.'},
              'limitations': ['Depths are clipped to10m and compared over finite positive self-filtered final head-depth pixels.',
                              'The saved query is used for fidelity conclusions. Earlier probe values remain unmodified provenance.',
                              'End-effector errors use the robot base frame and are not world-pose errors.',
                              'Base-versus-dead-reckoned differences are against integrated odometry, not measured world ground truth.',
                              'Differences can reflect controller/physics replay, object state, camera pose or rendering. No arbitrary fidelity exclusions are applied.',
                              'Captured RGB/depth and geometry labels share the frozen simulator state even when replay differs from the recorded video.'],
              'per_case': rows, 'highest_saved_depth_median_cases': worst_median,
              'lowest_saved_depth_within5cm_cases': worst_fraction,
              'maximum_recorded_frame_timestamp_error_s': max((abs(r['recorded_frame_proof']['timestamp_error_s']) for r in rows), default=None),
              'tool': {'path': str(Path(__file__).resolve()), 'sha256': tool_hash}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ('cases_completed', 'complete', 'cache_rows_reused',
                      'distributions', 'maximum_recorded_frame_timestamp_error_s')}, indent=2))


if __name__ == '__main__':
    main()
