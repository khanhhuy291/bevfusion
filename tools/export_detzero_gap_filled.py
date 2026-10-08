#!/usr/bin/env python3
"""Export gap boxes from an existing trusted local DetZero run, without rerunning models."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'DetZero-main-2'))
from integration import bridge
from integration.gap_fill import export_gap_filled


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, help='Existing branch containing tracks.pkl, refined_boxes.pkl and run_report.json')
    p.add_argument('--baseline', required=True, help='EXACT baseline JSON used as input to the existing run')
    p.add_argument('--data-root', required=True, help='EXACT dataset_local/calibration used by the existing run')
    p.add_argument('--version', default='v1.0-mini')
    p.add_argument('--output-dir', required=True, help='New or empty output directory')
    p.add_argument('--method', choices=['tracker', 'interpolate'], default='tracker')
    p.add_argument('--max-gap-seconds', type=float, default=0.6,
                   help='Maximum TOTAL elapsed time between two observed anchors, not time since the last hit')
    p.add_argument('--score-decay', type=float, default=0.8)
    p.add_argument('--min-score', type=float, default=0.1)
    p.add_argument('--duplicate-iou', type=float, default=0.1)
    p.add_argument('--duplicate-center-distance', type=float, default=0.5)
    args = p.parse_args()
    run, out = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    if out == run or (out.exists() and any(out.iterdir())):
        raise FileExistsError('Choose a new/empty output directory; source results are preserved')
    paths = {n: run / n for n in ['run_report.json', 'tracks.pkl', 'refined_boxes.pkl',
                                  'results_nusc_detzero_refined.json', 'results_nusc_detzero_tracking.json']}
    for path in [*paths.values(), Path(args.baseline)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_report = json.loads(paths['run_report.json'].read_text())
    original_args = source_report['arguments']
    if source_report.get('coordinate_mode', original_args.get('coordinate_mode')) != 'global':
        raise ValueError('Cached gap export requires a global-coordinate run')
    source_hash = hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest()
    if source_report.get('input_sha256') != source_hash:
        raise ValueError('Baseline differs from the exact JSON used by this run; check center correction and run directory')
    if original_args['version'] != args.version:
        raise ValueError('Dataset version differs from source run')
    classes = list(dict.fromkeys(c.strip() for c in original_args['classes'].split(',') if c.strip()))
    prepared = bridge.prepare(args.baseline, args.data_root, args.version, classes, original_args['min_score'])
    # These are trusted user-created local pipeline artifacts, not downloaded pickle inputs.
    tracks = pickle.loads(paths['tracks.pkl'].read_bytes())
    refined = pickle.loads(paths['refined_boxes.pkl'].read_bytes())
    if set(refined) != set(tracks):
        raise ValueError('Cached track and refined IDs differ')
    detection = json.loads(paths['results_nusc_detzero_refined.json'].read_text())
    tracking = json.loads(paths['results_nusc_detzero_tracking.json'].read_text())
    gap_detection, gap_tracking, report = export_gap_filled(
        prepared, tracks, refined, detection, tracking, args.method, args.max_gap_seconds,
        args.score_decay, args.min_score, args.duplicate_iou, args.duplicate_center_distance)
    report.update(arguments=vars(args), baseline_sha256=source_hash,
                  source_artifacts_sha256={n: hashlib.sha256(f.read_bytes()).hexdigest() for n, f in paths.items()})
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results_nusc_detzero_gap_filled.json').write_text(json.dumps(gap_detection, allow_nan=False))
    (out / 'results_nusc_detzero_tracking_gap_filled.json').write_text(json.dumps(gap_tracking, allow_nan=False))
    (out / 'gap_fill_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('Gap fill:', report['counts'])
    print('Added frames:', report['added_frames'])
    print('Outputs:', out)


if __name__ == '__main__':
    main()
