"""Audit saved box center/height/bottom changes; never modify predictions.

GT comparison uses previously frozen diagnostic same-class pairs (XY < .5 m,
score >= .1). GT bottom is a reference, not an independently fitted road plane.
Internal comparisons include all ten detector classes with score >= .1.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BRANCHES = ['baseline', 'tracking', 'grm', 'grm_prm']


def stats(values):
    values = np.asarray(values, dtype=float)
    return {'median': float(np.median(values)), 'p10': float(np.quantile(values, .1)),
            'p90': float(np.quantile(values, .9))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='outputs/vf-car-truck-sweeps2-01')
    args = parser.parse_args()
    run = Path(args.run_dir)
    out = run / 'analysis'
    paths = {b: run / b / ('results_nusc_center_corrected.json' if b == 'baseline'
                            else 'results_nusc_detzero_refined.json') for b in BRANCHES}
    predictions = {b: json.loads(path.read_text())['results'] for b, path in paths.items()}
    pairs = list(csv.DictReader((out / 'fixed_gt_pairs.csv').open()))
    for row in pairs:
        row['detector_class'] = predictions['baseline'][row['sample_token']][int(row['source_index'])]['detection_name']
    report = {'units': 'meters', 'reference': 'GT bottom = GT center Z - GT height / 2; not a road-plane estimate',
              'pair_policy': 'Frozen baseline score >= .1, same evaluation class, greedy XY < .5m; proximity pairing is diagnostic, not verified identity',
              'input_hashes': {b: hashlib.sha256(p.read_bytes()).hexdigest() for b, p in paths.items()},
              'gt_pairs': {}, 'all_detector_classes': {}, 'tracking_position_exceptions': []}
    for row in pairs:
        assert abs(float(row['grm_z_residual']) - float(row['tracking_z_residual'])) < 1e-6
        assert abs(float(row['grm_prm_height']) - float(row['grm_height'])) < 1e-6
    table = []
    for cls in ['car', 'motorcycle', 'pedestrian', 'car_detector_only', 'truck_detector_only']:
        rows = [p for p in pairs if (p['class'] == cls if cls in ['car', 'motorcycle', 'pedestrian']
                                    else p['detector_class'] == cls.removesuffix('_detector_only'))]
        if not rows:
            continue
        result = {'n': len(rows), 'methods': {}}
        for b in BRANCHES:
            z = np.array([float(p[b + '_z_residual']) for p in rows])
            h = np.array([float(p[b + '_height']) for p in rows])
            hgt = np.array([float(p['gt_height']) for p in rows])
            dh = h - np.array([float(p['baseline_height']) for p in rows])
            dz = z - np.array([float(p['baseline_z_residual']) for p in rows])
            bottom = z - (h - hgt) / 2
            result['methods'][b] = {'center_error_vs_gt': stats(z), 'bottom_error_vs_gt': stats(bottom),
                                    'height_change_vs_baseline': stats(dh), 'center_change_vs_baseline': stats(dz),
                                    'bottom_change_vs_baseline': stats(dz - dh / 2),
                                    'fraction_bottom_above_gt_by_10cm': float(np.mean(bottom > .1))}
            table.append({'class': cls, 'method': b, 'n': len(rows), 'center_residual_median_m': np.median(z),
                          'bottom_residual_median_m': np.median(bottom), 'height_change_median_m': np.median(dh),
                          'center_change_median_m': np.median(dz), 'bottom_change_median_m': np.median(dz - dh / 2)})
        report['gt_pairs'][cls] = result
    internal = {}
    for token, base_records in predictions['baseline'].items():
        for b in BRANCHES[1:]:
            other = predictions[b][token]
            assert len(other) == len(base_records)
            for a, c in zip(base_records, other):
                assert a['detection_name'] == c['detection_name'] and a['detection_score'] == c['detection_score']
        for i, a in enumerate(base_records):
            if a['detection_score'] < .1:
                continue
            tracked = predictions['tracking'][token][i]
            delta = np.array(tracked['translation']) - np.array(a['translation'])
            if np.linalg.norm(delta) > 1e-4:
                report['tracking_position_exceptions'].append({
                    'sample_token': token, 'source_index': i, 'class': a['detection_name'],
                    'delta_xyz_m': delta.tolist(), 'baseline': a, 'tracking': tracked})
            cls = a['detection_name']
            entries = internal.setdefault(cls, {b: [] for b in BRANCHES[1:]})
            for b in BRANCHES[1:]:
                c = predictions[b][token][i]
                dz = c['translation'][2] - a['translation'][2]
                dh = c['size'][2] - a['size'][2]
                change = max(np.max(np.abs(np.array(c['translation']) - a['translation'])),
                             np.max(np.abs(np.array(c['size']) - a['size'])))
                entries[b].append((dz, dh, dz - dh / 2, change > 1e-5))
    for cls, methods in sorted(internal.items()):
        report['all_detector_classes'][cls] = {}
        for b, entries in methods.items():
            values = np.array(entries)
            changed = values[:, 3].astype(bool)
            report['all_detector_classes'][cls][b] = {
                'n': len(entries), 'center_or_size_changed_n': int(changed.sum()),
                'center_change': stats(values[:, 0]), 'height_change': stats(values[:, 1]),
                'bottom_change': stats(values[:, 2]),
                'max_abs_bottom_change': float(np.abs(values[:, 2]).max()),
                'changed_only_bottom_change': stats(values[changed, 2]) if changed.any() else None}
    (out / 'z_bottom_audit.json').write_text(json.dumps(report, indent=2))
    with (out / 'z_bottom_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for cls, r in report['gt_pairs'].items():
        print(cls, r['n'], 'full paired bottom shift:', r['methods']['grm_prm']['bottom_change_vs_baseline']['median'])
    print('Saved', out / 'z_bottom_audit.json')


if __name__ == '__main__':
    main()
