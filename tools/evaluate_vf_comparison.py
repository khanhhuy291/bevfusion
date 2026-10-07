"""Custom frame-level VF detection evaluation; NOT official nuScenes NDS/AMOTA.

Uses nuScenes SDK center-distance AP/TP algorithms on explicitly selected tokens
and annotated classes. Velocity/attributes are intentionally not scored.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.data_classes import DetectionBox

CLASS_MAP = {'vehicle.car': 'car', 'vehicle.motorcycle': 'motorcycle',
             'human.pedestrian.adult': 'pedestrian'}
RANGES = {'car': 50, 'motorcycle': 40, 'pedestrian': 40}


def evaluate(root, version, prediction_path, classes, include_zero_points=False,
             car_includes_truck=False):
    root = Path(root)
    def table(name):
        return {x['token']: x for x in json.loads((root / version / f'{name}.json').read_text())}
    samples, egos, calibs, sensors = table('sample'), table('ego_pose'), table('calibrated_sensor'), table('sensor')
    annotations, instances, categories = table('sample_annotation'), table('instance'), table('category')
    origins = {}
    for sd in table('sample_data').values():
        if sd['is_key_frame'] and sensors[calibs[sd['calibrated_sensor_token']]['sensor_token']]['channel'] == 'LIDAR_TOP':
            origins[sd['sample_token']] = np.asarray(egos[sd['ego_pose_token']]['translation'])
    raw = json.loads(Path(prediction_path).read_text())
    if set(raw['results']) != set(samples):
        raise ValueError('Custom evaluation requires exactly all dataset sample tokens, including empty predictions')
    gt_by_sample = {token: [] for token in samples}
    counts, removed, ignored_pred, remapped_pred = Counter(), Counter(), Counter(), Counter()
    for a in annotations.values():
        category = categories[instances[a['instance_token']]['category_token']]['name']
        name = 'car' if car_includes_truck and category == 'vehicle.truck' else CLASS_MAP.get(category)
        if name not in classes: continue
        if not include_zero_points and a['num_lidar_pts'] == 0:
            removed['gt_zero_points'] += 1; continue
        delta = np.asarray(a['translation']) - origins[a['sample_token']]
        if np.linalg.norm(delta[:2]) >= RANGES[name]:
            removed['gt_outside_range'] += 1; continue
        gt_by_sample[a['sample_token']].append(DetectionBox(
            sample_token=a['sample_token'], translation=tuple(a['translation']), size=tuple(a['size']),
            rotation=tuple(a['rotation']), velocity=(0., 0.), ego_translation=tuple(delta),
            num_pts=a['num_lidar_pts'], detection_name=name, detection_score=-1., attribute_name=''))
        counts[name] += 1
    gt, pred = EvalBoxes(), EvalBoxes()
    for token in samples:
        gt.add_boxes(token, gt_by_sample[token])
        items = raw['results'][token]
        if len(items) > 500: raise ValueError('More than 500 predicted boxes in a sample')
        boxes = []
        for a in items:
            if a['sample_token'] != token: raise ValueError('Prediction sample_token mismatch')
            name = a['detection_name']
            if car_includes_truck and name == 'truck':
                name = 'car'
                remapped_pred['truck_to_car'] += 1
            if name not in classes:
                ignored_pred[name] += 1; continue
            numeric = a['translation'] + a['size'] + a['rotation'] + [a['detection_score']]
            if not np.isfinite(numeric).all() or min(a['size']) <= 0 or not 0 <= a['detection_score'] <= 1:
                raise ValueError('Invalid predicted geometry or score')
            delta = np.asarray(a['translation']) - origins[token]
            if np.linalg.norm(delta[:2]) >= RANGES[name]: continue
            boxes.append(DetectionBox(sample_token=token, translation=tuple(a['translation']), size=tuple(a['size']),
                         rotation=tuple(a['rotation']), velocity=(0., 0.), ego_translation=tuple(delta),
                         detection_name=name, detection_score=float(a['detection_score']), attribute_name=''))
        pred.add_boxes(token, boxes)
    per_class = {}
    for name in classes:
        if not counts[name]: raise ValueError(f'No evaluable GT for {name}; choose annotated classes explicitly')
        ap, tp = {}, {}
        for distance in [.5, 1., 2., 4.]:
            md = accumulate(gt, pred, name, center_distance, distance)
            ap[str(distance)] = float(calc_ap(md, min_recall=.1, min_precision=.1))
            if distance == 2.:
                tp = {m: float(calc_tp(md, min_recall=.1, metric_name=m)) for m in ['trans_err', 'scale_err', 'orient_err']}
        per_class[name] = {'gt_boxes': counts[name], 'AP': float(np.mean(list(ap.values()))), 'AP_by_distance_m': ap, **tp}
    return {'protocol': 'VF custom frame-level detection, nuScenes SDK AP/TP algorithms; not official nuScenes benchmark',
            'samples': len(samples), 'classes': list(classes), 'range_m': {n: RANGES[n] for n in classes},
            'prediction_sha256': hashlib.sha256(Path(prediction_path).read_bytes()).hexdigest(),
            'sample_tokens_sha256': hashlib.sha256('\n'.join(sorted(samples)).encode()).hexdigest(),
            'per_class': per_class, 'custom_mAP': float(np.mean([p['AP'] for p in per_class.values()])),
            'excluded': dict(removed), 'ignored_prediction_classes': dict(ignored_pred),
            'include_zero_points': include_zero_points,
            'car_includes_truck': car_includes_truck,
            'remapped_prediction_classes': dict(remapped_pred),
            'limitations': ['Partner num_lidar_pts copied, not recomputed.',
                           'Car and Rider mappings are dataset-specific.',
                           'No NDS, velocity, attribute, or global trajectory metrics.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='data/nuscenes_vf6_01_5hz')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--baseline', required=True)
    p.add_argument('--refined')
    p.add_argument('--baseline-data-root', help='Data root for baseline if different from --data-root')
    p.add_argument('--baseline-version', help='Version for baseline if different from --version')
    p.add_argument('--classes', default='car,motorcycle,pedestrian')
    p.add_argument('--include-zero-points', action='store_true', help='Sensitivity check using GT boxes with copied zero point counts')
    p.add_argument('--car-includes-truck', action='store_true',
                   help='Evaluate car and truck predictions together as VF Car, without modifying source JSON')
    p.add_argument('--output', default='outputs/vf6_01_5hz/comparison.json')
    a = p.parse_args()
    classes = a.classes.split(',')
    if not classes or set(classes) - set(RANGES) or len(set(classes)) != len(classes): raise ValueError('Unsupported or duplicate classes')
    report = {}
    b_root = a.baseline_data_root or a.data_root
    b_version = a.baseline_version or a.version
    if a.baseline:
        report['baseline'] = evaluate(b_root, b_version, a.baseline, classes, a.include_zero_points, a.car_includes_truck)
        print('baseline custom_mAP:', report['baseline']['custom_mAP'])
        for cls, metrics in report['baseline']['per_class'].items(): print(' ', cls, metrics)
    if a.refined:
        report['refined'] = evaluate(a.data_root, a.version, a.refined, classes, a.include_zero_points, a.car_includes_truck)
        print('refined custom_mAP:', report['refined']['custom_mAP'])
        for cls, metrics in report['refined']['per_class'].items(): print(' ', cls, metrics)
    if a.refined and a.baseline: report['delta_custom_mAP'] = report['refined']['custom_mAP'] - report['baseline']['custom_mAP']
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('Saved:', out)


if __name__ == '__main__': main()
