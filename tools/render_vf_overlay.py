"""Overlay real GT, BEVFusion and full DetZero boxes in camera, BEV and side views.

Uses saved geometric-center predictions, original camera calibration/ego poses,
and raw keyframe LiDAR. No inference, fabricated imagery or box corrections.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/vf_matplotlib')
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from render_vf_detection_comparison import COLORS, MAPPING, RANGE, corners, draw_bev, project, read, text, to_ego
from vf_iou3d import iou3d_matrix

METHODS = ['GT', 'BEVFusion', 'BEVFusion + DetZero']
EDGES = [(i, i + 4) for i in range(4)] + [(i, (i+1) % 4) for i in range(4)] + [(i+4, (i+1) % 4+4) for i in range(4)]


def line(canvas, a, b, color, dashed=False, thickness=2):
    height, width = canvas.shape[:2]
    ok, start, end = cv2.clipLine((0, 0, width, height), tuple(a), tuple(b))
    if not ok:
        return
    if not dashed:
        cv2.line(canvas, start, end, color, thickness, cv2.LINE_AA)
        return
    start, end = np.array(start, float), np.array(end, float)
    length = np.linalg.norm(end-start)
    if length < 1:
        return
    for offset in np.arange(0, length, 14):
        p = np.rint(start + (end-start) * offset / length).astype(int)
        q = np.rint(start + (end-start) * min(offset+8, length) / length).astype(int)
        cv2.line(canvas, tuple(p), tuple(q), color, thickness, cv2.LINE_AA)


def camera_overlay(image, groups, calib, pose, keys=None):
    canvas = image.copy()
    for method in METHODS:
        for record in groups[method]:
            if keys is not None and record['_key'] != keys[method]:
                continue
            xy = project(record, calib, pose)
            if xy is None:
                continue
            pts = np.rint(np.clip(xy, -10000, 10000)).astype(int)
            for a, b in EDGES:
                line(canvas, pts[a], pts[b], COLORS[method], method == 'BEVFusion',
                     thickness=4 if method == 'GT' else 2)
    return canvas


def bev_overlay(points, groups, pose, size=720, bounds=None, keys=None):
    canvas = draw_bev(points, [], COLORS['GT'], pose, size=size, bounds=bounds)
    xl, xh, yl, yh = bounds or (-40., 40., -40., 40.)
    for method in METHODS:
        for record in groups[method]:
            if keys is not None and record['_key'] != keys[method]:
                continue
            local = to_ego(corners(record).T, pose)[[2, 3, 7, 6]]
            pts = np.rint(np.column_stack([(yh-local[:, 1])/(yh-yl)*(size-1),
                                          (xh-local[:, 0])/(xh-xl)*(size-1)])).astype(int)
            for i in range(4):
                line(canvas, pts[i], pts[(i+1) % 4], COLORS[method], method == 'BEVFusion',
                     thickness=4 if method == 'GT' else 2)
    return canvas


def header(width, title, subtitle):
    canvas = np.full((110, width, 3), 18, np.uint8)
    text(canvas, title, (20, 28), scale=.7)
    x = 20
    for method in METHODS:
        label = method + (' (dashed)' if method == 'BEVFusion' else ' (solid)')
        text(canvas, label, (x, 58), COLORS[method], .6)
        x += 380
    text(canvas, subtitle, (20, 87), scale=.53)
    return canvas


def side_overlay(path, points_ego, ego, groups, keys, target, label):
    rotation = Quaternion(target['rotation']).rotation_matrix
    center = np.array(target['translation'])
    points_global = points_ego @ Quaternion(ego['rotation']).rotation_matrix.T + np.array(ego['translation'])
    points = (points_global-center) @ rotation
    half_height = target['size'][2]/2
    keep = (np.abs(points[:, 0]) < 4) & (np.abs(points[:, 1]) < max(.8, target['size'][0]/2+.3))
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.scatter(points[keep, 0], points[keep, 2], s=2, color='.6', alpha=.5, label='Keyframe LiDAR (GT-aligned slice)')
    for method in METHODS:
        record = next(r for r in groups[method] if r['_key'] == keys[method])
        pts = (corners(record).T-center) @ rotation
        color = np.array(COLORS[method][::-1])/255.
        for n, (a, b) in enumerate(EDGES):
            ax.plot(pts[[a, b], 0], pts[[a, b], 2], '--' if method == 'BEVFusion' else '-',
                    color=color, linewidth=2, label=method if n == 0 else None)
    ax.axhline(-half_height, color='green', linestyle=':', linewidth=1, label='GT bottom reference (not fitted road)')
    ax.set(xlim=(-4, 4), ylim=(-half_height-.7, max(3., half_height+.6)),
           xlabel='GT local longitudinal X (m)', ylabel='Height relative to GT center (m)', title=label)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='upper right')
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='outputs/vf-car-truck-sweeps2-01')
    parser.add_argument('--sensor-root', default='data/nuscenes_vf6_01_adapted')
    parser.add_argument('--refined', help='Alternative source-index-preserving refined detection JSON')
    parser.add_argument('--output-dir')
    parser.add_argument('--refinement-label', default='Original GRM+PRM, input threshold .1, z_policy=center')
    args = parser.parse_args()
    run, source = Path(args.run_dir), Path(args.sensor_root)
    out = Path(args.output_dir) if args.output_dir else run/'visual_overlay'
    out.mkdir(parents=True, exist_ok=True)
    meta = run/'dataset_local/v1.0-mini'
    def table(name):
        return {r['token']: r for r in read(meta/(name+'.json'))}
    samples, sd, calibs, egos, sensors, instances, categories, annotations = [table(name) for name in
        ['sample', 'sample_data', 'calibrated_sensor', 'ego_pose', 'sensor', 'instance', 'category', 'sample_annotation']]
    by_sample = {t: {} for t in samples}
    for row in sd.values():
        if row['is_key_frame']:
            channel = sensors[calibs[row['calibrated_sensor_token']]['sensor_token']]['channel']
            by_sample[row['sample_token']][channel] = row
    paths = {'BEVFusion': run/'baseline/results_nusc_center_corrected.json',
             'BEVFusion + DetZero': Path(args.refined) if args.refined else run/'grm_prm/results_nusc_detzero_refined.json'}
    predictions = {m: read(p)['results'] for m, p in paths.items()}
    old_manifest = read(run/'visual_comparison/manifest.json')
    requests = [(f['index'], f['sample_token'], f['focus'], f['camera']) for f in old_manifest['frames']]
    # Add a visible car example so all three evaluable classes have close-ups.
    pairs = list(csv.DictReader((run/'analysis/fixed_gt_pairs.csv').open()))
    token = sorted(samples.values(), key=lambda r: r['timestamp'])[30]['token']
    candidates = []
    for row in pairs:
        if row['sample_token'] != token or row['class'] != 'car':
            continue
        for channel, sensor_row in by_sample[token].items():
            if not channel.startswith('CAM_'):
                continue
            xy = project(annotations[row['gt_token']], calibs[sensor_row['calibrated_sensor_token']], egos[sensor_row['ego_pose_token']])
            if xy is None:
                continue
            low = np.maximum(xy.min(axis=0), 0)
            high = np.minimum(xy.max(axis=0), [sensor_row['width'], sensor_row['height']])
            area = float(np.prod(np.maximum(high-low, 0)))
            if area > 100:
                candidates.append((area, row, channel))
    _, focus, camera = max(candidates, key=lambda v: v[0])
    requests.append((30, token, focus, camera))
    truck_candidates = [entry for entry in candidates if predictions['BEVFusion'][token][int(entry[1]['source_index'])]['detection_name'] == 'truck']
    if truck_candidates:
        _, truck_focus, camera = max(truck_candidates, key=lambda v: v[0])
        truck_focus = dict(truck_focus, evaluation_class='car', **{'class': 'truck'})
        requests.append((30, token, truck_focus, camera))
    manifest = {'min_score': .1, 'refinement': args.refinement_label,
                'hashes': {m: hashlib.sha256(p.read_bytes()).hexdigest() for m, p in paths.items()},
                'GT_policy': old_manifest['GT_policy'], 'projection': old_manifest['projection'],
                'styles': {'GT': 'green solid thick', 'BEVFusion': 'orange dashed', 'BEVFusion + DetZero': 'blue solid thin'},
                'bottom_reference': 'GT bottom, not independently estimated road plane', 'frames': []}
    contact = []
    for index, token, focus, camera in requests:
        records = by_sample[token]
        lidar, cam = records['LIDAR_TOP'], records[camera]
        ego, lc = egos[lidar['ego_pose_token']], calibs[lidar['calibrated_sensor_token']]
        cp, cc = egos[cam['ego_pose_token']], calibs[cam['calibrated_sensor_token']]
        groups = {'GT': []}
        origin = np.array(ego['translation'])
        for ann in annotations.values():
            if ann['sample_token'] != token or ann['num_lidar_pts'] == 0:
                continue
            cls = MAPPING.get(categories[instances[ann['instance_token']]['category_token']]['name'])
            if cls in RANGE and np.linalg.norm((np.array(ann['translation'])-origin)[:2]) < RANGE[cls]:
                groups['GT'].append(dict(ann, _class=cls, _key=ann['token']))
        for method in paths:
            groups[method] = []
            for i, record in enumerate(predictions[method][token]):
                cls = 'car' if record['detection_name'] == 'truck' else record['detection_name']
                if cls in RANGE and record['detection_score'] >= .1 and np.linalg.norm((np.array(record['translation'])-origin)[:2]) < RANGE[cls]:
                    groups[method].append(dict(record, _class=cls, _key=i))
        image = cv2.imread(str(source/cam['filename']))
        if image is None:
            raise FileNotFoundError(source/cam['filename'])
        raw = np.fromfile(source/lidar['filename'], dtype=np.float32).reshape(-1, 5)[:, :3]
        points = raw @ Quaternion(lc['rotation']).rotation_matrix.T + np.array(lc['translation'])
        stem = f"frame_{index:03d}_{focus['class']}"
        title = f"Frame {index:03d} | {camera} | GT / BEVFusion / BEVFusion + DetZero overlay"
        subtitle = 'Same image, calibration and coordinate frame | confidence >= 0.1 | BEV: X forward, Y left'
        whole = np.hstack([cv2.resize(camera_overlay(image, groups, cc, cp), (1280, 720)), bev_overlay(points, groups, ego)])
        cv2.imwrite(str(out/(stem+'_scene.png')), np.vstack([header(2000, title, subtitle), whole]))
        target = annotations[focus['gt_token']]
        keys = {'GT': focus['gt_token'], 'BEVFusion': int(focus['source_index']), 'BEVFusion + DetZero': int(focus['source_index'])}
        focus = dict(focus)
        refined_record = predictions['BEVFusion + DetZero'][token][int(focus['source_index'])]
        def box(record):
            return SimpleNamespace(**{k: record[k] for k in ['translation', 'size', 'rotation']})
        focus['grm_prm_iou3d'] = float(iou3d_matrix([box(refined_record)], [box(target)])[0, 0])
        focus['grm_prm_z_residual'] = refined_record['translation'][2] - target['translation'][2]
        focus['grm_prm_height'] = refined_record['size'][2]
        xy = project(target, cc, cp)
        center = (xy.min(axis=0)+xy.max(axis=0))/2
        width = min(image.shape[1], max(260., np.ptp(xy[:, 0])*2.2))
        height = min(image.shape[0], width*.75)
        left = int(np.clip(center[0]-width/2, 0, image.shape[1]-width))
        top = int(np.clip(center[1]-height/2, 0, image.shape[0]-height))
        marked = camera_overlay(image, groups, cc, cp, keys)[top:top+int(height), left:left+int(width)]
        pc = to_ego(np.array(target['translation'])[None], ego)[0]
        radius = max(2.5, target['size'][1]/2+1.)
        bounds = (pc[0]-radius, pc[0]+radius, pc[1]-radius, pc[1]+radius)
        body = np.hstack([cv2.resize(marked, (800, 600)), bev_overlay(points, groups, ego, size=600, bounds=bounds, keys=keys)])
        dz = float(focus['grm_prm_z_residual'])-float(focus['baseline_z_residual'])
        dh = float(focus['grm_prm_height'])-float(focus['baseline_height'])
        caption = f"{focus['class']} | IoU 3D {float(focus['baseline_iou3d']):.3f} -> {float(focus['grm_prm_iou3d']):.3f} | bottom change {dz-dh/2:+.3f} m"
        zoom = np.vstack([header(1400, title, caption), body])
        cv2.imwrite(str(out/(stem+'_zoom.png')), zoom)
        side_overlay(out/(stem+'_height.png'), points, ego, groups, keys, target, caption)
        contact.append(cv2.resize(zoom, (700, 355)))
        manifest['frames'].append({'index': index, 'sample_token': token, 'camera': camera, 'focus': focus,
                                   'files': [stem+s+'.png' for s in ['_scene', '_zoom', '_height']],
                                   'camera_source': str(source/cam['filename']), 'lidar_source': str(source/lidar['filename'])})
        print(stem, camera, 'bottom change', round(dz-dh/2, 4))
    if len(contact) % 2:
        contact.append(np.full_like(contact[0], 18))
    cv2.imwrite(str(out/'contact_sheet.png'), np.vstack([np.hstack(contact[i:i+2]) for i in range(0, 6, 2)]))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out/'README.md').write_text('''# Overlaid comparison

Green thick solid = GT; orange dashed = BEVFusion; blue thin solid = BEVFusion + GRM + PRM, original input threshold 0.1.

`*_scene.png`: all evaluable boxes, overlaid in the same camera and BEV. `*_zoom.png`: only the selected GT/prediction proximity pair, overlaid. `*_height.png`: the same boxes projected onto the GT-local X/Z plane, with real keyframe LiDAR points in a lateral slice. Coordinates, dimensions and scores are unchanged.

GT car includes truck; display classes/ranges and nonzero-point GT policy match the earlier evaluator. Camera uses its own timestamp pose. BEV uses keyframe ego axes and equal metric scale. The GT bottom reference in height figures is NOT a fitted road plane. Close-by pairing is diagnostic, not independently verified identity. Examples are selected for visibility, not population statistics. See manifest for input hashes and exact source boxes.

Reproduce from repository root: `python3 tools/audit_vf_box_heights.py` then `python3 tools/render_vf_overlay.py`. This overlay renderer reuses the frame selection in `visual_comparison/manifest.json`.
''')
    with (out/'README.md').open('a') as f:
        f.write('\nActual refinement input: ' + args.refinement_label + '\n')
    print('Saved', out)


if __name__ == '__main__':
    main()
