#!/usr/bin/env python3
"""Offline detection -> class-preserving tracking -> optional GRM/PRM, without CRM."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import torch
import yaml
from easydict import EasyDict

DETZERO_ROOT = Path(__file__).resolve().parent.parent
for directory in [DETZERO_ROOT, DETZERO_ROOT / 'tracking', DETZERO_ROOT / 'refining', DETZERO_ROOT / 'utils']:
    sys.path.insert(0, str(directory))
from integration import bridge
from integration.gap_fill import export_gap_filled
from detzero_utils.config_utils import cfg_from_yaml_file
from detzero_track.models.detzero_tracker import DetZeroTracker
from detzero_refine.models import build_network

TRACKING_CLASSES = {'car', 'truck', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle'}


class StandaloneDataset:
    tta = False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results_path', required=True)
    p.add_argument('--data_root', default='data/nuscenes')
    p.add_argument('--version', default='v1.0-mini')
    p.add_argument('--tracking_cfg', default=str(DETZERO_ROOT / 'tracking/tools/cfgs/tk_model_cfgs/nuscenes_detzero_track.yaml'))
    p.add_argument('--checkpoint_dir', default=str(DETZERO_ROOT / 'checkpoints'))
    p.add_argument('--output_dir', default='outputs/detzero_refined')
    p.add_argument('--min_score', type=float, default=0.1)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--tracking-device', choices=['cpu', 'cuda'],
                   help='Optional CPU/Shapely association while refiners run on CUDA')
    p.add_argument('--classes', default='car,truck,bus,trailer,construction_vehicle,pedestrian,motorcycle,bicycle')
    p.add_argument('--refinement', choices=['none', 'geometry', 'full'], default='full')
    p.add_argument('--skip_refining', action='store_true', help='Compatibility alias for --refinement none')
    p.add_argument('--coordinate-mode', choices=['global', 'ego'], default='global',
                   help='ego is an explicitly uncompensated diagnostic; PRM is disabled')
    p.add_argument('--assume-stationary', action='store_true', help='Use identity poses as global ONLY if the sensor rig is known to be stationary')
    p.add_argument('--size_policy', choices=['auto', 'detector_prior', 'grm'], default='auto',
                   help='auto accepts GRM within 25%% of detector median, for every selected class')
    p.add_argument('--z-policy', choices=['center', 'preserve_bottom'], default='center')
    p.add_argument('--intensity-mode', choices=['unit', 'raw', 'tanh'], default='unit',
                   help='unit divides 0..255 by 255; transfer choice, not calibrated to Waymo')
    p.add_argument('--max-gap-seconds', type=float, default=0.6)
    p.add_argument('--allow-refine-errors', action='store_true', help='Record explicit per-track fallbacks instead of stopping on model errors')
    p.add_argument('--fill-missed-detections', action='store_true',
                   help='Also write separate detection/tracking JSONs with bounded internal gaps filled')
    p.add_argument('--gap-fill-method', choices=['tracker', 'interpolate'], default='tracker')
    p.add_argument('--gap-fill-max-seconds', type=float, default=0.6,
                   help='Maximum TOTAL elapsed time between the two observed anchors')
    p.add_argument('--gap-fill-score-decay', type=float, default=0.8)
    p.add_argument('--gap-fill-min-score', type=float, default=0.1)
    p.add_argument('--gap-fill-iou', type=float, default=0.1)
    p.add_argument('--gap-fill-center-distance', type=float, default=0.5)
    return p.parse_args()


def check_coordinates(prepared, coordinate_mode, refinement, assume_stationary=False):
    if coordinate_mode == 'ego':
        if not prepared['identity_ego_pose']:
            raise ValueError('ego mode currently requires identity ego poses; use global for a measured trajectory')
        if refinement == 'full':
            raise ValueError('PRM requires a consistent world frame. Use --coordinate-mode ego --refinement geometry (or none) for diagnostics only.')
    elif prepared['identity_ego_pose'] and not assume_stationary:
        raise ValueError('All ego poses are identity. Supply real poses, or explicitly use --coordinate-mode ego --refinement geometry. --assume-stationary requires a verified stationary sensor rig.')


def load_refining_models(checkpoint_dir, device, class_names=None, stages=('grm', 'prm')):
    models = {'grm': {}, 'prm': {}, 'checkpoints': {}}
    for cls in sorted(class_names if class_names is not None else bridge.MODEL_PREFIX):
        for stage in stages:
            prefix = bridge.MODEL_PREFIX[cls]
            path = Path(checkpoint_dir) / f'{prefix}_{stage}_model.pth'
            cfg_path = DETZERO_ROOT / f'refining/tools/cfgs/ref_model_cfgs/{prefix}_{stage}_model.yaml'
            cfg = EasyDict(yaml.safe_load(cfg_path.read_text()))
            model = build_network(cfg.MODEL, dataset=StandaloneDataset())
            options = {'map_location': 'cpu'}
            if 'weights_only' in inspect.signature(torch.load).parameters:
                options['weights_only'] = False  # User-supplied trusted local checkpoints.
            ckpt = torch.load(str(path), **options)
            model.load_state_dict(ckpt.get('model_state', ckpt), strict=True)
            models[stage][cls] = model.to(device).eval()
            models['checkpoints'][f'{cls}/{stage}'] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            print(f'[Checkpoint] {cls}/{stage}: all model keys matched', flush=True)
    return models


def run_tracking(prepared, tracking_cfg_path, device, max_gap_seconds=0.6):
    if max_gap_seconds <= 0:
        raise ValueError('max-gap-seconds must be positive')
    cfg = EasyDict(); cfg_from_yaml_file(tracking_cfg_path, cfg)
    cfg.MODEL.DEVICE = device
    tracks = {}
    for scene, frames in prepared['frames'].items():
        dt = np.diff([f['timestamp'] for f in frames.values()])
        scene_cfg = copy.deepcopy(cfg.MODEL)
        if len(dt):
            scene_cfg.TRACKING.FILTER.DELTA_T = float(np.median(dt))
            scene_cfg.TRACKING.TRACK_AGE.DEATH_AGE = max(1, int(np.ceil(max_gap_seconds / np.median(dt))))
        for name in prepared['classes']:
            selected = {}
            for idx, frame in frames.items():
                selected[idx] = frame.copy()
                mask = frame['nusc_name'] == name
                for key in ['boxes_global', 'name', 'nusc_name', 'score', 'source_index']:
                    selected[idx][key] = frame[key][mask].copy()
            if not any(len(f['score']) for f in selected.values()): continue
            result = DetZeroTracker(copy.deepcopy(scene_cfg)).forward(selected)
            for tid, track in result.items():
                track['sequence_name'] = scene
                track['nusc_name'] = name
                tracks[f'{scene}_{name}_{tid}'] = track
        print(f'[Tracking] {scene}: {len(frames)} frames', flush=True)
    return tracks


def collect_track_pointclouds(prepared, tracks, data_root, intensity_mode='unit'):
    # Read one cloud at a time; do not cache a complete multi-scene dataset in RAM.
    requests = defaultdict(list)
    for tid, track in tracks.items():
        track['pts'] = [None] * len(track['sample_idx'])
        for i, idx in enumerate(track['sample_idx']):
            requests[(track['sequence_name'], str(idx))].append((tid, i))
    for (scene, idx), objects in requests.items():
        frame = prepared['frames'][scene][idx]
        points = bridge.read_points(Path(data_root) / frame['lidar_path'], frame['pose'], intensity_mode)
        for tid, i in objects:
            tracks[tid]['pts'][i] = bridge.crop_points(points, tracks[tid]['boxes_global'][i], scale=1.1)


def tensor_batch(features, device):
    return {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v for k, v in features.items()}


def position_chunks(track, model, device, rng):
    """Up to 200 proposals per PRM call, with 16 context frames on either side."""
    n = len(track['sample_idx']); output = np.empty((n, 7), dtype=np.float32)
    for start in range(0, n, 168):
        end = min(start + 168, n); lo, hi = max(0, start-16), min(n, end+16)
        chunk = {k: (v[lo:hi] if k in ['boxes_global', 'score', 'sample_idx', 'pts', 'name'] else v)
                 for k, v in track.items()}
        cls = track['name'][0]
        features, origin = bridge.position_features(chunk, rng, with_class=(cls == 'Cyclist'))
        pred, _, _ = model(tensor_batch(features, device))
        boxes = bridge.position_to_global(pred['pred_boxes'][0, :hi-lo], origin)
        output[start:end] = boxes[start-lo:end-lo]
    return output


def run_refining(tracks, models, device, size_policy='auto', refinement='full', z_policy='center', allow_errors=False):
    refined, status = {}, {}
    for tid, track in tracks.items():
        original = track['boxes_global'][:, :7]
        info = {'frames': len(original), 'grm': 'not_requested', 'prm': 'not_requested'}
        status[tid] = info
        refined[tid] = original.copy()
        if refinement == 'none':
            info['status'] = 'tracking_only'; continue
        if sum(len(p) for p in track.get('pts', [])) < 10:
            info['status'] = 'insufficient_points'; continue
        rng = np.random.default_rng(int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16))
        cls = track['name'][0]
        try:
            prior = np.median(original[:, 3:6], axis=0)
            size = prior
            with torch.no_grad():
                if size_policy == 'detector_prior':
                    info['grm'] = 'detector_prior'
                else:
                    pred, _, _ = models['grm'][cls](tensor_batch(bridge.geometry_features(track, rng), device))
                    size_pred = pred['pred_boxes'][0, 3:6]
                    if not np.isfinite(size_pred).all() or np.any(size_pred <= 0):
                        raise ValueError('Invalid GRM dimensions')
                    ratio = size_pred / prior
                    if size_policy == 'grm' or np.all((ratio >= .75) & (ratio <= 1.25)):
                        size = size_pred; info['grm'] = 'accepted'
                    else: info['grm'] = 'rejected_using_prior'
                position = original.copy()
                if refinement == 'full':
                    position = position_chunks(track, models['prm'][cls], device, rng)
                    info['prm'] = 'applied'
                refined[tid] = bridge.combine_boxes(np.r_[np.zeros(3), size, 0], position,
                                                     original_boxes=original, z_policy=z_policy)
            info['status'] = 'processed'
        except Exception as exc:
            if not allow_errors: raise RuntimeError(f'Refinement failed for {tid}') from exc
            info.update(status='error_fallback', error=str(exc), grm='not_applied', prm='not_applied')
            print(f'[Fallback] {tid}: {exc}', flush=True)
    return refined, status


def export_tracking_results(prepared, tracks, refined_boxes, output_file, coordinate_mode='global'):
    output = {'meta': copy.deepcopy(prepared['original']['meta']),
              'results': {token: [] for token in prepared['original']['results']}}
    if coordinate_mode == 'ego':
        output['coordinate_mode'] = 'ego_uncompensated_diagnostic'
    claimed = set()
    for tid, track in sorted(tracks.items(), key=lambda kv: (-len(kv[1]['sample_idx']), kv[0])):
        name = track['nusc_name']
        if name not in TRACKING_CLASSES: continue
        boxes = refined_boxes.get(tid, track['boxes_global'][:, :7])
        velocities = (np.gradient(boxes[:, :2], track['timestamp'], axis=0)
                      if len(boxes) > 1 else track['boxes_global'][:, 7:9])
        for i, idx in enumerate(track['sample_idx']):
            token = prepared['frames'][track['sequence_name']][str(idx)]['sample_token']
            source = int(track['source_index'][i])
            # Matched observations only: no duplicate or fabricated detections.
            if source < 0 or (token, source) in claimed: continue
            original = prepared['original']['results'][token][source]
            if original['detection_name'] != name: raise ValueError('Source class mismatch')
            box = bridge.replace_box(original, boxes[i])
            record = {k: box[k] for k in ['sample_token', 'translation', 'size', 'rotation']}
            record.update(tracking_id=tid, tracking_name=name, tracking_score=original['detection_score'])
            if coordinate_mode == 'global':
                record['velocity'] = velocities[i].tolist()
            output['results'][token].append(record); claimed.add((token, source))
    Path(output_file).write_text(json.dumps(output, allow_nan=False))
    return output


def main():
    args = parse_args()
    if args.skip_refining: args.refinement = 'none'
    if not 0 <= args.min_score <= 1: raise ValueError('min_score must be in [0, 1]')
    if args.fill_missed_detections and args.coordinate_mode != 'global':
        raise ValueError('Gap export requires a consistent global frame')
    classes = list(dict.fromkeys(c.strip() for c in args.classes.split(',') if c.strip()))
    if not classes or set(classes) - set(bridge.CLASS_MAP): raise ValueError('Unsupported/empty classes')
    prepared = bridge.prepare(args.results_path, args.data_root, args.version, classes, args.min_score)
    check_coordinates(prepared, args.coordinate_mode, args.refinement, args.assume_stationary)
    tracks = run_tracking(prepared, args.tracking_cfg, args.tracking_device or args.device, args.max_gap_seconds)
    stages = []
    if args.refinement != 'none' and args.size_policy != 'detector_prior': stages.append('grm')
    if args.refinement == 'full': stages.append('prm')
    needed = {t['name'][0] for t in tracks.values()}
    models = load_refining_models(args.checkpoint_dir, args.device, needed, stages)
    if args.refinement != 'none': collect_track_pointclouds(prepared, tracks, args.data_root, args.intensity_mode)
    refined, status = run_refining(tracks, models, args.device, args.size_policy, args.refinement,
                                   args.z_policy, args.allow_refine_errors)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if models['checkpoints']:
        prepared['original']['meta']['use_external'] = True
    detection, replaced = bridge.export_detection(prepared, tracks, refined)
    if args.coordinate_mode == 'ego': detection['coordinate_mode'] = 'ego_uncompensated_diagnostic'
    (output_dir / 'results_nusc_detzero_refined.json').write_text(json.dumps(detection, allow_nan=False))
    tracking_name = 'tracks_ego_diagnostic.json' if args.coordinate_mode == 'ego' else 'results_nusc_detzero_tracking.json'
    tracking = export_tracking_results(prepared, tracks, refined, output_dir / tracking_name, args.coordinate_mode)
    gap_report = None
    if args.fill_missed_detections:
        gap_detection, gap_tracking, gap_report = export_gap_filled(
            prepared, tracks, refined, detection, tracking, method=args.gap_fill_method,
            max_gap_seconds=args.gap_fill_max_seconds, score_decay=args.gap_fill_score_decay,
            min_score=args.gap_fill_min_score, iou_threshold=args.gap_fill_iou,
            center_distance=args.gap_fill_center_distance)
        (output_dir / 'results_nusc_detzero_gap_filled.json').write_text(json.dumps(gap_detection, allow_nan=False))
        (output_dir / 'results_nusc_detzero_tracking_gap_filled.json').write_text(json.dumps(gap_tracking, allow_nan=False))
        (output_dir / 'gap_fill_report.json').write_text(json.dumps(gap_report, indent=2, allow_nan=False))
        print(f'[Gap fill] {gap_report["counts"]}', flush=True)
    (output_dir / 'tracks.pkl').write_bytes(pickle.dumps({k: {n: v for n, v in t.items() if n != 'pts'} for k, t in tracks.items()}, protocol=4))
    (output_dir / 'refined_boxes.pkl').write_bytes(pickle.dumps(refined, protocol=4))
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=DETZERO_ROOT, capture_output=True, text=True)
    diff = subprocess.run(['git', 'diff', 'HEAD', '--', '.'], cwd=DETZERO_ROOT, capture_output=True)
    report = {'arguments': vars(args), 'git_commit': git.stdout.strip(), 'input_sha256': prepared['source_sha256'],
              'git_diff_sha256': hashlib.sha256(diff.stdout).hexdigest(),
              'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__},
              'frames': sum(len(f) for f in prepared['frames'].values()),
              'tracking_config_sha256': hashlib.sha256(Path(args.tracking_cfg).read_bytes()).hexdigest(),
              'identity_ego_pose': prepared['identity_ego_pose'], 'coordinate_mode': args.coordinate_mode,
              'checkpoints': models['checkpoints'], 'tracks': len(tracks),
              'matched_boxes_replaced': replaced, 'status_counts': dict(Counter(s['status'] for s in status.values())),
              'grm_counts': dict(Counter(s['grm'] for s in status.values())),
              'prm_counts': dict(Counter(s['prm'] for s in status.values())), 'track_status': status,
              'gap_fill_counts': gap_report['counts'] if gap_report else None,
              'notes': ['Offline; untracked detections and detector scores are preserved.',
                        'No CRM. Ego diagnostics do not establish physical world trajectories.']}
    (output_dir / 'run_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Completed: {len(tracks)} tracks, {replaced} matched boxes replaced; {report["status_counts"]}')
    print(f'Outputs: {output_dir}')


if __name__ == '__main__': main()
