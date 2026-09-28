"""Audit custom nuScenes sensor/label contracts without importing CUDA or MMCV."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from pyquaternion import Quaternion


def validate(root, version='v1.0-trainval', infos=None, full=False):
    root = Path(root).resolve()
    def table(name):
        rows = json.loads((root / version / (name + '.json')).read_text())
        assert len({r['token'] for r in rows}) == len(rows), name + ': duplicate tokens'
        return {r['token']: r for r in rows}
    samples, scenes, sd = table('sample'), table('scene'), table('sample_data')
    egos, calibs, sensors = table('ego_pose'), table('calibrated_sensor'), table('sensor')
    annotations, instances, categories = table('sample_annotation'), table('instance'), table('category')
    required = {'LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}
    by_sample = {t: {} for t in samples}
    errors, warnings = [], []
    gaps, camera_dt, point_counts, image_sizes = [], [], [], Counter()
    intensity_min, intensity_max = float('inf'), -float('inf')
    total_points = 0
    for scene in scenes.values():
        seq = sorted((s for s in samples.values() if s['scene_token'] == scene['token']), key=lambda s: s['timestamp'])
        assert len(seq) == scene['nbr_samples'], 'scene length mismatch'
        assert seq[0]['token'] == scene['first_sample_token'] and seq[-1]['token'] == scene['last_sample_token']
        for i, sample in enumerate(seq):
            assert sample['prev'] == (seq[i-1]['token'] if i else '')
            assert sample['next'] == (seq[i+1]['token'] if i+1 < len(seq) else '')
        diff = np.diff([s['timestamp'] for s in seq]) / 1000
        assert np.all(diff > 0), 'non-increasing timestamp'
        gaps.extend(diff.tolist())
    for row in sd.values():
        assert row['sample_token'] in samples and row['ego_pose_token'] in egos
        calib = calibs[row['calibrated_sensor_token']]
        channel = sensors[calib['sensor_token']]['channel']
        if not row['is_key_frame']:
            continue
        assert channel not in by_sample[row['sample_token']], 'duplicate keyframe channel'
        by_sample[row['sample_token']][channel] = row
        path = root / row['filename']
        if not path.is_file():
            errors.append('Missing sensor: ' + str(path)); continue
        if channel.startswith('CAM_'):
            camera_dt.append(abs(row['timestamp'] - samples[row['sample_token']]['timestamp']) / 1000)
            if full:
                with Image.open(path) as img:
                    img.load()
                    image_sizes[str(img.size)] += 1
                    assert img.size == (row['width'], row['height']), 'image metadata size mismatch'
        elif channel == 'LIDAR_TOP':
            assert path.stat().st_size % 20 == 0, 'invalid float32 x5 point file'
            n = path.stat().st_size // 20
            point_counts.append(n); total_points += n
            if full:
                points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
                assert np.isfinite(points).all(), 'nonfinite point'
                intensity_min = min(intensity_min, float(points[:, 3].min()))
                intensity_max = max(intensity_max, float(points[:, 3].max()))
    for token, channels in by_sample.items():
        if not required <= set(channels): errors.append('Missing camera/lidar channel: ' + token)
    source = root / 'source_extrinsics.json'
    meta_path = root / 'conversion_meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if source.exists():
        matrices = json.loads(source.read_text())
        channel_to_source = {v: k for k, v in meta['camera_mapping'].items()}
        for calib in calibs.values():
            channel = sensors[calib['sensor_token']]['channel']
            if channel not in channel_to_source: continue
            pose = np.eye(4); pose[:3, :3] = Quaternion(calib['rotation']).rotation_matrix
            pose[:3, 3] = calib['translation']
            assert np.allclose(pose, np.linalg.inv(matrices[channel_to_source[channel]]), atol=2e-5), 'extrinsic mismatch: ' + channel
            expected = meta['camera_intrinsics'][channel_to_source[channel]]['K_new']
            assert np.allclose(calib['camera_intrinsic'], expected), 'intrinsic mismatch: ' + channel
    class_counts = Counter()
    for ann in annotations.values():
        assert ann['sample_token'] in samples
        cls = categories[instances[ann['instance_token']]['category_token']]['name']
        class_counts[cls] += 1
        assert np.isfinite(ann['translation'] + ann['size'] + ann['rotation']).all()
        assert min(ann['size']) > 0
        for key, opposite in [('prev', 'next'), ('next', 'prev')]:
            if ann[key]:
                other = annotations[ann[key]]
                assert other['instance_token'] == ann['instance_token'] and other[opposite] == ann['token'], 'broken annotation chain'
    identity = all(np.allclose(e['translation'], 0) and np.allclose(Quaternion(e['rotation']).rotation_matrix, np.eye(3)) for e in egos.values())
    if identity: warnings.append('Identity ego poses: global trajectories and velocities are not established.')
    if annotations and all(not a['attribute_tokens'] for a in annotations.values()): warnings.append('No GT attributes: do not report NDS/mAAE as a validated VF benchmark.')
    pkl_report = None
    if infos:
        data = pickle.loads(Path(infos).read_bytes())
        assert {i['token'] for i in data['infos']} == set(samples), 'info/sample token mismatch'
        missing_paths, outside_root_paths = 0, 0
        for item in data['infos']:
            paths = [item['lidar_path']] + [c['data_path'] for c in item['cams'].values()]
            missing_paths += sum(not Path(p).is_file() for p in paths)
            outside_root_paths += sum(root not in Path(p).resolve().parents for p in paths)
        pkl_report = {'frames': len(data['infos']), 'missing_paths': missing_paths,
                      'outside_root_paths': outside_root_paths, 'metadata': data['metadata']}
        if missing_paths: errors.append(f'PKL contains {missing_paths} missing paths; regenerate on the target machine.')
        if outside_root_paths: errors.append(f'PKL contains {outside_root_paths} paths outside this dataset root; regenerate.')
    report = {'data_root': str(root), 'version': version, 'scenes': len(scenes), 'samples': len(samples),
              'sensor_records': len(sd), 'annotations': len(annotations), 'instances': len(instances),
              'class_counts': dict(class_counts), 'identity_ego_pose': identity,
              'frame_dt_ms': [min(gaps), float(np.median(gaps)), max(gaps)] if gaps else [],
              'camera_abs_dt_ms': {'mean': float(np.mean(camera_dt)), 'max': max(camera_dt)} if camera_dt else {},
              'lidar_points': {'min': min(point_counts), 'max': max(point_counts), 'total': total_points},
              'image_sizes': dict(image_sizes), 'intensity_range': [intensity_min, intensity_max] if full else None,
              'zero_point_gt': sum(a['num_lidar_pts'] == 0 for a in annotations.values()),
              'infos': pkl_report, 'full_sensor_read': full, 'errors': errors, 'warnings': warnings,
              'detection_structure_ok': not errors, 'global_motion_ready': not identity}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/nuscenes_vf6_01_5hz')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--infos')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--output', default='outputs/vf6_01_5hz/data_audit.json')
    args = parser.parse_args()
    report = validate(args.data_root, args.version, args.infos, args.full)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    if report['errors']: raise SystemExit(1)


if __name__ == '__main__': main()
