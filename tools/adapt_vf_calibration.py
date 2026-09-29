"""Create a separate VF dataset in a reference nuScenes LiDAR/camera convention.

Camera: exact same-optical-center rotation/intrinsic homography (NO translation).
LiDAR: passive rigid coordinate change (NO simulation of a new scanner/viewpoint).
Annotations, sample tokens, timestamps and ego poses remain unchanged.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm


def rotation(q):
    return Quaternion(q).rotation_matrix


def camera_homography(source, target):
    if not np.allclose(source['translation'], target['translation']):
        raise ValueError('A depth-free camera homography cannot translate the optical center')
    return (np.asarray(target['camera_intrinsic']) @ rotation(target['rotation']).T @
            rotation(source['rotation']) @ np.linalg.inv(source['camera_intrinsic']))


def transform_points(points, source, target):
    out = points.copy()
    ego = points[:, :3].astype(np.float64) @ rotation(source['rotation']).T + source['translation']
    out[:, :3] = (ego - target['translation']) @ rotation(target['rotation'])
    return out


def model_affine(width, height, resize=.48, final_dim=(256, 704)):
    """Exact deterministic ImageAug3D parameters with bot_pct_lim=[0, 0]."""
    new_w, new_h = int(width * resize), int(height * resize)
    crop_x = int(max(0, new_w-final_dim[1]) / 2)
    crop_y = new_h-final_dim[0]
    return np.array([[resize, 0., -crop_x], [0., resize, -crop_y], [0., 0., 1.]])


def read_tables(root, version):
    folder = Path(root) / version
    return {p.stem: json.loads(p.read_text()) for p in folder.glob('*.json')}


def reference_rig(root, version, scene_name):
    tables = read_tables(root, version)
    scene = next(s for s in tables['scene'] if s['name'] == scene_name)
    sensor = {s['token']: s['channel'] for s in tables['sensor']}
    calibrated = {c['token']: c for c in tables['calibrated_sensor']}
    result = {}
    for sd in tables['sample_data']:
        if sd['is_key_frame'] and sd['sample_token'] == scene['first_sample_token']:
            c = calibrated[sd['calibrated_sensor_token']]
            result[sensor[c['sensor_token']]] = {'calibration': c, 'width': sd['width'], 'height': sd['height']}
    if 'LIDAR_TOP' not in result: raise ValueError('Reference scene has no LIDAR_TOP')
    return result, scene['first_sample_token']


def adapted_calibrations(tables, reference, mode):
    channels = {s['token']: s['channel'] for s in tables['sensor']}
    output = copy.deepcopy(tables['calibrated_sensor'])
    for c in output:
        channel = channels[c['sensor_token']]
        if channel == 'LIDAR_TOP' and mode in ('lidar', 'both'):
            ref = reference[channel]['calibration']
            c['rotation'] = ref['rotation']; c['translation'] = ref['translation']
        elif channel.startswith('CAM_') and mode in ('camera', 'both'):
            ref = reference[channel]['calibration']
            c['rotation'] = ref['rotation']; c['camera_intrinsic'] = ref['camera_intrinsic']
            # c['translation'] MUST remain the real VF camera center.
    return output


def adapt(source_root, output_root, reference_root, mode='both', version='v1.0-trainval',
          reference_version='v1.0-mini', reference_scene='scene-0103'):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError('Use a separate sibling output dataset; never overwrite the source')
    if output_root.exists(): raise FileExistsError(f'Output already exists: {output_root}; choose a new directory')
    if mode not in ('camera', 'lidar', 'both'): raise ValueError(mode)
    source = read_tables(source_root, version)
    if not all(np.allclose(e['translation'], 0) and np.allclose(rotation(e['rotation']), np.eye(3)) for e in source['ego_pose']):
        raise ValueError('This adapter is scoped to the VF identity-ego single-frame export')
    reference, reference_token = reference_rig(reference_root, reference_version, reference_scene)
    target = copy.deepcopy(source)
    target['calibrated_sensor'] = adapted_calibrations(source, reference, mode)
    old_calib = {c['token']: c for c in source['calibrated_sensor']}
    new_calib = {c['token']: c for c in target['calibrated_sensor']}
    sensors = {s['token']: s['channel'] for s in source['sensor']}
    output_root.mkdir(parents=True)
    report = {'mode': mode, 'source_root': str(source_root), 'reference_root': str(Path(reference_root).resolve()),
              'reference_version': reference_version, 'reference_scene': reference_scene,
              'reference_sample_token': reference_token, 'camera': {}, 'lidar': {},
              'status': 'building', 'ego_motion_available': False,
              'notes': ['Camera translation is physical VF, not nuScenes.',
                        'LiDAR is a coordinate change of existing returns, not resimulated rays.',
                        'Missing visibility, density and occlusions cannot be reconstructed.',
                        'GT/sample tokens/timestamps are unchanged; no GT used to choose the transform.']}
    manifest = output_root / 'calibration_adapter.json'
    manifest.write_text(json.dumps(report, indent=2))
    for sd in tqdm(target['sample_data'], desc=f'Adapt {mode}'):
        path = source_root / sd['filename']; out = output_root / sd['filename']
        out.parent.mkdir(parents=True, exist_ok=True)
        src = old_calib[sd['calibrated_sensor_token']]; dst = new_calib[sd['calibrated_sensor_token']]
        channel = sensors[src['sensor_token']]
        if channel.startswith('CAM_') and mode in ('camera', 'both'):
            image = cv2.imread(str(path))
            if image is None: raise FileNotFoundError(path)
            h = camera_homography(src, dst)
            width, height = reference[channel]['width'], reference[channel]['height']
            warped = cv2.warpPerspective(image, h, (width, height), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if not cv2.imwrite(str(out), warped, [cv2.IMWRITE_JPEG_QUALITY, 95]): raise IOError(out)
            sd['width'], sd['height'] = width, height
            if channel not in report['camera']:
                mask = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), h, (width, height), flags=cv2.INTER_NEAREST)
                affine = model_affine(width, height)
                crop_mask = cv2.warpPerspective(mask, affine, (704,256), flags=cv2.INTER_NEAREST)
                mask_path = output_root / 'calibration_masks' / (channel + '.png')
                mask_path.parent.mkdir(exist_ok=True); cv2.imwrite(str(mask_path), mask)
                report['camera'][channel] = {'homography': h.tolist(), 'width': width, 'height': height,
                                            'source_translation': src['translation'], 'target_translation': dst['translation'],
                                            'valid_pixel_fraction': float(np.mean(mask > 0)),
                                            'valid_model_crop_fraction': float(np.mean(crop_mask > 0))}
        elif channel == 'LIDAR_TOP' and mode in ('lidar', 'both'):
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
            if not np.isfinite(points).all(): raise ValueError('Nonfinite source LiDAR')
            transform_points(points, src, dst).tofile(out)
            report['lidar'] = {'source_rotation': src['rotation'], 'source_translation': src['translation'],
                               'target_rotation': dst['rotation'], 'target_translation': dst['translation']}
        else:
            out.symlink_to(os.path.relpath(path, out.parent))
    folder = output_root / version; folder.mkdir()
    for name, rows in target.items():
        (folder / (name+'.json')).write_text(json.dumps(rows))
    (output_root / 'sweeps').mkdir()
    if (source_root / 'maps').is_dir():
        (output_root / 'maps').symlink_to(os.path.relpath(source_root / 'maps', output_root), target_is_directory=True)
    for name in ['phenikaa_val_scenes.txt', 'selection_5hz.json', 'annotation_import_stats.json']:
        src = source_root / name
        if src.exists(): (output_root / name).write_bytes(src.read_bytes())
    report.update(status='complete', samples=len(target['sample']), sensor_records=len(target['sample_data']),
                  source_annotations_sha256=hashlib.sha256((source_root/version/'sample_annotation.json').read_bytes()).hexdigest(),
                  source_calibration_sha256=hashlib.sha256((source_root/version/'calibrated_sensor.json').read_bytes()).hexdigest())
    manifest.write_text(json.dumps(report, indent=2))
    (output_root / 'conversion_meta.json').write_text(json.dumps({
        'dataset': 'vf_calibration_ablation', 'source_root': str(source_root),
        'version': version, 'split': 'custom_val', 'ego_motion_available': False,
        'calibration_adapter': report}, indent=2))
    print('Created:', output_root)
    print('Next: create_vf_infos.py --root-path', output_root)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', default='data/nuscenes_vf6_01_5hz')
    p.add_argument('--output-root', required=True)
    p.add_argument('--reference-root', default='data/nuscenes')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--reference-version', default='v1.0-mini')
    p.add_argument('--reference-scene', default='scene-0103')
    p.add_argument('--mode', choices=['camera','lidar','both'], default='both')
    a = p.parse_args()
    adapt(a.source_root,a.output_root,a.reference_root,a.mode,a.version,a.reference_version,a.reference_scene)


if __name__ == '__main__': main()
