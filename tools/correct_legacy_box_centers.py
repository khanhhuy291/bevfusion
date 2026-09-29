"""Opt-in MIT legacy JSON center correction; never overwrites raw predictions.

Only use for a checkpoint whose decoded tensor Z represents geometric center
but was wrapped as bottom-centered and exported using gravity_center.
This is NOT a generic nuScenes correction. No GT is used by this tool.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion

CONVENTION = 'mit-legacy-center-as-bottom'


def lidar_global_poses(root, version):
    def table(name):
        return {r['token']: r for r in json.loads((Path(root)/version/(name+'.json')).read_text())}
    sensors, calibs, egos = table('sensor'), table('calibrated_sensor'), table('ego_pose')
    result = {}
    for row in table('sample_data').values():
        c = calibs[row['calibrated_sensor_token']]
        if row['is_key_frame'] and sensors[c['sensor_token']]['channel'] == 'LIDAR_TOP':
            e = egos[row['ego_pose_token']]
            r = Quaternion(e['rotation']).rotation_matrix @ Quaternion(c['rotation']).rotation_matrix
            t = Quaternion(e['rotation']).rotation_matrix @ np.asarray(c['translation']) + e['translation']
            result[row['sample_token']] = (r, t)
    return result


def correct(document, poses, convention):
    if convention != CONVENTION:
        raise ValueError('Explicit legacy source convention required')
    if 'legacy_center_correction' in document:
        raise ValueError('Already center-corrected; refusing to apply twice')
    if set(document['results']) != set(poses):
        raise ValueError('Prediction tokens and calibration dataset differ')
    out = copy.deepcopy(document)
    count = 0
    for token, boxes in out['results'].items():
        up = poses[token][0][:, 2]
        for box in boxes:
            if box['sample_token'] != token:
                raise ValueError('Prediction sample token mismatch')
            height = float(box['size'][2])
            xyz = np.asarray(box['translation'], dtype=float)
            if not np.isfinite(xyz).all() or not np.isfinite(height) or height <= 0:
                raise ValueError('Invalid box geometry')
            # Subtract along source LiDAR +Z expressed in GLOBAL, not global Z.
            box['translation'] = (xyz - up * height / 2).tolist()
            count += 1
    out['legacy_center_correction'] = {
        'source_convention': convention, 'boxes': count,
        'operation': 'global_center -= R_global_lidar[:,2] * predicted_height / 2',
        'target_convention': 'geometric center',
        'warning': 'Checkpoint-specific, not a generic nuScenes JSON operation. Raw export range filtering is not rerun.'}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, help='Exact calibration dataset used for inference')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source-convention', required=True, choices=[CONVENTION])
    a = p.parse_args()
    src, dst = Path(a.input), Path(a.output)
    if dst.exists() or src.resolve() == dst.resolve():
        raise FileExistsError('Choose a new output; existing files are never overwritten')
    raw = src.read_bytes()
    out = correct(json.loads(raw), lidar_global_poses(a.data_root, a.version), a.source_convention)
    out['legacy_center_correction']['source_sha256'] = hashlib.sha256(raw).hexdigest()
    out['legacy_center_correction']['calibration_sha256'] = hashlib.sha256((Path(a.data_root)/a.version/'calibrated_sensor.json').read_bytes()).hexdigest()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, allow_nan=False))
    print('Saved', dst, '| corrected boxes:', out['legacy_center_correction']['boxes'])


if __name__ == '__main__':
    main()
