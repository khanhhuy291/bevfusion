import os
import pickle
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type='lidar'):
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(Path(nusc.get_sample_data_path(sd_rec['token'])).resolve())

    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T
    sweep['sensor2lidar_translation'] = T
    return sweep

def create_vf_infos(root_path="data/nuscenes_vf6_01_5hz", version="v1.0-trainval", max_sweeps=0,
                    output_name="bevfusion_infos_val.pkl", max_sweep_age=1.0):
    if max_sweeps < 0 or max_sweep_age <= 0:
        raise ValueError('max_sweeps must be nonnegative and max_sweep_age positive')
    root_path = str(Path(root_path).resolve())
    print(f"Creating infos for {root_path} ({version})...")
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    pose_path = Path(root_path) / 'pose_report.json'
    pose_report = json.loads(pose_path.read_text()) if pose_path.exists() else {}
    conv_meta_path = Path(root_path) / 'conversion_meta.json'
    conv_meta = json.loads(conv_meta_path.read_text()) if conv_meta_path.exists() else {}
    moving = any(not np.allclose(p['translation'], 0) or
                 not np.allclose(Quaternion(p['rotation']).rotation_matrix, np.eye(3))
                 for p in nusc.ego_pose)
    motion_available = (bool(pose_report.get('available')) or bool(conv_meta.get('global_coord_mode'))) and moving
    if max_sweeps and not motion_available:
        raise ValueError('Sweeps require measured poses and pose_report.json or conversion_meta.json available')

    val_nusc_infos = []
    token2idx = {}

    for sample in tqdm(sorted(nusc.sample, key=lambda s: s['timestamp'])):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        lidar_path = str(Path(lidar_path).resolve())

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'radars': None,
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'prev_token': sample['prev']
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        previous = sd_rec['prev']
        seen = {lidar_token}
        last_timestamp = sd_rec['timestamp']
        while previous and len(info['sweeps']) < max_sweeps:
            if previous in seen:
                raise ValueError('Cycle in LiDAR sample_data history')
            seen.add(previous)
            sweep_sd = nusc.get('sample_data', previous)
            if sweep_sd['timestamp'] >= last_timestamp:
                raise ValueError('Sweep history must be strictly in the past')
            last_timestamp = sweep_sd['timestamp']
            if nusc.get('sample', sweep_sd['sample_token'])['scene_token'] != sample['scene_token']:
                raise ValueError('Sweep crosses scene boundary')
            sweep_cs = nusc.get('calibrated_sensor', sweep_sd['calibrated_sensor_token'])
            if nusc.get('sensor', sweep_cs['sensor_token'])['channel'] != 'LIDAR_TOP':
                raise ValueError('Non-LiDAR record in sweep history')
            if (sd_rec['timestamp'] - sweep_sd['timestamp']) / 1e6 > max_sweep_age:
                break
            sweep = obtain_sensor2top(nusc, previous, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat)
            if not Path(sweep['data_path']).is_file():
                raise FileNotFoundError(sweep['data_path'])
            info['sweeps'].append(sweep)
            previous = sweep_sd['prev']

        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            if cam not in sample['data']:
                raise ValueError(f"Missing {cam} for {sample['token']}")
            cam_token = sample['data'][cam]
            _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'][cam] = cam_info

        # Annotations (Ground Truth)
        annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        velocity = np.full((len(boxes), 2), np.nan, dtype=np.float32)
        valid_flag = np.array([anno.get('num_lidar_pts', 0) > 0 for anno in annotations], dtype=bool).reshape(-1)

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NameMapping:
                names[i] = NameMapping[names[i]]
        names = np.array(names)

        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1) if len(boxes) > 0 else np.zeros((0, 7))
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_velocity'] = velocity
        info['num_lidar_pts'] = np.array([a.get('num_lidar_pts', 0) for a in annotations])
        info['num_radar_pts'] = np.zeros(len(annotations), dtype=int)
        info['valid_flag'] = valid_flag

        val_nusc_infos.append(info)
        token2idx[info['token']] = ('val', len(val_nusc_infos) - 1)

    for info in val_nusc_infos:
        prev_token = info['prev_token']
        if prev_token == '':
            info['prev'] = -1
        else:
            prev_set, prev_idx = token2idx.get(prev_token, ('val', -1))
            info['prev'] = prev_idx

    metadata = {'version': version, 'split': 'custom_val',
                'ego_motion_available': motion_available, 'velocity_available': False,
                'max_sweeps': max_sweeps, 'max_sweep_age_seconds': max_sweep_age,
                'pose_source': pose_report.get('source', conv_meta.get('global_coord_mode', 'unknown')),
                'pose_limitations': pose_report.get('limitations', []),
                'box_convention': 'MIT legacy: center xyz, wlh, -yaw-pi/2'}
    out_val = {'infos': val_nusc_infos, 'metadata': metadata}

    val_path = os.path.join(root_path, output_name)

    with open(val_path, "wb") as f:
        pickle.dump(out_val, f, protocol=4)

    print(f"Exported {len(val_nusc_infos)} custom validation samples to {val_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", default="data/nuscenes_vf6_01_5hz")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--max-sweeps", type=int, default=0)
    parser.add_argument("--output-name", default="bevfusion_infos_val.pkl")
    parser.add_argument("--max-sweep-age", type=float, default=1.0)
    args = parser.parse_args()
    create_vf_infos(args.root_path, args.version, args.max_sweeps, args.output_name, args.max_sweep_age)
