"""nuScenes data contract and pure NumPy geometry, independent of BEVFusion."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

CLASS_MAP = {
    'car': 'Vehicle', 'truck': 'Vehicle', 'bus': 'Vehicle',
    'trailer': 'Vehicle', 'construction_vehicle': 'Vehicle',
    'pedestrian': 'Pedestrian', 'bicycle': 'Cyclist', 'motorcycle': 'Cyclist',
}
MODEL_PREFIX = {'Vehicle': 'vehicle', 'Pedestrian': 'pedestrian', 'Cyclist': 'cyclist'}


def wrap_yaw(yaw):
    return (np.asarray(yaw) + np.pi) % (2 * np.pi) - np.pi


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError('Invalid quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def pose(record):
    out = np.eye(4)
    out[:3, :3] = rotation(record['rotation'])
    out[:3, 3] = record['translation']
    return out


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def nusc_box(record):
    r = rotation(record['rotation'])
    w, length, height = record['size']
    box = np.array([*record['translation'], length, w, height,
                    np.arctan2(r[1, 0], r[0, 0])], dtype=np.float64)
    if not np.isfinite(box).all() or np.any(box[3:6] <= 0):
        raise ValueError('Invalid detector box')
    return box


def replace_box(record, box):
    out = copy.deepcopy(record)
    out['translation'] = np.asarray(box[:3]).tolist()
    out['size'] = np.asarray(box[[4, 3, 5]]).tolist()
    yaw = float(wrap_yaw(box[6]))
    out['rotation'] = [float(np.cos(yaw/2)), 0., 0., float(np.sin(yaw/2))]
    return out


def prepare(results_path, data_root, version, classes, min_score):
    """Read only prediction + pose tables, never annotations or GT matches."""
    root = Path(data_root).resolve()
    def table(name):
        rows = json.loads((root / version / (name + '.json')).read_text())
        return {r['token']: r for r in rows}
    samples, scenes = table('sample'), table('scene')
    calibs, egos, sensors = table('calibrated_sensor'), table('ego_pose'), table('sensor')
    lidar = {}
    for row in table('sample_data').values():
        calib = calibs[row['calibrated_sensor_token']]
        if row['is_key_frame'] and sensors[calib['sensor_token']]['channel'] == 'LIDAR_TOP':
            lidar[row['sample_token']] = row
    original = json.loads(Path(results_path).read_text())
    results = original['results']
    if not results:
        raise ValueError('No sample entries in detector JSON')
    if not set(results) <= set(samples):
        raise ValueError('Detector tokens do not belong to this dataset version')
    selected_scenes = {samples[t]['scene_token'] for t in results}
    # A partial scene changes time gaps and evaluation. Refuse silent truncation.
    expected = {t for t, s in samples.items() if s['scene_token'] in selected_scenes}
    if set(results) != expected:
        raise ValueError('Detector JSON must include every sample (including empty ones) of selected scenes')
    frames = {}
    for scene in sorted(selected_scenes):
        sequence = sorted((samples[t] for t in results if samples[t]['scene_token'] == scene),
                          key=lambda s: s['timestamp'])
        frames[scene] = {}
        for i, sample in enumerate(sequence):
            token = sample['token']; sd = lidar[token]
            path = root / sd['filename']
            if not path.is_file():
                raise FileNotFoundError(path)
            transform = pose(egos[sd['ego_pose_token']]) @ pose(calibs[sd['calibrated_sensor_token']])
            boxes, names, scores, indices = [], [], [], []
            for j, pred in enumerate(results[token]):
                if pred['sample_token'] != token:
                    raise ValueError('Detection sample_token mismatch')
                score = float(pred['detection_score'])
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Invalid detection confidence')
                box = nusc_box(pred)
                if pred['detection_name'] in classes and score >= min_score:
                    boxes.append(box); names.append(pred['detection_name'])
                    scores.append(score); indices.append(j)
            frames[scene][str(i)] = {
                'sequence_name': scene, 'scene_name': scenes[scene]['name'],
                'frame_id': i, 'sample_token': token,
                'timestamp': sample['timestamp'] / 1e6, 'pose': transform,
                'lidar_path': sd['filename'],
                'boxes_global': np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
                'name': np.asarray([CLASS_MAP[n] for n in names], dtype=str),
                'nusc_name': np.asarray(names, dtype=str),
                'score': np.asarray(scores, dtype=np.float32),
                'source_index': np.asarray(indices, dtype=np.int64),
            }
    return {'schema': 1, 'version': version, 'classes': list(classes),
            'min_score': min_score, 'frames': frames, 'original': original,
            'source_sha256': hashlib.sha256(Path(results_path).read_bytes()).hexdigest()}


def crop_points(points_global, box, scale=1.1):
    local = (points_global[:, :3] - box[:3]) @ yaw_matrix(box[6])
    mask = np.all(np.abs(local) <= box[3:6] * scale / 2, axis=1)
    return points_global[mask].copy()


def read_points(path, transform, intensity_mode):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 5:
        raise ValueError('nuScenes LiDAR must have 5 float32 values per point')
    raw = raw.reshape(-1, 5)
    raw = raw[np.isfinite(raw).all(axis=1)]
    xyz = raw[:, :3].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    intensity = raw[:, 3:4].copy()
    if intensity_mode == 'unit':
        intensity = np.clip(intensity / 255., 0., 1.)
    elif intensity_mode == 'tanh':
        intensity = np.tanh(intensity)
    elif intensity_mode != 'raw':
        raise ValueError(intensity_mode)
    return np.column_stack([xyz, intensity]).astype(np.float32)


def sampled(points, count, rng):
    if len(points) >= count:
        return points[np.sort(rng.choice(len(points), count, replace=False))].copy()
    return np.concatenate([points, np.zeros((count-len(points), points.shape[1]), dtype=np.float32)])


def corners(boxes):
    template = np.array([[1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],
                         [1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1]]) / 2
    return np.stack([(template * b[3:6]) @ yaw_matrix(b[6]).T + b[:3] for b in boxes])


def geometry_features(track, rng):
    boxes = np.asarray(track['boxes_global'])[:, :7]
    scores = np.asarray(track['score'])
    query = np.argsort(scores)[::-1][:3]
    query_boxes = boxes[query].copy()
    query_boxes[:, [0, 1, 2, 6]] = 0
    local = [p.copy() for p in track['pts']]
    for i, p in enumerate(local):
        p[:, :3] = (p[:, :3] - boxes[i, :3]) @ yaw_matrix(boxes[i, 6])
    query_points = np.stack([sampled(local[i], 256, rng) for i in query])
    memory = []
    for i, p in enumerate(local):
        memory.append(np.column_stack([p, boxes[i, 3:6]/2-p[:, :3],
                                       boxes[i, 3:6]/2+p[:, :3], np.full(len(p), scores[i])]))
    memory = sampled(np.concatenate(memory), 4096, rng)
    return {'geo_query_points': query_points[None].astype(np.float32),
            'geo_memory_points': memory[None].astype(np.float32),
            'geo_query_boxes': query_boxes[None].astype(np.float32),
            'geo_query_num': [len(query)]}


def position_features(track, rng, with_class=False):
    boxes = np.asarray(track['boxes_global'])[:, :7].copy()
    n = len(boxes)
    if not 1 <= n <= 200:
        raise ValueError('PRM chunks must have 1..200 frames')
    origin = boxes[n//2].copy()
    boxes[:, :3] = (boxes[:, :3]-origin[:3]) @ yaw_matrix(origin[6])
    boxes[:, 6] = wrap_yaw(boxes[:, 6]-origin[6])
    local = [p.copy() for p in track['pts']]
    for p in local:
        p[:, :3] = (p[:, :3]-origin[:3]) @ yaw_matrix(origin[6])
    vertices = np.concatenate([corners(boxes).reshape(n, 24), boxes[:, :3]], axis=1)
    cls_str = str(track['name'][0]) if isinstance(track['name'], (list, np.ndarray)) else str(track['name'])
    cls = np.eye(3)[['Vehicle', 'Pedestrian', 'Cyclist'].index(cls_str)]
    def encode(count):
        pts = np.stack([sampled(p, count, rng) for p in local])
        relative = np.tile(pts[:, :, :3], (1, 1, 9)) - vertices[:, None, :]
        score = np.broadcast_to(np.asarray(track['score'])[:, None, None], (n, count, 1))
        parts = [pts, relative, score]
        if with_class:
            parts.append(np.broadcast_to(cls, (n, count, 3)))
        features = np.concatenate(parts, axis=2).astype(np.float32)
        return np.pad(features, ((0, 200-n),(0,0),(0,0)))[None]
    return {'pos_query_points': encode(256), 'pos_memory_points': encode(48),
            'pos_trajectory': np.pad(boxes, ((0,200-n),(0,0)))[None].astype(np.float32),
            'padding_mask': (np.arange(200) >= n)[None]}, origin


def position_to_global(boxes, origin):
    result = np.asarray(boxes).copy()
    result[:, :3] = result[:, :3] @ yaw_matrix(origin[6]).T + origin[:3]
    result[:, 6] = wrap_yaw(result[:, 6] + origin[6])
    return result


def combine_boxes(geometry, position):
    out = np.asarray(position).copy()
    out[:, 3:6] = np.asarray(geometry)[3:6]
    if not np.isfinite(out).all() or np.any(out[:, 3:6] <= 0):
        raise ValueError('Refiner produced invalid boxes; output not accepted')
    return out


def export_detection(prepared, tracks, refined=None):
    """Replace matched source detections once; preserve scores and unprocessed boxes."""
    output = copy.deepcopy(prepared['original'])
    claimed = set(); replaced = 0
    # Prioritize longest tracks if reverse tracking claimed the same observation.
    for tid, track in sorted(tracks.items(), key=lambda kv: (-len(kv[1]['sample_idx']), kv[0])):
        boxes = refined.get(tid, track['boxes_global']) if refined is not None else track['boxes_global']
        if len(boxes) != len(track['sample_idx']):
            raise ValueError('Refined frame count differs from track')
        for i, frame_id in enumerate(track['sample_idx']):
            frame = prepared['frames'][track['sequence_name']][str(frame_id)]
            token = frame['sample_token']; source = int(track['source_index'][i])
            if source < 0 or (token, source) in claimed:
                continue
            original = output['results'][token][source]
            track_cls = track['name'][0] if isinstance(track['name'], (list, np.ndarray)) else track['name']
            if original['detection_name'] != track.get('nusc_name', None) and CLASS_MAP.get(original['detection_name']) != track_cls:
                raise ValueError('Track class differs from source detection')
            output['results'][token][source] = replace_box(original, np.asarray(boxes[i]))
            claimed.add((token, source)); replaced += 1
    return output, replaced
