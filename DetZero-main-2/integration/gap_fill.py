"""Optional offline gap export; no annotations or GT matching are used."""
import copy
from collections import Counter

import numpy as np
from shapely.geometry import Polygon

from . import bridge

TRACKING_CLASSES = {'car', 'truck', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle'}
RANGES = {'car': 50., 'truck': 50., 'bus': 50., 'trailer': 50.,
          'pedestrian': 40., 'motorcycle': 40., 'bicycle': 40.}


def bev_polygon(box):
    xy = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * box[3:5] / 2
    return Polygon(xy @ bridge.yaw_matrix(box[6])[:2, :2].T + box[:2])


def duplicate(box, name, records, iou_threshold, center_distance):
    """Also protect low-score source boxes and car/truck class ambiguity."""
    polygon = bev_polygon(box)
    for record in records:
        other_name = record.get('detection_name', record.get('tracking_name'))
        if name != other_name and {name, other_name} != {'car', 'truck'}:
            continue
        other = bridge.nusc_box(record)
        if np.linalg.norm(box[:2] - other[:2]) <= center_distance:
            return True
        other_polygon = bev_polygon(other)
        intersection = polygon.intersection(other_polygon).area
        union = polygon.area + other_polygon.area - intersection
        if union > 0 and intersection / union >= iou_threshold:
            return True
    return False


def export_gap_filled(prepared, tracks, refined, detection, tracking,
                      method='tracker', max_gap_seconds=0.6, score_decay=0.8,
                      min_score=0.1, iou_threshold=0.1, center_distance=0.5):
    """Append bounded missing observations to BOTH exports, preserving existing boxes.

    tracker: original Kalman position/yaw, interpolated refined dimensions.
    interpolate: time-interpolate refined anchor boxes, wrapping yaw on the short arc.
    Only gaps between observed detections in one retained track are eligible.
    max_gap_seconds is the FULL time between the two observed anchors.
    """
    if method not in {'tracker', 'interpolate'}:
        raise ValueError('Unknown gap fill method')
    values = [max_gap_seconds, score_decay, min_score, iou_threshold, center_distance]
    if not np.isfinite(values).all() or max_gap_seconds <= 0 or not 0 < score_decay < 1:
        raise ValueError('Gap length must be positive and score decay strictly between 0 and 1')
    if not 0 <= min_score <= 1 or not 0 < iou_threshold <= 1 or center_distance < 0:
        raise ValueError('Invalid gap fill score or duplicate thresholds')
    original = prepared['original']['results']
    if set(detection['results']) != set(original) or set(tracking['results']) != set(original):
        raise ValueError('Gap exports and baseline must cover identical sample tokens')
    out_detection, out_tracking = copy.deepcopy(detection), copy.deepcopy(tracking)
    stats, additions, candidates = Counter(), [], []
    exported_ids = {token: {r['tracking_id'] for r in records} for token, records in tracking['results'].items()}
    for token, records in detection['results'].items():
        if len(records) != len(original[token]) or len(records) > 500:
            raise ValueError('Input must be the observation-only export with at most 500 boxes/frame')
        for source, record in enumerate(records):
            baseline = original[token][source]
            if (record['sample_token'] != token or record['detection_name'] != baseline['detection_name']
                    or record['detection_score'] != baseline['detection_score']):
                raise ValueError('Input export no longer preserves source indices/classes/scores')
    for tid, track in sorted(tracks.items()):
        name = track['nusc_name']
        scene = track['sequence_name']
        frames = prepared['frames'][scene]
        frame_ids = [str(i) for i in track['sample_idx']]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError('Duplicate frame in cached track')
        raw = np.asarray(track['boxes_global'])[:, :7]
        boxes = np.asarray(refined.get(tid, raw))[:, :7]
        sources = np.asarray(track['source_index'], dtype=int)
        if boxes.shape != (len(frame_ids), 7) or len(sources) != len(frame_ids):
            raise ValueError('Cached track/refined frame alignment differs')
        times = np.array([frames[i]['timestamp'] for i in frame_ids])
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Track frame timestamps must increase')
        if 'timestamp' in track and not np.allclose(track['timestamp'], times, rtol=0, atol=1e-5):
            raise ValueError('Cached track belongs to different dataset timestamps')
        if 'pose' in track and not np.allclose(track['pose'], [frames[i]['pose'] for i in frame_ids],
                                               rtol=0, atol=1e-3):
            raise ValueError('Cached track belongs to different dataset poses')
        observed = np.flatnonzero(sources >= 0)
        for i in observed:
            token = frames[frame_ids[i]]['sample_token']
            if sources[i] >= len(original[token]) or original[token][sources[i]]['detection_name'] != name:
                raise ValueError('Cached track source does not match baseline class/index')
        if name not in TRACKING_CLASSES or len(observed) < 2:
            continue
        scene_frames = sorted(frames, key=lambda k: frames[k]['timestamp'])
        scene_times = np.array([frames[k]['timestamp'] for k in scene_frames])
        if not np.isfinite(scene_times).all() or np.any(np.diff(scene_times) <= 0):
            raise ValueError('Scene timestamps must increase')
        step = float(np.median(np.diff(scene_times)))
        lookup = {frame: i for i, frame in enumerate(frame_ids)}
        for left, right in zip(observed[:-1], observed[1:]):
            start, end = times[left], times[right]
            interior = [k for k in scene_frames if start < frames[k]['timestamp'] < end]
            if not interior:
                continue
            stats['missing_frames_in_observed_gaps'] += len(interior)
            if end - start > max_gap_seconds + 1e-9:
                stats['gap_too_long'] += len(interior)
                continue
            if any(tid not in exported_ids[frames[frame_ids[i]]['sample_token']] for i in [left, right]):
                # A reverse/overlapping track may have lost these observations to another ID at export.
                stats['anchors_not_exported_for_track'] += len(interior)
                continue
            anchors = [original[frames[frame_ids[i]]['sample_token']][sources[i]] for i in [left, right]]
            for frame_id in interior:
                frame = frames[frame_id]
                token, time = frame['sample_token'], frame['timestamp']
                fraction = (time - start) / (end - start)
                score = min(a['detection_score'] for a in anchors) * score_decay ** (
                    min(time - start, end - time) / step)
                if score < min_score:
                    stats['below_min_score'] += 1
                    continue
                i = lookup.get(frame_id)
                if i is not None and sources[i] >= 0:
                    continue
                if method == 'tracker':
                    if i is None:
                        stats['no_tracker_prediction'] += 1
                        continue
                    box = raw[i].copy()
                    box[3:6] = (1 - fraction) * boxes[left, 3:6] + fraction * boxes[right, 3:6]
                else:
                    box = (1 - fraction) * boxes[left] + fraction * boxes[right]
                    box[6] = bridge.wrap_yaw(boxes[left, 6] + fraction * bridge.wrap_yaw(
                        boxes[right, 6] - boxes[left, 6]))
                if not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                    stats['invalid_geometry'] += 1
                    continue
                ego = frame.get('ego_translation', np.asarray(frame['pose'])[:3, 3])
                if np.linalg.norm(box[:2] - np.asarray(ego)[:2]) >= RANGES[name]:
                    stats['outside_class_range'] += 1
                    continue
                template = copy.deepcopy(anchors[0 if fraction <= .5 else 1])
                template['sample_token'] = token
                template['detection_score'] = float(score)
                # Velocity is estimated from the two refined observations, not GT.
                template['velocity'] = ((boxes[right, :2] - boxes[left, :2]) / (end - start)).tolist()
                record = bridge.replace_box(template, box)
                candidates.append((score, len(observed), tid, token, record, box, end - start))
    # Highest-confidence candidates win when reverse/overlapping tracks claim the same region.
    for score, _, tid, token, record, box, gap in sorted(candidates, key=lambda x: (-x[0], -x[1], x[2], x[3])):
        name = record['detection_name']
        if duplicate(box, name, original[token], iou_threshold, center_distance) or duplicate(
                box, name, out_detection['results'][token], iou_threshold, center_distance):
            stats['duplicate_suppressed'] += 1
            continue
        if len(out_detection['results'][token]) >= 500 or len(out_tracking['results'][token]) >= 500:
            stats['frame_box_limit'] += 1
            continue
        out_detection['results'][token].append(record)
        track_record = {k: copy.deepcopy(record[k]) for k in
                        ['sample_token', 'translation', 'size', 'rotation', 'velocity']}
        track_record.update(tracking_id=tid, tracking_name=name, tracking_score=float(score))
        out_tracking['results'][token].append(track_record)
        additions.append({'sample_token': token, 'tracking_id': tid, 'class': name,
                          'detection_index': len(out_detection['results'][token]) - 1,
                          'tracking_index': len(out_tracking['results'][token]) - 1,
                          'score': float(score), 'anchor_span_seconds': float(gap), 'method': method})
        stats['added_boxes'] += 1
        stats['added_' + name] += 1
    stats.setdefault('added_boxes', 0)
    report = {'method': method, 'max_anchor_span_seconds': max_gap_seconds,
              'score_decay': score_decay, 'min_score': min_score,
              'duplicate_bev_iou': iou_threshold, 'duplicate_center_distance_m': center_distance,
              'counts': dict(stats), 'added_frames': len({a['sample_token'] for a in additions}),
              'additions': additions,
              'notes': ['Offline: requires observed anchors before AND after a gap; no extrapolation.',
                        'No GT/annotations used. Existing detections and tracker IDs are preserved.',
                        'Score is a heuristic decay of the smaller anchor confidence; no CRM.',
                        'Tracker mode uses Kalman center/yaw and refined anchor dimensions.',
                        'Interpolation mode interpolates refined anchors in actual time, including wrapped yaw.']}
    return out_detection, out_tracking, report
