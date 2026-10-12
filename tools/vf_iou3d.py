"""CPU IoU and detection AP for upright, yaw-rotated, center-origin boxes.

This is a custom VF protocol, not the official Waymo evaluator. Inputs use
nuScenes size order (width, length, height) and geometric centers in meters.
"""
import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Polygon


def validate_thresholds(thresholds):
    values = tuple(float(value) for value in thresholds)
    if (not values or len(set(values)) != len(values)
            or any(not np.isfinite(value) or not 0 < value <= 1 for value in values)):
        raise ValueError('IoU thresholds must be distinct finite values in (0, 1]')
    return values


def _geometry(box):
    center = np.asarray(box.translation, dtype=np.float64)
    width, length, height = np.asarray(box.size, dtype=np.float64)
    rotation = Quaternion(box.rotation).rotation_matrix
    # Match the usual autonomous-driving box convention: vertical height axis,
    # yaw about global Z. Fail explicitly rather than silently flattening tilt.
    if not np.allclose(rotation[:, 2], [0., 0., 1.], atol=1e-4):
        raise ValueError('VF IoU 3D requires upright boxes (yaw only); found roll/pitch')
    corners = np.array([[length, width], [length, -width],
                        [-length, -width], [-length, width]]) / 2
    polygon = Polygon(corners @ rotation[:2, :2].T + center[:2])
    return polygon, center[2] - height / 2, center[2] + height / 2, width * length * height


def iou3d_matrix(predictions, ground_truth):
    """Pairwise rotated 3D IoU; intersection area * height, not BEV IoU * height."""
    result = np.zeros((len(predictions), len(ground_truth)), dtype=np.float64)
    pred_geometry = [_geometry(box) for box in predictions]
    gt_geometry = [_geometry(box) for box in ground_truth]
    for i, (polygon, bottom, top, volume) in enumerate(pred_geometry):
        for j, (gt_polygon, gt_bottom, gt_top, gt_volume) in enumerate(gt_geometry):
            overlap_h = max(0., min(top, gt_top) - max(bottom, gt_bottom))
            if overlap_h == 0:
                continue
            intersection = polygon.intersection(gt_polygon).area * overlap_h
            union = volume + gt_volume - intersection
            result[i, j] = np.clip(intersection / union, 0., 1.)
    return result


def _average_precision(tp, fp, gt_count):
    """All-point interpolated precision-envelope area over recall [0, 1]."""
    recall = np.cumsum(tp, dtype=np.float64) / gt_count
    precision = np.cumsum(tp, dtype=np.float64) / np.maximum(np.cumsum(tp + fp), 1)
    recall = np.r_[0., recall, 1.]
    precision = np.r_[0., precision, 0.]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def evaluate_iou3d(gt, pred, classes, thresholds=(.25, .5, .7)):
    """Score-order, same-frame/class greedy matching, highest unmatched IoU.

    Each GT matches at most once. Equality at the IoU threshold is accepted.
    Equal scores are ordered by sorted sample token then input prediction index.
    No confidence cutoff is applied: AP uses every already-selected candidate.
    """
    thresholds = validate_thresholds(thresholds)
    per_class = {}
    for name in classes:
        frames, candidates, gt_count = {}, [], 0
        for token in sorted(gt.sample_tokens):
            targets = [box for box in gt[token] if box.detection_name == name]
            detections = [box for box in pred[token] if box.detection_name == name]
            frames[token] = iou3d_matrix(detections, targets)
            gt_count += len(targets)
            candidates.extend((box.detection_score, token, i) for i, box in enumerate(detections))
        if not gt_count:
            raise ValueError('No evaluable GT for ' + name)
        candidates.sort(key=lambda item: -item[0])
        by_threshold = {}
        for threshold in thresholds:
            matched = {token: np.zeros(matrix.shape[1], dtype=bool) for token, matrix in frames.items()}
            tp = np.zeros(len(candidates), dtype=np.int64)
            for index, (_, token, pred_index) in enumerate(candidates):
                available = np.flatnonzero(~matched[token])
                if not len(available):
                    continue
                overlap = frames[token][pred_index, available]
                best = int(np.argmax(overlap))
                if overlap[best] >= threshold:
                    matched[token][available[best]] = True
                    tp[index] = 1
            fp = 1 - tp
            true_count = int(tp.sum())
            by_threshold[str(threshold)] = {
                'AP': _average_precision(tp, fp, gt_count),
                'TP': true_count, 'FP': int(fp.sum()), 'FN': gt_count - true_count,
                'precision': true_count / max(len(candidates), 1),
                'recall': true_count / gt_count,
            }
        per_class[name] = {'gt_boxes': gt_count, 'predicted_boxes': len(candidates),
                           'by_threshold': by_threshold}
    mean_ap = {str(threshold): float(np.mean([
        values['by_threshold'][str(threshold)]['AP'] for values in per_class.values()
    ])) for threshold in thresholds}
    return {'protocol': 'VF custom upright rotated IoU 3D; not official Waymo/nuScenes',
            'AP_method': 'All-point interpolated precision envelope over recall [0, 1]; no minimum recall/precision cutoff',
            'matching': 'Descending confidence, best unmatched same-class GT per frame, IoU >= threshold; stable ties',
            'box_convention': 'Global geometric center; size [width,length,height]; yaw only',
            'confidence_cutoff': None,
            'count_policy': 'TP/FP/FN and precision/recall use all candidates after shared class/range filters',
            'thresholds': list(thresholds), 'per_class': per_class,
            'mAP_by_threshold': mean_ap, 'mean_mAP': float(np.mean(list(mean_ap.values())))}
