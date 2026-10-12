"""Analytic geometry and AP regressions for the custom VF IoU 3D evaluator."""
from pathlib import Path
import sys
import unittest

import numpy as np
from pyquaternion import Quaternion
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.vf_iou3d import evaluate_iou3d, iou3d_matrix, validate_thresholds


def box(center=(0., 0., 0.), size=(2., 2., 2.), yaw=0., score=.9,
        name='car', token='frame'):
    return DetectionBox(sample_token=token, translation=center, size=size,
                        rotation=tuple(Quaternion(axis=[0, 0, 1], radians=yaw)),
                        detection_name=name, detection_score=score)


def evaluate(targets, predictions, thresholds=(.5,)):
    gt, pred = EvalBoxes(), EvalBoxes()
    tokens = sorted({b.sample_token for b in targets + predictions})
    for token in tokens:
        gt.add_boxes(token, [b for b in targets if b.sample_token == token])
        pred.add_boxes(token, [b for b in predictions if b.sample_token == token])
    return evaluate_iou3d(gt, pred, ['car'], thresholds)['per_class']['car']['by_threshold']


class GeometryTests(unittest.TestCase):
    def test_identity_disjoint_height_and_partial_volume(self):
        target = box()
        predictions = [box(), box(center=(3., 0., 0.)), box(center=(0., 0., 1.)),
                       box(center=(1., 0., 1.)), box(center=(0., 0., 2.))]
        np.testing.assert_allclose(iou3d_matrix(predictions, [target])[:, 0],
                                   [1., 0., 1 / 3, 1 / 7, 0.])

    def test_rotated_rectangle(self):
        # Two centered 4x2 rectangles at 90 degrees: intersection=4, union=12.
        first = box(size=(2., 4., 2.))
        second = box(size=(2., 4., 2.), yaw=np.pi / 2)
        self.assertAlmostEqual(iou3d_matrix([first], [second])[0, 0], 1 / 3)

    def test_width_length_order_and_symmetry(self):
        first = box(size=(2., 4., 2.))
        second = box(center=(1., 0., 0.), size=(2., 4., 2.))
        self.assertAlmostEqual(iou3d_matrix([first], [second])[0, 0], .6)
        np.testing.assert_allclose(iou3d_matrix([first], [second]),
                                   iou3d_matrix([second], [first]).T)

    def test_tilt_is_not_silently_flattened(self):
        tilted = box()
        tilted.rotation = tuple(Quaternion(axis=[1, 0, 0], radians=.1))
        with self.assertRaisesRegex(ValueError, 'upright'):
            iou3d_matrix([tilted], [box()])


class APTests(unittest.TestCase):
    def test_perfect_empty_and_missing_gt(self):
        self.assertEqual(evaluate([box()], [box()])['0.5']['AP'], 1.)
        result = evaluate([box()], [])['0.5']
        self.assertEqual((result['AP'], result['TP'], result['FP'], result['FN']), (0., 0, 0, 1))
        result = evaluate([box(), box(center=(10., 0., 0.))], [box()])['0.5']
        self.assertEqual(result['AP'], .5)

    def test_score_order_false_positive_and_duplicate(self):
        predictions = [box(score=.8), box(center=(10., 0., 0.), score=.9), box(score=.7)]
        result = evaluate([box()], predictions)['0.5']
        self.assertEqual((result['AP'], result['TP'], result['FP'], result['FN']), (.5, 1, 2, 0))

    def test_iou_threshold_inclusive_and_geometry_can_improve_ap(self):
        target = box()
        oversized = box(size=(4., 2., 2.))
        before = evaluate([target], [oversized], (.5, .7))
        after = evaluate([target], [box()], (.5, .7))
        self.assertEqual(before['0.5']['AP'], 1.)
        self.assertEqual(before['0.7']['AP'], 0.)
        self.assertEqual(after['0.7']['AP'], 1.)

    def test_best_unmatched_gt_and_frame_class_isolation(self):
        result = evaluate([box(), box(center=(1., 0., 0.))],
                          [box(), box(center=(.1, 0., 0.), score=.8)], (.25,))['0.25']
        self.assertEqual(result['TP'], 2)
        result = evaluate([box()], [box(token='other'), box(name='pedestrian')])['0.5']
        self.assertEqual((result['TP'], result['FP'], result['FN']), (0, 1, 1))

    def test_invalid_thresholds(self):
        for thresholds in [(), (0.,), (1.1,), (.5, .5), (float('nan'),)]:
            with self.assertRaises(ValueError):
                validate_thresholds(thresholds)


if __name__ == '__main__':
    unittest.main()
