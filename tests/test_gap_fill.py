"""Regression checks for missed-frame export, suppression and cached-run reuse."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'DetZero-main-2'))
sys.path.insert(0, str(ROOT))
from integration import bridge
from integration.gap_fill import export_gap_filled


def record(token, x, name='car', score=.8):
    return dict(sample_token=token, translation=[x, 0., 1.], size=[2., 4., 2.],
                rotation=[1., 0., 0., 0.], velocity=[0., 0.], detection_name=name,
                detection_score=score, attribute_name='')


def fixture():
    original = {'meta': {}, 'results': {f't{i}': [] for i in range(5)}}
    original['results']['t1'] = [record('t1', 10.)]
    original['results']['t3'] = [record('t3', 12.)]
    frames = {str(i): dict(sample_token=f't{i}', timestamp=i*.2, pose=np.eye(4),
                          ego_translation=np.zeros(3)) for i in range(5)}
    prepared = {'original': original, 'frames': {'s': frames}}
    track = dict(nusc_name='car', sequence_name='s', sample_idx=np.arange(5).astype(str),
                 source_index=np.array([-1, 0, -1, 0, -1]), timestamp=np.arange(5)*.2,
                 boxes_global=np.array([[9+i, 0, 1, 4, 2, 2, 0] for i in range(5)], dtype=float))
    track['boxes_global'][2, 0] = 11.2  # Kalman position differs from interpolation.
    tracks = {'id': track}
    refined = {'id': track['boxes_global'].copy()}
    detection, _ = bridge.export_detection(prepared, tracks, refined)
    tracking = {'meta': {}, 'results': {f't{i}': [] for i in range(5)}}
    for i in [1, 3]:
        r = detection['results'][f't{i}'][0]
        tracking['results'][f't{i}'] = [dict(
            **{k: r[k] for k in ['sample_token', 'translation', 'size', 'rotation', 'velocity']},
            tracking_id='id', tracking_name='car', tracking_score=.8)]
    return prepared, tracks, refined, detection, tracking


class GapFillTests(unittest.TestCase):
    def test_tracker_fills_internal_gap_only_and_keeps_originals(self):
        args = fixture()
        original = copy.deepcopy(args[3])
        detection, tracking, report = export_gap_filled(*args)
        self.assertEqual(report['counts']['added_boxes'], 1)
        box = detection['results']['t2'][0]
        self.assertAlmostEqual(box['translation'][0], 11.2)
        self.assertAlmostEqual(box['detection_score'], .64)
        self.assertEqual(tracking['results']['t2'][0]['tracking_id'], 'id')
        self.assertEqual(tracking['results']['t2'][0]['translation'], box['translation'])
        self.assertEqual(detection['results']['t0'], [])
        self.assertEqual(detection['results']['t4'], [])
        self.assertEqual(detection['results']['t1'], original['results']['t1'])
        self.assertEqual(args[3], original)

    def test_irregular_time_interpolation_and_yaw_wrap(self):
        args = fixture()
        args[0]['frames']['s']['2']['timestamp'] = .3
        args[1]['id']['timestamp'][2] = .3
        args[2]['id'][1, 6] = np.deg2rad(179)
        args[2]['id'][3, 6] = np.deg2rad(-179)
        detection, _, _ = export_gap_filled(*args, method='interpolate')
        box = bridge.nusc_box(detection['results']['t2'][0])
        self.assertAlmostEqual(box[0], 10.5)
        self.assertAlmostEqual(abs(np.rad2deg(box[6])), 179.5)

    def test_missing_track_entry_requires_interpolation_mode(self):
        args = fixture()
        for key in ['sample_idx', 'source_index', 'timestamp', 'boxes_global']:
            args[1]['id'][key] = args[1]['id'][key][[1, 3]]
        args[2]['id'] = args[2]['id'][[1, 3]]
        detection, _, report = export_gap_filled(*args)
        self.assertEqual(report['counts']['no_tracker_prediction'], 1)
        self.assertEqual(detection['results']['t2'], [])
        detection, _, report = export_gap_filled(*args, method='interpolate')
        self.assertEqual(report['counts']['added_boxes'], 1)
        self.assertEqual(detection['results']['t2'][0]['translation'][0], 11.)

    def test_long_gap_low_confidence_and_outside_range_are_skipped(self):
        for options, expected in [({'max_gap_seconds': .3}, 'gap_too_long'),
                                  ({'min_score': .7}, 'below_min_score')]:
            _, _, report = export_gap_filled(*fixture(), **options)
            self.assertEqual(report['counts']['added_boxes'], 0)
            self.assertEqual(report['counts'][expected], 1)
        args = fixture()
        args[1]['id']['boxes_global'][2, 0] = 51.
        _, _, report = export_gap_filled(*args)
        self.assertEqual(report['counts']['outside_class_range'], 1)

    def test_low_score_detection_and_car_truck_ambiguity_suppress_duplicates(self):
        for name in ['car', 'truck']:
            args = fixture()
            existing = record('t2', 11.2, name, .01)
            args[0]['original']['results']['t2'] = [existing]
            args[3]['results']['t2'] = [copy.deepcopy(existing)]
            detection, _, report = export_gap_filled(*args)
            self.assertEqual(report['counts']['duplicate_suppressed'], 1)
            self.assertEqual(detection['results']['t2'], [existing])

    def test_bev_overlap_and_duplicate_track_candidates(self):
        args = fixture()
        args[1]['other'] = copy.deepcopy(args[1]['id'])
        args[2]['other'] = copy.deepcopy(args[2]['id'])
        _, tracking, report = export_gap_filled(*args)
        self.assertEqual(report['counts']['added_boxes'], 1)
        self.assertEqual(report['counts']['anchors_not_exported_for_track'], 1)
        self.assertEqual(len(tracking['results']['t2']), 1)
        args = fixture()
        existing = record('t2', 13.)  # >0.5m away but overlapping BEV boxes.
        args[0]['original']['results']['t2'] = [existing]
        args[3]['results']['t2'] = [copy.deepcopy(existing)]
        _, _, report = export_gap_filled(*args)
        self.assertEqual(report['counts']['duplicate_suppressed'], 1)

    def test_frame_limit_preserves_all_existing_detections(self):
        args = fixture()
        existing = [record('t2', 30, 'barrier') for _ in range(500)]
        args[0]['original']['results']['t2'] = existing
        args[3]['results']['t2'] = copy.deepcopy(existing)
        detection, _, report = export_gap_filled(*args)
        self.assertEqual(report['counts']['frame_box_limit'], 1)
        self.assertEqual(detection['results']['t2'], existing)

    def test_timestamp_and_source_mismatch_are_rejected(self):
        args = fixture()
        args[1]['id']['timestamp'][2] += .01
        with self.assertRaises(ValueError):
            export_gap_filled(*args)
        args = fixture()
        args[1]['id']['nusc_name'] = 'pedestrian'
        with self.assertRaises(ValueError):
            export_gap_filled(*args)

    def test_cached_cli_and_evaluator_end_to_end(self):
        from tools.evaluate_vf_comparison import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, run, out = root/'dataset', root/'run', root/'out'
            version = 'v1.0-mini'
            folder = dataset/version
            folder.mkdir(parents=True)
            run.mkdir()
            args = fixture()
            tables = dict(
                sample=[dict(token=f't{i}', scene_token='s', timestamp=int(i*200000)) for i in range(5)],
                scene=[dict(token='s', name='test')],
                sensor=[dict(token='l', channel='LIDAR_TOP')],
                calibrated_sensor=[dict(token='c', sensor_token='l', translation=[0,0,0], rotation=[1,0,0,0])],
                ego_pose=[dict(token='e', translation=[0,0,0], rotation=[1,0,0,0])],
                sample_data=[dict(token=f'sd{i}', sample_token=f't{i}', is_key_frame=True,
                                  filename='cloud.bin', calibrated_sensor_token='c', ego_pose_token='e') for i in range(5)],
                category=[dict(token='cat', name='vehicle.car')],
                instance=[dict(token='obj', category_token='cat')],
                sample_annotation=[dict(token=f'ann{i}', instance_token='obj', sample_token=f't{i}',
                                        translation=[9.+i, 0, 1], size=[2,4,2], rotation=[1,0,0,0], num_lidar_pts=10)
                                   for i in [1,2,3]])
            for name, rows in tables.items():
                (folder/(name+'.json')).write_text(json.dumps(rows))
            np.zeros((1,5), dtype=np.float32).tofile(dataset/'cloud.bin')
            baseline = root/'baseline.json'
            baseline.write_text(json.dumps(args[0]['original']))
            source_report = dict(coordinate_mode='global', input_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                                 arguments=dict(classes='car', min_score=.1, version=version))
            (run/'run_report.json').write_text(json.dumps(source_report))
            (run/'tracks.pkl').write_bytes(pickle.dumps(args[1]))
            (run/'refined_boxes.pkl').write_bytes(pickle.dumps(args[2]))
            (run/'results_nusc_detzero_refined.json').write_text(json.dumps(args[3]))
            (run/'results_nusc_detzero_tracking.json').write_text(json.dumps(args[4]))
            command = [sys.executable, str(ROOT/'tools/export_detzero_gap_filled.py'), '--run-dir', str(run),
                       '--baseline', str(baseline), '--data-root', str(dataset), '--version', version, '--output-dir', str(out)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((out/'gap_fill_report.json').read_text())
            self.assertEqual(report['counts']['added_boxes'], 1)
            before = evaluate(dataset, version, run/'results_nusc_detzero_refined.json', ['car'])
            after = evaluate(dataset, version, out/'results_nusc_detzero_gap_filled.json', ['car'])
            self.assertLess(before['custom_mAP'], after['custom_mAP'])
            self.assertAlmostEqual(after['custom_mAP'], 1.)
            # Re-running refuses to overwrite; a changed baseline is rejected before output is written.
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            baseline.write_text(baseline.read_text() + '\n')
            command[-1] = str(root/'wrong')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertFalse((root/'wrong').exists())


if __name__ == '__main__':
    unittest.main()
