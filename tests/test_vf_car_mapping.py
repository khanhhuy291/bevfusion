"""Check the VF Car protocol using a real SDK evaluation on synthetic boxes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_vf_comparison import evaluate


class VFCarMappingTests(unittest.TestCase):
    def test_truck_matches_car_only_when_mapping_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            version = root / 'v1.0-mini'
            version.mkdir()
            tables = {
                'sample': [{'token': 'frame'}],
                'ego_pose': [{'token': 'pose', 'translation': [0, 0, 0]}],
                'calibrated_sensor': [{'token': 'calib', 'sensor_token': 'lidar'}],
                'sensor': [{'token': 'lidar', 'channel': 'LIDAR_TOP'}],
                'sample_data': [{'token': 'sd', 'sample_token': 'frame', 'is_key_frame': True,
                                 'calibrated_sensor_token': 'calib', 'ego_pose_token': 'pose'}],
                'category': [{'token': 'category', 'name': 'vehicle.car'}],
                'instance': [{'token': 'instance', 'category_token': 'category'}],
                'sample_annotation': [{'token': 'gt', 'instance_token': 'instance',
                                       'sample_token': 'frame', 'num_lidar_pts': 10,
                                       'translation': [5, 0, 0], 'size': [2, 4, 2],
                                       'rotation': [1, 0, 0, 0]}],
            }
            for name, rows in tables.items():
                (version / f'{name}.json').write_text(json.dumps(rows))
            prediction = root / 'prediction.json'
            original = json.dumps({'results': {'frame': [{
                'sample_token': 'frame', 'translation': [5, 0, 0], 'size': [2, 4, 2],
                'rotation': [1, 0, 0, 0], 'detection_name': 'truck', 'detection_score': 0.9}]}})
            prediction.write_text(original)
            plain = evaluate(root, 'v1.0-mini', prediction, ['car'])
            merged = evaluate(root, 'v1.0-mini', prediction, ['car'], car_includes_truck=True)
            self.assertEqual(plain['custom_mAP'], 0)
            self.assertGreater(merged['custom_mAP'], 0.99)
            self.assertEqual(merged['remapped_prediction_classes'], {'truck_to_car': 1})
            self.assertEqual(merged['per_class']['car']['scale_err'], 0)
            self.assertEqual(prediction.read_text(), original)


if __name__ == '__main__':
    unittest.main()
