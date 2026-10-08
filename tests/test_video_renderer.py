"""Checks for real tracker IDs and matching full-scene video inputs."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import render_vf_videos as renderer


class VideoRendererTests(unittest.TestCase):
    def boxes(self):
        return [dict(sample_token='a', tracking_id=f's_{name}_1', tracking_name=name,
                     tracking_score=.8, translation=[0., 0., 10.], size=[2., 2., 2.],
                     rotation=[1., 0., 0., 0.]) for name in ['car', 'motorcycle']]

    def test_display_ids_are_distinct_across_classes_and_stable(self):
        boxes = self.boxes()
        labels = renderer.make_track_labels({'results': {'a': boxes, 'b': list(reversed(boxes))}})
        self.assertEqual(set(labels.values()), {'T001', 'T002'})
        self.assertEqual(labels, renderer.make_track_labels({'results': {'a': list(reversed(boxes))}}))
        boxes[0]['tracking_id'] = ''
        with self.assertRaises(ValueError):
            renderer.make_track_labels({'results': {'a': boxes}})

    def test_only_predicted_scenes_and_complete_matching_tokens(self):
        nusc = SimpleNamespace(sample=[{'token': 'a', 'scene_token': 's'},
                                       {'token': 'b', 'scene_token': 's'},
                                       {'token': 'c', 'scene_token': 'other'}],
                               scene=[{'token': 's'}, {'token': 'other'}])
        document = {'results': {'a': [], 'b': []}}
        self.assertEqual(renderer.select_scenes(nusc, [document, copy.deepcopy(document)]), [{'token': 's'}])
        with self.assertRaises(ValueError):
            renderer.select_scenes(nusc, [document, {'results': {'a': []}}])
        with self.assertRaises(ValueError):
            renderer.select_scenes(nusc, [{'results': {'a': []}}])
        with self.assertRaises(ValueError):
            renderer.select_scenes(nusc, [dict(document, coordinate_mode='ego_uncompensated_diagnostic')])

    def test_camera_and_bev_use_same_unambiguous_labels(self):
        boxes = self.boxes()
        labels = renderer.make_track_labels({'results': {'a': boxes}})
        pose = {'translation': [0., 0., 0.], 'rotation': [1., 0., 0., 0.]}
        camera = dict(pose, camera_intrinsic=[[100, 0, 100], [0, 100, 100], [0, 0, 1]])
        with patch.object(renderer.cv2, 'putText', wraps=cv2.putText) as draw:
            image = renderer.render_camera_view(np.zeros((200, 200, 3), dtype=np.uint8), boxes,
                                                camera, pose, pose, pose, is_tracking=True,
                                                track_labels=labels)
            self.assertEqual(image.shape, (270, 480, 3))
            strings = [call.args[1] for call in draw.call_args_list]
            self.assertIn('T001', strings)
            self.assertIn('T002', strings)
            camera_colors = {call.args[1]: call.args[5] for call in draw.call_args_list
                             if call.args[1] in labels.values()}
            self.assertNotEqual(camera_colors['T001'], camera_colors['T002'])
        with tempfile.TemporaryDirectory() as d:
            lidar = Path(d) / 'cloud.bin'
            np.zeros((2, 5), dtype=np.float32).tofile(lidar)
            with patch.object(renderer.cv2, 'putText', wraps=cv2.putText) as draw:
                renderer.render_lidar_bev(str(lidar), boxes, pose, pose,
                                          is_tracking=True, track_labels=labels)
                strings = [call.args[1] for call in draw.call_args_list]
                self.assertIn('T001', strings)
                self.assertIn('T002', strings)
                bev_colors = {call.args[1]: call.args[5] for call in draw.call_args_list
                              if call.args[1] in labels.values()}
                self.assertEqual(camera_colors, bev_colors)
            with self.assertRaises(FileNotFoundError):
                renderer.render_lidar_bev(str(Path(d) / 'missing.bin'), [], pose, pose)


if __name__ == '__main__':
    unittest.main()
