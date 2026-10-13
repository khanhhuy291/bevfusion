"""Coordinate and frozen-source contracts for the local inspection viewer."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('vf_inspector_server',Path(__file__).resolve().parents[1]/'apps/vf-inspector/server.py')
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)


def record(x):
    return {'translation':[x,0.,1.], 'rotation':[1.,0.,0.,0.], 'size':[2.,4.,2.],
            'detection_name':'car', 'detection_score':.9}


class InspectorTests(unittest.TestCase):
    def fixture(self):
        d=object.__new__(server.Dataset)
        ego={'translation':[0.,0.,0.],'rotation':[1.,0.,0.,0.]}
        sample={'token':'s','timestamp':0,'scene_token':'scene'}
        sensors={c:{'calibrated_sensor_token':'cal','ego_pose_token':'ego','width':300,'height':200} for c in ['LIDAR_TOP',*server.CHANNELS]}
        d.frame_rows=lambda index:(sample,sensors,ego)
        d.egos={'ego':ego};d.calibs={'cal':{**ego,'camera_intrinsic':[[100,0,150],[0,100,100],[0,0,1]]}}
        d.scenes={'scene':{'name':'synthetic'}};d.start_time=0
        d.baseline={'s':[record(.1),record(10.1)]};d.refined={'s':[record(10.1),record(.1)]}
        d.annotations={'s':[{**record(x),'group':'car','token':'gt'+str(i),'instance_token':'instance'+str(i),'num_lidar_pts':5} for i,x in enumerate([0.,10.])]}
        return d

    def test_refinement_does_not_get_a_more_favorable_gt_match(self):
        frame=self.fixture().frame(0)
        for index,o in enumerate(frame['objects']):
            self.assertEqual(o['source_index'],index)
            self.assertGreater(o['bev_iou'],.9)
            self.assertEqual(o['detzero_iou'],0.)
        self.assertEqual(len(frame['boxes']['gt']),2)

    def test_geometric_center_and_dimension_order(self):
        corners=server.global_corners(record(10.))
        np.testing.assert_allclose(corners.mean(axis=0),[10.,0.,1.])
        np.testing.assert_allclose(np.ptp(corners,axis=0),[4.,2.,2.])
        np.testing.assert_allclose(corners[[2,3,7,6],2],0.)

    def test_camera_pose_at_its_own_timestamp(self):
        d=self.fixture()
        d.egos['camera_ego']={'translation':[1.,0.,0.],'rotation':[1.,0.,0.,0.]}
        sample,sensors,ego=d.frame_rows(0)
        sensors['CAM_FRONT']['ego_pose_token']='camera_ego'
        frame=d.frame(0)
        transform=np.array(frame['cameras']['CAM_FRONT']['from_ego'])
        np.testing.assert_allclose(transform@np.array([3.,0.,1.,1.]),[2.,0.,1.,1.])
        np.testing.assert_allclose(frame['cameras']['CAM_BACK']['from_ego'],np.eye(4))


if __name__=='__main__':unittest.main()
