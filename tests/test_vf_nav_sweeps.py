"""NAV sensor transforms and actual exported history regression checks."""
import json
from pathlib import Path
import pickle
import runpy
import sys
import unittest
import numpy as np
from pyquaternion import Quaternion
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.data_converter.create_vf_infos import obtain_sensor2top


def pose(yaw,xyz):
    return dict(rotation=Quaternion(axis=[0,0,1],angle=yaw).elements.tolist(),translation=xyz)


def matrix(p):
    t=np.eye(4);t[:3,:3]=Quaternion(p['rotation']).rotation_matrix;t[:3,3]=p['translation'];return t


class FakeNuScenes:
    def __init__(self):
        self.tables={'sample_data':{'s':dict(token='s',calibrated_sensor_token='c',ego_pose_token='e',timestamp=100)},
                     'calibrated_sensor':{'c':pose(.3,[1,2,1.8])},'ego_pose':{'e':pose(-.7,[21,12,3])}}
    def get(self,table,token):return self.tables[table][token]
    def get_sample_data_path(self,token):return '/tmp/sensor.bin'


class NavTests(unittest.TestCase):
    def test_sensor_to_reference_matches_homogeneous_chain(self):
        n=FakeNuScenes();lc=pose(-1.2,[.9,0,1.8]);ego=pose(.8,[25,14,3.1])
        sw=obtain_sensor2top(n,'s',lc['translation'],Quaternion(lc['rotation']).rotation_matrix,
                             ego['translation'],Quaternion(ego['rotation']).rotation_matrix)
        expected=np.linalg.inv(matrix(ego)@matrix(lc))@matrix(n.get('ego_pose','e'))@matrix(n.get('calibrated_sensor','c'))
        np.testing.assert_allclose(sw['sensor2lidar_rotation'],expected[:3,:3],atol=1e-12)
        np.testing.assert_allclose(sw['sensor2lidar_translation'],expected[:3,3],atol=1e-12)

    def test_nav_configs_only_change_history_count(self):
        from torchpack.utils.config import Config
        for n in [0,2,5,9]:
            c=Config();c.load(str(ROOT/f'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/vf6_nav/sweeps{n}.yaml'),recursive=True)
            cfg=runpy.run_path(str(ROOT/'mmdet3d/utils/config.py'))['recursive_eval'](c)
            sw=next(p for p in cfg['data']['test']['pipeline'] if p['type']=='LoadPointsFromMultiSweeps')
            self.assertEqual(sw['sweeps_num'],n);self.assertFalse(sw['pad_empty_sweeps'])
            self.assertTrue(sw['test_mode'])
            self.assertEqual(cfg['model']['heads']['object']['bbox_coder']['score_threshold'],0)
            self.assertEqual(cfg['dataset_root'],'data/nuscenes_vf6_01_5hz_nav_calib/')

    @unittest.skipUnless((ROOT/'data/nuscenes_vf6_01_5hz_nav_calib/bevfusion_infos_val.pkl').exists(),'optional local NAV data')
    def test_real_history_is_past_bounded_and_matches_source_transform(self):
        root=ROOT/'data/nuscenes_vf6_01_5hz_nav_calib';data=pickle.loads((root/'bevfusion_infos_val.pkl').read_bytes())
        self.assertTrue(data['metadata']['ego_motion_available']);self.assertFalse(data['metadata']['velocity_available'])
        self.assertEqual(len(data['infos']),155)
        counts=[]
        for i in data['infos']:
            counts.append(len(i['sweeps']));age=[(i['timestamp']-s['timestamp'])/1e6 for s in i['sweeps']]
            self.assertTrue(all(0<a<=1 for a in age));self.assertTrue(all(a<b for a,b in zip(age,age[1:])))
            self.assertEqual(len({s['sample_data_token'] for s in i['sweeps']}),len(age))
            dst=matrix(dict(rotation=i['ego2global_rotation'],translation=i['ego2global_translation']))@matrix(dict(rotation=i['lidar2ego_rotation'],translation=i['lidar2ego_translation']))
            for s in i['sweeps']:
                self.assertTrue(Path(s['data_path']).is_file())
                src=matrix(dict(rotation=s['ego2global_rotation'],translation=s['ego2global_translation']))@matrix(dict(rotation=s['sensor2ego_rotation'],translation=s['sensor2ego_translation']))
                expected=np.linalg.inv(dst)@src
                np.testing.assert_allclose(s['sensor2lidar_rotation'],expected[:3,:3],atol=1e-10)
                np.testing.assert_allclose(s['sensor2lidar_translation'],expected[:3,3],atol=1e-10)
        self.assertEqual(sum(n==9 for n in counts),147)

if __name__=='__main__':unittest.main()
