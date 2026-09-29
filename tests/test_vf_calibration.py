"""CPU checks for image rays, sensor coordinates and ablation config contracts."""
import ast
import copy
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest

import cv2
import numpy as np
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.adapt_vf_calibration import camera_homography, transform_points, rotation, model_affine, adapt
from tools.diagnose_vf_pedestrian import project


def calib(q=None, t=None):
    return dict(rotation=q if q is not None else [1,0,0,0], translation=t if t is not None else [0,0,0],
                camera_intrinsic=[[20,0,16],[0,20,12],[0,0,1]])


class CalibrationTests(unittest.TestCase):
    def test_camera_rays_match_updated_extrinsics(self):
        src=calib(t=[1,2,3]);dst=calib(Quaternion(axis=[0,1,0], angle=.2).elements.tolist(), [1,2,3])
        dst['camera_intrinsic']=[[25,0,15],[0,27,10],[0,0,1]]
        points=np.array([[1,2,10],[2,3,12],[-1,1,8]], dtype=float)
        uv,_=project(points,src);expected,_=project(points,dst)
        h=camera_homography(src,dst)
        hp=np.c_[uv,np.ones(len(uv))] @ h.T
        np.testing.assert_allclose(hp[:,:2]/hp[:,2:],expected,atol=1e-10)
        dst['translation'][0]+=1
        with self.assertRaises(ValueError):camera_homography(src,dst)

    def test_lidar_roundtrip_preserves_ego_points_and_extra_channels(self):
        src=calib();dst=calib(Quaternion(axis=[1,2,3],angle=1.4).elements.tolist(),[.985793,0,1.84019])
        pts=np.random.RandomState(42).normal(size=(200,5)).astype(np.float32)
        changed=transform_points(pts,src,dst)
        np.testing.assert_allclose(changed[:,:3] @ rotation(dst['rotation']).T+dst['translation'],pts[:,:3],atol=3e-7)
        np.testing.assert_array_equal(changed[:,3:],pts[:,3:])
        np.testing.assert_allclose(transform_points(changed,dst,src),pts,atol=3e-7)

    def test_crop_diagnostic_matches_real_pipeline(self):
        tree=ast.parse((ROOT/'mmdet3d/datasets/pipelines/transforms_3d.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ImageAug3D')
        cls.decorator_list=[]
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('__init__', 'sample_augmentation')]
        space={'np':np};exec(compile(ast.Module(body=[cls],type_ignores=[]),'ImageAug3D','exec'),space)
        aug=space['ImageAug3D']((256,704),(.48,.48),(0,0),(0,0),False,False)
        for w,h in [(1920,1536),(1600,900)]:
            scale,_,crop,_,_=aug.sample_augmentation({'ori_shape':(w,h)})
            np.testing.assert_allclose(model_affine(w,h),[[scale,0,-crop[0]],[0,scale,-crop[1]],[0,0,1]])

    def test_all_modes_preserve_labels_and_physical_camera_center(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';reference=root/'reference'
            sensors=[dict(token='cam',channel='CAM_FRONT'),dict(token='lidar',channel='LIDAR_TOP')]
            cam=dict(calib(t=[1,0,1.7]),token='cc',sensor_token='cam')
            lidar=dict(calib(),token='lc',sensor_token='lidar')
            rows=[dict(token='im',is_key_frame=True,sample_token='s',calibrated_sensor_token='cc',filename='samples/cam.jpg',width=32,height=24),
                  dict(token='pc',is_key_frame=True,sample_token='s',calibrated_sensor_token='lc',filename='samples/pc.bin',width=0,height=0)]
            tables=dict(sensor=sensors,calibrated_sensor=[cam,lidar],sample_data=rows,ego_pose=[calib()],sample=[dict(token='s')],
                        scene=[dict(name='scene-0103',first_sample_token='s')],sample_annotation=[dict(token='gt',translation=[2,3,1])])
            target=copy.deepcopy(tables);target['calibrated_sensor'][0]['rotation']=Quaternion(axis=[0,1,0],angle=.1).elements.tolist()
            target['calibrated_sensor'][0]['translation']=[5,0,3]
            target['calibrated_sensor'][1]['translation']=[1,0,1.8]
            for folder,version,data in [(source,'v1.0-trainval',tables),(reference,'v1.0-mini',target)]:
                (folder/version).mkdir(parents=True)
                for name,records in data.items():(folder/version/(name+'.json')).write_text(json.dumps(records))
            (source/'samples').mkdir();(source/'maps').mkdir()
            cv2.imwrite(str(source/'samples/cam.jpg'),np.full((24,32,3),120,np.uint8))
            pts=np.array([[3,4,1,255,0]],np.float32);pts.tofile(source/'samples/pc.bin')
            for mode in ['camera','lidar','both']:
                out=root/mode;adapt(source,out,reference,mode)
                actual=json.loads((out/'v1.0-trainval/calibrated_sensor.json').read_text())
                self.assertEqual(actual[0]['translation'],cam['translation'])
                self.assertEqual(json.loads((out/'v1.0-trainval/sample_annotation.json').read_text()),tables['sample_annotation'])
                changed=np.fromfile(out/'samples/pc.bin',dtype=np.float32).reshape(-1,5)
                np.testing.assert_allclose(changed[:,:3] @ rotation(actual[1]['rotation']).T+actual[1]['translation'],pts[:,:3],atol=1e-6)
                self.assertTrue((out/'maps').is_dir())
                self.assertEqual((out/'samples/cam.jpg').is_symlink(),mode=='lidar')
                self.assertEqual((out/'samples/pc.bin').is_symlink(),mode=='camera')
                with self.assertRaises(FileExistsError):adapt(source,out,reference,mode)

    def test_configs_keep_pretrained_model_and_single_frame_input(self):
        from torchpack.utils.config import Config
        base=ROOT/'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/vf6_calibration'
        for mode,suffix in [('camera','camera'),('lidar','lidar'),('both','nusc_calib')]:
            configs=Config();configs.load(str(base/(mode+'.yaml')),recursive=True)
            cfg=runpy.run_path(str(ROOT/'mmdet3d/utils/config.py'))['recursive_eval'](configs)
            self.assertEqual(cfg['dataset_root'],'data/nuscenes_vf6_01_5hz_'+suffix+'/')
            self.assertEqual(len(cfg['object_classes']),10)
            self.assertEqual(set(cfg['model']['encoders']),{'camera','lidar'})
            self.assertEqual(cfg['data']['test']['ann_file'],cfg['dataset_root']+'bevfusion_infos_val.pkl')
            sweep=next(p for p in cfg['data']['test']['pipeline'] if p['type']=='LoadPointsFromMultiSweeps')
            self.assertEqual(sweep['sweeps_num'],0)
            self.assertFalse(sweep['pad_empty_sweeps'])

if __name__=='__main__':unittest.main()
