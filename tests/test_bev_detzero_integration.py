"""CPU regression checks for coordinate, class, refinement and evaluation contracts."""
import copy
import importlib.util
import json
import runpy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'DetZero-main-2'))
sys.path.insert(0, str(ROOT))
from integration import bridge as b
from integration import run_pipeline as r
from integration.gap_fill import export_gap_filled
from tools.evaluate_vf_comparison import evaluate

CFG = str(r.DETZERO_ROOT / 'tracking/tools/cfgs/tk_model_cfgs/nuscenes_detzero_track.yaml')


def record(name='car', token='t', x=10.):
    return dict(sample_token=token, translation=[x, 0., 1.], size=[2., 4., 2.], rotation=[1., 0., 0., 0.],
                velocity=[0., 0.], detection_name=name, detection_score=.8, attribute_name='')


def track(name='Vehicle', n=3):
    return {'name': np.array([name]*n), 'nusc_name': {'Vehicle':'car','Cyclist':'motorcycle','Pedestrian':'pedestrian'}[name],
            'boxes_global': np.array([[10.+i*.1, 0, 1, 4, 2, 2, .2, 0, 0] for i in range(n)], dtype=np.float32),
            'score': np.full(n, .8), 'sample_idx': np.arange(n).astype(str), 'source_index': np.zeros(n, dtype=int),
            'sequence_name': 's', 'timestamp': np.arange(n)*.2,
            'pts': [np.tile([10.+i*.1, 0, 1, .5], (12, 1)).astype(np.float32) for i in range(n)]}


class GeometryModel:
    def __init__(self): self.calls = 0
    def __call__(self, batch):
        self.calls += 1
        return {'pred_boxes': np.array([[0, 0, 0, 4.2, 2.1, 2.1, 0]], dtype=np.float32)}, {}, {}


class IdentityPositionModel:
    def __call__(self, batch):
        return {'pred_boxes': batch['pos_trajectory'].cpu().numpy()}, {}, {}


class IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torchpack'), 'torchpack belongs to the BEVFusion environment')
    def test_vf_config_keeps_checkpoint_layout_and_single_frame_input(self):
        from torchpack.utils.config import configs
        configs.load(str(ROOT / 'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser_vf6_01_5hz.yaml'), recursive=True)
        cfg = runpy.run_path(str(ROOT / 'mmdet3d/utils/config.py'))['recursive_eval'](configs)
        test = cfg['data']['test']
        self.assertEqual(test['load_interval'], 1)
        self.assertTrue(test['ann_file'].endswith('nuscenes_vf6_01_5hz/bevfusion_infos_val.pkl'))
        self.assertEqual(len(cfg['object_classes']), 10)
        self.assertEqual(set(cfg['model']['encoders']), {'camera', 'lidar'})
        sweep = next(p for p in test['pipeline'] if p['type'] == 'LoadPointsFromMultiSweeps')
        self.assertEqual(sweep['sweeps_num'], 0)
        self.assertFalse(sweep['pad_empty_sweeps'])
        self.assertEqual(test['pipeline'][-1]['keys'], ['img', 'points'])
        self.assertFalse({'LoadRadarPointsMultiSweeps','LoadBEVSegmentation','GTDepth','LoadAnnotations3D'} &
                         {p['type'] for p in test['pipeline']})

    def test_box_roundtrip_and_crop(self):
        a = record(); a['rotation'] = [np.cos(.4), 0, 0, np.sin(.4)]
        box = b.nusc_box(a); out = b.replace_box(a, box)
        np.testing.assert_allclose(out['translation'], a['translation'])
        np.testing.assert_allclose(out['size'], a['size'])
        np.testing.assert_allclose(out['rotation'], a['rotation'])
        pts = np.array([[10, 0, 1, .5], [100, 0, 1, .5]])
        self.assertEqual(len(b.crop_points(pts, box)), 1)

    def test_coordinate_guard(self):
        with self.assertRaises(ValueError): r.check_coordinates({'identity_ego_pose': True}, 'global', 'full')
        with self.assertRaises(ValueError): r.check_coordinates({'identity_ego_pose': True}, 'ego', 'full')
        r.check_coordinates({'identity_ego_pose': True}, 'ego', 'geometry')
        r.check_coordinates({'identity_ego_pose': True}, 'global', 'full', True)

    def test_prm_z_is_kept_by_default(self):
        original = track()['boxes_global'][:, :7]; pos = original.copy(); pos[:, 2] = 4
        geo = np.array([0,0,0,4,2,2,0])
        np.testing.assert_allclose(b.combine_boxes(geo, pos, original)[:, 2], 4)
        np.testing.assert_allclose(b.combine_boxes(geo, pos, original, 'preserve_bottom')[:, 2], 1)

    def test_class_preserving_tracker_and_empty_frames(self):
        frames = {}
        for i in range(5):
            names = ['car','truck'] if i in (1,2,3) else []
            frames[str(i)] = {'boxes_global': np.array([[10,0,1,4,2,2,0]]*len(names), dtype=np.float32).reshape(-1,7),
                             # Match bridge.prepare(), including string dtype on empty frames.
                             'name': np.array(['Vehicle']*len(names), dtype=str),
                             'nusc_name': np.array(names, dtype=str),
                             'score': np.full(len(names), .8), 'source_index': np.arange(len(names)),
                             'pose': np.eye(4), 'timestamp': i*.2}
        tracks = r.run_tracking({'frames': {'s': frames}, 'classes': ['car','truck']}, CFG, 'cpu')
        self.assertEqual({v['nusc_name'] for v in tracks.values()}, {'car','truck'})
        self.assertEqual(len(tracks), 2)
        for t in tracks.values():
            np.testing.assert_array_equal(t['sample_idx'], ['1','2','3'])
            self.assertTrue(np.all(np.diff(t['timestamp']) > 0))

    def test_export_retains_confidence_and_passthrough(self):
        prepared = {'original': {'meta': {}, 'results': {'t': [record(), record('barrier')]}},
                    'frames': {'s': {'0': {'sample_token': 't'}}}}
        t = track(n=1)
        boxes = t['boxes_global'][:, :7].copy(); boxes[:,0] = 11
        output, n = b.export_detection(prepared, {'a':t,'b':copy.deepcopy(t)}, {'a':boxes,'b':boxes})
        self.assertEqual(n, 1)
        self.assertEqual(len(output['results']['t']), 2)
        self.assertEqual(output['results']['t'][0]['detection_score'], .8)
        self.assertEqual(output['results']['t'][1], prepared['original']['results']['t'][1])
        self.assertEqual(prepared['original']['results']['t'][0]['translation'][0], 10)

    def test_real_tracker_missing_observation_can_be_exported(self):
        frames, original = {}, {'meta': {}, 'results': {}}
        for i in range(7):
            token = f't{i}'
            detections = [record(token=token, x=10+i*.1)] if i in [1, 2, 4, 5] else []
            original['results'][token] = detections
            frames[str(i)] = dict(
                boxes_global=np.asarray([b.nusc_box(a) for a in detections], dtype=np.float32).reshape(-1, 7),
                name=np.asarray(['Vehicle']*len(detections), dtype=str),
                nusc_name=np.asarray(['car']*len(detections), dtype=str),
                score=np.asarray([.8]*len(detections)), source_index=np.arange(len(detections)),
                sample_token=token, pose=np.eye(4), ego_translation=np.zeros(3), timestamp=i*.2)
        prepared = dict(frames={'s': frames}, classes=['car'], original=original)
        tracks = r.run_tracking(prepared, CFG, 'cpu')
        self.assertEqual(len(tracks), 1)
        tid, t = next(iter(tracks.items()))
        missing = list(t['sample_idx']).index('3')
        self.assertEqual(t['source_index'][missing], -1)
        refined = {tid: t['boxes_global'][:, :7].copy()}
        detection, _ = b.export_detection(prepared, tracks, refined)
        with tempfile.TemporaryDirectory() as tmp:
            tracking = r.export_tracking_results(prepared, tracks, refined, Path(tmp)/'tracking.json')
        filled, filled_tracking, report = export_gap_filled(prepared, tracks, refined, detection, tracking)
        self.assertEqual(report['counts']['added_boxes'], 1)
        self.assertEqual(len(filled['results']['t3']), 1)
        self.assertEqual(filled_tracking['results']['t3'][0]['tracking_id'], tid)
        self.assertEqual(filled['results']['t0'], [])
        self.assertEqual(filled['results']['t6'], [])

    def test_tracking_filters_construction_and_ego_velocity(self):
        t=track(n=1);t['nusc_name']='construction_vehicle'
        prepared={'original':{'meta':{},'results':{'t':[record('construction_vehicle')]}},'frames':{'s':{'0':{'sample_token':'t'}}}}
        with tempfile.TemporaryDirectory() as d:
            out=r.export_tracking_results(prepared,{'id':t},{},Path(d)/'o.json')
            self.assertEqual(out['results']['t'], [])
            t['nusc_name']='car';prepared['original']['results']['t']=[record()]
            out=r.export_tracking_results(prepared,{'id':t},{},Path(d)/'o.json','ego')
            self.assertNotIn('velocity',out['results']['t'][0])

    def test_grm_policy_runs_all_three_classes(self):
        for cls in b.MODEL_PREFIX:
            model = GeometryModel()
            out, status = r.run_refining({'id':track(cls)}, {'grm':{cls:model}}, 'cpu', 'grm', 'geometry')
            self.assertEqual(model.calls,1)
            self.assertEqual(status['id']['grm'],'accepted')
            np.testing.assert_allclose(out['id'][:,3],4.2)

    def test_sparse_and_failure_are_not_counted_as_refined(self):
        t=track();t['pts']=[np.empty((0,4))]*3
        _, status=r.run_refining({'id':t},{},'cpu',refinement='geometry')
        self.assertEqual(status['id']['status'],'insufficient_points')
        with self.assertRaises(RuntimeError): r.run_refining({'id':track()},{'grm':{}},'cpu',refinement='geometry')
        _, status=r.run_refining({'id':track()},{'grm':{}},'cpu',refinement='geometry',allow_errors=True)
        self.assertEqual(status['id']['status'],'error_fallback')

    def test_long_prm_track_keeps_frame_alignment(self):
        t=track(n=401)
        out=r.position_chunks(t,IdentityPositionModel(),'cpu',np.random.default_rng(0))
        np.testing.assert_allclose(out,t['boxes_global'][:,:7],atol=1e-5)

    def test_evaluator_perfect_empty_and_foreign_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'v1.0-trainval';p.mkdir()
            tables={'sample':[{'token':'t'}], 'ego_pose':[{'token':'e','translation':[0,0,0]}],
                    'calibrated_sensor':[{'token':'c','sensor_token':'l'}], 'sensor':[{'token':'l','channel':'LIDAR_TOP'}],
                    'sample_data':[{'token':'sd','sample_token':'t','is_key_frame':True,'calibrated_sensor_token':'c','ego_pose_token':'e'}],
                    'instance':[{'token':'i','category_token':'cat'}], 'category':[{'token':'cat','name':'vehicle.car'}],
                    'sample_annotation':[dict(token='a',instance_token='i',sample_token='t',translation=[10.,0.,1.],size=[2.,4.,2.],rotation=[1.,0.,0.,0.],num_lidar_pts=10)]}
            for name,rows in tables.items():(p/(name+'.json')).write_text(json.dumps(rows))
            prediction=root/'prediction.json';prediction.write_text(json.dumps({'results':{'t':[record()]}}))
            out=evaluate(root,'v1.0-trainval',prediction,['car'])
            self.assertAlmostEqual(out['custom_mAP'],1.)
            self.assertAlmostEqual(out['per_class']['car']['scale_err'],0.)
            prediction.write_text(json.dumps({'results':{'t':[]}}))
            self.assertEqual(evaluate(root,'v1.0-trainval',prediction,['car'])['custom_mAP'],0.)
            prediction.write_text(json.dumps({'results':{'wrong':[]}}))
            with self.assertRaises(ValueError):evaluate(root,'v1.0-trainval',prediction,['car'])

    @unittest.skipUnless((r.DETZERO_ROOT/'checkpoints/vehicle_grm_model.pth').exists(), 'Local trusted checkpoints not installed')
    def test_real_six_checkpoints_and_forward(self):
        torch.set_num_threads(2)
        models=r.load_refining_models(r.DETZERO_ROOT/'checkpoints','cpu')
        for cls in b.MODEL_PREFIX:
            out,status=r.run_refining({'id':track(cls)},models,'cpu','grm','full')
            self.assertEqual(status['id']['status'],'processed')
            self.assertEqual(status['id']['grm'],'accepted')
            self.assertEqual(status['id']['prm'],'applied')
            self.assertTrue(np.isfinite(out['id']).all())
            self.assertTrue((out['id'][:,3:6]>0).all())


if __name__=='__main__':unittest.main()
