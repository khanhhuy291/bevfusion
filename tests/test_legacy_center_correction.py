"""Checkpoint-convention regression: real coder algebra and tilted sensor frame."""
import ast
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from pyquaternion import Quaternion
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.correct_legacy_box_centers import correct,CONVENTION

class CenterTests(unittest.TestCase):
    def test_actual_legacy_encode_decode_then_export(self):
        tree=ast.parse((ROOT/'mmdet3d/core/bbox/coders/transfusion_bbox_coder.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));cls.decorator_list=[]
        scope={'torch':torch,'BaseBBoxCoder':object};exec(compile(ast.Module(body=[cls],type_ignores=[]),'coder','exec'),scope)
        coder=scope['TransFusionBBoxCoder']([-54,-54],8,[.075,.075],code_size=10)
        # Converter writes true geometric center; legacy GT loader origin=0 leaves Z unchanged.
        gt=torch.tensor([[10.,5.,-1.,.8,.7,1.8,.2,0.,0.]])
        targets=coder.encode(gt)
        decoded=coder.decode(torch.ones(1,1,1),targets[:,6:8].T[None],targets[:,3:6].T[None].clone(),
                             targets[:,:2].T[None].clone(),targets[:,2:3].T[None],targets[:,8:10].T[None])[0]['bboxes']
        torch.testing.assert_close(decoded,gt)
        exported=decoded[0,:3].numpy().copy();exported[2]+=decoded[0,5].item()/2
        d={'results':{'t':[dict(sample_token='t',translation=exported.tolist(),size=[.8,.7,1.8],detection_score=.5)]}}
        out=correct(d,{'t':(np.eye(3),np.zeros(3))},CONVENTION)
        np.testing.assert_allclose(out['results']['t'][0]['translation'],gt[0,:3],atol=1e-6)

    def test_tilted_lidar_global_axis_and_fields_unchanged(self):
        r=Quaternion(axis=[1,2,3],angle=.5).rotation_matrix;center=np.array([13.,9.,2.]);h=2.
        box=dict(sample_token='t',translation=(center+r[:,2]*h/2).tolist(),size=[1,1,h],rotation=[1,0,0,0],velocity=[2,3],detection_name='pedestrian',detection_score=.4)
        doc={'meta':{'use_lidar':True},'results':{'t':[box]}};original=copy.deepcopy(doc)
        out=correct(doc,{'t':(r,np.zeros(3))},CONVENTION)
        np.testing.assert_allclose(out['results']['t'][0]['translation'],center)
        self.assertEqual(doc,original)
        for k in box:
            if k!='translation':self.assertEqual(out['results']['t'][0][k],box[k])
        with self.assertRaises(ValueError):correct(out,{'t':(r,np.zeros(3))},CONVENTION)
        with self.assertRaises(ValueError):correct(doc,{'other':(r,np.zeros(3))},CONVENTION)
        with self.assertRaises(ValueError):correct(doc,{'t':(r,np.zeros(3))},'unknown')

if __name__=='__main__':unittest.main()
