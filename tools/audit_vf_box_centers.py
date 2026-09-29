"""Audit raw versus legacy-corrected LiDAR-Z residuals; fixed same-class XY matches."""
import argparse
import json,sys
from pathlib import Path
import numpy as np
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.correct_legacy_box_centers import lidar_global_poses

def audit(root,version,predpath):
 root=Path(root);read=lambda n:json.loads((root/version/(n+'.json')).read_text());cats={x['token']:x['name'] for x in read('category')};inst={x['token']:cats[x['category_token']] for x in read('instance')};gt=defaultdict(list);poses=lidar_global_poses(root,version);pred=json.loads(Path(predpath).read_text())['results']
 for a in read('sample_annotation'):
  c=inst[a['instance_token']];name='pedestrian' if c.startswith('human.pedestrian.') else {'vehicle.car':'car','vehicle.motorcycle':'motorcycle'}.get(c)
  if name and a['num_lidar_pts']>0:gt[(a['sample_token'],name)].append(a)
 out={}
 for name in ['car','motorcycle','pedestrian']:
  raw=[];fixed=[];score=[]
  for token,pp in pred.items():
   used=set();gg=gt[(token,name)]
   for p in sorted([p for p in pp if p['detection_name']==name and p['detection_score']>=.1],key=lambda p:-p['detection_score']):
    cand=[(np.linalg.norm(np.array(g['translation'][:2])-p['translation'][:2]),j,g) for j,g in enumerate(gg) if j not in used]
    if not cand:continue
    d,j,g=min(cand,key=lambda x:x[0])
    if d>=.5:continue
    used.add(j);delta=poses[token][0].T@(np.array(p['translation'])-g['translation']);raw.append(float(delta[2]));fixed.append(float(delta[2]-p['size'][2]/2));score.append(p['detection_score'])
  out[name]={'matches_xy_lt_0_5_score_ge_0_1':len(raw),'raw_median_dz_lidar':float(np.median(raw)) if raw else None,'corrected_median_dz_lidar':float(np.median(fixed)) if fixed else None,'raw_median_abs_dz_lidar':float(np.median(np.abs(raw))) if raw else None,'corrected_median_abs_dz_lidar':float(np.median(np.abs(fixed))) if raw else None}
 return out
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--version',default='v1.0-trainval')
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    report={'protocol':'Same-class greedy matching at XY <0.5m and score >=0.1; GT with LiDAR points. Fixed pairs for before/after residual comparison, not AP. Z measured in source LiDAR axes.',
            'per_class':audit(a.data_root,a.version,a.predictions)}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
