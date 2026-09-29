"""Audit VF camera crop coverage and LiDAR frame; optionally diagnose VM predictions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from tools.adapt_vf_calibration import read_tables, reference_rig, model_affine, rotation
except ModuleNotFoundError:
    from adapt_vf_calibration import read_tables, reference_rig, model_affine, rotation

CLASSES = {'vehicle.car':'car', 'vehicle.motorcycle':'motorcycle', 'human.pedestrian.adult':'pedestrian'}


def project(points, calibration):
    cam = (points - calibration['translation']) @ rotation(calibration['rotation'])
    pix = cam @ np.asarray(calibration['camera_intrinsic']).T
    return pix[:, :2] / np.where(abs(pix[:, 2:3]) > 1e-9, pix[:, 2:3], 1e-9), cam[:, 2]


def inside(uv, depth, width, height):
    return (depth > .1) & (uv[:,0]>=0) & (uv[:,0]<width) & (uv[:,1]>=0) & (uv[:,1]<height)


def apply_affine(uv, a):
    return uv @ a[:2,:2].T + a[:2,2]


def diagnose(root, reference_root, predictions=None, version='v1.0-trainval', reference_scene='scene-0103'):
    tables = read_tables(root, version)
    if not all(np.allclose(e['translation'], 0) and np.allclose(rotation(e['rotation']), np.eye(3)) for e in tables['ego_pose']):
        raise ValueError('This diagnostic is scoped to identity-ego VF')
    ref, token = reference_rig(reference_root, 'v1.0-mini', reference_scene)
    sensors = {s['token']:s['channel'] for s in tables['sensor']}
    calibs = {c['token']:c for c in tables['calibrated_sensor']}
    by_sample = defaultdict(dict)
    for sd in tables['sample_data']:
        if sd['is_key_frame']: by_sample[sd['sample_token']][sensors[calibs[sd['calibrated_sensor_token']]['sensor_token']]] = sd
    cats={c['token']:c['name'] for c in tables['category']}
    instances={i['token']:cats[i['category_token']] for i in tables['instance']}
    evaluated=defaultdict(list)
    for a in tables['sample_annotation']:
        name=CLASSES.get(instances[a['instance_token']])
        if name and a['num_lidar_pts']>0 and np.linalg.norm(a['translation'][:2]) < (50 if name=='car' else 40):
            evaluated[a['sample_token']].append((name,a))
    totals=defaultdict(Counter); sizes=defaultdict(list); z_source=defaultdict(list);z_reference=defaultdict(list)
    for sample,annotations in evaluated.items():
        points=np.array([a['translation'] for _,a in annotations])
        visible={n:np.zeros(len(points),bool) for n in ['raw_image_center','baseline_model_crop_center','reference_camera_model_crop_center']}
        sd_lidar=by_sample[sample]['LIDAR_TOP'];src_lidar=calibs[sd_lidar['calibrated_sensor_token']]
        pl=(points-src_lidar['translation']) @ rotation(src_lidar['rotation'])
        ref_lidar=ref['LIDAR_TOP']['calibration'];pr=(points-ref_lidar['translation']) @ rotation(ref_lidar['rotation'])
        for channel,sd in by_sample[sample].items():
            if not channel.startswith('CAM_'):continue
            c=calibs[sd['calibrated_sensor_token']]
            uv,depth=project(points,c);raw=inside(uv,depth,sd['width'],sd['height'])
            visible['raw_image_center'] |= raw
            visible['baseline_model_crop_center'] |= raw & inside(apply_affine(uv,model_affine(sd['width'],sd['height'])),depth,704,256)
            virtual=dict(ref[channel]['calibration']);virtual['translation']=c['translation']
            uv_new,dep_new=project(points,virtual)
            # Restrict to rays actually recorded in this physical source camera.
            visible['reference_camera_model_crop_center'] |= raw & inside(apply_affine(uv_new,model_affine(ref[channel]['width'],ref[channel]['height'])),dep_new,704,256)
        for i,(name,a) in enumerate(annotations):
            totals[name]['gt_evaluated']+=1
            for key,mask in visible.items():totals[name][key]+=int(mask[i])
            sizes[name].append(a['size']);z_source[name].append(pl[i,2]);z_reference[name].append(pr[i,2])
    report={'scope':'Geometric center coverage, not visibility/occlusion or recall/AP.',
            'image_augmentation':{'resize':.48,'final_dim':[256,704],'bot_pct_lim':[0,0]},
            'reference_scene':reference_scene,'reference_sample_token':token,
            'coverage':{n:dict(c) for n,c in totals.items()},
            'geometry':{n:{'median_wlh':np.median(sizes[n],axis=0).tolist(),
                           'median_center_z_source_lidar':float(np.median(z_source[n])),
                           'median_center_z_reference_lidar':float(np.median(z_reference[n]))} for n in sizes},
            'reference_lidar':ref['LIDAR_TOP']['calibration']}
    if predictions:
        pred=json.loads(Path(predictions).read_text())['results']
        if set(pred)!={s['token'] for s in tables['sample']}:raise ValueError('Prediction/sample tokens differ')
        scores=defaultdict(list)
        for items in pred.values():
            for item in items:scores[item['detection_name']].append(item['detection_score'])
        report['prediction_counts_and_scores']={n:{'count':len(s),'score_p10_p50_p90':np.quantile(s,[.1,.5,.9]).tolist()} for n,s in scores.items()}
        report['pedestrian_spatial_diagnostic']={}
        for threshold in [0.,.1,.25,.5]:
            counts=Counter()
            for sample,annotations in evaluated.items():
                gt=[a for name,a in annotations if name=='pedestrian']
                if not gt:continue
                candidates=[p for p in pred[sample] if p['detection_score']>=threshold]
                counts['gt']+=len(gt)
                if not candidates:counts['unmatched']+=len(gt);continue
                distance=np.linalg.norm(np.array([a['translation'][:2] for a in gt])[:,None,:]-np.array([p['translation'][:2] for p in candidates])[None,:,:],axis=2)
                cost=distance.copy();cost[distance>2.]=10000.
                rows,cols=linear_sum_assignment(cost);matched=0
                for i,j in zip(rows,cols):
                    if distance[i,j]<=2.:
                        counts['near_'+candidates[j]['detection_name']]+=1;matched+=1
                counts['unmatched']+=len(gt)-matched
            report['pedestrian_spatial_diagnostic'][str(threshold)]=dict(counts)
        report['prediction_diagnostic_note']='One-to-one center proximity within 2m to pedestrian GT across all predicted classes. Not the AP matching protocol; proximity alone does not establish object identity.'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='data/nuscenes_vf6_01_5hz')
    p.add_argument('--reference-root',default='data/nuscenes')
    p.add_argument('--predictions')
    p.add_argument('--output',default='outputs/vf6_01_5hz/calibration_diagnostic.json')
    a=p.parse_args();report=diagnose(a.data_root,a.reference_root,a.predictions)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
