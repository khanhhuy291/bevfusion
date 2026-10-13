"""Local, read-only camera / BEV / 3D inspector for saved VF detection outputs.

No inference, cloud upload or external frontend dependencies. Metadata comes
from the run's localized dataset; physical sensor files may use another root.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np
from pyquaternion import Quaternion
try:
    from .vf_iou3d import iou3d_matrix
except ImportError:
    from vf_iou3d import iou3d_matrix

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'apps/vf_object_inspector'
CLASS_MAP = {'vehicle.car':'car','vehicle.truck':'truck','vehicle.motorcycle':'motorcycle',
    'vehicle.bicycle':'bicycle','vehicle.bus.rigid':'bus','vehicle.bus.bendy':'bus',
    'vehicle.trailer':'trailer','vehicle.construction':'construction_vehicle',
    'movable_object.barrier':'barrier','movable_object.trafficcone':'traffic_cone',
    'human.pedestrian.adult':'pedestrian','human.pedestrian.child':'pedestrian',
    'human.pedestrian.police_officer':'pedestrian','human.pedestrian.construction_worker':'pedestrian'}
RANGES = {'car':50.,'motorcycle':40.,'pedestrian':40.}


def read(path): return json.loads(Path(path).read_text())
def eval_class(name): return 'car' if name=='truck' else name

def transform(pose):
    matrix=np.eye(4);matrix[:3,:3]=Quaternion(pose['rotation']).rotation_matrix
    matrix[:3,3]=pose['translation'];return matrix

def box_corners(record):
    # Same corner order as nuScenes Box.corners(): front face, then rear face.
    w,l,h=record['size']
    points=np.array([[l,l,l,l,-l,-l,-l,-l],[w,-w,-w,w,w,-w,-w,w],[h,h,-h,-h,h,h,-h,-h]],float).T/2
    return points@Quaternion(record['rotation']).rotation_matrix.T+record['translation']

def transformed(points, matrix):return points@matrix[:3,:3].T+matrix[:3,3]

def confidence_matches(records, targets, threshold):
    """One-frame evaluator-style confidence-ordered, one-to-one IoU matching."""
    output={}; matched=set()
    for name in RANGES:
        pred=[r for r in records if r['evaluated'] and r['eval_class']==name]
        gt=[r for r in targets if r['evaluated'] and r['eval_class']==name]
        matrix=iou3d_matrix([SimpleNamespace(**r['box']) for r in pred],[SimpleNamespace(**r['box']) for r in gt])
        for i in sorted(range(len(pred)),key=lambda k:(-pred[k]['score'],pred[k]['source_index'])):
            free=[j for j in range(len(gt)) if gt[j]['id'] not in matched]
            if not free:continue
            j=max(free,key=lambda j:matrix[i,j])
            if matrix[i,j]>=threshold:
                output[pred[i]['source_index']]=gt[j]['id'];matched.add(gt[j]['id'])
    return output


def diagnostic_pairs(records, targets, max_distance=.5):
    """Freeze same-class baseline source pairs for visual comparison, not TP/FP."""
    pairs={};claimed=set()
    for p in sorted(records,key=lambda p:(-p['score'],p['source_index'])):
        candidates=[g for g in targets if g['id'] not in claimed and g['eval_class']==p['eval_class']]
        if not candidates:continue
        target=min(candidates,key=lambda g:np.linalg.norm(np.array(p['box']['translation'])[:2]-g['box']['translation'][:2]))
        distance=np.linalg.norm(np.array(p['box']['translation'])[:2]-target['box']['translation'][:2])
        if distance<=max_distance:
            pairs[target['id']]=p['source_index'];claimed.add(target['id'])
    return pairs


class Dataset:
    def __init__(self,run,refined_dir,sensor_root,version):
        self.run=Path(run).resolve();self.refined_dir=refined_dir
        self.sensors=Path(sensor_root).resolve();self.version=version
        meta=self.run/'dataset_local'/version
        def table(name):return {x['token']:x for x in read(meta/(name+'.json'))}
        self.samples=sorted(table('sample').values(),key=lambda x:x['timestamp'])
        self.calibs=table('calibrated_sensor');self.egos=table('ego_pose')
        sensors=table('sensor');instances=table('instance');categories=table('category')
        self.by_sample=defaultdict(dict)
        for r in table('sample_data').values():
            if r['is_key_frame']:
                channel=sensors[self.calibs[r['calibrated_sensor_token']]['sensor_token']]['channel']
                self.by_sample[r['sample_token']][channel]=r
        self.annotations=defaultdict(list)
        for r in table('sample_annotation').values():
            category=categories[instances[r['instance_token']]['category_token']]['name']
            cls=CLASS_MAP.get(category)
            protocol_gt=category in {'vehicle.car','vehicle.truck','vehicle.motorcycle','human.pedestrian.adult'}
            if cls:self.annotations[r['sample_token']].append(dict(r,cls=cls,protocol_gt=protocol_gt))
        self.baseline=read(self.run/'baseline/results_nusc_center_corrected.json')['results']
        self.refined=read(self.run/refined_dir/'results_nusc_detzero_refined.json')['results']
        assert set(self.baseline)==set(self.refined)=={s['token'] for s in self.samples}
        for token,rows in self.baseline.items():
            if len(rows)!=len(self.refined[token]):raise ValueError('Requires detection JSONs without gap filling, with the original candidate order')
            for a,b in zip(rows,self.refined[token]):
                if a['detection_name']!=b['detection_name'] or a['detection_score']!=b['detection_score']:
                    raise ValueError('Baseline/refined source-index, class or confidence alignment differs')
        self.comparison=read(self.run/refined_dir/'comparison_iou3d.json') if (self.run/refined_dir/'comparison_iou3d.json').exists() else None
        self.stamp=hashlib.sha256((str(self.sensors)+str(meta)+str(self.samples[0]['timestamp'])).encode()).hexdigest()[:12]
        self.examples=[]
        example_path=self.run/'visual_overlay/manifest.json'
        if example_path.exists():
            for f in read(example_path)['frames']:
                focus=f.get('focus')
                if focus and (f['index'],focus['class']) in [(30,'motorcycle'),(30,'car'),(142,'pedestrian'),(80,'motorcycle')]:
                    self.examples.append({'frame':f['index'],'gt_id':'g:'+focus['gt_token'],'class':focus['class']})

    def sample(self,index):
        if not 0<=index<len(self.samples):raise ValueError('Frame index outside dataset')
        return self.samples[index]

    def metadata(self):
        return {'title':'VF Object Inspector','run':str(self.run.name),'detzero_run':self.refined_dir,
                'frames':len(self.samples),'cameras':sorted(k for k in self.by_sample[self.samples[0]['token']] if k.startswith('CAM_')),
                'examples':self.examples,'classes':sorted({p['detection_name'] for rows in self.baseline.values() for p in rows}),
                'pair_distance_m':.5,'summary':self.comparison,
                'policy':'Car includes truck. GT evaluation requires nonzero LiDAR points and ranges 50/40/40m. Viewer score filtering does not rerun DetZero.'}

    @lru_cache(maxsize=16)
    def frame(self,index,score,threshold,include_other):
        sample=self.sample(index);token=sample['token'];sensors=self.by_sample[token]
        lidar=sensors['LIDAR_TOP'];ego=transform(self.egos[lidar['ego_pose_token']]);inverse=np.linalg.inv(ego)
        cameras={}
        for name,row in sensors.items():
            if not name.startswith('CAM_'):continue
            calibration=self.calibs[row['calibrated_sensor_token']]
            camera_to_global=transform(self.egos[row['ego_pose_token']])@transform(calibration)
            cameras[name]={'width':row['width'],'height':row['height'],'intrinsic':calibration['camera_intrinsic'],
                'ego_to_camera':(np.linalg.inv(camera_to_global)@ego).tolist(),
                'image_url':f'/image/{index}/{name}?stamp={self.stamp}'}
        def wrap(box,cls,id,source=None,instance=None,points=None,protocol_gt=True):
            name=eval_class(cls); distance=float(np.linalg.norm((np.array(box['translation'])-ego[:3,3])[:2]))
            evaluated=protocol_gt and name in RANGES and distance<RANGES.get(name,0) and (points is None or points>0)
            geometry={k:box[k] for k in ['translation','size','rotation']}
            return {'id':id,'class':cls,'eval_class':name,'source_index':source,'instance':instance,
                'score':box.get('detection_score'),'evaluated':evaluated,'num_lidar_pts':points,'range_m':distance,
                'box':geometry,'center':transformed(np.array(box['translation'])[None],inverse)[0].tolist(),
                'corners':transformed(box_corners(box),inverse).tolist(),
                'bottom_z':float(box['translation'][2]-box['size'][2]/2)}
        gt=[wrap(a,a['cls'],'g:'+a['token'],instance=a['instance_token'],points=a['num_lidar_pts'],protocol_gt=a['protocol_gt']) for a in self.annotations[token]]
        if not include_other:gt=[g for g in gt if g['evaluated']]
        baseline=[wrap(p,p['detection_name'],'p:'+str(i),source=i) for i,p in enumerate(self.baseline[token]) if p['detection_score']>=score and (include_other or (eval_class(p['detection_name']) in RANGES and np.linalg.norm((np.array(p['translation'])-ego[:3,3])[:2])<RANGES[eval_class(p['detection_name'])]))]
        # Keep the same source candidates, even when a refined box crosses range.
        refined=[wrap(self.refined[token][p['source_index']],p['class'],p['id'],source=p['source_index']) for p in baseline]
        bmap={p['source_index']:p for p in baseline};rmap={p['source_index']:p for p in refined}
        matches_b=confidence_matches(baseline,gt,threshold);matches_r=confidence_matches(refined,gt,threshold)
        pairs=diagnostic_pairs(baseline,gt);objects=[];used=set(pairs.values())
        for g in gt:
            i=pairs.get(g['id']);b=bmap.get(i);r=rmap.get(i)
            overlaps=[None,None]
            if b:
                overlaps=[float(iou3d_matrix([SimpleNamespace(**p['box'])],[SimpleNamespace(**g['box'])])[0,0]) for p in [b,r]]
            objects.append({'id':g['id'],'class':g['class'],'eval_class':g['eval_class'],'gt':g['id'],
                'source_index':i,'instance':g['instance'],'iou_baseline':overlaps[0],'iou_detzero':overlaps[1],
                'delta_iou':None if overlaps[0] is None else overlaps[1]-overlaps[0],
                'range_m':g['range_m'],'evaluated':g['evaluated'],'kind':'gt'})
        for b in baseline:
            if b['source_index'] in used:continue
            objects.append({'id':b['id'],'class':b['class'],'eval_class':b['eval_class'],'gt':None,
                'source_index':b['source_index'],'instance':None,'iou_baseline':None,'iou_detzero':None,
                'delta_iou':None,'range_m':b['range_m'],'evaluated':b['evaluated'],'kind':'prediction'})
        objects.sort(key=lambda o:(o['kind']!='gt',o['range_m']))
        return {'index':index,'timestamp':sample['timestamp'],'sample_token':token,'cameras':cameras,
            'gt':gt,'baseline':baseline,'detzero':refined,'objects':objects,'score':score,'iou_threshold':threshold,
            'counts':{'gt':len(gt),'baseline':len(baseline),'detzero':len(refined),
                'tp_baseline':len(matches_b),'tp_detzero':len(matches_r)},
            'matches':{'baseline':matches_b,'detzero':matches_r},
            'pair_policy':'Confidence-ordered nearest same-class baseline/GT XY <= 0.5m, one-to-one; same source index frozen for DetZero. Diagnostic pairing is not a TP decision.'}

    @lru_cache(maxsize=8)
    def points(self,index,limit):
        token=self.sample(index)['token'];row=self.by_sample[token]['LIDAR_TOP'];cal=self.calibs[row['calibrated_sensor_token']]
        path=self.sensor_file(index,'LIDAR_TOP')
        raw=np.fromfile(path,dtype=np.float32)
        if raw.size%5:raise ValueError('Expected LiDAR float32 records with five columns')
        points=transformed(raw.reshape(-1,5)[:,:3],transform(cal))
        points=points[np.isfinite(points).all(axis=1)&(np.abs(points[:,:2]).max(axis=1)<85)&(points[:,2]>-6)&(points[:,2]<12)]
        if len(points)>limit:points=points[np.linspace(0,len(points)-1,limit,dtype=int)]
        return np.asarray(points,dtype='<f4').tobytes()

    def sensor_file(self,index,channel):
        token=self.sample(index)['token'];row=self.by_sample[token].get(channel)
        if row is None:raise ValueError('Unknown sensor channel')
        path=(self.sensors/row['filename']).resolve()
        if not path.is_relative_to(self.sensors):raise ValueError('Sensor filename escapes configured root')
        if not path.is_file():raise FileNotFoundError(f'Missing sensor file: {path}')
        return path


def handler(dataset):
    class Handler(BaseHTTPRequestHandler):
        def send(self,payload,type='application/json; charset=utf-8',status=200,cache=False):
            if not isinstance(payload,bytes):payload=json.dumps(payload,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',type);self.send_header('Content-Length',str(len(payload)))
            self.send_header('Cache-Control','private, max-age=3600' if cache else 'no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            try:
                url=urlparse(self.path);parts=url.path.strip('/').split('/');query=parse_qs(url.query)
                static={'':'index.html','index.html':'index.html','app.js':'app.js','style.css':'style.css','geometry.mjs':'geometry.mjs'}
                if url.path.strip('/') in static:
                    path=ASSETS/static[url.path.strip('/')];types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8'}
                    return self.send(path.read_bytes(),types[path.suffix])
                if parts==['api','meta']:return self.send(dataset.metadata())
                if len(parts)==3 and parts[:2]==['api','frame']:
                    score=float(query.get('score',['.1'])[0]);threshold=float(query.get('iou',['.5'])[0]);other=query.get('other',['0'])[0]=='1'
                    if not 0<=score<=1 or threshold not in [.25,.5,.7]:raise ValueError('Invalid score/IoU threshold')
                    return self.send(dataset.frame(int(parts[2]),score,threshold,other))
                if len(parts)==3 and parts[:2]==['api','points']:
                    limit=int(query.get('limit',['12000'])[0])
                    if not 100<=limit<=50000:raise ValueError('Point limit must be 100..50000')
                    return self.send(dataset.points(int(parts[2]),limit),'application/octet-stream',cache=True)
                if len(parts)==3 and parts[0]=='image':return self.send(dataset.sensor_file(int(parts[1]),parts[2]).read_bytes(),'image/jpeg',cache=True)
                if parts==['favicon.ico']:return self.send(b'','image/x-icon',204)
                self.send({'error':'Not found'},status=404)
            except (ValueError,KeyError) as error:self.send({'error':str(error)},status=400)
            except FileNotFoundError as error:self.send({'error':str(error)},status=404)
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as error:self.send({'error':str(error)},status=500)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',default='outputs/vf-car-truck-sweeps2-01')
    parser.add_argument('--refined-dir',default='grm_prm_zfix_OTAgmB')
    parser.add_argument('--sensor-root',default='data/nuscenes_vf6_01_adapted')
    parser.add_argument('--version',default='v1.0-mini')
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();data=Dataset(args.run_dir,args.refined_dir,args.sensor_root,args.version)
    server=ThreadingHTTPServer((args.host,args.port),handler(data))
    print(f'VF Object Inspector: http://{args.host}:{args.port}\nRun: {data.run}/{args.refined_dir}\nRead-only sensor root: {data.sensors}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
