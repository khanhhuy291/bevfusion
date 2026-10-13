#!/usr/bin/env python3
"""Local, read-only camera / BEV / 3D inspection of saved VF detections.

Sensor files come from data_root; world-frame metadata must come from the
prediction run's dataset_local (its poses/GT may be localized). No model inference.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.vf_iou3d import iou3d_matrix

STATIC = Path(__file__).parent / 'static'
CHANNELS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
RANGES = {'car': 50., 'motorcycle': 40., 'pedestrian': 40.}
GT_MAPPING = {'vehicle.car': 'car', 'vehicle.truck': 'car', 'vehicle.motorcycle': 'motorcycle',
              'human.pedestrian.adult': 'pedestrian'}


def read(path):
    return json.loads(Path(path).read_text())


def pose(row):
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(row['rotation']).rotation_matrix
    matrix[:3, 3] = row['translation']
    return matrix


def global_corners(record):
    w, length, h = np.asarray(record['size']) / 2
    xyz = np.array([[length,w,h], [length,-w,h], [length,-w,-h], [length,w,-h],
                    [-length,w,h], [-length,-w,h], [-length,-w,-h], [-length,w,-h]])
    return xyz @ Quaternion(record['rotation']).rotation_matrix.T + np.array(record['translation'])


def box(record):
    return SimpleNamespace(**{k: record[k] for k in ['translation', 'size', 'rotation']})


class Dataset:
    def __init__(self, data_root, run_dir, refined_path=None):
        self.data_root = Path(data_root).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.refined_path = Path(refined_path).resolve() if refined_path else self.run_dir/'grm_prm_zfix_OTAgmB/results_nusc_detzero_refined.json'
        self.baseline_path = self.run_dir/'baseline/results_nusc_center_corrected.json'
        self.meta = self.run_dir/'dataset_local/v1.0-mini'
        def table(name):
            return {x['token']: x for x in read(self.meta/(name+'.json'))}
        self.samples = sorted(table('sample').values(), key=lambda x: x['timestamp'])
        self.scenes = table('scene')
        self.calibs, self.egos, self.sensors = table('calibrated_sensor'), table('ego_pose'), table('sensor')
        instances, categories = table('instance'), table('category')
        self.annotations = defaultdict(list)
        for row in table('sample_annotation').values():
            name = categories[instances[row['instance_token']]['category_token']]['name']
            if name in GT_MAPPING:
                self.annotations[row['sample_token']].append(dict(row, category=name, group=GT_MAPPING[name]))
        self.sensor_data = defaultdict(dict)
        for row in table('sample_data').values():
            if row['is_key_frame']:
                calib = self.calibs[row['calibrated_sensor_token']]
                channel = self.sensors[calib['sensor_token']]['channel']
                self.sensor_data[row['sample_token']][channel] = row
        self.baseline = read(self.baseline_path)['results']
        self.refined = read(self.refined_path)['results']
        tokens = {r['token'] for r in self.samples}
        if set(self.baseline) != tokens or set(self.refined) != tokens:
            raise ValueError('Prediction tokens do not match run dataset_local metadata')
        for token in tokens:
            before, after = self.baseline[token], self.refined[token]
            if len(before) != len(after):
                raise ValueError('Viewer requires source-index-preserving output; do not use gap-filled JSON')
            for a, b in zip(before, after):
                if a['detection_name'] != b['detection_name'] or a['detection_score'] != b['detection_score']:
                    raise ValueError('Baseline/refined class or score order differs')
            for channel in ['LIDAR_TOP', *CHANNELS]:
                if channel not in self.sensor_data[token]:
                    raise ValueError(f'Missing {channel} in sample {token}')
                self.sensor_file(self.sensor_data[token][channel])
        self.start_time = self.samples[0]['timestamp']

    def sensor_file(self, row):
        path = (self.data_root / row['filename']).resolve()
        if not path.is_relative_to(self.data_root) or not path.is_file():
            raise FileNotFoundError(f'Sensor file unavailable under data root: {row["filename"]}')
        return path

    def frame_rows(self, index):
        if index < 0 or index >= len(self.samples):
            raise ValueError('Frame index out of bounds')
        sample = self.samples[index]
        records = self.sensor_data[sample['token']]
        ego = self.egos[records['LIDAR_TOP']['ego_pose_token']]
        return sample, records, ego

    def record(self, raw, key, group, ego, source_index=None):
        rotation = Quaternion(ego['rotation']).rotation_matrix
        translation = np.array(ego['translation'])
        center = (np.array(raw['translation']) - translation) @ rotation
        corners = (global_corners(raw) - translation) @ rotation
        orientation = rotation.T @ Quaternion(raw['rotation']).rotation_matrix
        return {'key': key, 'group': group, 'class': raw.get('detection_name', raw.get('category', group)),
                'position': center.tolist(), 'global_position': raw['translation'], 'size': raw['size'],
                'corners': corners.tolist(), 'yaw': float(np.arctan2(orientation[1,0], orientation[0,0])),
                'score': raw.get('detection_score'), 'source_index': source_index,
                'num_lidar_pts': raw.get('num_lidar_pts'), 'instance': raw.get('instance_token')}

    @lru_cache(maxsize=24)
    def frame(self, index, score=.1, include_zero=False, match_radius=1.):
        sample, sensors, ego = self.frame_rows(index)
        token = sample['token']
        origin = np.array(ego['translation'])
        gt_raw = [g for g in self.annotations[token] if (include_zero or g['num_lidar_pts'] > 0)
                  and np.linalg.norm((np.array(g['translation']) - origin)[:2]) < RANGES[g['group']]]
        # The same source index is retained even if refined boxes cross a range boundary.
        selected = [(i, a) for i,a in enumerate(self.baseline[token])
                    if ('car' if a['detection_name']=='truck' else a['detection_name']) in RANGES
                    and a['detection_score'] >= score
                    and np.linalg.norm((np.array(a['translation'])-origin)[:2]) < RANGES['car' if a['detection_name']=='truck' else a['detection_name']]]
        groups = {'gt': [], 'bev': [], 'detzero': []}
        for g in gt_raw:
            groups['gt'].append(self.record(g, 'gt:'+g['token'], g['group'], ego))
        for i,a in selected:
            group = 'car' if a['detection_name']=='truck' else a['detection_name']
            groups['bev'].append(self.record(a, 'pred:'+str(i), group, ego, i))
            groups['detzero'].append(self.record(self.refined[token][i], 'pred:'+str(i), group, ego, i))
        # Diagnostic pairing: nearest baseline XY, globally greedy one-to-one within a group.
        candidates = []
        for gi,g in enumerate(gt_raw):
            for pi,(source,p) in enumerate(selected):
                group = 'car' if p['detection_name']=='truck' else p['detection_name']
                if group != g['group']:
                    continue
                distance = float(np.linalg.norm((np.array(p['translation'])-g['translation'])[:2]))
                if distance <= match_radius:
                    candidates.append((distance,gi,pi))
        matched, used = {}, set()
        for distance,gi,pi in sorted(candidates):
            if gi not in matched and pi not in used:
                matched[gi] = pi
                used.add(pi)
        objects = []
        for gi,g in enumerate(gt_raw):
            pi = matched.get(gi)
            source = selected[pi][0] if pi is not None else None
            a = selected[pi][1] if pi is not None else None
            b = self.refined[token][source] if source is not None else None
            info = {'key': 'gt:'+g['token'], 'kind': 'gt', 'group': g['group'], 'gt': gi,
                    'bev': pi, 'detzero': pi, 'source_index': source, 'score': a['detection_score'] if a else None,
                    'instance': g['instance_token'], 'token': g['token']}
            for method,p in [('bev',a),('detzero',b)]:
                info[method+'_iou'] = float(iou3d_matrix([box(p)],[box(g)])[0,0]) if p else None
                info[method+'_xy_error'] = float(np.linalg.norm((np.array(p['translation'])-g['translation'])[:2])) if p else None
                info[method+'_bottom_error'] = float(p['translation'][2]-p['size'][2]/2-(g['translation'][2]-g['size'][2]/2)) if p else None
            objects.append(info)
        for pi,(source,a) in enumerate(selected):
            if pi in used:
                continue
            group = 'car' if a['detection_name']=='truck' else a['detection_name']
            objects.append({'key': 'pred:'+str(source), 'kind': 'prediction', 'group': group,
                            'gt': None, 'bev': pi, 'detzero': pi, 'source_index': source,
                            'score': a['detection_score'], 'token': None, 'instance': None,
                            'bev_iou': None, 'detzero_iou': None, 'bev_xy_error': None,
                            'detzero_xy_error': None, 'bev_bottom_error': None, 'detzero_bottom_error': None})
        cameras = {}
        world_from_ego = pose(ego)
        for channel in CHANNELS:
            row = sensors[channel]
            calib = self.calibs[row['calibrated_sensor_token']]
            cam_pose = self.egos[row['ego_pose_token']]
            cameras[channel] = {'width':row['width'], 'height':row['height'], 'K':calib['camera_intrinsic'],
                'from_ego': (np.linalg.inv(pose(calib)) @ np.linalg.inv(pose(cam_pose)) @ world_from_ego).tolist(),
                'image':f'/api/image?frame={index}&camera={channel}',
                'thumbnail':f'/api/image?frame={index}&camera={channel}&width=320'}
        return {'index': index, 'token': token, 'scene': self.scenes[sample['scene_token']]['name'],
                'time_s': (sample['timestamp']-self.start_time)/1e6, 'timestamp_us': sample['timestamp'],
                'cameras': cameras, 'boxes': groups, 'objects': objects,
                'counts': {k:len(v) for k,v in groups.items()}, 'point_url':f'/api/points?frame={index}',
                'pair_policy': f'Nearest baseline XY <= {match_radius:g}m; greedy one-to-one; frozen source index. Diagnostic, not evaluator TP/FP.'}

    @lru_cache(maxsize=12)
    def points(self, index, limit=18000):
        _,records,_ = self.frame_rows(index)
        lidar = records['LIDAR_TOP']
        raw = np.fromfile(self.sensor_file(lidar), dtype=np.float32)
        if len(raw)%5:
            raise ValueError('Expected raw LiDAR records of 5 float32 values')
        raw = raw.reshape(-1,5)[:,:3]
        calib = self.calibs[lidar['calibrated_sensor_token']]
        xyz = raw @ Quaternion(calib['rotation']).rotation_matrix.T + np.array(calib['translation'])
        mask = np.isfinite(xyz).all(axis=1) & (np.abs(xyz[:,:2]).max(axis=1)<65) & (xyz[:,2]>-5) & (xyz[:,2]<10)
        xyz = xyz[mask]
        if len(xyz)>limit:
            xyz = xyz[np.linspace(0,len(xyz)-1,limit,dtype=int)]
        return np.asarray(xyz,dtype='<f4').tobytes()

    @lru_cache(maxsize=36)
    def image(self, index, channel, width=0):
        if channel not in CHANNELS:
            raise ValueError('Unsupported camera')
        _,records,_ = self.frame_rows(index)
        path = self.sensor_file(records[channel])
        if not width:
            return path.read_bytes()
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError('Image cannot be decoded')
        im = cv2.resize(im,(width,round(im.shape[0]*width/im.shape[1])),interpolation=cv2.INTER_AREA)
        ok,jpg = cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,82])
        if not ok:
            raise ValueError('Thumbnail encoding failed')
        return jpg.tobytes()

    def info(self):
        return {'frames':len(self.samples),'classes':list(RANGES),'run':self.run_dir.name,
                'detzero_run':self.refined_path.parent.name,'data_root':str(self.data_root),
                'metadata_root':str(self.meta),'baseline':str(self.baseline_path),'refined':str(self.refined_path),
                'channels':CHANNELS,'point_format':'float32 little-endian ego xyz',
                'pair_policy':'Diagnostic nearest baseline XY; same source index for both methods, not evaluator TP/FP.',
                'gt_policy':'Same evaluation groups and class ranges, zero-point GT excluded by default.'}


class Handler(BaseHTTPRequestHandler):
    dataset = None

    def send(self, body, mime='application/json', status=200, cache=False):
        if isinstance(body,(dict,list)):
            body = json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
        headers = {}
        if mime=='application/json' and 'gzip' in self.headers.get('Accept-Encoding',''):
            body=gzip.compress(body);headers['Content-Encoding']='gzip'
        self.send_response(status)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','private, max-age=3600' if cache else 'no-cache')
        self.send_header('X-Content-Type-Options','nosniff')
        for k,v in headers.items():self.send_header(k,v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            url=urlparse(self.path);q=parse_qs(url.query)
            def value(name, default):return q.get(name,[default])[0]
            def index():return int(value('frame','0'))
            if url.path=='/api/info':return self.send(self.dataset.info())
            if url.path=='/api/frame':
                score=float(value('score','0.1'));radius=float(value('radius','1'))
                if not 0<=score<=1 or not .1<=radius<=5:raise ValueError('Invalid score/radius')
                return self.send(self.dataset.frame(index(),score,value('zero','0')=='1',radius))
            if url.path=='/api/points':
                limit=int(value('limit','18000'))
                if not 1000<=limit<=60000:raise ValueError('Point limit must be 1000..60000')
                return self.send(self.dataset.points(index(),limit),'application/octet-stream',cache=True)
            if url.path=='/api/image':
                width=int(value('width','0'))
                if width!=0 and not 160<=width<=640:raise ValueError('Invalid thumbnail width')
                return self.send(self.dataset.image(index(),value('camera','CAM_FRONT'),width),'image/jpeg',cache=True)
            names={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if url.path in names:
                file=STATIC/names[url.path]
                mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8'}[file.suffix]
                return self.send(file.read_bytes(),mime)
            self.send({'error':'Not found'},status=404)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            self.send({'error':str(exc)},status=400)
        except (BrokenPipeError,ConnectionResetError):
            pass
        except Exception as exc:
            print('Request failed:',repr(exc),flush=True)
            self.send({'error':'Unable to read this frame; check server log'},status=500)

    def log_message(self,fmt,*args):
        if ' 200 ' not in (fmt%args):super().log_message(fmt,*args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default=str(ROOT/'data/nuscenes_vf6_01_adapted'))
    p.add_argument('--run-dir',default=str(ROOT/'outputs/vf-car-truck-sweeps2-01'))
    p.add_argument('--refined',help='Detection JSON of another source-index-preserving DetZero run')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8765)
    args=p.parse_args()
    Handler.dataset=Dataset(args.data_root,args.run_dir,args.refined)
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'VF Object Inspector: http://{args.host}:{args.port}',flush=True)
    print('DetZero:',Handler.dataset.refined_path,flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
