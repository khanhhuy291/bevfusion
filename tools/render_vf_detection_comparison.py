"""Render real VF camera/BEV comparisons, using global geometric-center boxes.

No inference or generated imagery. Camera projections use the camera's own
timestamp pose; BEV uses the keyframe ego pose. Original outputs are read-only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

COLORS = {'GT': (80, 215, 70), 'BEVFusion': (0, 155, 255),
          'BEVFusion + DetZero': (255, 145, 35)}  # BGR
MAPPING = {'vehicle.car': 'car', 'vehicle.truck': 'car',
           'vehicle.motorcycle': 'motorcycle', 'human.pedestrian.adult': 'pedestrian'}
RANGE = {'car': 50., 'motorcycle': 40., 'pedestrian': 40.}
CLASS_ABBR = {'car': 'C', 'motorcycle': 'M', 'pedestrian': 'P'}


def read(path):
    return json.loads(path.read_text())


def text(canvas, value, origin, color=(230, 235, 240), scale=.6):
    cv2.putText(canvas, value, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, value, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def corners(record):
    return Box(record['translation'], record['size'], Quaternion(record['rotation'])).corners()


def project(record, calibration, pose):
    points = corners(record)
    points = Quaternion(pose['rotation']).rotation_matrix.T @ (points - np.array(pose['translation'])[:, None])
    points = Quaternion(calibration['rotation']).rotation_matrix.T @ (points - np.array(calibration['translation'])[:, None])
    if np.any(points[2] <= .2):
        return None
    image = np.array(calibration['camera_intrinsic']) @ points
    return (image[:2] / image[2]).T


def draw_camera(image, records, color, calibration, pose, focus_key=None):
    canvas = image.copy()
    width, height = image.shape[1], image.shape[0]
    for record in records:
        xy = project(record, calibration, pose)
        if xy is None or (xy[:, 0].max() < 0 or xy[:, 0].min() >= width
                          or xy[:, 1].max() < 0 or xy[:, 1].min() >= height):
            continue
        pts = np.rint(np.clip(xy, -10000, 10000)).astype(int)
        focused = record['_key'] == focus_key
        for i in range(4):
            for a, b in [(i, i+4), (i, (i+1)%4), (i+4, (i+1)%4+4)]:
                ok, p1, p2 = cv2.clipLine((0, 0, width, height), tuple(pts[a]), tuple(pts[b]))
                if ok:
                    cv2.line(canvas, p1, p2, color, 4 if focused else 2, cv2.LINE_AA)
        origin = (int(np.clip(xy[:,0].min(), 3, width-90)), int(np.clip(xy[:,1].min()-5, 15, height-5)))
        label = ('F ' if focused else '') + CLASS_ABBR[record['_class']]
        if record.get('detection_score') is not None:
            label += f" {record['detection_score']:.2f}"
        text(canvas, label, origin, color, .45)
    return canvas


def to_ego(points, pose):
    return (points - np.array(pose['translation'])) @ Quaternion(pose['rotation']).rotation_matrix


def draw_bev(points, records, color, pose, focus_key=None, size=640, bounds=None):
    canvas = np.full((size, size, 3), 22, np.uint8)
    # Ego X forward/up, Y left/left. Equal metric scales on both axes.
    xlow, xhigh, ylow, yhigh = bounds if bounds else (-40., 40., -40., 40.)
    def pixel(xyz):
        return np.column_stack([(yhigh-xyz[:,1])/(yhigh-ylow)*(size-1),
                                (xhigh-xyz[:,0])/(xhigh-xlow)*(size-1)])
    step=1 if xhigh-xlow<10 else 10
    for x in np.arange(np.ceil(xlow/step)*step,xhigh+1e-6,step):
        row = int(pixel(np.array([[x,0,0]]))[0,1])
        cv2.line(canvas, (0,row), (size-1,row), (48,48,48), 1)
        text(canvas, f'{x}m', (5,max(16,row-3)), scale=.4)
    for y in np.arange(np.ceil(ylow/step)*step,yhigh+1e-6,step):
        col = int(pixel(np.array([[0,y,0]]))[0,0])
        cv2.line(canvas, (col,0), (col,size-1), (48,48,48), 1)
    xy = np.rint(pixel(points)).astype(int)
    kept = (xy[:,0]>=0)&(xy[:,0]<size)&(xy[:,1]>=0)&(xy[:,1]<size)
    xy=xy[kept]
    canvas[xy[:,1],xy[:,0]]=(100,100,100)
    for record in records:
        local=to_ego(corners(record).T,pose)
        # nuScenes corners 2,3,7,6 form the bottom footprint; 0..3 is a face.
        outline=np.rint(pixel(local[[2,3,7,6]])).astype(int)
        focused=record['_key']==focus_key
        cv2.polylines(canvas,[outline],True,color,3 if focused else 1,cv2.LINE_AA)
        center=to_ego(np.array(record['translation'])[None],pose)
        p=np.rint(pixel(center)[0]).astype(int)
        if 0<=p[0]<size and 0<=p[1]<size:
            text(canvas,('F:' if focused else '')+CLASS_ABBR[record['_class']],
                 (int(p[0])+2,int(p[1])-2),color,.4)
    origin=np.rint(pixel(np.array([[0.,0.,0.]]))[0]).astype(int)
    cv2.drawMarker(canvas,tuple(origin),(255,255,255),cv2.MARKER_TRIANGLE_UP,12,2)
    text(canvas,'Forward +X / Left +Y | C=car, M=motorcycle, P=pedestrian', (10,size-12),scale=.39)
    return canvas


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',default='outputs/vf-car-truck-sweeps2-01')
    parser.add_argument('--sensor-root',default='data/nuscenes_vf6_01_adapted')
    parser.add_argument('--frames',type=int,nargs='+',default=[2,30,80,142])
    parser.add_argument('--min-score',type=float,default=.1)
    parser.add_argument('--output-dir')
    args=parser.parse_args()
    run=Path(args.run_dir);source=Path(args.sensor_root)
    out=Path(args.output_dir) if args.output_dir else run/'visual_comparison'
    out.mkdir(parents=True,exist_ok=True)
    meta=run/'dataset_local/v1.0-mini'
    def table(name):return {r['token']:r for r in read(meta/(name+'.json'))}
    samples=table('sample');sd=table('sample_data');calibs=table('calibrated_sensor')
    egos=table('ego_pose');sensors=table('sensor');instances=table('instance');categories=table('category')
    annotations=table('sample_annotation')
    ordered=sorted(samples.values(),key=lambda r:r['timestamp'])
    sensors_by_sample={t:{} for t in samples}
    for r in sd.values():
        if r['is_key_frame']:
            sensors_by_sample[r['sample_token']][sensors[calibs[r['calibrated_sensor_token']]['sensor_token']]['channel']]=r
    paths={'BEVFusion':run/'baseline/results_nusc_center_corrected.json',
           'BEVFusion + DetZero':run/'grm_prm/results_nusc_detzero_refined.json'}
    predictions={name:read(path)['results'] for name,path in paths.items()}
    assert all(set(p)==set(samples) for p in predictions.values())
    pairs=list(csv.DictReader((run/'analysis/fixed_gt_pairs.csv').open()))
    focus_classes={2:'pedestrian',30:'motorcycle',80:'motorcycle',142:'pedestrian'}
    manifest={'min_display_score':args.min_score,'frame_indexing':'zero-based timestamp order',
              'comparison':'Original full GRM+PRM min_score=0.1, not threshold-sweep best result',
              'GT_policy':'Evaluable GT: nonzero partner num_lidar_pts, shared class/range filters; car includes truck',
              'colors':{'GT':'green','BEVFusion':'orange','BEVFusion + DetZero':'blue'},
              'projection':'Global geometric centers; camera uses own timestamp ego pose; BEV uses keyframe ego axes',
              'hashes':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},'frames':[]}
    contact=[]
    for index in args.frames:
        sample=ordered[index];token=sample['token'];records=sensors_by_sample[token]
        lidar=records['LIDAR_TOP'];ego=egos[lidar['ego_pose_token']];calib=calibs[lidar['calibrated_sensor_token']]
        origin=np.array(ego['translation'])
        groups={'GT':[]}
        for annotation in annotations.values():
            if annotation['sample_token']!=token or annotation['num_lidar_pts']==0:continue
            cls=MAPPING.get(categories[instances[annotation['instance_token']]['category_token']]['name'])
            if cls not in RANGE or np.linalg.norm((np.array(annotation['translation'])-origin)[:2])>=RANGE[cls]:continue
            groups['GT'].append(dict(annotation,_class=cls,_key=annotation['token']))
        for method in paths:
            groups[method]=[]
            for i,record in enumerate(predictions[method][token]):
                cls='car' if record['detection_name']=='truck' else record['detection_name']
                if cls not in RANGE or record['detection_score']<args.min_score:continue
                if np.linalg.norm((np.array(record['translation'])-origin)[:2])>=RANGE[cls]:continue
                groups[method].append(dict(record,_class=cls,_key=i))
        selected=[r for r in pairs if int(r['frame'])==index and r['class']==focus_classes.get(index,'car')
                  and float(r['score'])>=args.min_score]
        def visible_area(annotation):
            area=0.
            for channel,row in records.items():
                if not channel.startswith('CAM_'):continue
                xy=project(annotation,calibs[row['calibrated_sensor_token']],egos[row['ego_pose_token']])
                if xy is None:continue
                low=np.maximum(xy.min(axis=0),0);high=np.minimum(xy.max(axis=0),[row['width'],row['height']])
                area=max(area,float(np.prod(np.maximum(high-low,0))))
            return area
        visible=[r for r in selected if visible_area(annotations[r['gt_token']])>100]
        if visible:selected=visible
        focus=None
        if selected:
            key=lambda r:float(r['grm_prm_iou3d'])-float(r['baseline_iou3d'])
            focus=min(selected,key=key) if focus_classes.get(index)=='pedestrian' else max(selected,key=key)
        target=annotations[focus['gt_token']] if focus else (groups['GT'][0] if groups['GT'] else None)
        # Select a camera where the illustrative target is visible at useful size.
        scores=[]
        for channel,row in records.items():
            if not channel.startswith('CAM_'):continue
            xy=project(target,calibs[row['calibrated_sensor_token']],egos[row['ego_pose_token']]) if target else None
            area=0.
            if xy is not None:
                low=np.maximum(xy.min(axis=0),0);high=np.minimum(xy.max(axis=0),[row['width'],row['height']])
                area=float(np.prod(np.maximum(high-low,0)))
            scores.append((area,channel))
        camera=max(scores)[1];cam_sd=records[camera]
        image=cv2.imread(str(source/cam_sd['filename']))
        if image is None:raise FileNotFoundError(source/cam_sd['filename'])
        assert image.shape[:2]==(cam_sd['height'],cam_sd['width'])
        cs=calibs[cam_sd['calibrated_sensor_token']];cam_pose=egos[cam_sd['ego_pose_token']]
        raw=np.fromfile(source/lidar['filename'],dtype=np.float32).reshape(-1,5)[:,:3]
        points_ego=raw@Quaternion(calib['rotation']).rotation_matrix.T+np.array(calib['translation'])
        panels=[]
        for method in ['GT','BEVFusion','BEVFusion + DetZero']:
            color=COLORS[method]
            focus_key=focus['gt_token'] if method=='GT' and focus else (int(focus['source_index']) if focus else None)
            cam=draw_camera(image,groups[method],color,cs,cam_pose,focus_key)
            cam=cv2.resize(cam,(640,360),interpolation=cv2.INTER_AREA)
            bev=draw_bev(points_ego,groups[method],color,ego,focus_key)
            header=np.full((48,640,3),17,np.uint8);text(header,method,(15,31),color,.8)
            sub=np.full((30,640,3),17,np.uint8);text(sub,camera+' | same camera and BEV scale',(15,21),scale=.48)
            panels.append(np.vstack([header,cam,sub,bev]))
        canvas=np.hstack(panels)
        title=np.full((62,1920,3),12,np.uint8)
        text(title,f'Frame {index:03d} | VF adapted | display confidence >= {args.min_score:.2f}',(20,25),scale=.65)
        caption='GT / BEVFusion / full GRM+PRM | F = illustrative corresponding object'
        if focus:caption+=f" | {focus['class']} IoU 3D: {float(focus['baseline_iou3d']):.3f} -> {float(focus['grm_prm_iou3d']):.3f}"
        text(title,caption,(20,50),scale=.55)
        canvas=np.vstack([title,canvas])
        filename=f'frame_{index:03d}_comparison.png';cv2.imwrite(str(out/filename),canvas)
        contact.append(cv2.resize(canvas,(960,590),interpolation=cv2.INTER_AREA))
        frame_info={'index':index,'sample_token':token,'camera':camera,'image_file':filename,
                    'camera_source':str(source/cam_sd['filename']),
                    'GT_count':len(groups['GT']),'baseline_display_count':len(groups['BEVFusion']),
                    'refined_display_count':len(groups['BEVFusion + DetZero']),
                    'focus':focus}
        # Zoomed camera comparison with the identical crop across all methods.
        if focus and target:
            focus_center=to_ego(np.array(target['translation'])[None],ego)[0]
            bounds=(focus_center[0]-2.5,focus_center[0]+2.5,focus_center[1]-2.5,focus_center[1]+2.5)
            zoom_bev=[]
            for method in ['GT','BEVFusion','BEVFusion + DetZero']:
                key=focus['gt_token'] if method=='GT' else int(focus['source_index'])
                chosen=[r for r in groups[method] if r['_key']==key]
                panel=draw_bev(points_ego,chosen,COLORS[method],ego,key,bounds=bounds)
                header=np.full((45,640,3),17,np.uint8);text(header,method,(15,30),COLORS[method],.7)
                zoom_bev.append(np.vstack([header,panel]))
            filename=f'frame_{index:03d}_bev_zoom.png'
            cv2.imwrite(str(out/filename),np.hstack(zoom_bev))
            frame_info['bev_zoom_file']=filename
            xy=project(target,cs,cam_pose)
            if xy is not None:
                center=(xy.min(axis=0)+xy.max(axis=0))/2
                width=max(360.,np.ptp(xy[:,0])*3);height=width*.75
                left=int(np.clip(center[0]-width/2,0,max(0,image.shape[1]-width)))
                top=int(np.clip(center[1]-height/2,0,max(0,image.shape[0]-height)))
                right=min(image.shape[1],left+int(width));bottom=min(image.shape[0],top+int(height))
                zoom=[]
                for method in ['GT','BEVFusion','BEVFusion + DetZero']:
                    key=focus['gt_token'] if method=='GT' else int(focus['source_index'])
                    chosen=[r for r in groups[method] if r['_key']==key]
                    marked=draw_camera(image,chosen,COLORS[method],cs,cam_pose,key)[top:bottom,left:right]
                    panel=cv2.resize(marked,(640,480),interpolation=cv2.INTER_AREA)
                    header=np.full((45,640,3),17,np.uint8);text(header,method,(15,30),COLORS[method],.7)
                    zoom.append(np.vstack([header,panel]))
                filename=f'frame_{index:03d}_camera_zoom.png';cv2.imwrite(str(out/filename),np.hstack(zoom))
                frame_info['camera_zoom_file']=filename
        manifest['frames'].append(frame_info)
        print(index,camera,'focus',focus['class'] if focus else None)
    if len(contact)==4:
        cv2.imwrite(str(out/'contact_sheet.png'),np.vstack([np.hstack(contact[:2]),np.hstack(contact[2:])]))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'README.md').write_text(
        '# GT / BEVFusion / BEVFusion + DetZero visual comparison\n\n'
        'Real saved VF adapted camera images and LiDAR; no inference rerun. '
        'Green: GT. Orange: BEVFusion baseline after legacy center correction. Blue: original full GRM+PRM at input min_score=0.1.\n\n'
        'Frames are zero-based timestamp order. Both prediction displays use confidence >=0.1. '
        'Only car (including truck), motorcycle and pedestrian are shown, with the evaluator class/range/GT-zero-point policy. '
        'BEV is a top-down visualization of LiDAR/boxes in ego axes, not a neural-network feature map. '
        'Camera is selected per frame for target visibility. All three panels within a frame share the exact image, projection, and BEV scale.\n\n'
        'Frames 2 and 142 illustrate pedestrian regression; frames 30 and 80 illustrate motorcycle improvement. '
        'F marks a diagnostic same-class proximity pair, not independently verified object identity. '
        'These selected examples do not estimate population error rates. The manifest records sources, filters and hashes.\n')
    print('Saved:',out)


if __name__=='__main__':main()
