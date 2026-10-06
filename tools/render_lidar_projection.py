"""Export clean camera images and depth-colored keyframe LiDAR overlays."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion


from nuscenes.utils.data_classes import Box


def transform(record):
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    matrix[:3, 3] = record['translation']
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--frame', type=int, default=1, help='One-based timestamp-sorted sample index')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-depth', type=float, default=60.)
    parser.add_argument('--point-size', type=int, default=2, help='Radius of projected LiDAR points in pixels')
    parser.add_argument('--draw-boxes', action='store_true', help='Draw 3D bounding boxes from ground truth annotations')
    args = parser.parse_args()
    if args.max_depth <= 0:
        parser.error('--max-depth must be positive')
    root = Path(args.data_root)
    def table(name):
        return json.loads((root / args.version / (name + '.json')).read_text())
    samples = sorted(table('sample'), key=lambda s: (s['timestamp'], s['token']))
    if not 1 <= args.frame <= len(samples):
        parser.error('--frame outside dataset range')
    sample = samples[args.frame - 1]
    sensors = {r['token']: r for r in table('sensor')}
    calibs = {r['token']: r for r in table('calibrated_sensor')}
    poses = {r['token']: r for r in table('ego_pose')}
    records = {sensors[calibs[r['calibrated_sensor_token']]['sensor_token']]['channel']: r
               for r in table('sample_data')
               if r['sample_token'] == sample['token'] and r['is_key_frame']}
    lidars = [r for ch, r in records.items() if sensors[calibs[r['calibrated_sensor_token']]['sensor_token']]['modality'] == 'lidar']
    if len(lidars) != 1:
        raise ValueError('Expected exactly one keyframe LiDAR')
    lidar = lidars[0]
    points = np.fromfile(root / lidar['filename'], dtype=np.float32).reshape(-1, 5)[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    points = np.column_stack((points, np.ones(len(points))))
    global_from_lidar = transform(poses[lidar['ego_pose_token']]) @ transform(calibs[lidar['calibrated_sensor_token']])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {'data_root': str(root), 'frame': args.frame, 'sample_token': sample['token'],
              'sample_timestamp': sample['timestamp'], 'lidar_file': lidar['filename'],
              'depth_color': 'TURBO: blue near, red far; camera depth 0..max_depth m',
              'max_depth_m': args.max_depth, 'keyframe_only': True, 'cameras': {}}
    anns = [a for a in table('sample_annotation') if a['sample_token'] == sample['token']] if args.draw_boxes else []
    camera_overlays = {}
    for channel, camera in sorted(records.items()):
        if not channel.startswith('CAM_'):
            continue
        image = cv2.imread(str(root / camera['filename']))
        if image is None:
            raise FileNotFoundError(camera['filename'])
        calib = calibs[camera['calibrated_sensor_token']]
        global_from_camera = transform(poses[camera['ego_pose_token']]) @ transform(calib)
        camera_from_lidar = np.linalg.inv(global_from_camera) @ global_from_lidar
        xyz = (camera_from_lidar @ points.T)[:3].T
        xyz = xyz[(xyz[:, 2] > .2) & (xyz[:, 2] <= args.max_depth)]
        projected = xyz @ np.asarray(calib['camera_intrinsic']).T
        uv = projected[:, :2] / projected[:, 2:3]
        h, w = image.shape[:2]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        uv, depth = uv[inside], xyz[inside, 2]
        overlay = image.copy()
        if len(depth):
            colors = cv2.applyColorMap(np.clip(depth / args.max_depth * 255, 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0]
            for i in np.argsort(depth)[::-1]:
                cv2.circle(overlay, tuple(np.rint(uv[i]).astype(int)), args.point_size, tuple(int(c) for c in colors[i]), -1, cv2.LINE_AA)

        if args.draw_boxes and len(anns) > 0:
            cam_t = np.array(calib['translation'])
            cam_r = Quaternion(calib['rotation'])
            ego_t = np.array(poses[camera['ego_pose_token']]['translation'])
            ego_r = Quaternion(poses[camera['ego_pose_token']]['rotation'])
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
            for a in anns:
                box = Box(a['translation'], a['size'], Quaternion(a['rotation']))
                box.translate(-ego_t)
                box.rotate(ego_r.inverse)
                box.translate(-cam_t)
                box.rotate(cam_r.inverse)
                corners = box.corners()
                if np.all(corners[2, :] > 0.5):
                    pts_b = np.asarray(calib['camera_intrinsic']) @ corners
                    uv_b = pts_b[:2] / pts_b[2]
                    if np.any((uv_b[0] >= 0) & (uv_b[0] < w)) and np.any((uv_b[1] >= 0) & (uv_b[1] < h)):
                        uv_b = uv_b.astype(int)
                        for e1, e2 in edges:
                            cv2.line(overlay, (uv_b[0, e1], uv_b[1, e1]), (uv_b[0, e2], uv_b[1, e2]), (0, 255, 0), 2, cv2.LINE_AA)

        camera_overlays[channel] = overlay
        for suffix, canvas in [('camera', image), ('lidar_overlay', overlay)]:
            if not cv2.imwrite(str(output / f'{channel}_{suffix}.png'), canvas):
                raise OSError('Image write failed')
        panels = []
        for title, canvas in [('CAMERA ONLY', image), ('LIDAR PROJECTION | blue: near, red: far', overlay)]:
            panel = cv2.copyMakeBorder(canvas, 64, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
            cv2.putText(panel, f'{channel} | Frame {args.frame} | {title}', (15, 42), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)
        if not cv2.imwrite(str(output / f'{channel}_comparison.png'), cv2.hconcat(panels)):
            raise OSError('Image write failed')
        report['cameras'][channel] = {'image_file': camera['filename'], 'projected_points': len(depth),
                                      'camera_minus_lidar_ms': (camera['timestamp'] - lidar['timestamp']) / 1000}
        print(channel, len(depth), 'points')

    # Build 6-camera mosaic grid
    grid_channels = [
        ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    ]
    if all(ch in camera_overlays for row in grid_channels for ch in row):
        grid_rows = []
        for row in grid_channels:
            row_imgs = []
            for ch in row:
                im = camera_overlays[ch]
                # Add channel label banner
                im_labeled = cv2.copyMakeBorder(im, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                cv2.putText(im_labeled, ch, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                # Resize for compact display
                im_resized = cv2.resize(im_labeled, (800, int(800 * im_labeled.shape[0] / im_labeled.shape[1])))
                row_imgs.append(im_resized)
            grid_rows.append(cv2.hconcat(row_imgs))
        mosaic = cv2.vconcat(grid_rows)
        cv2.imwrite(str(output / 'all_cameras_lidar_overlay.png'), mosaic)
        print('Saved 6-camera mosaic:', output / 'all_cameras_lidar_overlay.png')

    (output / 'projection_report.json').write_text(json.dumps(report, indent=2))
    print('Saved:', output)


if __name__ == '__main__':
    main()
