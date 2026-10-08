import os
import argparse
import subprocess
import sys
import json
import shutil
import colorsys
import cv2
import numpy as np
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

OBJECT_PALETTE = {
    'car': (0, 158, 255),               # BGR orange/gold
    'truck': (71, 99, 255),
    'construction_vehicle': (122, 150, 233),
    'bus': (0, 69, 255),
    'trailer': (0, 140, 255),
    'barrier': (144, 128, 112),
    'motorcycle': (99, 61, 255),
    'bicycle': (60, 20, 220),
    'pedestrian': (230, 0, 0),          # blue in BGR
    'traffic_cone': (79, 79, 47),
}


def draw_track_label(canvas, text, x, y, color):
    """Draw only the colored ID at the final video resolution, without a background."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(4, min(int(x), canvas.shape[1] - width - 5))
    y = max(height + 5, min(int(y), canvas.shape[0] - baseline - 5))
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def make_track_labels(document):
    """Use distinct display IDs while retaining original tracker IDs in a sidecar."""
    ids = set()
    for boxes in document['results'].values():
        for box in boxes:
            tid = box.get('tracking_id')
            if not isinstance(tid, str) or not tid.strip():
                raise ValueError('Tracking video requires a nonempty tracker ID on every box')
            ids.add(tid)
    return {tid: f'T{i + 1:03d}' for i, tid in enumerate(sorted(ids))}


def make_track_colors(labels):
    """Assign distinct bright BGR colors, stable across frames and camera/BEV views."""
    colors, used = {}, set()
    for i, display_id in enumerate(sorted(set(labels.values()))):
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.65 + 0.1 * (i % 3)
        value = 0.9 + 0.05 * ((i // 3) % 3)
        while True:
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            color = tuple(round(channel * 255) for channel in reversed(rgb))
            if color not in used:
                break
            hue = (hue + 0.001) % 1.0
        colors[display_id] = color
        used.add(color)
    return colors


def select_scenes(nusc, documents):
    """Require identical full-scene coverage across the before/after inputs."""
    samples = {s['token']: s for s in nusc.sample}
    tokens = set(documents[0]['results'])
    if not tokens or not tokens <= set(samples):
        raise ValueError('Prediction sample tokens are empty or do not belong to this dataset')
    if any(set(d['results']) != tokens for d in documents[1:]):
        raise ValueError('Before/after JSONs must contain the same sample tokens, including empty frames')
    scene_tokens = {samples[t]['scene_token'] for t in tokens}
    expected = {t for t, s in samples.items() if s['scene_token'] in scene_tokens}
    if tokens != expected:
        raise ValueError('Video inputs must cover all frames of each selected scene')
    for document in documents:
        if document.get('coordinate_mode') == 'ego_uncompensated_diagnostic':
            raise ValueError('This renderer projects global boxes; ego diagnostics are not supported')
        for token, boxes in document['results'].items():
            if any(b.get('sample_token') != token for b in boxes):
                raise ValueError('Box sample_token differs from its frame')
    return [s for s in nusc.scene if s['token'] in scene_tokens]


def render_camera_view(img: np.ndarray, boxes: list, cam_cs: dict, cam_pose: dict,
                       lidar_cs: dict, lidar_pose: dict,
                       target_w: int = 480, target_h: int = 270, cam_name: str = "",
                       is_tracking: bool = False, track_labels=None) -> np.ndarray:
    canvas = img.copy()
    orig_h, orig_w = canvas.shape[:2]
    tracking_tags = []
    track_colors = make_track_colors(track_labels or {
        b['tracking_id']: b['tracking_id'] for b in boxes if b.get('tracking_id')
    }) if is_tracking else {}
    intrinsic = np.array(cam_cs['camera_intrinsic'])

    l2e_t = np.array(lidar_cs['translation'])
    l2e_r = Quaternion(lidar_cs['rotation'])
    e2g_t = np.array(lidar_pose['translation'])
    e2g_r = Quaternion(lidar_pose['rotation'])

    g2e_t = np.array(cam_pose['translation'])
    g2e_r = Quaternion(cam_pose['rotation'])
    e2c_t = np.array(cam_cs['translation'])
    e2c_r = Quaternion(cam_cs['rotation'])

    for item in boxes:
        name = item.get('tracking_name', item.get('detection_name', 'car'))
        score = float(item.get('tracking_score', item.get('detection_score', 0.0)))
        center = np.array(item['translation'], dtype=float)
        b = Box(center, item['size'], Quaternion(item['rotation']), name=name, score=score)

        # LiDAR frame
        b.translate(-e2g_t)
        b.rotate(e2g_r.inverse)
        b.translate(-l2e_t)
        b.rotate(l2e_r.inverse)

        corners_lidar = b.corners()
        pts_ego_l = l2e_r.rotation_matrix @ corners_lidar + l2e_t[:, None]
        pts_glob = e2g_r.rotation_matrix @ pts_ego_l + e2g_t[:, None]
        pts_ego_c = g2e_r.rotation_matrix.T @ (pts_glob - g2e_t[:, None])
        corners_cam = e2c_r.rotation_matrix.T @ (pts_ego_c - e2c_t[:, None])

        if np.any(corners_cam[2, :] <= 0.2):
            continue

        corners_img = view_points(corners_cam, intrinsic, normalize=True)[:2, :]

        if (np.all(corners_img[0, :] < 0) or np.all(corners_img[0, :] >= orig_w) or
            np.all(corners_img[1, :] < 0) or np.all(corners_img[1, :] >= orig_h)):
            continue

        color = OBJECT_PALETTE.get(name, (0, 255, 0))
        tid = item.get('tracking_id', '')
        short_id = track_labels.get(tid, tid) if track_labels is not None else tid

        def pt(idx):
            return (int(round(corners_img[0, idx])), int(round(corners_img[1, idx])))

        for i in range(4):
            cv2.line(canvas, pt(i), pt(i + 4), color, 3, cv2.LINE_AA)
            cv2.line(canvas, pt(i), pt((i + 1) % 4), color, 3, cv2.LINE_AA)
            cv2.line(canvas, pt(i + 4), pt(((i + 1) % 4) + 4), color, 3, cv2.LINE_AA)

        min_y_idx = int(np.argmin(corners_img[1, :]))
        tag_x = int(np.clip(corners_img[0, min_y_idx] - 12, 10, orig_w - 95))
        tag_y = int(np.clip(corners_img[1, min_y_idx] - 6, 22, orig_h - 10))

        if is_tracking and short_id:
            tag_text = short_id
            tracking_tags.append((tag_text, tag_x * target_w / orig_w,
                                  tag_y * target_h / orig_h, track_colors[short_id]))
            continue
        else:
            tag_text = f"{name[:3].capitalize()} {score:.2f}"

        (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (tag_x - 2, tag_y - th - 4), (tag_x + tw + 4, tag_y + 3), color, -1)
        cv2.putText(canvas, tag_text, (tag_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    canvas = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    for text, x, y, color in tracking_tags:
        draw_track_label(canvas, text, x, y, color)
    cv2.rectangle(canvas, (5, 5), (5 + len(cam_name) * 8 + 14, 22), (20, 20, 25), -1)
    cv2.putText(canvas, cam_name, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_lidar_bev(lidar_path: str, boxes: list, lidar_cs: dict, lidar_pose: dict,
                     size: int = 540, pc_range: float = 50.0,
                     track_history: dict = None, is_tracking: bool = False,
                     track_labels=None) -> np.ndarray:
    bev = np.full((size, size, 3), 15, dtype=np.uint8)
    center_px = (size // 2, size // 2)
    track_colors = make_track_colors(track_labels or {
        b['tracking_id']: b['tracking_id'] for b in boxes if b.get('tracking_id')
    }) if is_tracking else {}

    for r in [15, 30, 45]:
        r_px = int(r / pc_range * (size // 2))
        cv2.circle(bev, center_px, r_px, (35, 38, 44), 1, cv2.LINE_AA)

    if not os.path.isfile(lidar_path):
        raise FileNotFoundError(lidar_path)
    if os.path.exists(lidar_path):
        raw = np.fromfile(lidar_path, dtype=np.float32)
        if raw.size % 5:
            raise ValueError('LiDAR file must contain five float32 values per point')
        pts = raw.reshape(-1, 5)[:, :3]
        pts = pts[np.isfinite(pts).all(axis=1)]
        mask = (np.abs(pts[:, 0]) <= pc_range) & (np.abs(pts[:, 1]) <= pc_range)
        pts = pts[mask]
        u = np.clip(((pts[:, 0] + pc_range) / (2 * pc_range) * (size - 1)).astype(np.int32), 0, size - 1)
        v = np.clip(((pc_range - pts[:, 1]) / (2 * pc_range) * (size - 1)).astype(np.int32), 0, size - 1)
        bev[v, u] = (220, 230, 240)

    # Render trajectory trails in tracking mode
    if is_tracking and track_history is not None:
        l_e2g_t = np.array(lidar_pose['translation'])
        l_e2g_r = Quaternion(lidar_pose['rotation'])
        l_l2e_t = np.array(lidar_cs['translation'])
        l_l2e_r = Quaternion(lidar_cs['rotation'])

        for tid, hist in track_history.items():
            if len(hist) < 2:
                continue
            trail_pts = []
            for g_pt in hist[-25:]: # last 25 frames (~5s)
                p_rel = g_pt - l_e2g_t
                p_ego = l_e2g_r.inverse.rotate(p_rel)
                p_lidar = l_l2e_r.inverse.rotate(p_ego - l_l2e_t)
                if abs(p_lidar[0]) <= pc_range and abs(p_lidar[1]) <= pc_range:
                    u = int((p_lidar[0] + pc_range) / (2 * pc_range) * (size - 1))
                    v = int((pc_range - p_lidar[1]) / (2 * pc_range) * (size - 1))
                    trail_pts.append((u, v))
            if len(trail_pts) >= 2:
                for k in range(len(trail_pts) - 1):
                    alpha = (k + 1) / len(trail_pts)
                    color = (int(255 * alpha), int(200 * alpha), int(50 * alpha)) # cyan/yellow
                    cv2.line(bev, trail_pts[k], trail_pts[k + 1], color, 1, cv2.LINE_AA)

    for item in boxes:
        name = item.get('tracking_name', item.get('detection_name', 'car'))
        b = Box(item['translation'], item['size'], Quaternion(item['rotation']), name=name)
        b.translate(-np.array(lidar_pose['translation']))
        b.rotate(Quaternion(lidar_pose['rotation']).inverse)
        b.translate(-np.array(lidar_cs['translation']))
        b.rotate(Quaternion(lidar_cs['rotation']).inverse)

        c = b.corners()[:2, [2, 3, 7, 6]]
        if np.all(np.abs(c) > pc_range * 1.2):
            continue

        u = ((c[0, :] + pc_range) / (2 * pc_range) * (size - 1)).astype(np.int32)
        v = ((pc_range - c[1, :]) / (2 * pc_range) * (size - 1)).astype(np.int32)
        poly = np.stack([u, v], axis=1)

        color = OBJECT_PALETTE.get(name, (0, 255, 0))
        cv2.polylines(bev, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        front_mid = ((u[0] + u[1]) // 2, (v[0] + v[1]) // 2)
        center = (int(u.mean()), int(v.mean()))
        cv2.line(bev, center, front_mid, color, 2, cv2.LINE_AA)

        if is_tracking and item.get('tracking_id', ''):
            tid = item['tracking_id']
            short_id = track_labels.get(tid, tid) if track_labels is not None else tid
            draw_track_label(bev, short_id, int(u.mean()) - 8, int(v.mean()) - 6,
                             track_colors[short_id])

    cv2.rectangle(bev, (5, 5), (140, 22), (20, 20, 25), -1)
    cv2.putText(bev, "LiDAR BEV (50m)", (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)
    return bev


def generate_video(model_title: str, results_json: str, output_name: str,
                   nusc, frames, is_tracking: bool = False, min_score: float = 0.25,
                   output_dir="outputs/vf6_01_5hz/videos", fps=5.0,
                   track_labels=None):
    print("=" * 60)
    print(f"Generating Video: {model_title}")
    print(f"Source JSON: {results_json}")
    print(f"Output File: {output_name}")
    print("=" * 60)

    document = json.loads(Path(results_json).read_text())
    data = document['results']
    if is_tracking and track_labels is None:
        track_labels = make_track_labels(document)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_video = out_dir / f"{output_name}_raw.mp4"
    fixed_video = out_dir / f"{output_name}.mp4"

    cam_layout = [
        [("CAM_FRONT_LEFT", "Front-Left"), ("CAM_FRONT", "Front"), ("CAM_FRONT_RIGHT", "Front-Right")],
        [("CAM_BACK_LEFT", "Back-Left"), ("CAM_BACK", "Back"), ("CAM_BACK_RIGHT", "Back-Right")]
    ]
    target_cam_w, target_cam_h = 480, 270
    lidar_size = 540

    track_history = {} if is_tracking else None
    vw = None

    for frm_idx, (_, token) in enumerate(frames):
        sample = nusc.get("sample", token)
        all_boxes = data[token]
        score_key = 'tracking_score' if is_tracking else 'detection_score'
        boxes = [b for b in all_boxes if b.get(score_key, 0.0) >= min_score]

        # Update track history in global coordinates
        if is_tracking:
            for b in boxes:
                tid = b.get('tracking_id', '')
                if tid:
                    track_history.setdefault(tid, []).append(np.array(b['translation']))

        lidar_sd_token = sample["data"]["LIDAR_TOP"]
        lidar_sd = nusc.get("sample_data", lidar_sd_token)
        lidar_path = nusc.get_sample_data_path(lidar_sd_token)
        lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        cam_rows = []
        for r in range(2):
            row_imgs = []
            for channel, display_name in cam_layout[r]:
                sd_token = sample["data"][channel]
                sd_record = nusc.get("sample_data", sd_token)
                cam_img_path = nusc.get_sample_data_path(sd_token)
                cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
                pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

                raw_img = cv2.imread(str(cam_img_path))
                if raw_img is None:
                    raise FileNotFoundError(cam_img_path)
                cam_canvas = render_camera_view(
                    raw_img, boxes, cs_record, pose_record,
                    lidar_cs=lidar_cs, lidar_pose=lidar_pose,
                    target_w=target_cam_w, target_h=target_cam_h,
                    cam_name=display_name, is_tracking=is_tracking, track_labels=track_labels
                )
                row_imgs.append(cam_canvas)
            cam_rows.append(np.hstack(row_imgs))
        cameras_canvas = np.vstack(cam_rows)

        bev_canvas = render_lidar_bev(
            str(lidar_path), boxes, lidar_cs, lidar_pose,
            size=lidar_size, pc_range=50.0,
            track_history=track_history, is_tracking=is_tracking, track_labels=track_labels
        )

        full_frame = np.hstack([cameras_canvas, bev_canvas])

        # Top banner with Title and Info
        banner = np.full((36, full_frame.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(banner, f"{fps:.2f} Hz | {model_title}", (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        status_txt = f"Frame [{frm_idx + 1:03d}/{len(frames):03d}] | Active Objects: {len(boxes)}"
        cv2.putText(banner, status_txt, (full_frame.shape[1] - 380, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        final_composite = np.vstack([banner, full_frame])

        if vw is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(raw_video), fourcc, fps, (final_composite.shape[1], final_composite.shape[0]))

        if not vw.isOpened():
            raise RuntimeError("VideoWriter failed to open")
        vw.write(final_composite)
        print(f"  [{model_title}] Frame {frm_idx + 1:03d}/{len(frames):03d} - Boxes: {len(boxes)}", end="\r")

    if vw is not None:
        vw.release()
    print(f"\n  -> Transcoding with ffmpeg: {fixed_video}...")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video),
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(fixed_video)], check=True)
    if raw_video.exists():
        raw_video.unlink()
    print(f"  -> SUCCESS: Saved {fixed_video} ({fixed_video.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Render JSON geometric centers, with no extra height offset.")
    parser.add_argument('--data-root', default='data/nuscenes_vf6_01_5hz')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--refined')
    parser.add_argument('--tracking')
    parser.add_argument('--output-dir', default='outputs/vf6_01_5hz/videos')
    parser.add_argument('--min-score', type=float, default=.25)
    args = parser.parse_args()
    if not 0 <= args.min_score <= 1:
        parser.error('--min-score must be in [0, 1]')
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg is required to encode H.264 videos; check the container before rendering')
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    configs = [('BEVFusion', args.baseline, 'bevfusion', False),
               ('BEVFusion + DetZero', args.refined, 'refined', False),
               ('BEVFusion + DetZero (track IDs)', args.tracking, 'tracking', True)]
    documents = {suffix: json.loads(Path(file).read_text())
                 for _, file, suffix, _ in configs if file}
    scenes = select_scenes(nusc, list(documents.values()))
    labels = make_track_labels(documents['tracking']) if args.tracking else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if labels is not None:
        (output_dir / 'tracking_id_map.json').write_text(json.dumps(labels, indent=2))
        (output_dir / 'tracking_id_colors.json').write_text(json.dumps(
            {'color_format': 'BGR', 'colors': make_track_colors(labels)}, indent=2))
    for scene in scenes:
        samples = sorted((s for s in nusc.sample if s['scene_token'] == scene['token']), key=lambda s: s['timestamp'])
        frames = [(s['timestamp'], s['token']) for s in samples]
        fps = 1e6 / float(np.median(np.diff([f[0] for f in frames]))) if len(frames) > 1 else 5.
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Scene timestamps must have positive frame intervals')
        for title, path, suffix, tracking in configs:
            if path:
                generate_video(title, path, scene['name'] + '_' + suffix, nusc, frames,
                               tracking, args.min_score, args.output_dir, fps,
                               labels if tracking else None)


if __name__ == '__main__':
    main()
