import os
import json
import cv2
import mmcv
import numpy as np
from pathlib import Path

# Đường dẫn
root_path = Path('data/nuscenes')
val_info_path = root_path / 'nuscenes_infos_val.pkl'
sample_json_path = root_path / 'v1.0-mini/sample.json'
viz_dir = Path('outputs/mini-viz')
output_dir = Path('outputs/demo-videos')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Tạo từ điển map: sample_token -> scene_token từ file json gốc
samples_raw = json.loads(sample_json_path.read_text())
token_to_scene = {s['token']: s['scene_token'] for s in samples_raw}

# 2. Đọc metadata val và phân nhóm theo đúng scene_token
val_data = mmcv.load(str(val_info_path))
scenes = {}

for info in val_data['infos']:
    token = info['token']
    timestamp = info['timestamp']
    scene_token = token_to_scene.get(token, 'unknown_scene')
    file_prefix = f"{timestamp}-{token}"
    
    if scene_token not in scenes:
        scenes[scene_token] = []
    scenes[scene_token].append((timestamp, file_prefix))

# Sắp xếp các frame theo thời gian tăng dần
for s_token in scenes:
    scenes[s_token].sort(key=lambda x: x[0])

print(f"Tìm thấy chính xác {len(scenes)} scenes trong tập validation:")
for idx, (s_token, frames) in enumerate(scenes.items()):
    print(f" - Scene {idx + 1} (token: {s_token[:8]}...): {len(frames)} frames")

# Layout camera
cam_order = [
    ['camera-1', 'camera-0', 'camera-2'],
    ['camera-4', 'camera-3', 'camera-5']
]

# 3. Xuất video cho từng scene
for idx, (s_token, frames) in enumerate(scenes.items()):
    video_path = output_dir / f"scene_{idx + 1}.mp4"
    vw = None
    target_cam_h, target_cam_w = 270, 480
    
    for _, prefix in frames:
        # Gom 6 camera
        rows = []
        for r in range(2):
            row_imgs = []
            for c in range(3):
                cam_name = cam_order[r][c]
                img_p = viz_dir / cam_name / f"{prefix}.png"
                if img_p.exists():
                    im = cv2.imread(str(img_p))
                    im = cv2.resize(im, (target_cam_w, target_cam_h))
                else:
                    im = np.zeros((target_cam_h, target_cam_w, 3), dtype=np.uint8)
                row_imgs.append(im)
            rows.append(np.hstack(row_imgs))
        cams_canvas = np.vstack(rows)

        # LiDAR BEV
        lidar_p = viz_dir / 'lidar' / f"{prefix}.png"
        total_h = cams_canvas.shape[0]
        if lidar_p.exists():
            lidar_im = cv2.imread(str(lidar_p))
            lidar_im = cv2.resize(lidar_im, (total_h, total_h))
        else:
            lidar_im = np.zeros((total_h, total_h, 3), dtype=np.uint8)

        full_frame = np.hstack([cams_canvas, lidar_im])

        if vw is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(str(video_path), fourcc, 2.0, (full_frame.shape[1], full_frame.shape[0]))

        vw.write(full_frame)

    if vw is not None:
        vw.release()
    print(f"Đã xuất xong: {video_path}")