import os
import cv2
import mmcv
import numpy as np
from pathlib import Path

# Đường dẫn dữ liệu
val_info_path = 'data/nuscenes/nuscenes_infos_val.pkl'
viz_dir = Path('outputs/mini-viz')
output_dir = Path('outputs/demo-videos')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Đọc metadata để lấy thứ tự thời gian và phân nhóm theo scene
val_data = mmcv.load(val_info_path)
scenes = {}

for info in val_data['infos']:
    scene_token = info.get('scene_token', 'default_scene')
    timestamp = info['timestamp']
    token = info['token']
    file_prefix = f"{timestamp}-{token}"
    
    if scene_token not in scenes:
        scenes[scene_token] = []
    scenes[scene_token].append((timestamp, file_prefix))

# Sắp xếp các frame theo thời gian tăng dần trong từng scene
for s_token in scenes:
    scenes[s_token].sort(key=lambda x: x[0])

print(f"Tìm thấy {len(scenes)} scenes trong tập validation.")

# Định nghĩa thứ tự 6 camera theo layout:
# Top: Front-Left (1), Front (0), Front-Right (2)
# Bottom: Back-Left (4), Back (3), Back-Right (5)
cam_order = [
    ['camera-1', 'camera-0', 'camera-2'],
    ['camera-4', 'camera-3', 'camera-5']
]

# 2. Xử lý xuất video cho từng scene
for idx, (s_token, frames) in enumerate(scenes.items()):
    video_path = output_dir / f"scene_{idx + 1}.mp4"
    vw = None
    target_cam_h, target_cam_w = 270, 480  # Scale kích thước camera để video vừa chuẩn HD
    
    for _, prefix in frames:
        # Gom 6 ảnh camera
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
        cams_canvas = np.vstack(rows)  # Kích thước: (540, 1440, 3)

        # Đọc ảnh LiDAR BEV và resize cùng chiều cao với cụm camera
        lidar_p = viz_dir / 'lidar' / f"{prefix}.png"
        total_h = cams_canvas.shape[0]
        if lidar_p.exists():
            lidar_im = cv2.imread(str(lidar_p))
            lidar_im = cv2.resize(lidar_im, (total_h, total_h))
        else:
            lidar_im = np.zeros((total_h, total_h, 3), dtype=np.uint8)

        # Ghép Cụm Camera (trái) + LiDAR (phải)
        full_frame = np.hstack([cams_canvas, lidar_im])

        # Khởi tạo VideoWriter với FPS = 2 (tần suất gốc của nuScenes) hoặc 4 nếu muốn chạy nhanh hơn
        if vw is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(str(video_path), fourcc, 2.0, (full_frame.shape[1], full_frame.shape[0]))

        vw.write(full_frame)

    if vw is not None:
        vw.release()
    print(f"Đã tạo xong: {video_path} ({len(frames)} frames)")