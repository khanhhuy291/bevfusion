import os
import json
import cv2
import colorsys
import numpy as np
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility


CLS_ABBR = {
    'pedestrian': 'Ped',
    'car': 'Car',
    'truck': 'Truck',
    'bus': 'Bus',
    'trailer': 'Trailer',
    'bicycle': 'Bike',
    'motorcycle': 'Moto',
    'construction_vehicle': 'Const'
}


OBJECT_PALETTE = {
    'car': (0, 158, 255),               # BGR
    'truck': (71, 99, 255),
    'construction_vehicle': (122, 150, 233),
    'bus': (0, 69, 255),
    'trailer': (0, 140, 255),
    'barrier': (144, 128, 112),
    'motorcycle': (99, 61, 255),
    'bicycle': (60, 20, 220),
    'pedestrian': (230, 0, 0),
    'traffic_cone': (79, 79, 47),
}


def render_camera_view(img: np.ndarray, boxes: list, cam_cs: dict, cam_pose: dict,
                       lidar_cs: dict, lidar_pose: dict,
                       target_w: int = 480, target_h: int = 270, cam_name: str = "") -> np.ndarray:
    """Project and draw 3D bounding boxes using BEVFusion's exact LiDAR-to-Camera geometry."""
    canvas = img.copy()
    orig_h, orig_w = canvas.shape[:2]
    intrinsic = np.array(cam_cs['camera_intrinsic'])

    # Geometry matrices: LiDAR -> Ego(t_lidar) -> Global -> Ego(t_cam) -> Camera
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
        b = Box(
            item['translation'], item['size'], Quaternion(item['rotation']),
            name=name, score=item.get('tracking_score', item.get('detection_score', -1))
        )

        # 1. Transform Global -> Ego(t_lidar) -> LiDAR frame
        b.translate(-e2g_t)
        b.rotate(e2g_r.inverse)
        b.translate(-l2e_t)
        b.rotate(l2e_r.inverse)

        # 2. Get 8 corners in LiDAR frame: (3, 8)
        corners_lidar = b.corners()

        # 3. Transform corners: LiDAR -> Ego(t_lidar) -> Global -> Ego(t_cam) -> Camera
        # Step 1: LiDAR -> Ego(t_lidar)
        pts_ego_l = l2e_r.rotation_matrix @ corners_lidar + l2e_t[:, None]
        # Step 2: Ego(t_lidar) -> Global
        pts_glob = e2g_r.rotation_matrix @ pts_ego_l + e2g_t[:, None]
        # Step 3: Global -> Ego(t_cam)
        pts_ego_c = g2e_r.rotation_matrix.T @ (pts_glob - g2e_t[:, None])
        # Step 4: Ego(t_cam) -> Camera
        corners_cam = e2c_r.rotation_matrix.T @ (pts_ego_c - e2c_t[:, None])

        # Filter out boxes behind camera lens (z <= 0.1)
        if np.any(corners_cam[2, :] <= 0.1):
            continue

        # Project to pixel space
        corners_img = view_points(corners_cam, intrinsic, normalize=True)[:2, :]

        # Check if box corners fall within image bounds
        if (np.all(corners_img[0, :] < 0) or np.all(corners_img[0, :] >= orig_w) or
            np.all(corners_img[1, :] < 0) or np.all(corners_img[1, :] >= orig_h)):
            continue

        color = OBJECT_PALETTE.get(name, (0, 255, 0))
        tid = item.get('tracking_id', '')
        short_id = str(tid).split('_')[-1] if tid else ''

        def pt(idx):
            return (int(round(corners_img[0, idx])), int(round(corners_img[1, idx])))

        # Draw wireframe edges matching BEVFusion:
        # NuScenes Box corners:
        # Front: 0, 1, 2, 3; Rear: 4, 5, 6, 7
        for i in range(4):
            cv2.line(canvas, pt(i), pt(i + 4), color, 3, cv2.LINE_AA)
            cv2.line(canvas, pt(i), pt((i + 1) % 4), color, 3, cv2.LINE_AA)
            cv2.line(canvas, pt(i + 4), pt(((i + 1) % 4) + 4), color, 3, cv2.LINE_AA)

        # Draw subtle track badge if tracking_id exists
        if short_id:
            min_y_idx = int(np.argmin(corners_img[1, :]))
            tag_x = int(np.clip(corners_img[0, min_y_idx] - 12, 10, orig_w - 90))
            tag_y = int(np.clip(corners_img[1, min_y_idx] - 6, 22, orig_h - 10))
            tag_text = f"#{short_id} {name[:3].capitalize()}"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(canvas, (tag_x - 2, tag_y - th - 3), (tag_x + tw + 3, tag_y + 3), color, -1)
            cv2.putText(canvas, tag_text, (tag_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Resize to layout target
    canvas = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    # Camera title overlay
    cv2.rectangle(canvas, (5, 5), (5 + len(cam_name) * 8 + 14, 22), (20, 20, 25), -1)
    cv2.putText(canvas, cam_name, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_lidar_bev(lidar_path: str, boxes: list, lidar_cs: dict, lidar_pose: dict,
                     size: int = 540, pc_range: float = 50.0) -> np.ndarray:
    """Render LiDAR point cloud and 3D bounding boxes in BEVFusion bird's eye view aesthetic."""
    bev = np.full((size, size, 3), 15, dtype=np.uint8) # Dark background matching BEVFusion
    center_px = (size // 2, size // 2)

    # Draw range rings (15m, 30m, 45m)
    for r in [15, 30, 45]:
        r_px = int(r / pc_range * (size // 2))
        cv2.circle(bev, center_px, r_px, (35, 38, 44), 1, cv2.LINE_AA)

    # Draw LiDAR points in white
    if os.path.exists(lidar_path):
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        mask = (np.abs(pts[:, 0]) <= pc_range) & (np.abs(pts[:, 1]) <= pc_range)
        pts = pts[mask]
        u = np.clip(((pts[:, 0] + pc_range) / (2 * pc_range) * (size - 1)).astype(np.int32), 0, size - 1)
        v = np.clip(((pc_range - pts[:, 1]) / (2 * pc_range) * (size - 1)).astype(np.int32), 0, size - 1)
        bev[v, u] = (220, 230, 240)

    # Draw boxes in BEV
    for item in boxes:
        name = item.get('tracking_name', item.get('detection_name', 'car'))
        b = Box(item['translation'], item['size'], Quaternion(item['rotation']), name=name)
        # Global -> Ego(t_lidar) -> LiDAR
        b.translate(-np.array(lidar_pose['translation']))
        b.rotate(Quaternion(lidar_pose['rotation']).inverse)
        b.translate(-np.array(lidar_cs['translation']))
        b.rotate(Quaternion(lidar_cs['rotation']).inverse)

        # 4 bottom corners in LiDAR frame: [Front-Right, Front-Left, Rear-Left, Rear-Right]
        c = b.corners()[:2, [2, 3, 7, 6]]
        if np.all(np.abs(c) > pc_range * 1.2):
            continue

        u = ((c[0, :] + pc_range) / (2 * pc_range) * (size - 1)).astype(np.int32)
        v = ((pc_range - c[1, :]) / (2 * pc_range) * (size - 1)).astype(np.int32)
        poly = np.stack([u, v], axis=1)

        color = OBJECT_PALETTE.get(name, (0, 255, 0))
        cv2.polylines(bev, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Heading indicator
        front_mid = ((u[0] + u[1]) // 2, (v[0] + v[1]) // 2)
        center = (int(u.mean()), int(v.mean()))
        cv2.line(bev, center, front_mid, color, 2, cv2.LINE_AA)

    # Title header
    cv2.rectangle(bev, (5, 5), (140, 22), (20, 20, 25), -1)
    cv2.putText(bev, "LiDAR BEV", (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)
    return bev


def main():
    print("=" * 60)
    print("BEVFUSION + DETZERO DEMO VIDEO GENERATOR WITH TRACK IDS")
    print("=" * 60)

    dataroot = Path("data/nuscenes")
    tracking_json_path = Path("outputs/detzero_refined/results_nusc_detzero_tracking.json")
    output_dir = Path("outputs/demo-videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading nuScenes dataset from {dataroot}...")
    nusc = NuScenes(version="v1.0-mini", dataroot=str(dataroot), verbose=False)

    print(f"Loading tracking results from {tracking_json_path}...")
    with open(tracking_json_path) as f:
        tracking_data = json.load(f)["results"]

    # Only process validation scenes that have predictions
    # Preserve exact order matching create_demo_video.py:
    # Scene 1: fcbccedd61424f1b85dcbf8f897f9754 (scene-0103, 40 frames)
    # Scene 2: 325cef682f064c55a255f2625c533b75 (scene-0916, 41 frames)
    target_scenes = [
        "fcbccedd61424f1b85dcbf8f897f9754",
        "325cef682f064c55a255f2625c533b75"
    ]

    scenes = {}
    for sample in nusc.sample:
        st = sample["scene_token"]
        if st in target_scenes:
            scenes.setdefault(st, []).append((sample["timestamp"], sample["token"]))

    for st in scenes:
        scenes[st].sort(key=lambda x: x[0])

    # Sort target_scenes in defined order
    ordered_scenes = [(st, scenes[st]) for st in target_scenes if st in scenes]

    print(f"Found {len(ordered_scenes)} validation scenes to process:")
    for idx, (st, frames) in enumerate(ordered_scenes):
        print(f" - Scene {idx + 1} (token: {st[:8]}...): {len(frames)} frames")

    # Camera layout definition: 2 rows x 3 columns
    cam_layout = [
        [("CAM_FRONT_LEFT", "Front-Left"), ("CAM_FRONT", "Front"), ("CAM_FRONT_RIGHT", "Front-Right")],
        [("CAM_BACK_LEFT", "Back-Left"), ("CAM_BACK", "Back"), ("CAM_BACK_RIGHT", "Back-Right")]
    ]

    target_cam_w, target_cam_h = 480, 270
    lidar_size = 540

    for idx, (st, frames) in enumerate(ordered_scenes):
        raw_video_path = output_dir / f"scene_{idx + 1}_detzero.mp4"
        fixed_video_path = output_dir / f"scene_{idx + 1}_detzero_fixed.mp4"
        print(f"\n[Processing Scene {idx + 1}/{len(ordered_scenes)}] -> {len(frames)} frames...")

        vw = None

        for frm_idx, (_, token) in enumerate(frames):
            sample = nusc.get("sample", token)
            boxes = tracking_data.get(token, [])

            # 1. Fetch LiDAR records for synchronization and BEV
            lidar_sd_token = sample["data"]["LIDAR_TOP"]
            lidar_sd = nusc.get("sample_data", lidar_sd_token)
            lidar_path = nusc.get_sample_data_path(lidar_sd_token)
            lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
            lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

            # 2. Render 6 Camera views with exact LiDAR-to-Camera geometry
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
                    cam_canvas = render_camera_view(
                        raw_img, boxes, cs_record, pose_record,
                        lidar_cs=lidar_cs, lidar_pose=lidar_pose,
                        target_w=target_cam_w, target_h=target_cam_h,
                        cam_name=display_name
                    )
                    row_imgs.append(cam_canvas)
                cam_rows.append(np.hstack(row_imgs))
            cameras_canvas = np.vstack(cam_rows) # (540, 1440, 3)

            # 3. Render LiDAR BEV view
            bev_canvas = render_lidar_bev(
                str(lidar_path), boxes, lidar_cs, lidar_pose,
                size=lidar_size, pc_range=50.0
            ) # (540, 540, 3)

            # 4. Stitch cameras and LiDAR BEV matching BEVFusion exact 1980x540 layout
            full_frame = np.hstack([cameras_canvas, bev_canvas]) # (540, 1980, 3)

            if vw is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # 2.0 FPS for nuScenes keyframe rate
                vw = cv2.VideoWriter(str(raw_video_path), fourcc, 2.0, (full_frame.shape[1], full_frame.shape[0]))

            vw.write(full_frame)
            print(f"  Frame [{frm_idx + 1:02d}/{len(frames):02d}] rendered: {len(boxes)} tracks", end="\r")

        if vw is not None:
            vw.release()
        print(f"\n  -> Raw video saved: {raw_video_path}")

        # 5. Transcode with ffmpeg for universal browser playback
        print(f"  -> Transcoding with ffmpeg -> {fixed_video_path}...")
        cmd = f"ffmpeg -y -loglevel error -i '{raw_video_path}' -vcodec libx264 -pix_fmt yuv420p '{fixed_video_path}'"
        os.system(cmd)
        print(f"  -> SUCCESS! Final video: {fixed_video_path}")

    print("\n" + "=" * 60)
    print("ALL DEMO VIDEOS GENERATED SUCCESSFULLY!")
    print(f"Output folder: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
