import argparse
import json
import os
import shutil
import cv2
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create demo video from BEVFusion visualization frames")
    parser.add_argument("--data-root", type=str, default="data/nuscenes",
                        help="Path to dataset root (e.g., data/nuscenes_vf6_01_5hz_nav_calib)")
    parser.add_argument("--viz-dir", type=str, default="outputs/mini-viz",
                        help="Path to folder containing camera-0..5 and lidar subfolders")
    parser.add_argument("--output-dir", type=str, default="outputs/demo-videos",
                        help="Output directory for generated mp4 videos")
    parser.add_argument("--prefix", type=str, default="scene",
                        help="Prefix for output video files (e.g., 'scene' or 'scene_detzero')")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second")
    args = parser.parse_args()

    root_path = Path(args.data_root)
    val_info_path = root_path / "bevfusion_infos_val.pkl"
    if not val_info_path.exists():
        val_info_path = root_path / "nuscenes_infos_val.pkl"

    sample_json_path = root_path / "v1.0-trainval/sample.json"
    if not sample_json_path.exists():
        sample_json_path = root_path / "v1.0-mini/sample.json"
    if not sample_json_path.exists():
        sample_json_path = root_path / "sample.json"

    viz_dir = Path(args.viz_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sample_json_path.exists():
        raise FileNotFoundError(f"Missing sample.json under {root_path}")

    # 1. Map sample_token -> scene_token and sample details from sample.json
    samples_raw = json.loads(sample_json_path.read_text())
    token_to_sample = {s["token"]: s for s in samples_raw}

    scenes = {}
    if val_info_path.exists():
        try:
            import mmcv
            val_data = mmcv.load(str(val_info_path))
            for info in val_data["infos"]:
                token = info["token"]
                timestamp = info["timestamp"]
                s_rec = token_to_sample.get(token, {})
                scene_token = s_rec.get("scene_token", "unknown_scene")
                file_prefix = f"{timestamp}-{token}"
                scenes.setdefault(scene_token, []).append((timestamp, file_prefix))
        except Exception as e:
            print(f"Warning: Could not load {val_info_path}: {e}, falling back to sample.json")

    if not scenes:
        # Fallback: scan existing frames in viz_dir
        cam0_dir = viz_dir / "camera-0"
        existing_prefixes = set()
        if cam0_dir.exists():
            for p in cam0_dir.glob("*.png"):
                existing_prefixes.add(p.stem)

        for s in samples_raw:
            prefix = f"{s['timestamp']}-{s['token']}"
            if not existing_prefixes or prefix in existing_prefixes:
                scene_token = s["scene_token"]
                scenes.setdefault(scene_token, []).append((s["timestamp"], prefix))

    # Sort frames in each scene by timestamp
    for s_token in scenes:
        scenes[s_token].sort(key=lambda x: x[0])

    print(f"Found {len(scenes)} scenes in {viz_dir}:")
    for idx, (s_token, frames) in enumerate(scenes.items()):
        print(f" - Scene {idx + 1} (token: {s_token[:8]}...): {len(frames)} frames")

    # Camera layout (2 rows x 3 columns) matching BEVFusion default
    cam_order = [
        ["camera-1", "camera-0", "camera-2"],
        ["camera-4", "camera-3", "camera-5"]
    ]

    has_ffmpeg = shutil.which("ffmpeg") is not None

    for idx, (s_token, frames) in enumerate(scenes.items()):
        raw_video_path = output_dir / f"{args.prefix}_{idx + 1}_raw.mp4"
        final_video_path = output_dir / f"{args.prefix}_{idx + 1}_fixed.mp4"
        standard_video_path = output_dir / f"{args.prefix}_{idx + 1}.mp4"

        vw = None
        target_cam_h, target_cam_w = 270, 480

        for frm_idx, (_, prefix) in enumerate(frames):
            # 1. Stitch 6 cameras
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
            cams_canvas = np.vstack(rows) # (540, 1440, 3)

            # 2. Stitch LiDAR BEV
            lidar_p = viz_dir / "lidar" / f"{prefix}.png"
            total_h = cams_canvas.shape[0]
            if lidar_p.exists():
                lidar_im = cv2.imread(str(lidar_p))
                lidar_im = cv2.resize(lidar_im, (total_h, total_h))
            else:
                lidar_im = np.zeros((total_h, total_h, 3), dtype=np.uint8)

            full_frame = np.hstack([cams_canvas, lidar_im]) # (540, 1980, 3)

            if vw is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                save_target = str(raw_video_path) if has_ffmpeg else str(standard_video_path)
                vw = cv2.VideoWriter(save_target, fourcc, args.fps, (full_frame.shape[1], full_frame.shape[0]))

            vw.write(full_frame)

        if vw is not None:
            vw.release()

        # 3. Transcode with ffmpeg if available for universal browser playback
        if has_ffmpeg and raw_video_path.exists():
            cmd = f"ffmpeg -y -loglevel error -i '{raw_video_path}' -vcodec libx264 -pix_fmt yuv420p '{final_video_path}'"
            os.system(cmd)
            # Also keep standard filename
            shutil.copyfile(str(final_video_path), str(standard_video_path))
            raw_video_path.unlink(missing_ok=True)
            print(f"Exported Scene {idx + 1}: {final_video_path} & {standard_video_path}")
        else:
            print(f"Exported Scene {idx + 1}: {standard_video_path}")


if __name__ == "__main__":
    main()