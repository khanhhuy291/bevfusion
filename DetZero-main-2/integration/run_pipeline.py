#!/usr/bin/env python3
"""End-to-end integration pipeline: BEVFusion detection -> DetZero tracking -> DetZero refining (GRM + PRM) -> NuScenes export."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from easydict import EasyDict

# Ensure DetZero subpackages can be imported
DETZERO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DETZERO_ROOT))
sys.path.insert(0, str(DETZERO_ROOT / "tracking"))
sys.path.insert(0, str(DETZERO_ROOT / "refining"))
sys.path.insert(0, str(DETZERO_ROOT / "utils"))

from integration import bridge
from detzero_utils.config_utils import cfg_from_yaml_file
from detzero_track.models.detzero_tracker import DetZeroTracker
from detzero_refine.models import build_network


class StandaloneDataset:
    """Minimal dataset stub for loading refining models without Waymo data dependencies."""
    def __init__(self):
        self.tta = False


def parse_args():
    parser = argparse.ArgumentParser(description="BEVFusion to DetZero Tracking & Refining Pipeline")
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to BEVFusion detection output (results_nusc.json)")
    parser.add_argument("--data_root", type=str, default="data/nuscenes",
                        help="Path to nuScenes dataset root directory")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        help="nuScenes version (e.g. v1.0-mini or v1.0-trainval)")
    parser.add_argument("--tracking_cfg", type=str,
                        default=str(DETZERO_ROOT / "tracking/tools/cfgs/tk_model_cfgs/nuscenes_detzero_track.yaml"),
                        help="Tracking config file")
    parser.add_argument("--checkpoint_dir", type=str,
                        default=str(DETZERO_ROOT / "checkpoints"),
                        help="Directory containing DetZero GRM and PRM checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs/detzero_refined",
                        help="Directory to save final refined and tracking outputs")
    parser.add_argument("--min_score", type=float, default=0.1,
                        help="Minimum detection confidence threshold for tracking input")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run models on ('cuda' or 'cpu')")
    parser.add_argument("--skip_refining", action="store_true",
                        help="Skip refining (GRM + PRM) and only perform tracking")
    parser.add_argument("--classes", type=str,
                        default="car,truck,bus,trailer,construction_vehicle,pedestrian,motorcycle,bicycle",
                        help="Comma-separated nuScenes classes to track and refine")
    return parser.parse_args()


def load_refining_models(checkpoint_dir: Path, device: str):
    """Load the 6 GRM and PRM checkpoints for Vehicle, Pedestrian, and Cyclist."""
    ref_cfg_root = DETZERO_ROOT / "refining/tools/cfgs/ref_model_cfgs"
    model_types = ["Vehicle", "Pedestrian", "Cyclist"]
    models = {"grm": {}, "prm": {}}

    for cls_name in model_types:
        prefix = bridge.MODEL_PREFIX[cls_name]

        import yaml

        # Load GRM
        grm_cfg_file = ref_cfg_root / f"{prefix}_grm_model.yaml"
        grm_ckpt_file = checkpoint_dir / f"{prefix}_grm_model.pth"
        if grm_cfg_file.is_file() and grm_ckpt_file.is_file():
            grm_raw = yaml.safe_load(grm_cfg_file.read_text())
            grm_cfg = EasyDict(grm_raw)
            grm_model = build_network(grm_cfg.MODEL, dataset=StandaloneDataset())
            ckpt = torch.load(str(grm_ckpt_file), map_location="cpu", weights_only=False)
            state = ckpt.get("model_state", ckpt)
            grm_model.load_state_dict(state, strict=False)
            grm_model.to(device).eval()
            models["grm"][cls_name] = grm_model
            print(f"[Refining] Loaded {cls_name} GRM from {grm_ckpt_file.name}")
        else:
            print(f"[Refining] Warning: missing GRM config or checkpoint for {cls_name}: {grm_ckpt_file}")

        # Load PRM
        prm_cfg_file = ref_cfg_root / f"{prefix}_prm_model.yaml"
        prm_ckpt_file = checkpoint_dir / f"{prefix}_prm_model.pth"
        if prm_cfg_file.is_file() and prm_ckpt_file.is_file():
            prm_raw = yaml.safe_load(prm_cfg_file.read_text())
            prm_cfg = EasyDict(prm_raw)
            prm_model = build_network(prm_cfg.MODEL, dataset=StandaloneDataset())
            ckpt = torch.load(str(prm_ckpt_file), map_location="cpu", weights_only=False)
            state = ckpt.get("model_state", ckpt)
            prm_model.load_state_dict(state, strict=False)
            prm_model.to(device).eval()
            models["prm"][cls_name] = prm_model
            print(f"[Refining] Loaded {cls_name} PRM from {prm_ckpt_file.name}")
        else:
            print(f"[Refining] Warning: missing PRM config or checkpoint for {cls_name}: {prm_ckpt_file}")

    return models


def run_tracking(prepared: Dict[str, Any], tracking_cfg_path: str, device: str) -> Dict[str, Dict[str, Any]]:
    """Run DetZero Kalman + Data Association tracker per scene."""
    tk_cfg = EasyDict()
    cfg_from_yaml_file(tracking_cfg_path, tk_cfg)
    tk_cfg.MODEL.DEVICE = device
    tk_cfg.MODEL.TRACKING.DATA_ASSOCIATION.device = device

    tracker = DetZeroTracker(tk_cfg.MODEL)
    all_tracks = {}

    for scene_id, frame_dict in prepared["frames"].items():
        print(f"[Tracking] Processing scene {scene_id} ({len(frame_dict)} frames)...")
        # TrackManager.forward expects frame_dict indexed by str(frame_id)
        scene_tracks = tracker.forward(frame_dict)
        print(f"  -> Generated {len(scene_tracks)} tracks for scene {scene_id}")
        for tid, track_info in scene_tracks.items():
            global_tid = f"{scene_id}_{tid}"
            track_info["sequence_name"] = str(scene_id)
            all_tracks[global_tid] = track_info

    return all_tracks


def collect_track_pointclouds(prepared: Dict[str, Any], tracks: Dict[str, Dict[str, Any]], data_root: Path):
    """Load LiDAR frames and crop points within bounding boxes for each tracked object."""
    # Group frames to read each LiDAR point cloud once per scene
    scene_lidar_cache = {}

    for tid, track in tracks.items():
        scene_id = track["sequence_name"]
        if isinstance(scene_id, (list, np.ndarray)):
            scene_id = str(scene_id[0]) if len(scene_id) > 0 else str(scene_id)
        else:
            scene_id = str(scene_id)
        track["sequence_name"] = scene_id

        track_boxes = track["boxes_global"]
        sample_indices = track["sample_idx"]

        track_pts = []
        for i, frm_idx in enumerate(sample_indices):
            frame_data = prepared["frames"][scene_id][str(frm_idx)]
            lidar_rel_path = frame_data["lidar_path"]
            lidar_full_path = data_root / lidar_rel_path

            cache_key = (scene_id, frm_idx)
            if cache_key not in scene_lidar_cache:
                pts_global = bridge.read_points(str(lidar_full_path), frame_data["pose"], intensity_mode="tanh")
                scene_lidar_cache[cache_key] = pts_global
            else:
                pts_global = scene_lidar_cache[cache_key]

            box = track_boxes[i]
            cropped = bridge.crop_points(pts_global, box, scale=1.1)
            track_pts.append(cropped)

        track["pts"] = track_pts

    print(f"[LiDAR] Extracted object point clouds for {len(tracks)} tracks.")


def run_refining(tracks: Dict[str, Dict[str, Any]], models: Dict[str, Dict[str, Any]], device: str) -> Dict[str, np.ndarray]:
    """Run GRM (geometry) and PRM (position/heading) models on each track."""
    refined_boxes = {}
    rng = np.random.default_rng(42)

    for tid, track in tracks.items():
        cls_name = track["name"]
        if isinstance(cls_name, (list, np.ndarray)):
            cls_name = cls_name[0]

        if cls_name not in models["grm"] or cls_name not in models["prm"]:
            refined_boxes[tid] = track["boxes_global"][:, :7]
            continue

        grm_model = models["grm"][cls_name]
        prm_model = models["prm"][cls_name]

        # Check that track has points
        total_pts = sum(len(p) for p in track.get("pts", []))
        if total_pts < 10:
            # Fallback to detector/tracker box if points are too sparse
            refined_boxes[tid] = track["boxes_global"][:, :7]
            continue

        try:
            # 1. Geometry Refinement (GRM)
            geo_feat = bridge.geometry_features(track, rng)
            geo_batch = {
                "geo_query_points": torch.from_numpy(geo_feat["geo_query_points"]).to(device),
                "geo_memory_points": torch.from_numpy(geo_feat["geo_memory_points"]).to(device),
                "geo_query_boxes": torch.from_numpy(geo_feat["geo_query_boxes"]).to(device),
                "geo_query_num": geo_feat["geo_query_num"]
            }
            with torch.no_grad():
                geo_preds, _, _ = grm_model(geo_batch)
                # geo_preds['pred_boxes'] shape is (1, 7) with sizes at [3:6]
                refined_size = geo_preds["pred_boxes"][0, 3:6]

            # 2. Position Refinement (PRM)
            with_class = (cls_name == "Cyclist")
            pos_feat, origin = bridge.position_features(track, rng, with_class=with_class)
            n_frames = len(track["sample_idx"])
            pos_batch = {
                "pos_query_points": torch.from_numpy(pos_feat["pos_query_points"]).to(device),
                "pos_memory_points": torch.from_numpy(pos_feat["pos_memory_points"]).to(device),
                "pos_trajectory": torch.from_numpy(pos_feat["pos_trajectory"]).to(device),
                "padding_mask": torch.from_numpy(pos_feat["padding_mask"]).to(device)
            }
            with torch.no_grad():
                pos_preds, _, _ = prm_model(pos_batch)
                # Local refined trajectory shape (1, 200, 7)
                local_boxes = pos_preds["pred_boxes"][0, :n_frames]
                refined_pos_global = bridge.position_to_global(local_boxes, origin)

            # 3. Combine Size + Position
            combined = bridge.combine_boxes(np.array([0, 0, 0, *refined_size, 0]), refined_pos_global)
            refined_boxes[tid] = combined

        except Exception as e:
            print(f"[Refining] Warning: refining failed for track {tid} ({cls_name}): {e}, keeping tracking box.")
            refined_boxes[tid] = track["boxes_global"][:, :7]

    return refined_boxes


def export_tracking_results(prepared: Dict[str, Any], tracks: Dict[str, Dict[str, Any]], refined_boxes: Dict[str, np.ndarray], output_file: Path):
    """Export predictions with tracking IDs formatted for nuScenes tracking evaluation."""
    tracking_output = copy.deepcopy(prepared["original"])
    tracking_results = {token: [] for token in tracking_output["results"]}

    VALID_TRACKING_NAMES = {'bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck'}

    for tid, track in tracks.items():
        boxes = refined_boxes.get(tid, track["boxes_global"][:, :7])
        cls_name = track["nusc_name"] if "nusc_name" in track else track["name"]
        if isinstance(cls_name, (list, np.ndarray)):
            cls_name = cls_name[0]

        for i, frm_idx in enumerate(track["sample_idx"]):
            frame_data = prepared["frames"][track["sequence_name"]][str(frm_idx)]
            token = frame_data["sample_token"]
            score = float(track["score"][i])
            box = boxes[i]
            source = int(track["source_index"][i])

            # Determine nuScenes tracking category
            tracking_name = None
            if source >= 0 and source < len(prepared["original"]["results"][token]):
                orig_name = prepared["original"]["results"][token][source].get("detection_name", "")
                if orig_name in VALID_TRACKING_NAMES:
                    tracking_name = orig_name

            if tracking_name is None:
                # Map from DetZero class
                if cls_name == "Pedestrian":
                    tracking_name = "pedestrian"
                elif cls_name == "Cyclist":
                    tracking_name = "bicycle"
                elif cls_name == "Vehicle":
                    tracking_name = "car"
                elif str(cls_name).lower() in VALID_TRACKING_NAMES:
                    tracking_name = str(cls_name).lower()
                else:
                    tracking_name = "car"

            # Convert to nuScenes record format
            yaw = float(bridge.wrap_yaw(box[6]))
            quat = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
            tracking_rec = {
                "sample_token": token,
                "translation": [float(box[0]), float(box[1]), float(box[2])],
                "size": [float(box[4]), float(box[3]), float(box[5])], # nuScenes [w, l, h]
                "rotation": quat,
                "velocity": [float(track["boxes_global"][i, 7]), float(track["boxes_global"][i, 8])],
                "tracking_id": str(tid),
                "tracking_name": tracking_name,
                "tracking_score": score
            }
            tracking_results[token].append(tracking_rec)

    tracking_output["results"] = tracking_results
    output_file.write_text(json.dumps(tracking_output, indent=2))
    print(f"[Export] Saved tracking results to {output_file}")


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    print("=" * 60)
    print("BEVFUSION -> DETZERO TRACKING & REFINING PIPELINE")
    print(f"Results path   : {args.results_path}")
    print(f"nuScenes root  : {data_root} ({args.version})")
    print(f"Checkpoints    : {checkpoint_dir}")
    print(f"Device         : {args.device}")
    print(f"Output dir     : {output_dir}")
    print("=" * 60)

    # 1. Prepare detector output and sync timestamps
    print("\n[Step 1/5] Preparing frames from detector output and nuScenes metadata...")
    prepared = bridge.prepare(
        results_path=args.results_path,
        data_root=str(data_root),
        version=args.version,
        classes=classes,
        min_score=args.min_score
    )
    n_scenes = len(prepared["frames"])
    n_frames = sum(len(f) for f in prepared["frames"].values())
    print(f"  -> Prepared {n_scenes} scenes, total {n_frames} frames.")

    # 2. Run DetZero Tracking
    print("\n[Step 2/5] Running DetZero Kalman & Data Association Tracking...")
    tracks = run_tracking(prepared, args.tracking_cfg, args.device)
    print(f"  -> Total tracks generated across all scenes: {len(tracks)}")

    # 3. Object point clouds extraction & 4. Refining
    refined_boxes = {}
    if not args.skip_refining:
        print("\n[Step 3/5] Extracting LiDAR point clouds for each tracked object...")
        collect_track_pointclouds(prepared, tracks, data_root)

        print("\n[Step 4/5] Loading GRM & PRM models and performing 3D refinement...")
        models = load_refining_models(checkpoint_dir, args.device)
        refined_boxes = run_refining(tracks, models, args.device)
        print(f"  -> Refined {len(refined_boxes)} tracks.")
    else:
        print("\n[Step 3-4/5] Skipping refining as requested (--skip_refining).")
        for tid, track in tracks.items():
            refined_boxes[tid] = track["boxes_global"][:, :7]

    # 5. Export back to nuScenes detection format
    print("\n[Step 5/5] Exporting refined predictions back to nuScenes format...")
    refined_output, replaced_count = bridge.export_detection(prepared, tracks, refined=refined_boxes)
    det_out_file = output_dir / "results_nusc_detzero_refined.json"
    det_out_file.write_text(json.dumps(refined_output, indent=2))
    print(f"  -> Replaced {replaced_count} detection boxes with refined boxes.")
    print(f"  -> Saved detection JSON: {det_out_file}")

    # Export tracking format as well
    track_out_file = output_dir / "results_nusc_detzero_tracking.json"
    export_tracking_results(prepared, tracks, refined_boxes, track_out_file)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"1. Refined Detection output : {det_out_file}")
    print(f"2. Tracking output          : {track_out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
