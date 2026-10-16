#!/usr/bin/env python3
"""Run LiDAR-only BEVFusion on PCD frames, then DetZero without CRM."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.pcd_lidar_utils import (compare_results, interpolate_tum_pose,
                                   model_box_to_nuscenes_record, prepare_model_points,
                                   read_binary_pcd, read_tum_trajectory,
                                   write_nuscenes_bridge_dataset)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd-dir", default="data/data_test_only_lidar")
    parser.add_argument("--trajectory", help="TUM pose file; default: <pcd-dir>/trajectory_optimized.tum")
    parser.add_argument("--max-pose-gap", type=float, default=0.11,
                        help="Maximum seconds from a PCD timestamp to its nearest TUM pose")
    parser.add_argument("--invert-trajectory", action="store_true",
                        help="Use only if the TUM file stores LiDAR-from-world instead of world-from-LiDAR")
    parser.add_argument("--config", default="configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml")
    parser.add_argument("--checkpoint", default="pretrained/lidar-only-det.pth")
    parser.add_argument("--output-dir", default="../bevfusion-adapter/runs/pcd-pretrained-smoke")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, help="Use the first N selected frames; default is all")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--transform", type=float, nargs=16, metavar="M")
    parser.add_argument("--no-bev", action="store_true")
    parser.add_argument("--detzero-refinement", choices=["none", "geometry", "full"], default="full")
    parser.add_argument("--detzero-checkpoint-dir", default="DetZero-main-2/checkpoints")
    parser.add_argument("--tracking-device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be in [0,1]")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.max_pose_gap <= 0:
        parser.error("--max-pose-gap must be positive")
    return args


def load_config(path):
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval
    configs.load(str(path), recursive=True)
    return Config(recursive_eval(configs), filename=str(path))


def build_detector(cfg, checkpoint, device):
    import torch
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_model
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("BEVFusion sparse voxel inference requires CUDA")
    torch.cuda.set_device(int(device.split(":", 1)[1]) if ":" in device else 0)
    cfg.model.train_cfg = None
    if "pretrained" in cfg.model:
        cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    return model.to(device).eval()


def save_bev(path, points, boxes, labels, classes, point_range):
    from mmdet3d.core.utils import visualize_lidar
    visualize_lidar(str(path), points, bboxes=boxes, labels=labels, classes=classes,
                    xlim=tuple(point_range[[0, 3]]), ylim=tuple(point_range[[1, 4]]),
                    radius=1, thickness=3)


def run_detector(args, files, output, trajectory):
    import torch
    from mmdet3d.core import LiDARInstance3DBoxes
    cfg = load_config(ROOT / args.config)
    classes = list(cfg.object_classes)
    point_range = np.asarray(cfg.point_cloud_range, dtype=np.float64)
    transform = np.asarray(args.transform, dtype=np.float64).reshape(4, 4) if args.transform else None
    model = build_detector(cfg, ROOT / args.checkpoint, args.device)
    predictions_dir = output / "baseline" / "predictions"
    bev_dir = output / "baseline" / "bev"
    predictions_dir.mkdir(parents=True)
    if not args.no_bev:
        bev_dir.mkdir()
    frames, results, timings, pose_gaps = [], {}, [], []
    warned_zero_intensity = False
    for index, path in enumerate(files, 1):
        raw, metadata = read_binary_pcd(path)
        if metadata["timestamp"] is None:
            raise ValueError(f"{path}: '# timestamp <seconds>' is required for tracking")
        intensity = raw[:, metadata["fields"].index("intensity")]
        if not warned_zero_intensity and np.all(intensity == 0):
            print("WARNING: PCD intensity is identically zero; predictions are out-of-domain.", flush=True)
            warned_zero_intensity = True
        points = prepare_model_points(raw, metadata["fields"], point_range, transform,
                                      args.intensity_scale)
        pose = interpolate_tum_pose(trajectory, metadata["timestamp"], args.max_pose_gap,
                                    args.invert_trajectory)
        token = f"pcd-{path.stem}"
        tensor = torch.from_numpy(points).to(args.device)
        metas = [dict(box_type_3d=LiDARInstance3DBoxes, token=token,
                      timestamp=metadata["timestamp"])]
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model(
                img=None, points=[tensor], camera2ego=None, lidar2ego=None,
                lidar2camera=None, lidar2image=None, camera_intrinsics=None,
                camera2lidar=None, img_aug_matrix=None, lidar_aug_matrix=None,
                metas=metas,
            )[0]
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        keep = prediction["scores_3d"] >= args.score_threshold
        boxes = prediction["boxes_3d"][keep]
        scores = prediction["scores_3d"][keep].numpy()
        labels = prediction["labels_3d"][keep].numpy()
        box_array = boxes.tensor.numpy()
        detections = [dict(class_id=int(label), class_name=classes[int(label)],
                           score=float(score), box=box.tolist())
                      for box, score, label in zip(box_array, scores, labels)]
        record = {"source": str(path), "sample_token": token,
                  "timestamp": metadata["timestamp"], "input_points": len(raw),
                  "model_range_points": len(points), "inference_ms": elapsed_ms,
                  "detections": detections}
        (predictions_dir / f"{path.stem}.json").write_text(json.dumps(record, indent=2) + "\n")
        if not args.no_bev:
            save_bev(bev_dir / f"{path.stem}.png", points, boxes, labels, classes, point_range)
        frames.append({"sample_token": token, "timestamp": metadata["timestamp"], "points": points,
                       "source": str(path), "input_points": len(raw), "pose": pose})
        results[token] = [model_box_to_nuscenes_record(token, box, score, classes[int(label)], pose)
                          for box, score, label in zip(box_array, scores, labels)]
        timings.append(elapsed_ms); pose_gaps.append(pose["nearest_gap"])
        print(f"[{index}/{len(files)}] {path.name}: {len(points)} points, "
              f"{len(scores)} detections >= {args.score_threshold:.2f}, {elapsed_ms:.1f} ms", flush=True)
    baseline_json = write_nuscenes_bridge_dataset(output / "dataset", frames, results)
    summary = {"config": str((ROOT / args.config).resolve()),
               "checkpoint": str((ROOT / args.checkpoint).resolve()),
               "pcd_dir": str(Path(args.pcd_dir).resolve()), "frames": len(frames),
               "detections": sum(map(len, results.values())), "score_threshold": args.score_threshold,
               "point_cloud_range": point_range.tolist(), "single_frame": True,
               "trajectory": trajectory["source"], "trajectory_convention": (
                   "lidar_from_world_inverted" if args.invert_trajectory else "world_from_lidar"),
               "max_nearest_pose_gap_s": float(max(pose_gaps)),
               "identity_ego_pose": False, "ground_truth_available": False,
               "zero_intensity": warned_zero_intensity, "mean_inference_ms": float(np.mean(timings)),
               "median_inference_ms": float(np.median(timings))}
    (output / "baseline" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return baseline_json


def run_detzero(args, baseline_json, output):
    command = [sys.executable, str(ROOT / "DetZero-main-2/integration/run_pipeline.py"),
               "--results_path", str(baseline_json), "--data_root", str(output / "dataset"),
               "--version", "v1.0-mini", "--output_dir", str(output / "detzero"),
               "--coordinate-mode", "global", "--refinement", args.detzero_refinement,
               "--device", args.device, "--tracking-device", args.tracking_device,
               "--checkpoint_dir", str((ROOT / args.detzero_checkpoint_dir).resolve()),
               "--min_score", str(args.score_threshold), "--size_policy", "auto",
               "--z-policy", "center", "--preserve-bottom-classes", "motorcycle"]
    log_path = output / "logs" / "detzero.log"
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONUNBUFFERED="1")
    with log_path.open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
        code = child.wait()
    if code:
        raise RuntimeError(f"DetZero failed (exit {code}); see {log_path}")


def main():
    args = parse_args()
    for required in [ROOT / args.config, ROOT / args.checkpoint]:
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.detzero_refinement in {"geometry", "full"}:
        checkpoint_dir = ROOT / args.detzero_checkpoint_dir
        stages = ("grm", "prm") if args.detzero_refinement == "full" else ("grm",)
        missing = [checkpoint_dir / f"{name}_{stage}_model.pth"
                   for name in ("vehicle", "pedestrian", "cyclist")
                   for stage in stages
                   if not (checkpoint_dir / f"{name}_{stage}_model.pth").is_file()]
        if missing:
            raise FileNotFoundError("Missing DetZero checkpoint(s): " + ", ".join(map(str, missing)))
    pcd_root = Path(args.pcd_dir).resolve()
    files = sorted(pcd_root.rglob("*.pcd"))[::args.stride]
    if args.max_frames is not None:
        files = files[:args.max_frames]
    if not files:
        raise FileNotFoundError(f"No PCD files under {pcd_root}")
    trajectory_path = Path(args.trajectory).resolve() if args.trajectory else pcd_root / "trajectory_optimized.tum"
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    trajectory = read_tum_trajectory(trajectory_path)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty; preserving existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    (output / "run_arguments.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    baseline_json = run_detector(args, files, output, trajectory)
    run_detzero(args, baseline_json, output)
    refined = output / "detzero" / "results_nusc_detzero_refined.json"
    comparison = compare_results(baseline_json, refined)
    comparison.update(frames=len(files), refinement=args.detzero_refinement,
                      coordinate_mode="global_from_interpolated_tum",
                      limitations=["TUM poses are assumed to be world-from-LiDAR transforms.",
                                   "No ground truth; accuracy improvement cannot be measured.",
                                   "Input intensity is zero in the supplied PCD sequence."])
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"Completed. Results: {output}\nComparison: {output / 'comparison.json'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        raise SystemExit(1)
