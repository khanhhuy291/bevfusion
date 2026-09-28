import argparse
import copy
import json
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from pyquaternion import Quaternion
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def global_to_lidar_boxes(preds_for_token, info, object_classes):
    """Convert nuScenes predictions (Global frame) to LiDARInstance3DBoxes (LiDAR frame)."""
    if not preds_for_token or info is None:
        return None, None, None

    e2g_t = np.array(info["ego2global_translation"])
    e2g_r = Quaternion(info["ego2global_rotation"])
    l2e_t = np.array(info["lidar2ego_translation"])
    l2e_r = Quaternion(info["lidar2ego_rotation"])

    locs, dims, yaws, scores, labels = [], [], [], [], []
    for item in preds_for_token:
        c_glob = np.array(item["translation"])
        c_ego = e2g_r.inverse.rotate(c_glob - e2g_t)
        c_lidar = l2e_r.inverse.rotate(c_ego - l2e_t)

        q_glob = Quaternion(item["rotation"])
        q_ego = e2g_r.inverse * q_glob
        q_lidar = l2e_r.inverse * q_ego

        yaw = -q_lidar.yaw_pitch_roll[0] - np.pi / 2
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

        w, l, h = item["size"]
        name = item.get("detection_name", item.get("tracking_name", "car"))
        score = float(item.get("detection_score", item.get("tracking_score", 1.0)))

        cls_idx = object_classes.index(name) if name in object_classes else -1

        locs.append(c_lidar)
        dims.append([w, l, h])
        yaws.append(yaw)
        scores.append(score)
        labels.append(cls_idx)

    if not locs:
        return None, None, None

    locs = np.array(locs)
    dims = np.array(dims)
    yaws = np.array(yaws)[:, None]
    scores = np.array(scores)
    labels = np.array(labels)

    # BEVFusion export stores translation[2] at roof/top-half elevation (z_centroid + h / 2)
    # due to LiDARInstance3DBoxes.gravity_center adding h/2. To pass true bottom-center
    # to LiDARInstance3DBoxes for ground-level rendering, subtract the full box height h.
    locs[:, 2] -= dims[:, 2]
    box_tensor = np.concatenate([locs, dims, yaws, np.zeros((len(locs), 2))], axis=-1)
    bboxes = LiDARInstance3DBoxes(box_tensor, box_dim=9)

    return bboxes, scores, labels


def main() -> None:
    try:
        dist.init()
        is_dist = dist.is_distributed()
    except Exception:
        is_dist = False

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--result", type=str, default=None,
                        help="Path to results JSON (nuScenes detection/tracking format)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.local_rank() if is_dist else 0)

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=is_dist,
        shuffle=False,
    )

    results_data = None
    info_map = None
    if args.result is not None:
        with open(args.result, "r") as f:
            raw_res = json.load(f)
            results_data = raw_res.get("results", raw_res)
        info_map = {info["token"]: info for info in dataset.data_infos}
        print(f"Loaded {len(results_data)} result entries from {args.result}")

    # build the model and load checkpoint (only needed if running live model inference)
    if args.mode == "pred" and args.result is None:
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])

        if args.result is not None:
            bboxes, scores, labels = global_to_lidar_boxes(
                results_data.get(metas["token"], []),
                info_map.get(metas["token"], None),
                cfg.object_classes,
            )
            if bboxes is not None:
                if args.bbox_classes is not None:
                    indices = np.isin(labels, args.bbox_classes)
                    bboxes = bboxes[indices]
                    scores = scores[indices]
                    labels = labels[indices]

                if args.bbox_score is not None:
                    indices = scores >= args.bbox_score
                    bboxes = bboxes[indices]
                    scores = scores[indices]
                    labels = labels[indices]
        else:
            if args.mode == "pred":
                with torch.inference_mode():
                    outputs = model(**data)

            if args.mode == "gt" and "gt_bboxes_3d" in data:
                bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
                labels = data["gt_labels_3d"].data[0][0].numpy()

                if args.bbox_classes is not None:
                    indices = np.isin(labels, args.bbox_classes)
                    bboxes = bboxes[indices]
                    labels = labels[indices]

                bboxes[..., 2] -= bboxes[..., 5] / 2
                bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
            elif args.mode == "pred" and "boxes_3d" in outputs[0]:
                bboxes = outputs[0]["boxes_3d"].tensor.numpy()
                scores = outputs[0]["scores_3d"].numpy()
                labels = outputs[0]["labels_3d"].numpy()

                if args.bbox_classes is not None:
                    indices = np.isin(labels, args.bbox_classes)
                    bboxes = bboxes[indices]
                    scores = scores[indices]
                    labels = labels[indices]

                if args.bbox_score is not None:
                    indices = scores >= args.bbox_score
                    bboxes = bboxes[indices]
                    scores = scores[indices]
                    labels = labels[indices]

                bboxes[..., 2] -= bboxes[..., 5] / 2
                bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
            else:
                bboxes = None
                labels = None

        if args.result is None:
            if args.mode == "gt" and "gt_masks_bev" in data:
                masks = data["gt_masks_bev"].data[0].numpy()
                masks = masks.astype(np.bool)
            elif args.mode == "pred" and "masks_bev" in outputs[0]:
                masks = outputs[0]["masks_bev"].numpy()
                masks = masks >= args.map_score
            else:
                masks = None
        else:
            masks = None

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()
