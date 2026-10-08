# VF original camera: four branches and before/after videos

Source: `data/nuscenes_vf6_01`, version `v1.0-mini`.
This source already contains undistorted images; it has not received the
virtual-camera homography used in `nuscenes_vf6_01_adapted`.

Local full audit: 156 frames, 1 scene, 15,475 annotations, 515 instances,
5,379 images at 1920x1536 and 307 LiDAR records. Every sensor file was read.
No missing files, invalid image dimensions or nonfinite points were found.
LiDAR has 262,603--288,837 points per record and intensity in [0,255].
GT, sample timestamps and ego poses equal those in the adapted source.
Two-sweep infos were generated and checked: 154 frames have two past sweeps,
one has one, and the initial frame has none. These Mac infos are audit artifacts;
regenerate infos on the company machine with the runner below.

Calibration quaternions are normalized and camera intrinsics match conversion
metadata. Frame-50 camera/LiDAR/GT projections were inspected, but physical
calibration is not independently established. Camera keyframe timestamp offsets
against sample timestamps reach 75.449 ms (median 33.7225 ms).
GT point counts are copied from partner labels, and GT velocity/attributes are
not validated. Use custom AP/TP metrics, not official VF NDS.

## Files to synchronize

Put the complete source dataset under `/home/khanhhuy/bevfusion/data/nuscenes_vf6_01`.
Update these files in the company project:

- `tools/run_vf_adapted_ablation.py`
- `tools/validate_nuscenes_data.py`
- `tools/evaluate_vf_comparison.py`
- `tools/render_vf_videos.py`

The existing converter, configs and pretrained BEVFusion/DetZero checkpoints
must remain available as in the previous successful run.

## Commands inside the existing GPU container

The project is mounted at `/workspace/bevfusion` and Python is `/opt/venv/bin/python`.
Run the following block. The subshell stops if a step fails.

```bash
(
set -e
cd /workspace/bevfusion

/opt/venv/bin/python tools/run_vf_adapted_ablation.py \
  --data-root data/nuscenes_vf6_01 \
  --sweeps 2 \
  --legacy-center-correction \
  --car-includes-truck \
  --output-dir outputs/vf-original-car-truck-sweeps2-01

command -v ffmpeg
/opt/venv/bin/python tools/render_vf_videos.py \
  --data-root outputs/vf-original-car-truck-sweeps2-01/dataset_local \
  --version v1.0-mini \
  --baseline outputs/vf-original-car-truck-sweeps2-01/baseline/results_nusc_center_corrected.json \
  --tracking outputs/vf-original-car-truck-sweeps2-01/grm_prm/results_nusc_detzero_tracking.json \
  --min-score 0.1 \
  --output-dir outputs/vf-original-car-truck-sweeps2-01/videos
)
```

The output directory must be new or empty. If FFmpeg is missing, model outputs
and evaluation results remain complete; provision FFmpeg and rerun only the
renderer. No model inference needs repeating for videos.

The runner checks dependencies and source structure, localizes UTM coordinates,
generates and audits infos, runs detection, corrects the checkpoint-specific box
center convention, runs tracking / GRM / GRM+PRM, and evaluates all four branches.
Tracking/refining classes: car, truck, motorcycle, pedestrian (score >=0.1).
Evaluation groups: car+truck -> Car, motorcycle -> current Rider mapping,
pedestrian -> Pedestrian. Both baseline and refined use the same mapping.

The localized dataset preserves source camera images, camera calibration and
LiDAR files through symlinks. It only subtracts the same scene origin from pose
and GT translations to avoid large-coordinate float32 precision loss. Model
input resize/crop remains the existing deterministic BEVFusion preprocessing;
the different source image aspect ratio changes the visible crop. This run does
not apply the virtual-camera adapter.

## Outputs on the host

`/home/khanhhuy/bevfusion/outputs/vf-original-car-truck-sweeps2-01/`

- `comparison.csv`: four-branch metrics.
- `baseline/`, `tracking/`, `grm/`, `grm_prm/`: predictions and custom metrics.
- `logs/eval_*.log`: evaluation logs; the other stage logs are alongside them.
- `source_data_audit.json`, `infos_data_audit.json`: company-machine audits.
- `experiment.json`: source and evaluation policy.
- `videos/`: baseline video, full DetZero video with track IDs, and ID map.

Compare with `outputs/vf-car-truck-sweeps2-01/comparison.csv` from the adapted run.
Keep checkpoint, sweeps, classes, score thresholds and evaluation policy fixed.
Do not reuse adapted predictions with this source calibration for rendering.


**Dataset mới đủ điều kiện chạy về cấu trúc và định dạng.** Tôi đã kiểm tra:

- **156 frame, 1 scene, 15.475 box GT**, đủ 6 camera và LiDAR.
- Đọc toàn bộ ảnh/LiDAR: không thiếu file, không phát hiện dữ liệu không hợp lệ.
- Tạo và kiểm tra infos 2 sweep thành công.
- GT, timestamp và ego pose giống bản adapted.

**Khác biệt chính:** bản này dùng ảnh **1920×1536 đã undistort**, chưa warp thành camera ảo 1600×900 như bản adapted. Camera keyframe lệch timestamp tới khoảng **75 ms**; calibration và đồng bộ chưa được xác nhận hoàn toàn. Tôi chưa chạy inference GPU tại máy này.

Bạn có thể xem [ảnh kiểm tra LiDAR + GT trên camera](/Users/khanhhuy/bevfusion/outputs/vf_original_audit/projection_frame50/all_cameras_lidar_overlay.png).

**1. Đồng bộ sang máy công ty**

Đặt dataset tại:

```text
/home/khanhhuy/bevfusion/data/nuscenes_vf6_01/
```

Cập nhật các file sau trong project máy công ty:

- [run_vf_adapted_ablation.py](/Users/khanhhuy/bevfusion/tools/run_vf_adapted_ablation.py)
- [validate_nuscenes_data.py](/Users/khanhhuy/bevfusion/tools/validate_nuscenes_data.py)
- [evaluate_vf_comparison.py](/Users/khanhhuy/bevfusion/tools/evaluate_vf_comparison.py)
- [render_vf_videos.py](/Users/khanhhuy/bevfusion/tools/render_vf_videos.py)

**2. Chạy toàn bộ trong container GPU đang dùng**

```bash
(
set -e
cd /workspace/bevfusion

/opt/venv/bin/python tools/run_vf_adapted_ablation.py \
  --data-root data/nuscenes_vf6_01 \
  --sweeps 2 \
  --legacy-center-correction \
  --car-includes-truck \
  --output-dir outputs/vf-original-car-truck-sweeps2-01

command -v ffmpeg

/opt/venv/bin/python tools/render_vf_videos.py \
  --data-root outputs/vf-original-car-truck-sweeps2-01/dataset_local \
  --version v1.0-mini \
  --baseline outputs/vf-original-car-truck-sweeps2-01/baseline/results_nusc_center_corrected.json \
  --tracking outputs/vf-original-car-truck-sweeps2-01/grm_prm/results_nusc_detzero_tracking.json \
  --min-score 0.1 \
  --output-dir outputs/vf-original-car-truck-sweeps2-01/videos
)
```

Lệnh chạy **detection → tracking → GRM → PRM → đánh giá bốn nhánh → hai video trước/sau**. Thư mục output phải mới hoặc rỗng.

Nếu thiếu FFmpeg, kết quả model và đánh giá vẫn đã lưu; bổ sung FFmpeg rồi chạy lại riêng lệnh render.

Script giữ ảnh, LiDAR và calibration của dataset mới. Nó vẫn đưa tọa độ UTM về gốc local để tránh mất độ chính xác, và áp dụng resize/crop đầu vào của BEVFusion.

**3. Kết quả trên máy công ty**

```text
/home/khanhhuy/bevfusion/outputs/vf-original-car-truck-sweeps2-01/
```

Trong đó:

| Kết quả | Đường dẫn tương đối |
|---|---|
| So sánh bốn nhánh | `comparison.csv` |
| Metric từng nhánh | `<nhánh>/custom_metrics.json` |
| Log đánh giá | `logs/eval_*.log` |
| Hai video | `videos/` |
| Tham số thí nghiệm | `experiment.json` |

So sánh với lần adapted **`vf-car-truck-sweeps2-01`** vì cùng dùng 2 sweep và quy tắc **Car = car + truck**.

Tôi cũng lưu hướng dẫn đầy đủ tại [RUN_VF_ORIGINAL_END_TO_END.md](/Users/khanhhuy/bevfusion/docs/RUN_VF_ORIGINAL_END_TO_END.md).