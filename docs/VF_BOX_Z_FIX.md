# VF box height / Z correction

**Khi nhận data mới và cần chạy cả BEVFusion:** dùng
[RUN_VF_NEW_DATA_END_TO_END.md](RUN_VF_NEW_DATA_END_TO_END.md).
Lệnh bên dưới dành cho chạy lại DetZero trên baseline đã lưu.

## Nguyên nhân đã xác định

Trong run `vf-car-truck-sweeps2-01`, GRM giảm chiều cao motorcycle trung vị khoảng 0.488 m. Với `z_policy=center`, tâm GRM-only giữ Z của tracking, còn PRM full chỉ đổi Z trung vị khoảng -0.002 m so với baseline. Do `bottom = center_z - height/2`, đáy cuối cùng tăng trung vị khoảng 0.241 m trên 1.563 cặp gần GT. Vì vậy box nhìn như bị nhấc lên ngay cả khi tâm gần như không tăng. PRM cũng có thể thay đổi tâm Z trực tiếp ở từng box.

Đã kiểm tra mọi class: car gần như giữ đáy theo trung vị, truck hạ xuống, pedestrian không có xu hướng nâng chung nhưng có từng trường hợp. Sáu class không được chọn refine giữ nguyên geometry. Xem `outputs/vf-car-truck-sweeps2-01/analysis/Z_BOTTOM_REPORT_VI.md` và `z_bottom_audit.json`.

Ngoài ra, reverse tracker gọi `KalmanFilter.update(two_stage=True)` cả khi association là single-stage. Filter trả box dự đoán, nhưng source_index vẫn gắn với detection đã match. Exporter vì vậy thay detection bằng prediction tracker ở 6 box (4 car, 1 truck, 1 motorcycle).

## Thay đổi code

- `track_manager.py`: single-stage reverse match cập nhật từ measurement; chế độ two-stage giữ hành vi cũ.
- `integration/run_pipeline.py`: thêm `--preserve-bottom-classes motorcycle`. Class được chỉ định giữ đáy detector/track khi ghép size GRM với pose PRM. XY/yaw vẫn lấy PRM; Z tính lại theo đáy và chiều cao mới. `--z-policy center` tiếp tục áp dụng cho các class còn lại. Policy từng track ghi vào run report.
- Runner VF và threshold sweep mặc định bật giữ đáy motorcycle. Runner general vẫn cần truyền flag rõ ràng. Để chạy lại policy cũ ở runner VF, dùng `--preserve-bottom-classes ''`; ở sweep dùng `VF_SWEEP_BOTTOM_CLASSES='' bash tools/run_vf_threshold_sweep.sh`.
- `tools/apply_vf_bottom_anchor.py`: áp dụng riêng sửa ghép Z trên output lưu, không đọc GT và không chạy lại model/tracker. Không ghi đè file nguồn.

Giữ đáy detector là một policy được kiểm chứng trên dataset này, không phải ước lượng mặt đường. Nó thay thế hiệu chỉnh Z của PRM cho class được chọn; nếu detector đã sai đáy thì sai số đó còn lại. Không áp dụng tự động lên mọi class.

## Kết quả kiểm tra

Chạy lại tracker CPU trên đủ 156 frame: 389 tracks, 6.381 detections được thay; lệch tâm tuyệt đối lớn nhất so với detector 3.813e-6 m (float32), lệch kích thước 0. Report: `outputs/vf-car-truck-sweeps2-01/z_fix/tracking_replay_report.json`.

Giữ nguyên prediction GRM/PRM đã lưu, threshold 0.1, đổi riêng policy đáy motorcycle (1.587 box):

| IoU 3D | AP motorcycle cũ (%) | AP motorcycle giữ đáy (%) | mAP 3 class cũ (%) | mAP giữ đáy (%) |
|---|---:|---:|---:|---:|
| 0.25 | 73.334 | 75.580 | 56.477 | 57.226 |
| 0.50 | 27.130 | 41.970 | 29.655 | 34.602 |
| 0.70 | 0.558 | 2.740 | 1.828 | 2.556 |

Car và pedestrian giữ nguyên AP trong phép thử chỉ motorcycle. Thử giữ đáy mọi class đạt mAP@0.7 2.336%, thấp hơn chỉ motorcycle 2.556%; không chọn policy toàn bộ class. Các số này là phép thử trên **prediction lưu**, chưa phải kết quả rerun GRM/PRM với tracker mới; features của vài track có thể thay đổi sau sửa reverse tracker. Cần chạy lệnh dưới để đo kết quả toàn pipeline trên GPU máy công ty.

## Chạy lại full trong Docker máy công ty

Đồng bộ các file code đã sửa trước. Chạy trong container `bevfusion-detection`, ở `/workspace/bevfusion`. Dataset_local của run gốc phải còn các symlink sensor hợp lệ trong container. Lệnh dùng cùng baseline/weights/threshold, tạo thư mục mới và chạy lại tracking + GRM + PRM; không chạy lại BEVFusion và không dùng CRM.

```bash
cd /workspace/bevfusion
(
set -euo pipefail
export OMP_NUM_THREADS=2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="/workspace/bevfusion${PYTHONPATH:+:$PYTHONPATH}"
vf_z_run=outputs/vf-car-truck-sweeps2-01
vf_z_output="$(mktemp -d "$vf_z_run/grm_prm_zfix_XXXXXX")"
/opt/venv/bin/python DetZero-main-2/integration/run_pipeline.py \
  --results_path "$vf_z_run/baseline/results_nusc_center_corrected.json" \
  --data_root "$vf_z_run/dataset_local" \
  --version v1.0-mini \
  --tracking_cfg DetZero-main-2/tracking/tools/cfgs/tk_model_cfgs/nuscenes_detzero_track.yaml \
  --checkpoint_dir DetZero-main-2/checkpoints \
  --coordinate-mode global --device cuda --tracking-device cpu \
  --classes car,truck,motorcycle,pedestrian \
  --min_score 0.1 --max-gap-seconds 0.6 --size_policy grm \
  --z-policy center --preserve-bottom-classes motorcycle \
  --intensity-mode unit --refinement full --output_dir "$vf_z_output" \
  2>&1 | tee "$vf_z_output/pipeline.log"
/opt/venv/bin/python tools/evaluate_vf_comparison.py \
  --data-root "$vf_z_run/dataset_local" --version v1.0-mini \
  --baseline "$vf_z_run/baseline/results_nusc_center_corrected.json" \
  --refined "$vf_z_output/results_nusc_detzero_refined.json" \
  --classes car,motorcycle,pedestrian --car-includes-truck \
  --iou-thresholds 0.25 0.5 0.7 --output "$vf_z_output/comparison_iou3d.json" \
  2>&1 | tee "$vf_z_output/evaluation.log"
printf 'Output: %s\n' "$vf_z_output"
)
```

## Sửa và đánh giá output đã lưu, không chạy GPU

Lệnh này chỉ đổi policy Z, giữ nguyên các kết quả tracker/GRM/PRM cũ. Cần file output mới chưa tồn tại.

```bash
/opt/venv/bin/python tools/apply_vf_bottom_anchor.py \
  --baseline outputs/vf-car-truck-sweeps2-01/baseline/results_nusc_center_corrected.json \
  --refined outputs/vf-car-truck-sweeps2-01/grm_prm/results_nusc_detzero_refined.json \
  --classes motorcycle \
  --output outputs/vf-car-truck-sweeps2-01/results_nusc_motorcycle_bottom_anchored.json
```

Ảnh chồng GT/baseline/DetZero sau đổi policy đã có tại `outputs/vf-car-truck-sweeps2-01/z_fix/visual_overlay`. Góc ngang frame 80 cho thấy đáy không nâng thêm, IoU 0.744 -> 0.805. Manifest ghi rõ đây là saved-output anchoring.
