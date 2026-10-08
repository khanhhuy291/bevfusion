# Bổ sung box ở frame BEVFusion bỏ sót

Nhánh này dùng track đã lưu, không chạy lại BEVFusion/GRM/PRM. Kết quả cũ được giữ để so sánh.

## Cách xử lý

- Chỉ xử lý khoảng trống giữa hai detection đã quan sát trong cùng track/scene và cùng ID đã xuất.
- Mặc định `tracker`: lấy tâm/yaw dự đoán bởi Kalman ở frame có `source_index=-1`; kích thước lấy từ hai box refined ở hai đầu khoảng trống.
- `interpolate`: nội suy box refined theo timestamp thật; yaw nội suy theo cung ngắn nhất. Có thể lấp frame chưa có phần tử dự đoán trong track, nhưng vẫn phải có hai quan sát của cùng track.
- Không kéo track ra trước lần thấy đầu tiên hoặc sau lần thấy cuối cùng, không nối hai ID khác nhau, không dùng GT.
- `--max-gap-seconds` giới hạn **toàn bộ khoảng thời gian từ detection đầu đến detection cuối**. VF khoảng 5 Hz: một frame mất có span khoảng 0.4 s; hai frame mất có span khoảng 0.6 s. nuScenes mini khoảng 2 Hz: một frame mất có span khoảng 1.0 s, cần chọn khoảng 1.1 s nếu muốn thử.
- Score bổ sung = `min(score hai đầu) × decay^(thời gian tới đầu gần nhất / chu kỳ frame trung vị)`. Với score hai đầu 0.8, một frame thiếu và decay 0.8, score mới khoảng 0.64. Đây là heuristic, không phải CRM hay xác suất đã được hiệu chỉnh.
- Bỏ ứng viên nếu trùng detection gốc hoặc kết quả refined (kể cả detection score thấp), hoặc trùng ứng viên đã nhận: IoU BEV ≥ 0.1 **hoặc** khoảng cách tâm ≤ 0.5 m. Car/truck được xét chung khi chống trùng.
- Lọc theo khoảng cách class, score tối thiểu, tối đa 500 box/frame; không xóa detection cũ để lấy chỗ.
- File detection và file tracking nhận cùng box bổ sung. Metadata của từng box thêm nằm trong `gap_fill_report.json`, không thêm trường lạ vào record nuScenes.
- Box thêm dùng velocity từ chênh lệch tâm refined hai đầu; attribute kế thừa đầu gần nhất. Nên theo dõi cả mAP và các lỗi khi đánh giá nuScenes.

Chế độ tracker chỉ xuất dự đoán còn lưu trong track. Nếu gap dài tới mức tracker đã tách ID hoặc xóa track thì chế độ này không phục hồi được object.

## Đồng bộ code sang máy công ty

Giữ nguyên đường dẫn tương đối của các file:

```text
DetZero-main-2/integration/bridge.py
DetZero-main-2/integration/gap_fill.py
DetZero-main-2/integration/run_pipeline.py
tools/export_detzero_gap_filled.py
tools/render_vf_videos.py
```

## VF raw: xuất, đánh giá và render lại

Chạy trong container đang có dữ liệu và kết quả. Thư mục `GAP_DIR` phải mới hoặc rỗng. Cần FFmpeg cho bước render video.

```bash
(
set -euo pipefail
cd /workspace/bevfusion

GAP_SOURCE_RUN=outputs/vf-original-car-truck-sweeps2-01
GAP_DIR="$GAP_SOURCE_RUN/grm_prm_gap_fill_tracker_01"

/opt/venv/bin/python tools/export_detzero_gap_filled.py \
  --run-dir "$GAP_SOURCE_RUN/grm_prm" \
  --baseline "$GAP_SOURCE_RUN/baseline/results_nusc_center_corrected.json" \
  --data-root "$GAP_SOURCE_RUN/dataset_local" \
  --version v1.0-mini \
  --method tracker \
  --max-gap-seconds 0.6 \
  --score-decay 0.8 \
  --min-score 0.1 \
  --output-dir "$GAP_DIR"

/opt/venv/bin/python tools/evaluate_vf_comparison.py \
  --data-root "$GAP_SOURCE_RUN/dataset_local" \
  --version v1.0-mini \
  --baseline "$GAP_SOURCE_RUN/grm_prm/results_nusc_detzero_refined.json" \
  --refined "$GAP_DIR/results_nusc_detzero_gap_filled.json" \
  --classes car,motorcycle,pedestrian \
  --car-includes-truck \
  --output "$GAP_DIR/comparison_gap_fill.json" \
  | tee "$GAP_DIR/evaluation.log"

/opt/venv/bin/python tools/render_vf_videos.py \
  --data-root "$GAP_SOURCE_RUN/dataset_local" \
  --version v1.0-mini \
  --baseline "$GAP_SOURCE_RUN/baseline/results_nusc_center_corrected.json" \
  --tracking "$GAP_DIR/results_nusc_detzero_tracking_gap_filled.json" \
  --min-score 0.1 \
  --output-dir "$GAP_DIR/videos_id_color"
)
```

Với VF adapted, thay `GAP_SOURCE_RUN` bằng `outputs/vf-car-truck-sweeps2-01`. Không trộn `dataset_local` hoặc baseline của hai lần chạy: tool kiểm tra hash input và timestamp/pose của cache.

## Đọc kết quả

```text
grm_prm_gap_fill_tracker_01/
├── results_nusc_detzero_gap_filled.json
├── results_nusc_detzero_tracking_gap_filled.json
├── gap_fill_report.json
├── comparison_gap_fill.json
├── evaluation.log
└── videos_id_color/
    ├── <scene>_bevfusion.mp4
    ├── <scene>_tracking.mp4
    ├── tracking_id_map.json
    └── tracking_id_colors.json
```

Trong `comparison_gap_fill.json`, `baseline` là **Tracking + GRM + PRM cũ**, `refined` là **cùng kết quả đó + box bổ sung**. `delta_custom_mAP` là chênh lệch dạng 0..1; nhân 100 để có điểm phần trăm.

`gap_fill_report.json → counts.added_boxes` cho biết số box thực sự thêm. Nếu bằng 0, xem các counter `gap_too_long`, `no_tracker_prediction`, `anchors_not_exported_for_track`, `below_min_score`, `duplicate_suppressed`. Không dùng GT để chọn tham số chỉ nhằm tăng score.

Có thể thử nội suy trong một nhánh khác bằng cách đổi `--method interpolate` và `GAP_DIR=.../grm_prm_gap_fill_interpolate_01`. Không ghi đè nhánh tracker để so sánh hai phương pháp.

## Bật trực tiếp khi chạy pipeline mới

Thêm vào lệnh `DetZero-main-2/integration/run_pipeline.py`:

```bash
--fill-missed-detections \
--gap-fill-method tracker \
--gap-fill-max-seconds 0.6 \
--gap-fill-score-decay 0.8 \
--gap-fill-min-score 0.1
```

Pipeline vẫn xuất file observation-only cũ và xuất thêm hai JSON có hậu tố `gap_filled`. Evaluator và renderer phải trỏ tới file mới nếu muốn chấm/hiển thị box bổ sung. Không bật flag thì hành vi xuất trước đây được giữ.

Đây là hậu xử lý offline có sử dụng quan sát tương lai để xác nhận hai đầu gap; không báo cáo như detector online. Box thêm có thể tăng recall nhưng cũng có thể tạo false positive, cần đo kết quả thực tế trên máy công ty.
