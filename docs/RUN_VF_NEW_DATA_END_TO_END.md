# Chạy end-to-end BEVFusion → DetZero khi có dataset VF mới

Chạy trong container `bevfusion-detection` trên máy công ty. Pipeline gồm: kiểm tra data → tạo infos → BEVFusion → sửa convention tâm của checkpoint hiện tại → tracking → GRM/PRM → đánh giá AP. Không cần CRM.

## 1. Data và code cần có

Dataset phải **đã chuyển sang định dạng VF/nuScenes giống các lần chạy hiện tại**, không phải thư mục ảnh/point cloud hoặc rosbag thô. Runner hiện hỗ trợ **một scene**, metadata ở `v1.0-mini/`, đủ 6 camera + `LIDAR_TOP`, các file sensor đúng đường dẫn metadata, calibration và ego pose thực đo để chạy 2 sweep + PRM. Không dùng pose identity thay cho chuyển động thật.

Để đánh giá AP ba nhóm hiện tại, data cần có GT đánh giá được cho car/truck, motorcycle và pedestrian. Runner này chưa hỗ trợ tự bỏ class thiếu GT hoặc chạy end-to-end cho data không có nhãn. Dataset nhiều scene hoặc version khác cần điều chỉnh runner trước.

Trên Ubuntu, đặt data mới trong `/home/khanhhuy/bevfusion/data/`; bind mount sẽ hiển thị dưới `/workspace/bevfusion/data/` trong container. Đồng bộ bản code đã sửa, gồm:

- `tools/run_vf_adapted_ablation.py`, `tools/validate_nuscenes_data.py`.
- `tools/data_converter/create_vf_infos.py`, `tools/correct_legacy_box_centers.py`.
- `tools/evaluate_vf_comparison.py`, `tools/vf_iou3d.py`.
- `DetZero-main-2/integration/run_pipeline.py` và các helper của `integration/`.
- `DetZero-main-2/tracking/detzero_track/models/tracking_modules/track_manager.py` có sửa reverse tracking.

Dùng cùng config và checkpoint hiện tại:

- Config: `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/vf6_nav/sweeps2.yaml`.
- Detector: `pretrained/bevfusion-det.pth`.
- GRM/PRM: 6 checkpoint vehicle/pedestrian/cyclist tại `DetZero-main-2/checkpoints/`.

`--legacy-center-correction` dành cho convention đã kiểm tra của chính checkpoint detector này; nếu thay checkpoint, phải kiểm tra lại convention, không tự áp cùng correction.

## 2. Mở container

Trên terminal Ubuntu, nếu container đang dừng:

```bash
sudo docker start -ai bevfusion-detection
```

Nếu container đang chạy, mở shell bằng:

```bash
sudo docker exec -it bevfusion-detection /bin/bash
```

## 3. Lệnh end-to-end cho data mới

**Chỉ sửa biến `vf_e2e_data` thành đường dẫn dataset mới trong container.** Ví dụ dưới dùng dataset adapted đang có; khi nhận dataset khác, thay giá trị này. Output tự có tên mới ở mỗi lần chạy nên không ghi đè run cũ.

Dán toàn bộ block vào Bash trong container:

```bash
(
set -euo pipefail
cd /workspace/bevfusion
export PATH="/opt/venv/bin:$PATH"

# Thay dòng này khi nhận dataset mới; dùng đường dẫn trong container.
vf_e2e_data=data/nuscenes_vf6_01_adapted

test -f "$vf_e2e_data/v1.0-mini/sample.json"
mkdir -p outputs
vf_e2e_output="$(mktemp -d outputs/vf-e2e-sweeps2-zfix_XXXXXX)"
printf 'Dataset: %s\nOutput: %s\n' "$vf_e2e_data" "$vf_e2e_output"

/opt/venv/bin/python tools/run_vf_adapted_ablation.py \
  --data-root "$vf_e2e_data" \
  --sweeps 2 \
  --legacy-center-correction \
  --car-includes-truck \
  --preserve-bottom-classes motorcycle \
  --output-dir "$vf_e2e_output" \
  2>&1 | tee "$vf_e2e_output/end_to_end.log"

# Runner đã đánh giá từng nhánh; lệnh này tạo thêm JSON so sánh baseline/full.
/opt/venv/bin/python tools/evaluate_vf_comparison.py \
  --data-root "$vf_e2e_output/dataset_local" \
  --version v1.0-mini \
  --baseline "$vf_e2e_output/baseline/results_nusc_center_corrected.json" \
  --refined "$vf_e2e_output/grm_prm/results_nusc_detzero_refined.json" \
  --classes car,motorcycle,pedestrian \
  --car-includes-truck \
  --iou-thresholds 0.25 0.5 0.7 \
  --output "$vf_e2e_output/comparison_iou3d.json" \
  2>&1 | tee "$vf_e2e_output/evaluation_iou3d.log"

printf '\nHoan tat. Gui thu muc ket qua: %s\n' "$vf_e2e_output"
)
```

Script runner tự đặt các biến môi trường cần thiết cho Torchpack/DetZero. Threshold đưa detection vào tracking/refining hiện **cố định 0.1**; lệnh này chưa chạy threshold sweep. Nếu một bước lỗi, block dừng và không chạy các bước sau; đọc `end_to_end.log` và log của bước đó trong `logs/`. Vì biến nằm trong subshell, xem tên output từ dòng `Output:` hoặc log terminal.

## 4. Pipeline và policy đang chạy

| Bước | Đầu vào / kết quả |
|---|---|
| Kiểm tra data | Đọc metadata, camera/LiDAR và kiểm tra dependency/GPU |
| Localize và infos | Tạo `dataset_local/`, giữ calibration/rotation/timestamp, trừ cùng origin khỏi pose và GT; tạo infos mới ngay trên máy công ty |
| BEVFusion | Camera + LiDAR với tối đa 2 sweep; lưu baseline detection |
| Center correction | Chuyển convention legacy của checkpoint sang tâm hình học |
| Tracking | Tách car, truck, motorcycle, pedestrian; score >=0.1; association CPU, reverse single-stage cập nhật box measurement |
| GRM | Ước lượng size; giữ đáy riêng motorcycle khi ghép box |
| GRM + PRM | Size từ GRM, XY/yaw từ PRM; motorcycle tính Z theo đáy detector/track và height mới; class khác dùng Z PRM |
| Đánh giá | AP khoảng cách và AP IoU 3D 0.25/0.5/0.7 trên cùng dataset/frame/GT policy |

Runner chạy độc lập bốn nhánh: `baseline`, `tracking`, `grm`, `grm_prm`. GRM/PRM chạy CUDA; detector score được giữ, không dùng CRM. Car và truck tách nhau khi tracking/refining, gộp thành nhóm Car khi đánh giá.

Giữ đáy motorcycle ngăn GRM nâng đáy chỉ vì giảm chiều cao; không phải fit mặt đường và không bảo đảm loại bỏ sai số Z vốn có của detector trên data mới. Xem [VF_BOX_Z_FIX.md](VF_BOX_Z_FIX.md) để hiểu bản sửa và kết quả kiểm chứng. AP là **VF custom**, không phải benchmark Waymo/nuScenes chính thức.

`dataset_local` dùng symlink đến sensor nguồn. Giữ thư mục data nguồn sau khi chạy để có thể crop/render lại. Dùng metadata `dataset_local` cùng prediction của run; không trộn với GT/pose nguồn khi đánh giá hoặc chiếu box.

## 5. Kết quả cần gửi để phân tích

Output trong container: `outputs/vf-e2e-sweeps2-zfix_<mã>/`. Trên host bind mount hiện tại: `/home/khanhhuy/bevfusion/outputs/` với cùng tên thư mục.

| File / thư mục | Nội dung |
|---|---|
| `comparison_iou3d.json` | So sánh baseline với full DetZero, AP từng class tại 0.25/0.5/0.7 |
| `comparison.csv` | Bảng tổng hợp bốn nhánh, có AP khoảng cách và IoU 3D |
| `baseline/results_nusc_center_corrected.json` | Detection BEVFusion dùng để so sánh |
| `tracking/`, `grm/`, `grm_prm/` | Detection, tracks, metrics và run report của từng nhánh |
| `dataset_local/` | Metadata/infos đúng hệ tọa độ của run |
| `experiment.json` | Source data, checkpoint, sweeps, class và policy giữ đáy |
| `logs/`, `end_to_end.log` | Log từng bước và toàn pipeline |
| `source_data_audit.json`, `infos_data_audit.json` | Kết quả kiểm tra data và infos |

Để phân tích số liệu, gửi cả thư mục output hoặc ít nhất `comparison_iou3d.json`, `comparison.csv`, `experiment.json`, `grm_prm/run_report.json`. Để kiểm tra hình học/chiếu box, cần thêm prediction từng nhánh, metadata `dataset_local` và sensor nguồn.

Với data mới, chạy một lần ở threshold 0.1 trước để kiểm tra pipeline và độ cao box; sau đó mới so sánh threshold trên cùng dataset. Nếu chỉ chạy lại DetZero trên baseline đã có, dùng lệnh ở [VF_BOX_Z_FIX.md](VF_BOX_Z_FIX.md), không cần chạy detector lần nữa.
