Đã cập nhật code để **tracking/refining cả `car` và `truck`**, rồi **gộp hai class thành Car khi đánh giá VF**. Tracking vẫn giữ class gốc; cả hai dùng model refining `Vehicle`. Kiểm tra evaluator đã pass.

**Trước tiên, đồng bộ 3 file này sang project trên máy công ty:**

- [run_vf_adapted_ablation.py](/Users/khanhhuy/bevfusion/tools/run_vf_adapted_ablation.py)
- [evaluate_vf_comparison.py](/Users/khanhhuy/bevfusion/tools/evaluate_vf_comparison.py)
- [render_vf_videos.py](/Users/khanhhuy/bevfusion/tools/render_vf_videos.py)

**1. Chạy lại toàn bộ trong container**

```bash
cd /workspace/bevfusion

/opt/venv/bin/python tools/run_vf_adapted_ablation.py \
  --sweeps 2 \
  --legacy-center-correction \
  --car-includes-truck \
  --output-dir outputs/vf-car-truck-sweeps2-01
```

Lệnh tự chạy:

1. Chuẩn bị dataset tọa độ local và infos.
2. BEVFusion detection, sửa tâm box.
3. Ba nhánh DetZero: tracking → tracking + GRM → tracking + GRM + PRM.
4. Đánh giá cả bốn nhánh với cùng quy tắc **Car = car + truck**.

DetZero nhận `car,truck,motorcycle,pedestrian`, score ≥ `0.1`. Quy tắc Rider hiện vẫn giữ nguyên: `Rider → motorcycle`.

**Thư mục output phải mới hoặc rỗng.** Script sẽ dừng nếu thư mục trên đã có kết quả.

**2. Khi bước trên báo `ALL FOUR BRANCHES COMPLETED`, tạo hai video**

Kiểm tra FFmpeg:

```bash
command -v ffmpeg
```

Nếu có đường dẫn FFmpeg, chạy:

```bash
/opt/venv/bin/python tools/render_vf_videos.py \
  --data-root outputs/vf-car-truck-sweeps2-01/dataset_local \
  --version v1.0-mini \
  --baseline outputs/vf-car-truck-sweeps2-01/baseline/results_nusc_center_corrected.json \
  --tracking outputs/vf-car-truck-sweeps2-01/grm_prm/results_nusc_detzero_tracking.json \
  --min-score 0.1 \
  --output-dir outputs/vf-car-truck-sweeps2-01/videos
```

Video sau có box đã tracking + refining và ID màu vàng. Truck có thể xuất hiện nếu track được giữ lại.

**3. Kết quả trên máy công ty**

Nếu dùng bind mount như README, kết quả nằm tại:

```text
/home/khanhhuy/bevfusion/outputs/vf-car-truck-sweeps2-01/
├── baseline/
├── tracking/
├── grm/
├── grm_prm/
├── comparison.csv
├── experiment.json
├── logs/
└── videos/
```

Mở `comparison.csv` để so sánh score. Do lần này đổi định nghĩa nhóm Car, hãy so sánh **bốn nhánh trong lần chạy mới**; score cũ dùng quy tắc khác.