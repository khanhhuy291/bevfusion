# Inference epoch_3.pth trên VF NAV calibration

Config: `configs/phenikaa_undistort/det/transfusion/secfpn/camera+lidar/swint_v0p075/nav_calib_inference.yaml`.
Checkpoint local: `pretrained/epoch_3.pth` (epoch 3, metadata MMCV 1.4.0).
582 tên tensor trùng model BEVFusion gốc, 5 tensor khác shape thuộc head
10 → 3 lớp. Chưa kiểm tra strict load/forward CUDA cho checkpoint mới.

Config giữ kiến trúc swint_v0p075, thứ tự car/motorcycle/pedestrian từ config
mentor, một keyframe không sweep, không radar, không GT đầu vào. Chỉ adapter
inference thay đường dẫn, loader không hỗ trợ dataset_root, workers=0 và
tắt tải pretrained Swin. Config training gốc không bị sửa.

Code mentor chưa có: `heatmap_nms_exempt_class_ids: [2]` là giả định chuyển
quy tắc miễn heatmap NMS của pedestrian sang index 2. Không thêm trọng số.
Khi thiếu tùy chọn này, head giữ hành vi nuScenes/Waymo cũ. `nms_type: null`
giữ nguyên config mentor, không dùng bảng post-NMS 10 lớp.

Không chứng nhận tương đương pipeline mentor: checkpoint không chứa config
hoặc CLASSES. nav_calib đã warp ảnh/đổi hệ LiDAR theo nuScenes, có thể khác
dữ liệu phenikaa_nuscenes_undistort khi training. Không tự chạy correction Z
của checkpoint bevfusion-det.pth lên model retrain này.

## Đồng bộ sang máy công ty

Chép toàn bộ `configs/phenikaa_undistort/`, `pretrained/epoch_3.pth`, bản mới
`mmdet3d/models/heads/bbox/transfusion.py` và `tools/test.py`. Config leaf cần
các default.yaml cha. Giữ các bản sửa Blackwell hiện có trong hai file Python.
Không cần rebuild Docker hoặc CUDA extension cho các thay đổi Python này.

Trong container bevfusion-detection:

```bash
cd /workspace/bevfusion
export OMP_NUM_THREADS=2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
set -o pipefail
mkdir -p outputs/phenikaa-epoch3-nav/smoke

python tools/data_converter/create_vf_infos.py \
  --root-path data/nuscenes_vf6_01_5hz_nav_calib \
  --version v1.0-trainval --max-sweeps 0 \
  --output-name phenikaa_epoch3_infos_val.pkl

torchpack dist-run -np 1 python tools/test.py \
  configs/phenikaa_undistort/det/transfusion/secfpn/camera+lidar/swint_v0p075/nav_calib_inference.yaml \
  pretrained/epoch_3.pth --strict-checkpoint \
  --out outputs/phenikaa-epoch3-nav/smoke/predictions.pkl --format-only \
  --eval-options jsonfile_prefix=outputs/phenikaa-epoch3-nav/smoke \
  --cfg-options data.test.ann_file=data/nuscenes_vf6_01_5hz_nav_calib/phenikaa_epoch3_infos_val.pkl data.test.load_interval=155 \
  2>&1 | tee outputs/phenikaa-epoch3-nav/smoke/run.log
```

Với bộ 155 frame hiện tại, load_interval=155 chọn một frame. Strict load
phải thành công và xuất results_nusc.json trước khi chạy hết bộ. Chạy full
với output riêng, load_interval=1; giữ ann_file như trên. Dùng evaluator
`tools/evaluate_vf_comparison.py` cho VF, không dùng evaluator mini_val/NDS.

## Kiểm tra local

Đã resolve kế thừa YAML và kiểm tra 3 lớp, sparse_shape, bounds, đường dẫn,
pipeline không annotation/sweep và không truyền dataset_root vào point
loader. Đã chạy nhánh chọn heatmap trên tensor CPU: override [2], override
rỗng, legacy nuScenes [8,9], legacy Waymo [1,2], từ chối index ngoài phạm vi.
Đây chưa phải kiểm tra end-to-end model CUDA.
