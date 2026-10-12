# So sánh detection VF trước/sau GRM và PRM bằng IoU 3D

`tools/evaluate_vf_comparison.py` xuất đồng thời AP theo khoảng cách tâm XY
hiện có và AP theo IoU 3D tại các ngưỡng mặc định 0.25, 0.5, 0.7.
Không cần checkpoint CRM, chạy lại BEVFusion hoặc chạy lại GRM/PRM nếu đã có
hai JSON detection. Không dùng JSON tracking thay cho JSON detection refined.

## Chạy trên máy công ty

Đồng bộ cả `tools/evaluate_vf_comparison.py` và `tools/vf_iou3d.py`.
Evaluator dùng NumPy, Shapely, pyquaternion và nuscenes-devkit, không cần CUDA
extension. Trong repo hoặc container, thay hai đường dẫn JSON bằng output
thực tế của cùng một lần chạy:

```bash
python tools/evaluate_vf_comparison.py \
  --data-root data/nuscenes_vf6_01_adapted \
  --version v1.0-mini \
  --baseline outputs/YOUR_RUN/baseline/results_nusc_center_corrected.json \
  --refined outputs/YOUR_RUN/grm_prm/results_nusc_detzero_refined.json \
  --classes car,motorcycle,pedestrian \
  --iou-thresholds 0.25 0.5 0.7 \
  --output outputs/YOUR_RUN/comparison_iou3d.json
```

Nếu dùng runner `tools/run_vf_adapted_ablation.py` cho lần chạy mới, đồng bộ
file runner này nữa: `comparison.csv` đã có mAP/AP IoU 3D và delta theo
ngưỡng cho baseline, tracking, GRM và GRM+PRM. Đánh giá lại JSON có sẵn bằng
lệnh trên không cần chạy lại toàn bộ runner/inference.

Nếu checkpoint không dùng correction center legacy, dùng JSON baseline thực
tế của checkpoint đó; không tự áp correction theo ví dụ tên file. Hai nhánh
phải cùng hệ tọa độ và quy ước geometric center. Nếu thí nghiệm gốc đã gộp
truck vào VF Car, thêm `--car-includes-truck` cho cả hai nhánh. Giữ nguyên
chính sách range và `--include-zero-points` giữa các lần so sánh.

## Đọc kết quả

- `baseline/refined.custom_mAP`: metric khoảng cách tâm cũ, giữ nguyên.
- `baseline/refined.iou3d.mAP_by_threshold`: mAP 3D ở từng ngưỡng.
- `baseline/refined.iou3d.per_class`: AP, TP, FP, FN, precision, recall từng lớp/ngưỡng.
- `delta_iou3d_mAP_by_threshold`: refined trừ baseline; đơn vị 0–1,
  nhân 100 để được chênh lệch điểm phần trăm.
- `iou3d.mean_mAP`: trung bình qua các ngưỡng được chọn, là metric custom.

AP 3D ghép prediction theo confidence giảm dần với GT cùng frame/class có
IoU cao nhất chưa được ghép; mỗi GT chỉ được ghép một lần. IoU đúng bằng
ngưỡng được chấp nhận. Score bằng nhau giữ thứ tự token tăng dần rồi thứ tự
prediction trong JSON. AP là diện tích dưới precision envelope nội suy qua
toàn bộ recall [0,1], không áp cutoff recall/precision 0.1 của nuScenes.
TP/FP/FN là số đếm trên toàn bộ candidate sau lọc class/range, không phải
tại confidence 0.1. Evaluator không tự lọc confidence hoặc sửa score.

IoU dùng diện tích giao hai hình chữ nhật xoay nhân chiều cao giao, chia thể
tích hợp. Box dùng global geometric center, size `[width,length,height]`,
yaw quanh Z. Box có roll/pitch bị từ chối rõ ràng; không âm thầm làm phẳng.
Sai convention Z, thứ tự kích thước hoặc frame tọa độ có thể làm IoU rất thấp.

## Quan hệ với DetZero gốc

DetZero gốc dùng evaluator Waymo: mặc định IoU 3D, ngưỡng Vehicle 0.7,
Pedestrian/Cyclist 0.5 và matcher Hungarian. Metric mới này dùng greedy
theo confidence và full-recall AP; không gọi nó là Waymo mAP/mAPH hay
nuScenes chính thức. Adapter tracking VF hiện dùng IoU BEV để association;
đó là một bước khác với đánh giá prediction bằng GT.

Thiếu CRM vẫn có thể đo thay đổi hình học từ GRM/PRM bằng IoU 3D. AP có tăng
hay không phụ thuộc box thực sự gần GT hơn và vượt các ngưỡng IoU; không
được bảo đảm chỉ vì đã chạy refinement.
