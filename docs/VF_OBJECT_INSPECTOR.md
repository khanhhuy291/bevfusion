# VF Object Inspector: camera, BEV và 3D

Viewer chạy cục bộ, chỉ đọc ảnh/LiDAR/metadata/prediction. Không chạy model, không sửa data, không cần CRM, không cần Node/npm/CDN. Giao diện so sánh **GT / BEVFusion / BEVFusion + DetZero**, dùng bản DetZero trong `grm_prm_zfix_OTAgmB`.

## Mở trên máy Mac hiện tại

```text
http://127.0.0.1:8765/?frame=80&source=2
```

Nếu server đã dừng, chạy ở terminal:

```bash
cd /Users/khanhhuy/bevfusion
python3 apps/vf-inspector/server.py \
  --data-root data/nuscenes_vf6_01_adapted \
  --run-dir outputs/vf-car-truck-sweeps2-01 \
  --port 8765
```

Python cần `numpy`, `opencv-python`, `pyquaternion`, `shapely` (môi trường hiện tại đã có). Server dùng standard-library HTTP; không cần FastAPI/Flask.

## Thao tác

- Thanh frame: slider, nhập số, nút trước/sau và phát. Frame đánh số từ 0 theo timestamp. Phím trái/phải chuyển frame; Space phát/dừng.
- Chọn camera bằng 6 thumbnail. Khi chọn object, `Tự chọn camera` chọn góc có GT/detection hiển thị lớn nhất.
- Chọn object từ danh sách hoặc click box trên camera/BEV/3D. `Tập trung` zoom vào object; `Chỉ object đang chọn` ẩn các box khác.
- GT xanh lá, BEVFusion cam nét đứt, DetZero xanh dương. Bật/tắt từng bộ box độc lập.
- Camera: cuộn zoom, kéo pan. BEV: cuộn zoom, kéo pan; X tiến lên, Y trái. 3D: kéo để xoay, Shift + kéo để pan, cuộn zoom.
- Nút phóng to mở mỗi view trên toàn màn hình ứng dụng; Escape đóng. Nút lưu ảnh xuất PNG của view hiện tại kèm frame/source/IoU.
- Bảng object hiển thị IoU 3D, kích thước, tâm ego, lệch XY và đáy so GT. Đáy so GT tính theo **global Z**, không phải khoảng cách đến mặt đường được fit.
- Score là **ngưỡng hiển thị**, không chạy lại tracking/refining. GT zero-point bị ẩn mặc định; có tùy chọn bật lại.
- Bộ lọc class hiện hỗ trợ các nhóm đánh giá: car (gồm truck), motorcycle, pedestrian. Chế độ `Không ghép GT` hiện prediction chưa có cặp để kiểm tra false-positive nghi vấn; không khẳng định chúng là FP chính thức.

## Quy tắc ghép và tọa độ

Ghép GT với baseline cùng nhóm class bằng khoảng cách XY, greedy một–một trong bán kính chọn (mặc định 1 m). DetZero được lấy **cùng source index** với baseline; không ghép lại riêng DetZero để chọn kết quả tốt hơn. Đây là cặp chẩn đoán, không phải matcher TP/FP/AP. Trong cảnh đông object, cần xác minh identity trên ảnh.

Frame/GT/ego pose/calibration lấy từ `run_dir/dataset_local/v1.0-mini`, vì prediction đã localize tọa độ. Sensor thật lấy từ `data-root`. Không dùng metadata UTM nguồn để chiếu prediction đã localize. Camera dùng ego pose riêng tại timestamp camera; BEV/3D dùng ego pose của LiDAR keyframe. LiDAR chuyển bằng calibrated sensor → ego, không cộng sai origin world. Lưới 3D Z=0 là mốc tọa độ ego, **không phải ground plane**.

Camera box được clip ở near plane trước chiếu. Viewer 3D dùng phép chiếu perspective tương tác của LiDAR xyz và 8 đỉnh box thật; hiển thị wireframe. LiDAR được lấy mẫu tối đa 18.000 điểm/frame trong ROI ±65 m, Z từ -5 đến 10 m; ảnh và box giữ nguyên.

Để so sánh từng object, tập prediction hiển thị được lọc theo class/score/range **của baseline**, rồi giữ cùng source ở DetZero, kể cả box DetZero vượt range. Vì vậy số box hiển thị có thể khác số candidate trong metric của từng phương pháp. Không dùng count của viewer thay cho evaluator.

## Máy công ty

Cách nhanh nhất là chạy viewer **ngoài Docker**, đọc data và output trên host:

```bash
cd /home/khanhhuy/bevfusion
python3 apps/vf-inspector/server.py \
  --data-root data/nuscenes_vf6_01_adapted \
  --run-dir outputs/vf-car-truck-sweeps2-01 \
  --port 8765
```

Nếu Python host thiếu dependency, dùng một môi trường Python có bốn thư viện ở trên. Mở `http://127.0.0.1:8765` trong trình duyệt máy công ty.

Nếu chạy trong container thì cần đưa cổng 8765 ra host (port mapping hoặc host network đã cấu hình khi tạo container), và bind server `--host 0.0.0.0`. Container đang có không tự mở thêm cổng bằng `docker start`; không cần tạo lại container detection chỉ để xem ảnh.

## Data hoặc run khác

```bash
python3 apps/vf-inspector/server.py \
  --data-root data/nuscenes_vf6_01_adapted \
  --run-dir outputs/vf-car-truck-sweeps2-01 \
  --refined outputs/vf-car-truck-sweeps2-01/grm_prm_zfix_OTAgmB/results_nusc_detzero_refined.json \
  --port 8765
```

Thay các đường dẫn tương ứng. Run phải có `dataset_local/v1.0-mini` và `baseline/results_nusc_center_corrected.json`. Refined JSON phải giữ nguyên số lượng/class/score/source order với baseline; không dùng JSON tracking hoặc gap-filled. Đường dẫn data/source trong info dialog ghi rõ nguồn đang đọc.

## Kiểm tra đã thực hiện

API trên 156 frame, ảnh thật/thumbnail, point binary xyz, IoU object mẫu, kích thước ảnh và xử lý input sai đã kiểm tra. Phép ghép giữ source và biến đổi góc camera có kiểm tra hồi quy. Trình duyệt tự động chưa kết nối được trong phiên này nên chưa xác nhận giao diện bằng browser QA thực tế.
