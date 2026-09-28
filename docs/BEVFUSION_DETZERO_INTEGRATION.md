# Kiến trúc BEVFusion → DetZero hiện tại

Hướng dẫn và lệnh thực thi: [RUN_TRACKING_AND_REFINING.md](RUN_TRACKING_AND_REFINING.md).
Code tích hợp là `DetZero-main-2/integration/`, sử dụng GRM/PRM, không dùng CRM.

## Ranh giới dữ liệu

BEVFusion xử lý camera + LiDAR, xuất box global qua exporter nuScenes chuẩn:
`translation` là tâm hình học, `size=[w,l,h]`, quaternion wxyz.
Bridge chuyển sang `[x,y,z,l,w,h,yaw]`, nối sample token với scene/timestamp và dùng
`T_global_lidar = T_global_ego @ T_ego_lidar` khi crop điểm. Chỉ đọc dự đoán và
metadata cảm biến/pose; không dùng annotation GT để nối track hoặc chạy refiner.

## Tracking

Chạy DetZero Kalman + IoU BEV/GNN riêng từng scene và từng class nuScenes.
Giữ nguồn `(sample_token, source_index)` cho mỗi observation; không ghép car với truck
chỉ vì cả hai dùng checkpoint Vehicle. Delta-time lấy từ timestamp cho cả hai chiều.
Frame rỗng vẫn được giữ. Track ít observation được lọc trước refining.

## Refining

GRM gom điểm đã chuẩn hóa theo từng box, ước lượng kích thước dùng chung cho track.
PRM chuẩn hóa theo box giữa cửa sổ, sửa tâm/yaw, sau đó chuyển lại global.
PRM chỉ chạy trên tọa độ có tính nhất quán theo thời gian. Cửa sổ tối đa 200 frame,
track dài được chia có context. Không có Confidence Refining Module: score detector được giữ.

`size_policy` và `z_policy` là các bước hậu xử lý có thể thay kết quả model; ghi chúng
trong `run_report.json` và so sánh ablation. Các checkpoint hiện có khớp cấu trúc model,
nhưng độ chính xác transfer Waymo→nuScenes/VF phải được đo trên dữ liệu độc lập.

## Hai chế độ tọa độ

- nuScenes-mini có ego pose: hỗ trợ tracking + GRM + PRM offline.
- VF6_01 5 Hz có identity pose do thiếu NAV/IMU: đủ để thử detection từng frame.
  Chỉ bật association + GRM ở chế độ `ego` tường minh; output mang nhãn diagnostic,
  không xuất vận tốc global và không bật PRM. Cần pose thật hoặc xác nhận sensor đứng yên
  để chạy đầy đủ. Không thể sửa thiếu odometry bằng đổi DELTA_T hoặc checkpoint.

## So sánh công bằng

Cùng detector JSON đầu vào, frame/token, GT và chính sách range/score. Chạy riêng:
BEVFusion, +tracking, +tracking/GRM, +tracking/GRM/PRM (khi pose cho phép).
Detection output bảo toàn số candidate, class, score và các box không xử lý;
không gọi đó là cơ chế xóa toàn bộ false positive. Tracking output có ID là một sản phẩm riêng.
Báo cáo frame-level detection riêng với tracking, không dùng video đẹp hơn làm bằng chứng metrics tăng.
VF dùng custom evaluator cho ba class có GT, không báo NDS khi không có velocity/attribute đáng tin cậy.
