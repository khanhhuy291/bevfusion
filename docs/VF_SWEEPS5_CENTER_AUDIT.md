# VF NAV 5 sweep: convention Z và kiểm tra calibration

## Kết luận thực nghiệm

Kiểm tra JSON 5 sweep từ VM, không chạy lại inference. Bản raw được giữ
nguyên; tạo JSON hiệu chỉnh riêng. Chạy kiểm tra độc lập trên JSON kết quả
nuScenes-mini có sẵn, không dùng GT VF để chọn hệ số dịch box.

Với cặp cùng class, greedy theo score giảm dần, XY <0.5 m và score >=0.1,
đo median sai số tuyệt đối theo trục Z LiDAR trên cùng các cặp:

| Dữ liệu | Class | Số cặp | Raw | Sau sửa |
|---|---|---:|---:|---:|
| VF NAV 5 sweep | car | 2653 | 1.060 m | 0.119 m |
| VF NAV 5 sweep | motorcycle | 1732 | 1.077 m | 0.095 m |
| VF NAV 5 sweep | pedestrian | 127 | 0.822 m | 0.101 m |
| nuScenes-mini | car | 1670 | 0.805 m | 0.073 m |
| nuScenes-mini | motorcycle | 146 | 0.641 m | 0.111 m |
| nuScenes-mini | pedestrian | 997 | 0.848 m | 0.069 m |

Đây là các cặp đã khớp tốt XY, không đánh giá box bị bỏ sót/nhầm lớp. Các
prediction mini dùng để kiểm chứng là kết quả mini-eval cũ, checkpoint hiện
chỉ có state_dict, không có metadata huấn luyện chứng nhận nguồn. Bằng
chứng gồm đường code legacy, kiểm thử encode/decode và kết quả độc lập mini.

## Convention đã truy ra

1. Converter nuScenes/VF ghi `b.center` (tâm hình học) vào gt_boxes.
2. Loader legacy tạo LiDARInstance3DBoxes với origin=(0.5,0.5,0), giữ Z đó
   trong tensor mà API coi là tâm đáy.
3. TransFusion lấy tensor GT, coder encode cộng h/2; decode trừ h/2.
   Nếu tái tạo target đúng, decoded Z lại là giá trị tâm hình học ban đầu.
4. Exporter dùng gravity_center, cộng thêm predicted h/2 vào decoded Z.

Do đó không thể sửa bằng một hằng số Z chung hay trừ global h/2 bất kể
rotation. Phép hiệu chỉnh cho kết quả legacy đã export là:

    center_global -= R_global_lidar[:, 2] * predicted_height / 2

Cả vị trí và orientation vẫn được biểu diễn trong global. Giữ nguyên size,
class, score, velocity; không đọc GT để hiệu chỉnh. Áp dụng tất cả class,
không riêng pedestrian. Trục LiDAR tham chiếu có roll/pitch nhỏ nên phép
sửa cũng thay đổi XY một lượng nhỏ.

Không thay đổi coder/training loader trong lần này: sửa những phần đó có
thể thay convention checkpoint. Không áp dụng correction này cho model
đã train/fine-tune theo convention chuẩn khác hoặc kết quả đã sửa rồi.
Công cụ yêu cầu flag convention rõ ràng, chặn ghi đè output và đánh dấu để
chặn áp dụng lần hai. Cần dùng chính dataset calibration đã chạy inference.
Raw export đã lọc range, công cụ không phục hồi box bị lọc trước đó.

## Metric VF sau sửa

| Metric | Raw 5 sweep | Center corrected |
|---|---:|---:|
| custom mAP | 50.7288% | 50.8355% |
| AP car | 61.7407% | 62.1041% |
| AP motorcycle | 81.2494% | 81.1945% |
| AP pedestrian | 9.1964% | 9.2078% |

AP dùng XY, do đó không kỳ vọng tăng mạnh. Thay đổi nhỏ do LiDAR Z nghiêng
so với global Z, làm XY cũng thay đổi. Mục tiêu là đúng tâm box cho
visualization, crop point và DetZero, không làm đẹp AP.

## Pedestrian và calibration

Dựng 28 ví dụ trên 27 frame (có chọn trường hợp, không phải mẫu ngẫu nhiên):
GT, điểm keyframe và prediction được đưa về ego keyframe trước khi chiếu;
camera dùng ego pose tại timestamp riêng, không giả định NAV=identity.
Trong 518 GT, greedy matching cùng lớp trong 2m của bản raw 5 sweep có:
- 104 không ghép được prediction; 6 trong số này có >=100 điểm keyframe.
- 285 ghép được nhưng score <0.1; 99 có >=100 điểm keyframe.
- 129 ghép được với score >=0.1; còn 335 prediction >=0.1 không ghép GT.
Số điểm ở đây chỉ đếm keyframe để kiểm tra geometry, không phải toàn bộ
số điểm 5 sweep vào model. Sửa Z không giải quyết bỏ sót hoặc nhầm Rider.

Đối chiếu 930 camera record: cùng điểm từ nguồn qua homography và qua
sensor2lidar/cam_intrinsic trong PKL mới, max sai khác ~2.52e-9 pixel.
Lấy mẫu điểm thực từ các file LiDAR đã đổi, đưa về ego nguồn: max sai khác
~7.56e-6 m. Kiểm tra này chứng minh pipeline nhất quán về đại số/số học,
không chứng nhận calibration vật lý đúng từng pixel hoặc pose 6-DoF chuẩn.

Camera sau warp vẫn ở vị trí VF, không chuyển optical center sang nuScenes.
CAM_BACK chỉ khoảng 59.1% pixel crop có dữ liệu ảnh nguồn; front ~98.5%,
front-left/right ~89.8/90.1%. Không thể tạo vùng nhìn mới bằng homography.
Giữ cấu hình hiện tại làm baseline; không thay extrinsic hàng loạt dựa vào
AP. Nếu tối ưu tiếp, thử camera-only/lidar-only và crop/FOV có kiểm soát.

## Lệnh trên VM — không cần chạy lại model

```bash
cd ~/bevfusion
git pull --ff-only
source ~/venvs/detzero/bin/activate

python tools/correct_legacy_box_centers.py \
  --data-root data/nuscenes_vf6_01_5hz_nav_calib \
  --input outputs/vf6_nav/sweeps5/results_nusc.json \
  --output outputs/vf6_nav/sweeps5/results_nusc_center_corrected.json \
  --source-convention mit-legacy-center-as-bottom

python tools/audit_vf_box_centers.py \
  --data-root data/nuscenes_vf6_01_5hz_nav_calib \
  --predictions outputs/vf6_nav/sweeps5/results_nusc.json \
  --output outputs/vf6_nav/sweeps5/center_audit.json

python tools/evaluate_vf_comparison.py \
  --data-root data/nuscenes_vf6_01_5hz_nav_calib \
  --baseline outputs/vf6_nav/sweeps5/results_nusc.json \
  --refined outputs/vf6_nav/sweeps5/results_nusc_center_corrected.json \
  --output outputs/vf6_nav/sweeps5/center_comparison.json
```

Dùng results_nusc_center_corrected.json làm input DetZero cho checkpoint
legacy này; không trừ thêm h/2 trong renderer/bridge. File predictions.pkl
và JSON raw giữ nguyên, nên các công cụ đọc trực tiếp PKL vẫn theo convention
raw cũ. Chưa chạy DetZero/GPU trong lần kiểm tra này.
