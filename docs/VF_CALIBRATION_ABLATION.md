# Thử calibration nuScenes trên VF6_01

Baseline VM: custom_mAP 0.411414; AP car 0.608259, motorcycle 0.620525,
pedestrian 0.005458. Chưa có kết quả GPU của cấu hình thử nghiệm này.

## Phát hiện đã kiểm tra trên dữ liệu

Config VF baseline giữ resize 0.48 và crop đáy của nuScenes. Ảnh VF 1920×1536
thành 921×737, rồi lấy vùng (108,481) kích thước 704×256. Tức là lấy phần ảnh
bắt đầu khoảng y=1002 trên ảnh gốc, cắt mất phần lớn vùng có đối tượng.
Đây là lỗi thích nghi preprocessing trong config VF trước đó.

Kiểm tra tâm GT trong phạm vi đánh giá, num_lidar_pts > 0, trên đủ 155 frame:

| Class | GT | Tâm trong ảnh gốc | Tâm trong crop baseline | Tâm trong crop sau đổi camera |
|---|---:|---:|---:|---:|
| car | 4846 | 4846 | 59 | 4606 |
| motorcycle | 2477 | 2477 | 197 | 2295 |
| pedestrian | 518 | 518 | 0 | 406 |

Cột cuối chỉ đếm tia nằm trong chính camera nguồn tương ứng và crop đầu ra.
Đây là độ phủ hình học của **tâm box**, không phải độ nhìn thấy sau che khuất,
không phải recall/AP và không có nghĩa baseline mất toàn bộ pixel của mọi box.
LiDAR vẫn cung cấp dữ liệu; cần inference mới để đo ảnh hưởng đến detection.

VF đang dùng point cloud trong hệ ego: LIDAR_TOP có R=I, t=0. Reference
nuScenes scene-0103 có t=[0.985793,0,1.84019] và rotation khác, gồm yaw khoảng
-90 độ cùng roll/pitch nhỏ. Phân bố tọa độ đầu vào khác checkpoint.
Ngoài ra, median chiều cao GT pedestrian được đánh giá là 1.286 m; cần xem
lại mẫu nhãn, đối tượng bị che khuất và sự khác biệt với dữ liệu huấn luyện,
chưa đủ bằng chứng kết luận nhãn sai. AP pedestrian gần như không tăng khi
nới khoảng cách 0.5 lên 4 m: lỗi vị trí nhỏ khó giải thích toàn bộ vấn đề.

## Phép đổi thực hiện

- Camera: dùng `H = K_ref R_ref^T R_VF K_VF^-1`, thay orientation/intrinsic,
  xuất ảnh 1600×900. **Giữ tâm camera vật lý VF**, không giả vờ chuyển camera
  sang vị trí nuScenes: việc đó cần depth, xử lý che khuất và vùng chưa quan sát.
- LiDAR: `p_ref = R_ref^T (R_VF p_VF + t_VF - t_ref)`, cập nhật calibration
  tương ứng. Giữ nguyên intensity và các cột khác. Đây là đổi hệ tọa độ,
  không tái tạo tia quét/mật độ/che khuất của một LiDAR mới.
- Chọn calibration frame đầu scene-0103 của nuScenes-mini, không dùng GT để
  chọn phép đổi. Đây là một rig tham chiếu, không phải calibration chung cho
  mọi scene nuScenes. Giữ nguyên sample token, timestamp và GT trong hệ ego.
- Converter tạo lại camera2lidar và thông tin exporter. Đầu ra detection được
  đổi ngược về hệ ego gốc nên so sánh trên cùng GT VF được.
- Có ba biến thể camera, lidar, both; baseline không bị sửa. Phải chạy riêng
  để biết nhánh nào giúp/hại. Camera sau warp có vùng đen: phần pixel hợp lệ
  trong crop khoảng 98.5% front, 89.8/90.1% front-left/right, 59.1% back,
  100% back-left/right. Không thể bổ sung vùng nhìn thiếu chỉ từ calibration.
- Thiếu ego motion vẫn giữ nguyên; phép đổi này không mở khóa PRM/global tracking.

## Chạy trên VM

Không cần tải lại dữ liệu. Cần hai thư mục đã có:
`data/nuscenes_vf6_01_5hz/v1.0-trainval` và `data/nuscenes/v1.0-mini`.
Adapter chỉ đọc JSON calibration từ nuScenes-mini, không dùng ảnh/GT mini.
Bản both tạo thêm ảnh và point cloud; không ghi đè dữ liệu gốc.
Nếu thư mục đích đã tồn tại, script dừng; chọn tên mới hoặc kiểm tra bản cũ
trước khi tự dọn. Các biến thể chỉ đổi một sensor dùng symlink tới sensor
không đổi: cần giữ nguyên dataset nguồn.

```bash
cd ~/bevfusion
git pull --ff-only
source ~/venvs/bevfusion/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# Chẩn đoán thêm score và các box gần pedestrian trong baseline VM.
python tools/diagnose_vf_pedestrian.py \
  --predictions outputs/vf6_01_5hz/bevfusion/results_nusc.json

# Tạo bản thử cả camera + hệ tọa độ LiDAR.
python tools/adapt_vf_calibration.py \
  --mode both \
  --output-root data/nuscenes_vf6_01_5hz_nusc_calib

python tools/data_converter/create_vf_infos.py \
  --root-path data/nuscenes_vf6_01_5hz_nusc_calib \
  --version v1.0-trainval

python tools/validate_nuscenes_data.py \
  --data-root data/nuscenes_vf6_01_5hz_nusc_calib \
  --infos data/nuscenes_vf6_01_5hz_nusc_calib/bevfusion_infos_val.pkl \
  --full --output outputs/vf6_01_5hz/calibration_adapted_audit.json

mkdir -p outputs/vf6_01_5hz/calibration_both
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/vf6_calibration/both.yaml \
  pretrained/bevfusion-det.pth \
  --out outputs/vf6_01_5hz/calibration_both/predictions.pkl \
  --format-only \
  --eval-options jsonfile_prefix=outputs/vf6_01_5hz/calibration_both

source ~/venvs/detzero/bin/activate
python tools/evaluate_vf_comparison.py \
  --data-root data/nuscenes_vf6_01_5hz \
  --baseline outputs/vf6_01_5hz/bevfusion/results_nusc.json \
  --refined outputs/vf6_01_5hz/calibration_both/results_nusc.json \
  --output outputs/vf6_01_5hz/calibration_comparison.json
```

Ở lệnh cuối, `--refined` chỉ là tên tham số nhận bộ dự đoán thứ hai; thí nghiệm
này **chưa chạy DetZero**. Đánh giá dùng data-root gốc, cùng bộ GT và bộ lọc.

Hai thí nghiệm tách nhánh dùng cùng chuỗi lệnh, thay các giá trị:

| Biến thể | --mode | Dataset đích | Config | Thư mục output |
|---|---|---|---|---|
| Camera | camera | data/nuscenes_vf6_01_5hz_camera | vf6_calibration/camera.yaml | outputs/vf6_01_5hz/calibration_camera |
| LiDAR | lidar | data/nuscenes_vf6_01_5hz_lidar | vf6_calibration/lidar.yaml | outputs/vf6_01_5hz/calibration_lidar |

Đường dẫn config đầy đủ có cùng tiền tố với both.yaml ở trên.
Chạy lại converter cho từng dataset; không dùng PKL baseline cho dataset đổi calibration.
Đổi `--refined` và `--output` đánh giá theo từng biến thể, tránh ghi đè so sánh.

## Đọc kết quả và quyết định tiếp

Ưu tiên AP pedestrian, đồng thời kiểm tra car/motorcycle và custom_mAP. Xem
`calibration_diagnostic.json` để biết lượng box/score từng lớp, cùng thống kê
box gần GT pedestrian theo các ngưỡng score. Đó là matching không phân lớp
trong bán kính 2 m, chỉ phục vụ điều tra; không phải confusion matrix có xác
nhận danh tính hay quy trình tính AP.

Nếu sửa đầu vào vẫn có AP thấp, xem các frame chứa pedestrian để phân biệt
bỏ sót, nhầm lớp, nhãn/độ che khuất và score thấp. Chưa có predictions VM ở
local nên hiện chưa kết luận pedestrian chủ yếu bị nhầm thành lớp nào.
Không đổi nhãn hoặc threshold đánh giá để làm đẹp baseline. Chưa ghép DetZero
vào so sánh này: cần ổn định detector trước khi đo hiệu quả tracking/refining.
