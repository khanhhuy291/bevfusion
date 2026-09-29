# VF6 NAV + sweep: chạy thí nghiệm có kiểm soát

Nguồn mới: `data/nuscenes_vf6_01_5hz_nav_sweeps`.
Bản đổi camera/LiDAR: `data/nuscenes_vf6_01_5hz_nav_calib`.
Giữ nguyên nguồn. Bản derived dùng cùng phương pháp calibration_both đã thử:
warp camera tại cùng optical center, đổi hệ LiDAR, giữ ego pose và GT global.

## Kiểm tra đã thực hiện local

- 155 keyframe, 930 ảnh, 306 LiDAR records (155 keyframe +151 non-keyframe).
- LiDAR history khoảng 10 Hz (median dt 0.1 s); nhãn/keyframe 5 Hz.
- Chỉ dùng sweep quá khứ, tối đa 9, age <=1 s; đúng 147/155 frame đủ 9.
  Keyframe trước cũng có thể là sweep của frame hiện tại.
- GT global NAV đổi ngược về ego khớp toàn bộ 15.389 box của bộ cũ:
  max sai số tâm ~5.7e-14 m, rotation matrix ~7.8e-16.
- Converter tạo `sensor2lidar_rotation/translation` bằng pose riêng của từng
  sensor; tests đối chiếu độc lập với chuỗi ma trận homogeneous 4x4.
- Kiểm tra đầy đủ sensor và đường dẫn keyframe/sweep trong PKL trên local.
- Chưa chạy inference GPU cho bản NAV; chưa biết AP sẽ tăng/giảm.

## Những giới hạn vẫn còn

Pose NAV XYZ+yaw, roll/pitch=0, không per-point deskew; chưa phải INS 6-DoF
được xác nhận. Heading/velocity agreement và report alignment nguồn là
bằng chứng hỗ trợ, chưa xác nhận mọi frame đúng. Trong source report,
median khoảng cách nearest-neighbor background sau bù ở 6 cặp vẫn khoảng
0.26–0.44 m. Không diễn giải đây là sai số pose chính xác vì khác mật độ,
che khuất và phương pháp loại đối tượng động chỉ là gần đúng.

9 sweep khoảng 0.9 s (một số trường hợp sát 1 s), không phải 1.8 s. Pose bù
chuyển động xe, không bù người/xe máy động. Cloud đã ghép nhiều LiDAR có
khoảng 270 nghìn điểm/frame: 9 sweep có thể gần 2.7 triệu điểm trước lọc,
tốn bộ nhớ và thời gian hơn baseline. Nên chạy 0 và 2 trước, sau đó 5 và 9.

`ego_motion_available=true` là có nguồn pose đo được, không phải chứng nhận
chất lượng. `velocity_available=false` và gt_velocity NaN được giữ lại:
chưa xác nhận vận tốc đối tượng/track identity nên không tự sinh GT velocity.
Không suy ra PRM đã đáng tin cậy chỉ vì pose khác zero.

## Chạy trên VM

Bạn tự đưa dataset mới lên VM tại đúng đường dẫn nguồn, không cần upload
bản derived đã tạo trên Mac. PKL phải tạo lại trên VM để đúng đường dẫn.
Các lệnh dưới dùng env bevfusion đã có numpy, OpenCV và nuscenes-devkit.
Nếu output derived đã tồn tại, adapter dừng để tránh ghi đè; không chạy lại
adapter khi bộ đó đã được tạo thành công, chỉ chạy converter khi cần.

```bash
cd ~/bevfusion
git pull --ff-only
source ~/venvs/bevfusion/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

python tools/adapt_vf_calibration.py \
  --source-root data/nuscenes_vf6_01_5hz_nav_sweeps \
  --output-root data/nuscenes_vf6_01_5hz_nav_calib \
  --mode both

python tools/data_converter/create_vf_infos.py \
  --root-path data/nuscenes_vf6_01_5hz_nav_calib \
  --max-sweeps 9 --max-sweep-age 1.0

python tools/validate_nuscenes_data.py \
  --data-root data/nuscenes_vf6_01_5hz_nav_calib \
  --infos data/nuscenes_vf6_01_5hz_nav_calib/bevfusion_infos_val.pkl \
  --full --output outputs/vf6_nav/data_audit.json
```

Chỉ tạo một PKL chứa tối đa 9 sweep; từng config quyết định dùng 0/2/5/9.
Loader test_mode=true lấy các sweep gần nhất, không lấy ngẫu nhiên và không
nhân bản frame hiện tại khi thiếu lịch sử. Cột thứ 5 là age theo giây.

Chạy hai cấu hình đầu (subshell dừng nếu một lệnh lỗi):

```bash
(
set -e
cd ~/bevfusion
source ~/venvs/bevfusion/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
for n in 0 2; do
  mkdir -p "outputs/vf6_nav/sweeps${n}"
  torchpack dist-run -np 1 python tools/test.py \
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/vf6_nav/sweeps${n}.yaml" \
    pretrained/bevfusion-det.pth \
    --out "outputs/vf6_nav/sweeps${n}/predictions.pkl" \
    --format-only \
    --eval-options "jsonfile_prefix=outputs/vf6_nav/sweeps${n}"
done

source ~/venvs/detzero/bin/activate
python tools/evaluate_vf_comparison.py \
  --data-root data/nuscenes_vf6_01_5hz_nav_calib \
  --baseline outputs/vf6_nav/sweeps0/results_nusc.json \
  --refined outputs/vf6_nav/sweeps2/results_nusc.json \
  --output outputs/vf6_nav/compare_0_2.json
)
```

Sau đó đổi vòng lặp thành `for n in 5 9; do` để thử thêm, và đánh giá từng
kết quả với baseline `sweeps0` bằng cùng script. Giữ các file comparison
riêng, ví dụ `compare_0_5.json`, `compare_0_9.json`.

**Không đưa prediction cũ dùng identity ego vào evaluator với GT NAV.**
Ngay cả khi token trùng, tọa độ global cũ và mới khác nhau. Baseline mới phải
là `vf6_nav/sweeps0`, giữ cùng pose/calibration/camera, chỉ khác số sweep.
Không dùng script `diagnose_vf_pedestrian.py` cũ cho NAV: script đó chủ động
chỉ hỗ trợ identity ego; dùng evaluator ở trên cho so sánh mới.

## Quyết định sau khi chạy

So AP cả 3 lớp, đặc biệt pedestrian, và thời gian/VRAM. Chọn số sweep trên
validation, không mặc định 9 tốt nhất. Nếu tăng sweep giảm AP, kiểm tra vùng
đối tượng động bị kéo dài, sai số roll/pitch, đồng bộ NAV và saturation của
voxelizer. Không thay ngưỡng score hay nhãn giữa các thí nghiệm. Quy ước Z
của prediction và Rider/Pedestrian vẫn cần xử lý độc lập trước DetZero.

Kết quả kiểm tra convention Z của checkpoint legacy và lệnh xuất JSON cho DetZero: [VF_SWEEPS5_CENTER_AUDIT.md](VF_SWEEPS5_CENTER_AUDIT.md).
