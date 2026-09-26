# Đánh giá nối BEVFusion → DetZero Tracking → Refining

Đã đọc source local tại `DetZero-main/`. Đây là thiết kế tích hợp dựa trên
code hiện tại, chưa phải pipeline đã được triển khai hoặc kiểm thử GPU.
Người dùng sẽ bổ sung checkpoint GRM/PRM/CRM sau; chưa kiểm tra được trọng số.

## Dữ liệu hiện có

- BEVFusion đã dự đoán 81 sample mini validation và xuất
  `outputs/mini-eval/results_nusc.json` trên VM.
- Hai scene tương ứng: `scene-0103` (40 frame), `scene-0916` (41 frame).
- Đo trực tiếp metadata local: delta-time của scene-0103 nằm trong
  0.399259–0.599961 giây; scene-0916 nằm trong 0.400353–0.501001 giây.
- DetZero local là một Git repo riêng (`TestModel_DetZero`), có thay đổi sẵn
  trong `.gitignore`, `daemon/prepare_object_data.py` và file chưa tracked.
  Không được ghi đè các thay đổi này hoặc thêm cả repo con vào Git BEVFusion.

## 1. Adapter detection

Nên dùng JSON nuScenes đã export làm ranh giới giữa hai môi trường Python.
Nó đã chứa tâm box, quaternion, kích thước và vận tốc trong hệ global, có
sample token. Không cần unpickle class `LiDARInstance3DBoxes` của BEVFusion
trong môi trường DetZero. JSON này đã qua lọc phạm vi lớp của nuScenes;
không hoàn toàn tương đương tập proposal thô trong `mini-predictions.pkl`.

Thêm adapter, ví dụ `tools/export_nuscenes_to_detzero.py`, để:

1. Nối sample token với scene, timestamp, LiDAR file và calibration.
2. Đánh số frame liên tục riêng từng scene, giữ mọi frame kể cả không có box.
3. Chuyển size nuScenes `[width, length, height]` thành DetZero
   `[length, width, height]`; dùng tâm hình học và yaw phù hợp.
4. Dùng `T_global_from_lidar = T_global_from_ego @ T_ego_from_lidar`.
5. Tạo dữ liệu mỗi frame gồm `sequence_name`, `frame_id`, `timestamp`,
   `pose`, `boxes_lidar` (N×7), `name`, `score`; hoặc chuẩn hóa trực tiếp
   `boxes_global` và bỏ bước transform lại trong processor.
6. Lưu manifest riêng ánh xạ `(scene, frame_id)` ↔ sample token, LiDAR file,
   lớp nuScenes gốc. Không đưa tùy tiện chuỗi metadata vào các field mà
   `DataProcessor` áp dụng boolean mask theo box.

Không copy nguyên `.tensor[:, :7]`: BEVFusion có chuyển tâm bằng
`gravity_center`, chuyển yaw `-yaw - pi/2` trong
`mmdet3d/datasets/nuscenes_dataset.py::output_to_nusc_box`.

Kiểm thử bắt buộc: vòng chuyển tọa độ đi/về, kích thước, yaw và chiếu box lên
ảnh/point cloud; kiểm tra đủ 81 token, 2 scene, không ghép chéo scene.

## 2. Tracking và thời gian

Các điểm phải chỉnh ở DetZero:

- `tracking/tools/cfgs/tk_model_cfgs/waymo_detzero_track.yaml`:
  `FILTER.DELTA_T: 0.1` là mặc định Waymo. Tạo config nuScenes riêng.
- `tracking/detzero_track/models/tracking_modules/track_manager.py`:
  reverse tracking có `delta_t=-0.1`; cần delta-time âm theo timestamp thật.
- `tracking/detzero_track/models/tracking_modules/kalman_filter/kalman_filter.py`:
  cập nhật ma trận chuyển trạng thái theo delta-time mỗi lần predict.
- `tracking/detzero_track/models/tracking_modules/post_process.py`:
  `velocity_optimize` đang nhân độ dịch chuyển với `10`; thay bằng chia
  cho chênh lệch thời gian. Truyền timestamp xuyên suốt cả forward/reverse.
- Xem lại `LEAST_AGE: 5`, ngưỡng association, track lifetime theo mật độ frame;
  5 quan sát ở 2 Hz có ý nghĩa khác 5 quan sát ở 10 Hz.
- Xử lý frame không có detection/track: reverse tracking hiện truy cập
  `frm_tracks[frm_id]` trực tiếp, có thể thiếu key.

`DatasetTemplate.assign_mode` hiện bật GT cho mọi split khác `test`. Vì vậy
chạy `--split val` sẽ tìm `waymo_infos_val.pkl`. Nên tách `assign_gt` khỏi tên
split và thêm dataset nuScenes. Nếu thử bằng chế độ `test` để bỏ GT, phải ghi
rõ dữ liệu thực tế vẫn là mini validation, không phải nuScenes test split.

Tracker Kalman/association không cần checkpoint học sâu.

## 3. Class mapping

DetZero dùng Vehicle/Pedestrian/Cyclist, BEVFusion detection dùng 10 lớp.
Khuyến nghị kiểm thử ban đầu chỉ `car → Vehicle`, giữ nhãn gốc trong manifest;
sau đó thêm `pedestrian → Pedestrian` và kiểm tra riêng mapping cyclist.
Không gộp bicycle/motorcycle hoặc mọi xe lớn vào một lớp rồi mặc định đánh giá
như 10 lớp nuScenes. Nếu muốn track toàn bộ lớp động theo nhãn nuScenes, dùng
config association riêng và tách chính sách mapping cho refiner.

Các lớp không refine phải được giữ nguyên nếu xuất kết quả detection đầy đủ.
Khi chỉ đánh giá subset lớp, phải công bố rõ subset.

## 4. Object data cho refining

`daemon/prepare_object_data.py` hiện:

- Đọc `waymo_processed_data/segment-*/NNNN.npy`.
- Dùng `pts[:, 5] == -1` làm mask NLZ và `tanh(intensity)`.
- Crop points theo box trong hệ global, lưu `refining/<Class>/<scene>.pkl`.

nuScenes LiDAR `.pcd.bin` có 5 float mỗi điểm và không có cột NLZ Waymo.
Thêm reader/cropper nuScenes riêng dùng manifest, lấy xyz và intensity;
không diễn giải ring index hay time-lag của BEVFusion thành NLZ.
Chọn normalization intensity dựa trên preprocessing của checkpoint refining,
không áp dụng tanh của Waymo một cách mặc định.

Mỗi track cần: `sequence_name`, `obj_id`, `name`, `sample_idx`, `boxes_global`,
`score`, `pose`, `hit`, `state`, `pts` và timestamp. Points phải cùng hệ tọa độ
với box khi crop, sau đó để dataset refiner biến đổi sang hệ tọa độ object.
Kiểm tra số điểm mỗi box, tỷ lệ crop rỗng, track có toàn bộ frame rỗng.

## 5. Dataset/config refining và checkpoint

`refining/detzero_refine/datasets/dataset.py` đang dùng `ImageSets/<split>.txt`,
đường dẫn Waymo và xử lý tên sequence bằng `.strip(...)`. Đó là xóa tập ký tự,
không phải xóa prefix/suffix chính xác; không dùng nguyên trạng cho scene/token
nuScenes. Thêm dataset/config nuScenes hoặc lớp đọc manifest chung.

Inference không cần GT, nhưng code hiện vẫn mang `matched`, `matched_tracklet`,
`gt_boxes_global` qua nhiều bước. Cần chế độ inference không GT rõ ràng:
không lọc mất unmatched tracks, không tính recall từ GT placeholder;
`--save_to_file` và `GENERATE_RECALL=False` là các cấu hình liên quan.
Nếu train/fine-tune thì phải tạo matching với GT nuScenes thật và tách train/val.

GRM dự đoán hình học/kích thước; PRM dự đoán vị trí/hướng; CRM hiệu chỉnh
confidence. GRM và PRM có thể chạy độc lập từ cùng object data rồi ghép lại.

Khi nhận checkpoint phải xác minh từng model, class, feature encoding và shape.
Loader `utils/detzero_utils/model_utils.py::load_params_from_file` chỉ nạp các
key/shape khớp và có thể để phần còn lại ngẫu nhiên. Đối với inference phải
kiểm tra đủ trọng số cần thiết, không chỉ thấy tiến trình không báo lỗi.
Checkpoint train vài bước trên Waymo chỉ chứng minh chạy được, không chứng minh
refine tốt cho nuScenes. Giảm batch cho T4, không đổi kiến trúc tùy ý để né OOM.

## 6. Export và đánh giá

`daemon/combine_output.py` ghép size từ GRM, vị trí/hướng từ PRM, score từ CRM
(tùy chọn), nhưng đường dẫn/lớp/output còn theo Waymo. Thêm exporter nuScenes:

- Ghép bằng `(scene, track ID, frame ID)`, kiểm tra thiếu/thừa/đổi thứ tự frame.
- Trả box về hệ global nuScenes và ánh xạ sample token, nhãn gốc.
- Track ID duy nhất xuyên scene, ví dụ `<scene_token>:<local_track_id>`.
- Xuất detection JSON riêng cho mAP/NDS và tracking JSON đúng schema riêng
  cho đánh giá tracking; giữ entry rỗng cho frame không có box.
- So sánh BEVFusion gốc, sau tracking và sau refining trên cùng split/phạm vi
  lớp. Không suy ra chất lượng tracking chỉ từ mAP/NDS hoặc video nhìn mượt.

## Thứ tự triển khai

1. Adapter và kiểm thử hình học/token, bắt đầu car.
2. Tracking timestamp thực và video có ID trên 2 scene mini.
3. Crop object points và kiểm tra trực quan.
4. Nạp checkpoint đủ key, chạy GRM/PRM; ghép và kiểm tra trước/sau.
5. Thêm CRM, exporter và đánh giá trên cùng dữ liệu.

---

## 7. Trạng thái triển khai hoàn tất (DetZero-main-2)

Pipeline đã được hiện thực hóa và kiểm thử thành công end-to-end:

1. **Adapter & Geometry Bridge (`DetZero-main-2/integration/bridge.py`):**
   - Đọc kết quả detection từ `results_nusc.json` của BEVFusion mà không cần phụ thuộc môi trường PyTorch BEVFusion.
   - Đồng bộ timestamp thực tế và ma trận biến đổi tọa độ ego/sensor sang hệ toạ độ global.
   - Ánh xạ các lớp nuScenes sang 3 lớp mục tiêu của DetZero (`Vehicle`, `Pedestrian`, `Cyclist`).
   - Xử lý đọc LiDAR point cloud `.pcd.bin` (5 kênh) và crop điểm chính xác quanh từng box 3D.
   - Mã hóa đặc trưng cho GRM (`geo_query_points`, `geo_memory_points`, `geo_query_boxes`) và PRM (`pos_query_points`, `pos_memory_points`, `pos_trajectory`, `padding_mask`).
   - Ghép box tinh chỉnh (size từ GRM + vị trí & hướng từ PRM) và xuất lại đúng schema NuScenes.

2. **DetZero Tracking thích ứng nuScenes (`DetZero-main-2/tracking/`):**
   - File cấu hình riêng: `nuscenes_detzero_track.yaml` với tần số 2 Hz (`DELTA_T: 0.5`, `LEAST_AGE: 2`).
   - Cập nhật delta-time động theo timestamp thật trong Kalman Filter cho cả forward pass và reverse tracking.
   - Tối ưu hóa vận tốc theo độ biến thiên thời gian thực tế $\Delta t$ thay vì nhân cố định 10.
   - Hỗ trợ chạy trên cả GPU (CUDA) và CPU (dùng Shapely exact polygon intersection fallback).

3. **Refining GRM + PRM độc lập (`DetZero-main-2/refining/`):**
   - Tách rời hoàn toàn khỏi cấu trúc dataset Waymo; nạp cấu hình model trực tiếp từ YAML.
   - Tự động nạp đủ 6 checkpoint:
     - `vehicle_grm_model.pth` + `vehicle_prm_model.pth`
     - `pedestrian_grm_model.pth` + `pedestrian_prm_model.pth`
     - `cyclist_grm_model.pth` + `cyclist_prm_model.pth`
   - Bỏ CRM theo yêu cầu (giữ nguyên độ tin cậy detector confidence).

4. **Script chạy toàn bộ quy trình (`DetZero-main-2/integration/run_pipeline.py`):**
   - Thực thi trọn vẹn 5 bước: `Prepare -> Track -> Crop LiDAR -> Refine (GRM+PRM) -> Export`.
   - Xuất 2 file kết quả:
     - `results_nusc_detzero_refined.json`: Kết quả detection sau refine để đánh giá mAP / NDS.
     - `results_nusc_detzero_tracking.json`: Kết quả tracking có `tracking_id` theo chuẩn NuScenes tracking.

---

## 8. Hướng dẫn chạy trên VM GPU (`detzero-gpu-01`)

### Bước 1: Đẩy mã nguồn từ máy Local lên GitHub bằng Git

Toàn bộ 6 checkpoint (.pth ~14-16MB/file, tổng ~89MB) và các module đã được cấu hình hợp lệ để commit qua Git (không vượt ngưỡng 100MB của GitHub):

```bash
git add DetZero-main-2 docs/BEVFUSION_DETZERO_INTEGRATION.md
git commit -m "feat: Integrate DetZero tracking and GRM/PRM refining with BEVFusion"
git push origin fix/nuscenes-mini-depth-lss
```

### Bước 2: Kéo mã nguồn về VM và kích hoạt môi trường

Trên máy ảo GPU (`hngtram11@detzero-gpu-01`):

```bash
cd ~/bevfusion
git pull origin fix/nuscenes-mini-depth-lss
conda activate detzero # hoặc môi trường DetZero của bạn
```

### Bước 3: Chạy pipeline nối BEVFusion sang DetZero Tracking & Refining

```bash
python DetZero-main-2/integration/run_pipeline.py \
    --results_path outputs/mini-eval/results_nusc.json \
    --data_root data/nuscenes \
    --version v1.0-mini \
    --tracking_cfg DetZero-main-2/tracking/tools/cfgs/tk_model_cfgs/nuscenes_detzero_track.yaml \
    --checkpoint_dir DetZero-main-2/checkpoints \
    --output_dir outputs/detzero_refined \
    --device cuda
```

*Ghi chú: Nếu chỉ muốn kiểm tra tracking trước khi chạy refining, thêm cờ `--skip_refining`.*

### Bước 4: Chấm điểm đánh giá kết quả sau tinh chỉnh (Evaluation)

**Đánh giá Detection (mAP, NDS sau refine):**
```bash
python -m nuscenes.eval.detection.evaluate \
    --result_path outputs/detzero_refined/results_nusc_detzero_refined.json \
    --output_dir outputs/detzero_refined/eval_detection \
    --eval_set val \
    --dataroot data/nuscenes \
    --version v1.0-mini
```

**Đánh giá Tracking (AMOTA, AMOTP, MOTA, MOTP):**
```bash
python -m nuscenes.eval.tracking.evaluate \
    outputs/detzero_refined/results_nusc_detzero_tracking.json \
    --output_dir outputs/detzero_refined/eval_tracking \
    --eval_set val \
    --dataroot data/nuscenes \
    --version v1.0-mini
```

