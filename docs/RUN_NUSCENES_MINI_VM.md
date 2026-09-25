# Chạy BEVFusion detection trên VM GCP

Máy: `detzero-gpu-01`, project `detzero-509317`, zone `asia-southeast1-a`.
Repo: `~/bevfusion`, nhánh `fix/nuscenes-mini-depth-lss`.

## Trạng thái bàn giao

- Code sửa converter, DepthLSS và import Numba đã được đưa lên Git.
- Môi trường riêng `~/venvs/bevfusion` đã cài Python 3.8.20,
  PyTorch 1.10.1+cu113, MMCV-full 1.4.0, MMDetection 2.20.0.
- VM có Tesla T4 và CUDA toolkit 11.7 tại `/usr/local/cuda`.
- Data, checkpoint, archive và upload dở do trợ lý tạo đã được xóa theo yêu cầu.
- Chưa build CUDA extensions, tạo metadata hoặc xác nhận inference thành công.
- Không thay đổi môi trường DetZero. Các bước dưới đây do người dùng chạy.

## 1. Kết nối và cập nhật code

Chạy trên Mac:

```bash
gcloud compute ssh hngtram11@detzero-gpu-01 \
  --project=detzero-509317 --zone=asia-southeast1-a
```

Các lệnh tiếp theo chạy trên VM:

```bash
cd ~/bevfusion
git switch fix/nuscenes-mini-depth-lss
git pull --ff-only
source ~/venvs/bevfusion/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS=4
nvidia-smi
nvcc --version
python -c 'import torch, mmcv; print(torch.__version__, mmcv.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

## 2. Build extensions

```bash
mkdir -p outputs
set -o pipefail
python setup.py develop 2>&1 | tee outputs/build.log
python -c 'from mmdet3d.models import build_model; from mmdet3d.ops import bev_pool; print("Imports OK")'
```

Chỉ tiếp tục khi build và import đều thành công. CUDA toolkit 11.7 trên VM
khác minor version với CUDA 11.3 của wheel PyTorch; cần kiểm tra kết quả build
thực tế. Chưa có xác nhận build thành công trong phiên bàn giao này.

## 3. Tải mini trực tiếp trên VM

Nguồn: https://www.nuscenes.org/tutorials/nuscenes_tutorial.html

```bash
mkdir -p data/nuscenes downloads pretrained
curl -L --fail --retry 3 -C - \
  https://www.nuscenes.org/data/v1.0-mini.tgz \
  -o downloads/v1.0-mini.tgz
tar -xzf downloads/v1.0-mini.tgz -C data/nuscenes
```

Kết quả: `data/nuscenes/{samples,sweeps,maps,v1.0-mini}`.
Giữ cả camera, LiDAR và radar vì pipeline detection hiện tại vẫn đọc radar.
Chưa cần map expansion cho config detection này.

Nếu muốn upload từ Mac thay vì tải trực tiếp, bỏ qua lệnh tải/giải nén trên
và chạy trên Mac (thư mục `~/bevfusion/data` trên VM phải tồn tại):

```bash
cd /Users/khanhhuy/bevfusion
gcloud compute scp --recurse data/nuscenes \
  hngtram11@detzero-gpu-01:~/bevfusion/data/ \
  --project=detzero-509317 --zone=asia-southeast1-a
```

## 4. Tải checkpoint trên VM

```bash
curl -L --fail --retry 3 \
  'https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1' \
  -o pretrained/bevfusion-det.pth
printf '%s\n' 'ee0a389213922343508db40dff0ec04767b36d8f512da34b093939c37aaf2122  pretrained/bevfusion-det.pth' | sha256sum -c -
```

Hash này đã được đối chiếu với checkpoint người dùng cung cấp trên Mac.

## 5. Tạo metadata cho inference

Gọi converter trực tiếp để chỉ tạo info, không tạo ground-truth database dùng
cho augmentation khi train. Chạy từ `~/bevfusion` để đường dẫn tương đối ổn định.

```bash
python - <<'PY'
from tools.data_converter.nuscenes_converter import create_nuscenes_infos
create_nuscenes_infos(
    root_path='data/nuscenes',
    info_prefix='nuscenes',
    version='v1.0-mini',
    max_sweeps=10,
)
PY

python - <<'PY'
from pathlib import Path
import json
import mmcv

root = Path('data/nuscenes')
records = json.loads((root / 'v1.0-mini/sample_data.json').read_text())
missing = [r['filename'] for r in records if not (root / r['filename']).is_file()]
assert not missing, missing[:10]
for split, expected in [('train', 323), ('val', 81)]:
    data = mmcv.load(str(root / f'nuscenes_infos_{split}.pkl'))
    assert data['metadata']['version'] == 'v1.0-mini'
    assert len(data['infos']) == expected, (split, len(data['infos']))
    assert 'radars' in data['infos'][0]
    print(split, len(data['infos']))
print('Sensor files:', len(records), '— no missing files')
PY
```

Đầu ra: `nuscenes_infos_train.pkl` và `nuscenes_infos_val.pkl`.
Không dùng info từ MMDetection3D bản mới thay thế.

## 6. Inference và đánh giá mini validation, một GPU

Môi trường hiện tại dùng `mpi4py-mpich`. Chạy một tiến trình trực tiếp với
`MASTER_HOST` để Torchpack khởi tạo distributed world-size 1; không cần dùng
launcher `torchpack dist-run` với các cờ dành riêng cho OpenMPI.

```bash
export MASTER_HOST=127.0.0.1:29500
export OMP_NUM_THREADS=4
python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth \
  --eval bbox \
  --out outputs/mini-predictions.pkl \
  --eval-options jsonfile_prefix=outputs/mini-eval \
  --cfg-options data.workers_per_gpu=2 model.encoders.camera.backbone.init_cfg=None \
  2>&1 | tee outputs/mini-eval.log
```

`init_cfg=None` tránh tải thêm checkpoint khởi tạo Swin khi đã có checkpoint
fusion đầy đủ. Kết quả mini không so trực tiếp với metric full validation trong README.

## 7. Xuất ảnh dự đoán

```bash
python tools/visualize.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --mode pred \
  --checkpoint pretrained/bevfusion-det.pth \
  --split val \
  --bbox-score 0.3 \
  --out-dir outputs/mini-viz \
  --model.encoders.camera.backbone.init_cfg null \
  --data.workers_per_gpu 2 \
  2>&1 | tee outputs/mini-viz.log
```

Ảnh nằm trong `outputs/mini-viz/camera-*` và `outputs/mini-viz/lidar`.
Nếu có lỗi, giữ lại log của bước lỗi; không coi cảnh báo thiếu/sai shape hàng
loạt khi nạp checkpoint là một lần chạy hợp lệ.
