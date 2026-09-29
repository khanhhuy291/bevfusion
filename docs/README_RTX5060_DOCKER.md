# Khởi động môi trường RTX 5060 và chạy thử nuScenes-mini

## Trạng thái xác nhận ngày 2026-09-30

Máy công ty: Ubuntu 22.04.5, RTX 5060 Laptop GPU 8 GB, driver 595.91.07.

| Thành phần | Trạng thái |
| --- | --- |
| Docker + NVIDIA Container Toolkit | Container đã nhận GPU |
| Image `bevfusion-blackwell:base` | Đã build trên máy công ty |
| Python 3.10, PyTorch 2.7.1+cu128 | Đã chạy phép nhân ma trận trên GPU |
| CUDA runtime 12.8, GPU capability 12.0 | Đã xác nhận trong container |
| DetZero integration | Người dùng xác nhận test đạt sau khi sửa dtype của frame rỗng |
| Sáu checkpoint GRM/PRM | Test nạp trọng số và forward trên CPU đạt |
| Image `bevfusion-blackwell:deps` | Đã build thành công; MMCV-full 1.7.2, MMDetection 2.28.2, import dependency và pip check đạt |
| CUDA operator MMCV | Chưa xác nhận chạy NMS trên GPU |
| GRM/PRM trên GPU | Chưa xác nhận forward model trên GPU |
| BEVFusion detection trên RTX 5060 | **Chưa sẵn sàng: chưa port/build CUDA extensions riêng của repo và kiểm tra model** |

`GPU COMPUTE OK` xác nhận PyTorch chạy GPU, không xác nhận mọi CUDA
operator của BEVFusion đã hoạt động. Một test bị skip vì chưa có torchpack
không có nghĩa môi trường BEVFusion đã hoàn tất.

## 1. Mở lại môi trường mỗi ngày

Các lệnh trong mục này chạy ở Terminal Ubuntu, bên ngoài container.
Không cần kích hoạt Conda `(base)` để dùng Docker.

Xem container đang chạy hay đã dừng:

```bash
sudo docker ps -a --filter name=bevfusion-work
```

Nếu trạng thái là `Exited`, mở lại container đã có:

```bash
sudo docker start -ai bevfusion-work
```

Nếu trạng thái là `Up`, mở thêm một shell vào container:

```bash
sudo docker exec -it bevfusion-work /bin/bash
```

Trong container, thư mục làm việc là `/workspace/bevfusion`:

```bash
cd /workspace/bevfusion
which python
python --version
```

Python phải nằm ở `/opt/venv/bin/python`. Không cần `conda activate` hay
`source activate`. Dòng `I have no name!` và cảnh báo tên group là do image
không khai báo tên tương ứng UID/GID; không phải lỗi khởi động Python.

Gõ `exit` để thoát shell. Nếu đây là shell chính từ `start -ai`, container
sẽ dừng; lần sau dùng lại `start -ai`. Không chạy lại `docker run --name
bevfusion-work` khi container cùng tên đã tồn tại.

## 2. Tạo container nếu chưa có

Image đã build từ Dockerfile ở `~/bevfusion-env/Dockerfile` trên máy công ty.
Kiểm tra image:

```bash
sudo docker image inspect bevfusion-blackwell:base --format '{{.Id}}'
id -u
id -g
```

Lệnh dưới dành cho UID/GID `1000:1000` và project ở
`/home/khanhhuy/bevfusion`. Nếu ID hoặc đường dẫn khác, thay đúng giá trị.
Chỉ chạy khi chưa có container `bevfusion-work`:

```bash
sudo docker run --gpus all -it --name bevfusion-work --shm-size=8g --user 1000:1000 --env HOME=/tmp --env PYTHONPATH=/workspace/bevfusion --mount type=bind,source=/home/khanhhuy/bevfusion,target=/workspace/bevfusion bevfusion-blackwell:base
```

`--shm-size=8g` là bộ nhớ chia sẻ của container, không tăng VRAM của GPU.

Hai đường dẫn sau cùng trỏ đến một thư mục qua bind mount:

| Ubuntu host | Trong container |
| --- | --- |
| `/home/khanhhuy/bevfusion` | `/workspace/bevfusion` |
| `/home/khanhhuy/bevfusion/data` | `/workspace/bevfusion/data` |
| `/home/khanhhuy/bevfusion/outputs` | `/workspace/bevfusion/outputs` |

Code, dữ liệu và kết quả trong project được giữ trên host khi container dừng.
Thay đổi ngoài bind mount chỉ nằm trong container nếu chưa đưa vào image.
Venv `/opt/venv` thuộc root; UID 1000 không cài thêm package vào venv này
được. Các dependency bổ sung nên được ghi vào Dockerfile rồi build image;
không dùng `pip install --user` để tạo một môi trường chồng lên venv.

## 3. Kiểm tra GPU và DetZero

Chạy trong container:

```bash
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device='cuda')
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
print('GPU COMPUTE OK')
PY

python DetZero-main-2/integration/run_pipeline.py --help
python tests/test_bev_detzero_integration.py -v
```

Test frame rỗng phải tạo `name` và `nusc_name` với `dtype=str`, giống
`bridge.prepare()`. Với NumPy 1.23.5, `np.array([])` mặc định là số và so
sánh với tên lớp có thể trả scalar thay vì mask, làm sai shape box.
Không bỏ qua test lỗi này. Test sáu checkpoint chạy trên CPU và sẽ skip
nếu checkpoint đầu tiên không có; kiểm tra log để biết chúng thực sự chạy.

## 4. Chuẩn bị detection trên nuScenes gốc

Trong hướng dẫn này, “nuScenes gốc” là **nuScenes-mini chính thức**, ở
`data/nuscenes`, không phải dữ liệu VF hoặc bản VF đã đổi calibration.
Chạy toàn bộ mini validation để có kết quả liên tục theo scene cho DetZero.

Kiểm tra dữ liệu và checkpoint trước, không cần MMCV:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('data/nuscenes')
for name in ['samples', 'sweeps', 'maps', 'v1.0-mini']:
    assert (root / name).is_dir(), f'Missing: {root / name}'
records = json.loads((root / 'v1.0-mini/sample_data.json').read_text())
missing = [r['filename'] for r in records if not (root / r['filename']).is_file()]
assert not missing, f'Missing {len(missing)} sensor files; examples: {missing[:5]}'
checkpoint = Path('pretrained/bevfusion-det.pth')
assert checkpoint.is_file() and checkpoint.stat().st_size > 0, checkpoint
print('Sensor records:', len(records))
print('DATA + CHECKPOINT FILES OK')
PY
```

Config detection gốc của repo còn đọc radar trong pipeline, dù model fusion
chỉ dùng camera/LiDAR. Giữ đủ file sensor; không thay bằng dataset VF hoặc
xoá radar để giảm dung lượng.

### Điều kiện còn thiếu trước khi chạy detector

Image `bevfusion-blackwell:base` chưa cài MMCV, MMDetection, torchpack,
mpi4py, Numba và chưa build các operator của repo. Dockerfile cũ dùng
PyTorch 1.10/CUDA 11.3, không phải hướng cài đặt cho RTX 5060.

Cần hoàn tất một bước port riêng: xác định bản MMCV/MMDetection tương thích,
sửa API C++/PyTorch cũ nếu cần, build kernel cho `sm_120`, và kiểm tra thực
thi sparse convolution, voxelization, BEV pooling, NMS trên GPU. Chỉ đặt
`TORCH_CUDA_ARCH_LIST=12.0` không đủ vì `setup.py` hiện ghi cứng `-gencode`.
Không cài MMCV mới nhất hoặc chạy `python setup.py develop` rồi coi đó là
cách sửa đầy đủ. Chưa có bộ dependency BEVFusion đã xác nhận trên máy này.

Các lệnh ở mục 6 là quy trình chạy **sau khi bước port/build đạt**, không
phải lệnh hoàn tất cài đặt image base hiện tại.

## 5. Build dependency thử nghiệm cho BEVFusion

Sau khi xác nhận data/checkpoint đủ và năm module `mmcv`, `mmdet`,
`torchpack`, `mpi4py`, `numba` còn thiếu, bước tiếp theo là build image
dependency riêng. File [Dockerfile.blackwell-deps](../docker/Dockerfile.blackwell-deps)
dùng MMCV-full 1.7.2 và MMDetection 2.28.2. Người dùng đã cung cấp log
build thành công trên máy công ty: MMCV biên dịch từ source, `pip check`
đạt và `Dependency imports OK 1.7.2 2.28.2`. Chưa kiểm tra CUDA operator
trên GPU hoặc model BEVFusion; không phải bộ phiên bản gốc hay bản nâng cấp
đã bảo đảm tương đương checkpoint.

Dockerfile giữ các phiên bản package base qua pip constraints, build MMCV
từ source cho `sm_120`, và kiểm tra import. Nó chưa build operator BEVFusion.
Image/container DetZero hiện tại vẫn được giữ để dùng lại.

Sau khi file Dockerfile mới có trên máy công ty, chạy từ Terminal Ubuntu
(gõ `exit` trước nếu đang ở shell chính của container):

```bash
cd /home/khanhhuy/bevfusion
set -o pipefail
sudo docker build --progress=plain -f docker/Dockerfile.blackwell-deps -t bevfusion-blackwell:deps docker 2>&1 | tee /home/khanhhuy/bevfusion-env/build-deps.log
```

Lần build dependency đầu tiên dừng ở mpi4py 3.1.6 với
`new_compiler() got an unexpected keyword argument 'dry_run'` trong
setuptools của môi trường build tạm. Dockerfile đã bổ sung
`--no-build-isolation` để dùng setuptools 75.3.0 có sẵn trong image base.
Build lại sau bản sửa này đã thành công trên máy công ty.

Build MMCV có thể lâu. Nếu thất bại, giữ nguyên log và xử lý lỗi đầu tiên;
không tiếp tục cài mmdet khác phiên bản hoặc tắt kiểm tra CUDA để vượt lỗi.

Nếu build thành công, kiểm tra NMS của MMCV thực sự chạy GPU bằng container
tạm (lệnh trên host):

```bash
sudo docker run --rm --gpus all bevfusion-blackwell:deps python -c "import torch; from mmcv.ops import nms; b=torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.]],device='cuda'); s=torch.tensor([0.9,0.8],device='cuda'); dets,keep=nms(b,s,0.5); torch.cuda.synchronize(); assert keep.tolist()==[0],keep; print('MMCV CUDA NMS OK')"
```

Thành công ở đây mới chỉ xác nhận dependency và một CUDA operator MMCV.
Còn phải port/build operator BEVFusion và kiểm tra model trước mục tiếp theo.
Container `bevfusion-work` vẫn dùng image base cũ; build image mới không tự
thay môi trường của container đó.

## 6. Detection mini validation sau khi BEVFusion đã sẵn sàng

Trong container, kiểm tra import; nếu thất bại thì dừng ở đây:

```bash
python - <<'PY'
import torch, mmcv, mmdet, numba
from mpi4py import MPI
from torchpack import distributed
from mmdet3d.models import build_model
from mmdet3d.ops import bev_pool, Voxelization, spconv
print('BEVFusion imports OK:', torch.__version__, mmcv.__version__, mmdet.__version__)
PY
```

Import đạt vẫn phải kiểm tra CUDA operator và checkpoint tương thích khi
port. Tạo metadata bằng converter của repo, bên trong container để đường
dẫn nhất quán; không tạo ground-truth database phục vụ training:

```bash
python - <<'PY'
from tools.data_converter.nuscenes_converter import create_nuscenes_infos
create_nuscenes_infos(
    root_path='data/nuscenes', info_prefix='nuscenes',
    version='v1.0-mini', max_sweeps=10,
)
PY
```

Đầu ra là `data/nuscenes/nuscenes_infos_train.pkl` và
`data/nuscenes/nuscenes_infos_val.pkl`. Chạy bằng OpenMPI/torchpack một GPU,
batch size 1, một worker; không đổi độ phân giải/voxel/checkpoint để né lỗi:

```bash
mkdir -p outputs/mini-blackwell
set -o pipefail
export OMP_NUM_THREADS=2
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth \
  --eval bbox \
  --out outputs/mini-blackwell/predictions.pkl \
  --eval-options jsonfile_prefix=outputs/mini-blackwell \
  --cfg-options data.samples_per_gpu=1 data.test.samples_per_gpu=1 data.workers_per_gpu=1 data.test.load_interval=1 model.encoders.camera.backbone.init_cfg=None \
  2>&1 | tee outputs/mini-blackwell/detection.log
```

Đây là lần chạy dự kiến, chưa được thực thi trên RTX 5060. Chưa bảo đảm
VRAM 8 GB đủ; nếu OOM, giữ log để xác định bước và dung lượng bộ nhớ.
Nếu checkpoint thiếu/sai shape nhiều key, dừng và xử lý tương thích trước.

Kết quả mong đợi: `predictions.pkl`, `results_nusc.json`,
`metrics_summary.json`, `detection.log` trong `outputs/mini-blackwell`.
Mini validation có 81 sample; không so metric mini trực tiếp với full val.
File `results_nusc.json` là đầu vào cho bước DetZero trong
[hướng dẫn tracking/refining](RUN_TRACKING_AND_REFINING.md).

## Nguồn tham khảo

- [Docker Engine trên Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [PyTorch 2.7 và Blackwell](https://pytorch.org/blog/pytorch-2-7/)
- [Lệnh cài các phiên bản PyTorch](https://pytorch.org/get-started/previous-versions/)
