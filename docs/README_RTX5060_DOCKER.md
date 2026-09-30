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
| CUDA operator MMCV | Người dùng xác nhận `MMCV CUDA NMS OK` trên RTX 5060 |
| GRM/PRM trên GPU | Log người dùng xác nhận full pipeline với `--device cuda`: sáu checkpoint khớp key, 209 track processed, 119 insufficient_points trên tổng 328 track |
| Build CUDA extensions BEVFusion | Log người dùng xác nhận đã tạo/copy đủ 12 extension; NVCC dùng `sm_120` |
| BEVFusion import và operator | Người dùng xác nhận registry, BEV pooling, voxelization, sparse convolution và NMS đạt |
| BEVFusion detection trên RTX 5060 | **Log người dùng xác nhận hoàn tất mini val 81/81: mAP 0.5811, NDS 0.5823; đã xuất predictions.pkl và results_nusc.json** |
| DetZero tracking trên kết quả mini | Log người dùng xác nhận 2 scene (41 + 40 frame), 328 track, 3451 matched box; `tracking_only: 328`, chạy CPU trong container detection |

`GPU COMPUTE OK` xác nhận PyTorch chạy GPU, không xác nhận mọi CUDA
operator của BEVFusion đã hoạt động. Một test bị skip vì chưa có torchpack
không có nghĩa môi trường BEVFusion đã hoàn tất.

## 1. Mở lại môi trường mỗi ngày

Các lệnh trong mục này chạy ở Terminal Ubuntu, bên ngoài container.
Không cần kích hoạt Conda `(base)` để dùng Docker.

**Dùng `bevfusion-detection` cho bước BEVFusion.** Container này đã có đủ
dependency và NMS chạy GPU thành công. `bevfusion-work` vẫn dùng image
base/DetZero, không tự nhận package từ image mới.

```bash
sudo docker start -ai bevfusion-detection
```

Nếu container detection đang `Up`, dùng `sudo docker exec -it
bevfusion-detection /bin/bash`. Các lệnh `bevfusion-work` bên dưới chỉ dành
cho môi trường base cũ.

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
`TORCH_CUDA_ARCH_LIST=12.0` không đủ với setup.py cũ ghi cứng `-gencode`;
cần cập nhật bản sửa build ở mục 6 trước.
Không cài MMCV mới nhất hoặc chạy `python setup.py develop` rồi coi đó là
cách sửa đầy đủ. Bộ dependency và bản sửa đã chạy mini detection được ghi ở mục 5–7.

Các lệnh ở mục 7 là quy trình chạy **sau khi bước port/build đạt**, không
phải lệnh hoàn tất cài đặt image base hiện tại.

## 5. Build dependency thử nghiệm cho BEVFusion

Sau khi xác nhận data/checkpoint đủ và năm module `mmcv`, `mmdet`,
`torchpack`, `mpi4py`, `numba` còn thiếu, bước tiếp theo là build image
dependency riêng. File [Dockerfile.blackwell-deps](../docker/Dockerfile.blackwell-deps)
dùng MMCV-full 1.7.2 và MMDetection 2.28.2. Người dùng đã cung cấp log
build thành công trên máy công ty: MMCV biên dịch từ source, `pip check`
đạt và `Dependency imports OK 1.7.2 2.28.2`. NMS MMCV và model BEVFusion
đã chạy trên GPU sau các bản sửa ở mục 6–7. Đây là bộ phiên bản port;
chưa đối chiếu số học với môi trường gốc.

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

## 6. Build extension riêng của BEVFusion

Bản sửa trong repo cho phép `setup.py` nhận `TORCH_CUDA_ARCH_LIST`, thay vì
ghi cứng `-gencode`; mặc định vẫn giữ 7.0/7.5/8.0/8.6 cho luồng cũ.
Hai chỗ gọi Tensor.type() cũ được thay bằng scalar_type()/is_cuda().
Người dùng đã gửi log build hoàn tất, copy đủ 12 file extension và có cờ
`-gencode=arch=compute_120,code=sm_120`. Các warning `Tensor.data<T>()`
deprecated chưa làm build thất bại. Operator và mini detection đã chạy đạt
sau các bản sửa Python ở mục 7.
Máy công ty cần nhận các thay đổi code này trước khi chạy lệnh dưới.

Trong container **bevfusion-detection**:

```bash
cd /workspace/bevfusion
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2
mkdir -p outputs/mini-blackwell
set -o pipefail
python setup.py build_ext --inplace 2>&1 | tee outputs/mini-blackwell/build-bevfusion.log
```

`build_ext --inplace` đặt extension cạnh mã nguồn trong bind mount, không
cần ghi vào venv do root sở hữu. PYTHONPATH đã trỏ vào repo. File `.so`
này phụ thuộc Linux/Python/PyTorch/CUDA của môi trường build; không copy
chúng sang Mac hoặc dùng với bộ PyTorch khác. Lỗi build cần xử lý trước
khi chạy tiếp; xem `tail -n 100 outputs/mini-blackwell/build-bevfusion.log`.

## 7. Detection mini validation sau khi BEVFusion đã sẵn sàng

Sau bản sửa registry, người dùng đã xác nhận `BEVFusion imports + registry OK`
và `BEVFUSION CUDA CHECKS OK` (BEV pooling, voxelization, sparse convolution,
NMS). Các kiểm tra nhỏ này không thay thế việc chạy model đầy đủ.

Lần kiểm tra import đầu tiên sau build bị lỗi `SparseConv2d is already
registered in conv layer`: MMCV 1.7 đã đăng ký các lớp cùng tên. Bản sửa
trong `mmdet3d/ops/spconv/conv.py` dùng `force=True` cho 10 lớp sparse của
repo, để registry chọn đúng implementation đi cùng SparseConvTensor,
CUDA extension và checkpoint của repo. Không thay bằng spconv của MMCV.
Đây là sửa Python, không cần build lại `.so`; chạy Python mới để kiểm tra.

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
batch size 1, nạp dữ liệu trong tiến trình chính (`workers_per_gpu=0`);
không đổi độ phân giải/voxel/checkpoint để né lỗi:

```bash
mkdir -p outputs/mini-blackwell
set -o pipefail
export OMP_NUM_THREADS=2
# Checkpoint pretrained/bevfusion-det.pth đã được người dùng cung cấp/tin cậy.
# Cho phép MMCV cũ đọc metadata checkpoint với PyTorch >=2.6.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth \
  --eval bbox \
  --out outputs/mini-blackwell/predictions.pkl \
  --eval-options jsonfile_prefix=outputs/mini-blackwell \
  --cfg-options data.samples_per_gpu=1 data.test.samples_per_gpu=1 data.workers_per_gpu=0 data.test.load_interval=1 model.encoders.camera.backbone.init_cfg=None \
  2>&1 | tee outputs/mini-blackwell/detection.log
```

Lần chạy đầu với một DataLoader worker bị SIGABRT trong libfabric tại
`fork()`. Nghi vấn là tạo worker sau khi MPI khởi tạo; thử lại với
`workers_per_gpu=0` để tránh fork từ DataLoader. Log chạy lại đã qua bước
nạp dữ liệu và checkpoint, nhưng dừng ở lỗi DDP dưới đây trước sample đầu tiên.
Tham khảo [Open MPI về fork](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/fork-system-popen.html).

Với MMCV 1.7.2/PyTorch 2.7.1, có thể gặp
`AttributeError: 'MMDistributedDataParallel' object has no attribute '_use_replicated_tensor_module'`.
`tools/test.py` bổ sung cờ này bằng `False` nếu thiếu, ngay sau khi tạo
MMDistributedDataParallel, để MMCV gọi module thường. Giữ nguyên cờ nếu
PyTorch cũ đã có. Đây là sửa Python, không cần build CUDA hay Docker lại.
Bản sửa đã kiểm tra cú pháp và hai nhánh có/không có thuộc tính trên máy local;
log người dùng sau khi thêm cả sửa scatter bên dưới đã xác nhận detection hoàn tất. Tham khảo
[MMCV 1.7.2 DDP forward](https://github.com/open-mmlab/mmcv/blob/v1.7.2/mmcv/parallel/distributed.py).

Log sau sửa DDP đã qua lỗi thiếu thuộc tính, nhưng dừng khi scatter dữ liệu:
`AttributeError: 'int' object has no attribute 'type'` trong `_get_stream`.
MMCV 1.x truyền ID GPU dạng số, còn PyTorch >=2.1 nhận `torch.device`.
`tools/test.py` điều chỉnh tham chiếu `_get_stream` của MMCV trong tiến trình
test: đổi ID số sang device, giữ nguyên device có sẵn, và gọi hàm gốc của
PyTorch để giữ cơ chế tạo/cache stream. Chỉ bật với MMCV <2 và PyTorch >=2.1;
không ghi đè file thư viện trong `/opt/venv`. Không cần build lại extension.
Đã kiểm tra cú pháp và chuyển đổi đối số local; log người dùng ngày 2026-09-30
xác nhận detection thực tế hoàn tất sau bản sửa này.

Đã chạy hết mini validation 81/81 trên RTX 5060 Laptop 8 GB với cấu hình
trên: mAP 0.5811, NDS 0.5823, mATE 0.4057, mASE 0.4462,
mAOE 0.4769, mAVE 0.4334, mAAE 0.3200. Đây là kết quả từ log người dùng;
không phải xác nhận training. Lần chạy DetZero sau đó đã xác nhận GRM/PRM
trên GPU: 209 track processed, 119 track insufficient_points. Đánh giá từ
log người dùng: mAP 0.5811 → 0.5776, NDS 0.5823 → 0.5804;
mATE 0.4057 → 0.4095, mASE 0.4462 → 0.4566,
mAOE 0.4769 → 0.4632, mAVE 0.4334 → 0.4346, mAAE giữ 0.3200.
Góc quay cải thiện nhưng tổng thể chưa vượt baseline detection; chưa đánh
giá chất lượng tracking (ID/association). Với `--size_policy auto`, báo cáo
ghi 158 GRM accepted, 51 rejected_using_prior và 209 PRM applied.
Accepted chỉ có nghĩa qua ngưỡng kích thước, không chứng minh tốt hơn GT.
Đối chứng `geometry + auto` từ log người dùng: mAP 0.5809, NDS 0.5812,
mATE 0.4058, mASE 0.4566, mAOE 0.4769, mAVE 0.4334.
So với geometry, thêm PRM trong full giảm mAP/NDS và tăng sai số vị trí,
nhưng giảm sai số góc quay. Geometry đã tăng sai số kích thước so với
baseline; chưa tách tác động GRM khỏi việc dùng median kích thước khi
`auto` từ chối GRM. Đối chứng tiếp theo phù hợp là
`--refinement geometry --size_policy detector_prior` (median kích thước,
không chạy GRM/PRM), với thư mục output riêng.
Track có tổng điểm crop dưới 10 được
giữ box tracking đầu vào, không chạy refiner; không tính là refine thành công.
Log có cảnh báo `destroy_process_group() was not called` khi thoát,
sau khi đã lưu prediction và in metric; cần phân biệt với lỗi inference.
Con số 70 task/s trong log thuộc bước đổi định dạng detection, không phải FPS model.

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
