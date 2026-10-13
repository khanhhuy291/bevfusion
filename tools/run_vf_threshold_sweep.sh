#!/usr/bin/env bash
# Run inside bevfusion-detection; only min_score changes across the five runs.
set -euo pipefail

vf_sweep_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$vf_sweep_repo"
vf_sweep_python="${VF_SWEEP_PYTHON:-/opt/venv/bin/python}"
vf_sweep_run="${1:-outputs/vf-car-truck-sweeps2-01}"
vf_sweep_dataset="$vf_sweep_run/dataset_local"
vf_sweep_baseline="$vf_sweep_run/baseline/results_nusc_center_corrected.json"

for vf_sweep_file in \
  "$vf_sweep_baseline" \
  "$vf_sweep_dataset/v1.0-mini/sample.json" \
  "$vf_sweep_dataset/v1.0-mini/sample_annotation.json" \
  tools/evaluate_vf_comparison.py tools/vf_iou3d.py
do
  if [[ ! -f "$vf_sweep_file" ]]; then
    echo "Missing file: $vf_sweep_file" >&2
    exit 1
  fi
done
"$vf_sweep_python" -c 'from tools.vf_iou3d import evaluate_iou3d; print("IoU 3D evaluator imports OK")'

export OMP_NUM_THREADS=2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$vf_sweep_repo${PYTHONPATH:+:$PYTHONPATH}"
# A unique directory preserves previous runs, including the original score=0.1.
vf_sweep_output="$(mktemp -d "$vf_sweep_run/threshold_sweep_XXXXXX")"
echo "Sweep output: $vf_sweep_output"

for vf_sweep_score in 0.05 0.10 0.15 0.20 0.25
do
  vf_sweep_case="$vf_sweep_output/score_$vf_sweep_score"
  mkdir "$vf_sweep_case"
  echo "Running DetZero with min_score=$vf_sweep_score"
  "$vf_sweep_python" DetZero-main-2/integration/run_pipeline.py \
    --results_path "$vf_sweep_baseline" \
    --data_root "$vf_sweep_dataset" \
    --version v1.0-mini \
    --tracking_cfg DetZero-main-2/tracking/tools/cfgs/tk_model_cfgs/nuscenes_detzero_track.yaml \
    --checkpoint_dir DetZero-main-2/checkpoints \
    --coordinate-mode global \
    --device cuda \
    --tracking-device cpu \
    --classes car,truck,motorcycle,pedestrian \
    --min_score "$vf_sweep_score" \
    --max-gap-seconds 0.6 \
    --size_policy grm \
    --z-policy center \
    --intensity-mode unit \
    --refinement full \
    --output_dir "$vf_sweep_case" \
    2>&1 | tee "$vf_sweep_case/pipeline.log"

  "$vf_sweep_python" tools/evaluate_vf_comparison.py \
    --data-root "$vf_sweep_dataset" \
    --version v1.0-mini \
    --baseline "$vf_sweep_baseline" \
    --refined "$vf_sweep_case/results_nusc_detzero_refined.json" \
    --classes car,motorcycle,pedestrian \
    --car-includes-truck \
    --iou-thresholds 0.25 0.5 0.7 \
    --output "$vf_sweep_case/comparison_iou3d.json" \
    2>&1 | tee "$vf_sweep_case/evaluation.log"
done

"$vf_sweep_python" - "$vf_sweep_output" <<'PY'
import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = []
input_hash = None
for score in ['0.05', '0.10', '0.15', '0.20', '0.25']:
    directory = root / ('score_' + score)
    report = json.loads((directory / 'run_report.json').read_text())
    metrics = json.loads((directory / 'comparison_iou3d.json').read_text())
    assert abs(report['arguments']['min_score'] - float(score)) < 1e-12
    assert report['arguments']['refinement'] == 'full'
    assert report['arguments']['size_policy'] == 'grm'
    if input_hash is None:
        input_hash = report['input_sha256']
    assert report['input_sha256'] == input_hash == metrics['baseline']['prediction_sha256']
    row = {'min_score': float(score), 'tracks': report['tracks'],
           'processed_tracks': report['status_counts'].get('processed', 0),
           'insufficient_points_tracks': report['status_counts'].get('insufficient_points', 0),
           'matched_boxes_replaced': report['matched_boxes_replaced'],
           'center_distance_mAP_percent': metrics['refined']['custom_mAP'] * 100}
    for threshold in ['0.25', '0.5', '0.7']:
        row['mAP3D_' + threshold + '_percent'] = metrics['refined']['iou3d']['mAP_by_threshold'][threshold] * 100
        row['delta_mAP3D_' + threshold + '_pp'] = metrics['delta_iou3d_mAP_by_threshold'][threshold] * 100
        for cls in ['car', 'motorcycle', 'pedestrian']:
            value = metrics['refined']['iou3d']['per_class'][cls]['by_threshold'][threshold]
            row[cls + '_AP3D_' + threshold + '_percent'] = value['AP'] * 100
            row[cls + '_TP3D_' + threshold] = value['TP']
            row[cls + '_FP3D_' + threshold] = value['FP']
            row[cls + '_FN3D_' + threshold] = value['FN']
    rows.append(row)
with (root / 'threshold_comparison.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('\nmin_score  mAP3D@0.5(%)  Car AP(%)  Motorcycle AP(%)  Pedestrian AP(%)')
for row in rows:
    print(f"{row['min_score']:9.2f} {row['mAP3D_0.5_percent']:13.4f} "
          f"{row['car_AP3D_0.5_percent']:10.4f} {row['motorcycle_AP3D_0.5_percent']:17.4f} "
          f"{row['pedestrian_AP3D_0.5_percent']:17.4f}")
print('Saved:', root / 'threshold_comparison.csv')
PY
