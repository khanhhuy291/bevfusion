cd /workspace/bevfusion

command -v ffmpeg

python tools/render_vf_videos.py \
  --data-root outputs/vf-adapted-sweeps2-01/dataset_local \
  --version v1.0-mini \
  --baseline outputs/vf-adapted-sweeps2-01/baseline/results_nusc_center_corrected.json \
  --tracking outputs/vf-adapted-sweeps2-01/grm_prm/results_nusc_detzero_tracking.json \
  --min-score 0.1 \
  --output-dir outputs/vf-adapted-sweeps2-01/videos