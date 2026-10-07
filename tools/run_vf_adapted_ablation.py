#!/usr/bin/env python3
"""Four VF ablations; preserves source images/calibration and uses scene-local UTM."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/nuscenes_vf6_01_adapted'
VERSION = 'v1.0-mini'
CONFIG = 'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/vf6_nav/sweeps2.yaml'
BRANCHES = {'tracking': 'none', 'grm': 'geometry', 'grm_prm': 'full'}


def read(path):
    return json.loads(Path(path).read_text())


def localize(source, target):
    """Translate one scene consistently; never modify source JSON or point files."""
    version = source / VERSION
    scenes, samples = read(version / 'scene.json'), read(version / 'sample.json')
    if len(scenes) != 1:
        raise ValueError('This adapter expects exactly one scene')
    poses = read(version / 'ego_pose.json')
    pose_by_token = {p['token']: p for p in poses}
    sd = read(version / 'sample_data.json')
    sensors = {s['token']: s for s in read(version / 'sensor.json')}
    calibs = {c['token']: c for c in read(version / 'calibrated_sensor.json')}
    first = min(samples, key=lambda s: s['timestamp'])['token']
    lidar = next(s for s in sd if s['sample_token'] == first and s['is_key_frame']
                 and sensors[calibs[s['calibrated_sensor_token']]['sensor_token']]['channel'] == 'LIDAR_TOP')
    origin = pose_by_token[lidar['ego_pose_token']]['translation'][:]
    missing = [s['filename'] for s in sd if not (source / s['filename']).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} sensor files: {missing[:5]}')
    annotations = read(version / 'sample_annotation.json')
    target.mkdir(parents=True)
    shutil.copytree(version, target / VERSION)
    # Relative sensor paths continue to work without copying large image/cloud files.
    for directory in ['samples', 'sweeps', 'maps']:
        if (source / directory).is_dir():
            (target / directory).symlink_to((source / directory).resolve(), target_is_directory=True)
    for rows, name in [(poses, 'ego_pose'), (annotations, 'sample_annotation')]:
        for row in rows:
            row['translation'] = [float(x) - float(o) for x, o in zip(row['translation'], origin)]
        (target / VERSION / f'{name}.json').write_text(json.dumps(rows))
    meta = read(source / 'conversion_meta.json') if (source / 'conversion_meta.json').exists() else {}
    meta.update(global_coord_mode='scene_local_utm', ablation_source=str(source),
                ablation_origin_utm=origin, ablation_frame_count=len(samples))
    (target / 'conversion_meta.json').write_text(json.dumps(meta, indent=2))
    # The source pose_report describes ENU while actual tables are UTM; record actual operation.
    (target / 'pose_report.json').write_text(json.dumps({
        'available': True, 'source': 'source UTM pose tables, translated to a scene-local origin',
        'coordinate_frame': 'UTM orientation/trajectory with constant translation removed',
        'origin_utm': origin, 'limitations': [
            'Source pose accuracy and physical calibration are not independently validated.',
            'Translation preserves source rotations, timestamps and relative sensor geometry.']}, indent=2))
    print(f'Prepared {len(samples)} frames; local origin {origin}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(SOURCE),
                        help='Source VF nuScenes dataset; images and calibration are preserved')
    parser.add_argument('--sweeps', type=int, choices=[0, 1, 2], default=2)
    parser.add_argument('--output-dir', default='outputs/vf-adapted-sweeps2-01')
    parser.add_argument('--legacy-center-correction', action='store_true',
                        help='Explicitly use the audited bevfusion-det.pth MIT legacy center convention')
    parser.add_argument('--car-includes-truck', action='store_true',
                        help='Track/refine truck too and evaluate car+truck together as VF Car')
    a = parser.parse_args()
    tracking_classes = 'car,truck,motorcycle,pedestrian' if a.car_includes_truck else 'car,motorcycle,pedestrian'
    os.chdir(ROOT)
    source = Path(a.data_root).resolve()
    output = Path(a.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a NEW output directory; existing run is preserved: {output}')
    for path in [source / VERSION / 'sample.json', ROOT / CONFIG,
                 ROOT / 'pretrained/bevfusion-det.pth', *[
                     ROOT / f'DetZero-main-2/checkpoints/{c}_{s}_model.pth'
                     for c in ['vehicle', 'pedestrian', 'cyclist'] for s in ['grm', 'prm']]]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not a.legacy_center_correction:
        raise ValueError('This run uses pretrained/bevfusion-det.pth; explicitly pass --legacy-center-correction for its audited legacy convention')
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'; logs.mkdir()
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT) + os.pathsep + env.get('PYTHONPATH', ''),
               OMP_NUM_THREADS='2', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', PYTHONUNBUFFERED='1')

    def run(name, command):
        print(f'\nSTEP: {name}\nLog: {logs / (name + ".log")}', flush=True)
        with (logs / f'{name}.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in child.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = child.wait()
        if code:
            raise RuntimeError(f'{name} failed (exit {code}); read {logs / (name + ".log")}. Later steps were not run.')

    py = sys.executable
    run('preflight', [py, '-c', "import torch, mmcv, mmdet, torchpack, filterpy, shapely, nuscenes; from mpi4py import MPI; from mmdet3d.models import build_model; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0))"])
    run('data_audit', [py, 'tools/validate_nuscenes_data.py', '--data-root', str(source),
                       '--version', VERSION, '--output', str(output / 'source_data_audit.json')])
    dataset = output / 'dataset_local'
    localize(source, dataset)
    (output / 'experiment.json').write_text(json.dumps({
        'source': str(source), 'dataset': str(dataset), 'version': VERSION,
        'sweeps': a.sweeps, 'checkpoint': 'pretrained/bevfusion-det.pth',
        'size_policy': 'grm', 'center_convention': 'mit-legacy-center-as-bottom',
        'tracking_classes': tracking_classes.split(','), 'car_includes_truck': a.car_includes_truck,
        'evaluation': 'custom VF AP/TP, three annotated classes; no NDS'}, indent=2))
    run('infos', [py, 'tools/data_converter/create_vf_infos.py', '--root-path', str(dataset),
                  '--version', VERSION, '--max-sweeps', str(a.sweeps), '--max-sweep-age', '1.0'])
    run('infos_audit', [py, 'tools/validate_nuscenes_data.py', '--data-root', str(dataset),
                        '--version', VERSION, '--infos', str(dataset / 'bevfusion_infos_val.pkl'),
                        '--output', str(output / 'infos_data_audit.json')])
    baseline_dir = output / 'baseline'; baseline_dir.mkdir()
    run('detection', ['torchpack', 'dist-run', '-np', '1', py, 'tools/test.py', CONFIG,
                     'pretrained/bevfusion-det.pth', '--format-only',
                     '--out', str(baseline_dir / 'predictions.pkl'), '--eval-options',
                     f'jsonfile_prefix={baseline_dir}', '--cfg-options',
                     f'data.test.dataset_root={dataset}',
                     f'data.test.ann_file={dataset / "bevfusion_infos_val.pkl"}',
                     f'data.test.pipeline.2.sweeps_num={a.sweeps}',
                     'data.samples_per_gpu=1', 'data.test.samples_per_gpu=1',
                     'data.workers_per_gpu=0', 'data.test.load_interval=1',
                     'model.encoders.camera.backbone.init_cfg=None'])
    baseline = baseline_dir / 'results_nusc_center_corrected.json'
    run('center_correction', [py, 'tools/correct_legacy_box_centers.py', '--data-root', str(dataset),
                              '--version', VERSION, '--input', str(baseline_dir / 'results_nusc.json'),
                              '--output', str(baseline), '--source-convention', 'mit-legacy-center-as-bottom'])
    for branch, mode in BRANCHES.items():
        run(branch, [py, 'DetZero-main-2/integration/run_pipeline.py', '--results_path', str(baseline),
                     '--data_root', str(dataset), '--version', VERSION, '--coordinate-mode', 'global',
                     '--device', 'cuda', '--tracking-device', 'cpu', '--classes', tracking_classes,
                     '--min_score', '0.1', '--max-gap-seconds', '0.6', '--size_policy', 'grm',
                     '--z-policy', 'center', '--intensity-mode', 'unit', '--refinement', mode,
                     '--output_dir', str(output / branch)])
    rows = []
    for branch in ['baseline', *BRANCHES]:
        prediction = baseline if branch == 'baseline' else output / branch / 'results_nusc_detzero_refined.json'
        metrics_path = output / branch / 'custom_metrics.json'
        run('eval_' + branch, [py, 'tools/evaluate_vf_comparison.py', '--data-root', str(dataset),
                              '--version', VERSION, '--baseline', str(prediction),
                              '--classes', 'car,motorcycle,pedestrian', '--output', str(metrics_path),
                              *(['--car-includes-truck'] if a.car_includes_truck else [])])
        metrics = read(metrics_path)['baseline']
        row = {'branch': branch, 'custom_mAP_percent': metrics['custom_mAP'] * 100}
        for cls, values in metrics['per_class'].items():
            row.update({cls + '_AP_percent': values['AP'] * 100,
                        **{cls + '_' + k: values[k] for k in ['trans_err', 'scale_err', 'orient_err']}})
        row['delta_custom_mAP_pp'] = row['custom_mAP_percent'] - (rows[0]['custom_mAP_percent'] if rows else row['custom_mAP_percent'])
        rows.append(row)
    with (output / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('\nBranch        custom mAP (%)    delta (pp)')
    for row in rows:
        print(f"{row['branch']:<13} {row['custom_mAP_percent']:>12.4f} {row['delta_custom_mAP_pp']:>12.4f}")
    print(f'\nALL FOUR BRANCHES COMPLETED\nResults: {output}\nCSV: {output / "comparison.csv"}')


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f'\nSTOPPED: {exc}', file=sys.stderr)
        sys.exit(1)
