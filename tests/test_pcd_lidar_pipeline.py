import json
import math

import numpy as np
import pytest

from tools.pcd_lidar_utils import (compare_results, interpolate_tum_pose,
                                   model_box_to_nuscenes_record, prepare_model_points,
                                   read_binary_pcd, read_tum_trajectory,
                                   write_identity_nuscenes, write_nuscenes_bridge_dataset)


def write_pcd(path, points, data="binary"):
    points = np.asarray(points, dtype="<f4")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "# timestamp 123.25\nVERSION 0.7\nFIELDS x y z intensity\n"
        "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\nDATA {data}\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii")); stream.write(points.tobytes())


def test_pcd_to_five_features_and_range_filter(tmp_path):
    path = tmp_path / "frame.pcd"
    raw = np.array([[1, 2, 0, 4], [20, 0, 0, 8]], dtype=np.float32)
    write_pcd(path, raw)
    points, meta = read_binary_pcd(path)
    assert meta["timestamp"] == 123.25
    model = prepare_model_points(points, meta["fields"], [-5, -5, -3, 5, 5, 3],
                                 intensity_scale=.5)
    np.testing.assert_array_equal(model, [[1, 2, 0, 2, 0]])
    assert model.dtype == np.float32 and model.flags.c_contiguous


def test_reader_rejects_non_binary_and_truncated_payload(tmp_path):
    path = tmp_path / "bad.pcd"
    write_pcd(path, [[1, 2, 3, 4]], data="ascii")
    with pytest.raises(ValueError, match="DATA binary"):
        read_binary_pcd(path)
    write_pcd(path, [[1, 2, 3, 4]])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="payload bytes"):
        read_binary_pcd(path)


def test_model_box_conversion_uses_gravity_center_and_nuscenes_yaw():
    record = model_box_to_nuscenes_record("t", [1, 2, -.5, 2, 4, 2, 0, 0, 0], .8, "car")
    assert record["translation"] == [1., 2., .5]
    assert record["size"] == [2., 4., 2.]
    assert record["rotation"] == pytest.approx([math.sqrt(.5), 0, 0, -math.sqrt(.5)])
    assert record["attribute_name"] == "vehicle.parked"


def test_tum_interpolation_and_global_box_conversion(tmp_path):
    trajectory_path = tmp_path / "trajectory.tum"
    trajectory_path.write_text(
        "# timestamp x y z qx qy qz qw\n"
        "1 0 0 0 0 0 0 1\n"
        f"3 2 0 0 0 0 {math.sqrt(.5)} {math.sqrt(.5)}\n"
    )
    trajectory = read_tum_trajectory(trajectory_path)
    pose = interpolate_tum_pose(trajectory, 2., max_nearest_gap=1.01)
    np.testing.assert_allclose(pose["translation"], [1, 0, 0])
    np.testing.assert_allclose(pose["rotation"], [math.cos(math.pi/8), 0, 0,
                                                   math.sin(math.pi/8)])
    record = model_box_to_nuscenes_record(
        "t", [1, 0, 0, 2, 4, 2, -math.pi/2, 1, 0], .8, "car", pose)
    np.testing.assert_allclose(record["translation"],
                               [1 + math.sqrt(.5), math.sqrt(.5), 1], atol=1e-7)
    np.testing.assert_allclose(record["velocity"], [math.sqrt(.5), math.sqrt(.5)], atol=1e-7)


def test_tum_rejects_extrapolation_and_sparse_match(tmp_path):
    path = tmp_path / "trajectory.tum"
    path.write_text("1 0 0 0 0 0 0 1\n3 0 0 0 0 0 0 1\n")
    trajectory = read_tum_trajectory(path)
    with pytest.raises(ValueError, match="outside trajectory"):
        interpolate_tum_pose(trajectory, .9, max_nearest_gap=2)
    with pytest.raises(ValueError, match="Nearest trajectory pose"):
        interpolate_tum_pose(trajectory, 2, max_nearest_gap=.5)


def test_identity_dataset_and_change_report(tmp_path):
    points = np.array([[1, 2, 0, 0, 0]], dtype=np.float32)
    base = model_box_to_nuscenes_record("t0", [1, 2, 0, 2, 4, 2, 0], .8, "car")
    path = write_identity_nuscenes(tmp_path / "dataset", [
        {"sample_token": "t0", "timestamp": 1., "points": points},
    ], {"t0": [base]})
    assert path == tmp_path / "baseline" / "results_nusc.json"
    binary = np.fromfile(tmp_path / "dataset/samples/LIDAR_TOP/t0.bin", dtype=np.float32)
    np.testing.assert_array_equal(binary.reshape(-1, 5), points)
    refined = json.loads(path.read_text())
    refined["results"]["t0"][0]["size"][0] += .2
    refined_path = tmp_path / "refined.json"
    refined_path.write_text(json.dumps(refined))
    report = compare_results(path, refined_path)
    assert report["paired_detections"] == 1
    assert report["size_l2_change_m"]["changed"] == 1
    assert report["quality_conclusion"] == "unavailable_without_ground_truth"


def test_bridge_dataset_writes_interpolated_pose(tmp_path):
    points = np.array([[1, 2, 0, 0, 0]], dtype=np.float32)
    record = model_box_to_nuscenes_record("t0", [1, 2, 0, 2, 4, 2, 0], .8, "car")
    pose = {"translation": np.array([3., 4., 5.]),
            "rotation": np.array([1., 0., 0., 0.])}
    write_nuscenes_bridge_dataset(tmp_path / "dataset", [
        {"sample_token": "t0", "timestamp": 1., "points": points, "pose": pose},
    ], {"t0": [record]})
    rows = json.loads((tmp_path / "dataset/v1.0-mini/ego_pose.json").read_text())
    assert rows[0]["translation"] == [3., 4., 5.]
    assert rows[0]["rotation"] == [1., 0., 0., 0.]
