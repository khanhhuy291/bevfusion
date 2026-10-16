"""Pure helpers for binary PCD inference and DetZero interchange."""

import json
import math
from pathlib import Path

import numpy as np


_NUMPY_TYPES = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
    ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
}


def _single(header, key, path):
    values = header.get(key, [])
    if len(values) != 1:
        raise ValueError(f"{path}: expected exactly one {key} line")
    return values[0]


def read_binary_pcd(path):
    """Read scalar fields from an uncompressed ``DATA binary`` PCD."""
    path = Path(path)
    header, comments = {}, []
    with path.open("rb") as stream:
        for _ in range(128):
            raw = stream.readline()
            if not raw:
                raise ValueError(f"{path}: truncated PCD header")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}: PCD header is not ASCII") from exc
            if not line:
                continue
            if line.startswith("#"):
                comments.append(line[1:].strip())
                continue
            parts = line.split()
            header.setdefault(parts[0].upper(), []).append(parts[1:])
            if parts[0].upper() == "DATA":
                break
        else:
            raise ValueError(f"{path}: PCD header is too long")

        if _single(header, "DATA", path) != ["binary"]:
            raise ValueError(f"{path}: only DATA binary PCD is supported")
        fields = _single(header, "FIELDS", path)
        sizes = [int(v) for v in _single(header, "SIZE", path)]
        types = [v.upper() for v in _single(header, "TYPE", path)]
        counts = [int(v) for v in _single(header, "COUNT", path)]
        if not fields or len(set(fields)) != len(fields):
            raise ValueError(f"{path}: fields must be non-empty and unique")
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f"{path}: inconsistent FIELDS/SIZE/TYPE/COUNT")
        if any(count != 1 for count in counts):
            raise ValueError(f"{path}: vector-valued fields are not supported")
        try:
            dtype = np.dtype([(name, _NUMPY_TYPES[(kind, size)])
                              for name, kind, size in zip(fields, types, sizes)])
        except KeyError as exc:
            raise ValueError(f"{path}: unsupported PCD TYPE/SIZE {exc.args[0]}") from exc
        count = int(_single(header, "POINTS", path)[0])
        width = int(_single(header, "WIDTH", path)[0])
        height = int(_single(header, "HEIGHT", path)[0])
        if count <= 0 or width <= 0 or height <= 0 or width * height != count:
            raise ValueError(f"{path}: invalid POINTS/WIDTH/HEIGHT")
        payload = stream.read()

    expected = count * dtype.itemsize
    if len(payload) != expected:
        raise ValueError(f"{path}: expected {expected} payload bytes, found {len(payload)}")
    records = np.frombuffer(payload, dtype=dtype, count=count)
    points = np.column_stack([records[name] for name in fields]).astype(np.float32, copy=False)
    if not np.isfinite(points).all():
        raise ValueError(f"{path}: point cloud contains NaN or Inf")
    timestamp = None
    for comment in comments:
        if comment.startswith("timestamp "):
            try:
                timestamp = float(comment.split(None, 1)[1])
            except ValueError as exc:
                raise ValueError(f"{path}: invalid timestamp comment") from exc
    return points, dict(fields=fields, points=count, width=width, height=height,
                        timestamp=timestamp, data="binary")


def prepare_model_points(points, fields, point_cloud_range, transform=None,
                         intensity_scale=1.0):
    """Convert named PCD columns to ``x,y,z,intensity,time_lag`` float32."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != len(fields):
        raise ValueError("points shape must match fields")
    indices = {name: index for index, name in enumerate(fields)}
    missing = [name for name in ("x", "y", "z", "intensity") if name not in indices]
    if missing:
        raise ValueError("PCD is missing required fields: " + ", ".join(missing))
    if not np.isfinite(intensity_scale):
        raise ValueError("intensity_scale must be finite")
    xyz = points[:, [indices["x"], indices["y"], indices["z"]]].astype(np.float64)
    if transform is not None:
        transform = np.asarray(transform, dtype=np.float64)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])):
            raise ValueError("transform must be a finite homogeneous 4x4 matrix")
        xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    intensity = points[:, indices["intensity"]].astype(np.float64) * intensity_scale
    output = np.column_stack([xyz, intensity, np.zeros(len(points))]).astype(np.float32)
    limits = np.asarray(point_cloud_range, dtype=np.float64)
    if limits.shape != (6,) or not np.isfinite(limits).all() or np.any(limits[:3] >= limits[3:]):
        raise ValueError("point_cloud_range must be [xmin,ymin,zmin,xmax,ymax,zmax]")
    mask = ((output[:, :3] >= limits[:3]) & (output[:, :3] < limits[3:])).all(axis=1)
    output = output[mask]
    if not len(output):
        raise ValueError("No points remain after transform/range filtering")
    return np.ascontiguousarray(output)


def quaternion_matrix(quaternion):
    """Return a 3x3 matrix for a normalized ``[w,x,y,z]`` quaternion."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must contain four finite values")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = quaternion / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def quaternion_multiply(left, right):
    """Hamilton product for two ``[w,x,y,z]`` quaternions."""
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64)
    product = np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2])
    return product / np.linalg.norm(product)


def read_tum_trajectory(path):
    """Read TUM ``timestamp tx ty tz qx qy qz qw`` as world-from-LiDAR poses."""
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != 8:
            raise ValueError(f"{path}:{line_number}: expected 8 TUM columns")
        try:
            rows.append([float(value) for value in values])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric value") from exc
    array = np.asarray(rows, dtype=np.float64)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f"{path}: trajectory needs at least two finite poses")
    if np.any(np.diff(array[:, 0]) <= 0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    quaternions = array[:, [7, 4, 5, 6]]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError(f"{path}: zero quaternion")
    quaternions /= norms[:, None]
    # q and -q describe the same rotation. A continuous sign makes SLERP stable.
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1
    return {"timestamps": array[:, 0], "translations": array[:, 1:4],
            "rotations": quaternions, "source": str(Path(path).resolve())}


def _slerp(left, right, fraction):
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if dot < 0:
        right, dot = -right, -dot
    if dot > .9995:
        result = left + fraction * (right - left)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    result = (math.sin((1 - fraction) * angle) * left
              + math.sin(fraction * angle) * right) / math.sin(angle)
    return result / np.linalg.norm(result)


def interpolate_tum_pose(trajectory, timestamp, max_nearest_gap=0.11, invert=False):
    """Interpolate a TUM pose at one timestamp without extrapolation."""
    timestamps = trajectory["timestamps"]
    timestamp = float(timestamp)
    if timestamp < timestamps[0] or timestamp > timestamps[-1]:
        raise ValueError(f"Timestamp {timestamp:.9f} is outside trajectory coverage")
    right = int(np.searchsorted(timestamps, timestamp, side="left"))
    if right < len(timestamps) and abs(timestamps[right] - timestamp) < 1e-9:
        left = right; fraction = 0.0
    else:
        left = max(0, right - 1)
        if right >= len(timestamps):
            right = len(timestamps) - 1
        span = timestamps[right] - timestamps[left]
        fraction = (timestamp - timestamps[left]) / span
    nearest_gap = min(abs(timestamp - timestamps[left]), abs(timestamps[right] - timestamp))
    if nearest_gap > max_nearest_gap:
        raise ValueError(f"Nearest trajectory pose is {nearest_gap:.6f}s away; limit is {max_nearest_gap:.6f}s")
    translation = ((1 - fraction) * trajectory["translations"][left]
                   + fraction * trajectory["translations"][right])
    rotation = _slerp(trajectory["rotations"][left], trajectory["rotations"][right], fraction)
    if invert:
        inverse_rotation = rotation * np.array([1., -1., -1., -1.])
        translation = -(quaternion_matrix(inverse_rotation) @ translation)
        rotation = inverse_rotation
    return {"translation": translation, "rotation": rotation,
            "nearest_gap": float(nearest_gap), "left_timestamp": float(timestamps[left]),
            "right_timestamp": float(timestamps[right])}


def model_box_to_nuscenes_record(sample_token, box, score, class_name,
                                  world_from_lidar=None):
    """Convert a BEVFusion LiDAR box to the JSON convention used by nuScenes."""
    box = np.asarray(box, dtype=np.float64)
    if box.shape[0] < 7 or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
        raise ValueError("invalid detector box")
    # LiDARInstance3DBoxes stores bottom z and [width, length, height].
    yaw = -float(box[6]) - math.pi / 2
    orientation = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    center = np.array([box[0], box[1], box[2] + box[5] / 2], dtype=np.float64)
    velocity_3d = np.r_[box[7:9] if box.shape[0] >= 9 else [0.0, 0.0], 0.0]
    if world_from_lidar is not None:
        pose_rotation = np.asarray(world_from_lidar["rotation"], dtype=np.float64)
        rotation_matrix = quaternion_matrix(pose_rotation)
        center = rotation_matrix @ center + np.asarray(world_from_lidar["translation"])
        velocity_3d = rotation_matrix @ velocity_3d
        orientation = quaternion_multiply(pose_rotation, orientation)
    velocity = velocity_3d[:2]
    speed = float(np.linalg.norm(velocity))
    if speed > 0.2 and class_name in {"car", "truck", "bus", "trailer", "construction_vehicle"}:
        attribute = "vehicle.moving"
    elif speed > 0.2 and class_name in {"bicycle", "motorcycle"}:
        attribute = "cycle.with_rider"
    elif class_name == "pedestrian":
        attribute = "pedestrian.standing"
    elif class_name == "bus":
        attribute = "vehicle.stopped"
    else:
        attribute = {
            "car": "vehicle.parked", "truck": "vehicle.parked",
            "trailer": "vehicle.parked", "construction_vehicle": "vehicle.parked",
            "bicycle": "cycle.without_rider", "motorcycle": "cycle.without_rider",
            "barrier": "", "traffic_cone": "",
        }.get(class_name, "")
    return {
        "sample_token": sample_token,
        "translation": center.tolist(),
        "size": [float(box[3]), float(box[4]), float(box[5])],
        "rotation": orientation.tolist(),
        "velocity": [float(v) for v in velocity],
        "detection_name": class_name, "detection_score": float(score),
        "attribute_name": attribute,
    }


def write_nuscenes_bridge_dataset(root, frames, results):
    """Write the minimal nuScenes tables/files consumed by the DetZero bridge."""
    root = Path(root)
    version = root / "v1.0-mini"
    lidar_dir = root / "samples" / "LIDAR_TOP"
    version.mkdir(parents=True, exist_ok=False)
    lidar_dir.mkdir(parents=True, exist_ok=True)
    scene_token, sensor_token, calib_token = "pcd-scene", "pcd-lidar-sensor", "pcd-lidar-calib"
    samples, sample_data, ego_poses = [], [], []
    for index, frame in enumerate(frames):
        token = frame["sample_token"]
        timestamp = int(round(float(frame["timestamp"]) * 1e6))
        ego_token, sd_token = f"pcd-ego-{index:06d}", f"pcd-sd-{index:06d}"
        filename = f"samples/LIDAR_TOP/{token}.bin"
        np.asarray(frame["points"], dtype=np.float32).tofile(root / filename)
        samples.append({"token": token, "timestamp": timestamp, "scene_token": scene_token,
                        "prev": frames[index - 1]["sample_token"] if index else "",
                        "next": frames[index + 1]["sample_token"] if index + 1 < len(frames) else ""})
        sample_data.append({"token": sd_token, "sample_token": token, "ego_pose_token": ego_token,
                            "calibrated_sensor_token": calib_token, "timestamp": timestamp,
                            "is_key_frame": True, "filename": filename,
                            "prev": f"pcd-sd-{index-1:06d}" if index else "",
                            "next": f"pcd-sd-{index+1:06d}" if index + 1 < len(frames) else ""})
        pose = frame.get("pose", {"translation": [0., 0., 0.], "rotation": [1., 0., 0., 0.]})
        ego_poses.append({"token": ego_token, "timestamp": timestamp,
                          "translation": np.asarray(pose["translation"], dtype=float).tolist(),
                          "rotation": np.asarray(pose["rotation"], dtype=float).tolist()})
    tables = {
        "sample": samples,
        "scene": [{"token": scene_token, "name": "pcd-tum-trajectory",
                   "first_sample_token": samples[0]["token"], "last_sample_token": samples[-1]["token"],
                   "nbr_samples": len(samples), "description": "PCD sequence without measured ego poses"}],
        "sample_data": sample_data, "ego_pose": ego_poses,
        "sensor": [{"token": sensor_token, "channel": "LIDAR_TOP", "modality": "lidar"}],
        "calibrated_sensor": [{"token": calib_token, "sensor_token": sensor_token,
                               "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}],
    }
    for name, rows in tables.items():
        (version / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    payload = {"meta": {"use_camera": False, "use_lidar": True, "use_radar": False,
                        "use_map": False, "use_external": False}, "results": results}
    result_path = root.parent / "baseline" / "results_nusc.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload) + "\n")
    return result_path


def write_identity_nuscenes(root, frames, results):
    """Compatibility wrapper for tests and explicitly stationary diagnostics."""
    return write_nuscenes_bridge_dataset(root, frames, results)


def compare_results(baseline_path, refined_path):
    """Describe paired box changes; without GT these are not quality metrics."""
    baseline = json.loads(Path(baseline_path).read_text())["results"]
    refined = json.loads(Path(refined_path).read_text())["results"]
    shifts, size_changes, yaw_changes = [], [], []
    paired = 0
    for token, before_records in baseline.items():
        after_records = refined[token]
        if len(before_records) != len(after_records):
            raise ValueError(f"Detection count changed for {token}; source-index comparison is ambiguous")
        for before, after in zip(before_records, after_records):
            if before["detection_name"] != after["detection_name"]:
                raise ValueError(f"Detection order/class changed for {token}")
            paired += 1
            shifts.append(float(np.linalg.norm(np.asarray(after["translation"]) - before["translation"])))
            size_changes.append(float(np.linalg.norm(np.asarray(after["size"]) - before["size"])))
            def yaw(record):
                w, _, _, z = record["rotation"]
                return 2 * math.atan2(z, w)
            yaw_changes.append(abs(float((yaw(after) - yaw(before) + math.pi) % (2 * math.pi) - math.pi)))
    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        return {"mean": float(values.mean()) if len(values) else 0.0,
                "median": float(np.median(values)) if len(values) else 0.0,
                "max": float(values.max()) if len(values) else 0.0,
                "changed": int(np.count_nonzero(values > 1e-5))}
    return {"paired_detections": paired, "center_shift_m": stats(shifts),
            "size_l2_change_m": stats(size_changes), "yaw_change_rad": stats(yaw_changes),
            "quality_conclusion": "unavailable_without_ground_truth",
            "warning": "These values measure output changes, not accuracy improvement."}
