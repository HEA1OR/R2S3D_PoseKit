from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def load_objects_spec(spec_json: Path) -> Dict[int, Dict]:
    payload = json.loads(Path(spec_json).read_text(encoding="utf-8"))
    entries = payload.get("objects", payload)
    objects: Dict[int, Dict] = {}

    if isinstance(entries, dict):
        items = entries.items()
    else:
        items = ((entry["object_id"], entry) for entry in entries)

    for key, entry in items:
        object_id = int(key)
        data = dict(entry)
        data["object_id"] = object_id
        if "mesh_path" not in data:
            raise ValueError(f"GT spec entry {object_id} is missing 'mesh_path'")
        data.setdefault("name", f"object_{object_id}")
        data.setdefault("scale_xyz", [1.0, 1.0, 1.0])
        data.setdefault("pos_xyz", [0.0, 0.0, 0.0])
        data.setdefault("quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        objects[object_id] = data
    return objects


def apply_runtime_pose_override(objects: Dict[int, Dict], gt_pose_json: Optional[Path]) -> Dict[int, Dict]:
    if gt_pose_json is None:
        return objects
    payload = json.loads(Path(gt_pose_json).read_text(encoding="utf-8"))
    updated = {int(k): dict(v) for k, v in objects.items()}
    for entry in payload.get("objects", []):
        object_id = int(entry["object_id"])
        if object_id not in updated:
            continue
        if "position_world_abs_xyz" in entry:
            updated[object_id]["pos_xyz"] = list(entry["position_world_abs_xyz"])
        if "quat_wxyz" in entry:
            updated[object_id]["quat_wxyz"] = list(entry["quat_wxyz"])
    return updated


def wxyz_to_matrix(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )
    return Rotation.from_quat(quat_xyzw).as_matrix()


def make_pose_matrix(pos_xyz, quat_wxyz):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = wxyz_to_matrix(quat_wxyz)
    pose[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    return pose


def scale_matrix(scale_xyz):
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0], mat[1, 1], mat[2, 2] = [float(v) for v in scale_xyz]
    return mat


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    homo = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (transform @ homo.T).T[:, :3]


def triangulate_faces(face_counts, face_indices):
    faces = []
    cursor = 0
    for count in face_counts:
        if count < 3:
            cursor += count
            continue
        poly = face_indices[cursor : cursor + count]
        for idx in range(1, count - 1):
            faces.append([poly[0], poly[idx], poly[idx + 1]])
        cursor += count
    return np.asarray(faces, dtype=np.int64)


def usd_matrix_to_numpy(matrix):
    arr = np.array(matrix, dtype=np.float64)
    return arr.reshape(4, 4).T


def load_visual_mesh_from_usd(usd_path):
    try:
        from pxr import Usd, UsdGeom
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading USD ground-truth meshes requires the 'pxr' module. "
            "Install an environment with usd-core/pxr, or run evaluation and rendering "
            "with a Python environment that already provides 'pxr'."
        ) from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    xform_cache = UsdGeom.XformCache()
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path_lower = str(prim.GetPath()).lower()
        if "visual" not in path_lower:
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or face_counts is None or face_indices is None:
            continue

        points = np.asarray(points, dtype=np.float64)
        faces = triangulate_faces(np.asarray(face_counts), np.asarray(face_indices))
        if len(points) == 0 or len(faces) == 0:
            continue

        local_to_world = usd_matrix_to_numpy(xform_cache.GetLocalToWorldTransform(prim))
        points = transform_points(points, local_to_world)

        all_vertices.append(points)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(points)

    if not all_vertices:
        raise RuntimeError(f"No visual meshes found in USD: {usd_path}")

    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def load_mesh(mesh_path):
    mesh_path = Path(mesh_path)
    if mesh_path.suffix.lower() == ".usd":
        return load_visual_mesh_from_usd(mesh_path)
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def load_gt_visual_mesh(mesh_path, scale_xyz):
    mesh = load_mesh(mesh_path).copy()
    mesh.apply_transform(scale_matrix(scale_xyz))
    return mesh


def compute_bounds(points):
    points = np.asarray(points, dtype=np.float64)
    return np.stack([points.min(axis=0), points.max(axis=0)], axis=0)


def bbox_center(bounds):
    bounds = np.asarray(bounds, dtype=np.float64)
    return 0.5 * (bounds[0] + bounds[1])


def bbox_iou_3d(bounds_a, bounds_b):
    inter_min = np.maximum(bounds_a[0], bounds_b[0])
    inter_max = np.minimum(bounds_a[1], bounds_b[1])
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_size))
    vol_a = float(np.prod(np.maximum(bounds_a[1] - bounds_a[0], 0.0)))
    vol_b = float(np.prod(np.maximum(bounds_b[1] - bounds_b[0], 0.0)))
    union = vol_a + vol_b - inter_vol
    if union <= 0:
        return 0.0
    return inter_vol / union


def camera_frame_alignment_tf(mode: str) -> np.ndarray:
    mode = str(mode)
    if mode in {"identity", "none"}:
        return np.eye(4, dtype=np.float64)
    if mode == "legacy_mobile_etot":
        return np.diag([-1.0, 1.0, -1.0, 1.0]).astype(np.float64)
    raise ValueError(f"Unsupported camera_frame_alignment mode: {mode}")


def world_pose_to_camera_pose(
    pose_world: np.ndarray,
    camera_extrinsics: np.ndarray,
    camera_extrinsics_type: str = "world_from_camera",
    camera_frame_alignment: str = "identity",
) -> np.ndarray:
    align_tf = camera_frame_alignment_tf(camera_frame_alignment)
    if camera_extrinsics_type == "world_from_camera":
        return align_tf @ np.linalg.inv(camera_extrinsics) @ pose_world
    if camera_extrinsics_type == "camera_from_world":
        return align_tf @ camera_extrinsics @ pose_world
    raise ValueError(f"Unsupported camera_extrinsics_type: {camera_extrinsics_type}")


def apply_transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    homo = np.concatenate([verts, np.ones((len(verts), 1), dtype=np.float64)], axis=1)
    return (transform @ homo.T).T[:, :3]


def project_points(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.clip(points_cam[:, 2], 1e-8, None)
    u = K[0, 0] * points_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def render_mask_cpu(mesh: trimesh.Trimesh, pose_camera: np.ndarray, K: np.ndarray, H: int, W: int) -> np.ndarray:
    verts_cam = apply_transform(np.asarray(mesh.vertices), pose_camera)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri_cam = verts_cam[faces]
    valid_depth = np.all(tri_cam[:, :, 2] > 1e-5, axis=1)
    tri_cam = tri_cam[valid_depth]
    if len(tri_cam) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    face_normals = np.cross(tri_cam[:, 1] - tri_cam[:, 0], tri_cam[:, 2] - tri_cam[:, 0])
    face_centers = tri_cam.mean(axis=1)
    facing_score = np.sum(face_normals * face_centers, axis=1)
    front_a = facing_score < 0
    front_b = facing_score > 0
    tri_cam = tri_cam[front_a if front_a.sum() >= front_b.sum() else front_b]
    if len(tri_cam) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    tri_uv = np.stack([project_points(tri, K) for tri in tri_cam], axis=0)
    mask = np.zeros((H, W), dtype=np.uint8)
    for tri in tri_uv:
        if not np.isfinite(tri).all():
            continue
        poly = np.round(tri).astype(np.int32)
        if poly[:, 0].max() < 0 or poly[:, 0].min() >= W or poly[:, 1].max() < 0 or poly[:, 1].min() >= H:
            continue
        cv2.fillConvexPoly(mask, poly, 255)
    return (mask > 0).astype(np.uint8)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a > 0, mask_b > 0).sum())
    union = float(np.logical_or(mask_a > 0, mask_b > 0).sum())
    if union <= 0:
        return 0.0
    return inter / union


def load_binary_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to load mask: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return (image > 0).astype(np.uint8)


def voxel_index_array(mesh: trimesh.Trimesh, pitch: float) -> np.ndarray:
    voxels = mesh.voxelized(pitch)
    try:
        voxels = voxels.fill()
    except Exception:
        pass
    points = np.asarray(voxels.points, dtype=np.float64)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.rint(points / pitch).astype(np.int64)


def voxel_volume_iou(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, resolution: int = 72):
    union_bounds = np.stack(
        [
            np.minimum(mesh_a.bounds[0], mesh_b.bounds[0]),
            np.maximum(mesh_a.bounds[1], mesh_b.bounds[1]),
        ],
        axis=0,
    )
    union_extent = union_bounds[1] - union_bounds[0]
    max_extent = float(np.max(union_extent))
    if max_extent <= 1e-8:
        return 0.0, None

    pitch = max(max_extent / float(resolution), 1e-5)
    indices_a = voxel_index_array(mesh_a, pitch)
    indices_b = voxel_index_array(mesh_b, pitch)
    if len(indices_a) == 0 and len(indices_b) == 0:
        return 0.0, pitch

    dtype = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    flat_a = np.ascontiguousarray(indices_a).view(dtype).reshape(-1)
    flat_b = np.ascontiguousarray(indices_b).view(dtype).reshape(-1)
    inter = np.intersect1d(flat_a, flat_b).size
    union = np.union1d(flat_a, flat_b).size
    if union == 0:
        return 0.0, pitch
    return float(inter) / float(union), pitch
