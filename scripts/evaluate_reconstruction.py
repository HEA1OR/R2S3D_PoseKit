#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from r2s3d_posekit.eval_utils import (
    apply_runtime_pose_override,
    bbox_center,
    bbox_iou_3d,
    compute_bounds,
    compute_iou,
    load_binary_mask,
    load_gt_visual_mesh,
    load_mesh,
    load_objects_spec,
    make_pose_matrix,
    render_mask_cpu,
    voxel_volume_iou,
    world_pose_to_camera_pose,
)


def observed_mask_path(camera_data_dir: Path, object_id: int) -> Path | None:
    path = camera_data_dir / f"object_{object_id}.png"
    return path if path.exists() else None


def evaluate_object(
    *,
    object_id: int,
    result_root: Path,
    gt_cfg: dict,
    gt_visual_mesh,
    camera_data_dir: Path | None,
    camera_frame: int,
    camera_extrinsics_type: str,
    camera_frame_alignment: str,
    volume_iou_resolution: int,
) -> dict:
    object_dir = result_root / f"object_{object_id}"
    pred_pose_path = object_dir / f"{object_id}_pose_world_scaled.npz"
    pred_mesh_path = object_dir / f"{object_id}.glb"
    if not pred_pose_path.exists():
        raise FileNotFoundError(f"Missing predicted world pose: {pred_pose_path}")
    if not pred_mesh_path.exists():
        raise FileNotFoundError(f"Missing predicted mesh: {pred_mesh_path}")

    pose_npz = np.load(pred_pose_path)
    pred_pose_world = pose_npz["pose_world"].astype(np.float64)
    pred_mesh_local = load_mesh(pred_mesh_path)

    gt_pose_world = make_pose_matrix(gt_cfg["pos_xyz"], gt_cfg["quat_wxyz"])

    pred_mesh_world = pred_mesh_local.copy()
    pred_mesh_world.apply_transform(pred_pose_world)
    gt_mesh_world = gt_visual_mesh.copy()
    gt_mesh_world.apply_transform(gt_pose_world)

    pred_bounds = compute_bounds(np.asarray(pred_mesh_world.vertices, dtype=np.float64))
    gt_bounds = compute_bounds(np.asarray(gt_mesh_world.vertices, dtype=np.float64))
    pred_bbox_center = bbox_center(pred_bounds)
    gt_bbox_center = bbox_center(gt_bounds)
    bbox_center_error_xyz = pred_bbox_center - gt_bbox_center
    bbox_center_error_l2 = float(np.linalg.norm(bbox_center_error_xyz))
    aabb_iou = float(bbox_iou_3d(gt_bounds, pred_bounds))
    volume_iou, voxel_pitch = voxel_volume_iou(
        gt_mesh_world,
        pred_mesh_world,
        resolution=volume_iou_resolution,
    )

    result = {
        "object_id": int(object_id),
        "name": gt_cfg["name"],
        "gt_mesh_path": gt_cfg["mesh_path"],
        "pred_mesh_path": str(pred_mesh_path),
        "gt_bbox_center_xyz": gt_bbox_center.tolist(),
        "pred_bbox_center_xyz": pred_bbox_center.tolist(),
        "bbox_center_error_xyz": bbox_center_error_xyz.tolist(),
        "bbox_center_error_l2_m": bbox_center_error_l2,
        "aabb_iou_3d": aabb_iou,
        "voxel_volume_iou_3d": float(volume_iou),
        "voxel_pitch_m": (float(voxel_pitch) if voxel_pitch is not None else None),
    }

    if camera_data_dir is not None:
        K_path = camera_data_dir / f"intrinsics_{camera_frame}.npy"
        E_path = camera_data_dir / f"extrinsics_{camera_frame}.npy"
        rgb_path = camera_data_dir / f"rgb_{camera_frame}.png"
        if K_path.exists() and E_path.exists() and rgb_path.exists():
            K = np.load(K_path).reshape(3, 3).astype(np.float64)
            world_from_camera = np.load(E_path).astype(np.float64)
            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if image is not None:
                H, W = image.shape[:2]
                gt_pose_camera = world_pose_to_camera_pose(
                    gt_pose_world,
                    camera_extrinsics=world_from_camera,
                    camera_extrinsics_type=camera_extrinsics_type,
                    camera_frame_alignment=camera_frame_alignment,
                )
                pred_pose_camera = (
                    pose_npz["pose_camera"].astype(np.float64)
                    if "pose_camera" in pose_npz.files
                    else world_pose_to_camera_pose(
                        pred_pose_world,
                        camera_extrinsics=world_from_camera,
                        camera_extrinsics_type=camera_extrinsics_type,
                        camera_frame_alignment=camera_frame_alignment,
                    )
                )
                gt_mask = render_mask_cpu(gt_visual_mesh, gt_pose_camera, K, H, W)
                pred_mask = render_mask_cpu(pred_mesh_local, pred_pose_camera, K, H, W)
                result["camera_view_mask_iou_gt_pred"] = compute_iou(gt_mask, pred_mask)

                obs_mask_file = observed_mask_path(camera_data_dir, object_id)
                if obs_mask_file is not None:
                    obs_mask = load_binary_mask(obs_mask_file)
                    result["camera_view_mask_iou_obs_pred"] = compute_iou(obs_mask, pred_mask)
                    result["camera_view_mask_iou_obs_gt"] = compute_iou(obs_mask, gt_mask)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate reconstructed meshes and world poses against GT meshes and GT world poses."
    )
    parser.add_argument("--result-root", type=Path, required=True, help="Directory containing object_*/ outputs.")
    parser.add_argument(
        "--gt-spec",
        type=Path,
        required=True,
        help="JSON file describing GT mesh path, scale, and world pose for each object.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Default: <result-root>/reconstruction_eval.json",
    )
    parser.add_argument(
        "--camera-data-dir",
        type=Path,
        default=None,
        help="Optional prepared evaluation camera data directory.",
    )
    parser.add_argument(
        "--gt-pose-json",
        type=Path,
        default=None,
        help="Optional runtime GT pose snapshot JSON used to override poses from --gt-spec.",
    )
    parser.add_argument("--camera-frame", type=int, default=0, help="Frame index used in the camera data files.")
    parser.add_argument(
        "--camera-extrinsics-type",
        type=str,
        choices=["world_from_camera", "camera_from_world"],
        default="world_from_camera",
        help="Interpretation of the saved extrinsics matrix.",
    )
    parser.add_argument(
        "--camera-frame-alignment",
        type=str,
        choices=["identity", "legacy_mobile_etot"],
        default="identity",
        help="Optional extra fixed camera-frame alignment used before evaluation rendering.",
    )
    parser.add_argument(
        "--volume-iou-resolution",
        type=int,
        default=72,
        help="Voxel resolution used for approximate mesh volume IoU.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_root = args.result_root.resolve()
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else (result_root / "reconstruction_eval.json").resolve()
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    gt_objects = load_objects_spec(args.gt_spec)
    gt_objects = apply_runtime_pose_override(gt_objects, args.gt_pose_json)

    camera_data_dir = (
        args.camera_data_dir.resolve()
        if args.camera_data_dir is not None and args.camera_data_dir.exists()
        else None
    )

    gt_mesh_cache = {}
    results = []
    for object_id, gt_cfg in sorted(gt_objects.items()):
        mesh_path = Path(gt_cfg["mesh_path"]).resolve()
        if mesh_path not in gt_mesh_cache:
            gt_mesh_cache[mesh_path] = load_gt_visual_mesh(mesh_path, gt_cfg["scale_xyz"])
        results.append(
            evaluate_object(
                object_id=object_id,
                result_root=result_root,
                gt_cfg=gt_cfg,
                gt_visual_mesh=gt_mesh_cache[mesh_path],
                camera_data_dir=camera_data_dir,
                camera_frame=args.camera_frame,
                camera_extrinsics_type=args.camera_extrinsics_type,
                camera_frame_alignment=args.camera_frame_alignment,
                volume_iou_resolution=args.volume_iou_resolution,
            )
        )

    summary = {
        "result_root": str(result_root),
        "gt_spec": str(args.gt_spec.resolve()),
        "camera_data_dir": (str(camera_data_dir) if camera_data_dir is not None else None),
        "object_count": len(results),
        "objects": results,
    }
    if results:
        summary["mean_bbox_center_error_l2_m"] = float(
            np.mean([item["bbox_center_error_l2_m"] for item in results])
        )
        summary["mean_aabb_iou_3d"] = float(np.mean([item["aabb_iou_3d"] for item in results]))
        summary["mean_voxel_volume_iou_3d"] = float(
            np.mean([item["voxel_volume_iou_3d"] for item in results])
        )
        mask_values = [
            item["camera_view_mask_iou_gt_pred"]
            for item in results
            if "camera_view_mask_iou_gt_pred" in item
        ]
        if mask_values:
            summary["mean_camera_view_mask_iou_gt_pred"] = float(np.mean(mask_values))

    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for item in results:
        mask_iou = item.get("camera_view_mask_iou_gt_pred")
        suffix = f" | Mask IoU={mask_iou:.4f}" if mask_iou is not None else ""
        print(
            f"object_{item['object_id']} {item['name']}: "
            f"bbox_center_err={item['bbox_center_error_l2_m']:.4f} m | "
            f"AABB IoU={item['aabb_iou_3d']:.4f} | "
            f"Volume IoU={item['voxel_volume_iou_3d']:.4f}"
            f"{suffix}"
        )

    if results:
        print(f"\nMean bbox-center pose error: {summary['mean_bbox_center_error_l2_m']:.4f} m")
        print(f"Mean AABB IoU: {summary['mean_aabb_iou_3d']:.4f}")
        print(f"Mean mesh volume IoU: {summary['mean_voxel_volume_iou_3d']:.4f}")
        if "mean_camera_view_mask_iou_gt_pred" in summary:
            print(f"Mean camera-view mask IoU: {summary['mean_camera_view_mask_iou_gt_pred']:.4f}")
    print(f"\nSaved JSON to: {output_json}")


if __name__ == "__main__":
    main()
