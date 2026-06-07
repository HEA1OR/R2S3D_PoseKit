#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from r2s3d_posekit.eval_utils import (
    apply_runtime_pose_override,
    compute_iou,
    load_binary_mask,
    load_gt_visual_mesh,
    load_mesh,
    load_objects_spec,
    make_pose_matrix,
    render_mask_cpu,
    world_pose_to_camera_pose,
)


def load_rgb(path: Path) -> np.ndarray:
    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Failed to load RGB image: {path}")
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def apply_transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    homo = np.concatenate([verts, np.ones((len(verts), 1), dtype=np.float64)], axis=1)
    return (transform @ homo.T).T[:, :3]


def project_points(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.clip(points_cam[:, 2], 1e-8, None)
    u = K[0, 0] * points_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def render_mask_gpu(
    *,
    mesh: trimesh.Trimesh,
    pose_camera: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    foundationpose_root: Path,
) -> np.ndarray:
    if str(foundationpose_root) not in sys.path:
        sys.path.insert(0, str(foundationpose_root))
    import torch
    import nvdiffrast.torch as dr
    from Utils import make_mesh_tensors, nvdiffrast_render  # type: ignore

    mesh_tensors = make_mesh_tensors(mesh, device="cuda")
    ob_in_cam = torch.as_tensor(pose_camera[None], device="cuda", dtype=torch.float32)
    glctx = dr.RasterizeCudaContext()
    _, depth, _ = nvdiffrast_render(
        K=np.asarray(K, dtype=np.float32),
        H=int(H),
        W=int(W),
        ob_in_cams=ob_in_cam,
        glctx=glctx,
        mesh_tensors=mesh_tensors,
    )
    return (depth[0].detach().float().cpu().numpy() > 1e-6).astype(np.uint8)


def render_mask_auto(
    *,
    mesh: trimesh.Trimesh,
    pose_camera: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    render_backend: str,
    foundationpose_root: Path | None,
) -> tuple[np.ndarray, str]:
    if render_backend == "cpu":
        return render_mask_cpu(mesh, pose_camera, K, H, W), "cpu_project_fill"
    if foundationpose_root is None:
        return render_mask_cpu(mesh, pose_camera, K, H, W), "cpu_project_fill"
    try:
        return (
            render_mask_gpu(
                mesh=mesh,
                pose_camera=pose_camera,
                K=K,
                H=H,
                W=W,
                foundationpose_root=foundationpose_root,
            ),
            "gpu_nvdiffrast",
        )
    except Exception as exc:
        print(
            f"Warning: GPU render unavailable ({type(exc).__name__}: {exc}). "
            "Falling back to CPU projection fill."
        )
        return render_mask_cpu(mesh, pose_camera, K, H, W), f"cpu_fallback:{type(exc).__name__}"


def make_overlap_mask_image(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    gt_only = (gt_mask > 0) & ~(pred_mask > 0)
    pred_only = (pred_mask > 0) & ~(gt_mask > 0)
    overlap = (gt_mask > 0) & (pred_mask > 0)
    out[gt_only] = (0, 255, 0)
    out[pred_only] = (255, 0, 0)
    out[overlap] = (255, 255, 0)
    return out


def blend_masks_on_rgb(rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = make_overlap_mask_image(gt_mask, pred_mask)
    out = rgb.astype(np.float32).copy()
    active = (gt_mask > 0) | (pred_mask > 0)
    out[active] = (1.0 - alpha) * out[active] + alpha * color[active].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_mask_contours(rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gt_contours, _ = cv2.findContours((gt_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours((pred_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, gt_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(canvas, pred_contours, -1, (0, 0, 255), 2)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def make_labeled_panel(images_rgb, labels):
    panels = []
    for image_rgb, label in zip(images_rgb, labels):
        canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            canvas,
            label,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    return np.concatenate(panels, axis=1)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render GT and predicted meshes into the camera view for qualitative overlap comparison."
    )
    parser.add_argument("--result-root", type=Path, required=True, help="Directory containing object_*/ outputs.")
    parser.add_argument("--gt-spec", type=Path, required=True, help="GT object spec JSON.")
    parser.add_argument("--camera-data-dir", type=Path, required=True, help="Prepared evaluation camera data directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <result-root>/camera_view_mesh_compare")
    parser.add_argument("--gt-pose-json", type=Path, default=None, help="Optional runtime GT pose snapshot override.")
    parser.add_argument("--camera-frame", type=int, default=0, help="Frame index used in the camera data files.")
    parser.add_argument(
        "--camera-extrinsics-type",
        type=str,
        choices=["world_from_camera", "camera_from_world"],
        default="world_from_camera",
    )
    parser.add_argument(
        "--camera-frame-alignment",
        type=str,
        choices=["identity", "legacy_mobile_etot"],
        default="identity",
    )
    parser.add_argument(
        "--render-backend",
        type=str,
        choices=["auto", "cpu"],
        default="auto",
        help="Use CPU by default, or try FoundationPose+nvdiffrast first when set to auto.",
    )
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=None,
        help="Installed FoundationPose root used only for optional GPU rendering.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_root = args.result_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (result_root / "camera_view_mesh_compare").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_data_dir = args.camera_data_dir.resolve()

    rgb = load_rgb(camera_data_dir / f"rgb_{args.camera_frame}.png")
    H, W = rgb.shape[:2]
    K = np.load(camera_data_dir / f"intrinsics_{args.camera_frame}.npy").reshape(3, 3).astype(np.float64)
    world_from_camera = np.load(camera_data_dir / f"extrinsics_{args.camera_frame}.npy").astype(np.float64)

    gt_objects = load_objects_spec(args.gt_spec)
    gt_objects = apply_runtime_pose_override(gt_objects, args.gt_pose_json)

    gt_mesh_cache = {}
    summary = {
        "result_root": str(result_root),
        "camera_data_dir": str(camera_data_dir),
        "gt_spec": str(args.gt_spec.resolve()),
        "objects": [],
    }

    render_backend_used = None
    for object_id, gt_cfg in sorted(gt_objects.items()):
        object_dir = output_dir / f"object_{object_id}"
        object_dir.mkdir(parents=True, exist_ok=True)

        pred_pose_npz = np.load(result_root / f"object_{object_id}" / f"{object_id}_pose_world_scaled.npz")
        pred_pose_world = pred_pose_npz["pose_world"].astype(np.float64)
        pred_pose_camera = (
            pred_pose_npz["pose_camera"].astype(np.float64)
            if "pose_camera" in pred_pose_npz.files
            else world_pose_to_camera_pose(
                pred_pose_world,
                camera_extrinsics=world_from_camera,
                camera_extrinsics_type=args.camera_extrinsics_type,
                camera_frame_alignment=args.camera_frame_alignment,
            )
        )

        gt_pose_world = make_pose_matrix(gt_cfg["pos_xyz"], gt_cfg["quat_wxyz"])
        gt_pose_camera = world_pose_to_camera_pose(
            gt_pose_world,
            camera_extrinsics=world_from_camera,
            camera_extrinsics_type=args.camera_extrinsics_type,
            camera_frame_alignment=args.camera_frame_alignment,
        )

        mesh_path = Path(gt_cfg["mesh_path"]).resolve()
        if mesh_path not in gt_mesh_cache:
            gt_mesh_cache[mesh_path] = load_gt_visual_mesh(mesh_path, gt_cfg["scale_xyz"])
        gt_mesh = gt_mesh_cache[mesh_path].copy()
        pred_mesh = load_mesh(result_root / f"object_{object_id}" / f"{object_id}.glb")

        gt_mask, backend_gt = render_mask_auto(
            mesh=gt_mesh,
            pose_camera=gt_pose_camera,
            K=K,
            H=H,
            W=W,
            render_backend=args.render_backend,
            foundationpose_root=(args.foundationpose_root.resolve() if args.foundationpose_root else None),
        )
        pred_mask, backend_pred = render_mask_auto(
            mesh=pred_mesh,
            pose_camera=pred_pose_camera,
            K=K,
            H=H,
            W=W,
            render_backend=args.render_backend,
            foundationpose_root=(args.foundationpose_root.resolve() if args.foundationpose_root else None),
        )
        render_backend_used = backend_pred if render_backend_used is None else render_backend_used

        obs_mask_path = camera_data_dir / f"object_{object_id}.png"
        obs_mask = load_binary_mask(obs_mask_path) if obs_mask_path.exists() else None

        overlap_rgb = make_overlap_mask_image(gt_mask, pred_mask)
        blended_rgb = blend_masks_on_rgb(rgb, gt_mask, pred_mask)
        contour_rgb = draw_mask_contours(rgb, gt_mask, pred_mask)

        gt_mask_rgb = np.zeros_like(rgb)
        gt_mask_rgb[gt_mask > 0] = (0, 255, 0)
        pred_mask_rgb = np.zeros_like(rgb)
        pred_mask_rgb[pred_mask > 0] = (255, 0, 0)

        panel_rgb = make_labeled_panel(
            [rgb, gt_mask_rgb, pred_mask_rgb, blended_rgb, contour_rgb, overlap_rgb],
            ["RGB", "GT Mask", "Pred Mask", "Blend", "Contours", "Overlap"],
        )

        save_rgb(object_dir / "rgb.png", rgb)
        save_rgb(object_dir / "gt_mask.png", gt_mask_rgb)
        save_rgb(object_dir / "pred_mask.png", pred_mask_rgb)
        save_rgb(object_dir / "overlap_mask.png", overlap_rgb)
        save_rgb(object_dir / "blend_on_rgb.png", blended_rgb)
        save_rgb(object_dir / "contours_on_rgb.png", contour_rgb)
        save_rgb(object_dir / "comparison_panel.png", panel_rgb)

        summary["objects"].append(
            {
                "object_id": int(object_id),
                "name": gt_cfg["name"],
                "output_dir": str(object_dir),
                "render_backend_gt": backend_gt,
                "render_backend_pred": backend_pred,
                "mask_iou_gt_pred": float(compute_iou(gt_mask, pred_mask)),
                "mask_iou_obs_pred": (float(compute_iou(obs_mask, pred_mask)) if obs_mask is not None else None),
                "mask_iou_obs_gt": (float(compute_iou(obs_mask, gt_mask)) if obs_mask is not None else None),
            }
        )
        print(f"object_{object_id} {gt_cfg['name']}: saved camera-view comparison to {object_dir}")

    if render_backend_used is not None:
        summary["render_backend"] = render_backend_used
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
