#!/usr/bin/env python3

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh


SUPPORTED_MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}


class MeshSizeAdjuster:
    DEFAULT_K = np.array(
        [
            [906.461181640625, 0, 635.8511962890625],
            [0, 905.659912109375, 350.6916809082031],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        dataset_path: str,
        mesh_path: Optional[str] = None,
        frame_name: Optional[str] = None,
        depth_scale: float = 1000.0,
        opt_iters: int = 5,
        pose_refine_iter: int = 3,
        track_refine_iter: int = 4,
        scale_step_cap: float = 1.1,
        scale_exponent: float = 0.5,
        projection_min_pixels: float = 8.0,
        iou_stop: float = 0.98,
        max_work_mesh_faces: int = 100000,
        upright_lock_after_iter: int = 2,
        upright_lock_angle_deg: float = 1.0,
        camera_extrinsics_path: Optional[str] = None,
        camera_extrinsics_type: str = "world_from_camera",
        camera_frame_alignment: str = "identity",
    ):
        """
        Support both:
        1. Legacy structure:
           mesh/, mask/, raw_depth/, raw_rgb/
        2. FoundationPose demo structure:
           mesh/, masks/, depth/, rgb/, cam_K.txt
        """
        self.dataset_path = Path(dataset_path).resolve()
        self.code_dir = Path(__file__).resolve().parent
        self.mesh_path = Path(mesh_path).resolve() if mesh_path else None
        self.frame_name = str(frame_name) if frame_name is not None else None
        self.depth_scale = float(depth_scale)
        self.opt_iters = int(opt_iters)
        self.pose_refine_iter = int(pose_refine_iter)
        self.track_refine_iter = int(track_refine_iter)
        self.scale_step_cap = float(scale_step_cap)
        self.scale_exponent = float(scale_exponent)
        self.projection_min_pixels = float(projection_min_pixels)
        self.iou_stop = float(iou_stop)
        self.max_work_mesh_faces = int(max_work_mesh_faces)
        self.upright_lock_after_iter = int(upright_lock_after_iter)
        self.upright_lock_angle_deg = float(upright_lock_angle_deg)
        self.camera_extrinsics_path = (
            Path(camera_extrinsics_path).resolve() if camera_extrinsics_path else None
        )
        self.camera_extrinsics_type = str(camera_extrinsics_type)
        self.camera_frame_alignment = str(camera_frame_alignment)

        self.mesh_dir = self._resolve_optional_dir("mesh")
        self.mask_dir = self._resolve_required_dir("mask", "masks")
        self.depth_dir = self._resolve_required_dir("raw_depth", "depth")
        self.rgb_dir = self._resolve_optional_dir("raw_rgb", "rgb")
        self.K = self._load_camera_matrix()
        self.camera_extrinsics = self._load_camera_extrinsics()

        self._fp_ready = False
        self._fp_torch = None
        self._FoundationPose = None
        self._ScorePredictor = None
        self._PoseRefinePredictor = None
        self._make_mesh_tensors = None
        self._nvdiffrast_render = None
        self._draw_posed_3d_box = None
        self._draw_xyz_axis = None
        self._dr = None
        self._scorer = None
        self._refiner = None
        self._glctx = None
        self._estimator = None
        self._default_rot_grid = None

    def _sync_cuda(self) -> None:
        if self._fp_torch is None:
            return
        try:
            if self._fp_torch.cuda.is_available():
                self._fp_torch.cuda.synchronize()
        except Exception:
            pass

    def _start_timer(self) -> float:
        self._sync_cuda()
        return time.perf_counter()

    def _finish_timer(self, start_time: float) -> float:
        self._sync_cuda()
        return float(time.perf_counter() - start_time)

    def _resolve_optional_dir(self, *candidates: str) -> Optional[Path]:
        for name in candidates:
            path = self.dataset_path / name
            if path.exists():
                return path
        return None

    def _resolve_required_dir(self, *candidates: str) -> Path:
        path = self._resolve_optional_dir(*candidates)
        if path is None:
            joined = ", ".join(str(self.dataset_path / name) for name in candidates)
            raise FileNotFoundError(f"Required directory not found. Tried: {joined}")
        return path

    def _load_camera_matrix(self) -> np.ndarray:
        cam_k_file = self.dataset_path / "cam_K.txt"
        if cam_k_file.exists():
            return np.loadtxt(cam_k_file).reshape(3, 3).astype(np.float64)
        return self.DEFAULT_K.copy()

    def _load_camera_extrinsics(self) -> Optional[np.ndarray]:
        if self.camera_extrinsics_path is None:
            return None
        if not self.camera_extrinsics_path.exists():
            raise FileNotFoundError(
                f"Camera extrinsics file not found: {self.camera_extrinsics_path}"
            )
        extrinsics = np.load(self.camera_extrinsics_path)
        extrinsics = np.asarray(extrinsics, dtype=np.float64)
        if extrinsics.shape == (3, 4):
            extrinsics = np.vstack([extrinsics, np.array([0.0, 0.0, 0.0, 1.0])])
        if extrinsics.shape != (4, 4):
            raise ValueError(
                f"Camera extrinsics must have shape (4,4) or (3,4), got {extrinsics.shape}"
            )
        return extrinsics

    def _list_image_files(self, directory: Path) -> Dict[str, Path]:
        files: Dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                files[path.stem] = path
        return files

    def _collect_mesh_files(self) -> List[Path]:
        source = self.mesh_path if self.mesh_path is not None else self.mesh_dir
        if source is None:
            raise FileNotFoundError(
                "No mesh source provided. Pass --mesh_path or create dataset_path/mesh."
            )

        if source.is_file():
            if source.suffix.lower() not in SUPPORTED_MESH_EXTENSIONS:
                raise ValueError(f"Unsupported mesh format: {source}")
            return [source]

        if source.is_dir():
            mesh_files = [
                path
                for path in sorted(source.iterdir())
                if path.is_file() and path.suffix.lower() in SUPPORTED_MESH_EXTENSIONS
            ]
            if not mesh_files:
                raise FileNotFoundError(f"No supported mesh files found in: {source}")
            return mesh_files

        raise FileNotFoundError(f"Mesh source not found: {source}")

    def _resolve_observation_files(self, mesh_stem: str) -> Tuple[str, Path, Path, Path]:
        mask_files = self._list_image_files(self.mask_dir)
        depth_files = self._list_image_files(self.depth_dir)
        rgb_files = self._list_image_files(self.rgb_dir) if self.rgb_dir else {}

        common_stems = sorted(set(mask_files) & set(depth_files))
        if self.rgb_dir is None:
            raise FileNotFoundError(
                "RGB directory is required for iterative FoundationPose optimization."
            )
        common_stems = sorted(set(common_stems) & set(rgb_files))
        if not common_stems:
            raise FileNotFoundError(
                "No matching rgb/mask/depth frame triplets found in the scene directory."
            )

        candidates: List[str] = []
        if self.frame_name is not None:
            candidates.append(self.frame_name)
        candidates.append(mesh_stem)
        if len(common_stems) == 1:
            candidates.append(common_stems[0])
        candidates.extend(common_stems)

        tried = []
        for stem in candidates:
            if stem in tried:
                continue
            tried.append(stem)
            if stem in rgb_files and stem in mask_files and stem in depth_files:
                return stem, rgb_files[stem], mask_files[stem], depth_files[stem]

        raise FileNotFoundError(
            "Unable to match mesh to an rgb/mask/depth frame. "
            f"Mesh stem: {mesh_stem}, frame_name: {self.frame_name}, available frames: {common_stems}"
        )

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        if mask.ndim == 3:
            best_channel = np.argmax([mask[..., i].sum() for i in range(mask.shape[2])])
            mask = mask[..., best_channel]
        return mask

    def _load_rgb(self, rgb_path: Path) -> np.ndarray:
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Failed to read RGB image: {rgb_path}")
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    def _load_depth_in_meters(self, depth_path: Path) -> np.ndarray:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to read depth: {depth_path}")
        depth = depth.astype(np.float32)
        if np.nanmax(depth) > 100:
            depth /= self.depth_scale
        return depth

    def _load_observation(self, mesh_stem: str) -> Dict:
        frame_name, rgb_path, mask_path, depth_path = self._resolve_observation_files(mesh_stem)
        rgb = self._load_rgb(rgb_path)
        mask = (self._load_mask(mask_path) > 127).astype(np.uint8)
        depth = self._load_depth_in_meters(depth_path)
        if rgb.shape[:2] != mask.shape[:2] or rgb.shape[:2] != depth.shape[:2]:
            raise ValueError(
                f"RGB/mask/depth shape mismatch for frame {frame_name}: "
                f"rgb={rgb.shape[:2]}, mask={mask.shape[:2]}, depth={depth.shape[:2]}"
            )
        return {
            "frame_name": frame_name,
            "rgb_path": rgb_path,
            "mask_path": mask_path,
            "depth_path": depth_path,
            "rgb": rgb,
            "mask": mask,
            "depth": depth,
            "K": self.K.copy(),
            "H": int(mask.shape[0]),
            "W": int(mask.shape[1]),
        }

    def _load_trimesh(self, mesh_path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Unsupported mesh type for {mesh_path}: {type(mesh)}")
        if len(mesh.vertices) == 0:
            raise ValueError(f"Mesh has no vertices: {mesh_path}")
        return mesh

    def _ensure_foundationpose(self, debug_dir: Path) -> None:
        if self._fp_ready:
            return

        import torch
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
        from Utils import (
            bilateral_filter_depth,
            depth2xyzmap,
            draw_posed_3d_box,
            draw_xyz_axis,
            erode_depth,
            make_mesh_tensors,
            nvdiffrast_render,
            set_logging_format,
            set_seed,
        )
        import nvdiffrast.torch as dr

        set_logging_format()
        set_seed(0)

        self._fp_torch = torch
        self._FoundationPose = FoundationPose
        self._PoseRefinePredictor = PoseRefinePredictor
        self._ScorePredictor = ScorePredictor
        self._bilateral_filter_depth = bilateral_filter_depth
        self._depth2xyzmap = depth2xyzmap
        self._make_mesh_tensors = make_mesh_tensors
        self._nvdiffrast_render = nvdiffrast_render
        self._draw_posed_3d_box = draw_posed_3d_box
        self._draw_xyz_axis = draw_xyz_axis
        self._dr = dr
        self._erode_depth = erode_depth
        self._scorer = self._ScorePredictor()
        self._refiner = self._PoseRefinePredictor()
        self._glctx = self._dr.RasterizeCudaContext()
        self._fp_ready = True
        debug_dir.mkdir(parents=True, exist_ok=True)

    def load_depth_from_mask(self, mask: np.ndarray, depth: np.ndarray) -> float:
        valid_mask = (mask > 0) & np.isfinite(depth) & (depth > 1e-6)
        valid_count = int(valid_mask.sum())
        if valid_count < 30:
            raise ValueError(f"Too few valid depth pixels inside mask: {valid_count}")
        return float(np.median(depth[valid_mask]))

    def _compute_obb_frame(self, mesh: trimesh.Trimesh) -> Dict:
        mesh_to_obb, extents = trimesh.bounds.oriented_bounds(mesh)
        obb_to_mesh = np.linalg.inv(mesh_to_obb)
        return {
            "mesh_to_obb": mesh_to_obb.astype(np.float64),
            "obb_to_mesh": obb_to_mesh.astype(np.float64),
            "extents": np.asarray(extents, dtype=np.float64),
            "center_obj": obb_to_mesh[:3, 3].astype(np.float64),
            "axes_obj": obb_to_mesh[:3, :3].astype(np.float64),
        }

    def _apply_obb_scaling(
        self, base_mesh: trimesh.Trimesh, obb_frame: Dict, scale_xyz: np.ndarray
    ) -> trimesh.Trimesh:
        mesh_to_obb = obb_frame["mesh_to_obb"]
        obb_to_mesh = obb_frame["obb_to_mesh"]
        verts = np.asarray(base_mesh.vertices, dtype=np.float64)
        verts_obb = verts @ mesh_to_obb[:3, :3].T + mesh_to_obb[:3, 3]
        verts_obb *= scale_xyz.reshape(1, 3)
        verts_scaled = verts_obb @ obb_to_mesh[:3, :3].T + obb_to_mesh[:3, 3]
        mesh = base_mesh.copy()
        mesh.vertices = verts_scaled.astype(np.float32)
        return mesh

    def _mask_bbox_size(self, mask: np.ndarray) -> Tuple[int, int]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in mask.")
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return int(w), int(h)

    def _build_work_mesh(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Dict]:
        info = {
            "work_mesh_simplified": False,
            "work_mesh_vertices": int(len(mesh.vertices)),
            "work_mesh_faces": int(len(mesh.faces)),
        }
        if len(mesh.faces) <= self.max_work_mesh_faces:
            return mesh.copy(), info

        if not isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
            print(
                "  Warning: large mesh uses non-ColorVisuals; skipping simplification to avoid losing texture."
            )
            return mesh.copy(), info

        import open3d as o3d

        tri = o3d.geometry.TriangleMesh()
        tri.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
        tri.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
        vertex_colors = np.asarray(mesh.visual.vertex_colors)[..., :3].astype(np.float64) / 255.0
        tri.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        simple = tri.simplify_quadric_decimation(
            target_number_of_triangles=self.max_work_mesh_faces
        )
        simple.remove_degenerate_triangles()
        simple.remove_duplicated_triangles()
        simple.remove_duplicated_vertices()
        simple.remove_non_manifold_edges()
        simple.compute_vertex_normals()

        work_mesh = trimesh.Trimesh(
            vertices=np.asarray(simple.vertices),
            faces=np.asarray(simple.triangles),
            process=False,
        )
        simple_colors = np.asarray(simple.vertex_colors)
        if len(simple_colors) == len(work_mesh.vertices):
            color_rgba = np.concatenate(
                [
                    np.clip(simple_colors * 255.0, 0, 255).astype(np.uint8),
                    np.full((len(simple_colors), 1), 255, dtype=np.uint8),
                ],
                axis=1,
            )
            work_mesh.visual.vertex_colors = color_rgba

        info = {
            "work_mesh_simplified": True,
            "work_mesh_vertices": int(len(work_mesh.vertices)),
            "work_mesh_faces": int(len(work_mesh.faces)),
            "original_mesh_vertices": int(len(mesh.vertices)),
            "original_mesh_faces": int(len(mesh.faces)),
        }
        print(
            f"  Simplified work mesh for estimation: "
            f"{len(mesh.vertices)}v/{len(mesh.faces)}f -> "
            f"{len(work_mesh.vertices)}v/{len(work_mesh.faces)}f"
        )
        return work_mesh, info

    def calculate_scale_from_depth(self, mesh: trimesh.Trimesh, observation: Dict) -> Tuple[float, Dict]:
        real_depth = self.load_depth_from_mask(observation["mask"], observation["depth"])
        _, extents = trimesh.bounds.oriented_bounds(mesh)
        mesh_max_size = float(np.max(extents))
        if mesh_max_size <= 0:
            raise ValueError(f"Invalid mesh extent: {extents}")

        width_px, height_px = self._mask_bbox_size(observation["mask"])
        fx = float(observation["K"][0, 0])
        fy = float(observation["K"][1, 1])
        real_width = (width_px * real_depth) / fx
        real_height = (height_px * real_depth) / fy
        real_max_size = max(real_width, real_height)
        if real_max_size <= 0:
            raise ValueError(f"Estimated real size is invalid: {real_max_size}")

        scale = real_max_size / mesh_max_size
        info = {
            "observation_frame": observation["frame_name"],
            "rgb_file": str(observation["rgb_path"]),
            "mask_file": str(observation["mask_path"]),
            "depth_file": str(observation["depth_path"]),
            "real_depth_m": float(real_depth),
            "mesh_original_size_m": float(mesh_max_size),
            "mask_bbox_pixels": [int(width_px), int(height_px)],
            "estimated_real_size_m": float(real_max_size),
            "calculated_scale": float(scale),
            "real_width_m": float(real_width),
            "real_height_m": float(real_height),
        }
        return float(scale), info

    def _camera_pose_to_centered_pose(self, pose_camera: np.ndarray) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("Estimator is not initialized.")
        tf_to_center = (
            self._estimator.get_tf_to_centered_mesh().detach().cpu().numpy().astype(np.float64)
        )
        return pose_camera.astype(np.float64) @ np.linalg.inv(tf_to_center)

    def _camera_frame_alignment_tf(self) -> np.ndarray:
        if self.camera_frame_alignment in {"identity", "none"}:
            return np.eye(4, dtype=np.float64)
        if self.camera_frame_alignment == "legacy_mobile_etot":
            return np.diag([-1.0, 1.0, -1.0, 1.0]).astype(np.float64)
        raise ValueError(
            f"Unsupported camera_frame_alignment: {self.camera_frame_alignment}"
        )

    def _camera_rotation_world_to_camera(self) -> Optional[np.ndarray]:
        if self.camera_extrinsics is None:
            return None
        camera_frame_align = self._camera_frame_alignment_tf()[:3, :3]
        if self.camera_extrinsics_type == "world_from_camera":
            return camera_frame_align @ np.linalg.inv(self.camera_extrinsics[:3, :3])
        if self.camera_extrinsics_type == "camera_from_world":
            return camera_frame_align @ self.camera_extrinsics[:3, :3].astype(np.float64)
        raise ValueError(
            f"Unsupported camera_extrinsics_type: {self.camera_extrinsics_type}"
        )

    def _build_upright_locked_rot_grid(self, upright_lock: Dict, obb_frame: Dict) -> np.ndarray:
        camera_R_world = self._camera_rotation_world_to_camera()
        if camera_R_world is None:
            raise RuntimeError("Camera extrinsics are required for upright-lock rot-grid building.")

        axis_obj = (
            obb_frame["axes_obj"][:, upright_lock["axis"]].astype(np.float64)
            * float(upright_lock["axis_sign"])
        )
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        align_world = self._rotation_from_vectors(axis_obj, world_z)

        rot_grid = []
        for yaw_deg in np.arange(0.0, 360.0, 30.0):
            yaw = np.deg2rad(yaw_deg)
            c = float(np.cos(yaw))
            s = float(np.sin(yaw))
            yaw_world = np.array(
                [
                    [c, -s, 0.0],
                    [s, c, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            world_R_object = yaw_world @ align_world
            camera_R_object = camera_R_world @ world_R_object
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = camera_R_object
            rot_grid.append(pose)
        return np.stack(rot_grid, axis=0)

    def _camera_vertical_direction(self) -> np.ndarray:
        camera_R_world = self._camera_rotation_world_to_camera()
        if camera_R_world is None:
            raise RuntimeError("Camera extrinsics are required for vertical-direction projection.")
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        direction = camera_R_world @ world_z
        direction /= np.linalg.norm(direction)
        return direction

    def _project_pose_candidates_to_upright_constraint(
        self,
        poses: np.ndarray,
        upright_lock: Dict,
        obb_frame: Dict,
    ) -> np.ndarray:
        poses_np = np.asarray(poses, dtype=np.float64).copy()
        target_dir_cam = self._camera_vertical_direction()
        axis_obj = (
            obb_frame["axes_obj"][:, upright_lock["axis"]].astype(np.float64)
            * float(upright_lock["axis_sign"])
        )

        for idx in range(len(poses_np)):
            current_axis_cam = poses_np[idx, :3, :3] @ axis_obj
            align_rot = self._rotation_from_vectors(current_axis_cam, target_dir_cam)
            poses_np[idx, :3, :3] = align_rot @ poses_np[idx, :3, :3]
        return poses_np

    def _register_with_upright_constraint(
        self,
        observation: Dict,
        upright_lock: Dict,
        obb_frame: Dict,
    ) -> np.ndarray:
        est = self._estimator
        if est is None:
            raise RuntimeError("Estimator is not initialized.")

        depth = self._erode_depth(observation["depth"], radius=2, device="cuda")
        depth = self._bilateral_filter_depth(depth, radius=2, device="cuda")
        valid = (depth >= 0.001) & (observation["mask"] > 0)
        if valid.sum() < 4:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = est.guess_translation(
                depth=depth,
                mask=observation["mask"],
                K=observation["K"],
            )
            return pose

        est.H, est.W = depth.shape[:2]
        est.K = observation["K"]
        est.ob_id = None
        est.ob_mask = observation["mask"]

        rot_grid = np.asarray(upright_lock["rot_grid"], dtype=np.float64)
        center = est.guess_translation(depth=depth, mask=observation["mask"], K=observation["K"])
        poses_np = rot_grid.copy()
        poses_np[:, :3, 3] = center.reshape(1, 3)
        poses_np = self._project_pose_candidates_to_upright_constraint(
            poses_np,
            upright_lock=upright_lock,
            obb_frame=obb_frame,
        )

        xyz_map = self._depth2xyzmap(depth, observation["K"])

        for _ in range(self.pose_refine_iter):
            poses_refined, _ = self._refiner.predict(
                mesh=est.mesh,
                mesh_tensors=est.mesh_tensors,
                rgb=observation["rgb"],
                depth=depth,
                K=observation["K"],
                ob_in_cams=poses_np,
                normal_map=None,
                xyz_map=xyz_map,
                glctx=est.glctx,
                mesh_diameter=est.diameter,
                iteration=1,
                get_vis=False,
            )
            if self._fp_torch.is_tensor(poses_refined):
                poses_np = poses_refined.detach().cpu().numpy().astype(np.float64)
            else:
                poses_np = np.asarray(poses_refined, dtype=np.float64)
            poses_np = self._project_pose_candidates_to_upright_constraint(
                poses_np,
                upright_lock=upright_lock,
                obb_frame=obb_frame,
            )

        scores, _ = self._scorer.predict(
            mesh=est.mesh,
            rgb=observation["rgb"],
            depth=depth,
            K=observation["K"],
            ob_in_cams=poses_np,
            normal_map=None,
            mesh_tensors=est.mesh_tensors,
            glctx=est.glctx,
            mesh_diameter=est.diameter,
            get_vis=False,
        )
        ids = self._fp_torch.as_tensor(scores).argsort(descending=True)
        best_idx = int(ids[0].item())
        best_pose_centered = poses_np[best_idx]
        tf_to_center = est.get_tf_to_centered_mesh().detach().cpu().numpy().astype(np.float64)
        best_pose = best_pose_centered @ tf_to_center

        est.pose_last = self._fp_torch.as_tensor(
            best_pose_centered,
            device="cuda",
            dtype=self._fp_torch.float32,
        )
        est.best_id = best_idx
        est.poses = self._fp_torch.as_tensor(poses_np, device="cuda", dtype=self._fp_torch.float32)
        est.scores = scores
        return best_pose.astype(np.float64)

    def _estimate_pose(
        self,
        mesh: trimesh.Trimesh,
        observation: Dict,
        debug_dir: Path,
        obb_frame: Optional[Dict] = None,
        upright_lock: Optional[Dict] = None,
        previous_pose_camera: Optional[np.ndarray] = None,
        prefer_tracking: bool = False,
    ) -> Tuple[np.ndarray, str]:
        self._ensure_foundationpose(debug_dir)
        estimation_mode = "register"
        if self._estimator is None:
            self._estimator = self._FoundationPose(
                model_pts=np.asarray(mesh.vertices),
                model_normals=np.asarray(mesh.vertex_normals),
                mesh=mesh,
                scorer=self._scorer,
                refiner=self._refiner,
                glctx=self._glctx,
                debug=0,
                debug_dir=str(debug_dir),
            )
        else:
            self._estimator.debug_dir = str(debug_dir)
            self._estimator.reset_object(
                model_pts=np.asarray(mesh.vertices),
                model_normals=np.asarray(mesh.vertex_normals),
                mesh=mesh,
            )
            self._estimator.glctx = self._glctx

        if self._default_rot_grid is None:
            self._default_rot_grid = self._estimator.rot_grid.detach().clone()

        if upright_lock is not None:
            if obb_frame is None:
                raise ValueError("obb_frame is required when upright_lock is enabled.")
            if "rot_grid" not in upright_lock:
                upright_lock["rot_grid"] = self._build_upright_locked_rot_grid(
                    upright_lock, obb_frame
                )
            pose = self._register_with_upright_constraint(
                observation=observation,
                upright_lock=upright_lock,
                obb_frame=obb_frame,
            )
            estimation_mode = "register_upright_locked"
            return pose.astype(np.float64), estimation_mode

        if self._default_rot_grid is not None:
            self._estimator.rot_grid = self._default_rot_grid.detach().clone()

        pose = self._estimator.register(
            K=observation["K"],
            rgb=observation["rgb"],
            depth=observation["depth"],
            ob_mask=observation["mask"],
            iteration=self.pose_refine_iter,
        )
        return pose.astype(np.float64), estimation_mode

    def _render_mask(self, mesh: trimesh.Trimesh, pose: np.ndarray, observation: Dict) -> np.ndarray:
        mesh_tensors = self._make_mesh_tensors(mesh)
        ob_in_cam = self._fp_torch.as_tensor(
            pose[None], device="cuda", dtype=self._fp_torch.float32
        )
        _, depth, _ = self._nvdiffrast_render(
            K=observation["K"],
            H=observation["H"],
            W=observation["W"],
            ob_in_cams=ob_in_cam,
            glctx=self._glctx,
            mesh_tensors=mesh_tensors,
        )
        render_mask = (depth[0].detach().cpu().numpy() > 1e-6).astype(np.uint8)
        return render_mask

    def _compute_mask_center(self, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros(2, dtype=np.float64)
        return np.array([xs.mean(), ys.mean()], dtype=np.float64)

    def _mask_projection_length(
        self, mask: np.ndarray, direction_uv: np.ndarray, origin_uv: np.ndarray
    ) -> float:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0.0
        points = np.stack([xs, ys], axis=1).astype(np.float64)
        direction = direction_uv.astype(np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return 0.0
        direction /= norm
        projections = (points - origin_uv.reshape(1, 2)) @ direction
        return float(projections.max() - projections.min())

    def _project_axis_directions(
        self,
        pose: np.ndarray,
        obb_frame: Dict,
        current_scale_xyz: np.ndarray,
        K: np.ndarray,
    ) -> List[Dict]:
        axes = []
        center_obj = obb_frame["center_obj"]
        axes_obj = obb_frame["axes_obj"]
        extents = obb_frame["extents"] * current_scale_xyz
        center_cam = pose[:3, :3] @ center_obj + pose[:3, 3]
        if center_cam[2] <= 1e-6:
            return axes

        center_uv = np.array(
            [
                K[0, 0] * center_cam[0] / center_cam[2] + K[0, 2],
                K[1, 1] * center_cam[1] / center_cam[2] + K[1, 2],
            ],
            dtype=np.float64,
        )

        for axis_idx in range(3):
            axis_vec_obj = axes_obj[:, axis_idx] * max(extents[axis_idx] * 0.5, 1e-4)
            endpoint_obj = center_obj + axis_vec_obj
            endpoint_cam = pose[:3, :3] @ endpoint_obj + pose[:3, 3]
            if endpoint_cam[2] <= 1e-6:
                continue
            endpoint_uv = np.array(
                [
                    K[0, 0] * endpoint_cam[0] / endpoint_cam[2] + K[0, 2],
                    K[1, 1] * endpoint_cam[1] / endpoint_cam[2] + K[1, 2],
                ],
                dtype=np.float64,
            )
            direction_uv = endpoint_uv - center_uv
            if np.linalg.norm(direction_uv) < 1e-6:
                continue
            axes.append(
                {
                    "axis": axis_idx,
                    "center_uv": center_uv,
                    "direction_uv": direction_uv,
                }
            )
        return axes

    def _compute_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        intersection = float(np.logical_and(mask_a > 0, mask_b > 0).sum())
        union = float(np.logical_or(mask_a > 0, mask_b > 0).sum())
        if union <= 0:
            return 0.0
        return intersection / union

    def _camera_pose_to_world_pose(self, pose_camera: np.ndarray) -> Optional[np.ndarray]:
        if self.camera_extrinsics is None:
            return None
        camera_frame_align = self._camera_frame_alignment_tf()
        if self.camera_extrinsics_type == "world_from_camera":
            return self.camera_extrinsics @ camera_frame_align @ pose_camera
        if self.camera_extrinsics_type == "camera_from_world":
            return np.linalg.inv(self.camera_extrinsics) @ camera_frame_align @ pose_camera
        raise ValueError(
            f"Unsupported camera_extrinsics_type: {self.camera_extrinsics_type}"
        )

    def _world_pose_to_camera_pose(self, pose_world: np.ndarray) -> Optional[np.ndarray]:
        if self.camera_extrinsics is None:
            return None
        camera_frame_align = self._camera_frame_alignment_tf()
        if self.camera_extrinsics_type == "world_from_camera":
            return camera_frame_align @ np.linalg.inv(self.camera_extrinsics) @ pose_world
        if self.camera_extrinsics_type == "camera_from_world":
            return camera_frame_align @ self.camera_extrinsics @ pose_world
        raise ValueError(
            f"Unsupported camera_extrinsics_type: {self.camera_extrinsics_type}"
        )

    def _rotation_from_vectors(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        a = np.asarray(vec_a, dtype=np.float64)
        b = np.asarray(vec_b, dtype=np.float64)
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-8 or b_norm < 1e-8:
            return np.eye(3, dtype=np.float64)
        a /= a_norm
        b /= b_norm

        cross = np.cross(a, b)
        cross_norm = np.linalg.norm(cross)
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

        if cross_norm < 1e-8:
            if dot > 0.0:
                return np.eye(3, dtype=np.float64)
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(a[0]) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = np.cross(a, ref)
            axis /= np.linalg.norm(axis)
            k = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                dtype=np.float64,
            )
            return np.eye(3, dtype=np.float64) + 2.0 * (k @ k)

        k = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + k + (k @ k) * ((1.0 - dot) / (cross_norm ** 2))

    def _find_closest_world_vertical_axis(
        self,
        pose_camera: np.ndarray,
        obb_frame: Dict,
    ) -> Optional[Dict]:
        if self.camera_extrinsics is None:
            return None
        pose_world = self._camera_pose_to_world_pose(pose_camera)
        if pose_world is None:
            return None

        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        best_record = None

        for axis_idx in range(3):
            axis_obj = obb_frame["axes_obj"][:, axis_idx].astype(np.float64)
            axis_world = pose_world[:3, :3] @ axis_obj
            axis_norm = np.linalg.norm(axis_world)
            if axis_norm < 1e-8:
                continue
            axis_world /= axis_norm
            dot = float(np.clip(axis_world @ world_z, -1.0, 1.0))
            abs_dot = abs(dot)
            if best_record is None or abs_dot > best_record["abs_dot"]:
                best_record = {
                    "axis": int(axis_idx),
                    "axis_sign": float(1.0 if dot >= 0.0 else -1.0),
                    "dot_to_world_z": float(dot),
                    "abs_dot": float(abs_dot),
                    "angle_deg": float(np.degrees(np.arccos(abs_dot))),
                    "nearest_world_axis": "+Z" if dot >= 0.0 else "-Z",
                }

        return best_record

    def _detect_upright_axis_lock(
        self,
        pose_camera: np.ndarray,
        obb_frame: Dict,
    ) -> Optional[Dict]:
        best_record = self._find_closest_world_vertical_axis(pose_camera, obb_frame)
        threshold_dot = float(np.cos(np.deg2rad(self.upright_lock_angle_deg)))
        if best_record is None or best_record["abs_dot"] < threshold_dot:
            return None
        return best_record

    def _apply_upright_axis_lock(
        self,
        pose_camera: np.ndarray,
        obb_frame: Dict,
        upright_lock: Dict,
    ) -> np.ndarray:
        pose_world = self._camera_pose_to_world_pose(pose_camera)
        if pose_world is None:
            return pose_camera

        axis_obj = (
            obb_frame["axes_obj"][:, upright_lock["axis"]].astype(np.float64)
            * float(upright_lock["axis_sign"])
        )
        axis_world = pose_world[:3, :3] @ axis_obj
        align_rot = self._rotation_from_vectors(axis_world, np.array([0.0, 0.0, 1.0]))
        aligned_world_pose = pose_world.copy()
        aligned_world_pose[:3, :3] = align_rot @ pose_world[:3, :3]
        aligned_camera_pose = self._world_pose_to_camera_pose(aligned_world_pose)
        if aligned_camera_pose is None:
            return pose_camera
        return aligned_camera_pose.astype(np.float64)

    def _save_mask_overlap_visualization(
        self,
        rgb: np.ndarray,
        observed_mask: np.ndarray,
        render_mask: np.ndarray,
        output_path: Path,
    ) -> None:
        vis = rgb.copy()
        observed_only = (observed_mask > 0) & ~(render_mask > 0)
        render_only = (render_mask > 0) & ~(observed_mask > 0)
        overlap = (observed_mask > 0) & (render_mask > 0)

        vis[observed_only] = (0.45 * vis[observed_only] + 0.55 * np.array([0, 255, 0])).astype(
            np.uint8
        )
        vis[render_only] = (0.45 * vis[render_only] + 0.55 * np.array([255, 0, 0])).astype(
            np.uint8
        )
        vis[overlap] = (0.35 * vis[overlap] + 0.65 * np.array([255, 255, 0])).astype(np.uint8)
        cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    def _save_foundationpose_detection_visualization(
        self,
        mesh: trimesh.Trimesh,
        pose: np.ndarray,
        observation: Dict,
        output_path: Path,
    ) -> None:
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)
        center_pose = pose @ np.linalg.inv(to_origin)
        axis_scale = max(float(np.max(extents)) * 0.6, 0.03)

        vis = self._draw_posed_3d_box(
            observation["K"],
            img=observation["rgb"].copy(),
            ob_in_cam=center_pose,
            bbox=bbox,
        )
        vis = self._draw_xyz_axis(
            vis,
            ob_in_cam=center_pose,
            scale=axis_scale,
            K=observation["K"],
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    def _optimize_mesh_scale(
        self,
        mesh_path: Path,
        base_mesh: trimesh.Trimesh,
        observation: Dict,
        output_root: Path,
    ) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray, Dict]:
        iso_scale, init_info = self.calculate_scale_from_depth(base_mesh, observation)
        current_scale_xyz = np.full(3, iso_scale, dtype=np.float64)
        obb_frame = self._compute_obb_frame(base_mesh)
        work_base_mesh, work_mesh_info = self._build_work_mesh(base_mesh)
        history: List[Dict] = []
        debug_root = output_root / f"{mesh_path.stem}_iter_debug"
        debug_root.mkdir(parents=True, exist_ok=True)

        final_pose = None
        final_render_mask = None
        upright_lock = None

        for iteration_idx in range(self.opt_iters):
            iter_start = self._start_timer()

            mesh_scale_start = self._start_timer()
            current_work_mesh = self._apply_obb_scaling(
                work_base_mesh, obb_frame, current_scale_xyz
            )
            mesh_scale_time = self._finish_timer(mesh_scale_start)
            iter_dir = debug_root / f"iter_{iteration_idx:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            pose_start = self._start_timer()
            pose, estimation_mode = self._estimate_pose(
                current_work_mesh,
                observation,
                iter_dir,
                obb_frame=obb_frame,
                upright_lock=upright_lock,
                previous_pose_camera=final_pose,
                prefer_tracking=False,
            )
            pose_time = self._finish_timer(pose_start)

            upright_align_time = 0.0
            if self.camera_extrinsics is not None and (iteration_idx + 1) >= self.upright_lock_after_iter:
                if upright_lock is None:
                    upright_lock = self._detect_upright_axis_lock(pose, obb_frame)
                    if upright_lock is not None:
                        print(
                            "  Upright-axis lock enabled: "
                            f"axis={upright_lock['axis']} "
                            f"sign={int(upright_lock['axis_sign'])} "
                            f"angle={upright_lock['angle_deg']:.2f}deg "
                            f"after iter={iteration_idx + 1}; "
                            f"subsequent register candidates={len(upright_lock['rot_grid']) if 'rot_grid' in upright_lock else 12}"
                        )
                if upright_lock is not None:
                    upright_align_start = self._start_timer()
                    pose = self._apply_upright_axis_lock(pose, obb_frame, upright_lock)
                    upright_align_time = self._finish_timer(upright_align_start)

            closest_vertical_axis = self._find_closest_world_vertical_axis(pose, obb_frame)

            render_start = self._start_timer()
            render_mask = self._render_mask(current_work_mesh, pose, observation)
            render_time = self._finish_timer(render_start)

            metric_start = self._start_timer()
            iou = self._compute_iou(observation["mask"], render_mask)

            obs_origin = self._compute_mask_center(observation["mask"])
            render_origin = self._compute_mask_center(render_mask)
            axis_records = []
            updated_scale_xyz = current_scale_xyz.copy()

            for axis_info in self._project_axis_directions(
                pose=pose,
                obb_frame=obb_frame,
                current_scale_xyz=current_scale_xyz,
                K=observation["K"],
            ):
                obs_len = self._mask_projection_length(
                    observation["mask"], axis_info["direction_uv"], obs_origin
                )
                render_len = self._mask_projection_length(
                    render_mask, axis_info["direction_uv"], render_origin
                )

                update_factor = 1.0
                valid_for_update = (
                    obs_len >= self.projection_min_pixels
                    and render_len >= self.projection_min_pixels
                )
                if valid_for_update:
                    raw_ratio = obs_len / max(render_len, 1e-6)
                    update_factor = float(
                        np.clip(
                            raw_ratio ** self.scale_exponent,
                            1.0 / self.scale_step_cap,
                            self.scale_step_cap,
                        )
                    )
                    updated_scale_xyz[axis_info["axis"]] *= update_factor

                axis_records.append(
                    {
                        "axis": int(axis_info["axis"]),
                        "obs_projection_pixels": float(obs_len),
                        "render_projection_pixels": float(render_len),
                        "update_factor": float(update_factor),
                        "updated": bool(valid_for_update),
                    }
                )
            metric_time = self._finish_timer(metric_start)

            history.append(
                {
                    "iteration": int(iteration_idx),
                    "iou": float(iou),
                    "scale_xyz_before": current_scale_xyz.astype(float).tolist(),
                    "scale_xyz_after": updated_scale_xyz.astype(float).tolist(),
                    "axis_updates": axis_records,
                    "pose": pose.astype(float).tolist(),
                    "estimation_mode": estimation_mode,
                    "upright_axis_lock": (
                        {
                            "axis": int(upright_lock["axis"]),
                            "axis_sign": float(upright_lock["axis_sign"]),
                            "angle_deg_at_lock": float(upright_lock["angle_deg"]),
                        }
                        if upright_lock is not None
                        else None
                    ),
                    "timing_sec": {
                        "mesh_scaling": float(mesh_scale_time),
                        "pose_estimation": float(pose_time),
                        "upright_axis_align": float(upright_align_time),
                        "mask_render": float(render_time),
                        "projection_and_scale_update": float(metric_time),
                    },
                }
            )

            vis_start = self._start_timer()
            cv2.imwrite(
                str(iter_dir / "observed_mask.png"),
                (observation["mask"] * 255).astype(np.uint8),
            )
            cv2.imwrite(str(iter_dir / "render_mask.png"), (render_mask * 255).astype(np.uint8))
            self._save_mask_overlap_visualization(
                observation["rgb"],
                observation["mask"],
                render_mask,
                iter_dir / "mask_overlap.png",
            )
            self._save_foundationpose_detection_visualization(
                current_work_mesh,
                pose,
                observation,
                iter_dir / "foundationpose_detection.png",
            )
            vis_time = self._finish_timer(vis_start)
            iter_total_time = self._finish_timer(iter_start)
            history[-1]["timing_sec"]["visualization_save"] = float(vis_time)
            history[-1]["timing_sec"]["iteration_total"] = float(iter_total_time)

            print(
                f"  Iter {iteration_idx + 1}/{self.opt_iters}: "
                f"IoU={iou:.4f}, scale_xyz(before)={current_scale_xyz.round(6).tolist()}, "
                f"scale_xyz(after)={updated_scale_xyz.round(6).tolist()}, "
                f"pose_mode={estimation_mode}"
            )
            if closest_vertical_axis is not None:
                print(
                    "     vertical-nearest: "
                    f"axis={closest_vertical_axis['axis']} "
                    f"target={closest_vertical_axis['nearest_world_axis']} "
                    f"angle={closest_vertical_axis['angle_deg']:.2f}deg"
                )
            print(
                "     timing: "
                f"mesh_scaling={mesh_scale_time:.3f}s "
                f"pose_estimation={pose_time:.3f}s "
                f"upright_axis_align={upright_align_time:.3f}s "
                f"mask_render={render_time:.3f}s "
                f"projection_update={metric_time:.3f}s "
                f"visualization_save={vis_time:.3f}s "
                f"total={iter_total_time:.3f}s"
            )
            for axis_record in axis_records:
                print(
                    "     "
                    f"axis={axis_record['axis']} "
                    f"obs={axis_record['obs_projection_pixels']:.2f}px "
                    f"render={axis_record['render_projection_pixels']:.2f}px "
                    f"factor={axis_record['update_factor']:.4f} "
                    f"updated={axis_record['updated']}"
                )

            current_scale_xyz = updated_scale_xyz
            final_pose = pose
            final_render_mask = render_mask

            if iou >= self.iou_stop:
                print(f"  Early stop: IoU {iou:.4f} >= {self.iou_stop:.4f}")
                break

        final_mesh_start = self._start_timer()
        final_mesh = self._apply_obb_scaling(base_mesh, obb_frame, current_scale_xyz)
        final_work_mesh = self._apply_obb_scaling(work_base_mesh, obb_frame, current_scale_xyz)
        final_mesh_time = self._finish_timer(final_mesh_start)

        final_pose_start = self._start_timer()
        final_pose, final_estimation_mode = self._estimate_pose(
            final_work_mesh,
            observation,
            debug_root / "final_pose",
            obb_frame=obb_frame,
            upright_lock=upright_lock,
            previous_pose_camera=final_pose,
            prefer_tracking=False,
        )
        final_pose_time = self._finish_timer(final_pose_start)

        final_upright_align_time = 0.0
        if upright_lock is not None:
            final_upright_align_start = self._start_timer()
            final_pose = self._apply_upright_axis_lock(final_pose, obb_frame, upright_lock)
            final_upright_align_time = self._finish_timer(final_upright_align_start)
        final_closest_vertical_axis = self._find_closest_world_vertical_axis(final_pose, obb_frame)

        final_render_start = self._start_timer()
        final_render_mask = self._render_mask(final_work_mesh, final_pose, observation)
        final_render_time = self._finish_timer(final_render_start)
        final_iou = self._compute_iou(observation["mask"], final_render_mask)

        final_vis_start = self._start_timer()
        cv2.imwrite(
            str(debug_root / "final_render_mask.png"),
            (final_render_mask * 255).astype(np.uint8),
        )
        self._save_mask_overlap_visualization(
            observation["rgb"],
            observation["mask"],
            final_render_mask,
            debug_root / "final_mask_overlap.png",
        )
        self._save_foundationpose_detection_visualization(
            final_work_mesh,
            final_pose,
            observation,
            debug_root / "final_foundationpose_detection.png",
        )
        final_vis_time = self._finish_timer(final_vis_start)

        print(
            "  Final pass timing: "
            f"pose_mode={final_estimation_mode} "
            f"mesh_scaling={final_mesh_time:.3f}s "
            f"pose_estimation={final_pose_time:.3f}s "
            f"upright_axis_align={final_upright_align_time:.3f}s "
            f"mask_render={final_render_time:.3f}s "
            f"visualization_save={final_vis_time:.3f}s"
        )
        if final_closest_vertical_axis is not None:
            print(
                "  Final vertical-nearest: "
                f"axis={final_closest_vertical_axis['axis']} "
                f"target={final_closest_vertical_axis['nearest_world_axis']} "
                f"angle={final_closest_vertical_axis['angle_deg']:.2f}deg"
            )

        summary = {
            "initial_scale_info": init_info,
            "work_mesh_info": work_mesh_info,
            "iterations": history,
            "final_scale_xyz": current_scale_xyz.astype(float).tolist(),
            "final_iou": float(final_iou),
            "final_pose": final_pose.astype(float).tolist(),
            "final_pose_estimation_mode": final_estimation_mode,
            "upright_axis_lock": (
                {
                    "axis": int(upright_lock["axis"]),
                    "axis_sign": float(upright_lock["axis_sign"]),
                    "angle_deg_at_lock": float(upright_lock["angle_deg"]),
                }
                if upright_lock is not None
                else None
            ),
            "debug_dir": str(debug_root),
            "final_timing_sec": {
                "mesh_scaling": float(final_mesh_time),
                "pose_estimation": float(final_pose_time),
                "upright_axis_align": float(final_upright_align_time),
                "mask_render": float(final_render_time),
                "visualization_save": float(final_vis_time),
            },
        }
        return final_mesh, current_scale_xyz, final_pose, summary

    def save_scaled_mesh(self, mesh: trimesh.Trimesh, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))
        print(f"  Saved scaled mesh to: {output_path}")

    def process_all(
        self,
        output_dir: Optional[str] = None,
        save_scaled_meshes: bool = True,
        save_npz: bool = True,
    ) -> Dict:
        mesh_files = self._collect_mesh_files()
        if output_dir:
            output_root = Path(output_dir).resolve()
        else:
            scene_tag = f"{self.dataset_path.parent.name}_{self.dataset_path.name}"
            output_root = (self.code_dir / "adjust_pose_outputs" / scene_tag).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict] = {}

        print("=" * 60)
        print(f"Dataset path: {self.dataset_path}")
        print(f"Mask dir:      {self.mask_dir}")
        print(f"Depth dir:     {self.depth_dir}")
        print(f"RGB dir:       {self.rgb_dir if self.rgb_dir else 'N/A'}")
        print(f"Mesh source:   {self.mesh_path if self.mesh_path else self.mesh_dir}")
        print(f"Output dir:    {output_root}")
        print(
            f"Optimization:  opt_iters={self.opt_iters}, pose_refine_iter={self.pose_refine_iter}, "
            f"track_refine_iter={self.track_refine_iter}(reserved), "
            f"max_work_mesh_faces={self.max_work_mesh_faces}, "
            f"upright_lock_after_iter={self.upright_lock_after_iter}, "
            f"upright_lock_angle_deg={self.upright_lock_angle_deg}, "
            f"scale_step_cap={self.scale_step_cap}, scale_exponent={self.scale_exponent}"
        )
        print(
            f"Camera extrinsics: "
            f"{self.camera_extrinsics_path if self.camera_extrinsics_path else 'N/A'} "
            f"(type={self.camera_extrinsics_type}, alignment={self.camera_frame_alignment})"
        )
        print("=" * 60)

        for mesh_path in mesh_files:
            basename = mesh_path.stem
            print(f"\n{'=' * 60}")
            print(f"Processing mesh: {mesh_path}")
            print(f"{'=' * 60}")

            try:
                base_mesh = self._load_trimesh(mesh_path)
                observation = self._load_observation(basename)
                final_mesh, scale_xyz, final_pose, summary = self._optimize_mesh_scale(
                    mesh_path=mesh_path,
                    base_mesh=base_mesh,
                    observation=observation,
                    output_root=output_root,
                )
                final_pose_world = self._camera_pose_to_world_pose(final_pose)
                results[basename] = {
                    "success": True,
                    "observation_frame": observation["frame_name"],
                    "final_scale_xyz": scale_xyz.astype(float).tolist(),
                    "final_pose": final_pose.astype(float).tolist(),
                    "final_pose_world": (
                        final_pose_world.astype(float).tolist()
                        if final_pose_world is not None
                        else None
                    ),
                    "camera_extrinsics_path": (
                        str(self.camera_extrinsics_path)
                        if self.camera_extrinsics_path is not None
                        else None
                    ),
                    "camera_extrinsics_type": self.camera_extrinsics_type,
                    **summary,
                }
            except Exception as exc:
                print(f"  Failed: {exc}")
                results[basename] = {"success": False, "error": str(exc)}
                continue

            print(f"  Final scale_xyz: {np.round(scale_xyz, 6).tolist()}")
            print(f"  Final IoU: {summary['final_iou']:.4f}")
            upright_lock_info = summary.get("upright_axis_lock")
            if upright_lock_info is not None:
                print(
                    "  Fixed world-Z mode: activated "
                    f"(axis={upright_lock_info['axis']}, "
                    f"sign={int(upright_lock_info['axis_sign'])}, "
                    f"angle_at_lock={upright_lock_info['angle_deg_at_lock']:.2f}deg)"
                )
            else:
                print("  Fixed world-Z mode: not activated")
            if results[basename].get("final_pose_world") is not None:
                print("  Saved both camera-frame and world-frame object poses.")

            scaled_mesh_path = output_root / mesh_path.name
            if save_scaled_meshes:
                self.save_scaled_mesh(final_mesh, scaled_mesh_path)
                results[basename]["scaled_mesh"] = str(scaled_mesh_path)

            if save_npz:
                npz_path = output_root / f"{basename}_pose_scaled.npz"
                final_pose_world = self._camera_pose_to_world_pose(final_pose)
                np.savez(
                    npz_path,
                    pose=np.asarray(final_pose, dtype=np.float32),
                    pose_world=(
                        np.asarray(final_pose_world, dtype=np.float32)
                        if final_pose_world is not None
                        else np.eye(4, dtype=np.float32)
                    ),
                    scale_xyz=np.asarray(scale_xyz, dtype=np.float32),
                    camera_extrinsics=(
                        np.asarray(self.camera_extrinsics, dtype=np.float32)
                        if self.camera_extrinsics is not None
                        else np.eye(4, dtype=np.float32)
                    ),
                )
                print(f"  Saved NPZ to: {npz_path}")
                results[basename]["pose_npz"] = str(npz_path)
                if final_pose_world is not None:
                    npz_world_path = output_root / f"{basename}_pose_world_scaled.npz"
                    np.savez(
                        npz_world_path,
                        pose_world=np.asarray(final_pose_world, dtype=np.float32),
                        pose_camera=np.asarray(final_pose, dtype=np.float32),
                        scale_xyz=np.asarray(scale_xyz, dtype=np.float32),
                        camera_extrinsics=np.asarray(self.camera_extrinsics, dtype=np.float32),
                    )
                    print(f"  Saved world-pose NPZ to: {npz_world_path}")
                    results[basename]["pose_world_npz"] = str(npz_world_path)

        results_path = output_root / "scale_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        successful = sum(1 for value in results.values() if value.get("success"))
        total = len(results)
        print(f"\n{'=' * 60}")
        print(f"Results saved to: {results_path}")
        print(f"Summary: {successful}/{total} meshes processed successfully")
        if successful > 0:
            ious = [value["final_iou"] for value in results.values() if value.get("success")]
            print(f"Final IoU range: {min(ious):.4f} - {max(ious):.4f}")
            print(f"Mean final IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
            upright_locked = [
                name
                for name, value in results.items()
                if value.get("success") and value.get("upright_axis_lock") is not None
            ]
            if upright_locked:
                print(f"Meshes using fixed world-Z mode: {', '.join(upright_locked)}")
            else:
                print("Meshes using fixed world-Z mode: none")
            print(
                "Next step example:\n"
                f"  python {Path(__file__).with_name('run_demo.py')} "
                f"--mesh_file {output_root / mesh_files[0].name} "
                f"--test_scene_dir {self.dataset_path}"
            )
        print(f"{'=' * 60}\n")
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Adjust mesh size with iterative FoundationPose-guided anisotropic optimization."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Scene directory containing masks/depth/rgb/cam_K.txt",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=None,
        help="Path to a mesh file or a directory of meshes. If omitted, dataset_path/mesh is used.",
    )
    parser.add_argument(
        "--frame_name",
        type=str,
        default=None,
        help="Frame stem used to pick rgb/mask/depth, for example '0'. Defaults to an automatic match.",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="Depth image scale when the stored values are in millimeters.",
    )
    parser.add_argument(
        "--opt_iters",
        type=int,
        default=5,
        help="Number of scale optimization iterations.",
    )
    parser.add_argument(
        "--pose_refine_iter",
        type=int,
        default=3,
        help="FoundationPose register refinement iterations for the first optimization step.",
    )
    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=4,
        help="Reserved track_one iterations parameter kept for compatibility; the current optimization loop always uses register.",
    )
    parser.add_argument(
        "--scale_step_cap",
        type=float,
        default=1.1,
        help="Maximum per-step multiplicative scale update on each axis.",
    )
    parser.add_argument(
        "--scale_exponent",
        type=float,
        default=0.5,
        help="Exponent used when converting projection-length ratio to axis scale update.",
    )
    parser.add_argument(
        "--projection_min_pixels",
        type=float,
        default=8.0,
        help="Minimum projection length required before updating an axis.",
    )
    parser.add_argument(
        "--iou_stop",
        type=float,
        default=0.98,
        help="Early-stop threshold on rendered-mask IoU.",
    )
    parser.add_argument(
        "--max_work_mesh_faces",
        type=int,
        default=100000,
        help="If a mesh is larger than this, create a simplified work mesh for pose estimation/rendering.",
    )
    parser.add_argument(
        "--upright_lock_after_iter",
        type=int,
        default=2,
        help="Start checking world-frame upright-axis locking from this optimization iteration (1-based).",
    )
    parser.add_argument(
        "--upright_lock_angle_deg",
        type=float,
        default=10.0,
        help="If an object axis is within this many degrees of world +Z, lock it to the world +Z direction.",
    )
    parser.add_argument(
        "--camera_extrinsics_path",
        type=str,
        default=None,
        help="Optional 4x4 camera extrinsics file used to convert object pose from camera frame to world frame.",
    )
    parser.add_argument(
        "--camera_extrinsics_type",
        type=str,
        choices=["world_from_camera", "camera_from_world"],
        default="world_from_camera",
        help="Interpretation of --camera_extrinsics_path. Default assumes the matrix is world_T_camera.",
    )
    parser.add_argument(
        "--camera_frame_alignment",
        type=str,
        choices=["identity", "legacy_mobile_etot"],
        default="identity",
        help=(
            "Optional extra fixed camera-frame alignment. Use 'identity' for correctly saved "
            "world_T_camera extrinsics. Use 'legacy_mobile_etot' only for older camera exports."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: dataset_path/mesh_scaled",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not export scaled mesh files.",
    )
    parser.add_argument(
        "--no-npz",
        action="store_true",
        help="Do not export *_pose_scaled.npz files.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Reserved for compatibility. Visualization is not implemented in this script.",
    )
    args = parser.parse_args()

    if args.visualize:
        print("Warning: --visualize is currently ignored.")

    adjuster = MeshSizeAdjuster(
        dataset_path=args.dataset_path,
        mesh_path=args.mesh_path,
        frame_name=args.frame_name,
        depth_scale=args.depth_scale,
        opt_iters=args.opt_iters,
        pose_refine_iter=args.pose_refine_iter,
        track_refine_iter=args.track_refine_iter,
        scale_step_cap=args.scale_step_cap,
        scale_exponent=args.scale_exponent,
        projection_min_pixels=args.projection_min_pixels,
        iou_stop=args.iou_stop,
        max_work_mesh_faces=args.max_work_mesh_faces,
        upright_lock_after_iter=args.upright_lock_after_iter,
        upright_lock_angle_deg=args.upright_lock_angle_deg,
        camera_extrinsics_path=args.camera_extrinsics_path,
        camera_extrinsics_type=args.camera_extrinsics_type,
        camera_frame_alignment=args.camera_frame_alignment,
    )
    return adjuster.process_all(
        output_dir=args.output_dir,
        save_scaled_meshes=not args.no_save,
        save_npz=not args.no_npz,
    )


if __name__ == "__main__":
    main()
