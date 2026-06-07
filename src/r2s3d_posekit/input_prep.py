from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OBJECT_ID_PATTERNS = [
    re.compile(r"^object_(\d+)$"),
    re.compile(r"^(\d+)$"),
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask image: {path}")
    if mask.ndim == 3:
        channel_sums = [mask[..., i].sum() for i in range(mask.shape[2])]
        mask = mask[..., int(np.argmax(channel_sums))]
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def read_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    return np.asarray(depth)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR))


def save_mask(path: Path, mask: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8))


def save_depth_png(path: Path, depth: np.ndarray, depth_scale: float = 1000.0) -> None:
    ensure_dir(path.parent)
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.dtype.kind == "f":
        depth_mm = np.clip(
            np.round(depth * float(depth_scale)),
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
    else:
        depth_mm = depth.astype(np.uint16, copy=False)
    cv2.imwrite(str(path), depth_mm)


def write_intrinsics(path: Path, K: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.savetxt(path, np.asarray(K, dtype=np.float64).reshape(-1), fmt="%.12f")


def load_intrinsics(
    intrinsics_path: Optional[Path] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> np.ndarray:
    if intrinsics_path is not None:
        intrinsics_path = Path(intrinsics_path)
        if intrinsics_path.suffix.lower() == ".npy":
            K = np.load(intrinsics_path)
        else:
            K = np.loadtxt(intrinsics_path)
        K = np.asarray(K, dtype=np.float64)
        if K.size != 9:
            raise ValueError(
                f"Intrinsics file must contain 9 values, got shape {K.shape}: {intrinsics_path}"
            )
        return K.reshape(3, 3)

    values = [fx, fy, cx, cy]
    if any(v is None for v in values):
        raise ValueError(
            "Provide either --intrinsics or all of --fx --fy --cx --cy."
        )
    K = np.array(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def infer_mask_map(mask_dir: Path, mask_glob: str = "*.png") -> Dict[int, Path]:
    mask_dir = Path(mask_dir)
    paths = sorted(mask_dir.glob(mask_glob))
    if not paths:
        raise FileNotFoundError(f"No masks matched {mask_glob!r} in {mask_dir}")
    result: Dict[int, Path] = {}
    for path in paths:
        object_id = None
        for pattern in OBJECT_ID_PATTERNS:
            matched = pattern.match(path.stem)
            if matched:
                object_id = int(matched.group(1))
                break
        if object_id is None:
            raise ValueError(
                "Mask filename stem must look like '<id>' or 'object_<id>', "
                f"got: {path.name}"
            )
        result[object_id] = path
    return result


def copy_optional_file(src: Optional[Path], dst: Path) -> None:
    if src is None:
        return
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Optional file was requested but not found: {src}")
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def prepare_multi_object_workspace(
    *,
    rgb_path: Path,
    depth_path: Path,
    mask_map: Dict[int, Path],
    output_dir: Path,
    K: np.ndarray,
    depth_scale: float = 1000.0,
    extrinsics_path: Optional[Path] = None,
    gt_pose_json: Optional[Path] = None,
) -> Dict:
    output_dir = Path(output_dir).resolve()
    sam3d_dir = ensure_dir(output_dir / "sam3d_inputs")
    fp_root = ensure_dir(output_dir / "foundationpose_inputs")
    eval_camera_dir = ensure_dir(output_dir / "evaluation_camera_data")

    rgb = read_rgb(rgb_path)
    depth = read_depth(depth_path)

    save_rgb(sam3d_dir / "image.png", rgb)
    save_rgb(eval_camera_dir / "rgb_0.png", rgb)
    save_depth_png(eval_camera_dir / "depth_0.png", depth, depth_scale=depth_scale)
    np.save(eval_camera_dir / "intrinsics_0.npy", np.asarray(K, dtype=np.float64))
    write_intrinsics(eval_camera_dir / "cam_K.txt", K)
    copy_optional_file(extrinsics_path, eval_camera_dir / "extrinsics_0.npy")
    copy_optional_file(gt_pose_json, eval_camera_dir / "object_poses_0.json")

    objects = []
    for object_id, mask_path in sorted(mask_map.items()):
        mask = read_mask(mask_path)
        save_mask(sam3d_dir / f"{object_id}.png", mask)
        save_mask(eval_camera_dir / f"object_{object_id}.png", mask)

        scene_dir = ensure_dir(fp_root / f"object_{object_id}")
        ensure_dir(scene_dir / "rgb")
        ensure_dir(scene_dir / "depth")
        ensure_dir(scene_dir / "masks")
        ensure_dir(scene_dir / "mesh")

        save_rgb(scene_dir / "rgb" / "0.png", rgb)
        save_depth_png(scene_dir / "depth" / "0.png", depth, depth_scale=depth_scale)
        save_mask(scene_dir / "masks" / "0.png", mask)
        write_intrinsics(scene_dir / "cam_K.txt", K)

        objects.append(
            {
                "object_id": int(object_id),
                "mask_source": str(Path(mask_path).resolve()),
                "foundationpose_scene_dir": str(scene_dir),
            }
        )

    manifest = {
        "rgb_path": str(Path(rgb_path).resolve()),
        "depth_path": str(Path(depth_path).resolve()),
        "intrinsics": np.asarray(K, dtype=np.float64).tolist(),
        "sam3d_input_dir": str(sam3d_dir),
        "foundationpose_input_root": str(fp_root),
        "evaluation_camera_data_dir": str(eval_camera_dir),
        "objects": objects,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def prepare_single_object_scene(
    *,
    rgb_path: Path,
    depth_path: Path,
    mask_path: Path,
    object_id: int,
    output_dir: Path,
    K: np.ndarray,
    depth_scale: float = 1000.0,
) -> Dict:
    manifest = prepare_multi_object_workspace(
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_map={int(object_id): Path(mask_path)},
        output_dir=output_dir,
        K=K,
        depth_scale=depth_scale,
    )
    return manifest
