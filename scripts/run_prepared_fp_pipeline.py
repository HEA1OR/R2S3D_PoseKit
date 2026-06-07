#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from r2s3d_posekit.input_prep import (
    ensure_dir,
    read_depth,
    read_mask,
    read_rgb,
    save_depth_png,
    save_mask,
    save_rgb,
    write_intrinsics,
)


def split_python_cmd(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("Python command must not be empty.")
    return parts


def run_subprocess(command: list[str], cwd: Path | None = None) -> None:
    print("\n$ " + " ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True, cwd=(str(cwd) if cwd is not None else None))


def parse_object_id(name: str) -> int:
    if not name.startswith("object_"):
        raise ValueError(f"Prepared object directory must be named like object_<id>, got: {name}")
    return int(name.split("_", 1)[1])


def load_cam_k(path: Path) -> np.ndarray:
    K = np.loadtxt(path)
    K = np.asarray(K, dtype=np.float64)
    if K.size != 9:
        raise ValueError(f"cam_K.txt must contain 9 values, got shape {K.shape}: {path}")
    return K.reshape(3, 3)


def discover_prepared_objects(prepared_root: Path) -> list[dict]:
    objects = []
    for scene_dir in sorted(prepared_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        try:
            object_id = parse_object_id(scene_dir.name)
        except Exception:
            continue
        mesh_candidates = sorted((scene_dir / "mesh").glob(f"{object_id}.*"))
        if not mesh_candidates:
            mesh_candidates = sorted((scene_dir / "mesh").glob("*"))
        if not mesh_candidates:
            raise FileNotFoundError(f"No mesh file found under {scene_dir / 'mesh'}")
        entry = {
            "object_id": object_id,
            "scene_dir": scene_dir.resolve(),
            "rgb": (scene_dir / "rgb" / "0.png").resolve(),
            "depth": (scene_dir / "depth" / "0.png").resolve(),
            "mask": (scene_dir / "masks" / "0.png").resolve(),
            "mesh": mesh_candidates[0].resolve(),
            "cam_k": (scene_dir / "cam_K.txt").resolve(),
        }
        for key in ["rgb", "depth", "mask", "mesh", "cam_k"]:
            if not Path(entry[key]).exists():
                raise FileNotFoundError(f"Missing required prepared file: {entry[key]}")
        objects.append(entry)
    if not objects:
        raise FileNotFoundError(f"No prepared object_* directories found in {prepared_root}")
    return objects


def build_camera_data_from_prepared(
    *,
    prepared_objects: list[dict],
    output_dir: Path,
    depth_scale: float,
    camera_extrinsics_path: Path | None,
    gt_pose_json: Path | None,
) -> Path:
    output_dir = ensure_dir(output_dir.resolve())
    ref = prepared_objects[0]
    rgb = read_rgb(Path(ref["rgb"]))
    depth = read_depth(Path(ref["depth"]))
    K = load_cam_k(Path(ref["cam_k"]))

    save_rgb(output_dir / "rgb_0.png", rgb)
    save_depth_png(output_dir / "depth_0.png", depth, depth_scale=depth_scale)
    np.save(output_dir / "intrinsics_0.npy", K)
    write_intrinsics(output_dir / "cam_K.txt", K)

    for entry in prepared_objects:
        mask = read_mask(Path(entry["mask"]))
        save_mask(output_dir / f"object_{entry['object_id']}.png", mask)

    if camera_extrinsics_path is not None:
        output_dir.joinpath("extrinsics_0.npy").write_bytes(camera_extrinsics_path.read_bytes())
    if gt_pose_json is not None:
        output_dir.joinpath("object_poses_0.json").write_bytes(gt_pose_json.read_bytes())
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the iterative FoundationPose scale+pose stage from an already prepared per-object FoundationPose input root."
    )
    parser.add_argument(
        "--prepared-fp-root",
        type=Path,
        required=True,
        help="Prepared per-object FoundationPose input root containing object_0, object_1, ...",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for optimization outputs.")
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        required=True,
        help="Installed FoundationPose root that already contains the patched adjust_pose.py and Utils.py.",
    )
    parser.add_argument(
        "--foundationpose-python",
        type=str,
        default="python",
        help="Python launcher for FoundationPose, for example 'python' or 'conda run -n foundationpose python'.",
    )
    parser.add_argument(
        "--eval-python",
        type=str,
        default="python",
        help="Python launcher for evaluation and rendering scripts.",
    )
    parser.add_argument(
        "--reuse-mesh-root",
        type=Path,
        default=None,
        help="Optional reconstructed mesh root. If given, object_<id>/mesh/<id>.glb will be replaced by <reuse-mesh-root>/result_<id>/<id>.glb before optimization.",
    )
    parser.add_argument(
        "--camera-data-dir",
        type=Path,
        default=None,
        help="Optional camera data dir. If omitted, one is synthesized from the prepared object folders.",
    )
    parser.add_argument(
        "--camera-extrinsics-path",
        type=Path,
        default=None,
        help="Optional camera extrinsics .npy copied into the synthesized camera data directory and passed into adjust_pose.py.",
    )
    parser.add_argument(
        "--gt-pose-json",
        type=Path,
        default=None,
        help="Optional runtime GT pose JSON copied into the synthesized camera data directory and used during evaluation.",
    )
    parser.add_argument(
        "--gt-spec",
        type=Path,
        default=None,
        help="Optional GT object spec JSON. Enables evaluation and camera-view overlap rendering.",
    )
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth multiplier used for PNG export.")
    parser.add_argument("--opt-iters", type=int, default=8, help="Number of iterative optimization steps.")
    parser.add_argument("--pose-refine-iter", type=int, default=3, help="FoundationPose register refine iterations.")
    parser.add_argument("--track-refine-iter", type=int, default=4, help="Reserved compatibility parameter.")
    parser.add_argument(
        "--upright-lock-angle-deg",
        type=float,
        default=10.0,
        help="World-Z axis locking threshold used by the patched adjust_pose.py.",
    )
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
    parser.add_argument("--skip-eval", action="store_true", help="Do not run quantitative evaluation.")
    parser.add_argument("--skip-render", action="store_true", help="Do not render camera-view overlap images.")
    parser.add_argument(
        "--render-backend",
        type=str,
        choices=["auto", "cpu"],
        default="auto",
        help="Rendering backend for camera-view overlap images.",
    )
    return parser


def locate_reuse_mesh(mesh_root: Path, object_id: int) -> Path:
    candidates = [
        mesh_root / f"result_{object_id:03d}" / f"{object_id}.glb",
        mesh_root / f"{object_id}.glb",
        mesh_root / f"object_{object_id}" / f"{object_id}.glb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = sorted(mesh_root.rglob(f"{object_id}.glb"))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"Could not locate a reused mesh for object {object_id} inside {mesh_root}")


def main() -> None:
    args = build_parser().parse_args()
    prepared_root = args.prepared_fp_root.resolve()
    output_dir = ensure_dir(args.output_dir.resolve())
    prepared_objects = discover_prepared_objects(prepared_root)

    camera_data_dir = (
        args.camera_data_dir.resolve()
        if args.camera_data_dir is not None
        else build_camera_data_from_prepared(
            prepared_objects=prepared_objects,
            output_dir=output_dir / "camera_data_from_prepared",
            depth_scale=args.depth_scale,
            camera_extrinsics_path=(args.camera_extrinsics_path.resolve() if args.camera_extrinsics_path else None),
            gt_pose_json=(args.gt_pose_json.resolve() if args.gt_pose_json else None),
        )
    )

    fp_adjust_path = args.foundationpose_root.resolve() / "adjust_pose.py"
    if not fp_adjust_path.exists():
        raise FileNotFoundError(
            f"Patched FoundationPose adjust_pose.py not found: {fp_adjust_path}\n"
            f"Run scripts/install_foundationpose_patches.py first."
        )

    result_root = ensure_dir(output_dir / "fp_outputs")
    timings = {"per_object_pose_sec": {}}
    extrinsics_path = camera_data_dir / "extrinsics_0.npy"

    for entry in prepared_objects:
        object_id = int(entry["object_id"])
        scene_dir = Path(entry["scene_dir"])
        mesh_path = Path(entry["mesh"])
        if args.reuse_mesh_root is not None:
            mesh_path = locate_reuse_mesh(args.reuse_mesh_root.resolve(), object_id)

        object_result_dir = ensure_dir(result_root / f"object_{object_id}")
        command = split_python_cmd(args.foundationpose_python) + [
            str(fp_adjust_path),
            "--dataset_path",
            str(scene_dir),
            "--mesh_path",
            str(mesh_path),
            "--frame_name",
            "0",
            "--output_dir",
            str(object_result_dir),
            "--opt_iters",
            str(args.opt_iters),
            "--pose_refine_iter",
            str(args.pose_refine_iter),
            "--track_refine_iter",
            str(args.track_refine_iter),
            "--upright_lock_angle_deg",
            str(args.upright_lock_angle_deg),
            "--camera_extrinsics_type",
            args.camera_extrinsics_type,
            "--camera_frame_alignment",
            args.camera_frame_alignment,
        ]
        if extrinsics_path.exists():
            command.extend(["--camera_extrinsics_path", str(extrinsics_path)])

        t0 = time.perf_counter()
        run_subprocess(command, cwd=args.foundationpose_root.resolve())
        timings["per_object_pose_sec"][f"object_{object_id}"] = float(time.perf_counter() - t0)

    if args.gt_spec is not None and not args.skip_eval:
        eval_cmd = split_python_cmd(args.eval_python) + [
            str(REPO_ROOT / "scripts" / "evaluate_reconstruction.py"),
            "--result-root",
            str(result_root),
            "--gt-spec",
            str(args.gt_spec.resolve()),
            "--camera-data-dir",
            str(camera_data_dir),
            "--camera-frame",
            "0",
            "--camera-extrinsics-type",
            args.camera_extrinsics_type,
            "--camera-frame-alignment",
            args.camera_frame_alignment,
        ]
        if args.gt_pose_json is not None:
            eval_cmd.extend(["--gt-pose-json", str(args.gt_pose_json.resolve())])
        run_subprocess(eval_cmd)

    if args.gt_spec is not None and not args.skip_render:
        render_cmd = split_python_cmd(args.eval_python) + [
            str(REPO_ROOT / "scripts" / "render_camera_overlap.py"),
            "--result-root",
            str(result_root),
            "--gt-spec",
            str(args.gt_spec.resolve()),
            "--camera-data-dir",
            str(camera_data_dir),
            "--camera-frame",
            "0",
            "--camera-extrinsics-type",
            args.camera_extrinsics_type,
            "--camera-frame-alignment",
            args.camera_frame_alignment,
            "--render-backend",
            args.render_backend,
            "--foundationpose-root",
            str(args.foundationpose_root.resolve()),
        ]
        if args.gt_pose_json is not None:
            render_cmd.extend(["--gt-pose-json", str(args.gt_pose_json.resolve())])
        run_subprocess(render_cmd)

    summary = {
        "prepared_fp_root": str(prepared_root),
        "camera_data_dir": str(camera_data_dir),
        "result_root": str(result_root),
        "reuse_mesh_root": (str(args.reuse_mesh_root.resolve()) if args.reuse_mesh_root is not None else None),
        "timings_sec": timings,
    }
    if timings["per_object_pose_sec"]:
        values = list(timings["per_object_pose_sec"].values())
        summary["timings_sec"]["mean_pose_sec_per_object"] = float(sum(values) / len(values))

    summary_path = output_dir / "prepared_fp_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== Prepared FP Pipeline Summary ==========")
    print(f"Prepared root:       {prepared_root}")
    print(f"Camera data dir:     {camera_data_dir}")
    print(f"Result root:         {result_root}")
    if args.reuse_mesh_root is not None:
        print(f"Reused mesh root:    {args.reuse_mesh_root.resolve()}")
    for name, sec in sorted(timings["per_object_pose_sec"].items()):
        print(f"{name} pose stage:   {sec:.3f}s")
    if "mean_pose_sec_per_object" in summary["timings_sec"]:
        print(f"Mean pose/object:    {summary['timings_sec']['mean_pose_sec_per_object']:.3f}s")
    print(f"Summary JSON:        {summary_path}")


if __name__ == "__main__":
    main()
