#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from r2s3d_posekit.input_prep import infer_mask_map, load_intrinsics, prepare_multi_object_workspace


def split_python_cmd(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("Python command must not be empty.")
    return parts


def run_subprocess(command: list[str], cwd: Path | None = None) -> None:
    print("\n$ " + " ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True, cwd=(str(cwd) if cwd is not None else None))


def locate_mesh(mesh_root: Path, object_id: int) -> Path:
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
    raise FileNotFoundError(f"Could not locate mesh for object {object_id} inside {mesh_root}")


def prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end RGB/depth/mask -> SAM3D reconstruction -> FoundationPose iterative optimization pipeline."
    )
    parser.add_argument("--rgb", type=Path, required=True, help="RGB image path.")
    parser.add_argument("--depth", type=Path, required=True, help="Depth image path (*.png or *.npy).")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory containing per-object mask PNGs such as 0.png or object_0.png.",
    )
    parser.add_argument("--workspace-dir", type=Path, required=True, help="Workspace directory for all generated files.")
    parser.add_argument("--sam3d-root", type=Path, required=True, help="Installed SAM3D repository root.")
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        required=True,
        help="Installed FoundationPose repository root, already patched with this repo's adjust_pose.py and Utils.py.",
    )
    parser.add_argument("--intrinsics", type=Path, default=None, help="Optional camera intrinsics file.")
    parser.add_argument("--fx", type=float, default=None, help="Camera fx if --intrinsics is not given.")
    parser.add_argument("--fy", type=float, default=None, help="Camera fy if --intrinsics is not given.")
    parser.add_argument("--cx", type=float, default=None, help="Camera cx if --intrinsics is not given.")
    parser.add_argument("--cy", type=float, default=None, help="Camera cy if --intrinsics is not given.")
    parser.add_argument("--mask-glob", type=str, default="*.png", help="Mask glob inside --mask-dir.")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth multiplier used for uint16 export.")
    parser.add_argument(
        "--extrinsics",
        type=Path,
        default=None,
        help="Optional camera extrinsics matrix saved into the evaluation camera directory and passed into FoundationPose.",
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
    parser.add_argument(
        "--gt-spec",
        type=Path,
        default=None,
        help="Optional GT object spec JSON. Enables evaluation and camera-view overlap rendering.",
    )
    parser.add_argument(
        "--gt-pose-json",
        type=Path,
        default=None,
        help="Optional runtime GT pose snapshot JSON used to override poses from --gt-spec.",
    )
    parser.add_argument(
        "--sam3d-python",
        type=str,
        default="python",
        help="Python launcher for SAM3D, for example 'python' or 'conda run -n sam3d-objects python'.",
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
        help="Python launcher for evaluation/render scripts.",
    )
    parser.add_argument(
        "--reuse-mesh-root",
        type=Path,
        default=None,
        help="Skip SAM3D and reuse existing reconstructed meshes from this directory.",
    )
    parser.add_argument("--skip-sam3d", action="store_true", help="Skip SAM3D and reuse meshes from --reuse-mesh-root or workspace.")
    parser.add_argument("--skip-pose", action="store_true", help="Only prepare inputs and optionally run SAM3D.")
    parser.add_argument("--skip-eval", action="store_true", help="Do not run quantitative evaluation.")
    parser.add_argument("--skip-render", action="store_true", help="Do not render camera-view overlap images.")
    parser.add_argument("--prepare-only", action="store_true", help="Stop after workspace preparation.")
    parser.add_argument("--opt-iters", type=int, default=8, help="Number of iterative scale optimization steps.")
    parser.add_argument("--pose-refine-iter", type=int, default=3, help="FoundationPose register refine iterations.")
    parser.add_argument("--track-refine-iter", type=int, default=4, help="Reserved compatibility parameter.")
    parser.add_argument(
        "--upright-lock-angle-deg",
        type=float,
        default=10.0,
        help="World-Z axis locking threshold used by the patched adjust_pose.py.",
    )
    parser.add_argument(
        "--render-backend",
        type=str,
        choices=["auto", "cpu"],
        default="auto",
        help="Camera-view mesh overlap rendering backend.",
    )
    return parser


def main() -> None:
    args = prepare_parser().parse_args()
    workspace_dir = args.workspace_dir.resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    K = load_intrinsics(
        intrinsics_path=args.intrinsics,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
    )
    mask_map = infer_mask_map(args.mask_dir, mask_glob=args.mask_glob)
    object_ids = sorted(mask_map)

    timings = {
        "prepare_sec": 0.0,
        "sam3d_sec": None,
        "per_object_pose_sec": {},
        "evaluation_sec": None,
        "render_sec": None,
    }

    t0 = time.perf_counter()
    manifest = prepare_multi_object_workspace(
        rgb_path=args.rgb,
        depth_path=args.depth,
        mask_map=mask_map,
        output_dir=workspace_dir,
        K=K,
        depth_scale=args.depth_scale,
        extrinsics_path=args.extrinsics,
        gt_pose_json=args.gt_pose_json,
    )
    timings["prepare_sec"] = float(time.perf_counter() - t0)
    manifest_path = workspace_dir / "manifest.json"
    print(f"Prepared workspace manifest: {manifest_path}")

    if args.prepare_only:
        print("Preparation completed; stopping because --prepare-only was set.")
        return

    sam3d_output_dir = workspace_dir / "sam3d_outputs"
    if not args.skip_sam3d:
        t0 = time.perf_counter()
        run_subprocess(
            split_python_cmd(args.sam3d_python)
            + [
                str(REPO_ROOT / "scripts" / "run_sam3d_reconstruction.py"),
                "--sam3d-root",
                str(args.sam3d_root.resolve()),
                "--input-dir",
                str(Path(manifest["sam3d_input_dir"]).resolve()),
                "--output-dir",
                str(sam3d_output_dir),
            ]
        )
        timings["sam3d_sec"] = float(time.perf_counter() - t0)
    else:
        sam3d_output_dir = (
            args.reuse_mesh_root.resolve()
            if args.reuse_mesh_root is not None
            else (workspace_dir / "sam3d_outputs").resolve()
        )
        print(f"Skipping SAM3D; reusing meshes from {sam3d_output_dir}")

    if args.skip_pose:
        print("Pose stage skipped because --skip-pose was set.")
        return

    fp_adjust_path = args.foundationpose_root.resolve() / "adjust_pose.py"
    if not fp_adjust_path.exists():
        raise FileNotFoundError(
            f"Patched FoundationPose adjust_pose.py not found: {fp_adjust_path}\n"
            f"Run scripts/install_foundationpose_patches.py first."
        )

    result_root = workspace_dir / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    camera_data_dir = Path(manifest["evaluation_camera_data_dir"]).resolve()

    for object_entry in manifest["objects"]:
        object_id = int(object_entry["object_id"])
        scene_dir = Path(object_entry["foundationpose_scene_dir"]).resolve()
        mesh_path = locate_mesh(sam3d_output_dir, object_id)
        target_mesh_path = scene_dir / "mesh" / f"{object_id}.glb"
        target_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mesh_path, target_mesh_path)

        object_result_dir = result_root / f"object_{object_id}"
        object_result_dir.mkdir(parents=True, exist_ok=True)

        command = split_python_cmd(args.foundationpose_python) + [
            str(fp_adjust_path),
            "--dataset_path",
            str(scene_dir),
            "--mesh_path",
            str(target_mesh_path),
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
        extrinsics_path = camera_data_dir / "extrinsics_0.npy"
        if extrinsics_path.exists():
            command.extend(["--camera_extrinsics_path", str(extrinsics_path)])

        t0 = time.perf_counter()
        run_subprocess(command, cwd=args.foundationpose_root.resolve())
        timings["per_object_pose_sec"][f"object_{object_id}"] = float(time.perf_counter() - t0)

    if args.gt_spec is not None and not args.skip_eval:
        t0 = time.perf_counter()
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
        timings["evaluation_sec"] = float(time.perf_counter() - t0)

    if args.gt_spec is not None and not args.skip_render:
        t0 = time.perf_counter()
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
        ]
        if args.gt_pose_json is not None:
            render_cmd.extend(["--gt-pose-json", str(args.gt_pose_json.resolve())])
        if args.foundationpose_root is not None:
            render_cmd.extend(["--foundationpose-root", str(args.foundationpose_root.resolve())])
        run_subprocess(render_cmd)
        timings["render_sec"] = float(time.perf_counter() - t0)

    summary = {
        "workspace_dir": str(workspace_dir),
        "object_ids": object_ids,
        "manifest_path": str(manifest_path),
        "result_root": str(result_root),
        "sam3d_output_dir": str(sam3d_output_dir),
        "timings_sec": timings,
    }
    if timings["per_object_pose_sec"]:
        pose_values = list(timings["per_object_pose_sec"].values())
        summary["timings_sec"]["mean_pose_sec_per_object"] = float(sum(pose_values) / len(pose_values))

    summary_path = workspace_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== Pipeline Summary ==========")
    print(f"Workspace:           {workspace_dir}")
    print(f"Objects:             {object_ids}")
    print(f"Prepare:             {timings['prepare_sec']:.3f}s")
    if timings["sam3d_sec"] is not None:
        print(f"SAM3D total:         {timings['sam3d_sec']:.3f}s")
    if timings["per_object_pose_sec"]:
        for name, sec in sorted(timings["per_object_pose_sec"].items()):
            print(f"{name} pose stage:   {sec:.3f}s")
        print(f"Mean pose/object:    {summary['timings_sec']['mean_pose_sec_per_object']:.3f}s")
    if timings["evaluation_sec"] is not None:
        print(f"Evaluation:          {timings['evaluation_sec']:.3f}s")
    if timings["render_sec"] is not None:
        print(f"Render overlap:      {timings['render_sec']:.3f}s")
    print(f"Summary JSON:        {summary_path}")


if __name__ == "__main__":
    main()
