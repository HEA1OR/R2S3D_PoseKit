#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from r2s3d_posekit.input_prep import infer_mask_map, load_intrinsics, prepare_multi_object_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a generic RGB/depth/mask scene into SAM3D and FoundationPose workspaces."
    )
    parser.add_argument("--rgb", type=Path, required=True, help="RGB image path, usually *.png.")
    parser.add_argument(
        "--depth",
        type=Path,
        required=True,
        help="Depth image path, either *.png or *.npy.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory containing per-object mask PNGs such as 0.png or object_0.png.",
    )
    parser.add_argument(
        "--mask-glob",
        type=str,
        default="*.png",
        help="Glob used to collect masks inside --mask-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Workspace directory to create.",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help="Optional intrinsics file containing a 3x3 K matrix or 9 flattened values.",
    )
    parser.add_argument("--fx", type=float, default=None, help="Camera fx if --intrinsics is not given.")
    parser.add_argument("--fy", type=float, default=None, help="Camera fy if --intrinsics is not given.")
    parser.add_argument("--cx", type=float, default=None, help="Camera cx if --intrinsics is not given.")
    parser.add_argument("--cy", type=float, default=None, help="Camera cy if --intrinsics is not given.")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Multiplier used when saving float depth to uint16 PNG for downstream tools.",
    )
    parser.add_argument(
        "--extrinsics",
        type=Path,
        default=None,
        help="Optional world_T_camera or camera_T_world matrix to store for later world-pose conversion.",
    )
    parser.add_argument(
        "--gt-pose-json",
        type=Path,
        default=None,
        help="Optional runtime GT pose JSON copied into the evaluation camera data directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    K = load_intrinsics(
        intrinsics_path=args.intrinsics,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
    )
    mask_map = infer_mask_map(args.mask_dir, mask_glob=args.mask_glob)
    manifest = prepare_multi_object_workspace(
        rgb_path=args.rgb,
        depth_path=args.depth,
        mask_map=mask_map,
        output_dir=args.output_dir,
        K=K,
        depth_scale=args.depth_scale,
        extrinsics_path=args.extrinsics,
        gt_pose_json=args.gt_pose_json,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nPrepared workspace: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
