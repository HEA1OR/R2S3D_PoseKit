#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def clear_proxy_env() -> None:
    for key in [
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    ]:
        os.environ.pop(key, None)


def parse_mask_ids(input_dir: Path) -> list[int]:
    mask_ids = []
    for path in sorted(input_dir.glob("*.png")):
        if path.stem == "image":
            continue
        try:
            mask_ids.append(int(path.stem))
        except ValueError:
            continue
    if not mask_ids:
        raise FileNotFoundError(f"No integer-named mask PNGs found in {input_dir}")
    return mask_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM3D reconstruction from a prepared image+mask directory."
    )
    parser.add_argument(
        "--sam3d-root",
        type=Path,
        required=True,
        help="Path to the installed SAM3D repository.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Prepared input directory containing image.png and object masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where result_000/... or exported meshes will be written.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional SAM3D pipeline.yaml. Default: <sam3d-root>/checkpoints/hf/pipeline.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed passed into SAM3D inference.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable SAM3D compile mode if desired.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sam3d_root = args.sam3d_root.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = (
        args.config.resolve()
        if args.config is not None
        else (sam3d_root / "checkpoints" / "hf" / "pipeline.yaml").resolve()
    )

    clear_proxy_env()
    notebook_dir = sam3d_root / "notebook"
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))

    os.chdir(sam3d_root)

    from inference import Inference, load_image, load_masks  # type: ignore

    if not config_path.exists():
        raise FileNotFoundError(f"SAM3D config not found: {config_path}")

    image_path = input_dir / "image.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Prepared SAM3D image not found: {image_path}")

    image = load_image(str(image_path))
    mask_ids = parse_mask_ids(input_dir)
    masks = load_masks(str(input_dir), mask_ids, extension=".png")
    inference = Inference(str(config_path), compile=bool(args.compile))

    results = {
        "sam3d_root": str(sam3d_root),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config": str(config_path),
        "objects": [],
    }

    total_infer = 0.0
    total_export = 0.0
    print("\n========== SAM3D Reconstruction ==========\n")
    for object_id, mask in zip(mask_ids, masks):
        infer_t0 = time.perf_counter()
        output = inference(image, mask, seed=args.seed)
        infer_dt = time.perf_counter() - infer_t0

        result_dir = output_dir / f"result_{object_id:03d}"
        result_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = result_dir / f"{object_id}.glb"

        export_t0 = time.perf_counter()
        output["glb"].export(mesh_path)
        export_dt = time.perf_counter() - export_t0

        total_infer += infer_dt
        total_export += export_dt
        results["objects"].append(
            {
                "object_id": int(object_id),
                "mesh_path": str(mesh_path),
                "inference_sec": float(infer_dt),
                "mesh_export_sec": float(export_dt),
                "total_sec": float(infer_dt + export_dt),
            }
        )
        print(
            f"object_{object_id:02d} | "
            f"inference={infer_dt:.3f}s | export={export_dt:.3f}s | total={infer_dt + export_dt:.3f}s"
        )

    object_count = max(len(results["objects"]), 1)
    results["total_inference_sec"] = float(total_infer)
    results["total_mesh_export_sec"] = float(total_export)
    results["total_reconstruction_sec"] = float(total_infer + total_export)
    results["mean_per_object_sec"] = float((total_infer + total_export) / object_count)

    summary_path = output_dir / "sam3d_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n==========================================")
    print(f"Total inference:       {total_infer:.3f}s")
    print(f"Total mesh export:     {total_export:.3f}s")
    print(f"Total reconstruction:  {total_infer + total_export:.3f}s")
    print(f"Mean per object:       {(total_infer + total_export) / object_count:.3f}s")
    print(f"Saved summary to:      {summary_path}")


if __name__ == "__main__":
    main()
