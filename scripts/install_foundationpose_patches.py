#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = REPO_ROOT / "patches" / "foundationpose"
PATCH_FILES = ["adjust_pose.py", "Utils.py"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy the patched FoundationPose files from this repo into an installed FoundationPose checkout."
    )
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        required=True,
        help="Path to the installed FoundationPose repository.",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".bak_r2s3d",
        help="Suffix used when backing up existing files before replacement.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Replace target files directly without keeping a local backup.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target_root = args.foundationpose_root.resolve()
    if not target_root.exists():
        raise FileNotFoundError(f"FoundationPose root not found: {target_root}")

    print(f"Patch source:  {PATCH_DIR}")
    print(f"Target root:   {target_root}\n")

    for filename in PATCH_FILES:
        src = PATCH_DIR / filename
        dst = target_root / filename
        if not src.exists():
            raise FileNotFoundError(f"Patch file not found: {src}")
        if dst.exists() and not args.no_backup:
            backup_path = dst.with_name(f"{dst.name}{args.backup_suffix}")
            shutil.copy2(dst, backup_path)
            print(f"Backed up {dst.name} -> {backup_path.name}")
        shutil.copy2(src, dst)
        print(f"Installed {filename} -> {dst}")

    print("\nFoundationPose patch install finished.")


if __name__ == "__main__":
    main()
