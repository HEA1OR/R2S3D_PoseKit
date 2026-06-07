# FoundationPose Patch Notes

This repository ships two files under `patches/foundationpose/` that are meant to replace files inside an installed `FoundationPose` checkout:

- `adjust_pose.py`
- `Utils.py`

## Why these patches exist

### `adjust_pose.py`

This is the iterative mesh scale and pose optimization script used by the project. Compared with a vanilla one-shot longest-edge scaling flow, this version adds:

- repeated scale refinement using projected masks,
- iterative `register()` based pose updates,
- optional world-frame upright-axis locking logic,
- debug outputs and saved pose files in both camera and world coordinates.

The public copy inside this repository has also been cleaned up for open-source release:

- machine-specific absolute default paths were removed,
- `--dataset_path` is now an explicit required argument,
- the default upright lock threshold is set to `10` degrees.

### `Utils.py`

This patched file is included because the workflow depends on the custom mesh-loading path used in your local FoundationPose setup, especially for `.glb` meshes produced by `SAM3D`.

If you do not replace the original `FoundationPose/Utils.py`, the end-to-end pipeline here may fail to load or render reconstructed meshes correctly.

## Installation

Use:

```bash
python scripts/install_foundationpose_patches.py \
  --foundationpose-root /path/to/FoundationPose
```

By default, the installer saves a backup beside each original file before replacement.
