# R2S3D PoseKit

`R2S3D PoseKit` is the lightweight open-source layer around `SAM3D` and `FoundationPose` for:

- preparing raw `RGB + depth + per-object mask` inputs,
- reconstructing one mesh per object with `SAM3D`,
- running iterative anisotropic scale optimization plus pose estimation with a patched `FoundationPose`,
- evaluating and visualizing the final reconstruction in camera and world coordinates.

This repository does **not** vendor `SAM3D` or `FoundationPose`. Install those two projects separately, then use the scripts here as the glue code.

## What lives here

- `scripts/prepare_workspace.py`
  Converts generic `rgb/depth/mask` inputs into the directory layouts expected by `SAM3D` and `FoundationPose`.
- `scripts/run_sam3d_reconstruction.py`
  Reconstructs a `*.glb` mesh for each object mask from a prepared SAM3D input folder.
- `scripts/install_foundationpose_patches.py`
  Copies the patched `adjust_pose.py` and `Utils.py` into an installed `FoundationPose` checkout.
- `scripts/run_pipeline.py`
  End-to-end entry point from raw input images to final optimized results.
- `scripts/run_prepared_fp_pipeline.py`
  Runs only the iterative FoundationPose scale+pose stage from an already prepared `object_0 ... object_N` input root, with optional mesh replacement from fresh SAM3D outputs.
- `scripts/run_dataset_pipeline.sh`
  Unified bash entry for generic dataset folders. It accepts either a prepared layout or a raw-scene layout, and switches behavior with `--mode from-sam3d`, `--mode skip-sam3d`, or `--mode full-eval`.
- `scripts/evaluate_reconstruction.py`
  Computes pose error, 3D AABB IoU, voxel volume IoU, and camera-view mask IoU.
- `scripts/render_camera_overlap.py`
  Produces qualitative overlay images comparing GT and reconstructed meshes in the camera view.
- `examples/etot_prepared_iter8/`
  A concrete four-object local example containing `raw_scene`, `sam3d_inputs`, `prepared_fp_inputs`, `gt_meshes`, `camera_data`, and placeholder output folders.
- `patches/foundationpose/adjust_pose.py`
  Patched iterative scale-and-pose optimizer.
- `patches/foundationpose/Utils.py`
  Patched FoundationPose utility file that includes the `.glb` support used in this workflow.

## Input format

The public interface is intentionally simple:

- one RGB image: `rgb.png`
- one depth image: `depth.png` or `depth.npy`
- one mask PNG per object inside a folder
  - accepted names include `0.png`, `1.png`, ...
  - accepted names also include `object_0.png`, `object_1.png`, ...
- intrinsics:
  - either a `3x3` matrix file via `--intrinsics`
  - or `--fx --fy --cx --cy`

Optional:

- camera extrinsics matrix via `--extrinsics`
- runtime GT pose snapshot via `--gt-pose-json`
- GT object specification JSON via `--gt-spec`

## Install flow

1. Install `SAM3D` in its own environment.
2. Install `FoundationPose` in its own environment.
3. Clone this repository.
4. Install this repository's Python dependencies in the environment you want to use for preparation, evaluation, and visualization:

```bash
pip install -e .
```

If your GT meshes are `USD`, use:

```bash
pip install -e .[usd]
```

5. Copy the patched files into `FoundationPose`:

```bash
python scripts/install_foundationpose_patches.py \
  --foundationpose-root /path/to/FoundationPose
```

`Utils.py` is intentionally included here because the workflow depends on the `.glb` mesh support in that patched file. Replace the original `FoundationPose/Utils.py` with this version before running the pipeline.

## Quick start

### 1. Prepare a workspace

```bash
python scripts/prepare_workspace.py \
  --rgb /path/to/rgb.png \
  --depth /path/to/depth.npy \
  --mask-dir /path/to/masks \
  --intrinsics /path/to/intrinsics.npy \
  --extrinsics /path/to/extrinsics.npy \
  --output-dir /path/to/workspace
```

### 2. Run the full pipeline

You can pass a plain interpreter path or a full launcher command. This makes it easy to work with separate conda environments.

```bash
python scripts/run_pipeline.py \
  --rgb /path/to/rgb.png \
  --depth /path/to/depth.npy \
  --mask-dir /path/to/masks \
  --intrinsics /path/to/intrinsics.npy \
  --extrinsics /path/to/extrinsics.npy \
  --workspace-dir /path/to/workspace \
  --sam3d-root /path/to/sam-3d-objects \
  --foundationpose-root /path/to/FoundationPose \
  --sam3d-python "conda run -n sam3d-objects python" \
  --foundationpose-python "conda run -n foundationpose python" \
  --eval-python "conda run -n foundationpose python" \
  --opt-iters 8
```

### 3. Run evaluation and qualitative rendering separately

```bash
python scripts/evaluate_reconstruction.py \
  --result-root /path/to/workspace/results \
  --gt-spec configs/example_gt_objects.json \
  --camera-data-dir /path/to/workspace/evaluation_camera_data
```

```bash
python scripts/render_camera_overlap.py \
  --result-root /path/to/workspace/results \
  --gt-spec configs/example_gt_objects.json \
  --camera-data-dir /path/to/workspace/evaluation_camera_data \
  --foundationpose-root /path/to/FoundationPose
```

### 4. Run only the prepared FoundationPose stage

This mode is useful when you already have per-object folders such as `object_0`, `object_1`, ... that contain `rgb/depth/masks/mesh/cam_K.txt`.

```bash
python scripts/run_prepared_fp_pipeline.py \
  --prepared-fp-root /path/to/prepared_fp_inputs \
  --output-dir /path/to/prepared_fp_run \
  --foundationpose-root /path/to/FoundationPose \
  --foundationpose-python "conda run -n foundationpose python" \
  --camera-extrinsics-path /path/to/extrinsics.npy \
  --opt-iters 8
```

If you also want to replace the prepared meshes with fresh SAM3D outputs before optimization:

```bash
python scripts/run_prepared_fp_pipeline.py \
  --prepared-fp-root /path/to/prepared_fp_inputs \
  --reuse-mesh-root /path/to/sam3d_outputs \
  --output-dir /path/to/prepared_fp_run \
  --foundationpose-root /path/to/FoundationPose \
  --foundationpose-python "conda run -n foundationpose python" \
  --camera-extrinsics-path /path/to/extrinsics.npy \
  --opt-iters 8
```

### 5. Unified bash entry for generic dataset folders

The bash wrapper accepts either a prepared layout:

```text
input_root/
  sam3d_inputs/
  prepared_fp_inputs/         or foundationpose_inputs/
  camera_data/                or evaluation_camera_data/
  sam3d_outputs/
  fp_outputs/
```

or a raw-scene layout:

```text
input_root/
  raw_scene/
    rgb.png
    depth.npy | depth.png
    masks/
    cam_K.txt | intrinsics.npy | intrinsics.txt
    extrinsics.npy            optional
    object_poses_0.json       optional
```

When only `raw_scene/` exists, the script auto-generates a working directory under `input_root/generated_workspace/`.

To keep the open-source setup lightweight, the default path assumes evaluation GT meshes are in standard mesh formats such as `glb`, `obj`, `ply`, `stl`, or `off`, so the same `foundationpose` environment can run both optimization and evaluation.

You can use the unified bash wrapper in `scripts/`:

Run SAM3D and then optimize:

```bash
bash scripts/run_dataset_pipeline.sh \
  --mode from-sam3d \
  --input-root /path/to/input_root
```

Skip SAM3D and directly optimize:

```bash
bash scripts/run_dataset_pipeline.sh \
  --mode skip-sam3d \
  --input-root /path/to/input_root
```

Run the full route with quantitative evaluation and overlap rendering:

```bash
bash scripts/run_dataset_pipeline.sh \
  --mode full-eval \
  --input-root /path/to/input_root
```

## Workspace layout

After preparation and running, the workspace is organized like this:

```text
workspace/
  manifest.json
  sam3d_inputs/
    image.png
    0.png
    1.png
    ...
  foundationpose_inputs/
    object_0/
      rgb/0.png
      depth/0.png
      masks/0.png
      mesh/
    object_1/
      ...
  evaluation_camera_data/
    rgb_0.png
    depth_0.png
    intrinsics_0.npy
    extrinsics_0.npy
    object_0.png
    object_1.png
    ...
  sam3d_outputs/
  results/
    object_0/
    object_1/
    ...
```

## Local example

See [examples/etot_prepared_iter8/README.md](/home/ps/xwj/R2S3D_PoseKit/examples/etot_prepared_iter8/README.md).

That example is based on your local prepared input set, with one object branch intentionally removed so the published example keeps 4 objects, and is organized like this:

```text
examples/etot_prepared_iter8/
  raw_scene/
  sam3d_inputs/
  prepared_fp_inputs/
  gt_meshes/
  camera_data/
  sam3d_outputs/
  fp_outputs/
```

For that concrete example, the three bash usages are:

```bash
bash /home/ps/xwj/R2S3D_PoseKit/scripts/run_dataset_pipeline.sh \
  --mode from-sam3d \
  --input-root /home/ps/xwj/R2S3D_PoseKit/examples/etot_prepared_iter8
```

```bash
bash /home/ps/xwj/R2S3D_PoseKit/scripts/run_dataset_pipeline.sh \
  --mode skip-sam3d \
  --input-root /home/ps/xwj/R2S3D_PoseKit/examples/etot_prepared_iter8
```

```bash
bash /home/ps/xwj/R2S3D_PoseKit/scripts/run_dataset_pipeline.sh \
  --mode full-eval \
  --input-root /home/ps/xwj/R2S3D_PoseKit/examples/etot_prepared_iter8
```

## GT spec format

See `configs/example_gt_objects.json`.

Each object entry needs:

- `object_id`
- `name`
- `mesh_path`
- `scale_xyz`
- `pos_xyz`
- `quat_wxyz`

If you also provide `--gt-pose-json`, the runtime world pose inside that JSON overrides `pos_xyz` and `quat_wxyz`.

## Notes

- `patches/foundationpose/adjust_pose.py` in this repository has its local absolute defaults removed for public use.
- The default public pipeline uses `--opt-iters 8`, while the patched FoundationPose script itself keeps its own CLI defaults.
- `render_camera_overlap.py` falls back to a CPU renderer if `nvdiffrast` or the patched FoundationPose utilities are unavailable.
- For the default two-environment setup, use GT meshes in standard formats such as `glb`, `obj`, `ply`, `stl`, or `off`. USD GT meshes remain optional, but they require `pxr` in the runtime environment.
- See [NOTICE.md](/home/ps/xwj/R2S3D_PoseKit/NOTICE.md) before publishing. It calls out the FoundationPose-derived patch files and the included local example assets.
