# ETOT Prepared Iter8 Example

This example is built from your local `/home/ps/xwj/FoundationPose/adjust_pose_inputs/etot_prepared_iter8` dataset, but the `object_3 / drawer` branch has been intentionally removed here.

The example therefore keeps 4 objects:

- `object_0` pen
- `object_1` holder_black
- `object_2` tennis
- `object_4` holder_white

## Folder layout

```text
examples/etot_prepared_iter8/
  raw_scene/
    rgb.png
    depth.png
    cam_K.txt
    masks/object_0.png object_1.png object_2.png object_4.png
  sam3d_inputs/
    image.png
    0.png 1.png 2.png 4.png
  prepared_fp_inputs/
    object_0/
    object_1/
    object_2/
    object_4/
  gt_meshes/
    pen.glb
    holder_black.glb
    tennis.glb
    holder_white.glb
  camera_data/
    rgb_0.png
    depth_0.npy
    intrinsics_0.npy
    extrinsics_0.npy
    object_0.png object_1.png object_2.png object_4.png
    object_0_pose_world_abs.npy object_1_pose_world_abs.npy object_2_pose_world_abs.npy object_4_pose_world_abs.npy
    object_poses_0.json
  sam3d_outputs/
  fp_outputs/
```

## What each folder means

- `raw_scene/`
  Generic RGB/depth/mask input view in the open-source interface style.
- `sam3d_inputs/`
  The exact folder layout expected by `scripts/run_sam3d_reconstruction.py`.
- `prepared_fp_inputs/`
  The exact folder layout expected by `scripts/run_prepared_fp_pipeline.py`.
- `gt_meshes/`
  Example GT meshes already converted into standard `glb` files, so evaluation does not need `pxr`.
- `camera_data/`
  Evaluation and world-pose reference data copied from the local camera export, including `object_poses_0.json` and each object's `object_<id>_pose_world_abs.npy`.
- `sam3d_outputs/`
  Target folder for reconstructed meshes exported by SAM3D.
- `fp_outputs/`
  Target folder for iterative scale optimization and pose estimation results.

## Unified bash usage

Use the single generic bash entry in `scripts/` and switch behavior with `--mode`.

For this example, the GT meshes are already stored as `glb`, so the whole flow is intended to run inside just the `sam3d-objects` and `foundationpose` environments.

### Mode 1: run from SAM3D to final

This is the local two-step example:

1. reconstruct four meshes into `sam3d_outputs/`
2. feed those meshes into the iterative FoundationPose stage and save results into `fp_outputs/from_sam3d/`

```bash
bash /home/ps/xwj/R2S3D_PoseKit/scripts/run_dataset_pipeline.sh \
  --mode from-sam3d \
  --input-root /home/ps/xwj/R2S3D_PoseKit/examples/etot_prepared_iter8
```

### Mode 2: skip SAM3D and run only scale+pose optimization

This route directly reuses the example meshes already stored in `prepared_fp_inputs/object_*/mesh/`.

```bash
bash /home/ps/xwj/R2S3D_PoseKit/scripts/run_dataset_pipeline.sh \
  --mode skip-sam3d \
  --input-root /home/ps/xwj/R2S3D_PoseKit/examples/etot_prepared_iter8
```

### Mode 3: full run with mesh quality evaluation

This is the most complete mode. It runs:

1. `SAM3D` reconstruction
2. iterative FoundationPose scale and pose optimization
3. quantitative evaluation
4. camera-view overlap rendering

```bash
bash /home/ps/xwj/R2S3D_PoseKit/scripts/run_dataset_pipeline.sh \
  --mode full-eval \
  --input-root /home/ps/xwj/R2S3D_PoseKit/examples/etot_prepared_iter8
```

This full route produces:

- optimized meshes and poses in `fp_outputs/full_eval/fp_outputs/`
- metric JSON from `scripts/evaluate_reconstruction.py`
- camera-view overlap images from `scripts/render_camera_overlap.py`

## Notes

- The example folders already contain real local files, not placeholders.
- `camera_data/object_poses_0.json` and `camera_data/object_<id>_pose_world_abs.npy` provide the saved world-coordinate pose references for the 4 kept objects.
- `prepared_fp_inputs/` is still sizable because it includes four `*.glb` meshes.
- If you later want a lighter public repo, this example can be trimmed or moved to Git LFS.
