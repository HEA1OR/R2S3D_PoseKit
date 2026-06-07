#!/usr/bin/env bash
set -euo pipefail

MODE=""
INPUT_ROOT=""
REUSE_MESH_ROOT=""
SAM3D_ROOT="${SAM3D_ROOT:-/home/ps/xwj/sam-3d-objects}"
FOUNDATIONPOSE_ROOT="${FOUNDATIONPOSE_ROOT:-/home/ps/xwj/FoundationPose}"
SAM3D_ENV="${SAM3D_ENV:-sam3d-objects}"
FOUNDATIONPOSE_ENV="${FOUNDATIONPOSE_ENV:-foundationpose}"
PREP_ENV="${PREP_ENV:-${FOUNDATIONPOSE_ENV}}"
OPT_ITERS="${OPT_ITERS:-8}"
GT_SPEC=""
GT_POSE_JSON=""
MASK_GLOB="${MASK_GLOB:-*.png}"
DEPTH_SCALE="${DEPTH_SCALE:-1000.0}"
GENERATED_WORKSPACE_NAME="${GENERATED_WORKSPACE_NAME:-generated_workspace}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_dataset_pipeline.sh --mode MODE --input-root DIR [options]

Modes:
  from-sam3d   Run SAM3D reconstruction, then iterative FoundationPose optimization.
  skip-sam3d   Skip SAM3D and run iterative FoundationPose optimization.
  full-eval    Run SAM3D reconstruction, iterative FoundationPose optimization, evaluation, and overlap rendering.

Required:
  --mode MODE
  --input-root DIR

Accepted input-root layouts:

1. Prepared/workspace style:
  DIR/
    sam3d_inputs/                optional for skip-sam3d
    prepared_fp_inputs/          or foundationpose_inputs/
    camera_data/                 or evaluation_camera_data/
    sam3d_outputs/               optional
    fp_outputs/                  optional

2. Raw-scene style:
  DIR/
    raw_scene/
      rgb.png
      depth.npy | depth.png
      masks/
      cam_K.txt | intrinsics.npy | intrinsics.txt
      extrinsics.npy             optional
      object_poses_0.json        optional

In raw-scene mode, the script auto-generates a working directory:
  DIR/generated_workspace/

Optional:
  --reuse-mesh-root DIR         Reuse existing reconstructed meshes for skip-sam3d or prepared runs.
  --sam3d-root DIR
  --foundationpose-root DIR
  --sam3d-env NAME
  --foundationpose-env NAME
  --prep-env NAME
  --opt-iters N
  --gt-spec FILE
  --gt-pose-json FILE
  --mask-glob GLOB
  --depth-scale VALUE
  --generated-workspace-name NAME
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --reuse-mesh-root)
      REUSE_MESH_ROOT="$2"
      shift 2
      ;;
    --sam3d-root)
      SAM3D_ROOT="$2"
      shift 2
      ;;
    --foundationpose-root)
      FOUNDATIONPOSE_ROOT="$2"
      shift 2
      ;;
    --sam3d-env)
      SAM3D_ENV="$2"
      shift 2
      ;;
    --foundationpose-env)
      FOUNDATIONPOSE_ENV="$2"
      shift 2
      ;;
    --prep-env)
      PREP_ENV="$2"
      shift 2
      ;;
    --opt-iters)
      OPT_ITERS="$2"
      shift 2
      ;;
    --gt-spec)
      GT_SPEC="$2"
      shift 2
      ;;
    --gt-pose-json)
      GT_POSE_JSON="$2"
      shift 2
      ;;
    --mask-glob)
      MASK_GLOB="$2"
      shift 2
      ;;
    --depth-scale)
      DEPTH_SCALE="$2"
      shift 2
      ;;
    --generated-workspace-name)
      GENERATED_WORKSPACE_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${MODE}" || -z "${INPUT_ROOT}" ]]; then
  usage
  exit 1
fi

case "${MODE}" in
  from-sam3d|skip-sam3d|full-eval)
    ;;
  *)
    echo "Unsupported --mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac

INPUT_ROOT="$(realpath "${INPUT_ROOT}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_SCENE_DIR="${INPUT_ROOT}/raw_scene"
GENERATED_WORKSPACE_DIR="${INPUT_ROOT}/${GENERATED_WORKSPACE_NAME}"

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Required directory not found: ${path}" >&2
    exit 1
  fi
}

prepared_fp_has_meshes() {
  local prepared_root="$1"
  if find "${prepared_root}" -path '*/mesh/*' -type f | grep -q .; then
    return 0
  fi
  return 1
}

resolve_first_existing_dir() {
  for path in "$@"; do
    if [[ -n "${path}" && -d "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

resolve_first_existing_file() {
  for path in "$@"; do
    if [[ -n "${path}" && -f "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

auto_prepare_from_raw_scene() {
  local raw_scene_dir="$1"
  local generated_dir="$2"
  local rgb_path=""
  local depth_path=""
  local intrinsics_path=""
  local extrinsics_path=""
  local gt_pose_json_path=""

  rgb_path="$(resolve_first_existing_file \
    "${raw_scene_dir}/rgb.png" \
    "${raw_scene_dir}/image.png")"
  depth_path="$(resolve_first_existing_file \
    "${raw_scene_dir}/depth.npy" \
    "${raw_scene_dir}/depth.png")"
  intrinsics_path="$(resolve_first_existing_file \
    "${raw_scene_dir}/intrinsics.npy" \
    "${raw_scene_dir}/intrinsics.txt" \
    "${raw_scene_dir}/cam_K.txt")"
  extrinsics_path="$(resolve_first_existing_file \
    "${raw_scene_dir}/extrinsics.npy" \
    "${raw_scene_dir}/extrinsics_0.npy" || true)"
  gt_pose_json_path="$(resolve_first_existing_file \
    "${raw_scene_dir}/object_poses_0.json" \
    "${INPUT_ROOT}/object_poses_0.json" || true)"

  require_dir "${raw_scene_dir}/masks"
  if [[ -z "${rgb_path}" || -z "${depth_path}" || -z "${intrinsics_path}" ]]; then
    echo "raw_scene is missing required files. Need rgb, depth, masks, and intrinsics." >&2
    exit 1
  fi

  mkdir -p "${generated_dir}"
  CMD=(
    conda run -n "${PREP_ENV}" python "${REPO_ROOT}/scripts/prepare_workspace.py"
    --rgb "${rgb_path}"
    --depth "${depth_path}"
    --mask-dir "${raw_scene_dir}/masks"
    --mask-glob "${MASK_GLOB}"
    --intrinsics "${intrinsics_path}"
    --depth-scale "${DEPTH_SCALE}"
    --output-dir "${generated_dir}"
  )
  if [[ -n "${extrinsics_path}" ]]; then
    CMD+=(--extrinsics "${extrinsics_path}")
  fi
  if [[ -n "${gt_pose_json_path}" ]]; then
    CMD+=(--gt-pose-json "${gt_pose_json_path}")
  fi
  "${CMD[@]}"
}

SAM3D_INPUT_DIR="$(resolve_first_existing_dir \
  "${INPUT_ROOT}/sam3d_inputs" \
  "${GENERATED_WORKSPACE_DIR}/sam3d_inputs" || true)"
PREPARED_FP_ROOT="$(resolve_first_existing_dir \
  "${INPUT_ROOT}/prepared_fp_inputs" \
  "${INPUT_ROOT}/foundationpose_inputs" \
  "${GENERATED_WORKSPACE_DIR}/prepared_fp_inputs" \
  "${GENERATED_WORKSPACE_DIR}/foundationpose_inputs" || true)"
CAMERA_DATA_DIR="$(resolve_first_existing_dir \
  "${INPUT_ROOT}/camera_data" \
  "${INPUT_ROOT}/evaluation_camera_data" \
  "${GENERATED_WORKSPACE_DIR}/camera_data" \
  "${GENERATED_WORKSPACE_DIR}/evaluation_camera_data" || true)"

NEEDS_PREP=0
if [[ "${MODE}" == "from-sam3d" || "${MODE}" == "full-eval" ]]; then
  if [[ -z "${SAM3D_INPUT_DIR}" || -z "${PREPARED_FP_ROOT}" || -z "${CAMERA_DATA_DIR}" ]]; then
    NEEDS_PREP=1
  fi
elif [[ "${MODE}" == "skip-sam3d" ]]; then
  if [[ -z "${PREPARED_FP_ROOT}" || -z "${CAMERA_DATA_DIR}" ]]; then
    NEEDS_PREP=1
  fi
fi

if [[ "${NEEDS_PREP}" == "1" ]]; then
  if [[ ! -d "${RAW_SCENE_DIR}" ]]; then
    echo "Missing prepared inputs, and no raw_scene/ directory was found for auto-preparation under ${INPUT_ROOT}" >&2
    exit 1
  fi
  auto_prepare_from_raw_scene "${RAW_SCENE_DIR}" "${GENERATED_WORKSPACE_DIR}"
  SAM3D_INPUT_DIR="$(resolve_first_existing_dir "${GENERATED_WORKSPACE_DIR}/sam3d_inputs")"
  PREPARED_FP_ROOT="$(resolve_first_existing_dir "${GENERATED_WORKSPACE_DIR}/foundationpose_inputs")"
  CAMERA_DATA_DIR="$(resolve_first_existing_dir "${GENERATED_WORKSPACE_DIR}/evaluation_camera_data")"
fi

SAM3D_OUTPUT_DIR="${INPUT_ROOT}/sam3d_outputs"
FP_OUTPUT_BASE="${INPUT_ROOT}/fp_outputs"
mkdir -p "${SAM3D_OUTPUT_DIR}" "${FP_OUTPUT_BASE}"

require_dir "${PREPARED_FP_ROOT}"
require_dir "${CAMERA_DATA_DIR}"

if [[ "${MODE}" != "skip-sam3d" ]]; then
  require_dir "${SAM3D_INPUT_DIR}"
fi

if [[ "${MODE}" == "skip-sam3d" && -z "${REUSE_MESH_ROOT}" ]]; then
  if ! prepared_fp_has_meshes "${PREPARED_FP_ROOT}"; then
    echo "skip-sam3d mode requires meshes inside ${PREPARED_FP_ROOT} or an explicit --reuse-mesh-root." >&2
    exit 1
  fi
fi

if [[ -z "${GT_POSE_JSON}" ]]; then
  GT_POSE_JSON="$(resolve_first_existing_file \
    "${INPUT_ROOT}/object_poses_0.json" \
    "${CAMERA_DATA_DIR}/object_poses_0.json" || true)"
fi

if [[ -z "${GT_SPEC}" ]]; then
  GT_SPEC="$(resolve_first_existing_file \
    "${INPUT_ROOT}/gt_objects_4.json" \
    "${INPUT_ROOT}/gt_objects.json" || true)"
fi

echo "== R2S3D Dataset Pipeline =="
echo "Mode:                ${MODE}"
echo "Input root:          ${INPUT_ROOT}"
echo "SAM3D root:          ${SAM3D_ROOT}"
echo "FoundationPose root: ${FOUNDATIONPOSE_ROOT}"
echo "Prepare env:         ${PREP_ENV}"
echo "SAM3D env:           ${SAM3D_ENV}"
echo "FoundationPose env:  ${FOUNDATIONPOSE_ENV}"
echo "OPT_ITERS:           ${OPT_ITERS}"
echo "SAM3D input dir:     ${SAM3D_INPUT_DIR:-<unused>}"
echo "Prepared FP root:    ${PREPARED_FP_ROOT}"
echo "Camera data dir:     ${CAMERA_DATA_DIR}"
if [[ -n "${REUSE_MESH_ROOT}" ]]; then
  echo "Reuse mesh root:     ${REUSE_MESH_ROOT}"
fi
if [[ -n "${GT_SPEC}" ]]; then
  echo "GT spec:             ${GT_SPEC}"
fi
if [[ -n "${GT_POSE_JSON}" ]]; then
  echo "GT pose JSON:        ${GT_POSE_JSON}"
fi
if [[ -d "${GENERATED_WORKSPACE_DIR}" ]]; then
  echo "Generated workspace: ${GENERATED_WORKSPACE_DIR}"
fi
echo

if [[ "${MODE}" == "from-sam3d" || "${MODE}" == "full-eval" ]]; then
  conda run -n "${SAM3D_ENV}" python "${REPO_ROOT}/scripts/run_sam3d_reconstruction.py" \
    --sam3d-root "${SAM3D_ROOT}" \
    --input-dir "${SAM3D_INPUT_DIR}" \
    --output-dir "${SAM3D_OUTPUT_DIR}"
fi

case "${MODE}" in
  from-sam3d)
    FP_OUTPUT_DIR="${FP_OUTPUT_BASE}/from_sam3d"
    mkdir -p "${FP_OUTPUT_DIR}"
    conda run -n "${FOUNDATIONPOSE_ENV}" python "${REPO_ROOT}/scripts/run_prepared_fp_pipeline.py" \
      --prepared-fp-root "${PREPARED_FP_ROOT}" \
      --reuse-mesh-root "${SAM3D_OUTPUT_DIR}" \
      --output-dir "${FP_OUTPUT_DIR}" \
      --foundationpose-root "${FOUNDATIONPOSE_ROOT}" \
      --foundationpose-python "conda run -n ${FOUNDATIONPOSE_ENV} python" \
      --eval-python "conda run -n ${FOUNDATIONPOSE_ENV} python" \
      --camera-data-dir "${CAMERA_DATA_DIR}" \
      --opt-iters "${OPT_ITERS}"
    ;;
  skip-sam3d)
    FP_OUTPUT_DIR="${FP_OUTPUT_BASE}/skip_sam3d"
    mkdir -p "${FP_OUTPUT_DIR}"
    CMD=(
      conda run -n "${FOUNDATIONPOSE_ENV}" python "${REPO_ROOT}/scripts/run_prepared_fp_pipeline.py"
      --prepared-fp-root "${PREPARED_FP_ROOT}"
      --output-dir "${FP_OUTPUT_DIR}"
      --foundationpose-root "${FOUNDATIONPOSE_ROOT}"
      --foundationpose-python "conda run -n ${FOUNDATIONPOSE_ENV} python"
      --eval-python "conda run -n ${FOUNDATIONPOSE_ENV} python"
      --camera-data-dir "${CAMERA_DATA_DIR}"
      --opt-iters "${OPT_ITERS}"
    )
    if [[ -n "${REUSE_MESH_ROOT}" ]]; then
      CMD+=(--reuse-mesh-root "${REUSE_MESH_ROOT}")
    fi
    "${CMD[@]}"
    ;;
  full-eval)
    if [[ -z "${GT_SPEC}" ]]; then
      echo "full-eval mode requires a GT spec file. Provide --gt-spec or place gt_objects_4.json/gt_objects.json under the input root." >&2
      exit 1
    fi
    FP_OUTPUT_DIR="${FP_OUTPUT_BASE}/full_eval"
    mkdir -p "${FP_OUTPUT_DIR}"
    CMD=(
      conda run -n "${FOUNDATIONPOSE_ENV}" python "${REPO_ROOT}/scripts/run_prepared_fp_pipeline.py"
      --prepared-fp-root "${PREPARED_FP_ROOT}"
      --reuse-mesh-root "${SAM3D_OUTPUT_DIR}"
      --output-dir "${FP_OUTPUT_DIR}"
      --foundationpose-root "${FOUNDATIONPOSE_ROOT}"
      --foundationpose-python "conda run -n ${FOUNDATIONPOSE_ENV} python"
      --eval-python "conda run -n ${FOUNDATIONPOSE_ENV} python"
      --camera-data-dir "${CAMERA_DATA_DIR}"
      --gt-spec "${GT_SPEC}"
      --opt-iters "${OPT_ITERS}"
    )
    if [[ -n "${GT_POSE_JSON}" ]]; then
      CMD+=(--gt-pose-json "${GT_POSE_JSON}")
    fi
    "${CMD[@]}"
    ;;
esac

echo
echo "Completed mode: ${MODE}"
echo "Output root: ${FP_OUTPUT_DIR}"
