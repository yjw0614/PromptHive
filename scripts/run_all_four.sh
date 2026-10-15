#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
MODE="${MODE:-pt-strict}"
DEVICE="${DEVICE:-cuda:1}"
LIMIT="${LIMIT:-0}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-/datanas01/nas01/Student-home/2024U/LLS/WSSS_SAM/pretrained/sam_vit_l_0b3195.pth}"
DATA_ROOT="${DATA_ROOT:-/LAI_Data/2024U/Data/FewShot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/results/${MODE}}"

run_dataset() {
  local name="$1"
  local bundle_dir="$2"
  local image_dir="$3"
  local mask_dir="$4"

  "${PYTHON}" "${PACKAGE_ROOT}/run_eval.py" \
    --mode "${MODE}" \
    --bundle-dir "${bundle_dir}" \
    --image-dir "${image_dir}" \
    --mask-dir "${mask_dir}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --output-dir "${OUTPUT_ROOT}/${name}" \
    --device "${DEVICE}" \
    --limit "${LIMIT}"
}

find_data_dir() {
  local parent="$1"
  local pattern="$2"
  local match
  match="$(find "${parent}" -mindepth 1 -maxdepth 1 -type d -name "${pattern}" -print -quit)"
  if [[ -z "${match}" ]]; then
    echo "No directory matching ${pattern} under ${parent}" >&2
    return 1
  fi
  printf '%s\n' "${match}"
}

FSS_PROTOCOL="${FSS_PROTOCOL:-$(find_data_dir "${DATA_ROOT}/FSS1000" '*_protocol')}"
FSS_BUNDLE="${FSS_BUNDLE:-$(find_data_dir "${FSS_PROTOCOL}" 'features_*_vitg14')}"
VOC_PROTOCOL="${VOC_PROTOCOL:-$(find_data_dir "${DATA_ROOT}/VOC2012" '*_protocol')}"
VOC_BUNDLE="${VOC_BUNDLE:-$(find_data_dir "${VOC_PROTOCOL}" 'features_*_vitg14')}"
ISIC_BASE="${ISIC_BASE:-${DATA_ROOT}/ISIC/ISIC}"
ISIC_BUNDLE="${ISIC_BUNDLE:-$(find_data_dir "${ISIC_BASE}" 'features_*')}"
KVASIR_BASE="${KVASIR_BASE:-${DATA_ROOT}/Kvasir/Kvasir}"
KVASIR_BUNDLE="${KVASIR_BUNDLE:-$(find_data_dir "${KVASIR_BASE}" 'features_*')}"

run_dataset fss1000 \
  "${FSS_BUNDLE}" \
  "${FSS_PROTOCOL}/target_images" \
  "${FSS_PROTOCOL}/target_masks"

run_dataset voc2012 \
  "${VOC_BUNDLE}" \
  "${VOC_PROTOCOL}/target_images" \
  "${VOC_PROTOCOL}/target_masks"

run_dataset isic \
  "${ISIC_BUNDLE}" \
  "${ISIC_BASE}/target_images" \
  "${ISIC_BASE}/target_masks"

run_dataset kvasir \
  "${KVASIR_BUNDLE}" \
  "${KVASIR_BASE}/target_images" \
  "${KVASIR_BASE}/target_masks"
