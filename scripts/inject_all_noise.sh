#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CLEAN_DIR="${ROOT_DIR}/data/clean"
INJECT_NOISE="${BUILD_DIR}/inject_noise"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target inject_noise

count=0
while IFS= read -r -d '' image_path
do
    relative_path="${image_path#"${CLEAN_DIR}/"}"
    echo "Injecting noise: ${relative_path}"
    "${INJECT_NOISE}" "${relative_path}"
    count=$((count + 1))
done < <(find "${CLEAN_DIR}" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print0)

echo "Processed ${count} images"
