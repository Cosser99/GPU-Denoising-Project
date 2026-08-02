#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <noisy-image> [cpu|cuda_reference|cuda_optimized] [all|mean|gaussian|median|bilateral|fft]"
    exit 1
fi

IMAGE_PATH="$1"
BACKEND="${2:-cuda_optimized}"
FILTER_GROUP="${3:-all}"

case "${BACKEND}" in
    cpu)
        TARGET="generate_filtered_cpu"
        ;;
    cuda_reference)
        TARGET="generate_filtered_cuda_reference"
        ;;
    cuda_optimized)
        TARGET="generate_filtered_cuda_optimized"
        ;;
    *)
        echo "Unknown backend: ${BACKEND}"
        echo "Usage: $0 <noisy-image> [cpu|cuda_reference|cuda_optimized] [all|mean|gaussian|median|bilateral|fft]"
        exit 1
        ;;
esac

case "${FILTER_GROUP}" in
    all|mean|gaussian|median|bilateral|fft)
        ;;
    *)
        echo "Unknown filter group: ${FILTER_GROUP}"
        echo "Usage: $0 <noisy-image> [cpu|cuda_reference|cuda_optimized] [all|mean|gaussian|median|bilateral|fft]"
        exit 1
        ;;
esac

GENERATE_FILTERED="${BUILD_DIR}/${TARGET}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target "${TARGET}"

echo "Generating filtered images: ${IMAGE_PATH}"
"${GENERATE_FILTERED}" "${IMAGE_PATH}" "${FILTER_GROUP}"
echo "Completed with ${TARGET}"
