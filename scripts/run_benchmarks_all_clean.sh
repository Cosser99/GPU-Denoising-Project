#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <cpu|cuda|cuda_optimized> [repetitions] [all|mean|gaussian|median|bilateral|fft]"
    exit 1
fi

backend="$1"
repetitions="${2:-100}"
filter_group="${3:-all}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CLEAN_DIR="${ROOT_DIR}/data/clean"
REPORT_DIR="${ROOT_DIR}/reports"

case "${backend}" in
    cpu)
        executable="${BUILD_DIR}/denoising_benchmark_cpu"
        ;;
    cuda)
        executable="${BUILD_DIR}/denoising_benchmark_cuda_reference"
        ;;
    cuda_optimized)
        executable="${BUILD_DIR}/denoising_benchmark_cuda_optimized"
        ;;
    *)
        echo "Unknown backend: ${backend}"
        echo "Use: cpu, cuda, cuda_optimized"
        exit 1
        ;;
esac

cmake -S "${ROOT_DIR}" -B build \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.1 \
    -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel

count=0
while IFS= read -r -d '' image_path
do
    relative_path="${image_path#"${CLEAN_DIR}/"}"
    dimension="$(dirname "${relative_path}")"
    image_name="$(basename "${relative_path}")"
    image_base="${image_name%.*}"

    if [ "${dimension}" = "." ]; then
        dimension="root"
    fi

    output_dir="${REPORT_DIR}/${backend}/${dimension}"
    output_path="${output_dir}/report_${image_base}.csv"

    mkdir -p "${output_dir}"
    echo "Benchmarking ${backend}: ${relative_path} -> ${output_path#"${ROOT_DIR}/"}"
    "${executable}" "${image_path}" "${output_path}" "${repetitions}" "${filter_group}"
    count=$((count + 1))
done < <(find "${CLEAN_DIR}" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print0)

echo "Processed ${count} images"
