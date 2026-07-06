#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 [repetitions] [all|mean|gaussian|median|bilateral|fft]"
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_BENCHMARKS="${ROOT_DIR}/scripts/run_benchmarks_all_clean.sh"

repetitions="${1:-100}"
filter_group="${2:-all}"

for backend in cpu cuda cuda_optimized
do
    echo
    echo "Running ${backend} report..."
    "${RUN_BENCHMARKS}" "${backend}" "${repetitions}" "${filter_group}"
done

echo
echo "All reports completed."
echo "Reports saved under: ${ROOT_DIR}/reports"
