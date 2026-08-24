#pragma once

#include <cstdint>

#include "Filter.h"
#include "Types.h"

namespace idl
{
    template <typename PixelT>
    class Benchmark
    {
        public:
            static_assert(std::is_arithmetic_v<PixelT>, "PixelT must be an arithmetic type");

            BenchmarkResult run(
                const Filter<PixelT>& filter,
                const Image<PixelT>& input,
                const Image<PixelT>& original,
                uint32_t repetitions = 1
            );
    };
}

#include "Benchmark.tpp"
