cmake_minimum_required(VERSION 3.20)

project(
    GPUDenoising
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(IDL_ENABLE_CUDA "Build CUDA backends when a CUDA compiler is available" ON)

set(IDL_CPU_FILTER_SOURCES
    src/cpu/MeanFilter.cpp
    src/cpu/GaussianFilter.cpp
    src/cpu/MedianFilter.cpp
    src/cpu/BilateralFilter.cpp
    src/cpu/FFTFilter.cpp
)

set(IDL_CPU_NOISE_SOURCES
    src/noise/GaussianNoise.cpp
    src/noise/SaltAndPepperNoise.cpp
    src/noise/PoissonNoise.cpp
)

# Public code shared by every backend.  Filter headers remain backend-agnostic.
add_library(
    idl_core
    STATIC
    src/core/Errors.cpp
    src/core/Timer.cpp
    src/utils/ImageIO.cpp
)

target_include_directories(
    idl_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# A backend defines the same idl::MeanFilter, idl::GaussianFilter, ... symbols.
# Backend libraries must therefore be linked one at a time into an executable.
function(idl_add_filter_backend target)
    add_library(${target} STATIC ${ARGN})
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(${target} PUBLIC idl_core)
endfunction()

idl_add_filter_backend(idl_filters_cpu ${IDL_CPU_FILTER_SOURCES})

add_library(idl_noise_cpu STATIC ${IDL_CPU_NOISE_SOURCES})
target_include_directories(idl_noise_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(idl_noise_cpu PUBLIC idl_core)

add_executable(inject_noise src/InjectNoise.cpp)
target_link_libraries(inject_noise PRIVATE idl_noise_cpu)

add_executable(denoising_benchmark_cpu src/BenchmarkMain.cpp)
target_link_libraries(denoising_benchmark_cpu PRIVATE idl_filters_cpu idl_noise_cpu)
target_compile_definitions(denoising_benchmark_cpu PRIVATE IDL_BENCHMARK_BACKEND="cpu")

add_executable(generate_filtered_cpu src/GenerateFilteredImages.cpp)
target_link_libraries(generate_filtered_cpu PRIVATE idl_filters_cpu)
target_compile_definitions(generate_filtered_cpu PRIVATE IDL_FILTER_OUTPUT_BACKEND="cpu")

if(IDL_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        function(idl_add_cuda_filter target)
            idl_add_filter_backend(${target} ${ARGN})
            target_link_libraries(${target} PUBLIC CUDA::cudart)
            set_target_properties(${target} PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
            )
        endfunction()

        set(IDL_CUDA_FILTER_SOURCES
            src/cuda/MeanFilter.cu
            src/cuda/GaussianFilter.cu
            src/cuda/MedianFilter.cu
            src/cuda/BilateralFilter.cu
            src/cuda/FFTFilter.cu
        )

        set(IDL_CUDA_OPTIMIZED_FILTER_SOURCES
            src/cuda_optimized/MeanFilter.cu
            src/cuda_optimized/GaussianFilter.cu
            src/cuda_optimized/MedianFilter.cu
            src/cuda_optimized/BilateralFilter.cu
            src/cuda_optimized/FFTFilter.cu
        )

        # A complete CUDA backend is linked one at a time, just like the CPU backend.
        idl_add_cuda_filter(idl_filters_cuda_reference ${IDL_CUDA_FILTER_SOURCES})
        idl_add_cuda_filter(idl_filters_cuda_optimized ${IDL_CUDA_OPTIMIZED_FILTER_SOURCES})
        target_link_libraries(idl_filters_cuda_reference PUBLIC CUDA::cufft)
        target_link_libraries(idl_filters_cuda_optimized PUBLIC CUDA::cufft)

        add_executable(denoising_benchmark_cuda_reference src/BenchmarkMain.cpp)
        target_link_libraries(denoising_benchmark_cuda_reference PRIVATE idl_filters_cuda_reference idl_noise_cpu)
        target_compile_definitions(denoising_benchmark_cuda_reference PRIVATE IDL_BENCHMARK_BACKEND="cuda_reference")

        add_executable(denoising_benchmark_cuda_optimized src/BenchmarkMain.cpp)
        target_link_libraries(denoising_benchmark_cuda_optimized PRIVATE idl_filters_cuda_optimized idl_noise_cpu)
        target_compile_definitions(denoising_benchmark_cuda_optimized PRIVATE IDL_BENCHMARK_BACKEND="cuda_optimized")

        add_executable(generate_filtered_cuda_reference src/GenerateFilteredImages.cpp)
        target_link_libraries(generate_filtered_cuda_reference PRIVATE idl_filters_cuda_reference)
        target_compile_definitions(generate_filtered_cuda_reference PRIVATE IDL_FILTER_OUTPUT_BACKEND="cuda_reference")

        add_executable(generate_filtered_cuda_optimized src/GenerateFilteredImages.cpp)
        target_link_libraries(generate_filtered_cuda_optimized PRIVATE idl_filters_cuda_optimized)
        target_compile_definitions(generate_filtered_cuda_optimized PRIVATE IDL_FILTER_OUTPUT_BACKEND="cuda_optimized")
    else()
        message(STATUS "CUDA compiler not found: CUDA backends are skipped.")
    endif()
endif()
