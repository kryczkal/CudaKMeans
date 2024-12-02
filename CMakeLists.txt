cmake_minimum_required(VERSION 3.12)
project(CudaKMeans LANGUAGES CXX CUDA)

cmake_policy(SET CMP0146 NEW) # Use the modern CUDA support
cmake_policy(SET CMP0104 NEW) # Require explicit CUDA architectures

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

include_directories(include)

file(GLOB_RECURSE CPP_SOURCES src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)

set(SOURCES ${CPP_SOURCES} ${CUDA_SOURCES} main.cpp)

set_source_files_properties(${CPP_SOURCES} main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(CudaKMeans ${SOURCES})

set_target_properties(CudaKMeans PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

find_package(CUDAToolkit REQUIRED)

target_include_directories(CudaKMeans PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(CudaKMeans CUDA::cudart)

# Optionally, set compile options for CUDA files
target_compile_options(CudaKMeans PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

# Add different flags for Release and Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Configuring Release build...")
    target_compile_options(CudaKMeans PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>  # Optimize for performance and disable debug macros
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    )
else()
    message(STATUS "Configuring Debug build...")
    target_compile_options(CudaKMeans PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-g -DDEBUG>  # Include debug symbols
            $<$<COMPILE_LANGUAGE:CUDA>:-g -G>      # Enable debug symbols and host/device memory checks
    )
endif()
