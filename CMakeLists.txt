cmake_minimum_required(VERSION 3.28)
project(CudaKMeans CUDA)

set(CMAKE_CUDA_STANDARD 26)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(CudaKMeans main.cu)

set_target_properties(CudaKMeans PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
