cmake_minimum_required(VERSION 3.12)
project(CudaKMeans LANGUAGES CXX CUDA)

#cmake_policy(SET CMP0146 NEW) # Use the modern CUDA support
#cmake_policy(SET CMP0104 NEW) # Require explicit CUDA architectures

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

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
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --extended-lambda>
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

include(FetchContent)

FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG master
)
FetchContent_MakeAvailable(raylib)

target_link_libraries(CudaKMeans raylib)

## Fetch GLFW
#FetchContent_Declare(
#        glfw
#        GIT_REPOSITORY https://github.com/glfw/glfw.git
#        GIT_TAG        3.4
#)
#
## Fetch GLAD
#FetchContent_Declare(
#        glad
#        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
#        GIT_TAG        v2.0.8
#)
#
## Fetch GLM
#FetchContent_Declare(
#        glm
#        GIT_REPOSITORY https://github.com/g-truc/glm.git
#        GIT_TAG        1.0.1
#)
#
#FetchContent_MakeAvailable(glfw glad glm)
#target_link_libraries(CudaKMeans glfw glad glm)
#target_include_directories(CudaKMeans PRIVATE ${glfw_SOURCE_DIR}/include ${glad_SOURCE_DIR}/include ${glm_SOURCE_DIR})
#
#file(COPY shaders DESTINATION ${CMAKE_BINARY_DIR})