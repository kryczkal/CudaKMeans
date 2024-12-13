cmake_minimum_required(VERSION 3.12)
project(CudaKMeans LANGUAGES C CXX CUDA)

option(USE_VISUALIZER "Enable OpenGL visualization" OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Include paths
include_directories(include)

file(GLOB_RECURSE CPP_SOURCES src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)

set(SOURCES ${CPP_SOURCES} ${CUDA_SOURCES} main.cpp)

list(REMOVE_ITEM CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Visualizer.cpp)
set_source_files_properties(${CPP_SOURCES} main.cpp PROPERTIES LANGUAGE CUDA)
list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Visualizer.cpp)

add_executable(CudaKMeans ${SOURCES})

set_target_properties(CudaKMeans PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

find_package(CUDAToolkit REQUIRED)
target_include_directories(CudaKMeans PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(CudaKMeans CUDA::cudart)

# Compile options for CUDA
target_compile_options(CudaKMeans PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --extended-lambda>
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Configuring Release build...")
    target_compile_options(CudaKMeans PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    )
else()
    message(STATUS "Configuring Debug build...")
    target_compile_options(CudaKMeans PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-g -DDEBUG>
            $<$<COMPILE_LANGUAGE:CUDA>:-g -G>
    )
endif()

include(FetchContent)
if (${USE_VISUALIZER} STREQUAL "ON")

    # Fetch GLFW
    FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG        3.4
    )

    # Fetch GLM
    FetchContent_Declare(
            glm
            GIT_REPOSITORY https://github.com/g-truc/glm.git
            GIT_TAG        1.0.1
    )

    FetchContent_MakeAvailable(glfw glm)
    target_link_libraries(CudaKMeans
            glfw glm
    )
    target_include_directories(CudaKMeans PRIVATE
            ${glfw_SOURCE_DIR}/include
            ${glad_SOURCE_DIR}/include
            ${glm_SOURCE_DIR}
    )

    FetchContent_Declare(
            glad
            GIT_REPOSITORY https://github.com/Dav1dde/glad.git
            GIT_TAG        658f48e72aee3c6582e80b05ac0f8787a64fe6bb #v2.0.6
            GIT_PROGRESS   TRUE
            SOURCE_SUBDIR  cmake
    )
    FetchContent_MakeAvailable(glad)
    glad_add_library(glad_gl_core_43 STATIC REPRODUCIBLE LOADER API gl:core=4.3)
    target_link_libraries(CudaKMeans
            glad_gl_core_43)

    # Copy shaders to binary dir
    file(COPY shaders DESTINATION ${CMAKE_BINARY_DIR})
    target_compile_definitions(CudaKMeans PRIVATE USE_VISUALIZER)
endif()

target_compile_definitions(CudaKMeans PRIVATE PROJECT_BINARY_DIR="${CMAKE_BINARY_DIR}")