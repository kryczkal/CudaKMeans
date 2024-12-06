//
// Created by wookie on 11/22/24.
//
#include "GeneralUtils.h"
#include "CudaUtils.h"
#include <algorithm>
#include <cmath>
#include <iostream>

bool GeneralUtils::fitsInGpuGlobalMemory(uint mem_size_bytes, uint device_id)
{
    cudaDeviceProp prop{};
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device_id));
    size_t freeMem, totalMem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
    return mem_size_bytes <= freeMem;
}