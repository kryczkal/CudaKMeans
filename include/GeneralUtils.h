//
// Created by wookie on 11/22/24.
//

#ifndef CUDAKMEANS_GENERALUTILS_H
#define CUDAKMEANS_GENERALUTILS_H

#include <cinttypes>

class GeneralUtils
{
    public:
    /**
     * @brief Function to check if the given memory size fits into the global memory of the device
     * @param mem_size_bytes - size of the memory in bytes
     * @param device_id - ID of the device to check
     * @return true if the memory fits, false otherwise
     */
    static bool fitsInGpuGlobalMemory(uint mem_size_bytes, uint device_id = 0);
};

#endif // CUDAKMEANS_GENERALUTILS_H
