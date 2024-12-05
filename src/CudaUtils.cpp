//
// Created by wookie on 11/22/24.
//
#include "CudaUtils.h"
#include <iostream>

#include <cuda_runtime.h>
#include <cstdio>

void check_last_cuda_error(char const* file, int line)
{
}

void CudaUtils::printCudaDeviceInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("\nNumber of CUDA Devices: %d\n", deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        printf("\n==============================\n");
        printf(" Device ID: %d\n", device);
        printf("==============================\n");
        printf("  Name                : %s\n", prop.name);
        printf("  Compute Capability  : %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory : %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Memory Bus Width    : %d bits\n", prop.memoryBusWidth);
        printf("  Memory Clock Rate   : %.2f GHz\n", prop.memoryClockRate / 1.0e6);
        printf("  ----------------------------\n");
        printf("  Shared Memory / Block         : %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("  Shared Memory / Multiprocessor: %.2f KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
        printf("  Constant Memory               : %.2f KB\n", prop.totalConstMem / 1024.0);
        printf("  ----------------------------\n");
        printf("  Warp Size                     : %d\n", prop.warpSize);
        printf("  Max Threads / Block           : %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads / Multiprocessor  : %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Total Multiprocessors         : %d\n", prop.multiProcessorCount);
        printf("  Registers / Block             : %d\n", prop.regsPerBlock);
        printf("  Registers / Multiprocessor    : %d\n", prop.regsPerMultiprocessor);
        printf("  ----------------------------\n");
        printf("  Max Grid Size                 : (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max Block Dim                 : (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  ----------------------------\n");

        // Calculating theoretical maximum FLOPS
        double peakFLOPS = 0.0;
        if (prop.major >= 2) { // Fermi and later
            int cudaCoresPerSM = 0;
            switch (prop.major) {
                case 2: // Fermi
                    cudaCoresPerSM = prop.minor == 0 ? 32 : 48;
                    break;
                case 3: // Kepler
                    cudaCoresPerSM = 192;
                    break;
                case 5: // Maxwell
                    cudaCoresPerSM = 128;
                    break;
                case 6: // Pascal
                    cudaCoresPerSM = (prop.minor == 1 || prop.minor == 2) ? 128 : 64;
                    break;
                case 7: // Volta and Turing
                    cudaCoresPerSM = 64;
                    break;
                case 8: // Ampere
                    cudaCoresPerSM = (prop.minor == 0) ? 64 : 128;
                    break;
                default:
                    cudaCoresPerSM = 64; // Estimate for future architectures
                    break;
            }
            peakFLOPS = 2.0 * prop.multiProcessorCount * cudaCoresPerSM * (prop.clockRate * 1e3);
            printf("  Theoretical Peak FLOPS        : %.2f GFLOPS\n", peakFLOPS / 1e9);
        } else {
            printf("  Theoretical Peak FLOPS        : Not Available for Compute < 2.0\n");
        }

        // Calculating theoretical memory bandwidth
        double bandwidthGBps = (prop.memoryBusWidth / 8.0) * (prop.memoryClockRate * 1e3) / 1.0e9;
        printf("  Theoretical Bandwidth         : %.2f GB/s\n", bandwidthGBps);
        printf("==============================\n");
    }
}

void CudaUtils::check_cuda_error(cudaError_t err, const char *func, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void CudaUtils::check_last_cuda_error(const char *file, int line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
