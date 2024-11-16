#include <iostream>
#include <cuda_runtime.h>

void checkCudaErrors(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << __LINE__ << " : " << __FILE__ << " : " << __FUNCTION__ << " : " << cudaGetErrorString(error) << std::endl;
    }
}
