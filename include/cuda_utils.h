//
// Created by wookie on 11/22/24.
//

#ifndef CUDAKMEANS_CUDA_UTILS_H
#define CUDAKMEANS_CUDA_UTILS_H

void cudaDeviceInfo();
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
void check_cuda_error(cudaError_t err, char const* func, char const* file, int line);

#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)
void check_last_cuda_error(char const* file, int line);

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#endif //CUDAKMEANS_CUDA_UTILS_H
