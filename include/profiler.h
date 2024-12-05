//
// Created by wookie on 11/22/24.
//

#ifndef CUDAKMEANS_PROFILER_H
#define CUDAKMEANS_PROFILER_H

#include <functional>
#include <cuda_runtime.h>
#include "CudaUtils.h"

template <class T>
struct PerformanceResult {
    float latency_ms;
    float flops;
    float bandwidth_GBps;
};

template <class T>
PerformanceResult<T> measure_performance(std::function<T(cudaStream_t)> bound_function,
                                         cudaStream_t stream,
                                         size_t num_repeats = 10,
                                         size_t num_warmups = 10,
                                         size_t num_operations = 0,
                                         size_t num_bytes = 0);

#include "profiler.hpp"


#endif //CUDAKMEANS_PROFILER_H
