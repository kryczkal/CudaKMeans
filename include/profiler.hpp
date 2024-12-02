#ifndef PROFILER_HPP
#define PROFILER_HPP

template <class T>
PerformanceResult<T> measure_performance(std::function<T(cudaStream_t)> bound_function,
                                         cudaStream_t stream,
                                         size_t num_repeats,
                                         size_t num_warmups,
                                         size_t num_operations,
                                         size_t num_bytes)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (size_t i = 0; i < num_warmups; ++i) {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i = 0; i < num_repeats; ++i) {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float latency = time / num_repeats;

    PerformanceResult<T> result;
    result.latency_ms = latency;
    if (num_operations > 0) {
        result.flops = num_operations / (latency / 1000.0f);
    } else {
        result.flops = 0.0f;
    }

    if (num_bytes > 0) {
        result.bandwidth_GBps = (num_bytes / 1e9) / (latency / 1000.0f);
    } else {
        result.bandwidth_GBps = 0.0f;
    }

    return result;
}

#endif // PROFILER_HPP
