//
// Created by wookie on 12/5/24.
//

#include "PerformanceClock.h"
#include "CudaUtils.h"
#include <cstdio>
#include <stdexcept>

PerformanceClock::PerformanceClock() = default;

PerformanceClock::~PerformanceClock()
{
    // Destroy any remaining start events
    for (auto &pair : startEvents)
    {
        CHECK_CUDA_ERROR(cudaEventDestroy(pair.second));
    }
    startEvents.clear();
}

void PerformanceClock::start(MEASURED_PHASE phase)
{
    if (startEvents.find(phase) != startEvents.end())
    {
        throw std::runtime_error("Cannot start a phase that is already started");
    }
    cudaEvent_t startEvent;
    CHECK_CUDA_ERROR(cudaEventCreate(&startEvent));
    CHECK_CUDA_ERROR(cudaEventRecord(startEvent));
    startEvents[phase] = startEvent;

    // Initialize cumulative time for this phase if it doesn't exist
    if (cumulativeTimes.find(phase) == cumulativeTimes.end())
    {
        cumulativeTimes[phase] = 0.0;
    }
}

void PerformanceClock::stop(MEASURED_PHASE phase)
{
    auto it = startEvents.find(phase);
    if (it == startEvents.end())
    {
        throw std::runtime_error("Cannot stop a phase that was not started");
    }
    cudaEvent_t startEvent = it->second;

    cudaEvent_t stopEvent;
    CHECK_CUDA_ERROR(cudaEventCreate(&stopEvent));
    CHECK_CUDA_ERROR(cudaEventRecord(stopEvent));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopEvent));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

    cumulativeTimes[phase] += milliseconds;

    CHECK_CUDA_ERROR(cudaEventDestroy(startEvent));
    CHECK_CUDA_ERROR(cudaEventDestroy(stopEvent));

    startEvents.erase(it);
}

[[maybe_unused]] void PerformanceClock::reset()
{
    cumulativeTimes.clear();
    for (auto &pair : startEvents)
    {
        CHECK_CUDA_ERROR(cudaEventDestroy(pair.second));
    }
    startEvents.clear();
}

const char *phaseToString(MEASURED_PHASE phase)
{
    switch (phase)
    {
    case MEASURED_PHASE::DATA_TRANSFER:
        return "Data transfer time";
    case MEASURED_PHASE::KERNEL:
        return "Kernel execution time";
    case MEASURED_PHASE::DATA_TRANSFER_BACK:
        return "Data transfer back time";
    case MEASURED_PHASE::CPU_COMPUTATION:
        return "CPU computation time";
    default:
        return "Unknown phase";
    }
}

void PerformanceClock::printResults(std::optional<std::string> kernel_name) const
{
    if (kernel_name.has_value())
    {
        printf("%-25s: %s\n", "Kernel", kernel_name.value().c_str());
    }
    double total_time = 0.0;
    for (const auto &pair : cumulativeTimes)
    {
        MEASURED_PHASE phase = pair.first;
        double time          = pair.second;
        total_time += time;
        const char *phase_name = phaseToString(phase);
        printf("%-25s: %10.3f ms\n", phase_name, time);
    }
    printf("%-25s: %10.3f ms\n", "Total time", total_time);
}
