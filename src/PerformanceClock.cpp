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
        throw std::runtime_error("PerformanceClock::start: Cannot start a phase that is already started");
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
        throw std::runtime_error("PerformanceClock::stop: Cannot stop a phase that was not started");
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
    case MEASURED_PHASE::THRUST:
        return "Thrust time";
    case MEASURED_PHASE::LABEL_ASSIGNMENT:
        return "Label assignment time";
    case MEASURED_PHASE::CENTROID_UPDATE:
        return "Centroid update time";
    default:
        return "Unknown phase";
    }
}

void PerformanceClock::printResults(std::optional<std::string> kernel_name) const
{
    printDelimiter();
    if (kernel_name.has_value())
    {
        printf("%-25s: %s\n", "Kernel", kernel_name.value().c_str());
    }
    printDelimiter();
    double sum_of_times = 0.0;
    for (const auto &pair : cumulativeTimes)
    {
        MEASURED_PHASE phase = pair.first;
        if (phase == MEASURED_PHASE::TOTAL)
        {
            continue;
        }
        double time = pair.second;
        sum_of_times += time;
        const char *phase_name = phaseToString(phase);
        printf("%-25s: %10.3f ms\n", phase_name, time);
    }
    printDelimiter();
    if (cumulativeTimes.find(MEASURED_PHASE::TOTAL) != cumulativeTimes.end())
    {
        double everything_else = cumulativeTimes.at(MEASURED_PHASE::TOTAL) - sum_of_times;
        printf("%-25s: %10.3f ms\n", "Sum of above", sum_of_times);
        printDelimiter();
        printf("%-25s: %10.3f ms\n", "Debug work", everything_else);
        printf("%-25s: %10.3f ms\n", "Total time", cumulativeTimes.at(MEASURED_PHASE::TOTAL));
    }
    else
    {
        printf("Total time: %10.3f ms\n", sum_of_times);
    }
    printDelimiter();

    printf("Debug work - time spent on operations not measured such as calculating exactly which labels changed (work "
           "of no consequence to the algorithm itself)\n");
    printDelimiter();
}

void PerformanceClock::printDelimiter() { printf("%-25s:\n", "-------------------------"); }
