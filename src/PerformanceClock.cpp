//
// Created by wookie on 12/5/24.
//

#include <stdexcept>
#include <optional>
#include "PerformanceClock.h"
#include "CudaUtils.h"

PerformanceClock::PerformanceClock() {
    CHECK_CUDA_ERROR(cudaEventCreate(&startEvent));
    CHECK_CUDA_ERROR(cudaEventCreate(&stopEvent));
}

PerformanceClock::~PerformanceClock() {
    CHECK_CUDA_ERROR(cudaEventDestroy(startEvent));
    CHECK_CUDA_ERROR(cudaEventDestroy(stopEvent));
}

void PerformanceClock::start(MEASURED_PHASE phase) {
    if (current_phase != MEASURED_PHASE::NONE) {
        throw std::runtime_error("Cannot start a new phase before stopping the previous one");
    }
    current_phase = phase;
    CHECK_CUDA_ERROR(cudaEventRecord(startEvent));
}

void PerformanceClock::stop(MEASURED_PHASE phase) {
    if (current_phase != phase || current_phase == MEASURED_PHASE::NONE) {
        throw std::runtime_error("Cannot stop a phase that was not started");
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stopEvent));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopEvent));
    CHECK_CUDA_ERROR(cudaEventSynchronize(startEvent));
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
    switch (phase) {
        case MEASURED_PHASE::DATA_TRANSFER:
            data_transfer_time += milliseconds;
            break;
        case MEASURED_PHASE::KERNEL:
            kernel_execution_time += milliseconds;
            break;
        case MEASURED_PHASE::DATA_TRANSFER_BACK:
            data_transfer_back_time += milliseconds;
            break;
        case MEASURED_PHASE::CPU_COMPUTATION:
            cpu_computation_time += milliseconds;
            break;
        default:
            throw std::runtime_error("Unknown phase");
    }
    current_phase = MEASURED_PHASE::NONE;
}

void PerformanceClock::reset() {
    data_transfer_time = 0;
    kernel_execution_time = 0;
    data_transfer_back_time = 0;
    current_phase = MEASURED_PHASE::NONE;
}

void PerformanceClock::printResults(std::optional<std::string> kernel_name) const {
    if (kernel_name.has_value()) {
        printf("%-25s: %s\n", "Kernel", kernel_name.value().c_str());
    }
    printf("%-25s: %10.3f ms\n", "Data transfer time", data_transfer_time);
    printf("%-25s: %10.3f ms\n", "Kernel execution time", kernel_execution_time);
    printf("%-25s: %10.3f ms\n", "Data transfer back time", data_transfer_back_time);
    printf("%-25s: %10.3f ms\n", "CPU computation time", cpu_computation_time);
}

