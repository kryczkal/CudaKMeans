//
// Created by wookie on 12/5/24.
//

#ifndef CUDAKMEANS_PERFORMANCECLOCK_H
#define CUDAKMEANS_PERFORMANCECLOCK_H

#include <cuda_runtime.h>


enum class MEASURED_PHASE {
    DATA_TRANSFER,
    KERNEL,
    DATA_TRANSFER_BACK,
    CPU_COMPUTATION,
    NONE
};

/*
 * Enables measuring performance of a given kernel across different stages:
 * Data transfer
 * Kernel execution
 * Data transfer back
 */
class PerformanceClock {
public:
    // Constructor and destructor
    PerformanceClock();
    ~PerformanceClock();

    // Public methods
    void start(MEASURED_PHASE phase);
    void stop(MEASURED_PHASE phase);
    void reset();
    void printResults(std::optional<std::string> kernel_name = std::nullopt) const;

    // Public fields
    double data_transfer_time;
    double kernel_execution_time;
    double data_transfer_back_time;
    double cpu_computation_time;
private:
    // Private fields
    enum MEASURED_PHASE current_phase = MEASURED_PHASE::NONE;
    cudaEvent_t startEvent, stopEvent;

};
#endif //CUDAKMEANS_PERFORMANCECLOCK_H
