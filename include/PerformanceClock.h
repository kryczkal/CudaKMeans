//
// Created by wookie on 12/5/24.
//

#ifndef CUDAKMEANS_PERFORMANCECLOCK_H
#define CUDAKMEANS_PERFORMANCECLOCK_H

#include <cuda_runtime.h>
#include <optional>
#include <string>
#include <unordered_map>

enum class MEASURED_PHASE
{
    DATA_TRANSFER,
    DATA_TRANSFER_BACK,
    KERNEL,
    THRUST,
    CPU_COMPUTATION,
    TOTAL
};

/*
 * Enables measuring performance of different phases:
 * Data transfer
 * Kernel execution
 * Data transfer back
 * CPU computation
 */
class PerformanceClock
{
    public:
    // Constructor and destructor
    PerformanceClock();
    ~PerformanceClock();

    // Public methods
    void start(MEASURED_PHASE phase);
    void stop(MEASURED_PHASE phase);

    [[maybe_unused]] void reset();
    void printResults(std::optional<std::string> kernel_name = std::nullopt) const;
    static void printDelimiter();

    private:
    // Private fields
    std::unordered_map<MEASURED_PHASE, cudaEvent_t> startEvents;
    std::unordered_map<MEASURED_PHASE, double> cumulativeTimes;
};

#endif // CUDAKMEANS_PERFORMANCECLOCK_H
