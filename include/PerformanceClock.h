//
// Created by wookie on 12/5/24.
//

#ifndef CUDAKMEANS_PERFORMANCECLOCK_H
#define CUDAKMEANS_PERFORMANCECLOCK_H

#include <cuda_runtime.h>
#include <optional>
#include <string>
#include <unordered_map>

/**
 * @enum MEASURED_PHASE
 * @brief An enumeration of the different phases of the K-Means algorithm that can be measured.
 */
enum class MEASURED_PHASE
{
    DATA_TRANSFER,
    DATA_TRANSFER_BACK,
    KERNEL,
    THRUST,
    CPU_COMPUTATION,
    TOTAL,
    LABEL_ASSIGNMENT,
    CENTROID_UPDATE,
};

/**
 * @class Performance Clock
 * @brief A class for measuring the performance of different phases of the K-Means algorithm.
 * The class uses CUDA events to measure the time taken by different phases of the algorithm.
 * The measured times are accumulated and can be printed out at the end of the algorithm.
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
