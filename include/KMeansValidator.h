//
// Created by wookie on 12/5/24
//

#ifndef CUDAKMEANS_KMEANSVALIDATOR_H
#define CUDAKMEANS_KMEANSVALIDATOR_H

#include <string>
#include <vector>

#include <cstdint>
#include <string>

/**
 * @class KMeansValidator
 * @brief Validates the results of the K-Means algorithm by comparing centroids and labels from two different files.
 * One file is considered the ground truth and the other is the results to be validated.
 */
class [[maybe_unused]] KMeansValidator
{
    public:
    KMeansValidator()  = delete;
    ~KMeansValidator() = delete;

    /**
     * @brief Validates two sets of K-Means results.
     * @param file1 The path to the first results file.
     * @param file2 The path to the second results file.
     * @param d The number of dimensions.
     * @param k The number of clusters.
     * @return True if validation is successful, false otherwise.
     */
    static bool ValidateResults(const std::string &truthFile, const std::string &testedFile, int d, int k);

    // Constants
    static constexpr double labelMismatchTolerancePercent = 0.01;
    static constexpr double centroidDiffTolerance         = 0.15;
};
#endif // CUDAKMEANS_KMEANSVALIDATOR_H
