//
// Created by wookie on 12/5/24
//


#ifndef CUDAKMEANS_KMEANSVALIDATOR_H
#define CUDAKMEANS_KMEANSVALIDATOR_H

#include <string>
#include <vector>


/**
 * @class KMeansValidator
 * @brief Validates the output of the K-Means algorithm by comparing it against an expected result.
 *
 * This class provides functionality to load expected and computed outputs,
 * and to compare them for correctness in centroids and cluster assignments.
 */
class KMeansValidator {
public:
    /**
     * @brief Loads the expected output from a text file.
     * @param filename The path to the file containing the expected output.
     * @return True if the expected output was successfully loaded, false otherwise.
     */
    bool LoadExpectedOutput(const std::string& filename);

    /**
     * @brief Loads the computed output from a text file.
     * @param filename The path to the file containing the computed output.
     * @return True if the computed output was successfully loaded, false otherwise.
     */
    bool LoadComputedOutput(const std::string& filename);

    /**
     * @brief Compares the expected and computed outputs.
     * @return True if both outputs match, false otherwise.
     */
    bool CompareOutputs() const;

private:
    // Expected output
    std::vector<std::vector<float>> expectedCentroids;
    std::vector<int> expectedLabels;

    // Computed output
    std::vector<std::vector<float>> computedCentroids;
    std::vector<int> computedLabels;

    /**
     * @brief Compares two sets of centroids for equality.
     * @return True if all centroids match within a tolerance, false otherwise.
     */
    bool CompareCentroids() const;

    /**
     * @brief Compares two sets of labels for equality.
     * @return True if all labels match, false otherwise.
     */
    bool CompareLabels() const;

    /**
     * @brief Helper function to read centroids and labels from a file.
     * @param filename The file to read.
     * @param centroids Vector to store the centroids.
     * @param labels Vector to store the labels.
     * @return True if the file was successfully read, false otherwise.
     */
    bool LoadOutput(const std::string& filename, std::vector<std::vector<float>>& centroids, std::vector<int>& labels);
};

#endif //CUDAKMEANS_KMEANSVALIDATOR_H
