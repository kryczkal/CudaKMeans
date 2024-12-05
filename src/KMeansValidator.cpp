//
// Created by wookie on 12/5/24.
//

#include "KMeansValidator.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

/**
 * @brief Helper function to read centroids and labels from a file.
 * @param filename The file to read.
 * @param centroids Vector to store the centroids.
 * @param labels Vector to store the labels.
 * @return True if the file was successfully read, false otherwise.
 */
bool KMeansValidator::LoadOutput(const std::string& filename, std::vector<std::vector<float>>& centroids, std::vector<int>& labels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    bool readingCentroids = true;

    while (std::getline(file, line)) {
        std::istringstream iss(line);

        if (readingCentroids) {
            // Attempt to read a line of centroid coordinates
            std::vector<float> centroid;
            float value;
            while (iss >> value) {
                centroid.push_back(value);
            }

            if (centroid.empty()) {
                readingCentroids = false; // Switch to reading labels
            } else {
                centroids.push_back(centroid);
            }
        }

        if (!readingCentroids) {
            // Read labels
            int label;
            if (iss >> label) {
                labels.push_back(label);
            }
        }
    }

    file.close();
    return true;
}

bool KMeansValidator::LoadExpectedOutput(const std::string& filename) {
    return LoadOutput(filename, expectedCentroids, expectedLabels);
}

bool KMeansValidator::LoadComputedOutput(const std::string& filename) {
    return LoadOutput(filename, computedCentroids, computedLabels);
}

bool KMeansValidator::CompareCentroids() const {
    if (expectedCentroids.size() != computedCentroids.size()) {
        std::cerr << "Centroid count mismatch: "
                  << "expected " << expectedCentroids.size() << ", "
                  << "computed " << computedCentroids.size() << std::endl;
        return false;
    }

    const float tolerance = 1e-4f; // Tolerance for floating-point comparisons
    for (size_t i = 0; i < expectedCentroids.size(); ++i) {
        if (expectedCentroids[i].size() != computedCentroids[i].size()) {
            std::cerr << "Centroid dimension mismatch at index " << i << std::endl;
            return false;
        }

        for (size_t j = 0; j < expectedCentroids[i].size(); ++j) {
            float diff = std::fabs(expectedCentroids[i][j] - computedCentroids[i][j]);
            if (diff > tolerance) {
                std::cerr << "Centroid value mismatch at index (" << i << ", " << j << "): "
                          << "expected " << expectedCentroids[i][j] << ", "
                          << "computed " << computedCentroids[i][j] << ", "
                          << "difference " << diff << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool KMeansValidator::CompareLabels() const {
    if (expectedLabels.size() != computedLabels.size()) {
        std::cerr << "Label count mismatch: "
                  << "expected " << expectedLabels.size() << ", "
                  << "computed " << computedLabels.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < expectedLabels.size(); ++i) {
        if (expectedLabels[i] != computedLabels[i]) {
            std::cerr << "Label mismatch at index " << i << ": "
                      << "expected " << expectedLabels[i] << ", "
                      << "computed " << computedLabels[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool KMeansValidator::CompareOutputs() const {
    bool centroidsMatch = CompareCentroids();
    bool labelsMatch = CompareLabels();

    if (centroidsMatch && labelsMatch) {
        std::cout << "Validation successful: outputs match." << std::endl;
        return true;
    } else {
        std::cerr << "Validation failed: outputs do not match." << std::endl;
        return false;
    }
}
