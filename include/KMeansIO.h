//
// Created by wookie on 12/5/24.
//

#ifndef CUDAKMEANS_KMEANSIO_H
#define CUDAKMEANS_KMEANSIO_H

#include <string>
#include <cstdint>

/**
 * @class KMeansIO
 * @brief Handles input and output operations for the K-Means algorithm.
 *
 * This class provides methods for loading data in both text and binary formats and writing results
 * in the specified text format for the K-Means algorithm.
 */
class KMeansIO {
public:
    /**
     * @brief Loads data from a text file.
     * @param filename The path to the input text file.
     * @param data A pointer to a float array where the loaded data will be stored.
     * @param N The number of points (output parameter).
     * @param d The number of dimensions (output parameter).
     * @param k The number of clusters (output parameter).
     * @return True if the data was successfully loaded, false otherwise.
     */
    static bool LoadDataFromTextFile(const std::string& filename, float*& data, int64_t& N, int64_t& d, int64_t& k);

    /**
     * @brief Loads data from a binary file.
     * @param filename The path to the input binary file.
     * @param data A pointer to a float array where the loaded data will be stored.
     * @param N The number of points (output parameter).
     * @param d The number of dimensions (output parameter).
     * @param k The number of clusters (output parameter).
     * @return True if the data was successfully loaded, false otherwise.
     */
    static bool LoadDataFromBinaryFile(const std::string& filename, float*& data, int64_t& N, int64_t& d, int64_t& k);

    /**
     * @brief Writes the results to a text file.
     * @param filename The path to the output text file.
     * @param centroids A pointer to a float array containing the centroid coordinates.
     * @param labels A pointer to an int array containing the labels for each data point.
     * @param N The number of points.
     * @param d The number of dimensions.
     * @param k The number of clusters.
     * @return True if the results were successfully written, false otherwise.
     */
    static bool WriteResultsToTextFile(const std::string& filename, const float* centroids, const int* labels, int64_t N, int64_t d, int64_t k);
private:
    KMeansIO() = delete;
    ~KMeansIO() = delete;
};


#endif //CUDAKMEANS_KMEANSIO_H
