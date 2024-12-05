//
// Created by wookie on 11/22/24.
//

#ifndef CUDAKMEANS_GENERALUTILS_H
#define CUDAKMEANS_GENERALUTILS_H

#include <cinttypes>

class GeneralUtils
{
    public:
    /**
     * @brief Function to check if the given memory size fits into the global memory of the device
     * @param mem_size_bytes - size of the memory in bytes
     * @param device_id - ID of the device to check
     * @return true if the memory fits, false otherwise
     */
    static bool fitsInGpuGlobalMemory(uint mem_size_bytes, uint device_id = 0);

    /**
     * @brief Function to check if the given memory size fits into the RAM
     * @param mem_size_bytes - size of the memory in bytes
     * @return true if the memory fits, false otherwise
     */
    [[maybe_unused]] static bool fitsInRam(uint mem_size_bytes);
    static uint getTotalRam();

    /**
     * @brief Function to visualize K-means clustering result
     * @param data - input data points
     * @param centroids - cluster centroids
     * @param labels - cluster labels
     * @param N - number of points
     * @param D - number of dimensions
     * @param K - number of clusters
     * @param width - width of the visualization grid
     * @param height - height of the visualization grid
     *
     * The function visualizes the K-means clustering result by projecting the data points and centroids to 2D space
     * using PCA and then mapping the projected points to a grid.
     */
    [[maybe_unused]] static void
    visualizeKmeans(const float *data, const float *centroids, const int *labels, uint N, uint D, uint K);
};

#endif // CUDAKMEANS_GENERALUTILS_H
