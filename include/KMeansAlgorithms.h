//
// Created by wookie on 11/22/24.
//

#ifndef CUDAKMEANS_KMEANSALGORITHMS_H
#define CUDAKMEANS_KMEANSALGORITHMS_H

#include "CudaUtils.h"
#include <cuda_runtime.h>
#include <iostream>

static constexpr unsigned int max_iter = 100;
class KMeansAlgorithms
{
    public:
    static void Cpu(const float *data, float *centroids, int *labels, int n, int d, int k);

    /**
     * @brief Perform KMeans clustering using the atomic add and shared memory approach
     * @tparam D Dimensionality of the data
     * @tparam K Number of clusters
     * @param h_data Data points
     * @param h_centroids Initial centroids
     * @param h_labels Output labels
     * @param n Number of data points
     */
    template <int D, int K> static void AtomicAddShmem(float *h_data, float *h_centroids, int *h_labels, int n);

    /**
     * @brief Perform KMeans clustering using the thrust library
     * @param h_data Data points
     * @param h_centroids Initial centroids
     * @param h_labels Output labels
     * @param n Number of data points
     * @param d Dimensionality of the data
     * @param k Number of clusters
     */
    static void ThrustVersion(float *h_data, float *h_centroids, int *h_labels, int n, int d, int k);
};

#include "KMeansAlgorithms.tpp"

#endif // CUDAKMEANS_KMEANSALGORITHMS_H
