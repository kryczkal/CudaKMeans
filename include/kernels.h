//
// Created by wookie on 11/22/24.
//

#ifndef CUDAKMEANS_KERNELS_H
#define CUDAKMEANS_KERNELS_H

#include <cuda_runtime.h>

/**
 * @brief Kernel that assigns each data point to the nearest centroid
 * @tparam D Dimensionality of the data
 * @tparam K Number of clusters
 * @param data Data points
 * @param centroids Cluster centroids
 * @param labels Output labels
 * @param did_change Flag indicating if any point changed its cluster
 * @param n Number of data points
 */
template <int D, int K>
__global__ void shmem_labeling(const float *data, const float *centroids, int *labels, bool *did_change, int n);

/**
 * @brief Kernel that assings each data point to nearest centroid (not templated) (fastest)
 * @tparam D Dimensionality of the data
 * @tparam K Number of clusters
 * @param data Data points
 * @param labels Cluster labels
 * @param centroids Cluster centroids
 * @param counts Number of points in each cluster
 * @param n Number of data points
 */
__global__ void
shmem_labeling(const float *data, const float *centroids, int *labels, bool *did_change, int n, int d, int k);

/**
 *  @brief Kernel that populates updates the centroids with the sum of the data points in each cluster and the counts
 *  with the number of points in each cluster
 *  @tparam D Dimensionality of the data
 *  @tparam K Number of clusters
 *  @param data Data points
 *  @param labels Cluster labels
 *  @param centroids Cluster centroids
 *  @param counts Number of points in each cluster
 *  @param n Number of data points
 */
template <int D, int K>
__global__ void
atomic_add_shmem_centroid_update(const float *data, const int *labels, float *centroids, int *counts, int n);

#include "kernels.tpp"

#endif // CUDAKMEANS_KERNELS_H
