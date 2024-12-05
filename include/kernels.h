//
// Created by wookie on 11/22/24.
//

#ifndef CUDAKMEANS_KERNELS_H
#define CUDAKMEANS_KERNELS_H

#include <cuda_runtime.h>

// data - input data points
// centroids - cluster centroids
// n - number of points - up to 50e6
// D - number of dimensions - 1-20
// K - number of clusters - 2-20

/*
 * Naive k-means algorithm
 */
__global__ void naive_labeling(const float *data, const float *centroids, int *labels, int n, int d, int k);
__global__ void
naive_centroid_update(const float *data, const int *labels, float *centroids, int n, int d, int k, bool *changes);

/*
 * Shared memory reduction k-means algorithm
 */
__global__ void
shmem_labeling(const float *data, const float *centroids, int *labels, bool *did_change, int n, int d, int k);
__global__ void atomic_add_shmem_centroid_update(
    const float *data, const int *labels, float *centroids, int *counts, int n, int d, int k
);

#endif // CUDAKMEANS_KERNELS_H
