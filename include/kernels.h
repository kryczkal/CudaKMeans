//
// Created by wookie on 11/22/24.
//

#ifndef CUDAKMEANS_KERNELS_H
#define CUDAKMEANS_KERNELS_H

#include <cuda_runtime.h>

template <int D, int K>
__global__ void shmem_labeling(const float *data, const float *centroids, int *labels, bool *did_change, int n);

__global__ void
shmem_labeling(const float *data, const float *centroids, int *labels, bool *did_change, int n, int d, int k);

template <int D, int K>
__global__ void
atomic_add_shmem_centroid_update(const float *data, const int *labels, float *centroids, int *counts, int n);

struct LocalHistogram;

__global__ void compute_local_histograms(
    const float *data, const int *labels, float *partial_sums, int *partial_counts, int n, int d, int k,
    int points_per_block
);

__global__ void
tree_reduce_centroids(float *partial_sums, int *partial_counts, float *final_centroids, int num_blocks, int d, int k);

#include "kernels.tpp"

#endif // CUDAKMEANS_KERNELS_H
