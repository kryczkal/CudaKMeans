#include "kernels.h"

static constexpr float FLT_MAX = std::numeric_limits<float>::max();

template<int D, int K>
__global__ void shmem_labeling(const float *data, const float *centroids, int *labels, bool* did_change, int n) {
    extern __shared__ float s_centroids[]; // Shared memory for centroids

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;

    // Load centroids into shared memory
    for (unsigned int i = tid; i < K * D; i += blockDim.x) {
        s_centroids[i] = centroids[i];
    }
    __syncthreads();

    // Each thread processes multiple data points
    for (unsigned int i = gid; i < n; i += gridDim.x * blockDim.x) {
        const float *data_point = &data[i * D];

        // Find the nearest centroid
        int label = -1;
        float min_dist = FLT_MAX;
        for (int c = 0; c < K; ++c) {
            float dist = 0;
            for (int j = 0; j < D; ++j) {
                float diff = data_point[j] - s_centroids[c * D + j];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                label = c;
            }
        }

        if (labels[i] != label) {
            *did_change = true;
        }
        labels[i] = label;
    }
}


template<int D, int K>
__global__ void
atomic_add_shmem_centroid_update(const float *data, const int *labels, float *centroids, int *counts, int n)
{
    extern __shared__ float s_data[]; // shared memory

    // s_sums: k*d floats for partial centroid sums
    // s_counts: k ints for partial counts
    float *s_sums = s_data;
    int *s_counts = (int *)&s_sums[K * D];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory
    for (unsigned int i = tid; i < K * D; i += blockDim.x) {
        s_sums[i] = 0.0f;
    }
    for (unsigned int i = tid; i < K; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();

    // Accumulate local sums
    for (unsigned int i = global_id; i < n; i += gridDim.x * blockDim.x) {
        int label = labels[i];
        atomicAdd(&s_counts[label], 1);
        for (int j = 0; j < D; ++j) {
            atomicAdd(&s_sums[label * D + j], data[i * D + j]);
        }
    }
    __syncthreads();

    // Write shared results to global memory
    for (unsigned int i = tid; i < K * D; i += blockDim.x) {
        atomicAdd(&centroids[i], s_sums[i]);
    }
    for (unsigned int i = tid; i < K; i += blockDim.x) {
        atomicAdd(&counts[i], s_counts[i]);
    }
}
