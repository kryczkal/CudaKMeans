#include "kernels.h"

static constexpr float FLT_MAX = std::numeric_limits<float>::max();

__global__ void naive_labeling(const float *data, const float *centroids, int *labels, int n, int d, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float min_dist = FLT_MAX;
        int label = -1;
        // Compute the distance between the data point and each centroid
        for (int j = 0; j < k; j++) {
            float dist = 0;
            // Compute the Euclidean distance (without the square root)
            for (int l = 0; l < d; l++) {
                float diff = data[i * d + l] - centroids[j * d + l];
                dist += diff * diff;
            }
            // Update the closest centroid
            if (dist < min_dist) {
                min_dist = dist;
                label = j;
            }
        }
        // Assign the data point to the closest centroid
        labels[i] = label;
    }
}


__global__ void naive_centroid_update(const float* data, const int* labels, float* centroids, int n, int d, int k, bool* changes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (labels[j] == i) {
                count++;
                for (int l = 0; l < d; l++) {
                    atomicAdd(&centroids[i * d + l], data[j * d + l]);
                }
            }
        }
        if (count > 0) {
            for (int l = 0; l < d; l++) {
                centroids[i * d + l] /= count;
            }
            *changes = true;
        }
    }
}

__global__ void shmem_labeling(const float *data, const float *centroids, int *labels, bool* did_change, int n, int d, int k) {
    extern __shared__ float s_centroids[]; // Shared memory for centroids

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load centroids into shared memory
    for (int i = tid; i < k * d; i += blockDim.x) {
        s_centroids[i] = centroids[i];
    }
    __syncthreads();

    // Each thread processes multiple data points
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        const float *data_point = &data[i * d];

        // Find the nearest centroid
        int label = -1;
        float min_dist = FLT_MAX;
        for (int c = 0; c < k; ++c) {
            float dist = 0;
            for (int j = 0; j < d; ++j) {
                float diff = data_point[j] - s_centroids[c * d + j];
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

__global__ void
atomic_add_shmem_centroid_update(const float *data, const int *labels, float *centroids, int *counts, int n, int d, int k)
{
    extern __shared__ float s_data[]; // shared memory

    // s_sums: k*d floats for partial centroid sums
    // s_counts: k ints for partial counts
    float *s_sums = s_data;
    int *s_counts = (int *)&s_sums[k * d];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory
    for (int i = tid; i < k * d; i += blockDim.x) {
        s_sums[i] = 0.0f;
    }
    for (int i = tid; i < k; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();

    // Accumulate local sums
    for (int i = global_id; i < n; i += gridDim.x * blockDim.x) {
        int label = labels[i];
        atomicAdd(&s_counts[label], 1);
        for (int j = 0; j < d; ++j) {
            atomicAdd(&s_sums[label * d + j], data[i * d + j]);
        }
    }
    __syncthreads();

    // Write shared results to global memory
    for (int i = tid; i < k * d; i += blockDim.x) {
        atomicAdd(&centroids[i], s_sums[i]);
    }
    for (int i = tid; i < k; i += blockDim.x) {
        atomicAdd(&counts[i], s_counts[i]);
    }
}