#include "kernels.h"

__global__ void
shmem_labeling(const float *data, const float *centroids, int *labels, bool *did_change, int n, int d, int k)
{
    extern __shared__ float s_centroids[]; // Shared memory for centroids

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;

    // Load centroids into shared memory
    for (unsigned int i = tid; i < k * d; i += blockDim.x)
    {
        s_centroids[i] = centroids[i];
    }
    __syncthreads();

    // Each thread processes multiple data points
    for (unsigned int i = gid; i < n; i += gridDim.x * blockDim.x)
    {
        const float *data_point = &data[i * d];

        // Find the nearest centroid
        int label      = -1;
        float min_dist = FLT_MAX;
        for (int c = 0; c < k; ++c)
        {
            float dist = 0;
            for (int j = 0; j < d; ++j)
            {
                float diff = data_point[j] - s_centroids[c * d + j];
                dist += diff * diff;
            }
            if (dist < min_dist)
            {
                min_dist = dist;
                label    = c;
            }
        }

        if (labels[i] != label)
        {
            *did_change = true;
        }
        labels[i] = label;
    }
}

struct LocalHistogram
{
    float *sums; // k * d values
    int *counts; // k values

    __device__ void initialize(float *shared_mem, int k, int d)
    {
        sums   = shared_mem;
        counts = reinterpret_cast<int *>(shared_mem + k * d);

        // Zero initialization of local histogram
        for (int i = threadIdx.x; i < k; i += blockDim.x)
        {
            counts[i] = 0;
            for (int j = 0; j < d; j++)
            {
                sums[i * d + j] = 0.0f;
            }
        }
        __syncthreads();
    }
};

__global__ void compute_local_histograms(
    const float *data, const int *labels, float *partial_sums, int *partial_counts, int n, int d, int k,
    int points_per_block
)
{
    extern __shared__ float shared_mem[];
    LocalHistogram histogram;
    histogram.initialize(shared_mem, k, d);

    // Each thread processes its chunk of data
    int start_idx = blockIdx.x * points_per_block + threadIdx.x;
    int stride    = blockDim.x;
    int end_idx   = min(start_idx + points_per_block, n);

    // Compute local histogram
    for (int idx = start_idx; idx < end_idx; idx += stride)
    {
        int label = labels[idx];
        if (label >= 0 && label < k)
        {
            atomicAdd(&histogram.counts[label], 1);
            for (int d_idx = 0; d_idx < d; d_idx++)
            {
                atomicAdd(&histogram.sums[label * d + d_idx], data[idx * d + d_idx]);
            }
        }
    }
    __syncthreads();

    // Write local histogram to global memory
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < k; i++)
        {
            partial_counts[blockIdx.x * k + i] = histogram.counts[i];
            for (int j = 0; j < d; j++)
            {
                partial_sums[blockIdx.x * k * d + i * d + j] = histogram.sums[i * d + j];
            }
        }
    }
}

__global__ void
tree_reduce_centroids(float *partial_sums, int *partial_counts, float *final_centroids, int num_blocks, int d, int k)
{
    extern __shared__ float shared_mem[];
    float *shared_sums = shared_mem;
    int *shared_counts = reinterpret_cast<int *>(shared_mem + k * d);

    // Zero initialization
    for (int idx = threadIdx.x; idx < k * d; idx += blockDim.x)
    {
        shared_sums[idx] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < k; idx += blockDim.x)
    {
        shared_counts[idx] = 0;
    }
    __syncthreads();

    // Accumulate all sums and counts into shared memory
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x)
    {
        for (int c = 0; c < k; c++)
        {
            int count = partial_counts[b * k + c];
            if (count > 0)
            {
                atomicAdd(&shared_counts[c], count);
                for (int dim = 0; dim < d; dim++)
                {
                    atomicAdd(&shared_sums[c * d + dim], partial_sums[b * k * d + c * d + dim]);
                }
            }
        }
    }
    __syncthreads();

    // Final division to compute centroids
    if (threadIdx.x < k)
    {
        int c = threadIdx.x;
        if (shared_counts[c] > 0)
        {
            float inv_count = 1.0f / shared_counts[c];
            for (int dim = 0; dim < d; dim++)
            {
                final_centroids[c * d + dim] = shared_sums[c * d + dim] * inv_count;
            }
        }
    }
}
