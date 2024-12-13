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