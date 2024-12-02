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
