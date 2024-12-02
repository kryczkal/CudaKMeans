//
// Created by wookie on 11/22/24.
//

#include <iostream>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <cinttypes>

#include "profiler.h"
#include "kernels.h"
#include "wrappers.h"
#include "utils.h"

static constexpr uint64_t N = 1e5;
static constexpr uint64_t D = 2;
static constexpr uint64_t K = 3;
int main() {
    cudaDeviceInfo();

    float* h_data, *h_centroids;
    int* h_labels;

    h_data = static_cast<float *>(malloc(N * D * sizeof(float)));
    h_centroids = static_cast<float *>(malloc(K * D * sizeof(float)));
    h_labels = static_cast<int *>(malloc(N * sizeof(int)));

    srand(time(nullptr));
    for (int i = 0; i < N * D; ++i) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * D; ++i) {
        // Pick first K points as centroids
        h_centroids[i] = h_data[i];
    }

    cpu_kmeans(h_data, h_centroids, h_labels, N, D, K, 100);
    visualize_kmeans(h_data, h_centroids, h_labels, N, D, K, 80, 24);
    naive_kmeans(h_data, h_centroids, h_labels, N, D, K, 100);
    visualize_kmeans(h_data, h_centroids, h_labels, N, D, K, 80, 24);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    free(h_data);
    free(h_centroids);
    free(h_labels);

    return EXIT_SUCCESS;
}