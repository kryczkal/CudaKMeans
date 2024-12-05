//
// Created by wookie on 11/22/24.
//

#include <iostream>
#include <cuda_runtime.h>
#include "CudaUtils.h"
#include <cinttypes>
#include <format>

#include "profiler.h"
#include "kernels.h"
#include "wrappers.h"
#include "GeneralUtils.h"

static constexpr uint64_t N = 1e8;
static constexpr uint64_t D = 5;
static constexpr uint64_t K = 4;
int main() {
    CudaUtils::printCudaDeviceInfo();

    double total_size_bytes = N * D * sizeof(float) + K * D * sizeof(float) + N * sizeof(int);
    if (!GeneralUtils::fitsInGlobalMemory(total_size_bytes)) {
        std::cerr << "Requested data: " << total_size_bytes / 1e9 << " GB" << " does not fit into global memory" << std::endl;
    }

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

//    cpu_kmeans(h_data, h_centroids, h_labels, N, D, K, 100);
//    visualize_kmeans(h_data, h_centroids, h_labels, N, D, K, 80, 24);
//    naive_kmeans(h_data, h_centroids, h_labels, N, D, K, 100);
//    visualize_kmeans(h_data, h_centroids, h_labels, N, D, K, 80, 24);
    reduction_v1_kmeans(h_data, h_centroids, h_labels, N, D, K, 100);
    GeneralUtils::visualizeKmeans(h_data, h_centroids, h_labels, N, D, K, 80, 24);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    free(h_data);
    free(h_centroids);
    free(h_labels);

    return EXIT_SUCCESS;
}