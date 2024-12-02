//
// Created by wookie on 11/22/24.
//

#include <chrono>
#include "wrappers.h"
#include "kernels.h"

void cpu_kmeans(float *data, float *centroids, int *labels, int n, int d, int k, int max_iter) {
    printf("Running CPU k-means\n");
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    float *new_centroids = new float[k * d]();
    int *counts = new int[k]();

    bool changes = true;
    for (int i = 0; i < max_iter && changes; ++i) {
        changes = false;

        // Labeling
        for (int j = 0; j < n; ++j) {
            float min_dist = INFINITY;
            int min_idx = -1;
            for (int l = 0; l < k; ++l) {
                float dist = 0;
                for (int m = 0; m < d; ++m) {
                    dist += (data[j * d + m] - centroids[l * d + m]) * (data[j * d + m] - centroids[l * d + m]);
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = l;
                }
            }
            if (labels[j] != min_idx) {
                labels[j] = min_idx;
                changes = true;
            }
            for (int l = 0; l < d; ++l) {
                new_centroids[min_idx * d + l] += data[j * d + l];
            }
            counts[min_idx]++;
        }

        // Centroid update
        for (int j = 0; j < k; ++j) {
            if (counts[j] == 0) {
                continue;
            }
            for (int l = 0; l < d; ++l) {
                new_centroids[j * d + l] /= counts[j];
            }
        }

        // Copy new centroids
        for (int j = 0; j < k * d; ++j) {
            centroids[j] = new_centroids[j];
            new_centroids[j] = 0;
        }

        // Reset counts
        for (int j = 0; j < k; ++j) {
            counts[j] = 0;
        }
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("CPU k-means took %ld ms\n", duration);


    delete[] new_centroids;
    delete[] counts;
}

void naive_kmeans(float *data, float *centroids, int *labels, int n, int d, int k, int max_iter) {
    float *d_data, *d_centroids;
    int *d_labels;
    bool *d_changes;

    CHECK_CUDA_ERROR(cudaMalloc(&d_data, n * d * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, k * d * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_changes, sizeof(bool)));

    printf("Copying data to device\n");
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));

    CHECK_CUDA_ERROR(cudaMemcpy(d_data, data, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, centroids, k * d * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Copying data to device took %f ms\n", milliseconds);

    printf("Running naive k-means\n");

    dim3 block(32 * 32);
    dim3 grid(CEIL_DIV(n, block.x));

    float label_time_ms = 0;
    float centroid_time_ms = 0;

    bool changes = true;
    for (int i = 0; i < max_iter && changes; ++i) {
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        naive_labeling<<<grid, block>>>(d_data, d_centroids, d_labels, n, d, k);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        label_time_ms += milliseconds;
        CHECK_LAST_CUDA_ERROR();

        CHECK_CUDA_ERROR(cudaEventRecord(start));
        naive_centroid_update<<<1, k>>>(d_data, d_labels, d_centroids, n, d, k, d_changes);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        centroid_time_ms += milliseconds;
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaMemcpy(&changes, d_changes, sizeof(bool), cudaMemcpyDeviceToHost));
    }

    printf("Labeling took %f ms\n", label_time_ms);
    printf("Centroid update took %f ms\n", centroid_time_ms);

    CHECK_CUDA_ERROR(cudaMemcpy(labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(centroids, d_centroids, k * d * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_centroids));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_changes));
}

