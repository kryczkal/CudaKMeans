//
// Created by wookie on 11/22/24.
//

#include "KMeansWrappers.h"
#include "PerformanceClock.h"
#include "kernels.h"
#include <chrono>

void KMeansWrappers::Cpu(const float *data, float *centroids, int *labels, int n, int d, int k, int max_iter)
{
    PerformanceClock clock;

    auto new_centroids = new float[k * d]();
    auto counts        = new float[k]();

    clock.start(MEASURED_PHASE::CPU_COMPUTATION);
    bool changes = true;
    for (int i = 0; i < max_iter && changes; ++i)
    {
        changes = false;

        // Labeling
        for (int j = 0; j < n; ++j)
        {
            float min_dist = INFINITY;
            int min_idx    = -1;
            for (int l = 0; l < k; ++l)
            {
                float dist = 0;
                for (int m = 0; m < d; ++m)
                {
                    dist += (data[j * d + m] - centroids[l * d + m]) * (data[j * d + m] - centroids[l * d + m]);
                }
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_idx  = l;
                }
            }
            if (labels[j] != min_idx)
            {
                labels[j] = min_idx;
                changes   = true;
            }
            for (int l = 0; l < d; ++l)
            {
                new_centroids[min_idx * d + l] += data[j * d + l];
            }
            counts[min_idx]++;
        }

        // Centroid update
        for (int j = 0; j < k; ++j)
        {
            if (counts[j] == 0)
            {
                continue;
            }
            for (int l = 0; l < d; ++l)
            {
                new_centroids[j * d + l] /= (float)counts[j];
            }
        }

        // Copy new centroids
        for (int j = 0; j < k * d; ++j)
        {
            centroids[j]     = new_centroids[j];
            new_centroids[j] = 0;
        }

        // Reset counts
        for (int j = 0; j < k; ++j)
        {
            counts[j] = 0;
        }
    }
    clock.stop(MEASURED_PHASE::CPU_COMPUTATION);
    clock.printResults("CPU k-means");

    delete[] new_centroids;
    delete[] counts;
}

void KMeansWrappers::Naive(float *data, float *centroids, int *labels, int n, int d, int k, int max_iter)
{
    PerformanceClock clock;
    float *d_data, *d_centroids;
    int *d_labels;
    bool *d_changes;

    CHECK_CUDA_ERROR(cudaMalloc(&d_data, n * d * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, k * d * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_changes, sizeof(bool)));

    clock.start(MEASURED_PHASE::DATA_TRANSFER);
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, data, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, centroids, k * d * sizeof(float), cudaMemcpyHostToDevice));
    clock.stop(MEASURED_PHASE::DATA_TRANSFER);

    printf("Running naive k-means\n");

    dim3 block(32 * 32);
    dim3 grid(CEIL_DIV(n, block.x));

    float label_time_ms    = 0;
    float centroid_time_ms = 0;

    bool changes = true;
    for (int i = 0; i < max_iter && changes; ++i)
    {
        clock.start(MEASURED_PHASE::KERNEL);
        naive_labeling<<<grid, block>>>(d_data, d_centroids, d_labels, n, d, k);
        CHECK_LAST_CUDA_ERROR();

        naive_centroid_update<<<1, k>>>(d_data, d_labels, d_centroids, n, d, k, d_changes);
        CHECK_LAST_CUDA_ERROR();
        clock.stop(MEASURED_PHASE::KERNEL);

        clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
        CHECK_CUDA_ERROR(cudaMemcpy(&changes, d_changes, sizeof(bool), cudaMemcpyDeviceToHost));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);
    }

    printf("Labeling took %f ms\n", label_time_ms);
    printf("Centroid update took %f ms\n", centroid_time_ms);

    clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
    CHECK_CUDA_ERROR(cudaMemcpy(labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(centroids, d_centroids, k * d * sizeof(float), cudaMemcpyDeviceToHost));
    clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);

    clock.printResults("Naive k-means");

    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_centroids));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_changes));
}

void KMeansWrappers::AtomicAddShmem(float *h_data, float *h_centroids, int *h_labels, int n, int d, int k, int max_iter)
{
    PerformanceClock clock;

    float *d_data, *d_centroids;
    int *d_labels, *d_counts;
    bool *d_did_change;

    size_t data_size      = n * d * sizeof(float);
    size_t centroids_size = k * d * sizeof(float);
    size_t labels_size    = n * sizeof(int);
    size_t counts_size    = k * sizeof(int);

    CHECK_CUDA_ERROR(cudaMalloc(&d_data, data_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, centroids_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, labels_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_counts, counts_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_did_change, sizeof(bool)));

    clock.start(MEASURED_PHASE::DATA_TRANSFER);
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));
    clock.stop(MEASURED_PHASE::DATA_TRANSFER);

    int threads_per_block = 256;
    int blocks_per_grid   = (n + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_size_label  = k * d * sizeof(float);                       // For labeling kernel
    size_t shared_mem_size_update = (k * d * sizeof(float)) + (k * sizeof(int)); // For centroid update

    // Host arrays for counting points in each cluster - Debug only
    auto counts = new int[k];

    // Host convergence flag
    auto h_did_change = new bool;

    auto old_labels = new int[n];
    memcpy(old_labels, h_labels, n * sizeof(int));

    for (int iter = 1; iter <= max_iter; ++iter)
    {
        // Labeling step
        // Reset did_change flag
        clock.start(MEASURED_PHASE::DATA_TRANSFER);
        CHECK_CUDA_ERROR(cudaMemset(d_did_change, 0, sizeof(bool)));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER);
        // Assign labels
        clock.start(MEASURED_PHASE::KERNEL);
        shmem_labeling<<<blocks_per_grid, threads_per_block, shared_mem_size_label>>>(
            d_data, d_centroids, d_labels, d_did_change, n, d, k
        );
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        clock.stop(MEASURED_PHASE::KERNEL);

        // Convergence check
        clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
        CHECK_CUDA_ERROR(cudaMemcpy(h_did_change, d_did_change, sizeof(bool), cudaMemcpyDeviceToHost));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);
        if (!*h_did_change)
        {
            break;
        }

        // Copy labels back to host to check how many changed
        // I do not count this in the measured time since it is not necessary for the convergence check
        // And only here for debugging purposes
        CHECK_CUDA_ERROR(cudaMemcpy(h_labels, d_labels, labels_size, cudaMemcpyDeviceToHost));

        int changed_count = 0;
        for (int i = 0; i < n; i++)
        {
            if (h_labels[i] != old_labels[i])
            {
                changed_count++;
            }
        }
        // Print iteration info
        printf("Iteration %d: %d points changed their cluster\n", iter, changed_count);

        // Reset centroids and counts on device before centroid update
        clock.start(MEASURED_PHASE::KERNEL);
        CHECK_CUDA_ERROR(cudaMemset(d_centroids, 0, centroids_size));
        CHECK_CUDA_ERROR(cudaMemset(d_counts, 0, counts_size));

        // Centroid update step
        atomic_add_shmem_centroid_update<<<blocks_per_grid, threads_per_block, shared_mem_size_update>>>(
            d_data, d_labels, d_centroids, d_counts, n, d, k
        );
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        clock.stop(MEASURED_PHASE::KERNEL);

        // Copy counts and centroids to host
        clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
        CHECK_CUDA_ERROR(cudaMemcpy(counts, d_counts, counts_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);

        clock.start(MEASURED_PHASE::CPU_COMPUTATION);
        // Update centroids on host
        // If h_counts[i] > 0 , compute new centroid
        // If h_counts[i] == 0, keep the old centroid
        for (int i = 0; i < k; ++i)
        {
            if (counts[i] > 0)
            {
                for (int j = 0; j < d; ++j)
                {
                    h_centroids[i * d + j] /= (float)counts[i];
                }
            }
        }
        clock.stop(MEASURED_PHASE::CPU_COMPUTATION);

        // Copy updated centroids back to device for next iteration
        clock.start(MEASURED_PHASE::DATA_TRANSFER);
        CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER);

        memcpy(old_labels, h_labels, n * sizeof(int)); // Not in measured time since used only for debugging
    }
    // Copy centroids back to host
    clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost));
    clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);

    clock.printResults("Reduction v1 k-means atomicAdd version");

    delete[] counts;
    delete[] old_labels;

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
}