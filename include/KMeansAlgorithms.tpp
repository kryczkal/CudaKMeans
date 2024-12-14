#include "KMeansAlgorithms.h"
#include "PerformanceClock.h"
#include "kernels.h"

#include "Point.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

template <int D, int K>
void KMeansAlgorithms::AtomicAddShmem(float *h_data, float *h_centroids, int *h_labels, int n)
{
    PerformanceClock clock;
    clock.start(MEASURED_PHASE::TOTAL);

    clock.start(MEASURED_PHASE::DATA_TRANSFER);
    float *d_data, *d_centroids;
    int *d_labels, *d_counts;
    bool *d_did_change;

    size_t data_size      = n * D * sizeof(float);
    size_t centroids_size = K * D * sizeof(float);
    size_t labels_size    = n * sizeof(int);
    size_t counts_size    = K * sizeof(int);

    CHECK_CUDA_ERROR(cudaMalloc(&d_data, data_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, centroids_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, labels_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_counts, counts_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_did_change, sizeof(bool)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));

    int threads_per_block = 64;
    int blocks_per_grid   = (n + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_size_label  = K * D * sizeof(float);                       // For labeling kernel
    size_t shared_mem_size_update = (K * D * sizeof(float)) + (K * sizeof(int)); // For centroid update

    // Host convergence flag
    auto h_did_change = new bool;
    clock.stop(MEASURED_PHASE::DATA_TRANSFER);

    // Host arrays for counting points in each cluster - Debug only
    auto counts = new int[K];

    auto old_labels = new int[n];
    memcpy(old_labels, h_labels, n * sizeof(int));

    for (int iter = 1; iter <= max_iter; ++iter)
    {
        // Reset did_change flag
        clock.start(MEASURED_PHASE::DATA_TRANSFER);
        CHECK_CUDA_ERROR(cudaMemset(d_did_change, 0, sizeof(bool)));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER);

        // Assign labels
        clock.start(MEASURED_PHASE::LABEL_ASSIGNMENT);
        // Non-templated version of this kernel works faster
        shmem_labeling<<<blocks_per_grid, threads_per_block, shared_mem_size_label>>>(
                d_data, d_centroids, d_labels, d_did_change, n, D, K
        );
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        clock.stop(MEASURED_PHASE::LABEL_ASSIGNMENT);

        // Check convergence
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
        clock.start(MEASURED_PHASE::CENTROID_UPDATE);
        CHECK_CUDA_ERROR(cudaMemset(d_centroids, 0, centroids_size));
        CHECK_CUDA_ERROR(cudaMemset(d_counts, 0, counts_size));

        // Centroid update step
        atomic_add_shmem_centroid_update<D, K><<<blocks_per_grid, threads_per_block, shared_mem_size_update>>>(
                d_data, d_labels, d_centroids, d_counts, n
        );
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        clock.stop(MEASURED_PHASE::CENTROID_UPDATE);

        // Copy counts and centroids to host
        clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
        CHECK_CUDA_ERROR(cudaMemcpy(counts, d_counts, counts_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);

        clock.start(MEASURED_PHASE::CENTROID_UPDATE);
        // Update centroids on host
        // If h_counts[i] > 0 , compute new centroid
        // If h_counts[i] == 0, keep the old centroid
        for (int i = 0; i < K; ++i)
        {
            if (counts[i] > 0)
            {
                for (int j = 0; j < D; ++j)
                {
                    h_centroids[i * D + j] /= (float)counts[i];
                }
            }
        }
        clock.stop(MEASURED_PHASE::CENTROID_UPDATE);

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

    clock.stop(MEASURED_PHASE::TOTAL);
    clock.printResults("Reduction v1 k-means atomicAdd version");

    delete[] counts;
    delete[] old_labels;

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
}