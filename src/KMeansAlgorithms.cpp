//
// Created by wookie on 11/22/24.
//

#include "KMeansAlgorithms.h"
#include "PerformanceClock.h"
#include "Point.cuh"

void KMeansAlgorithms::Cpu(const float *data, float *centroids, int *labels, int n, int d, int k)
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

void KMeansAlgorithms::ThrustVersion(float *h_data, float *h_centroids, int *h_labels, int n, int d, int k)
{
    PerformanceClock clock;
    clock.start(MEASURED_PHASE::TOTAL);

    // Allocate device memory for data, centroids, labels, and convergence flag
    clock.start(MEASURED_PHASE::DATA_TRANSFER);

    float *d_data, *d_centroids;
    int *d_labels;
    bool *d_did_change;

    size_t data_size = n * d * sizeof(float);
    size_t centroids_size = k * d * sizeof(float);
    size_t labels_size = n * sizeof(int);

    CHECK_CUDA_ERROR(cudaMalloc(&d_data, data_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, centroids_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, labels_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_did_change, sizeof(bool)));

    // Copy data and initial centroids to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_size_label = k * d * sizeof(float); // For labeling kernel

    // Host flag for convergence
    auto h_did_change = new bool;
    clock.stop(MEASURED_PHASE::DATA_TRANSFER);

    // Keep old for counting changes - debug only
    auto old_labels = new int[n];
    memcpy(old_labels, h_labels, n * sizeof(int));

    // Create device pointer for data and labels
    thrust::device_ptr<float> d_data_ptr(d_data);
    thrust::device_ptr<int> d_labels_ptr(d_labels);

    // Pre-allocate raw device memory for temporary arrays
    int *d_labels_temp_raw;
    float* d_data_temp_raw;
    int *d_keys_out_raw;
    float* d_sums_raw;
    int *d_counts_raw;

    CHECK_CUDA_ERROR(cudaMalloc(&d_labels_temp_raw, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_data_temp_raw, n * d * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys_out_raw, k * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sums_raw, k * d * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_counts_raw, k * sizeof(int)));

    // Create thrust device pointers for the allocated arrays
    thrust::device_ptr<int> d_labels_temp(d_labels_temp_raw);
    thrust::device_ptr<float> d_data_temp(d_data_temp_raw);
    thrust::device_ptr<int> d_keys_out(d_keys_out_raw);
    thrust::device_ptr<float> d_sums(d_sums_raw);
    thrust::device_ptr<int> d_counts(d_counts_raw);

    for (int iter = 1; iter <= max_iter; ++iter)
    {
        // Reset did_change flag
        clock.start(MEASURED_PHASE::DATA_TRANSFER);
        CHECK_CUDA_ERROR(cudaMemset(d_did_change, 0, sizeof(bool)));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER);

        // Assign labels
        clock.start(MEASURED_PHASE::LABEL_ASSIGNMENT);
        shmem_labeling<<<blocks_per_grid, threads_per_block, shared_mem_size_label>>>(
                d_data, d_centroids, d_labels, d_did_change, n, d, k);
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        clock.stop(MEASURED_PHASE::LABEL_ASSIGNMENT);

        // Check convergence
        clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
        CHECK_CUDA_ERROR(cudaMemcpy(h_did_change, d_did_change, sizeof(bool), cudaMemcpyDeviceToHost));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);
        if (!*h_did_change)
        {
            // No changes mean convergence
            break;
        }

        // Copy labels back to host for debugging
        CHECK_CUDA_ERROR(cudaMemcpy(h_labels, d_labels, labels_size, cudaMemcpyDeviceToHost));
        int changed_count = 0;
        for (int i = 0; i < n; i++)
        {
            if (old_labels[i] != h_labels[i])
                changed_count++;
        }
        printf("Iteration %d: %d points changed their cluster\n", iter, changed_count);

        // Compute new centroids using thrust reduce_by_key without disturbing original order

        // Copy labels and points into temporary arrays
        clock.start(MEASURED_PHASE::DATA_TRANSFER);
        thrust::copy(thrust::device, d_labels_ptr, d_labels_ptr + n, d_labels_temp);
        thrust::copy(thrust::device, d_data_ptr, d_data_ptr + n * d, d_data_temp);
        clock.stop(MEASURED_PHASE::DATA_TRANSFER);

        clock.start(MEASURED_PHASE::CENTROID_UPDATE);
        // Sort points by cluster label (in the temporary arrays)
        thrust::sort_by_key(thrust::device, d_labels_temp, d_labels_temp + n, d_data_temp);

        // First reduction: sum up all points in each cluster
        auto end_pair = thrust::reduce_by_key(thrust::device, d_labels_temp, d_labels_temp + n, d_data_temp, d_keys_out, d_sums);
        int num_clusters = (int)(end_pair.first - d_keys_out);

        // Second reduction: count how many points are in each cluster
        thrust::fill(thrust::device, d_counts, d_counts + k, 0);
        thrust::reduce_by_key(thrust::device, d_labels_temp, d_labels_temp + n, thrust::constant_iterator<int>(1), d_keys_out, d_counts);

        // Copy results back to host
        std::vector<int> h_keys_out(num_clusters);
        std::vector<float> h_sums(num_clusters * d);
        std::vector<int> h_counts(k);

        CHECK_CUDA_ERROR(
                cudaMemcpy(h_keys_out.data(), d_keys_out_raw, num_clusters * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(
                cudaMemcpy(h_sums.data(), d_sums_raw, num_clusters * d * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(
                cudaMemcpy(h_counts.data(), d_counts_raw, k * sizeof(int), cudaMemcpyDeviceToHost));

        // Compute new centroids
        for (int cluster_id = 0; cluster_id < k; ++cluster_id)
        {
            auto it = std::find(h_keys_out.begin(), h_keys_out.end(), cluster_id);
            if (it != h_keys_out.end())
            {
                int idx = (int)(it - h_keys_out.begin());
                int c   = h_counts[idx];
                if (c > 0)
                {
                    for (int dim = 0; dim < d; ++dim)
                    {
                        h_centroids[cluster_id * d + dim] = h_sums[idx * d + dim] / c;
                    }
                }
            }
            // If no points, keep old centroid
        }

        // Copy updated centroids back to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));
        clock.stop(MEASURED_PHASE::CENTROID_UPDATE);

        memcpy(old_labels, h_labels, n * sizeof(int)); // Not in measured time since used only for debugging
    }

    // Copy final centroids back to host
    clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost));
    clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);

    clock.stop(MEASURED_PHASE::TOTAL);
    clock.printResults("Thrust-based k-means");

    // Cleanup
    delete h_did_change;

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_did_change);
}
