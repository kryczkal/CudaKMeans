//
// Created by wookie on 11/22/24.
//

#include "KMeansAlgorithms.h"
#include "PerformanceClock.h"
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

// Helper to calculate optimal number of points per block based on available shared memory
__host__ int calculate_optimal_points_per_block(int d, int k)
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    size_t shared_mem_per_block = props.sharedMemPerBlock;
    size_t histogram_size       = k * d * sizeof(float) + k * sizeof(int);
    // Leave some shared memory for other purposes
    size_t available_shared_mem = shared_mem_per_block * 0.8;

    // Calculate maximum points that can fit in a block
    int max_points = (available_shared_mem - histogram_size) / (sizeof(float) * d);

    // Round down to nearest multiple of 32 for coalescing
    max_points = (max_points / 32) * 32;

    // Limit maximum points to avoid too large blocks
    return std::min(max_points, 1024);
}

void update_centroids_tree_reduction(const float *d_data, const int *d_labels, float *d_centroids, int n, int d, int k)
{
    // Calculate optimal parameters
    int points_per_block        = calculate_optimal_points_per_block(d, k);
    const int threads_per_block = 256;
    const int num_blocks        = (n + points_per_block - 1) / points_per_block;

    // Allocate memory for partial results
    float *d_partial_sums;
    int *d_partial_counts;
    CHECK_CUDA_ERROR(cudaMalloc(&d_partial_sums, num_blocks * k * d * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_partial_counts, num_blocks * k * sizeof(int)));

    // Initialize memory
    CHECK_CUDA_ERROR(cudaMemset(d_partial_sums, 0, num_blocks * k * d * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_partial_counts, 0, num_blocks * k * sizeof(int)));

    // Shared memory size for local histograms
    size_t shared_mem_size = k * d * sizeof(float) + k * sizeof(int);

    // Compute local histograms
    compute_local_histograms<<<num_blocks, threads_per_block, shared_mem_size>>>(
        d_data, d_labels, d_partial_sums, d_partial_counts, n, d, k, points_per_block
    );
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    // Tree reduction
    tree_reduce_centroids<<<1, threads_per_block, shared_mem_size>>>(
        d_partial_sums, d_partial_counts, d_centroids, num_blocks, d, k
    );
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    // Cleanup
    cudaFree(d_partial_sums);
    cudaFree(d_partial_counts);
}

void KMeansAlgorithms::TreeReduction(float *h_data, float *h_centroids, int *h_labels, int n, int d, int k)
{
    PerformanceClock clock;
    clock.start(MEASURED_PHASE::TOTAL);

    // Allocate device memory
    clock.start(MEASURED_PHASE::DATA_TRANSFER);
    float *d_data, *d_centroids;
    int *d_labels;
    bool *d_did_change;

    size_t data_size      = n * d * sizeof(float);
    size_t centroids_size = k * d * sizeof(float);
    size_t labels_size    = n * sizeof(int);

    CHECK_CUDA_ERROR(cudaMalloc(&d_data, data_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, centroids_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, labels_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_did_change, sizeof(bool)));

    // Copy data and initial centroids to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));

    int threads_per_block        = 256;
    int blocks_per_grid          = (n + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size_label = k * d * sizeof(float);

    // Host flag for convergence
    auto h_did_change = new bool;
    clock.stop(MEASURED_PHASE::DATA_TRANSFER);

    // Keep old labels for counting changes - debug only
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
        shmem_labeling<<<blocks_per_grid, threads_per_block, shared_mem_size_label>>>(
            d_data, d_centroids, d_labels, d_did_change, n, d, k
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

        // Copy labels back to host for debugging
        CHECK_CUDA_ERROR(cudaMemcpy(h_labels, d_labels, labels_size, cudaMemcpyDeviceToHost));
        int changed_count = 0;
        for (int i = 0; i < n; i++)
        {
            if (h_labels[i] != old_labels[i])
                changed_count++;
        }
        printf("Iteration %d: %d points changed their cluster\n", iter, changed_count);

        // Update centroids using tree reduction
        clock.start(MEASURED_PHASE::CENTROID_UPDATE);
        update_centroids_tree_reduction(d_data, d_labels, d_centroids, n, d, k);
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        clock.stop(MEASURED_PHASE::CENTROID_UPDATE);

        memcpy(old_labels, h_labels, n * sizeof(int));
    }

    // Copy final results back to host
    clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_labels, d_labels, labels_size, cudaMemcpyDeviceToHost));
    clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);

    clock.stop(MEASURED_PHASE::TOTAL);
    clock.printResults("Tree-based k-means reduction");

    // Cleanup
    delete[] old_labels;
    delete h_did_change;

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_did_change);
}

#define D 4
void KMeansAlgorithms::ThrustVersion(
    float *h_data, float *h_centroids, int *h_labels, int n, int d, int k, int max_iter
)
{
    if (d != D)
    {
        throw std::runtime_error("Thrust version only supports 4 dimensions");
    }

    PerformanceClock clock;

    // Allocate device memory
    float *d_data, *d_centroids;
    int *d_labels;
    bool *d_did_change;

    size_t data_size      = n * d * sizeof(float);
    size_t centroids_size = k * d * sizeof(float);
    size_t labels_size    = n * sizeof(int);

    clock.start(MEASURED_PHASE::DATA_TRANSFER);
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, data_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, centroids_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, labels_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_did_change, sizeof(bool)));

    // Copy data and initial centroids to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));
    clock.stop(MEASURED_PHASE::DATA_TRANSFER);

    int threads_per_block = 256;
    int blocks_per_grid   = (n + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_size_label = k * d * sizeof(float); // For labeling kernel

    // Host flag for convergence
    auto h_did_change = new bool;

    // Keep old labels for counting changes - debug only
    auto old_labels = new int[n];
    memcpy(old_labels, h_labels, n * sizeof(int));

    // Initialize device labels to 0
    CHECK_CUDA_ERROR(cudaMemset(d_labels, 0, labels_size));

    clock.start(MEASURED_PHASE::DATA_TRANSFER);
    // Wrap raw pointers with thrust device pointers
    thrust::device_ptr<float> d_data_ptr(d_data);
    thrust::device_ptr<int> d_labels_ptr(d_labels);
    auto *d_points = reinterpret_cast<Point<D> *>(thrust::raw_pointer_cast(d_data_ptr));
    thrust::device_ptr<Point<D>> d_points_ptr(d_points);

    // Temporary vectors used for sorting and reducing each iteration
    thrust::device_vector<int> d_labels_temp(n);
    thrust::device_vector<Point<D>> d_points_temp(n);
    thrust::device_vector<int> d_keys_out(k);
    thrust::device_vector<Point<D>> d_sums(k);
    thrust::device_vector<int> d_counts(k);
    clock.stop(MEASURED_PHASE::DATA_TRANSFER);

    for (int iter = 1; iter <= max_iter; ++iter)
    {
        // Reset did_change
        clock.start(MEASURED_PHASE::DATA_TRANSFER);
        CHECK_CUDA_ERROR(cudaMemset(d_did_change, 0, sizeof(bool)));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER);

        // Labeling step
        clock.start(MEASURED_PHASE::KERNEL);
        shmem_labeling<<<blocks_per_grid, threads_per_block, shared_mem_size_label>>>(
            d_data, d_centroids, d_labels, d_did_change, n, d, k
        );
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        clock.stop(MEASURED_PHASE::KERNEL);

        // Check convergence
        clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
        CHECK_CUDA_ERROR(cudaMemcpy(h_did_change, d_did_change, sizeof(bool), cudaMemcpyDeviceToHost));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);

        if (!*h_did_change)
        {
            // No changes means convergence
            break;
        }

        // Copy labels back to host for debugging
        CHECK_CUDA_ERROR(cudaMemcpy(h_labels, d_labels, labels_size, cudaMemcpyDeviceToHost));
        int changed_count = 0;
        for (int i = 0; i < n; i++)
        {
            if (h_labels[i] != old_labels[i])
                changed_count++;
        }
        printf("Iteration %d: %d points changed their cluster\n", iter, changed_count);

        // Compute new centroids using thrust reduce_by_key without disturbing original order
        clock.start(MEASURED_PHASE::THRUST);

        // Copy original data and labels into temporary buffers
        thrust::copy(thrust::device, d_labels_ptr, d_labels_ptr + n, d_labels_temp.begin());
        thrust::copy(thrust::device, d_points_ptr, d_points_ptr + n, d_points_temp.begin());

        // Sort points by cluster label (in temporary arrays)
        thrust::sort_by_key(thrust::device, d_labels_temp.begin(), d_labels_temp.end(), d_points_temp.begin());

        // First reduction: sum up all points per cluster
        auto end_pair = thrust::reduce_by_key(
            thrust::device, d_labels_temp.begin(), d_labels_temp.end(), d_points_temp.begin(), d_keys_out.begin(),
            d_sums.begin()
        );
        int num_clusters_found = (int)(end_pair.first - d_keys_out.begin());

        // Second reduction: count how many points per cluster
        // Since each point contributes '1', we can reuse a constant iterator
        thrust::fill(d_counts.begin(), d_counts.end(), 0);
        thrust::reduce_by_key(
            thrust::device, d_labels_temp.begin(), d_labels_temp.end(), thrust::constant_iterator<int>(1),
            thrust::make_discard_iterator(), d_counts.begin()
        );

        // Copy results back to host
        thrust::host_vector<int> h_keys_out       = d_keys_out;
        thrust::host_vector<Point<D>> h_sums_host = d_sums;
        thrust::host_vector<int> h_counts_host    = d_counts;
        clock.stop(MEASURED_PHASE::THRUST);

        // Compute final centroids on the host
        clock.start(MEASURED_PHASE::CPU_COMPUTATION);
        for (int cluster_id = 0; cluster_id < k; ++cluster_id)
        {
            // Find this cluster in h_keys_out
            auto it = std::find(h_keys_out.begin(), h_keys_out.begin() + num_clusters_found, cluster_id);
            if (it != h_keys_out.begin() + num_clusters_found)
            {
                int idx = (int)(it - h_keys_out.begin());
                int c   = h_counts_host[idx];
                if (c > 0)
                {
                    for (int dim = 0; dim < d; ++dim)
                    {
                        h_centroids[cluster_id * d + dim] = h_sums_host[idx].coords[dim] / (float)c;
                    }
                }
                // If cluster is empty, keep old centroid (no change)
            }
            else
            {
                // No points in this cluster, do not change centroid
            }
        }
        clock.stop(MEASURED_PHASE::CPU_COMPUTATION);

        // Copy updated centroids back to device
        clock.start(MEASURED_PHASE::DATA_TRANSFER);
        CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER);

        // Update old_labels for debugging
        memcpy(old_labels, h_labels, n * sizeof(int));
    }

    // Copy final centroids back to host
    clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost));
    clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);

    clock.printResults("Thrust-based k-means");

    // Cleanup
    delete[] old_labels;
    delete h_did_change;

    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_centroids));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_did_change));
}