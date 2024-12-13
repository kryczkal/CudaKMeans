//
// Created by wookie on 11/22/24.
//

#include "KMeansAlgorithms.h"
#include "PerformanceClock.h"

#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

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

    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));
    clock.stop(MEASURED_PHASE::DATA_TRANSFER);

    int threads_per_block = 256;
    int blocks_per_grid   = (n + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_size_label = k * d * sizeof(float);

    auto h_did_change = new bool;
    auto old_labels   = new int[n];
    memcpy(old_labels, h_labels, n * sizeof(int));

    // Initialize device labels to -1
    CHECK_CUDA_ERROR(cudaMemset(d_labels, -1, labels_size));

    thrust::device_ptr<float> d_data_ptr(d_data);
    thrust::device_ptr<int> d_labels_ptr(d_labels);

    thrust::device_vector<int> d_labels_temp(n);
    // Create a device vector to store rearranged points
    thrust::device_vector<float> d_points_temp(n * d);
    thrust::device_vector<int> d_indices(n); // used for permutations

    // For summation and key extraction after sorting
    thrust::device_vector<int> d_keys_out(k);
    thrust::device_vector<float> d_sums_dim(k);
    thrust::device_vector<int> d_counts(k);

    for (int iter = 1; iter <= max_iter; ++iter)
    {
        // Reset did_change
        clock.start(MEASURED_PHASE::DATA_TRANSFER);
        CHECK_CUDA_ERROR(cudaMemset(d_did_change, 0, sizeof(bool)));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER);

        // Labeling kernel
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
            break;
        }

        // Copy labels back to host - not needed for computation but useful for debugging
        CHECK_CUDA_ERROR(cudaMemcpy(h_labels, d_labels, labels_size, cudaMemcpyDeviceToHost));
        int changed_count = 0;
        for (int i = 0; i < n; i++)
        {
            if (h_labels[i] != old_labels[i])
                changed_count++;
        }
        printf("Iteration %d: %d points changed their cluster\n", iter, changed_count);

        // Prepare data for centroid recomputation

        clock.start(MEASURED_PHASE::DATA_TRANSFER);
        // Copy current labels into d_labels_temp
        thrust::copy(d_labels_ptr, d_labels_ptr + n, d_labels_temp.begin());

        // Create a sequence of indices [0, 1, 2, ..., n-1]
        thrust::sequence(d_indices.begin(), d_indices.end(), 0);
        clock.stop(MEASURED_PHASE::DATA_TRANSFER);

        clock.start(MEASURED_PHASE::THRUST);
        // Sort the indices by labels - this is used to later rearrange points
        thrust::sort_by_key(d_labels_temp.begin(), d_labels_temp.end(), d_indices.begin());

        // Rearrange points dimension-by-dimension using a transform iterator to map [0..n-1] -> [original_index*d +
        // dim]
        for (int dim = 0; dim < d; ++dim)
        {
            // Create a transform iterator that maps an index i to i*d + dim
            auto dim_map_iter = thrust::make_transform_iterator(
                thrust::counting_iterator<int>(0),
                [=] __host__ __device__(int idx)
                {
                    return idx * d + dim;
                }
            );

            // Create a permutation iterator that picks the correct dim-th coordinate of each point:
            auto dim_data_iter = thrust::make_permutation_iterator(d_data_ptr, dim_map_iter);

            // Rearrange points according to sorted labels
            thrust::gather(d_indices.begin(), d_indices.end(), dim_data_iter, d_points_temp.begin() + dim * n);
        }

        // Now d_points_temp is stored in "dimension-major" form:
        // For dim = 0, coordinates are at d_points_temp[0...n-1]
        // For dim = 1, coordinates are at d_points_temp[n...2n-1], etc.
        // Corresponding labels for each point are in d_labels_temp (sorted).

        // First, get counts per cluster:
        thrust::fill(d_counts.begin(), d_counts.end(), 0);
        thrust::reduce_by_key(
            d_labels_temp.begin(), d_labels_temp.end(), thrust::constant_iterator<int>(1),
            thrust::make_discard_iterator(), d_counts.begin()
        );
        clock.stop(MEASURED_PHASE::THRUST);

        // Reduce by key, dimension by dimension to sum coordinates by cluster.
        // For each dim:
        // - Use a slice of d_points_temp as values
        // - Reduce by key using d_labels_temp as keys
        // - Store sums in a temporary vector and then combine

        // We'll store partial results on the host
        clock.start(MEASURED_PHASE::DATA_TRANSFER);
        thrust::host_vector<int> h_counts_host = d_counts;
        thrust::host_vector<int> h_keys_host(k);
        thrust::host_vector<float> h_sums_for_dim(k);
        clock.stop(MEASURED_PHASE::DATA_TRANSFER);

        // Perform the dimension-wise reductions
        for (int dim = 0; dim < d; ++dim)
        {
            // reduce_by_key for this dimension
            clock.start(MEASURED_PHASE::THRUST);
            auto values_begin = d_points_temp.begin() + dim * n;
            auto end_pair     = thrust::reduce_by_key(
                d_labels_temp.begin(), d_labels_temp.end(), values_begin, d_keys_out.begin(), d_sums_dim.begin()
            );

            int num_clusters_found = (int)(end_pair.first - d_keys_out.begin());
            clock.stop(MEASURED_PHASE::THRUST);
            // Copy results to host
            clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
            thrust::host_vector<int> h_keys_out   = d_keys_out;
            thrust::host_vector<float> h_sums_dim = d_sums_dim;
            clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);

            // Accumulate dimension sums into h_centroids
            clock.start(MEASURED_PHASE::CPU_COMPUTATION);
            for (int cluster_id = 0; cluster_id < k; ++cluster_id)
            {
                int c = h_counts_host[cluster_id];
                if (c > 0)
                {
                    // If a cluster has no points (is not found in keys), skip it
                    auto it = std::find(h_keys_out.begin(), h_keys_out.begin() + num_clusters_found, cluster_id);
                    if (it != h_keys_out.begin() + num_clusters_found)
                    {
                        int idx                           = (int)(it - h_keys_out.begin());
                        float sum_dim                     = h_sums_dim[idx];
                        h_centroids[cluster_id * d + dim] = sum_dim / (float)c;
                    }
                }
            }
            clock.stop(MEASURED_PHASE::CPU_COMPUTATION);
        }

        // Copy updated centroids back to device
        clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
        CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));
        clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);

        // Update old_labels for debugging
        memcpy(old_labels, h_labels, n * sizeof(int));
    }

    // Copy final centroids back to host
    clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost));
    clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);

    clock.printResults("Thrust-based k-means");

    delete[] old_labels;
    delete h_did_change;

    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_centroids));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_did_change));
}
