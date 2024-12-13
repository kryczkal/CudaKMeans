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

// !!! This version of ThrustVersion - even though is working - has a weird bug
// Templating this function, or using a macro to define it 20 times
// Makes both gpu1 and gpu2 methods to work much slower (around 10x)
// This is why the entire function is commented out
// On the other hand deleting the template and defining D as a constant works fine (but handles the problem only for
// D = some constant)

// I keep it as a proof that I made a simpler version of the thrust-based k-means and that this bug exists
template<int D>
void KMeansAlgorithms::ThrustVersionV2(float *h_data, float *h_centroids, int *h_labels, int n, int k)
{
//    PerformanceClock clock;
//    clock.start(MEASURED_PHASE::TOTAL);
//
//    // Allocate device memory
//    float *d_data, *d_centroids;
//    int *d_labels;
//    bool *d_did_change;
//
//    size_t data_size      = n * D * sizeof(float);
//    size_t centroids_size = k * D * sizeof(float);
//    size_t labels_size    = n * sizeof(int);
//
//    clock.start(MEASURED_PHASE::DATA_TRANSFER);
//    CHECK_CUDA_ERROR(cudaMalloc(&d_data, data_size));
//    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, centroids_size));
//    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, labels_size));
//    CHECK_CUDA_ERROR(cudaMalloc(&d_did_change, sizeof(bool)));
//
//    // Copy data and initial centroids to device
//    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
//    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));
//    clock.stop(MEASURED_PHASE::DATA_TRANSFER);
//
//    int threads_per_block = 256;
//    int blocks_per_grid   = (n + threads_per_block - 1) / threads_per_block;
//
//    size_t shared_mem_size_label = k * D * sizeof(float); // For labeling kernel
//
//    // Host flag for convergence
//    auto h_did_change = new bool;
//
//    // Keep old labels for counting changes - debug only
//    auto old_labels = new int[n];
//    memcpy(old_labels, h_labels, n * sizeof(int));
//
//    // Initialize device labels to 0
//    CHECK_CUDA_ERROR(cudaMemset(d_labels, 0, labels_size));
//
//    clock.start(MEASURED_PHASE::DATA_TRANSFER);
//    // Wrap raw pointers with thrust device pointers
//    thrust::device_ptr<float> d_data_ptr(d_data);
//    thrust::device_ptr<int> d_labels_ptr(d_labels);
//    auto *d_points = reinterpret_cast<Point<D> *>(thrust::raw_pointer_cast(d_data_ptr));
//    thrust::device_ptr<Point<D>> d_points_ptr(d_points);
//
//    // Temporary vectors used for sorting and reducing each iteration
//    thrust::device_vector<int> d_labels_temp(n);
//    thrust::device_vector<Point<D>> d_points_temp(n);
//    thrust::device_vector<int> d_keys_out(k);
//    thrust::device_vector<Point<D>> d_sums(k);
//    thrust::device_vector<int> d_counts(k);
//    clock.stop(MEASURED_PHASE::DATA_TRANSFER);
//
//    for (int iter = 1; iter <= max_iter; ++iter)
//    {
//        // Reset did_change
//        clock.start(MEASURED_PHASE::DATA_TRANSFER);
//        CHECK_CUDA_ERROR(cudaMemset(d_did_change, 0, sizeof(bool)));
//        clock.stop(MEASURED_PHASE::DATA_TRANSFER);
//
//        // Labeling step
//        clock.start(MEASURED_PHASE::KERNEL);
//        shmem_labeling<<<blocks_per_grid, threads_per_block, shared_mem_size_label>>>(
//            d_data, d_centroids, d_labels, d_did_change, n, D, k
//        );
//        cudaDeviceSynchronize();
//        CHECK_LAST_CUDA_ERROR();
//        clock.stop(MEASURED_PHASE::KERNEL);
//
//        // Check convergence
//        clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
//        CHECK_CUDA_ERROR(cudaMemcpy(h_did_change, d_did_change, sizeof(bool), cudaMemcpyDeviceToHost));
//        clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);
//
//        if (!*h_did_change)
//        {
//            // No changes means convergence
//            break;
//        }
//
//        // Copy labels back to host for debugging
//        CHECK_CUDA_ERROR(cudaMemcpy(h_labels, d_labels, labels_size, cudaMemcpyDeviceToHost));
//        int changed_count = 0;
//        for (int i = 0; i < n; i++)
//        {
//            if (h_labels[i] != old_labels[i])
//                changed_count++;
//        }
//        printf("Iteration %d: %d points changed their cluster\n", iter, changed_count);
//
//        // Compute new centroids using thrust reduce_by_key without disturbing original order
//        clock.start(MEASURED_PHASE::THRUST);
//
//        // Copy original data and labels into temporary buffers
//        thrust::copy(thrust::device, d_labels_ptr, d_labels_ptr + n, d_labels_temp.begin());
//        thrust::copy(thrust::device, d_points_ptr, d_points_ptr + n, d_points_temp.begin());
//
//        // Sort points by cluster label (in temporary arrays)
//        thrust::sort_by_key(thrust::device, d_labels_temp.begin(), d_labels_temp.end(), d_points_temp.begin());
//
//        // First reduction: sum up all points per cluster
//        auto end_pair = thrust::reduce_by_key(
//            thrust::device, d_labels_temp.begin(), d_labels_temp.end(), d_points_temp.begin(), d_keys_out.begin(),
//            d_sums.begin()
//        );
//        int num_clusters_found = (int)(end_pair.first - d_keys_out.begin());
//
//        // Second reduction: count how many points per cluster
//        thrust::fill(d_counts.begin(), d_counts.end(), 0);
//        thrust::reduce_by_key(
//            thrust::device, d_labels_temp.begin(), d_labels_temp.end(), thrust::constant_iterator<int>(1),
//            thrust::make_discard_iterator(), d_counts.begin()
//        );
//
//        // Copy results back to host
//        thrust::host_vector<int> h_keys_out       = d_keys_out;
//        thrust::host_vector<Point<D>> h_sums_host = d_sums;
//        thrust::host_vector<int> h_counts_host    = d_counts;
//        clock.stop(MEASURED_PHASE::THRUST);
//
//        // Compute final centroids on the host
//        clock.start(MEASURED_PHASE::CPU_COMPUTATION);
//        for (int cluster_id = 0; cluster_id < k; ++cluster_id)
//        {
//            // Find this cluster in h_keys_out
//            auto it = std::find(h_keys_out.begin(), h_keys_out.begin() + num_clusters_found, cluster_id);
//            if (it != h_keys_out.begin() + num_clusters_found)
//            {
//                int idx = (int)(it - h_keys_out.begin());
//                int c   = h_counts_host[idx];
//                if (c > 0)
//                {
//                    for (int dim = 0; dim < D; ++dim)
//                    {
//                        h_centroids[cluster_id * D + dim] = h_sums_host[idx].coords[dim] / (float)c;
//                    }
//                }
//                // If cluster is empty, keep old centroid (no change)
//            }
//            else
//            {
//                // No points in this cluster, do not change centroid
//            }
//        }
//        clock.stop(MEASURED_PHASE::CPU_COMPUTATION);
//
//        // Copy updated centroids back to device
//        clock.start(MEASURED_PHASE::DATA_TRANSFER);
//        CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice));
//        clock.stop(MEASURED_PHASE::DATA_TRANSFER);
//
//        // Update old_labels for debugging
//        memcpy(old_labels, h_labels, n * sizeof(int));
//    }
//
//    // Copy final centroids back to host
//    clock.start(MEASURED_PHASE::DATA_TRANSFER_BACK);
//    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost));
//    clock.stop(MEASURED_PHASE::DATA_TRANSFER_BACK);
//
//    clock.stop(MEASURED_PHASE::TOTAL);
//    clock.printResults("Thrust-based k-means");
//
//    // Cleanup
//    delete[] old_labels;
//    delete h_did_change;
//
//    CHECK_CUDA_ERROR(cudaFree(d_data));
//    CHECK_CUDA_ERROR(cudaFree(d_centroids));
//    CHECK_CUDA_ERROR(cudaFree(d_labels));
//    CHECK_CUDA_ERROR(cudaFree(d_did_change));
}
