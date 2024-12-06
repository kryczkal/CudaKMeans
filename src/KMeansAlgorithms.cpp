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

    // Available shared memory per block
    size_t shared_mem_per_block = props.sharedMemPerBlock;

    // Memory needed for local histogram
    size_t histogram_size = k * d * sizeof(float) + k * sizeof(int);

    // Leave some shared memory for other purposes
    size_t available_shared_mem = shared_mem_per_block * 0.8;

    // Calculate maximum points that can fit in a block
    int max_points = (available_shared_mem - histogram_size) / (sizeof(float) * d);

    // Round down to nearest multiple of 32 for coalescing
    max_points = (max_points / 32) * 32;

    // Limit maximum points to avoid too large blocks
    return std::min(max_points, 1024);
}

struct LocalHistogram {
    float* sums;        // k * d values
    int* counts;        // k values

    __device__ void initialize(float* shared_mem, int k, int d) {
        sums = shared_mem;
        counts = reinterpret_cast<int*>(shared_mem + k * d);

        // Zero initialization of local histogram
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            counts[i] = 0;
            for (int j = 0; j < d; j++) {
                sums[i * d + j] = 0.0f;
            }
        }
        __syncthreads();
    }
};

__global__ void compute_local_histograms(
        const float* data,
        const int* labels,
        float* partial_centroids,
        int* partial_counts,
        int n, int d, int k,
        int points_per_block
) {
    extern __shared__ float shared_mem[];
    LocalHistogram histogram;
    histogram.initialize(shared_mem, k, d);

    // Each thread processes its chunk of data
    int start_idx = blockIdx.x * points_per_block + threadIdx.x;
    int stride = blockDim.x;
    int end_idx = min(start_idx + points_per_block, n);

    // Compute local histogram
    for (int idx = start_idx; idx < end_idx; idx += stride) {
        int label = labels[idx];
        if (label >= 0 && label < k) {
            atomicAdd(&histogram.counts[label], 1);
            for (int d_idx = 0; d_idx < d; d_idx++) {
                atomicAdd(&histogram.sums[label * d + d_idx], data[idx * d + d_idx]);
            }
        }
    }
    __syncthreads();

    // Write local histogram to global memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++) {
            partial_counts[blockIdx.x * k + i] = histogram.counts[i];
            if (histogram.counts[i] > 0) {
                for (int j = 0; j < d; j++) {
                    partial_centroids[blockIdx.x * k * d + i * d + j] =
                            histogram.sums[i * d + j] / histogram.counts[i];
                }
            }
        }
    }
}

__global__ void tree_reduce_centroids(
        float* partial_centroids,
        int* partial_counts,
        float* final_centroids,
        int num_blocks, int d, int k
) {
    extern __shared__ float shared_mem[];
    float* shared_centroids = shared_mem;
    int* shared_counts = reinterpret_cast<int*>(shared_mem + k * d);

    int tid = threadIdx.x;
    int elements_per_thread = (k * d + k + blockDim.x - 1) / blockDim.x;

    // Initialize shared memory
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < k * d) {
            shared_centroids[idx] = 0.0f;
        }
        if (idx < k) {
            shared_counts[idx] = 0;
        }
    }
    __syncthreads();

    // Tree reduction
    for (int stride = 1; stride < num_blocks; stride *= 2) {
        for (int i = tid; i < k; i += blockDim.x) {
            int base_idx = blockIdx.x * stride * 2;
            if (base_idx + stride < num_blocks) {
                int count1 = partial_counts[base_idx * k + i];
                int count2 = partial_counts[(base_idx + stride) * k + i];
                int total_count = count1 + count2;

                if (total_count > 0) {
                    for (int j = 0; j < d; j++) {
                        float val1 = partial_centroids[base_idx * k * d + i * d + j] * count1;
                        float val2 = partial_centroids[(base_idx + stride) * k * d + i * d + j] * count2;
                        partial_centroids[base_idx * k * d + i * d + j] =
                                (val1 + val2) / total_count;
                    }
                    partial_counts[base_idx * k + i] = total_count;
                }
            }
        }
        __syncthreads();
    }

    // Write final results
    if (blockIdx.x == 0 && tid < k) {
        for (int j = 0; j < d; j++) {
            final_centroids[tid * d + j] = partial_centroids[tid * d + j];
        }
    }
}

void update_centroids_tree_reduction(
        const float* d_data,
        const int* d_labels,
        float* d_centroids,
        int n, int d, int k
) {
    // Calculate optimal parameters
    int points_per_block = calculate_optimal_points_per_block(d, k);
    const int threads_per_block = 256;
    const int num_blocks = (n + points_per_block - 1) / points_per_block;

    // Allocate memory for partial results
    float* d_partial_centroids;
    int* d_partial_counts;
    CHECK_CUDA_ERROR(cudaMalloc(&d_partial_centroids, num_blocks * k * d * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_partial_counts, num_blocks * k * sizeof(int)));

    // Shared memory size for local histograms
    size_t shared_mem_size = k * d * sizeof(float) + k * sizeof(int);

    // Step 1: Compute local histograms
    compute_local_histograms<<<num_blocks, threads_per_block, shared_mem_size>>>(
            d_data, d_labels, d_partial_centroids, d_partial_counts,
            n, d, k, points_per_block
    );

    // Step 2: Tree reduction
    tree_reduce_centroids<<<1, threads_per_block, shared_mem_size>>>(
            d_partial_centroids, d_partial_counts, d_centroids,
            num_blocks, d, k
    );

    // Cleanup
    cudaFree(d_partial_centroids);
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