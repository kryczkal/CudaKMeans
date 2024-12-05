//
// Created by wookie on 11/22/24.
//

#ifndef CUDAKMEANS_KMEANSWRAPPERS_H
#define CUDAKMEANS_KMEANSWRAPPERS_H

#include "CudaUtils.h"
#include <cuda_runtime.h>
#include <iostream>

/*
 * Wrappers that will run the k-means algorithm:
 * 1. Run kernel for labeling
 * 2. Run kernel for centroid update
 * 3. Repeat 1-2 until convergence or max iterations
 */

class KMeansWrappers
{
    public:
    static void Cpu(const float *data, float *centroids, int *labels, int n, int d, int k, int max_iter = 100);
    static void Naive(float *data, float *centroids, int *labels, int n, int d, int k, int max_iter = 100);
    static void ReductionV1(float *data, float *centroids, int *labels, int n, int d, int k, int max_iter = 100);
};

#endif // CUDAKMEANS_KMEANSWRAPPERS_H
