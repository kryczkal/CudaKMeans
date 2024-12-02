//
// Created by wookie on 11/22/24.
//

#ifndef CUDAKMEANS_WRAPPERS_H
#define CUDAKMEANS_WRAPPERS_H

#include <iostream>
#include <cuda_runtime.h>
#include "cuda_utils.h"

/*
 * Wrappers that will run the k-means algorithm:
 * 1. Run kernel for labeling
 * 2. Run kernel for centroid update
 * 3. Repeat 1-2 until convergence or max iterations
 */

void cpu_kmeans(float* data, float* centroids, int* labels, int n, int d, int k, int max_iter = 100);

void naive_kmeans(float* data, float* centroids, int* labels, int n, int d, int k, int max_iter = 100);


#endif //CUDAKMEANS_WRAPPERS_H
