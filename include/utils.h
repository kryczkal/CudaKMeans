//
// Created by wookie on 11/22/24.
//

#ifndef CUDAKMEANS_UTILS_H
#define CUDAKMEANS_UTILS_H

#include <cinttypes>

void visualize_kmeans(const float* data, const float* centroids, const int* labels, uint64_t N, uint64_t D, uint64_t K,
                      int width = 80, int height = 24);

#endif //CUDAKMEANS_UTILS_H
