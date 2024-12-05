//
// Created by wookie on 12/5/24.
//

#ifndef CUDAKMEANS_POINT_HU
#define CUDAKMEANS_POINT_HU

#include "cuda_runtime.h"

#pragma pack(push,1)
template<int D>
struct Point {
    float coords[D];

    __host__ __device__ inline
    Point() {
        for (int i = 0; i < D; i++) {
            coords[i] = 0.0f;
        }
    }

    __host__ __device__ inline
    Point(const Point &other) {
        for (int i = 0; i < D; i++) {
            coords[i] = other.coords[i];
        }
    }

    __host__ __device__ inline
    Point& operator=(const Point &other) {
        if (this != &other) {
            for (int i = 0; i < D; i++) {
                coords[i] = other.coords[i];
            }
        }
        return *this;
    }

    __host__ __device__ inline
    Point operator+(const Point &other) const {
        Point result;
        for (int i = 0; i < D; i++) {
            result.coords[i] = coords[i] + other.coords[i];
        }
        return result;
    }

    __host__ __device__ inline
    Point& operator+=(const Point &other) {
        for (int i = 0; i < D; i++) {
            coords[i] += other.coords[i];
        }
        return *this;
    }

    // Sort by the first coordinate
    __host__ __device__ inline
    bool operator<(const Point &other) const {
        return coords[0] < other.coords[0];
    }
};
#pragma pack(pop)

#endif //CUDAKMEANS_POINT_HU
