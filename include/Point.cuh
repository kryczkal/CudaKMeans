//
// Created by wookie on 12/5/24.
//

#ifndef CUDAKMEANS_POINT_HU
#define CUDAKMEANS_POINT_HU

#include "cuda_runtime.h"

class Point {
public:
    float* coords;
    int dim;

    // Constructor
    __host__ __device__ inline
    Point(int d = 0) : dim(d) {
        if (d > 0) {
            coords = new float[d];
            for (int i = 0; i < d; i++) {
                coords[i] = 0.0f;
            }
        } else {
            coords = nullptr;
        }
    }

    // Copy constructor
    __host__ __device__ inline
    Point(const Point& other) : dim(other.dim) {
        if (dim > 0) {
            coords = new float[dim];
            for (int i = 0; i < dim; i++) {
                coords[i] = other.coords[i];
            }
        } else {
            coords = nullptr;
        }
    }

    // Assignment operator
    __host__ __device__ inline
    Point& operator=(const Point& other) {
        if (this != &other) {
            // Delete old data
            delete[] coords;

            dim = other.dim;
            if (dim > 0) {
                coords = new float[dim];
                for (int i = 0; i < dim; i++) {
                    coords[i] = other.coords[i];
                }
            } else {
                coords = nullptr;
            }
        }
        return *this;
    }

    // Addition operator
    __host__ __device__ inline
    Point operator+(const Point& other) const {
        Point result(dim);
        for (int i = 0; i < dim; i++) {
            result.coords[i] = coords[i] + other.coords[i];
        }
        return result;
    }

    // Addition assignment operator
    __host__ __device__ inline
    Point& operator+=(const Point& other) {
        for (int i = 0; i < dim; i++) {
            coords[i] += other.coords[i];
        }
        return *this;
    }

    // Less than operator (for sorting)
    __host__ __device__ inline
    bool operator<(const Point& other) const {
        return coords[0] < other.coords[0];
    }

    // Destructor
    __host__ __device__ inline
    ~Point() {
        delete[] coords;
    }
};
#endif //CUDAKMEANS_POINT_HU
