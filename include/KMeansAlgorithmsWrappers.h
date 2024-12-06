//
// Created by wookie on 12/6/24.
//

#ifndef CUDAKMEANS_KMEANSALGORITHMSWRAPPERS_H
#define CUDAKMEANS_KMEANSALGORITHMSWRAPPERS_H

#include "Dispatchers.h"
#include "KMeansAlgorithms.h"

struct AtomicAddShmemLauncher
{
    float *data;
    float *centroids;
    int *labels;
    int n;

    struct AtomicAddShmemKernel
    {
        const AtomicAddShmemLauncher &wrapper;

        template <int D_, int K_> void cluster_launch()
        {
            KMeansAlgorithms::AtomicAddShmem<D_, K_>(wrapper.data, wrapper.centroids, wrapper.labels, wrapper.n);
        }
    };

    template <int D> void dimension_launch(int k)
    {
        AtomicAddShmemKernel kernelFunc{*this};
        ClusterDispatcher<2, 20>::dispatch<D>(k, kernelFunc);
    }

    void run(int D, int K) {
        DimensionDispatcher<2, 20>::dispatch(D, K, *this);
    }
};

struct ThrustVersionLauncher
{
    float *data;
    float *centroids;
    int *labels;
    int n;

    void run(int D, int K) {
        KMeansAlgorithms::ThrustVersion(data, centroids, labels, n, D, K);
    }
};

#endif // CUDAKMEANS_KMEANSALGORITHMSWRAPPERS_H
