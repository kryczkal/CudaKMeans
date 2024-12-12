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

    void launch(int d, int k) { DimensionDispatcher<2, 20>::dispatch(d, k, *this); }
};

#endif // CUDAKMEANS_KMEANSALGORITHMSWRAPPERS_H
