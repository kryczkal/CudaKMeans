//
// Created by wookie on 12/6/24.
//

#ifndef CUDAKMEANS_DISPATCHERS_H
#define CUDAKMEANS_DISPATCHERS_H

#include <stdexcept>

// A set of dispatchers that effectively unroll to a switch statement at compile time

template <int D_MIN, int D_MAX> struct DimensionDispatcher
{
    template <typename Func> static void dispatch(int d, int k, Func f)
    {
        if (d == D_MIN)
        {
            f.template dimension_launch<D_MIN>(k);
        }
        else
        {
            DimensionDispatcher<D_MIN + 1, D_MAX>::dispatch(d, k, f);
        }
    }
};

// Base case for recursion
template <int D_MAX> struct DimensionDispatcher<D_MAX, D_MAX>
{
    template <typename Func> static void dispatch(int d, int k, Func f)
    {
        if (d == D_MAX)
        {
            f.template dimension_launch<D_MAX>(k);
        }
        else
        {
            throw std::runtime_error("Dimension out of range");
        }
    }
};

template <int K_MIN, int K_MAX> struct ClusterDispatcher
{
    template <int D, typename KernelFunc> static void dispatch(int k, KernelFunc kernelFunc)
    {
        if (k == K_MIN)
        {
            kernelFunc.template cluster_launch<D, K_MIN>();
        }
        else
        {
            ClusterDispatcher<K_MIN + 1, K_MAX>::template dispatch<D>(k, kernelFunc);
        }
    }
};

// Base case for recursion
template <int K_MAX> struct ClusterDispatcher<K_MAX, K_MAX>
{
    template <int D, typename KernelFunc> static void dispatch(int k, KernelFunc kernelFunc)
    {
        if (k == K_MAX)
        {
            kernelFunc.template cluster_launch<D, K_MAX>();
        }
        else
        {
            throw std::runtime_error("Cluster count out of range");
        }
    }
};

#endif // CUDAKMEANS_DISPATCHERS_H
