//
// Created by wookie on 12/6/24.
//

#ifndef CUDAKMEANS_DISPATCHERS_H
#define CUDAKMEANS_DISPATCHERS_H

template <int D_MIN, int D_MAX> struct DimensionDispatcher
{
    template <typename Func> static void dispatch(int d, int k, Func f)
    {
        if (d == D_MIN)
        {
            // Now dispatch on K
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
            // d is out of range, handle error or default
            // For simplicity, do nothing or throw runtime_error
        }
    }
};

template <int K_MIN, int K_MAX> struct ClusterDispatcher
{
    template <int D, typename KernelFunc> static void dispatch(int k, KernelFunc kernelFunc)
    {
        if (k == K_MIN)
        {
            // We found the correct K, call the kernel function
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
            // k out of range, handle error
        }
    }
};

#endif // CUDAKMEANS_DISPATCHERS_H
