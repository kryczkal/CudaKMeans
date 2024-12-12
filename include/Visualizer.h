//
// Created by wookie on 12/7/24.
//

#ifndef CUDAKMEANS_VISUALIZER_H
#define CUDAKMEANS_VISUALIZER_H

#include "raylib.h"
#include <random>
#include <vector>

namespace RaylibVisualizer
{

class Visualizer
{
    public:
    Visualizer();
    ~Visualizer();

    /**
     * @brief Visualize the clustering results in 3D space.
     * @param data The data points (N x 3).
     * @param centroids The cluster centroids (K x 3).
     * @param labels The cluster labels for each point.
     * @param N Number of data points.
     * @param K Number of clusters.
     */
    void visualize3D(const float *data, const float *centroids, const int *labels, int N, int K);

    private:
    void drawAxes();
    void normalizeData(
        const float *data, float *normalizedData, int N, const float *centroids, float *normalizedCentroids, int K,
        float scale
    );
    std::vector<int> downsampleData(int N, int sampleSize);
    std::vector<Color> generateClusterColors(int K);

    static constexpr int maxRenderPoints = 1e3; // Maximum points to render for performance
};

} // namespace RaylibVisualizer

#endif // CUDAKMEANS_VISUALIZER_H
