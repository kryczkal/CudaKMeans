//
// Created by wookie on 11/22/24.
//

#include "KMeansAlgorithms.h"
#include "PerformanceClock.h"
#include "Point.cuh"

void KMeansAlgorithms::Cpu(const float *data, float *centroids, int *labels, int n, int d, int k)
{
    PerformanceClock clock;

    auto new_centroids = new float[k * d]();
    auto counts        = new float[k]();

    clock.start(MEASURED_PHASE::CPU_COMPUTATION);
    bool changes = true;
    for (int i = 0; i < max_iter && changes; ++i)
    {
        changes = false;

        // Labeling
        for (int j = 0; j < n; ++j)
        {
            float min_dist = INFINITY;
            int min_idx    = -1;
            for (int l = 0; l < k; ++l)
            {
                float dist = 0;
                for (int m = 0; m < d; ++m)
                {
                    dist += (data[j * d + m] - centroids[l * d + m]) * (data[j * d + m] - centroids[l * d + m]);
                }
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_idx  = l;
                }
            }
            if (labels[j] != min_idx)
            {
                labels[j] = min_idx;
                changes   = true;
            }
            for (int l = 0; l < d; ++l)
            {
                new_centroids[min_idx * d + l] += data[j * d + l];
            }
            counts[min_idx]++;
        }

        // Centroid update
        for (int j = 0; j < k; ++j)
        {
            if (counts[j] == 0)
            {
                continue;
            }
            for (int l = 0; l < d; ++l)
            {
                new_centroids[j * d + l] /= (float)counts[j];
            }
        }

        // Copy new centroids
        for (int j = 0; j < k * d; ++j)
        {
            centroids[j]     = new_centroids[j];
            new_centroids[j] = 0;
        }

        // Reset counts
        for (int j = 0; j < k; ++j)
        {
            counts[j] = 0;
        }
    }
    clock.stop(MEASURED_PHASE::CPU_COMPUTATION);
    clock.printResults("CPU k-means");

    delete[] new_centroids;
    delete[] counts;
}