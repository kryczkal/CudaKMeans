//
// Created by wookie on 12/5/24.
//

#include "KMeansValidator.h"
#include "KMeansIO.h"
#include <cmath>

bool KMeansValidator::ValidateResults(const std::string &truthFile, const std::string &testedFile, int d, int k)
{
    float *centroids1 = nullptr;
    int *labels1      = nullptr;
    int N1            = 0;

    float *centroids2 = nullptr;
    int *labels2      = nullptr;
    int N2            = 0;

    // Load results from the first file
    if (!KMeansIO::LoadResultsFromTextFile(truthFile, centroids1, labels1, N1, d, k))
    {
        fprintf(stderr, "Failed to load results from %s\n", truthFile.c_str());
        return false;
    }

    // Load results from the second file
    if (!KMeansIO::LoadResultsFromTextFile(testedFile, centroids2, labels2, N2, d, k))
    {
        fprintf(stderr, "Failed to load results from %s\n", testedFile.c_str());
        delete[] centroids1;
        delete[] labels1;
        return false;
    }

    // Check if the number of data points is the same
    if (N1 != N2)
    {
        fprintf(stderr, "Mismatch in number of data points: %d vs %d\n", N2, N1);
        delete[] centroids1;
        delete[] labels1;
        delete[] centroids2;
        delete[] labels2;
        return false;
    }

    // Compare centroids
    double centroid_diff = 0.0;
    for (int i = 0; i < k * d; ++i)
    {
        double diff = centroids1[i] - centroids2[i];
        centroid_diff += diff * diff;
    }
    centroid_diff = std::sqrt(centroid_diff);

    // Compare labels
    int label_mismatches = 0;
    for (int i = 0; i < N1; ++i)
    {
        if (labels1[i] != labels2[i])
        {
            label_mismatches++;
        }
    }

    // Output the differences
    printf("Total centroid difference (Euclidean distance): %f\n", centroid_diff);
    printf("Number of label mismatches: %d out of %d\n", label_mismatches, N1);

    // Clean up
    delete[] centroids1;
    delete[] labels1;
    delete[] centroids2;
    delete[] labels2;

    return label_mismatches <= labelMismatchTolerancePercent * N1 && centroid_diff <= centroidDiffTolerance;
}
