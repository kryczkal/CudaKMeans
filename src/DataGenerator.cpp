//
// Created by wookie on 12/7/24.
//

#include "DataGenerator.h"
#include <random>
#include <stdexcept>

[[maybe_unused]] float *DataGenerator::generateData(bool normalize) const
{
    // Use the generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    auto *data = new float[N * D];
    for (int i = 0; i < N * D; ++i)
    {
        if (normalize)
        {
            data[i] = dis(gen);
        }
        else
        {
            data[i] = dis(gen) * std::numeric_limits<float>::max();
        }
    }
    return data;
}

[[maybe_unused]] float *DataGenerator::generateGaussianData(int numDistributions, bool normalize = true) const
{
    if (numDistributions <= 0)
    {
        throw std::invalid_argument("Number of distributions must be positive.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    // Define distribution for selecting which Gaussian to sample from
    std::uniform_int_distribution<int> distSelect(0, numDistributions - 1);

    // Define distributions for Gaussian parameters
    // Means are chosen uniformly in a range, and std dev is fixed
    std::uniform_real_distribution<float> meanDis(-10.0f, 10.0f);
    float stdDev = 1.0f;

    // Generate random means for each Gaussian distribution
    std::vector<std::vector<float>> means(numDistributions, std::vector<float>(D));
    for (int k = 0; k < numDistributions; ++k)
    {
        for (int d = 0; d < D; ++d)
        {
            means[k][d] = meanDis(gen);
        }
    }

    // Create normal distributions for each dimension of each Gaussian
    std::vector<std::vector<std::normal_distribution<float>>> gaussianDistributions;
    gaussianDistributions.reserve(numDistributions);
    for (int k = 0; k < numDistributions; ++k)
    {
        std::vector<std::normal_distribution<float>> dimDistributions;
        dimDistributions.reserve(D);
        for (int d = 0; d < D; ++d)
        {
            dimDistributions.emplace_back(means[k][d], stdDev);
        }
        gaussianDistributions.emplace_back(std::move(dimDistributions));
    }

    float *data = new float[N * D];

    for (int i = 0; i < N; ++i)
    {
        int selectedDist = distSelect(gen);
        for (int d = 0; d < D; ++d)
        {
            float value     = gaussianDistributions[selectedDist][d](gen);
            data[i * D + d] = normalize ? value : value * std::numeric_limits<float>::max();
        }
    }

    return data;
}