//
// Created by wookie on 12/7/24.
//

#include "DataGenerator.h"
#include <random>

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
