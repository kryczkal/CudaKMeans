//
// Created by wookie on 12/7/24.
//

#ifndef CUDAKMEANS_DATAGENERATOR_H
#define CUDAKMEANS_DATAGENERATOR_H

/**
 * @brief Class to generate random data points for KMeans clustering.
 */
class DataGenerator
{
    public:
    DataGenerator(int N, int K, int D) : N(N), K(K), D(D) {}
    ~DataGenerator() = default;

    /*
     * @brief Generate random data points for the given N, K, and D.
     * @return A pointer to the generated data points.
     */
    [[maybe_unused]] [[nodiscard]] float *generateData(bool normalize = true) const;

    private:
    int N;
    int K;
    int D;
};

#endif // CUDAKMEANS_DATAGENERATOR_H
