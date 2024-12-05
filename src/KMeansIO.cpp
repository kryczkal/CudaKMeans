//
// Created by wookie on 12/5/24.
//

#include "KMeansIO.h"
#include <cstdio>
#include <cstdlib>
#include <cerrno>

bool KMeansIO::LoadDataFromTextFile(const std::string& filename, float*& data, int64_t& N, int64_t& d, int64_t& k) {
    FILE* fp = fopen(filename.c_str(), "r");
    if (!fp) {
        perror("Failed to open file for reading");
        return false;
    }

    // Read N, d, k from the first line
    if (fscanf(fp, "%ld %ld %ld", &N, &d, &k) != 3) {
        fprintf(stderr, "Failed to read N, d, k from file\n");
        fclose(fp);
        return false;
    }

    // Allocate memory for data
    data = (float*)malloc(N * d * sizeof(float));
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for data\n");
        fclose(fp);
        return false;
    }

    // Read N lines of data
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            if (fscanf(fp, "%f", &data[i * d + j]) != 1) {
                fprintf(stderr, "Failed to read data at point %ld, dimension %ld\n", i, j);
                free(data);
                fclose(fp);
                return false;
            }
        }
    }

    fclose(fp);
    return true;
}

bool KMeansIO::LoadDataFromBinaryFile(const std::string& filename, float*& data, int64_t& N, int64_t& d, int64_t& k) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        perror("Failed to open binary file for reading");
        return false;
    }

    // Read N, d, k from the first 12 bytes
    int32_t N_int32, d_int32, k_int32;
    if (fread(&N_int32, sizeof(int32_t), 1, fp) != 1 ||
        fread(&d_int32, sizeof(int32_t), 1, fp) != 1 ||
        fread(&k_int32, sizeof(int32_t), 1, fp) != 1) {
        fprintf(stderr, "Failed to read N, d, k from binary file\n");
        fclose(fp);
        return false;
    }
    N = static_cast<int64_t>(N_int32);
    d = static_cast<int64_t>(d_int32);
    k = static_cast<int64_t>(k_int32);

    // Allocate memory for data
    data = (float*)malloc(N * d * sizeof(float));
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for data\n");
        fclose(fp);
        return false;
    }

    // Read N * d floats
    size_t num_read = fread(data, sizeof(float), N * d, fp);
    if (num_read != static_cast<size_t>(N * d)) {
        fprintf(stderr, "Failed to read data from binary file\n");
        free(data);
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

bool KMeansIO::WriteResultsToTextFile(const std::string& filename, const float* centroids, const int* labels, int64_t N, int64_t d, int64_t k) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        perror("Failed to open file for writing");
        return false;
    }

    // First k lines: centroids
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            if (fprintf(fp, "%f", centroids[i * d + j]) < 0) {
                fprintf(stderr, "Failed to write centroid data\n");
                fclose(fp);
                return false;
            }
            if (j < d - 1) {
                if (fprintf(fp, " ") < 0) {
                    fprintf(stderr, "Failed to write space\n");
                    fclose(fp);
                    return false;
                }
            }
        }
        if (fprintf(fp, "\n") < 0) {
            fprintf(stderr, "Failed to write newline\n");
            fclose(fp);
            return false;
        }
    }

    // Next N lines: labels
    for (int64_t i = 0; i < N; ++i) {
        const char* format = (i < N - 1) ? "%d\n" : "%d";
        if (fprintf(fp, format, labels[i]) < 0) {
            fprintf(stderr, "Failed to write label data\n");
            fclose(fp);
            return false;
        }
    }


    fclose(fp);
    return true;
}
