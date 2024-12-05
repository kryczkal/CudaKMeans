//
// Created by wookie on 12/5/24.
//

#include "KMeansIO.h"
#include <cstdio>
#include <cstdlib>

bool KMeansIO::LoadDataFromTextFile(const std::string& filename, float*& data, int& N, int& d, int& k) {
    FILE* fp = fopen(filename.c_str(), "r");
    if (!fp) {
        perror("Failed to open file for reading");
        return false;
    }

    // Read N, d, k from the first line
    if (fscanf(fp, "%d %d %d", &N, &d, &k) != 3) {
        fprintf(stderr, "Failed to read N, d, k from file\n");
        fclose(fp);
        return false;
    }

    // Allocate memory for data
    data = new float[N * d];

    // Read N lines of data
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            if (fscanf(fp, "%f", &data[i * d + j]) != 1) {
                fprintf(stderr, "Failed to read data at point %d, dimension %d\n", i, j);
                free(data);
                fclose(fp);
                return false;
            }
        }
    }

    fclose(fp);
    return true;
}

bool KMeansIO::LoadDataFromBinaryFile(const std::string& filename, float*& data, int& N, int& d, int& k) {
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
    N = static_cast<int>(N_int32);
    d = static_cast<int>(d_int32);
    k = static_cast<int>(k_int32);

    // Allocate memory for data
    data = new float[N * d];

    // Read N * d floats
    size_t num_read = fread(data, sizeof(float), N * d, fp);
    if (num_read != static_cast<int>(N * d)) {
        fprintf(stderr, "Failed to read data from binary file\n");
        free(data);
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

bool KMeansIO::WriteResultsToTextFile(const std::string& filename, const float* centroids, const int* labels, int N, int d, int k) {
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        perror("Failed to open file for writing");
        return false;
    }

    // First k lines: centroids
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d; ++j) {
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
    for (int i = 0; i < N; ++i) {
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

bool
KMeansIO::LoadResultsFromTextFile(const std::string &filename, float *&centroids, int *&labels, int &N, int d,
                                  int k) {
    FILE* fp = fopen(filename.c_str(), "r");
    if (!fp) {
        perror("Failed to open file for reading");
        return false;
    }

    // Allocate memory for centroids
    centroids = new float[k * d];

    // Read centroids
    char line[1024];
    for (int i = 0; i < k; ++i) {
        if (!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "Failed to read centroid line %d\n", i);
            fclose(fp);
            delete[] centroids;
            return false;
        }

        char* token = strtok(line, " \t\n");
        for (int j = 0; j < d; ++j) {
            if (!token) {
                fprintf(stderr, "Insufficient data in centroid line %d\n", i);
                fclose(fp);
                delete[] centroids;
                return false;
            }
            centroids[i * d + j] = static_cast<float>(atof(token));
            token = strtok(nullptr, " \t\n");
        }
    }

    // Remember the position of labels in the file
    long labels_start_pos = ftell(fp);

    // Count the number of labels (N)
    N = 0;
    while (fgets(line, sizeof(line), fp)) {
        N++;
    }

    // Allocate memory for labels
    labels = new int[N];

    // Rewind to the start of labels
    fseek(fp, labels_start_pos, SEEK_SET);

    // Read labels
    for (int i = 0; i < N; ++i) {
        if (!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "Failed to read label line %d\n", i);
            fclose(fp);
            delete[] centroids;
            delete[] labels;
            return false;
        }
        labels[i] = atoi(line);
    }

    fclose(fp);
    return true;}
