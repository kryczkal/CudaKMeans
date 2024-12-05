//
// Created by wookie on 11/22/24.
//

#include <iostream>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include "KMeansWrappers.h"
#include "KMeansIO.h"
#include "CudaUtils.h"
#include "GeneralUtils.h"
#include "KMeansIO.h"

/**
 * @brief Prints the correct usage of the program to the standard error.
 */
void print_usage() {
    std::cerr << "Usage:\n";
    std::cerr << "    KMeans data_format computation_method input_file output_file\n";
    std::cerr << "Where:\n";
    std::cerr << "    data_format: txt or bin\n";
    std::cerr << "    computation_method: cpu, gpu1, or gpu2\n";
    std::cerr << "    input_file: path to the input file\n";
    std::cerr << "    output_file: path to the output file\n";
}

/**
 * @enum DataFormat
 * @brief Represents the data format of the input file.
 */
enum class DataFormat { TXT, BIN };

/**
 * @enum ComputationMethod
 * @brief Represents the computation method to be used.
 */
enum class ComputationMethod { CPU, GPU1, GPU2 };

int main(int argc, char* argv[]) {
    if (argc != 5) {
        print_usage();
        return EXIT_FAILURE;
    }

    std::string data_format_str = argv[1];
    std::string computation_method_str = argv[2];
    std::string input_file = argv[3];
    std::string output_file = argv[4];

    // Map strings to enums for data formats
    std::unordered_map<std::string, DataFormat> data_format_map = {
            {"txt", DataFormat::TXT},
            {"bin", DataFormat::BIN}
    };

    // Map strings to enums for computation methods
    std::unordered_map<std::string, ComputationMethod> computation_method_map = {
            {"cpu", ComputationMethod::CPU},
            {"gpu1", ComputationMethod::GPU1},
            {"gpu2", ComputationMethod::GPU2}
    };

    // Validate data_format
    DataFormat data_format;
    auto df_it = data_format_map.find(data_format_str);
    if (df_it != data_format_map.end()) {
        data_format = df_it->second;
    } else {
        std::cerr << "Error: data_format must be 'txt' or 'bin'\n";
        print_usage();
        return EXIT_FAILURE;
    }

    // Validate computation_method
    ComputationMethod computation_method;
    auto cm_it = computation_method_map.find(computation_method_str);
    if (cm_it != computation_method_map.end()) {
        computation_method = cm_it->second;
    } else {
        std::cerr << "Error: computation_method must be 'cpu', 'gpu1', or 'gpu2'\n";
        print_usage();
        return EXIT_FAILURE;
    }

    // Print CUDA device info if using GPU methods
    if (computation_method == ComputationMethod::GPU1 || computation_method == ComputationMethod::GPU2) {
        CudaUtils::printCudaDeviceInfo();
    }

    std::cout << "Chosen data format: " << data_format_str << std::endl;
    std::cout << "Loading data from: " << input_file << std::endl;

    // Load data
    float* data = nullptr;
    int64_t N = 0, d = 0, k = 0;
    bool success = false;
    if (data_format == DataFormat::TXT) {
        success = KMeansIO::LoadDataFromTextFile(input_file, data, N, d, k);
    } else if (data_format == DataFormat::BIN) {
        success = KMeansIO::LoadDataFromBinaryFile(input_file, data, N, d, k);
    }

    if (!success) {
        std::cerr << "Failed to load data from input file\n";
        return EXIT_FAILURE;
    }


    uint64_t total_size_bytes = N * d * sizeof(float) + k * d * sizeof(float) + N * sizeof(int);
    std::cout << "Loaded data: N=" << N << ", d=" << d << ", k=" << k << std::endl;
    std::cout << "Total size: " << (double)total_size_bytes / 1e9 << " GB\n";

    // Check if data fits into GPU global memory
    if (computation_method == ComputationMethod::GPU1 || computation_method == ComputationMethod::GPU2) {
        if (!GeneralUtils::fitsInGpuGlobalMemory(total_size_bytes)) {
            std::cerr << "Requested data: " << (double)total_size_bytes / 1e9 << " GB does not fit into GPU global memory\n";
            free(data);
            return EXIT_FAILURE;
        }
    }

    std::cout << "Allocating memory for centroids and labels\n";

    // Allocate centroids and labels
    auto* centroids = static_cast<float*>(malloc(k * d * sizeof(float)));
    int* labels = static_cast<int*>(malloc(N * sizeof(int)));

    if (!centroids || !labels) {
        std::cerr << "Failed to allocate memory for centroids or labels\n";
        free(data);
        return EXIT_FAILURE;
    }

    std::cout << "Initializing centroids and labels\n";
    // Initialize centroids to first k points
    std::memcpy(centroids, data, k * d * sizeof(float));

    // Initialize labels to -1
    std::fill(labels, labels + N, -1);

    const int max_iterations = 100;

    std::cout << "Running KMeans algorithm using " << computation_method_str << " method\n";

    // Run the KMeans algorithm based on the computation method
    switch (computation_method) {
        case ComputationMethod::CPU:
            KMeansWrappers::Cpu(data, centroids, labels, N, d, k, max_iterations);
            break;
        case ComputationMethod::GPU1:
            KMeansWrappers::ReductionV1(data, centroids, labels, N, d, k, max_iterations);
            break;
        case ComputationMethod::GPU2:
            KMeansWrappers::ReductionV1(data, centroids, labels, N, d, k, max_iterations);
            break;
        default:
            // Should not reach here due to earlier validation
            std::cerr << "Invalid computation_method\n";
            free(data);
            free(centroids);
            free(labels);
            return EXIT_FAILURE;
    }

    std::cout << "Writing results to: " << output_file << std::endl;
    // Write results to the output file
    if (!KMeansIO::WriteResultsToTextFile(output_file, centroids, labels, N, d, k)) {
        std::cerr << "Failed to write results to output file\n";
        free(data);
        free(centroids);
        free(labels);
        return EXIT_FAILURE;
    }

    std::cout << "Cleaning up memory\n";
    // Clean up allocated memory
    free(data);
    free(centroids);
    free(labels);

    std::cout << "Done\n";
    return EXIT_SUCCESS;
}
