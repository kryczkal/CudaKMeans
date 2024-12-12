//
// Created by wookie on 11/22/24.
//

#include "CudaUtils.h"
#include "DataGenerator.h"
#include "Dispatchers.h"
#include "GeneralUtils.h"
#include "KMeansAlgorithms.h"
#include "KMeansAlgorithmsWrappers.h"
#include "KMeansIO.h"
#include "KMeansValidator.h"
#include "Visualizer.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>

/**
 * @brief Prints the correct usage of the program to the standard error.
 */
void print_usage()
{
    std::cerr << "Usage:\n";
    std::cerr << "    KMeans data_format computation_method input_file output_file [-c | compare] [results_file_path] "
                 "[-g | gen_data] [N] [d] [k] [-s | --show_visualization]\n";
    std::cerr << "Where:\n";
    std::cerr << "    data_format: txt or bin\n";
    std::cerr << "    computation_method: cpu, gpu1, or gpu2\n";
    std::cerr << "    input_file: path to the input file\n";
    std::cerr << "    output_file: path to the output file\n";
    std::cerr << "    -c or compare: optional flag to compare results with a ground truth file\n";
    std::cerr << "    results_file_path: path to the ground truth file .txt\n";
    std::cerr << "    -g or gen_data: optional flag to generate random data - if used, N, d, and k must be provided\n"
                 " and the input_file will be ignored\n";
    std::cerr << "    N: number of data points\n";
    std::cerr << "    d: number of dimensions\n";
    std::cerr << "    k: number of clusters\n";
    std::cerr << "    -s or --show_visualization: optional flag to show visualization of the data and clusters\n";
}

/**
 * @enum DataFormat
 * @brief Represents the data format of the input file.
 */
enum class DataFormat
{
    TXT,
    BIN
};

/**
 * @enum ComputationMethod
 * @brief Represents the computation method to be used.
 */
enum class ComputationMethod
{
    CPU,
    GPU1,
    GPU2
};

/**
 * @brief Parses optional arguments.
 * @param argc The number of arguments.
 * @param argv The array of arguments.
 * @param compare_results A reference to a boolean to store whether to compare results.
 * @param results_file_path A reference to a string to store the path to the results file.
 * @param generate_data A reference to a boolean to store whether to generate random data.
 * @param show_visualization A reference to a boolean to store whether to show visualization.
 * @param N A reference to an integer to store the number of data points.
 * @param d A reference to an integer to store the number of dimensions.
 * @param k A reference to an integer to store the number of clusters.
 * @return void
 */
void parse_opt_args(
    int argc, char *const *argv, bool &compare_results, std::string &results_file_path, bool &generate_data,
    bool &show_visualization, int &N, int &d, int &k
)
{
    enum class optional_arg
    {
        COMPARE,
        GEN_DATA,
        SHOW_VISUALIZATION
    };

    std::unordered_map<std::string, optional_arg> optional_args = {
        {                "-c",            optional_arg::COMPARE},
        {           "compare",            optional_arg::COMPARE},
        {                "-g",           optional_arg::GEN_DATA},
        {          "gen_data",           optional_arg::GEN_DATA},
        {                "-s", optional_arg::SHOW_VISUALIZATION},
        {"show_visualization", optional_arg::SHOW_VISUALIZATION}
    };
    for (int i = 5; i < argc; ++i)
    {
        // Check if the argument is an optional argument
        auto it = optional_args.find(argv[i]);
        if (it != optional_args.end())
        {
            // Handle optional arguments
            switch (it->second)
            {
            case optional_arg::COMPARE:
            {
                if (i + 1 < argc)
                {
                    compare_results   = true;
                    results_file_path = argv[i + 1];
                    i += 1;
                }
                else
                {
                    std::cerr << "Error: missing results_file_path\n";
                    print_usage();
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case optional_arg::GEN_DATA:
            {
                generate_data = true;
                if (i + 3 < argc)
                {
                    N = std::stoi(argv[i + 1]);
                    d = std::stoi(argv[i + 2]);
                    k = std::stoi(argv[i + 3]);
                    i += 3;
                }
                else
                {
                    std::cerr << "Error: missing N, d, or k\n";
                    print_usage();
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case optional_arg::SHOW_VISUALIZATION:
            {
                show_visualization = true;
                break;
            }
            default:
            {
                std::cerr << "Error: invalid optional argument\n";
                print_usage();
                exit(EXIT_FAILURE);
            }
            }
        }
        else
        {
            std::cerr << "Error: invalid argument\n";
            print_usage();
            exit(EXIT_FAILURE);
        }
    }
}

//* @brief Gets the data format from a string.
//* @param data_format_str The string representing the data format.
//* @param data_format_map The map of strings to data formats.
//* @return The data format.
DataFormat getFormat(const std::string &data_format_str, std::unordered_map<std::string, DataFormat> &data_format_map)
{
    DataFormat data_format;
    auto df_it = data_format_map.find(data_format_str);
    if (df_it != data_format_map.end())
    {
        data_format = df_it->second;
    }
    else
    {
        std::cerr << "Error: data_format must be 'txt' or 'bin'\n";
        print_usage();
        exit(EXIT_FAILURE);
    }
    return data_format;
}

/**
 * @brief Gets the computation method from a string.
 * @param computation_method_str The string representing the computation method.
 * @param computation_method_map The map of strings to computation methods.
 * @return The computation method.
 */
ComputationMethod getMethod(
    const std::string &computation_method_str,
    std::unordered_map<std::string, ComputationMethod> &computation_method_map
)
{
    ComputationMethod computation_method;
    auto cm_it = computation_method_map.find(computation_method_str);
    if (cm_it != computation_method_map.end())
    {
        computation_method = cm_it->second;
    }
    else
    {
        std::cerr << "Error: computation_method must be 'cpu', 'gpu1', or 'gpu2'\n";
        print_usage();
        exit(EXIT_FAILURE);
    }
    return computation_method;
}

/**
 * @brief Loads data from a file.
 * @param input_file The path to the input file.
 * @param data_format The data format of the input file.
 * @param N A reference to an integer to store the number of data points.
 * @param d A reference to an integer to store the number of dimensions.
 * @param k A reference to an integer to store the number of clusters.
 * @param data A reference to a pointer to store the data.
 * @return void
 */
void loadData(const std::string &input_file, const DataFormat &data_format, int &N, int &d, int &k, float *&data)
{
    bool success = false;
    if (data_format == DataFormat::TXT)
    {
        success = KMeansIO::LoadDataFromTextFile(input_file, data, N, d, k);
    }
    else if (data_format == DataFormat::BIN)
    {
        success = KMeansIO::LoadDataFromBinaryFile(input_file, data, N, d, k);
    }

    if (!success)
    {
        std::cerr << "Failed to load data from input file\n";
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Runs the KMeans algorithm based on the computation method.
 * @param computation_method The computation method to be used.
 * @param data The data points.
 * @param centroids The centroids.
 * @param labels The labels.
 * @param N The number of data points.
 * @param d The number of dimensions.
 * @param k The number of clusters.
 * @return void
 */
void RunKmeans(
    const ComputationMethod &computation_method, float *&data, float *&centroids, int *&labels, int &N, int d, int k
)
{
    switch (computation_method)
    {
    case ComputationMethod::CPU:
    {
        KMeansAlgorithms::Cpu(data, centroids, labels, N, d, k);
    }
    break;
    case ComputationMethod::GPU1:
    {
        AtomicAddShmemLauncher launcher{data, centroids, labels, N};
        launcher.launch(d, k);
    }
    break;
    case ComputationMethod::GPU2:
    {
        //        KMeansAlgorithms::TreeReduction(data, centroids, labels, N, d, k);
        KMeansAlgorithms::ThrustVersion(data, centroids, labels, N, d, k, 100);
    }
    break;
    default:
    {
        std::cerr << "Invalid computation_method\n";
        free(data);
        free(centroids);
        free(labels);
        exit(EXIT_FAILURE);
    }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        print_usage();
        return EXIT_FAILURE;
    }

    std::string data_format_str        = argv[1];
    std::string computation_method_str = argv[2];
    std::string input_file             = argv[3];
    std::string output_file            = argv[4];

    // Parse optional arguments
    bool compare_results = false;
    std::string results_file_path;
    bool generate_data      = false;
    bool show_visualization = false;
    int N = 0, d = 0, k = 0;
    parse_opt_args(argc, argv, compare_results, results_file_path, generate_data, show_visualization, N, d, k);

    if (generate_data)
        std::cout << "Generating random data: N=" << N << ", d=" << d << ", k=" << k << std::endl;

    // Map strings to enums for data formats
    std::unordered_map<std::string, DataFormat> data_format_map = {
        {"txt", DataFormat::TXT},
        {"bin", DataFormat::BIN}
    };

    // Map strings to enums for computation methods
    std::unordered_map<std::string, ComputationMethod> computation_method_map = {
        { "cpu",  ComputationMethod::CPU},
        {"gpu1", ComputationMethod::GPU1},
        {"gpu2", ComputationMethod::GPU2}
    };

    DataFormat data_format               = getFormat(data_format_str, data_format_map);
    ComputationMethod computation_method = getMethod(computation_method_str, computation_method_map);

    // Print CUDA device info if using GPU methods
    if (computation_method == ComputationMethod::GPU1 || computation_method == ComputationMethod::GPU2)
    {
        CudaUtils::printCudaDeviceInfo();
    }

    std::cout << "Chosen data format: " << data_format_str << std::endl;

    // Load or generate data
    float *data = nullptr;
    if (generate_data)
    {
        std::cout << "Generating random data\n";
        DataGenerator data_generator{N, k, d};
        data = data_generator.generateData();
    }
    else
    {
        std::cout << "Loading data from: " << input_file << std::endl;
        loadData(input_file, data_format, N, d, k, data);
    }

    uint64_t total_size_bytes = N * d * sizeof(float) + k * d * sizeof(float) + N * sizeof(int);
    std::cout << "Loaded data: N=" << N << ", d=" << d << ", k=" << k << std::endl;
    std::cout << "Total size: " << (double)total_size_bytes / 1e9 << " GB\n";

    // Check if data fits into GPU global memory
    if (computation_method == ComputationMethod::GPU1 || computation_method == ComputationMethod::GPU2)
    {
        if (!GeneralUtils::fitsInGpuGlobalMemory(total_size_bytes))
        {
            std::cerr << "Requested data: " << (double)total_size_bytes / 1e9
                      << " GB does not fit into GPU global memory\n";
            free(data);
            return EXIT_FAILURE;
        }
    }

    std::cout << "Allocating memory for centroids and labels\n";

    // Allocate centroids and labels
    auto *centroids = new float[k * d];
    int *labels     = new int[N];

    std::cout << "Initializing centroids and labels\n";
    // Initialize centroids to first k points
    std::memcpy(centroids, data, k * d * sizeof(float));

    // Initialize labels to -1
    std::fill(labels, labels + N, -1);

    std::cout << "Running KMeans algorithm using " << computation_method_str << " method\n";

    // Run the KMeans algorithm based on the computation method
    RunKmeans(computation_method, data, centroids, labels, N, d, k);

    std::cout << "Writing results to: " << output_file << std::endl;

    // Write results to the output file
    if (!KMeansIO::WriteResultsToTextFile(output_file, centroids, labels, N, d, k))
    {
        std::cerr << "Failed to write results to output file\n";
        free(data);
        free(centroids);
        free(labels);
        return EXIT_FAILURE;
    }

    if (show_visualization)
    {
        if (d == 3)
        {
            std::cout << "Showing visualization\n";
            RaylibVisualizer::Visualizer visualizer;
            visualizer.visualize3D(data, centroids, labels, N, k);
        }
        else
        {
            std::cerr << "Visualization is only supported for 3D data\n";
        }
    }

    std::cout << "Cleaning up memory\n";

    // Clean up allocated memory
    free(data);
    free(centroids);
    free(labels);

    // Compare results if requested
    if (compare_results)
    {
        std::cout << "Comparing results with ground truth file: " << results_file_path << std::endl;
        if (!KMeansValidator::ValidateResults(results_file_path, output_file, d, k))
        {
            std::cerr << "Results are beyond the assumed tolerance\n";
            return EXIT_FAILURE;
        }
        std::cout << "Results are within the assumed error tolerance\n";
    }

    std::cout << "Done\n";
    return EXIT_SUCCESS;
}
