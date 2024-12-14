# CUDA K-Means Clustering

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/language-C++-blue.svg)
![CUDA](https://img.shields.io/badge/platform-CUDA%2011.0%2B-blue.svg)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Validation](#validation)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Overview

CUDA K-Means Clustering is a high-performance implementation of the K-Means algorithm leveraging NVIDIA's CUDA platform for parallel computation. This project includes both CPU and GPU versions of the algorithm, tools for data generation and validation, and an optional OpenGL-based 3D visualizer to interactively explore clustering results.

![](https://github.com/kryczkal/CudaKMeans/blob/main/gifs/2.gif)
![](https://github.com/kryczkal/CudaKMeans/blob/main/gifs/1.gif)

## Features

- **CPU Implementation**: A straightforward CPU-based K-Means algorithm for comparison and baseline measurements.
- **GPU Implementation**: Optimized CUDA-based K-Means leveraging shared memory and atomic operations for efficient parallel processing.
- **Thrust Integration**: Utilizes the Thrust library for high-level GPU operations to simplify and accelerate computations.
- **Data Generation**: Tools to generate synthetic datasets, including uniform and Gaussian-distributed data points.
- **Validation**: Mechanisms to validate clustering results against a file with ground truth results
- **Visualization**: Optional OpenGL-based 3D visualizer to render data points and cluster centroids interactively.
- **Performance Measurement**: Comprehensive performance tracking using CUDA events to monitor different phases of the algorithm.

## Installation

### Prerequisites

- **Operating System**: Linux, (Not tested under Windows)
- **Compiler**: NVIDIA CUDA Compiler (`nvcc`) and a C++17 compatible compiler
- **CUDA Toolkit**: Version 11.0 or higher
- **CMake**: Version 3.10 or higher
- **OpenGL Libraries** (Optional for visualization):
  - [GLFW](https://www.glfw.org/)
  - [GLAD](https://glad.dav1d.de/)
  - [GLM](https://glm.g-truc.net/0.9.9/index.html)

### Clone the Repository

```bash
git clone https://github.com/yourusername/cuda-kmeans-clustering.git
cd cuda-kmeans-clustering
```

### Build the Project

1. **Create a Build Directory**:

```bash
mkdir build
cd build
```

2. **Configure with CMake**:
```bash
cmake .. -DUSE_VISUALIZER=OFF
```

If you want to use the visualizer, just run it with `-DUSE_VISUALIZER=ON`
CmakeLists.txt leverages FetchContent to automatically download and compile the required OpenGL libraries (GLFW, GLAD, GLM)

3. **Compile**:

```bash
make -j$(nproc)
```

## Usage

After building, you can run the K-Means clustering on synthetic or custom datasets.

### Generate Synthetic Data

Use the `DataGenerator` to create synthetic datasets.

```cpp
#include "DataGenerator.h"

// Example: Generate Gaussian distributed data
DataGenerator generator(N, K, D);
float* data = generator.generateGaussianData(numDistributions, normalize);
```

### Run K-Means

Execute the clustering algorithm by specifying the dataset and parameters.

```bash
./CudaKMeans data_format computation_method input_file output_file [-c | compare] [results_file_path] "
                 "[-g | gen_data] [N] [d] [k] [-s | --show_visualization]
```

### Command-Line Arguments

- `data_format`: `txt` or `bin`
- `computation_method`: `cpu`, `gpu1`, or `gpu2`
- `input_file`: Path to the input data file
- `output_file`: Path to save the clustering results
- `-c` or `--compare`: (Optional) Compare results with a ground truth file
- `results_file_path`: Path to the ground truth `.txt` file (required if using `--compare`)
- `-g` or `--gen_data`: (Optional) Generate random data instead of reading from a file
- `N`: Number of data points (required if using `--gen_data`)
- `d`: Number of dimensions (required if using `--gen_data`)
- `k`: Number of clusters (required if using `--gen_data`)
- `-s` or `--show_visualization`: (Optional) Show visualization of the data and clusters

### Important Notes

**Argument Parsing Caveat**: The current implementation requiers always specifying an input file even if the `--gen_data` flag is used. When `--gen_data` is used, the program will ignore the `input_file` and generate synthetic data based on the provided `N`, `d`, and `k` parameters. 

## Visualization

If compiled with the `USE_VISUALIZER` flag, the project includes an OpenGL-based 3D visualizer to interactively explore clustering results.

### Controls

- **Movement**: `W/A/S/D` keys to move the camera.
- **Rotation**:
  - `H/K/U/J/Y/I` keys to rotate the view.
- **Exit**: `ESC` key to close the visualizer.


## Validation

The project includes a validation tool to compare clustering results against ground truth files.

### Usage

```bash
./cuda-kmeans-clustering --truth truth_results.txt --tested tested_results.txt --validate
```

### Validation Criteria

- **Centroid Difference**: Euclidean distance between corresponding centroids must be below a specified tolerance.
- **Label Mismatches**: The number of mismatched labels must be within an acceptable percentage of the total data points.

## Performance

Performance metrics are tracked using CUDA events, measuring different phases such as data transfer, kernel execution, and centroid updates.

### Example Output

Upon completion, the program outputs a summary of the time spent in each phase:
-------------------------:
Kernel : Reduction v1 k-means atomicAdd version
Data transfer time : 123.456 ms
Label assignment time : 78.910 ms
Centroid update time : 45.678 ms
Sum of above : 247.044 ms
Debug work : 5.000 ms
Total time : 252.044 ms
-------------------------:
Debug work - time spent on operations not measured such as calculating exactly which labels changed (work of no consequence to the algorithm itself)
-------------------------:

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
# CudaKMeans
