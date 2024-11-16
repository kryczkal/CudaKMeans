#include <iostream>
#include <cuda_runtime.h>

static constexpr bool DEBUG = false;
static constexpr int N = 4092;
static constexpr int M = 4092;

struct matMulKernelInfo {
    void (*kernel)(float*, float*, float*, int, int);
    size_t floatingPointOps;
    size_t bytes_read;
    size_t bytes_written;
};

void launchAndTimeKernel(
        matMulKernelInfo kernelInfo,
        float *d_matA,
        float *d_matB,
        float *d_matC,
        int N,
        int M);
void CudaDeviceInfo();
void checkCudaErrors(cudaError_t error);
float *initMatrix(float *mat, int sizeX, int sizeY);
void randomizeMatrix(float *mat, int sizeX, int sizeY);
void printMatrix(float *mat, int sizeX, int sizeY);
void checkCudaErrors(cudaError_t error);


__global__ void matMul(float *matA, float *matB, float *matC, int sizeX, int sizeY) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < sizeX && y < sizeY) {
        float sum = 0;
        for (int i = 0; i < sizeY; i++) {
            sum += matA[x * sizeY + i] * matB[i * sizeY + y];
        }
        matC[x * sizeY + y] = sum;
    }
}

__global__ void matMul2(float *matA, float *matB, float *matC, int sizeX, int sizeY) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < sizeX && y < sizeY) {
        float sum = 0;
        for (int i = 0; i < sizeY; i++) {
            sum += matA[x * sizeY + i] * matB[i * sizeY + y];
        }
        matC[x * sizeY + y] = sum;
    }
}

int main() {
    CudaDeviceInfo();

    float *h_matA = nullptr, *h_matB = nullptr, *h_matC = nullptr;
    [[maybe_unused]] float *d_matA = nullptr, *d_matB = nullptr, *d_matC = nullptr;
    cudaError_t err = cudaSuccess;

    h_matA = initMatrix(h_matA, N, M);
    h_matB = initMatrix(h_matB, M, N);
    h_matC = initMatrix(h_matC, N, N);

    randomizeMatrix(h_matA, N, M);
    randomizeMatrix(h_matB, M, N);

    if constexpr (DEBUG) {
        std::cout << "Matrix A:" << std::endl;
        printMatrix(h_matA, N, M);
        std::cout << "Matrix B:" << std::endl;
        printMatrix(h_matB, M, N);
    }

    checkCudaErrors(cudaMalloc((void**)&d_matA, N * M * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_matB, M * N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_matC, N * N * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_matA, h_matA, N * M * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_matB, h_matB, M * N * sizeof(float), cudaMemcpyHostToDevice));

    matMulKernelInfo matMulInfo = {matMul, (size_t)2 * N * N * M, 3 * N * M * sizeof(float), N * N * sizeof(float)};
    matMulKernelInfo matMul2Info = {matMul2, (size_t)2 * N * N * M, 3 * N * M * sizeof(float), N * N * sizeof(float)};
    launchAndTimeKernel(matMulInfo, d_matA, d_matB, d_matC, N, M);
    launchAndTimeKernel(matMul2Info, d_matA, d_matB, d_matC, N, M);


    checkCudaErrors(cudaMemcpy(h_matC, d_matC, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    if constexpr (DEBUG) {
        std::cout << "Matrix C:" << std::endl;
        printMatrix(h_matC, N, N);
    }

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
    free(h_matA);
    free(h_matB);
    free(h_matC);
}

void launchAndTimeKernel(
        matMulKernelInfo kernelInfo,
        float *d_matA,
        float *d_matB,
        float *d_matC,
        int N,
        int M) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    void (*kernel)(float*, float*, float*, int, int) = kernelInfo.kernel;

    // Warm up
    for (int i = 0; i < 10; i++) {
        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matA, d_matB, d_matC, N, N);
        if (cudaPeekAtLastError() != cudaSuccess) {
            std::cerr << "Error in kernel launch" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matA, d_matB, d_matC, N, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if (cudaPeekAtLastError() != cudaSuccess) {
        std::cerr << "Error in kernel launch" << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t total_flops = kernelInfo.floatingPointOps;
    size_t total_bytes_read = kernelInfo.bytes_read;
    size_t total_bytes_written = kernelInfo.bytes_written;
    size_t total_bytes = total_bytes_read + total_bytes_written;
    float gflops = (total_flops / elapsedTime) / 1e6;
    float gbytes = (total_bytes / elapsedTime) / 1e6;

    std::cout << "Elapsed time: " << elapsedTime << "ms" << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;
    std::cout << "GB/s: " << gbytes << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CudaDeviceInfo() {
    int count = 0;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);

    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Compute Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Memory Bus Width: %d\n", prop.memoryBusWidth);
        printf("  ----------------\n");
        printf("  Total Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Shared Memory Per Block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  Shared Memory Per MultiProcessor: %luKB\n", prop.sharedMemPerMultiprocessor >> 10);
        printf("  Total Constant Memory: %luKB\n", prop.totalConstMem >> 10);
        printf("  ----------------\n");
        printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Regsters Per Block: %d\n", prop.regsPerBlock);
        printf("  Total MultiProcessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads Per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max Regsters Per MultiProcessor: %d\n", prop.regsPerMultiprocessor);
        printf("  ----------------\n");
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("---------------------------\n");
    }
    printf("\n");
};
void checkCudaErrors(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << __LINE__ << " : " << __FILE__ << " : " << __FUNCTION__ << " : " << cudaGetErrorString(error) << std::endl;
    }
}

float *initMatrix(float *mat, int sizeX, int sizeY) {
    mat = (float*)calloc(sizeX * sizeY, sizeof(float));
    if (mat == NULL) {
        std::cerr << "Failed to allocate memory for matA" << std::endl;
        exit(EXIT_FAILURE);
    }
    return mat;
}

void randomizeMatrix(float *mat, int sizeX, int sizeY) {
    for (int i = 0; i < sizeX * sizeY; i++) {
        mat[i] = rand() / (float)RAND_MAX;
    }
}

void printMatrix(float *mat, int sizeX, int sizeY) {
    for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeY; j++) {
            std::cout << mat[i * sizeY + j] << " ";
        }
        std::cout << std::endl;
    }
}