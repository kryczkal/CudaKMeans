//
// Created by wookie on 11/22/24.
//
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cinttypes>

// Function to visualize K-means clustering result
void visualize_kmeans(const float* data, const float* centroids, const int* labels, uint64_t N, uint64_t D, uint64_t K,
                      const int width, const int height) {
    // Compute mean vector
    float* mean = new float[D]();
    for (uint64_t n = 0; n < N; ++n)
        for (uint64_t d = 0; d < D; ++d)
            mean[d] += data[n * D + d];
    for (uint64_t d = 0; d < D; ++d)
        mean[d] /= N;

    // Center the data
    float* centered_data = new float[N * D];
    for (uint64_t n = 0; n < N; ++n)
        for (uint64_t d = 0; d < D; ++d)
            centered_data[n * D + d] = data[n * D + d] - mean[d];

    // Center the centroids
    float* centered_centroids = new float[K * D];
    for (uint64_t k = 0; k < K; ++k)
        for (uint64_t d = 0; d < D; ++d)
            centered_centroids[k * D + d] = centroids[k * D + d] - mean[d];

    // Compute covariance matrix
    float* cov = new float[D * D]();
    for (uint64_t i = 0; i < D; ++i)
        for (uint64_t j = 0; j < D; ++j)
            for (uint64_t n = 0; n < N; ++n)
                cov[i * D + j] += centered_data[n * D + i] * centered_data[n * D + j];
    for (uint64_t i = 0; i < D * D; ++i)
        cov[i] /= (N - 1);

    // Eigenvalue decomposition using Jacobi method
    float* eigenvectors = new float[D * D]();
    for (uint64_t i = 0; i < D; ++i)
        eigenvectors[i * D + i] = 1.0f;
    float* eigenvalues = new float[D];
    for (uint64_t i = 0; i < D; ++i)
        eigenvalues[i] = cov[i * D + i];

    const int max_sweeps = 100;
    const float epsilon = 1e-6f;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        float max_offdiag = 0.0f;
        uint64_t p = 0, q = 0;
        for (uint64_t i = 0; i < D - 1; ++i) {
            for (uint64_t j = i + 1; j < D; ++j) {
                float offdiag = fabsf(cov[i * D + j]);
                if (offdiag > max_offdiag) {
                    max_offdiag = offdiag;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_offdiag < epsilon)
            break;
        float phi = 0.5f * atanf((2.0f * cov[p * D + q]) / (cov[q * D + q] - cov[p * D + p] + 1e-12f));
        float cos_phi = cosf(phi);
        float sin_phi = sinf(phi);

        // Rotate covariance matrix
        for (uint64_t i = 0; i < D; ++i) {
            float c_ip = cov[i * D + p];
            float c_iq = cov[i * D + q];
            cov[i * D + p] = c_ip * cos_phi - c_iq * sin_phi;
            cov[p * D + i] = cov[i * D + p];
            cov[i * D + q] = c_ip * sin_phi + c_iq * cos_phi;
            cov[q * D + i] = cov[i * D + q];
        }
        float c_pp = cov[p * D + p];
        float c_qq = cov[q * D + q];
        float c_pq = cov[p * D + q];
        cov[p * D + p] = c_pp * cos_phi * cos_phi + c_qq * sin_phi * sin_phi - 2.0f * c_pq * cos_phi * sin_phi;
        cov[q * D + q] = c_pp * sin_phi * sin_phi + c_qq * cos_phi * cos_phi + 2.0f * c_pq * cos_phi * sin_phi;
        cov[p * D + q] = 0.0f;
        cov[q * D + p] = 0.0f;

        // Rotate eigenvectors
        for (uint64_t i = 0; i < D; ++i) {
            float e_ip = eigenvectors[i * D + p];
            float e_iq = eigenvectors[i * D + q];
            eigenvectors[i * D + p] = e_ip * cos_phi - e_iq * sin_phi;
            eigenvectors[i * D + q] = e_ip * sin_phi + e_iq * cos_phi;
        }
    }
    for (uint64_t i = 0; i < D; ++i)
        eigenvalues[i] = cov[i * D + i];

    // Find indices of the two largest eigenvalues
    uint64_t idx1 = 0, idx2 = 1;
    if (eigenvalues[idx2] > eigenvalues[idx1])
        std::swap(idx1, idx2);
    for (uint64_t i = 2; i < D; ++i) {
        if (eigenvalues[i] > eigenvalues[idx1]) {
            idx2 = idx1;
            idx1 = i;
        } else if (eigenvalues[i] > eigenvalues[idx2]) {
            idx2 = i;
        }
    }

    // Get the top two eigenvectors
    float* evec1 = new float[D];
    float* evec2 = new float[D];
    for (uint64_t i = 0; i < D; ++i) {
        evec1[i] = eigenvectors[i * D + idx1];
        evec2[i] = eigenvectors[i * D + idx2];
    }

    // Project the centered data onto the top two eigenvectors
    float* projected_data = new float[N * 2];
    for (uint64_t n = 0; n < N; ++n) {
        float proj1 = 0.0f, proj2 = 0.0f;
        for (uint64_t d = 0; d < D; ++d) {
            proj1 += centered_data[n * D + d] * evec1[d];
            proj2 += centered_data[n * D + d] * evec2[d];
        }
        projected_data[n * 2 + 0] = proj1;
        projected_data[n * 2 + 1] = proj2;
    }

    // Project the centered centroids
    float* projected_centroids = new float[K * 2];
    for (uint64_t k = 0; k < K; ++k) {
        float proj1 = 0.0f, proj2 = 0.0f;
        for (uint64_t d = 0; d < D; ++d) {
            proj1 += centered_centroids[k * D + d] * evec1[d];
            proj2 += centered_centroids[k * D + d] * evec2[d];
        }
        projected_centroids[k * 2 + 0] = proj1;
        projected_centroids[k * 2 + 1] = proj2;
    }

    // Map the projected data to terminal coordinates

    // Find min and max of projected data
    float min_x = projected_data[0], max_x = projected_data[0];
    float min_y = projected_data[1], max_y = projected_data[1];
    for (uint64_t n = 0; n < N; ++n) {
        float x = projected_data[n * 2 + 0];
        float y = projected_data[n * 2 + 1];
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }
    for (uint64_t k = 0; k < K; ++k) {
        float x = projected_centroids[k * 2 + 0];
        float y = projected_centroids[k * 2 + 1];
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }

    // Compute scale factors
    float scale_x = (max_x - min_x) > 0 ? (width - 1) / (max_x - min_x) : 1.0f;
    float scale_y = (max_y - min_y) > 0 ? (height - 1) / (max_y - min_y) : 1.0f;

    // Initialize grid
    char grid[height][width];
    int color_grid[height][width]; // For storing color index
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j) {
            grid[i][j] = ' ';
            color_grid[i][j] = -1;
        }

    // Define colors for labels
    const char* colors[] = {
            "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m",
            "\033[36m", "\033[37m", "\033[91m", "\033[92m", "\033[93m",
            "\033[94m", "\033[95m", "\033[96m", "\033[97m"
    };
    int num_colors = sizeof(colors) / sizeof(colors[0]);

    // Map data points to grid
    for (uint64_t n = 0; n < N; ++n) {
        float x = projected_data[n * 2 + 0];
        float y = projected_data[n * 2 + 1];
        int grid_x = static_cast<int>((x - min_x) * scale_x);
        int grid_y = static_cast<int>((y - min_y) * scale_y);
        grid_x = std::min(std::max(grid_x, 0), width - 1);
        grid_y = std::min(std::max(grid_y, 0), height - 1);
        grid[grid_y][grid_x] = '.';
        color_grid[grid_y][grid_x] = labels[n] % num_colors;
    }

    // Map centroids to grid
    for (uint64_t k = 0; k < K; ++k) {
        float x = projected_centroids[k * 2 + 0];
        float y = projected_centroids[k * 2 + 1];
        int grid_x = static_cast<int>((x - min_x) * scale_x);
        int grid_y = static_cast<int>((y - min_y) * scale_y);
        grid_x = std::min(std::max(grid_x, 0), width - 1);
        grid_y = std::min(std::max(grid_y, 0), height - 1);
        grid[grid_y][grid_x] = 'X';
        color_grid[grid_y][grid_x] = k % num_colors;
    }

    // Print the grid
    for (int i = height - 1; i >= 0; --i) {
        for (int j = 0; j < width; ++j) {
            int color_idx = color_grid[i][j];
            if (color_idx >= 0)
                printf("%s%c\033[0m", colors[color_idx], grid[i][j]);
            else
                printf("%c", grid[i][j]);
        }
        printf("\n");
    }

    // Cleanup
    delete[] mean;
    delete[] centered_data;
    delete[] centered_centroids;
    delete[] cov;
    delete[] eigenvectors;
    delete[] eigenvalues;
    delete[] evec1;
    delete[] evec2;
    delete[] projected_data;
    delete[] projected_centroids;
}
