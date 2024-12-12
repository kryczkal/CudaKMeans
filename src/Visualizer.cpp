// home/wookie/Projects/ImageProfileConverter/
//  Created by wookie on 12/6/24.
//

#include "Visualizer.h"

namespace RaylibVisualizer
{
#include <raymath.h>

Visualizer::Visualizer()
{
    InitWindow(1200, 800, "K-Means Clustering Visualization");
    SetTargetFPS(60);
}

Visualizer::~Visualizer() { CloseWindow(); }

void Visualizer::visualize3D(const float *data, const float *centroids, const int *labels, int N, int K)
{
    std::vector<Color> clusterColors = generateClusterColors(K);

    Camera3D camera   = {0};
    camera.position   = {10.0f, 10.0f, 20.0f};
    camera.target     = {0.0f, 0.0f, 0.0f};
    camera.up         = {0.0f, 1.0f, 0.0f};
    camera.fovy       = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float rotationSpeed = 0.02f;
    float movementSpeed = 0.2f;

    auto *normalizedData      = new float[N * 3];
    auto *normalizedCentroids = new float[K * 3];
    normalizeData(data, normalizedData, N, centroids, normalizedCentroids, K, 14.0f);
    std::vector<int> sampledIndices = downsampleData(N, maxRenderPoints);

    while (!WindowShouldClose())
    {
        // Compute local axes
        Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
        Vector3 right   = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
        Vector3 up      = Vector3Normalize(camera.up);

        // Handle movement (relative to orientation)
        Vector3 movement = {0.0f, 0.0f, 0.0f};
        if (IsKeyDown(KEY_W))
            movement = Vector3Add(movement, Vector3Scale(forward, movementSpeed));
        if (IsKeyDown(KEY_S))
            movement = Vector3Subtract(movement, Vector3Scale(forward, movementSpeed));
        if (IsKeyDown(KEY_A))
            movement = Vector3Subtract(movement, Vector3Scale(right, movementSpeed));
        if (IsKeyDown(KEY_D))
            movement = Vector3Add(movement, Vector3Scale(right, movementSpeed));

        // Apply movement
        camera.position = Vector3Add(camera.position, movement);
        camera.target   = Vector3Add(camera.target, movement);

        // Handle rotations
        if (IsKeyDown(KEY_H))
        { // Yaw left
            forward = Vector3RotateByAxisAngle(forward, up, rotationSpeed);
        }
        if (IsKeyDown(KEY_K))
        { // Yaw right
            forward = Vector3RotateByAxisAngle(forward, up, -rotationSpeed);
        }
        if (IsKeyDown(KEY_U))
        { // Pitch up
            forward = Vector3RotateByAxisAngle(forward, right, -rotationSpeed);
        }
        if (IsKeyDown(KEY_J))
        { // Pitch down
            forward = Vector3RotateByAxisAngle(forward, right, rotationSpeed);
        }
        if (IsKeyDown(KEY_Y))
        { // Roll left
            up = Vector3RotateByAxisAngle(up, forward, -rotationSpeed);
        }
        if (IsKeyDown(KEY_I))
        { // Roll right
            up = Vector3RotateByAxisAngle(up, forward, rotationSpeed);
        }

        // Recompute camera target and up
        camera.target = Vector3Add(camera.position, forward);
        camera.up     = up;

        BeginDrawing();
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

        // Draw the axes
        drawAxes();

        // Draw sampled data points
        for (int index : sampledIndices)
        {
            Vector3 point = {normalizedData[index * 3], normalizedData[index * 3 + 1], normalizedData[index * 3 + 2]};
            DrawSphere(point, 0.05f, clusterColors[labels[index]]);
        }

        // Draw centroids
        for (int k = 0; k < K; ++k)
        {
            Vector3 centroid = {
                normalizedCentroids[k * 3], normalizedCentroids[k * 3 + 1], normalizedCentroids[k * 3 + 2]
            };
            DrawSphereWires(centroid, 0.2f, 6, 6, clusterColors[k]);
        }

        EndMode3D();

        // Display instructions
        DrawText("WASD: Move, H/U/J/K: Rotate, Y/I: Roll, ESC: Exit", 10, 10, 20, DARKGRAY);

        EndDrawing();
    }

    delete[] normalizedData;
    delete[] normalizedCentroids;
}

void Visualizer::drawAxes()
{
    DrawLine3D({-5.0f, 0.0f, 0.0f}, {5.0f, 0.0f, 0.0f}, RED);
    DrawLine3D({0.0f, -5.0f, 0.0f}, {0.0f, 5.0f, 0.0f}, GREEN);
    DrawLine3D({0.0f, 0.0f, -5.0f}, {0.0f, 0.0f, 5.0f}, BLUE);
}

void Visualizer::normalizeData(
    const float *data, float *normalizedData, int N, const float *centroids, float *normalizedCentroids, int K,
    float scale
)
{
    float minX = data[0], maxX = data[0];
    float minY = data[1], maxY = data[1];
    float minZ = data[2], maxZ = data[2];

    // Find min and max for each axis
    for (int i = 0; i < N; ++i)
    {
        minX = std::min(minX, data[i * 3]);
        maxX = std::max(maxX, data[i * 3]);
        minY = std::min(minY, data[i * 3 + 1]);
        maxY = std::max(maxY, data[i * 3 + 1]);
        minZ = std::min(minZ, data[i * 3 + 2]);
        maxZ = std::max(maxZ, data[i * 3 + 2]);
    }

    // Normalize data to fit within [-scale, scale] range
    float rangeX = maxX - minX;
    float rangeY = maxY - minY;
    float rangeZ = maxZ - minZ;

    for (int i = 0; i < N; ++i)
    {
        normalizedData[i * 3]     = scale * ((data[i * 3] - minX) / rangeX - 0.5f);
        normalizedData[i * 3 + 1] = scale * ((data[i * 3 + 1] - minY) / rangeY - 0.5f);
        normalizedData[i * 3 + 2] = scale * ((data[i * 3 + 2] - minZ) / rangeZ - 0.5f);
    }

    for (int i = 0; i < K; ++i)
    {
        normalizedCentroids[i * 3]     = scale * ((centroids[i * 3] - minX) / rangeX - 0.5f);
        normalizedCentroids[i * 3 + 1] = scale * ((centroids[i * 3 + 1] - minY) / rangeY - 0.5f);
        normalizedCentroids[i * 3 + 2] = scale * ((centroids[i * 3 + 2] - minZ) / rangeZ - 0.5f);
    }
}

std::vector<int> Visualizer::downsampleData(int N, int sampleSize)
{
    std::vector<int> indices;
    if (N <= sampleSize)
    {
        indices.resize(N);
        for (int i = 0; i < N; ++i) indices[i] = i;
    }
    else
    {
        indices.reserve(sampleSize);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, N - 1);

        for (int i = 0; i < sampleSize; ++i)
        {
            indices.push_back(dis(gen));
        }
    }
    return indices;
}

std::vector<Color> Visualizer::generateClusterColors(int K)
{
    std::vector<Color> colors;
    for (int i = 0; i < K; ++i)
    {
        colors.push_back(Color{
            static_cast<unsigned char>(GetRandomValue(50, 255)), static_cast<unsigned char>(GetRandomValue(50, 255)),
            static_cast<unsigned char>(GetRandomValue(50, 255)), 255
        });
    }
    return colors;
}
} // namespace RaylibVisualizer
