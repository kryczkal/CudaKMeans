//
//  Created by wookie on 12/6/24.
//

#include "Visualizer.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef USE_VISUALIZER
#include <GLFW/glfw3.h>
#include <glad/gl.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

VisualizerOpenGL::VisualizerOpenGL() {}
VisualizerOpenGL::~VisualizerOpenGL() {}

static std::string readFile(const char *path)
{
    std::ifstream file(path, std::ios::in);
    if (!file)
        throw std::runtime_error(std::string("Failed to open file: ") + path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

unsigned int VisualizerOpenGL::createShaderProgram(const char *vertPath, const char *fragPath)
{
    std::string vertCode = readFile(vertPath);
    std::string fragCode = readFile(fragPath);

    const char *vShaderCode = vertCode.c_str();
    const char *fShaderCode = fragCode.c_str();

    unsigned int vertexShader   = glCreateShader(GL_VERTEX_SHADER);
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    unsigned int program        = glCreateProgram();

    glShaderSource(vertexShader, 1, &vShaderCode, nullptr);
    glCompileShader(vertexShader);

    int success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        throw std::runtime_error(std::string("Vertex shader compilation failed: ") + infoLog);
    }

    glShaderSource(fragmentShader, 1, &fShaderCode, nullptr);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        throw std::runtime_error(std::string("Fragment shader compilation failed: ") + infoLog);
    }

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        throw std::runtime_error(std::string("Shader program linking failed: ") + infoLog);
    }

    return program;
}

void VisualizerOpenGL::initOpenGL()
{
    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1200, 800, "K-Means Clustering Visualization (OpenGL)", nullptr, nullptr);
    if (!window)
        throw std::runtime_error("Failed to create GLFW window");
    glfwMakeContextCurrent(window);

    if (!gladLoadGL(glfwGetProcAddress))
        throw std::runtime_error("Failed to initialize GLAD");

    glEnable(GL_DEPTH_TEST);

    shaderProgram = createShaderProgram(
        (std::string(PROJECT_BINARY_DIR) + "/shaders/basic.vert").c_str(),
        (std::string(PROJECT_BINARY_DIR) + "/shaders/basic.frag").c_str()
    );
}

void VisualizerOpenGL::cleanupOpenGL()
{
    if (shaderProgram)
        glDeleteProgram(shaderProgram);
    if (window)
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

void VisualizerOpenGL::visualize3D(const float *data, const float *centroids, const int *labels, int N, int K)
{
    std::cout << "Using OpenGL visualizer\n";
    std::cout << "Press W/A/S/D to move, H/K/U/J/Y/I to rotate, and ESC to exit\n";

    initOpenGL();

    float rotationSpeed = 0.02f;
    float movementSpeed = 0.2f;

    // Normalize data
    auto *normalizedData      = new float[N * 3];
    auto *normalizedCentroids = new float[K * 3];
    normalizeData(data, normalizedData, N, centroids, normalizedCentroids, K, 14.0f);

    std::vector<int> sampledIndices      = downsampleData(N, maxRenderPoints);
    std::vector<glm::vec3> clusterColors = generateClusterColors(K);

    // Camera setup
    glm::vec3 cameraPos    = glm::vec3(10.0f, 10.0f, 20.0f);
    glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cameraUp     = glm::vec3(0.0f, 1.0f, 0.0f);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Input handling
        glm::vec3 forward = glm::normalize(cameraTarget - cameraPos);
        glm::vec3 right   = glm::normalize(glm::cross(forward, cameraUp));
        glm::vec3 up      = glm::normalize(cameraUp);

        glm::vec3 movement(0.0f);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            movement += forward * movementSpeed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            movement -= forward * movementSpeed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            movement -= right * movementSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            movement += right * movementSpeed;

        cameraPos += movement;
        cameraTarget += movement;

        // Yaw left
        if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS)
            forward = glm::angleAxis(rotationSpeed, up) * forward;

        // Yaw right
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
            forward = glm::angleAxis(-rotationSpeed, up) * forward;

        // Pitch up
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
            forward = glm::angleAxis(-rotationSpeed, right) * forward;

        // Pitch down
        if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
            forward = glm::angleAxis(rotationSpeed, right) * forward;

        // Roll left
        if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS)
            up = glm::angleAxis(-rotationSpeed, forward) * up;

        // Roll right
        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
            up = glm::angleAxis(rotationSpeed, forward) * up;

        forward = glm::normalize(forward);
        up      = glm::normalize(up);

        cameraTarget = cameraPos + forward;
        cameraUp     = up;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            break;

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Compute MVP
        glm::mat4 proj  = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
        glm::mat4 view  = glm::lookAt(cameraPos, cameraTarget, cameraUp);
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 mvp   = proj * view * model;

        // Draw everything
        drawAxes(mvp);
        drawPoints(normalizedData, labels, (int)sampledIndices.size(), clusterColors, mvp);
        drawWireSpheres(normalizedCentroids, K, clusterColors, mvp);

        glfwSwapBuffers(window);
    }

    delete[] normalizedData;
    delete[] normalizedCentroids;

    cleanupOpenGL();
}

void VisualizerOpenGL::drawAxes(const glm::mat4 &mvp)
{
    // Simple line drawing for axes
    float axisData[] = {// X axis: red
                        -5.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,

                        // Y axis: green
                        0.0f, -5.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 1.0f, 0.0f,

                        // Z axis: blue
                        0.0f, 0.0f, -5.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 1.0f
    };

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(axisData), axisData, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0); // pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);

    glEnableVertexAttribArray(1); // color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));

    glUseProgram(shaderProgram);
    int mvpLoc = glGetUniformLocation(shaderProgram, "uMVP");
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

    glDrawArrays(GL_LINES, 0, 6);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
}

void VisualizerOpenGL::drawPoints(
    const float *points, const int *labels, int count, const std::vector<glm::vec3> &colors, const glm::mat4 &mvp
)
{
    std::vector<float> vertexData(count * 6); // pos + color
    for (int i = 0; i < count; i++)
    {
        glm::vec3 c           = colors[labels[i]];
        vertexData[i * 6 + 0] = points[i * 3 + 0];
        vertexData[i * 6 + 1] = points[i * 3 + 1];
        vertexData[i * 6 + 2] = points[i * 3 + 2];
        vertexData[i * 6 + 3] = c.r;
        vertexData[i * 6 + 4] = c.g;
        vertexData[i * 6 + 5] = c.b;
    }

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));

    glUseProgram(shaderProgram);
    int mvpLoc = glGetUniformLocation(shaderProgram, "uMVP");
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

    glPointSize(5.0f);
    glDrawArrays(GL_POINTS, 0, count);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
}

void VisualizerOpenGL::generateSphereWireframe(
    std::vector<glm::vec3> &sphereVertices, float radius, int segments, int rings
)
{
    // line: pair of points.
    // segments: how many segments in horizontal circle
    // rings: how many rings from pole to pole

    // vertical lines
    for (int i = 0; i < segments; i++)
    {
        float theta = (2.0f * 3.14159f * i) / segments;
        // line from top to bottom
        for (int j = 0; j < rings; j++)
        {
            float phi1 = (3.14159f * j) / (rings);
            float phi2 = (3.14159f * (j + 1)) / rings;
            glm::vec3 p1(radius * sin(phi1) * cos(theta), radius * cos(phi1), radius * sin(phi1) * sin(theta));
            glm::vec3 p2(radius * sin(phi2) * cos(theta), radius * cos(phi2), radius * sin(phi2) * sin(theta));
            sphereVertices.push_back(p1);
            sphereVertices.push_back(p2);
        }
    }

    for (int j = 1; j < rings; j++)
    {
        float phi = (3.14159f * j) / (rings);
        for (int i = 0; i < segments; i++)
        {
            float theta1 = (2.0f * 3.14159f * i) / segments;
            float theta2 = (2.0f * 3.14159f * (i + 1)) / segments;

            glm::vec3 p1(radius * sin(phi) * cos(theta1), radius * cos(phi), radius * sin(phi) * sin(theta1));
            glm::vec3 p2(radius * sin(phi) * cos(theta2), radius * cos(phi), radius * sin(phi) * sin(theta2));
            sphereVertices.push_back(p1);
            sphereVertices.push_back(p2);
        }
    }
}

void VisualizerOpenGL::drawWireSpheres(
    const float *centroids, int K, const std::vector<glm::vec3> &colors, const glm::mat4 &mvp
)
{
    // Pre-generate sphere wireframe
    std::vector<glm::vec3> sphereVertices;
    generateSphereWireframe(sphereVertices, 0.2f, 6, 6);

    // Just transform sphere vertices by adding centroid position and coloring
    for (int k = 0; k < K; k++)
    {
        glm::vec3 cpos(centroids[k * 3 + 0], centroids[k * 3 + 1], centroids[k * 3 + 2]);
        glm::vec3 ccol = colors[k];

        std::vector<float> vertexData(sphereVertices.size() * 6);
        for (size_t i = 0; i < sphereVertices.size(); i++)
        {
            glm::vec3 v           = sphereVertices[i] + cpos;
            vertexData[i * 6 + 0] = v.x;
            vertexData[i * 6 + 1] = v.y;
            vertexData[i * 6 + 2] = v.z;
            vertexData[i * 6 + 3] = ccol.r;
            vertexData[i * 6 + 4] = ccol.g;
            vertexData[i * 6 + 5] = ccol.b;
        }

        unsigned int VAO, VBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));

        glUseProgram(shaderProgram);
        int mvpLoc = glGetUniformLocation(shaderProgram, "uMVP");
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

        // Since it's a wireframe, just draw lines
        glDrawArrays(GL_LINES, 0, (int)sphereVertices.size());

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        glDeleteBuffers(1, &VBO);
        glDeleteVertexArrays(1, &VAO);
    }
}

void VisualizerOpenGL::normalizeData(
    const float *data, float *normalizedData, int N, const float *centroids, float *normalizedCentroids, int K,
    float scale
)
{
    float minX = data[0], maxX = data[0];
    float minY = data[1], maxY = data[1];
    float minZ = data[2], maxZ = data[2];

    for (int i = 0; i < N; ++i)
    {
        minX = std::min(minX, data[i * 3]);
        maxX = std::max(maxX, data[i * 3]);
        minY = std::min(minY, data[i * 3 + 1]);
        maxY = std::max(maxY, data[i * 3 + 1]);
        minZ = std::min(minZ, data[i * 3 + 2]);
        maxZ = std::max(maxZ, data[i * 3 + 2]);
    }

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

std::vector<int> VisualizerOpenGL::downsampleData(int N, int sampleSize)
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

std::vector<glm::vec3> VisualizerOpenGL::generateClusterColors(int K)
{
    std::vector<glm::vec3> colors;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(50, 255);

    for (int i = 0; i < K; ++i)
    {
        colors.push_back(glm::vec3(dis(gen) / 255.0f, dis(gen) / 255.0f, dis(gen) / 255.0f));
    }
    return colors;
}
#else
void VisualizerDummy::visualize3D(const float *data, const float *centroids, const int *labels, int N, int K)
{
    std::cerr << "Visualization enabled when compiled with USE_VISUALIZER flag\n";
}
#endif
