#ifndef CUDAKMEANS_VISUALIZER_OPENGL_H
#define CUDAKMEANS_VISUALIZER_OPENGL_H

#include <random>
#include <stdexcept>
#include <vector>
#ifdef USE_VISUALIZER
#include <GLFW/glfw3.h>
#include <glad/gl.h>
#include <glm/glm.hpp>
#endif

class Visualizer
{
    public:
    virtual ~Visualizer() {}
    /**
     * @brief Visualize the clustering results in 3D space
     * @param data The data points (N x 3).
     * @param centroids The cluster centroids (K x 3).
     * @param labels The cluster labels for each point.
     * @param N Number of data points.
     * @param K Number of clusters.
     */
    virtual void visualize3D(const float *data, const float *centroids, const int *labels, int N, int K) = 0;
};

#ifdef USE_VISUALIZER
class VisualizerOpenGL : public Visualizer
{
    public:
    VisualizerOpenGL();
    ~VisualizerOpenGL();

    void visualize3D(const float *data, const float *centroids, const int *labels, int N, int K) override;

    private:
    void initOpenGL();
    void cleanupOpenGL();
    unsigned int createShaderProgram(const char *vertPath, const char *fragPath);

    void drawAxes(const glm::mat4 &mvp);
    void drawPoints(
        const float *points, const int *labels, int count, const std::vector<glm::vec3> &colors, const glm::mat4 &mvp
    );
    void drawWireSpheres(const float *centroids, int K, const std::vector<glm::vec3> &colors, const glm::mat4 &mvp);

    static void normalizeData(
        const float *data, float *normalizedData, int N, const float *centroids, float *normalizedCentroids, int K,
        float scale
    );
    static std::vector<int> downsampleData(int N, int sampleSize);
    static std::vector<glm::vec3> generateClusterColors(int K);

    // Helper for sphere wireframe
    void generateSphereWireframe(std::vector<glm::vec3> &sphereVertices, float radius, int segments, int rings);

    GLFWwindow *window         = nullptr;
    unsigned int shaderProgram = 0;

    static constexpr int maxRenderPoints = 1e6; // Maximum points to render for performance
};
#else
class VisualizerDummy : public Visualizer
{
    public:
    void visualize3D(const float *data, const float *centroids, const int *labels, int N, int K) override;
};
#endif

#endif // CUDAKMEANS_VISUALIZER_OPENGL_H
