// Minimal CUDA k-means implementation.
// Build with: nvcc -O2 simple_kmeans.cu -o simple_kmeans

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do {                                           \
    const cudaError_t error = (call);                                  \
    if (error != cudaSuccess) {                                        \
        std::cerr << "CUDA error in " << #call << ": "                \
                  << cudaGetErrorString(error)                         \
                  << " (line " << __LINE__ << ")\n";                 \
        return 1;                                                       \
    }                                                                  \
} while (0)

constexpr int kPointCount = 6000;
constexpr int kClusterCount = 3;
constexpr int kDimensions = 2;
constexpr int kMaxIterations = 30;
constexpr int kThreadsPerBlock = 256;
constexpr float kConvergenceTolerance = 0.001f;

// One thread handles one point and atomically updates its cluster totals.
__global__ void kmeans_step(const float* points, const float* centers,
                            float* sums, int* counts, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int best_cluster = 0;
    float best_distance = FLT_MAX;
    for (int c = 0; c < kClusterCount; ++c) {
        float dx = points[2 * i]     - centers[2 * c];
        float dy = points[2 * i + 1] - centers[2 * c + 1];
        float distance = dx * dx + dy * dy;
        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = c;
        }
    }

    atomicAdd(&counts[best_cluster], 1);
    atomicAdd(&sums[2 * best_cluster], points[2 * i]);
    atomicAdd(&sums[2 * best_cluster + 1], points[2 * i + 1]);
}

int main() {
    std::mt19937 generator(42);
    std::normal_distribution<float> noise(0.0f, 0.7f);
    std::vector<float> points(kDimensions * kPointCount);

    // Generate three noisy 2D groups around fixed centers.
    const float true_centers[kClusterCount][kDimensions] = {
        {0, 0}, {6, 6}, {12, 0}
    };
    for (int i = 0; i < kPointCount; ++i) {
        const int cluster = i % kClusterCount;
        points[2 * i]     = true_centers[cluster][0] + noise(generator);
        points[2 * i + 1] = true_centers[cluster][1] + noise(generator);
    }

    std::vector<float> centers = {0, 1, 6, 5, 12, 1};
    float* device_points = nullptr;
    float* device_centers = nullptr;
    float* device_sums = nullptr;
    int* device_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&device_points, points.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_centers, centers.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_sums, centers.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_counts, kClusterCount * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_points, points.data(), points.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::vector<float> sums(kDimensions * kClusterCount);
    std::vector<int> counts(kClusterCount);
    std::cout << "[INFO] points=" << kPointCount
              << " clusters=" << kClusterCount << '\n';

    for (int iteration = 1; iteration <= kMaxIterations; ++iteration) {
        CUDA_CHECK(cudaMemcpy(device_centers, centers.data(),
                              centers.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(device_sums, 0, sums.size() * sizeof(float)));
        CUDA_CHECK(cudaMemset(device_counts, 0, counts.size() * sizeof(int)));

        kmeans_step<<<(kPointCount + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock>>>(device_points, device_centers,
                                           device_sums, device_counts,
                                           kPointCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(sums.data(), device_sums,
                              sums.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(counts.data(), device_counts,
                              counts.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));

        float shift = 0.0f;
        for (int c = 0; c < kClusterCount; ++c) {
            float old_x = centers[2 * c], old_y = centers[2 * c + 1];
            if (counts[c] > 0) {
                centers[2 * c]     = sums[2 * c] / counts[c];
                centers[2 * c + 1] = sums[2 * c + 1] / counts[c];
            }
            float dx = centers[2 * c] - old_x;
            float dy = centers[2 * c + 1] - old_y;
            shift = fmaxf(shift, sqrtf(dx * dx + dy * dy));
        }

        std::cout << "[ITER " << iteration << "] shift=" << shift
                  << " counts=" << counts[0] << "," << counts[1]
                  << "," << counts[2] << "\n";
        if (shift < kConvergenceTolerance) break;
    }

    std::cout << "[RESULT] centers:\n";
    for (int c = 0; c < kClusterCount; ++c)
        std::cout << "  " << c << ": (" << centers[2 * c] << ", "
                  << centers[2 * c + 1] << ")\n";

    cudaFree(device_points);
    cudaFree(device_centers);
    cudaFree(device_sums);
    cudaFree(device_counts);
    return 0;
}
