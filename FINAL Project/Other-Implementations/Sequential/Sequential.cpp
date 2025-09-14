//  g++-15 -o sequential sequential.cpp Utilities.cpp
#include "Utilities.hpp"


#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define NUM_ANGLES 90

/**
 * @brief Compute the squared L2 norm of each row in the sparse matrix and the total weight sum.
 * @param projector Sparse projection matrix in CSR format.
 * @param rowWeights Vector to store the squared L2 norm of each row.
 * @param totalRays Total number of rays (rows in the sinogram).
 * @param totalWeightSum Variable to store the total sum of row weights.
 */
void computeRowWeights(const SparseMatrix& projector, size_t totalRays, float& totalWeightSum) {
    std::vector<float> rowWeights(totalRays, 0.0f);

    for (size_t r = 0; r < totalRays; ++r) {
        int rowStart = projector.rows[r];
        int rowEnd = projector.rows[r + 1];

        for (int i = rowStart; i < rowEnd; ++i) {
            rowWeights[r] += projector.vals[i] * projector.vals[i];
        }
    }

    // Compute total weight sum
    totalWeightSum = std::accumulate(rowWeights.begin(), rowWeights.end(), 0.0f);
}

/**
 * @brief Perform Cimmino reconstruction.
 * @param maxIterations Number of iterations to perform.
 * @param projector Sparse projection matrix.
 * @param header Header information for the sparse matrix.
 * @param totalRays Total number of rays (rows in the sinogram).
 * @param sinogram Input sinogram data.
 * @return Reconstructed image as a flat vector.
 */
void cimminoReconstruct(int maxIterations,
    const SparseMatrix& projector,
    const SparseMatrixHeader& header,
    std::vector<float>& reconstructedVector,
    const size_t& totalRays,
    const std::vector<float>& sinogram,
    const float& totalWeightSum) {

    size_t imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;

    std::vector<float> residuals(totalRays);

    for (int iter = 0; iter < maxIterations; ++iter) {

        // Pass 1: Calculate all residuals
        std::fill(residuals.begin(), residuals.end(), 0.0f);

        for (size_t r = 0; r < totalRays; ++r) {
            float dotProduct = 0.0f;
            int rowStart = projector.rows[r];
            int rowEnd = projector.rows[r + 1];

            for (size_t i = rowStart; i < rowEnd; ++i) {
                dotProduct += projector.vals[i] * reconstructedVector[projector.cols[i]];
            }
            residuals[r] = sinogram[r] - dotProduct;
        }

        // Pass 2: Update reconstructedVector
        for (size_t r = 0; r < totalRays; ++r) {
            float residual = residuals[r];

            float scalar = (2.0f / totalWeightSum) * residual;
            int rowStart = projector.rows[r];
            int rowEnd = projector.rows[r + 1];

            for (size_t i = rowStart; i < rowEnd; ++i) {
                int index = projector.cols[i];
                float weight = projector.vals[i];
                reconstructedVector[index] += scalar * weight;
            }
        }
    }
    std::cout << "Reconstruction for " << maxIterations << " iterations complete." << std::endl;
    // return reconstructedVector;
}

/**
 * @brief Times the execution of a method and returns the duration in microseconds.
 * @param methodToTime The method to be timed.
 * @return The duration of the method execution in microseconds.
 * Inspired by Maksym's code from lecture 14/07/2025
 */
static double timeMethod_ms(const std::function<void()>& methodToTime) {
    auto start = std::chrono::high_resolution_clock::now();
    methodToTime();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

int main(int argc, const char* argv[]) {
    // Set geometry parameters
    auto numDetectors = static_cast<int>(std::ceil(2 * std::sqrt(2) * IMAGE_WIDTH));

    Geometry geom = { IMAGE_WIDTH,  IMAGE_HEIGHT, NUM_ANGLES, numDetectors };

    size_t totalRays = static_cast<size_t>(geom.nAngles * geom.nDetectors);

    // Sparse matrix parameters
    SparseMatrixHeader header = { 0, 0, 0 };
    SparseMatrix projector;

    // Load projection matrix from file
    if (!loadSparseMatrixBinary("projection_256.bin", projector, header, totalRays)) {
        std::cerr << "Failed to load sparse projection matrix." << std::endl;
        return -1;
    }

    // Load sinogram from file
    std::vector<float> sinogram(totalRays, 0.0f);
    loadSinogram("sinogram_256.bin", sinogram, totalRays);

    // Compute row weights and total weight sum
    float totalWeightSum = 0.0f;
    computeRowWeights(projector, totalRays, totalWeightSum);

    // Reconstruct image and time execution
    std::vector<float> reconstructedImage(IMAGE_WIDTH * IMAGE_HEIGHT, 0.0f);
    int numIterations = 10;

    auto totalReconstructTime = timeMethod_ms([&]() {
        cimminoReconstruct(numIterations, projector, header, reconstructedImage, totalRays, sinogram, totalWeightSum);
        });

    std::cout << "Total reconstruction time (ms): " << totalReconstructTime << std::endl;

    // Save image to txt file for viewing 
    saveImage("image_seq.txt", reconstructedImage, geom.imageWidth, geom.imageHeight);

    // Log performance
    logPerformance("Sequential", geom, numIterations, totalReconstructTime);
}

