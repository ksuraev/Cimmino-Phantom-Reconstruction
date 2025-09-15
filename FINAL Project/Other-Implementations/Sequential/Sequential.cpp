/**
 * @file Sequential.cpp
 * @brief Sequential implementation of Cimmino's algorithm for CT image reconstruction.
 */

#include "../include/Utilities.hpp"

#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define NUM_ANGLES 90

 /**
  * @brief Compute the squared L2 norm of each row in the sparse matrix and the total weight sum.
  * @param projector Sparse projection matrix in CSR format.
  * @param totalRays Total number of rays (rows in the sinogram).
  * @param totalWeightSum Variable to store the total sum of row weights.
  */
void computeTotalWeight(const SparseMatrix& projector, size_t totalRays, float& totalWeightSum) {
    totalWeightSum = 0.0f;
    for (size_t r = 0; r < totalRays; ++r) {
        double rowNormSq = 0.0f;
        for (int i = projector.rows[r]; i < projector.rows[r + 1]; ++i)
            rowNormSq += static_cast<double>(projector.vals[i] * projector.vals[i]);
        totalWeightSum += static_cast<float>(rowNormSq);;
    }
}

/**
 * @brief Perform Cimmino reconstruction.
 * @param maxIterations Number of iterations to perform.
 * @param projector Sparse projection matrix.
 * @param header Header information for the sparse matrix.
 * @param totalRays Total number of rays (rows in the sinogram).
 * @param sinogram Input sinogram data.
 * @param totalWeightSum Total sum of row weights.
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
}

/**
 * @brief Calculate the error norm between the phantom and the reconstructed image.
 * @param phantom The original phantom image as a flat vector.
 * @param approximation The reconstructed image as a flat vector.
 * @return The L2 norm of the error between the phantom and the approximation.
 */
double calculateErrorNorm(std::vector<float>& phantom, std::vector<float>& approximation) {
    if (phantom.size() != approximation.size()) {
        std::cerr << "Error: Vectors must be of the same size to calculate error norm." << std::endl;
        return -1.0f;
    }
    std::vector<float> A = approximation;
    std::vector<float> P = phantom;

    // Flip phantom vertically to match orientation
    for (size_t y = 0; y < IMAGE_HEIGHT / 2; ++y) {
        for (size_t x = 0; x < IMAGE_WIDTH; ++x) {
            std::swap(P[y * IMAGE_WIDTH + x], P[(IMAGE_HEIGHT - 1 - y) * IMAGE_WIDTH + x]);
        }
    }

    // Transpose approximation
    for (size_t y = 0; y < IMAGE_HEIGHT; ++y) {
        for (size_t x = y + 1; x < IMAGE_WIDTH; ++x) {
            std::swap(A[y * IMAGE_WIDTH + x], A[x * IMAGE_WIDTH + y]);
        }
    }

    // Compute L2 norm of the difference
    double sse = 0.0;
    for (size_t i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
        double d = (double)A[i] - (double)P[i];
        sse += d * d;
    }
    double norm = std::sqrt(sse);
    std::cout << "Sum of Squared Errors (SSE): " << norm << std::endl;
    return norm;
}

int main(int argc, const char* argv[]) {
    // Allow program to run from execution script or local folder
    std::string basePath;
    try {
        basePath = getBasePath();
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Default value
    int numIterations = 100;

    if (argc > 1) {
        numIterations = std::atoi(argv[1]); // Convert the first argument to an integer
    }

    // Set geometry parameters
    auto numDetectors = static_cast<int>(std::ceil(2 * std::sqrt(2) * IMAGE_WIDTH));

    Geometry geom = { IMAGE_WIDTH,  IMAGE_HEIGHT, NUM_ANGLES, numDetectors };

    size_t totalRays = static_cast<size_t>(geom.nAngles * geom.nDetectors);

    // Sparse matrix parameters
    SparseMatrixHeader header = { 0, 0, 0 };
    SparseMatrix projector;

    // Load projection matrix from file
    if (!loadSparseMatrixBinary(basePath + "data/projection_256_astra.bin", projector, header, totalRays)) {
        std::cerr << "Failed to load sparse projection matrix." << std::endl;
        return -1;
    }

    // Load sinogram from file
    std::vector<float> sinogram(totalRays, 0.0f);
    if (!loadSinogram(basePath + "data/sinogram_256.bin", sinogram, totalRays)) {
        std::cerr << "Failed to load sinogram." << std::endl;
        return -1;
    }

    // Load phantom from file
    std::vector<float> phantom = loadPhantom((basePath + "data/phantom_256.txt").c_str(), geom);
    if (phantom.empty()) {
        std::cerr << "Failed to load phantom." << std::endl;
        return -1;
    }

    // Compute row weights and total weight sum
    float totalWeightSum = 0.0f;
    computeTotalWeight(projector, totalRays, totalWeightSum);

    // Reconstruct image and time execution
    std::vector<float> reconstructedImage(IMAGE_WIDTH * IMAGE_HEIGHT, 0.0f);

    auto totalReconstructTime = timeMethod_ms([&]() {
        cimminoReconstruct(numIterations, projector, header, reconstructedImage, totalRays, sinogram, totalWeightSum);
        });

    std::cout << "Total reconstruction time (ms): " << totalReconstructTime << std::endl;

    // Save reconstructed image to file
    std::string imageSaveFileName = basePath + "data/image_seq_" + std::to_string(numIterations) + ".txt";

    // Calculate error norm between phantom and reconstruction
    double errorNorm = calculateErrorNorm(phantom, reconstructedImage);

    // Save image to txt file for viewing 
    saveImage(imageSaveFileName, reconstructedImage, geom.imageWidth, geom.imageHeight);

    // Log performance
    logPerformance("Sequential", geom, numIterations, totalReconstructTime, errorNorm, basePath + "logs/performance_log.csv");
}

