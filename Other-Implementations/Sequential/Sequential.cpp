/**
 * @file Sequential.cpp
 * @brief Sequential implementation of Cimmino's algorithm for CT image reconstruction.
 */

#include "../include/Utilities.hpp"

constexpr uint32_t IMAGE_WIDTH = 256;
constexpr uint32_t IMAGE_HEIGHT = 256;
constexpr uint32_t NUM_ANGLES = 90;

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
 * @brief Precompute the L2 norm of the phantom image.
 * @param phantom The original phantom image as a flat vector.
 * @param phantomNorm Variable to store the computed L2 norm of the phantom.
 */
void precomputePhantomNorm(std::vector<float>& phantom, double& phantomNorm) {
    float phantomNormSum = 0.0f;
    for (const auto& val : phantom) {
        phantomNormSum += val * val;
    }
    phantomNorm = sqrt(static_cast<double>(phantomNormSum));
}

/**
 * @brief Calculate the error norm between the phantom and the reconstructed image.
 * @param phantom The original phantom image as a flat vector.
 * @param approximation The reconstructed image as a flat vector.
 * @param phantomNorm The precomputed L2 norm of the phantom.
 * @return The L2 norm of the error between the phantom and the approximation.
 * Computed the same as metal kernels to ensure consistency.
 */
double calculateErrorNorm(std::vector<float>& phantom, std::vector<float>& approximation, double phantomNorm) {
    std::vector<float> A = approximation;
    std::vector<float> P = phantom;

    float differenceSum = 0.0f;

    for (size_t i = 0; i < A.size(); ++i) {
        float currentValue = A[i];
        float phantomValue = phantom[i];
        float difference = currentValue - phantomValue;

        differenceSum += difference * difference;
    }

    double differenceNorm = std::sqrt(static_cast<double>(differenceSum));
    double relativeErrorNorm = differenceNorm / phantomNorm;
    return relativeErrorNorm;
}

/**
 * @brief Perform Cimmino's algorithm for CT image reconstruction.
 * @param maxIterations The maximum number of iterations to perform.
 * @param projector The sparse projection matrix.
 * @param header The header information for the sparse matrix.
 * @param reconstructedVector The reconstructed image vector (output).
 * @param phantom The original phantom image for error calculation.
 * @param totalRays The total number of rays (rows in the sinogram).
 * @param sinogram The input sinogram data.
 * @param totalWeightSum The total sum of row weights.
 * @param phantomNorm The precomputed L2 norm of the phantom image.
 * @param relativeErrorNorm Variable to store the relative error norm after reconstruction.
 */
void cimminoReconstruct(int maxIterations,
    const SparseMatrix& projector,
    const SparseMatrixHeader& header,
    std::vector<float>& reconstructedVector,
    std::vector<float>& phantom,
    const size_t& totalRays,
    const std::vector<float>& sinogram,
    const float& totalWeightSum,
    double phantomNorm,
    double& relativeErrorNorm) {

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

        // Check for convergence every 50 iterations
        if ((iter + 1) % 50 == 0) {
            relativeErrorNorm = calculateErrorNorm(phantom, reconstructedVector, phantomNorm);
            if (relativeErrorNorm < 1e-2) {
                std::cout << "Converged after " << (iter + 1) << " iterations with relative error norm: " << relativeErrorNorm << std::endl;
                break;
            }
        }
    }
    std::cout << "Reconstruction for " << maxIterations << " iterations complete." << std::endl;
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

    int numIterations = 100;

    if (argc > 1) {
        numIterations = std::atoi(argv[1]);
    }

    // Set geometry parameters
    auto numDetectors = static_cast<uint32_t>(std::ceil(2 * std::sqrt(2) * IMAGE_WIDTH));
    Geometry geom = { IMAGE_WIDTH,  IMAGE_HEIGHT, NUM_ANGLES, numDetectors };
    size_t totalRays = static_cast<size_t>(geom.nAngles * geom.nDetectors);

    // Load projection matrix from file
    SparseMatrixHeader header = { 0, 0, 0 };
    SparseMatrix projector;
    if (!loadSparseMatrixBinary(basePath + "data/projection_256.bin", projector, header, totalRays)) {
        std::cerr << "Failed to load sparse projection matrix." << std::endl;
        return -1;
    }

    // Load sinogram from file
    std::vector<float> sinogram(totalRays, 0.0f);
    if (!loadSinogram(basePath + "data/sinogram_256.txt", sinogram)) {
        std::cerr << "Failed to load sinogram." << std::endl;
        return -1;
    }

    // Load phantom from file
    std::vector<float> phantom = loadPhantom((basePath + "data/phantom_256.txt").c_str(), geom);
    if (phantom.empty()) {
        std::cerr << "Failed to load phantom." << std::endl;
        return -1;
    }

    // Flip phantom vertically to match orientation
    for (size_t y = 0; y < IMAGE_HEIGHT / 2; ++y) {
        for (size_t x = 0; x < IMAGE_WIDTH; ++x) {
            std::swap(phantom[y * IMAGE_WIDTH + x], phantom[(IMAGE_HEIGHT - 1 - y) * IMAGE_WIDTH + x]);
        }
    }

    // Precompute phantom norm for error calculation
    double phantomNorm = 0.0;
    precomputePhantomNorm(phantom, phantomNorm);

    // Compute row weights and total weight sum
    float totalWeightSum = 0.0f;
    computeTotalWeight(projector, totalRays, totalWeightSum);

    // Reconstruct image and time execution
    std::vector<float> reconstructedImage(IMAGE_WIDTH * IMAGE_HEIGHT, 0.0f);
    double relativeErrorNorm = 0.0;
    auto totalReconstructTime = timeMethod_ms([&]() {
        cimminoReconstruct(numIterations, projector, header, reconstructedImage, phantom, totalRays, sinogram, totalWeightSum, phantomNorm, relativeErrorNorm);
        });

    std::cout << "Total reconstruction time (ms): " << totalReconstructTime << std::endl;

    // Save reconstructed image to file
    std::string imageSaveFileName = basePath + "data/image_seq_" + std::to_string(numIterations) + ".txt";
    saveImage(imageSaveFileName, reconstructedImage, geom.imageWidth, geom.imageHeight);

    // Log performance
    logPerformance("Sequential", geom, numIterations, totalReconstructTime, relativeErrorNorm, basePath + "logs/performance_log.csv");
}

