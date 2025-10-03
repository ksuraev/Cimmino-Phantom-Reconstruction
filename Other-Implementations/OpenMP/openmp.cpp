/**
 * @file openmp.cpp
 * @brief Implementation of Cimmino's reconstruction algorithm using OpenMP for parallelisation.
 * Compile with: g++-15 -fopenmp -o openmp openmp.cpp ../utilities/Utilities.cpp
 */

#include "../include/Utilities.hpp"
#include <omp.h>
#include <algorithm> 

constexpr uint32_t IMAGE_WIDTH = 256;
constexpr uint32_t IMAGE_HEIGHT = 256;
constexpr uint32_t NUM_ANGLES = 90;
constexpr uint32_t NUM_THREADS = 8;

/**
 * @brief Compute the squared L2 norm of each row in the sparse matrix and the total weight sum.
 * @param projector Sparse projection matrix in CSR format.
 * @param totalRays Total number of rays (rows in the sinogram).
 * @param totalWeightSum Variable to store the total sum of row weights.
 */
void computeTotalWeight(const SparseMatrix& projector, size_t totalRays, float& totalWeightSum) {
    totalWeightSum = 0.0f;
#pragma omp parallel for reduction(+:totalWeightSum) schedule(static) num_threads(NUM_THREADS)
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
 * @brief Calculate the relative error norm between the phantom and the reconstructed image.
 * @param phantom The original phantom image as a flat vector.
 * @param approximation The reconstructed image as a flat vector.
 * @param phantomNorm The precomputed L2 norm of the phantom image.
 * @return The L2 norm of the error between the phantom and the approximation.
 * Computed the same as metal kernels to ensure consistency.
 */
double calculateErrorNorm(std::vector<float>& phantom, std::vector<float>& approximation, double phantomNorm) {
    std::vector<float> A = approximation;
    std::vector<float> P = phantom;

    float differenceSum = 0.0f;
#pragma omp parallel for reduction(+:differenceSum) schedule(static) num_threads(NUM_THREADS)
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
 * @brief Perform Cimmino's reconstruction algorithm using OpenMP for parallelization.
 * @param numIterations Number of iterations to perform.
 * @param projector Sparse projection matrix in CSR format.
 * @param header Header information for the sparse matrix.
 * @param x The image vector to be reconstructed (input and output).
 * @param totalRays Total number of rays (rows in the sinogram).
 * @param sinogram The measured sinogram data as a flat vector.
 * @param phantom The original phantom image for error calculation.
 * @param totalWeightSum Precomputed total sum of row weights.
 * @param phantomNorm Precomputed L2 norm of the phantom image.
 * @param relativeErrorNorm Variable to store the final relative error norm after reconstruction.
 */
void cimminoReconstruct(int numIterations,
    SparseMatrix& projector,
    SparseMatrixHeader& header,
    std::vector<float>& x,
    const size_t& totalRays,
    const std::vector<float>& sinogram,
    std::vector<float>& phantom,
    const float& totalWeightSum,
    const double phantomNorm,
    double& relativeErrorNorm) {

    size_t imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;

    std::vector<float> residuals(totalRays, 0.0f);
    std::vector<float> localX(imageSize, 0.0f);

    for (size_t i = 0; i < numIterations; ++i) {
        // Clear residuals and local update vector
        std::fill(residuals.begin(), residuals.end(), 0.0f);

        // Calculate all residuals for each ray in parallel
#pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t r = 0; r < totalRays; ++r) {
            float dotProduct = 0.0f;
            int rowStart = projector.rows[r];
            int rowEnd = projector.rows[r + 1];
            if (rowStart == rowEnd) continue; // Skip empty rows

#pragma omp simd reduction(+:dotProduct)
            for (int i = rowStart; i < rowEnd; ++i) {
                dotProduct += projector.vals[i] * x[projector.cols[i]];
            }
            residuals[r] = sinogram[r] - dotProduct;
        }

        // Accumulate per-ray contributions into localX (GPU-like atomics)
        std::fill(localX.begin(), localX.end(), 0.0f);

#pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        for (size_t r = 0; r < totalRays; ++r) {
            int rowStart = projector.rows[r];
            int rowEnd = projector.rows[r + 1];
            if (rowStart == rowEnd) continue; // Skip empty rows

            float residual = residuals[r];
            float scalar = (2.0f / totalWeightSum) * residual;

            for (size_t i = rowStart; i < rowEnd; ++i) {
                int   pixelIndex = projector.cols[i];
                float contribution = scalar * projector.vals[i];
#pragma omp atomic
                localX[pixelIndex] += contribution;
            }
        }

        // Update global image vector
#pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        for (size_t i = 0; i < imageSize; ++i) {
            x[i] += localX[i];
        }
        // Check for convergence every 50 iterations
        if ((i + 1) % 50 == 0) {
            relativeErrorNorm = calculateErrorNorm(phantom, x, phantomNorm);
            if (relativeErrorNorm < 1e-2) {
                std::cout << "Converged after " << (i + 1) << " iterations with relative error norm: " << relativeErrorNorm << std::endl;
                break;
            }
        }

    }
    std::cout << "Reconstruction for " << numIterations << " iterations complete." << std::endl;
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
    uint32_t numDetectors = static_cast<uint32_t>(std::ceil(2 * std::sqrt(2) * IMAGE_WIDTH));
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
    // Flip phantom vertically to align with A
    for (size_t y = 0; y < IMAGE_HEIGHT / 2; ++y) {
        for (size_t x = 0; x < IMAGE_WIDTH; ++x) {
            std::swap(phantom[y * IMAGE_WIDTH + x], phantom[(IMAGE_HEIGHT - 1 - y) * IMAGE_WIDTH + x]);
        }
    }

    // Precompute phantom norm for error calculation
    double phantomNorm = 0.0;
    precomputePhantomNorm(phantom, phantomNorm);

    float totalWeightSum = 0.0f;
    computeTotalWeight(projector, totalRays, totalWeightSum);

    // Reconstruct image and time execution
    double relativeErrorNorm = 0.0;
    std::vector<float> reconstructed(IMAGE_WIDTH * IMAGE_HEIGHT, 0.0f);
    double totalReconstructTime = timeMethod_ms([&]() {
        cimminoReconstruct(numIterations, projector, header, reconstructed, totalRays, sinogram, phantom, totalWeightSum, phantomNorm, relativeErrorNorm);
        });

    std::cout << "Reconstruction time for " << numIterations << " iterations: " << totalReconstructTime << " ms." << std::endl;

    // Save reconstructed image to file
    std::string imageSaveFileName = basePath + "data/image_omp_" + std::to_string(numIterations) + ".txt";
    saveImage(imageSaveFileName, reconstructed, geom.imageWidth, geom.imageHeight);

    // Log performance data
    logPerformance("openmp", geom, numIterations, totalReconstructTime, relativeErrorNorm, basePath + "logs/performance_log.csv");
}

