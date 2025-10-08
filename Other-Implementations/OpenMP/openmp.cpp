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
constexpr uint32_t NUM_ANGLES = 360;
constexpr uint32_t NUM_THREADS = 8;

constexpr const char* LOG_FILE = "logs/performance_log.csv";
constexpr const char* PROJECTION_FILE = "data/projection_256.bin";
constexpr const char* PHANTOM_FILE = "data/phantom_256.txt";

constexpr float RELAXATION_FACTOR = 350.0f;

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
        float phantomValue = P[i];
        float difference = currentValue - phantomValue;

        differenceSum += difference * difference;
    }

    double differenceNorm = std::sqrt(static_cast<double>(differenceSum));
    double relativeErrorNorm = differenceNorm / phantomNorm;
    return relativeErrorNorm;
}

/**
 * @brief Compute the sinogram by multiplying the sparse projection matrix with the phantom data.
 * @param phantomData The phantom image data as a flat vector.
 * @param projector The sparse projection matrix in CSR format.
 * @param totalRays The total number of rays (rows in the sinogram).
 * @param sinogram The output sinogram vector (modified in place).
 * @note Simulates performing a scan.
 */
void computeSinogram(
    std::vector<float>& phantomData,
    const SparseMatrix& projector,
    const size_t& totalRays,
    std::vector<float>& sinogram) {

    // Initialise sinogram to zero
    std::fill(sinogram.begin(), sinogram.end(), 0.0f);

    // Compute sinogram by multiplying projector with phantom data
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
    for (size_t r = 0; r < totalRays; ++r) {
        int rowStart = projector.rows[r];
        int rowEnd = projector.rows[r + 1];

        float dotProduct = 0.0f;
        for (size_t i = rowStart; i < rowEnd; ++i) {
            dotProduct += projector.vals[i] * phantomData[projector.cols[i]];
        }
        sinogram[r] = dotProduct;
    }
}

/**
 * @brief Normalise each row of the sparse projection matrix to have unit L2 norm.
 * @param projector The sparse projection matrix in CSR format (modified in place).
 * @param totalRays The total number of rays (rows in the sinogram).
 */
void normaliseProjectionMatrix(SparseMatrix& projector, size_t totalRays, float& totalWeightSum) {
    totalWeightSum = 0.0f;
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
    for (size_t i = 0; i < totalRays; ++i) {
        double rowNormSq = 0.0;

        // Compute the squared norm of the row
        for (size_t j = projector.rows[i]; j < projector.rows[i + 1]; ++j) {
            rowNormSq += static_cast<double>(projector.vals[j] * projector.vals[j]);
        }

        float rowNorm = static_cast<float>(sqrt(rowNormSq));
        if (rowNorm > 0.0f) {
            // Normalise the row and accumulate the normalised weight sum
            for (size_t j = projector.rows[i]; j < projector.rows[i + 1]; ++j) {
                projector.vals[j] /= rowNorm;
            }
#pragma omp atomic
            totalWeightSum += 1.0f;  // Each normalised row has unit norm
        }
    }
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
    const SparseMatrix& projector,
    const SparseMatrixHeader& header,
    std::vector<float>& x,
    const size_t& totalRays,
    const std::vector<float>& sinogram,
    std::vector<float>& phantom,
    const float totalWeightSum,
    const double phantomNorm,
    double& relativeErrorNorm) {

    size_t imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;

    std::vector<float> residuals(totalRays, 0.0f);
    std::vector<float> localX(imageSize, 0.0f);

    for (size_t i = 0; i < numIterations; ++i) {
        // Calculate all residuals for each ray in parallel
#pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t r = 0; r < totalRays; ++r) {
            float dotProduct = 0.0f;
            int rowStart = projector.rows[r];
            int rowEnd = projector.rows[r + 1];

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

            float residual = residuals[r];
            float scalar = RELAXATION_FACTOR * (2.0f / totalWeightSum) * residual;

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
            if (x[i] + localX[i] < 0.0f) x[i] = 0.0f;
            else x[i] += localX[i];
        }
        // Check for convergence every 50 iterations
        if ((i + 1) % 50 == 0) {
            relativeErrorNorm = calculateErrorNorm(phantom, x, phantomNorm);
            if (relativeErrorNorm < 1e-2) {
                std::cout << "Converged after " << (i + 1) << " iterations with relative error norm: " << relativeErrorNorm << std::endl;
                break;
            }
        }
        // Clear residuals 
        std::fill(residuals.begin(), residuals.end(), 0.0f);
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
    if (!loadSparseMatrixBinary(basePath + PROJECTION_FILE, projector, header, totalRays)) {
        std::cerr << "Failed to load sparse projection matrix." << std::endl;
        return -1;
    }
    float totalWeightSum = 0.0f;
    normaliseProjectionMatrix(projector, totalRays, totalWeightSum);
    std::cout << "Total weight sum after normalisation: " << totalWeightSum << std::endl;
    // Load phantom from file
    std::vector<float> phantom = loadPhantom((basePath + PHANTOM_FILE).c_str(), geom);
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

    // Compute sinogram
    std::vector<float> sinogram(totalRays, 0.0f);
    auto scanTime = timeMethod_ms([&]() {
        computeSinogram(phantom, projector, totalRays, sinogram);
        });
    std::cout << "Sinogram computation time (ms): " << scanTime << std::endl;

    // Precompute phantom norm for error calculation
    double phantomNorm = 0.0;
    precomputePhantomNorm(phantom, phantomNorm);
    std::cout << "Phantom L2 norm: " << phantomNorm << std::endl;
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
    logPerformance("openmp-enhanced", geom, numIterations, totalReconstructTime, relativeErrorNorm, scanTime, basePath + LOG_FILE);
}

