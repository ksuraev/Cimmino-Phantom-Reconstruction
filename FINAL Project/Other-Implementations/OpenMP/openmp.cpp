// g++-15 -fopenmp -o openmp openmp.cpp ../utilities/Utilities.cpp

#include "../include/Utilities.hpp"
#include <omp.h>
#include <algorithm> 


#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define NUM_ANGLES 90

#define NUM_THREADS 8

/**
 * @brief Compute the squared L2 norm of each row in the sparse matrix and the total weight sum.
 * @param projector Sparse projection matrix in CSR format.
 * @param totalRays Total number of rays (rows in the sinogram).
 * @param totalWeightSum Variable to store the total sum of row weights.
 */
void computeRowWeights(const SparseMatrix& projector, size_t totalRays, float& totalWeightSum) {
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
 * @brief Reconstruct image using Cimmino's algorithm with OpenMP parallelisation.
 * @param numIterations The number of iterations to perform.
 * @param projector The sparse projection matrix.
 * @param header The header information for the sparse matrix.
 * @param x The reconstructed image vector (output).
 * @param totalRays The total number of rays (rows in the sinogram).
 * @param sinogram The input sinogram data.
 * @param totalWeightSum The total sum of row weights.
 */
void cimminoReconstruct(int numIterations,
    SparseMatrix& projector,
    SparseMatrixHeader& header,
    std::vector<float>& x,
    const size_t& totalRays,
    const std::vector<float>& sinogram,
    const float& totalWeightSum) {

    size_t imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;

    std::vector<float> residuals(totalRays, 0.0f);
    std::vector<float> localX(imageSize, 0.0f);

    for (int iter = 0; iter < numIterations; ++iter) {
        // Clear residuals
        std::fill(residuals.begin(), residuals.end(), 0.0f);

        // Pass 1: Calculate all residuals
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

            for (int i = rowStart; i < rowEnd; ++i) {
                int   pixelIndex = projector.cols[i];
                float contribution = scalar * projector.vals[i];
#pragma omp atomic
                localX[pixelIndex] += contribution;
            }
        }

        // Apply the update once
#pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        for (size_t i = 0; i < imageSize; ++i) {
            x[i] += localX[i];
        }

    }
    std::cout << "Reconstruction for " << numIterations << " iterations complete." << std::endl;
}

double calculateErrorNorm(const std::vector<float> phantom, const std::vector<float>approximation) {
    if (phantom.size() != approximation.size()) {
        std::cerr << "Error: Vectors must be of the same size to calculate error norm." << std::endl;
        return -1.0f;
    }

    // Work on copies so we don't mutate inputs
    std::vector<float> A = approximation;
    std::vector<float> P = phantom;

    // Transpose A
    for (size_t y = 0; y < IMAGE_HEIGHT; ++y) {
        for (size_t x = y + 1; x < IMAGE_WIDTH; ++x) {
            std::swap(A[y * IMAGE_WIDTH + x], A[x * IMAGE_WIDTH + y]);  // IMAGE_WIDTH, not IMAGE_HEIGHT
        }
    }

    // Flip P vertically to align with A
    for (size_t y = 0; y < IMAGE_HEIGHT / 2; ++y) {
        for (size_t x = 0; x < IMAGE_WIDTH; ++x) {
            std::swap(P[y * IMAGE_WIDTH + x], P[(IMAGE_HEIGHT - 1 - y) * IMAGE_WIDTH + x]);
        }
    }

    double sse = 0.0;
#pragma omp parallel for reduction(+:sse) schedule(static) num_threads(NUM_THREADS)
    for (size_t i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
        double d = (double)A[i] - (double)P[i];
        sse += d * d;
    }
    // Print sse
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
    int numDetectors = static_cast<int>(std::ceil(2 * std::sqrt(2) * IMAGE_WIDTH));

    Geometry geom = { IMAGE_WIDTH,  IMAGE_HEIGHT, NUM_ANGLES, numDetectors };

    size_t totalRays = static_cast<size_t>(geom.nAngles * geom.nDetectors);

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

    float totalWeightSum = 0.0f;
    computeRowWeights(projector, totalRays, totalWeightSum);

    // Reconstruct image and time execution
    std::vector<float> reconstructed(IMAGE_WIDTH * IMAGE_HEIGHT, 0.0f);
    double totalReconstructTime = timeMethod_ms([&]() {
        cimminoReconstruct(numIterations, projector, header, reconstructed, totalRays, sinogram, totalWeightSum);
        });

    std::cout << "Reconstruction time for " << numIterations << " iterations: " << totalReconstructTime << " ms." << std::endl;

    std::string imageSaveFileName = basePath + "data/image_omp_" + std::to_string(numIterations) + ".txt";

    saveImage(imageSaveFileName, reconstructed, geom.imageWidth, geom.imageHeight);

    // Calculate error norm between phantom and reconstruction
    double errorNorm = calculateErrorNorm(phantom, reconstructed);

    std::cout << "Error norm between phantom and reconstruction: " << errorNorm << std::endl;

    logPerformance("openmp", geom, numIterations, totalReconstructTime, errorNorm, basePath + "logs/performance_log.csv");
}

