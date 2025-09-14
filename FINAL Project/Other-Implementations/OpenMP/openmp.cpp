// g++-15 -fopenmp -o openmp openmp.cpp Utilities.cpp

#include "../include/Utilities.hpp"
#include "omp.h"

#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define NUM_ANGLES 90

#define NUM_THREADS 8

/**
 * @brief Compute the squared L2 norm of each row in the sparse matrix and the total weight sum.
 * @param projector Sparse projection matrix in CSR format.
 * @param rowWeights Vector to store the squared L2 norm of each row.
 * @param totalRays Total number of rays (rows in the sinogram).
 * @param totalWeightSum Variable to store the total sum of row weights.
 */
void computeRowWeights(const SparseMatrix& projector, size_t totalRays, float& totalWeightSum) {
    std::vector<float> rowWeights(totalRays, 0.0f);

#pragma omp parallel num_threads(NUM_THREADS) 
    {
        std::vector<float> localRowWeights(totalRays, 0.0f);

#pragma omp for 
        for (size_t r = 0; r < totalRays; ++r) {
            int rowStart = projector.rows[r];
            int rowEnd = projector.rows[r + 1];

            for (int i = rowStart; i < rowEnd; ++i) {
                localRowWeights[r] += projector.vals[i] * projector.vals[i];
            }
        }

#pragma omp reduction(+:rowWeights)
        for (size_t r = 0; r < totalRays; ++r) {
            rowWeights[r] += localRowWeights[r];
        }
    }

    // Compute total weight sum
    totalWeightSum = std::accumulate(rowWeights.begin(), rowWeights.end(), 0.0f);
}

void cimminoReconstruct(int maxIterations,
    SparseMatrix& projector,
    SparseMatrixHeader& header,
    std::vector<float>& x,
    const size_t& totalRays,
    const std::vector<float>& sinogram, const float& totalWeightSum) {

    size_t imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;

    std::vector<float> residuals(totalRays, 0.0f);

    for (int iter = 0; iter < maxIterations; ++iter) {
        // Pass 1: Calculate all residuals
        std::fill(residuals.begin(), residuals.end(), 0.0f);

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

        // Pass 2: Update x
#pragma omp parallel num_threads(NUM_THREADS)
        {
            std::vector<float> localX(imageSize, 0.0f);

#pragma omp for 
            for (size_t r = 0; r < totalRays; ++r) {
                float residual = residuals[r];

                float scalar = (2.0f / totalWeightSum) * residual;
                int rowStart = projector.rows[r];
                int rowEnd = projector.rows[r + 1];

                for (int i = rowStart; i < rowEnd; ++i) {
                    int index = projector.cols[i];
                    float weight = projector.vals[i];
                    localX[index] += scalar * weight;
                }
            }

#pragma omp reduction(+:x)
            for (size_t i = 0; i < imageSize; ++i) {
                x[i] += localX[i];
            }
        }
    }

    std::cout << "Reconstruction for " << maxIterations << " iterations complete." << std::endl;
}




int main(int argc, const char* argv[]) {
    // Set geometry parameters
    auto numDetectors = static_cast<int>(std::ceil(2 * std::sqrt(2) * IMAGE_WIDTH));

    Geometry geom = { IMAGE_WIDTH,  IMAGE_HEIGHT, NUM_ANGLES,
                     numDetectors };

    size_t totalRays = static_cast<size_t>(geom.nAngles * geom.nDetectors);

    SparseMatrixHeader header = { 0, 0, 0 };
    SparseMatrix projector;

    // Load projection matrix from file
    if (!loadSparseMatrixBinary("../data/projection_256.bin", projector, header, totalRays)) {
        std::cerr << "Failed to load sparse projection matrix." << std::endl;
        return -1;
    }

    // Load sinogram from file
    std::vector<float> sinogram(totalRays, 0.0f);

    if (!loadSinogram("../data/sinogram_256.bin", sinogram, totalRays)) {
        std::cerr << "Failed to load sinogram." << std::endl;
        return -1;
    }

    float totalWeightSum = 0.0f;
    computeRowWeights(projector, totalRays, totalWeightSum);

    // Reconstruct image and time execution
    int numIterations = 1000;
    std::vector<float> reconstructed(IMAGE_WIDTH * IMAGE_HEIGHT, 0.0f);
    double totalReconstructTime = timeMethod_ms([&]() {
        cimminoReconstruct(numIterations, projector, header, reconstructed, totalRays, sinogram, totalWeightSum);
        });

    std::cout << "Reconstruction time for " << numIterations << " iterations: " << totalReconstructTime << " ms." << std::endl;

    saveImage("image_omp.txt", reconstructed, geom.imageWidth, geom.imageHeight);

    logPerformance("openmp", geom, numIterations, totalReconstructTime);
}

