// g++-15 -fopenmp -o openmp openmp.cpp Utilities.cpp

#include "../include/Utilities.hpp"
#include <omp.h>


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
    totalWeightSum = 0.0f;
#pragma omp parallel for reduction(+:totalWeightSum) schedule(static) num_threads(NUM_THREADS)
    for (size_t r = 0; r < totalRays; ++r) {
        double rowNormSq = 0.0f;
        for (int i = projector.rows[r]; i < projector.rows[r + 1]; ++i)
            rowNormSq += static_cast<double>(projector.vals[i] * projector.vals[i]);
        // if (rowNormSq < 1e-9) rowNormSq = 1.0;
        totalWeightSum += static_cast<float>(rowNormSq);;
    }
    std::cout << "Total weight sum: " << totalWeightSum << std::endl;
}

void cimminoReconstruct(int maxIterations,
    SparseMatrix& projector,
    SparseMatrixHeader& header,
    std::vector<float>& x,
    const size_t& totalRays,
    const std::vector<float>& sinogram,
    const float& totalWeightSum) {

    size_t imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;

    std::vector<float> residuals(totalRays, 0.0f);

    for (int iter = 0; iter < maxIterations; ++iter) {
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

        // Pass 2: Update x
#pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        for (size_t r = 0; r < totalRays; ++r) {
            float residual = residuals[r];
            float scalar = (2.0f / totalWeightSum) * residual;
            int rowStart = projector.rows[r];
            int rowEnd = projector.rows[r + 1];
            if (rowStart == rowEnd) continue; // Skip empty rows
            for (int i = rowStart; i < rowEnd; ++i) {
                int   pixelIndex = projector.cols[i];
                float weight = projector.vals[i];
                float contribution = scalar * weight;
#pragma omp atomic
                x[pixelIndex] += contribution;
            }
        }
    }

    std::cout << "Reconstruction for " << maxIterations << " iterations complete." << std::endl;
}

double calculateErrorNorm(std::vector<float>& phantom, std::vector<float>& approximation) {
    if (phantom.size() != approximation.size()) {
        std::cerr << "Error: Vectors must be of the same size to calculate error norm." << std::endl;
        return -1.0f;
    }
    // // Flip the phantom vertically
    // std::vector<float> flippedPhantom = flipImageVertically(phantom, IMAGE_WIDTH, IMAGE_HEIGHT);

    // // Transpose the temp matrix in place for correct orientation and comparison
    // for (size_t y = 0; y < IMAGE_HEIGHT; ++y) {
    //     for (size_t x = y + 1; x < IMAGE_WIDTH; ++x) {
    //         std::swap(approximation[y * IMAGE_WIDTH + x], approximation[x * IMAGE_HEIGHT + y]);
    //     }
    // }

    // Compute residual error directly
    int imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;
    double sumOfSquares = 0.0;
    for (int i = 0; i < imageSize; ++i) {
        float diff = approximation[i] - phantom[i];
        sumOfSquares += diff * diff;
    }
    double finalUpdateNorm = sqrt(sumOfSquares);

    std::cout << "Final update norm (L2): " << finalUpdateNorm << std::endl;
    return finalUpdateNorm;
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

    double errorNorm = calculateErrorNorm(phantom, reconstructed);

    logPerformance("openmp", geom, numIterations, totalReconstructTime, errorNorm, basePath + "logs/performance_log.csv");
}

