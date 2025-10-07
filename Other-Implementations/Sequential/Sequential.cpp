/**
 * @file Sequential.cpp
 * @brief Sequential implementation of Cimmino's algorithm for CT image reconstruction.
 */

#include "../include/Utilities.hpp"

constexpr uint32_t IMAGE_WIDTH = 256;
constexpr uint32_t IMAGE_HEIGHT = 256;
constexpr uint32_t NUM_ANGLES = 360;

constexpr const char* LOG_FILE = "logs/performance_log.csv";
constexpr const char* PROJECTION_FILE = "data/projection_256.bin";
constexpr const char* PHANTOM_FILE = "data/phantom_256.txt";

constexpr float RELAXATION_FACTOR = 350.0f;

/**
 * @brief Find the maximum value in a 2D texture.
 * @param texture The 2D texture represented as a vector of vectors.
 * @return The maximum value found in the texture.
 */
float findMaxValue(const std::vector<std::vector<float>>& texture) {
    float maxVal = 0.0f;

    for (size_t y = 0; y < texture.size(); ++y) {
        for (size_t x = 0; x < texture[y].size(); ++x) {
            maxVal = std::max(maxVal, texture[y][x]);
        }
    }

    return maxVal;
}

/**
 * @brief Normalise a 2D texture by dividing each element by the maximum value.
 * If the maximum value is zero or negative, all elements are set to zero.
 * @param texture The 2D texture represented as a vector of vectors (modified in place).
 * @param maxVal The maximum value used for normalisation.
 */
void normaliseTexture(std::vector<std::vector<float>>& texture, float maxVal) {
    if (maxVal <= 0.0f) {
        // If maxVal is 0 or negative, set all values to 0
        for (size_t y = 0; y < texture.size(); ++y) {
            for (size_t x = 0; x < texture[y].size(); ++x) {
                texture[y][x] = 0.0f;
            }
        }
    }
    else {
        // Normalize each value by dividing by maxVal
        for (size_t y = 0; y < texture.size(); ++y) {
            for (size_t x = 0; x < texture[y].size(); ++x) {
                texture[y][x] /= maxVal;
            }
        }
    }
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

    // Initialize sinogram to zero
    std::fill(sinogram.begin(), sinogram.end(), 0.0f);

    // Compute sinogram by multiplying projector with phantom data
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
 * @brief Normalise each row of the sparse projection matrix to have unit L2 norm.
 * @param projector The sparse projection matrix in CSR format (modified in place).
 * @param totalRays The total number of rays (rows in the sinogram).
 */
void normaliseProjectionMatrix(SparseMatrix& projector, size_t totalRays, float& totalWeightSum) {
    totalWeightSum = 0.0F;
    for (size_t i = 0; i < totalRays; ++i) {
        double rowNormSq = 0.0;

        // Compute the squared norm of the row
        for (size_t j = projector.rows[i]; j < projector.rows[i + 1]; ++j) {
            rowNormSq += static_cast<double>(projector.vals[j] * projector.vals[j]);
        }

        float rowNorm = static_cast<float>(sqrt(rowNormSq));
        if (rowNorm > 0.0F) {
            // Normalise the row and accumulate the normalised weight sum
            for (size_t j = projector.rows[i]; j < projector.rows[i + 1]; ++j) {
                projector.vals[j] /= rowNorm;
            }
            totalWeightSum += 1.0F;  // Each normalised row has unit norm
        }
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

    for (int i = 0; i < maxIterations; ++i) {

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

            float scalar = RELAXATION_FACTOR * (2.0f / totalWeightSum) * residual;
            int rowStart = projector.rows[r];
            int rowEnd = projector.rows[r + 1];

            for (size_t i = rowStart; i < rowEnd; ++i) {
                int index = projector.cols[i];
                float weight = projector.vals[i];
                float contribution = scalar * weight;
                if (reconstructedVector[index] + contribution < 0.0f) {
                    reconstructedVector[index] = 0.0f;
                }
                else {
                    reconstructedVector[index] += contribution;
                }
            }
        }

        // Check for convergence every 50 iterations
        if ((i + 1) % 50 == 0) {
            relativeErrorNorm = calculateErrorNorm(phantom, reconstructedVector, phantomNorm);
            if (relativeErrorNorm < 1e-2) {
                std::cout << "Converged after " << (i + 1) << " iterations with relative error norm: " << relativeErrorNorm << std::endl;
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
    if (!loadSparseMatrixBinary(basePath + PROJECTION_FILE, projector, header, totalRays)) {
        std::cerr << "Failed to load sparse projection matrix." << std::endl;
        return -1;
    }
    float totalWeightSum = 0.0f;
    normaliseProjectionMatrix(projector, totalRays, totalWeightSum);

    // Load phantom from file
    std::vector<float> phantom = loadPhantom((basePath + PHANTOM_FILE).c_str(), geom);
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

    // Load sinogram from file
    std::vector<float> sinogram(totalRays, 0.0f);
    auto scanTime = timeMethod_ms([&]() {
        computeSinogram(phantom, projector, totalRays, sinogram);
        });
    std::cout << "Sinogram computation time (ms): " << scanTime << std::endl;

    // Precompute phantom norm for error calculation
    double phantomNorm = 0.0;
    precomputePhantomNorm(phantom, phantomNorm);


    // Reconstruct image and time execution
    std::vector<float> reconstructedImage(IMAGE_WIDTH * IMAGE_HEIGHT, 0.0f);
    double relativeErrorNorm = 0.0;
    auto totalReconstructTime = timeMethod_ms([&]() {
        cimminoReconstruct(numIterations, projector, header, reconstructedImage, phantom, totalRays, sinogram, totalWeightSum, phantomNorm, relativeErrorNorm);
        });

    std::cout << "Total reconstruction time (ms): " << totalReconstructTime << " with relative error norm: " << relativeErrorNorm << std::endl;

    // Save reconstructed image to file
    std::string imageSaveFileName = basePath + "data/image_seq_" + std::to_string(numIterations) + ".txt";
    saveImage(imageSaveFileName, reconstructedImage, geom.imageWidth, geom.imageHeight);

    // Log performance
    logPerformance("Sequential", geom, numIterations, totalReconstructTime, relativeErrorNorm, scanTime, basePath + LOG_FILE);
}

