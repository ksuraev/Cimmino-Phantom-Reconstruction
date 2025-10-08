// Implementation of AlgorithmTester class for testing reconstruction and sinogram computation algorithms.

#include "AlgorithmTester.hpp"
#include "MetalUtilities.hpp"

AlgorithmTester::AlgorithmTester(MetalContext &context, const Geometry &geom) : MTLComputeEngine(context, geom) {}

void AlgorithmTester::generateTestProjector(Geometry &geom) {
    // Load projection matrix for small test case
    loadProjectionMatrix("/metal-data/projection_" + std::to_string(geom.imageWidth) + ".bin");
}

void AlgorithmTester::generateTestPhantom(uint size) {
    // Create phantom with all 1s
    std::vector<float> phantomData(size * size, 1.0f);

    // Load phantom into buffer, compute phantom norm
    initialisePhantom(phantomData);
}

void AlgorithmTester::generateTestSinogram(uint angles, uint detectors) {
    // Create sinogram with all 1s
    std::vector<float> sinogramData(angles * detectors, 1.0f);

    // Load sinogram into buffer
    sinogramBuffer = metalUtils->createBuffer(angles * detectors * sizeof(float), MTL::ResourceStorageModeShared, sinogramData.data());
}

void AlgorithmTester::testReconstruction(Geometry &geom, uint numIterations, double &finalErrorNorm) {
    generateTestProjector(geom);
    generateTestPhantom(geom.imageWidth);
    generateTestSinogram(geom.nAngles, geom.nDetectors);

    // Run reconstruction
    reconstructImage(numIterations, finalErrorNorm, 350.0f);

    std::cout << "Final error norm: " << finalErrorNorm << std::endl;

    // Check accuracy of reconstruction against known solution
    if (!checkReconstructionAccuracy(numIterations)) {
        std::cerr << "WARNING: Reconstructed image does not match the expected solution." << std::endl;
    }
}

void AlgorithmTester::testSinogramComputation(Geometry &geom) {
    generateTestProjector(geom);

    // Create phantom with all 1s
    std::vector<float> phantomData(geom.imageWidth * geom.imageWidth, 1.0f);

    double scanTime = 0.0;
    computeSinogram(phantomData, scanTime);

    // Check accuracy of sinogram against known solution
    if (!checkSinogramAccuracy(geom.imageWidth)) {
        std::cerr << "WARNING: Computed sinogram does not match the expected solution." << std::endl;
    }
}

bool AlgorithmTester::checkReconstructionAccuracy(uint numIterations) {
    // Load solution (computed in R code)
    std::vector<float> solutionData =
        loadPhantom(std::string(PROJECT_BASE_PATH) + "/metal-data/solution_" + std::to_string(numIterations) + ".txt", geom);

    // Get reconstructed image data from buffer
    std::vector<float> reconstructedData(geom.imageWidth * geom.imageHeight);
    std::memcpy(reconstructedData.data(), reconstructedBuffer->contents(), geom.imageWidth * geom.imageHeight * sizeof(float));

    // Compare reconstructed data to solution with small tolerance for floating point errors
    for (size_t i = 0; i < solutionData.size(); ++i) {
        if (std::abs(solutionData[i] - reconstructedData[i]) > 1e-6) {
            std::cerr << "WARNING: Discrepancy found at index " << i << ": solution=" << solutionData[i]
                      << ", reconstructed=" << reconstructedData[i] << std::endl;
            return false;
        }
    }
    std::cout << "Reconstructed image matches the solution within tolerance." << std::endl;
    return true;
}

bool AlgorithmTester::checkSinogramAccuracy(uint imageSize) {
    // Load solution (computed in R code)
    std::vector<float> solutionData;
    loadSinogram(std::string(PROJECT_BASE_PATH) + "/metal-data/sino_sol_" + std::to_string(imageSize) + ".txt", solutionData,
                 geom.nAngles * geom.nDetectors);

    // Get computed sinogram data from buffer
    std::vector<float> computedData(geom.nAngles * geom.nDetectors);
    std::memcpy(computedData.data(), sinogramBuffer->contents(), geom.nAngles * geom.nDetectors * sizeof(float));

    // Compare computed data to solution with small tolerance for floating point errors
    for (size_t i = 0; i < solutionData.size(); ++i) {
        if (std::abs(solutionData[i] - computedData[i]) > 1e-4) {
            std::cerr << "WARNING: Discrepancy found at index " << i << ": solution=" << solutionData[i] << ", computed=" << computedData[i]
                      << std::endl;
            return false;
        }
    }
    std::cout << "Computed sinogram matches the solution within tolerance." << std::endl;
    return true;
}