#include "AlgorithmTester.hpp"
#include <string>
#include "MetalUtilities.hpp"

AlgorithmTester::AlgorithmTester(MetalContext &context, const Geometry &geom) : MTLComputeEngine(context, geom) {}

void AlgorithmTester::generateTestProjector(uint angles, uint detectors, uint imageSize) {
    // Load projection matrix for small test case
    loadProjectionMatrix("/metal-data/projection_32.bin");
}
void AlgorithmTester::generateTestPhantom(uint size) {
    // Create phantom with all 1s
    std::vector<float> phantomData(size * size, 1.0f);

    initialisePhantom(phantomData);

    // // Load phantom into buffer
    // phantomBuffer = metalUtils->createBuffer(size * size * sizeof(float), MTL::ResourceStorageModeShared, phantomData.data());
}

void AlgorithmTester::generateTestSinogram(uint angles, uint detectors) {
    // Create sinogram with all 1s
    std::vector<float> sinogramData(angles * detectors, 1.0f);

    // Load sinogram into buffer
    sinogramBuffer = metalUtils->createBuffer(angles * detectors * sizeof(float), MTL::ResourceStorageModeShared, sinogramData.data());
}

void AlgorithmTester::testReconstruction(Geometry &geom, int numIterations, double &finalErrorNorm) {
    generateTestProjector(geom.nAngles, geom.nDetectors, geom.imageWidth);
    generateTestPhantom(geom.imageWidth);
    generateTestSinogram(geom.nAngles, geom.nDetectors);

    // Run reconstruction
    reconstructImage(numIterations, finalErrorNorm);

    std::cout << "Final error norm: " << finalErrorNorm << std::endl;

    // Check accuracy of reconstruction against known solution
    if (!checkAccuracy(numIterations)) {
        std::cerr << "Reconstructed image does not match the expected solution." << std::endl;
    }
}

bool AlgorithmTester::checkAccuracy(int numIterations) {
    // Load solution (computed in R code)
    std::vector<float> solutionData =
        loadPhantom(std::string(PROJECT_BASE_PATH) + "/metal-data/solution_" + std::to_string(numIterations) + ".txt", geom);

    // Get reconstructed image data from buffer
    std::vector<float> reconstructedData(geom.imageWidth * geom.imageHeight);
    std::memcpy(reconstructedData.data(), reconstructedBuffer->contents(), geom.imageWidth * geom.imageHeight * sizeof(float));

    // Compare reconstructed data to solution with small tolerance for floating point errors
    for (size_t i = 0; i < solutionData.size(); ++i) {
        if (std::abs(solutionData[i] - reconstructedData[i]) > 1e-6) {
            std::cerr << "Discrepancy found at index " << i << ": solution=" << solutionData[i]
                      << ", reconstructed=" << reconstructedData[i] << std::endl;
            return false;
        }
    }
    std::cout << "Reconstructed image matches the solution within tolerance." << std::endl;
    return true;
}