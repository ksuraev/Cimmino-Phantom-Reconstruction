// Implementation of SinogramNormaliser class for normalising sinogram data using MTLComputeEngine functions
// For testing of normalisation performance

#include "sinogramNormaliser.hpp"

SinogramNormaliser::SinogramNormaliser(MetalContext &context, const Geometry &geom) : MTLComputeEngine(context, geom) {}

double SinogramNormaliser::normaliseSinogram(const std::string &fileName, uint nAngles, uint nDetectors) {
    std::vector<float> sinogramData;

    // Load sinogram data
    if (!loadSinogram(fileName, sinogramData, nAngles * nDetectors)) {
        std::cerr << "Error loading sinogram from file: " << fileName << std::endl;
        exit(-1);
    }

    // Load sinogram into texture
    MTL::Texture *testSinogramTexture =
        createTexture(nDetectors, nAngles, MTL::PixelFormatR32Float, MTL::TextureUsageShaderRead);
    MTL::Region region = MTL::Region::Make2D(0, 0, nDetectors, nAngles);
    testSinogramTexture->replaceRegion(region, 0, sinogramData.data(), nDetectors * sizeof(float));

    // Time the normalisation process
    auto normTime = timeMethod_ms([&]() {
        float maxVal = 0.0f;
        findMaxValInTexture(testSinogramTexture, maxVal);
        std::cout << "Max value in sinogram: " << maxVal << std::endl;
        normaliseTexture(testSinogramTexture, maxVal);
    });

    std::cout << "Sinogram normalisation time: " << normTime << " ms" << std::endl;

    return normTime;
}