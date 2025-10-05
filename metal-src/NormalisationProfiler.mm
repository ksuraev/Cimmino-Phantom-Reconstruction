// Implementation of NormalisationProfiler class for normalising sinogram data using MTLComputeEngine functions
// For testing of normalisation performance

#include "NormalisationProfiler.hpp"
#include "../metal-include/MetalUtilities.hpp"

NormalisationProfiler::NormalisationProfiler(MetalContext &context, const Geometry &geom)
    : MTLComputeEngine(context, geom) {}

double NormalisationProfiler::normaliseSinogram(const std::string &fileName, uint nAngles, uint nDetectors) {
    std::vector<float> sinogramData;

    // Load sinogram data
    if (!loadSinogram(fileName, sinogramData, nAngles * nDetectors)) {
        std::cerr << "Error loading sinogram from file: " << fileName << std::endl;
        exit(-1);
    }

    // Load sinogram into texture
    MTL::Texture *sinogramTexture =
        metalUtils->createTexture(nDetectors, nAngles, MTL::PixelFormatR32Float, MTL::TextureUsageShaderRead);
    MTL::Region region = MTL::Region::Make2D(0, 0, nDetectors, nAngles);
    sinogramTexture->replaceRegion(region, 0, sinogramData.data(), nDetectors * sizeof(float));

    // Time the normalisation process
    auto normalisationTime = timeMethod_ms([&]() {
        float maxVal = 0.0f;
        findMaxValInTexture(sinogramTexture, maxVal);
        normaliseTexture(sinogramTexture, maxVal);
        std::cout << "Max value in sinogram: " << maxVal << std::endl;
    });

    std::cout << "Sinogram normalisation time: " << normalisationTime << " ms" << std::endl;

    return normalisationTime;
}

void NormalisationProfiler::logPerformance(const std::string &logFilePath, double normalisationTime) {
    std::ofstream logFile;
    std::ifstream fileExists(logFilePath);
    bool writeHeader = !fileExists.good();
    fileExists.close();

    // Open the file in append mode, so we don't overwrite previous results
    logFile.open(logFilePath, std::ios::out | std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open log file for writing." << std::endl;
        return;
    }

    if (writeHeader) {
        logFile << "Timestamp,ImageWidth,ImageHeight,NumAngles,NumDetectors,NormalisationTime,Type\n";
    }

    // Get the current system time for the log entry
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm *local_tm = std::localtime(&now_time);

    // Write the new data row to CSV file
    logFile << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") << "," << geom.imageWidth << "," << geom.imageHeight << ","
            << geom.nAngles << "," << geom.nDetectors << "," << normalisationTime << "," << "Metal" << "\n";

    logFile.close();
    std::cout << "Performance metrics logged." << std::endl;
}