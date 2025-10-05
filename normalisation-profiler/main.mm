// This program is used to test the execution time of sinogram normalisation using Metal in isolation.
// Results logged to normalisation_log.csv

#include "../metal-include/NormalisationProfiler.hpp"

constexpr int IMAGE_WIDTH = 4096;
constexpr int IMAGE_HEIGHT = 4096;
constexpr int NUM_ANGLES = 720;

constexpr const char LOG_FILE[] = "/metal-logs/normalisation_log.csv";
constexpr const char SINOGRAM_TEST_FILE[] = "/metal-data/sinogram_4096_720_test.txt";

int main(int argc, char **argv) {
    if (IMAGE_WIDTH != IMAGE_HEIGHT) {
        std::cerr << "Image width and height must be equal." << std::endl;
        return -1;
    }

    try {
        NS::AutoreleasePool *pPool = NS::AutoreleasePool::alloc()->init();

        // Set geometry parameters for scanner
        uint numDetectors = std::ceil(2 * std::sqrt(2) * IMAGE_WIDTH);
        Geometry geom = {IMAGE_WIDTH, IMAGE_HEIGHT, NUM_ANGLES, numDetectors};

        MetalContext context = MetalContext();

        NormalisationProfiler NormalisationProfiler(context, geom);

        // Time sinogram normalisation
        auto time = NormalisationProfiler.normaliseSinogram(std::string(PROJECT_BASE_PATH) + SINOGRAM_TEST_FILE,
                                                            geom.nAngles, geom.nDetectors);

        // Log metrics to CSV file
        auto logFilePath = std::string(PROJECT_BASE_PATH) + LOG_FILE;
        NormalisationProfiler.logPerformance(logFilePath, time);

        pPool->release();
    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    }
    return 0;
}
