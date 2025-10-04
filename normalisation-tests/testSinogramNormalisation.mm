// This program is to test the execution time of sinogram normalisation using Metal
// Results logged to normalisation_log.csv

#include "../metal-include/sinogramNormaliser.hpp"

constexpr int IMAGE_WIDTH = 256;
constexpr int IMAGE_HEIGHT = 256;
constexpr int NUM_ANGLES = 180;

constexpr const char LOG_FILE[] = "normalisation_log.csv";
constexpr const char SINOGRAM_TEST_FILE[] = "sinogram_256_180_test.txt";

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

        SinogramNormaliser sinogramNormaliser(context, geom);

        // Test sinogram normalisation
        auto time = sinogramNormaliser.normaliseSinogram(
            std::string(PROJECT_BASE_PATH) + "/metal-data/" + SINOGRAM_TEST_FILE, geom.nAngles, geom.nDetectors);

        // Log metrics to CSV file
        auto logFilePath = std::string(PROJECT_BASE_PATH) + "/metal-logs/" + LOG_FILE;
        std::ofstream logFile;
        std::ifstream fileExists(logFilePath);
        bool writeHeader = !fileExists.good();
        fileExists.close();

        // Open the file in append mode, so we don't overwrite previous results
        logFile.open(logFilePath, std::ios::out | std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "Error: Could not open log file for writing." << std::endl;
            return 1;
        }

        if (writeHeader) {
            logFile << "Timestamp,ImageWidth,ImageHeight,NumAngles,NumDetectors,NormalisationTime,Type\n";
        }

        // Get the current system time for the log entry
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm *local_tm = std::localtime(&now_time);

        // Write the new data row to CSV file
        logFile << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") << "," << geom.imageWidth << "," << geom.imageHeight
                << "," << geom.nAngles << "," << geom.nDetectors << "," << time << "," << "Metal" << "\n";

        logFile.close();
        std::cout << "Performance metrics logged." << std::endl;

        pPool->release();
    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    }
    return 0;
}
