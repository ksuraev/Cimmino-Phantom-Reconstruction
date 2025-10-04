// OpenMP implementation of texture normalisation for normalisation performance testing
// g++-15 -o norm normalisationTimer.cpp ../utilities/Utilities.cpp
#include <omp.h>
#include "../include/Utilities.hpp"

constexpr uint32_t IMAGE_WIDTH = 256;
constexpr uint32_t IMAGE_HEIGHT = 256;
constexpr uint32_t NUM_ANGLES = 360;
constexpr int NUM_THREADS = 8;

constexpr const char* LOG_FILE = "/logs/normalisation_log.csv";
constexpr const char* SINOGRAM_FILE = "/data/sinogram_256_360_test.txt";

/**
 * @brief Find the maximum value in a 2D texture using OpenMP for parallelisation.
 * @param texture The 2D texture (vector of vectors) to search.
 * @return The maximum value found in the texture.
 */
float findMaxValue(const std::vector<std::vector<float>>& texture) {
    float maxVal = 0.0f;
#pragma omp parallel for num_threads(NUM_THREADS) collapse(2) reduction(max:maxVal)
    for (size_t y = 0; y < texture.size(); ++y) {
        for (size_t x = 0; x < texture[0].size(); ++x) {
            maxVal = std::max(maxVal, texture[y][x]);
        }
    }
    std::cout << "Max value found: " << maxVal << std::endl;
    return maxVal;
}

/**
 * @brief Normalise a 2D texture by dividing each element by the maximum value using OpenMP for parallelisation.
 * @param texture The 2D texture (vector of vectors) to normalise.
 * @param maxVal The maximum value used for normalization.
 */
void normaliseTexture(std::vector<std::vector<float>>& texture, float maxVal) {
    if (maxVal <= 0.0f) {
        // If maxVal is 0 or negative, set all values to 0
#pragma omp parallel for  num_threads(NUM_THREADS) collapse(2)
        for (size_t y = 0; y < texture.size(); ++y) {
            for (size_t x = 0; x < texture[0].size(); ++x) {
                texture[y][x] = 0.0f;
            }
        }
    }
    else {
        // Normalize each value by dividing by maxVal
#pragma omp parallel for  num_threads(NUM_THREADS) collapse(2)
        for (size_t y = 0; y < texture.size(); ++y) {
            for (size_t x = 0; x < texture[0].size(); ++x) {
                texture[y][x] /= maxVal;
            }
        }
    }
}

int main(int argc, const char* argv[]) {
    std::string basePath;
    try {
        basePath = getBasePath();
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Set geometry parameters
    auto numDetectors = static_cast<uint32_t>(std::ceil(2 * std::sqrt(2) * IMAGE_WIDTH));
    Geometry geom = { IMAGE_WIDTH,  IMAGE_HEIGHT, NUM_ANGLES, numDetectors };
    size_t totalRays = static_cast<size_t>(geom.nAngles * geom.nDetectors);

    // Load sinogram from file
    std::vector<float> sinogram(totalRays, 0.0f);
    if (!loadSinogram(basePath + SINOGRAM_FILE, sinogram)) {
        std::cerr << "Failed to load sinogram." << std::endl;
        return -1;
    }

    // Convert sinogram to 2D for normalisation
    std::vector<std::vector<float>> sinogram2D(geom.nAngles, std::vector<float>(geom.nDetectors, 0.0f));
    for (size_t a = 0; a < geom.nAngles; ++a) {
        for (size_t d = 0; d < geom.nDetectors; ++d) {
            sinogram2D[a][d] = sinogram[a * geom.nDetectors + d];
        }
    }

    // Time the normalisation process
    auto time = timeMethod_ms([&]() {
        float maxVal = findMaxValue(sinogram2D);
        normaliseTexture(sinogram2D, maxVal);
        });

    std::cout << "Sinogram normalisation time (ms): " << time << std::endl;

    // Log performance metrics to CSV file
    auto logFilePath = basePath + LOG_FILE;
    std::ofstream logFile;

    // Check if the file already exists
    std::ifstream fileExists(logFilePath);
    bool writeHeader = !fileExists.good();
    fileExists.close();

    logFile.open(logFilePath, std::ios::out | std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open log file for writing." << std::endl;
        return 1;
    }
    if (writeHeader) {
        logFile << "Timestamp,ImageWidth,ImageHeight,NumAngles,NumDetectors,NormalisationTime,Type\n";
    }

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_tm = std::localtime(&now_time);

    // Write the new data row to CSV file
    logFile << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") << "," << geom.imageWidth << "," << geom.imageHeight
        << "," << geom.nAngles << "," << geom.nDetectors << "," << time << "," << "OpenMP" << "\n";

    logFile.close();
    std::cout << "Performance metrics logged." << std::endl;
}

