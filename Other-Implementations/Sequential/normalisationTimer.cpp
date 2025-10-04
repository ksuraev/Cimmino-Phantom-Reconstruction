// clang++ -o norm normalisationTimer.cpp ../utilities/Utilities.cpp
#include "../include/Utilities.hpp"

constexpr uint32_t IMAGE_WIDTH = 4096;
constexpr uint32_t IMAGE_HEIGHT = 4096;
constexpr uint32_t NUM_ANGLES = 720;

constexpr const char* LOG_FILE = "/logs/normalisation_log.csv";
constexpr const char* SINOGRAM_FILE = "/data/sinogram_4096_720_test.txt";


// Equivalent to metal kernel findMaxValInTexture
float findMaxValue(const std::vector<std::vector<float>>& texture) {
    float maxVal = 0.0f;

    for (size_t y = 0; y < texture.size(); ++y) {
        for (size_t x = 0; x < texture[y].size(); ++x) {
            maxVal = std::max(maxVal, texture[y][x]);
        }
    }

    return maxVal;
}

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

    auto time = timeMethod_ms([&]() {
        // Find max value in sinogram
        float maxVal = findMaxValue(sinogram2D);
        // Normalise sinogram
        normaliseTexture(sinogram2D, maxVal);
        });
    std::cout << "Sinogram normalisation time (ms): " << time << std::endl;

    // Save normalised sinogram to file
    std::string sinogramSaveFileName = basePath + "data/sinogram_seq_" + std::to_string(IMAGE_WIDTH) + ".txt";

    // Convert back to 1D for saving
    std::vector<float> normalisedSinogram;
    for (const auto& row : sinogram2D) {
        normalisedSinogram.insert(normalisedSinogram.end(), row.begin(), row.end());
    }
    saveImage(sinogramSaveFileName, normalisedSinogram, geom.nDetectors, geom.nAngles);

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
        logFile << "Timestamp,ImageWidth,ImageHeight,NumAngles,NumDetectors,NormalisationTime\n";
    }

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_tm = std::localtime(&now_time);

    // Write the new data row to CSV file
    logFile << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") << "," << geom.imageWidth << "," << geom.imageHeight
        << "," << geom.nAngles << "," << geom.nDetectors << "," << time << "," << "Sequential" << "\n";

    logFile.close();
    std::cout << "Performance metrics logged." << std::endl;
}

