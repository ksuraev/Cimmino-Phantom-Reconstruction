//
//  Utilities.cpp
//  MetalCalculations
//
//  Created by Kate Suraev on 4/9/2025.
//
#include "../include/Utilities.hpp"

std::string getBasePath() {
    const char* envBasePath = std::getenv("PROJECT_BASE_PATH");
    if (envBasePath == nullptr) {
        throw std::runtime_error("Environment variable PROJECT_BASE_PATH is not set.");
    }

    std::string basePath(envBasePath);

    // Ensure basePath ends with a '/'
    if (!basePath.empty() && basePath.back() != '/') {
        basePath += '/';
    }

    return basePath;
}

bool loadSparseMatrixBinary(const std::string& binFileName, SparseMatrix& matrix, SparseMatrixHeader header, size_t totalRays) {
    std::ifstream file(binFileName, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening binary file " << binFileName << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&header), sizeof(SparseMatrixHeader));
    if (!file) {
        std::cerr << "Error reading matrix header from " << binFileName << std::endl;
        return false;
    }

    std::cout << "Loading Matrix Dimensions: " << header.num_rows << "x" << header.num_cols << ", Non-zero elements: " << header.num_non_zero << std::endl;

    matrix.rows.resize(totalRays + 1); // + 1 for CSR format
    matrix.cols.resize(header.num_non_zero);
    matrix.vals.resize(header.num_non_zero);

    file.read(reinterpret_cast<char*>(matrix.rows.data()), (header.num_rows + 1) * sizeof(int));
    if (!file) {
        std::cerr << "Error reading row data from " << binFileName << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(matrix.cols.data()), header.num_non_zero * sizeof(int));
    if (!file) {
        std::cerr << "Error reading column data from " << binFileName << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(matrix.vals.data()), header.num_non_zero * sizeof(float));
    if (!file) {
        std::cerr << "Error reading value data from " << binFileName << std::endl;
        return false;
    }

    std::cout << "Sparse matrix successfully loaded." << std::endl;
    return true;
}

std::vector<float> loadPhantom(const char* filename, const Geometry& geom) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open phantom file " << filename
            << std::endl;
        exit(-1);
    }
    std::vector<float> phantomData;
    float value;
    while (file >> value) {
        phantomData.push_back(value);
    }
    file.close();

    size_t expectedSize = geom.imageWidth * geom.imageHeight;
    if (phantomData.size() != expectedSize) {
        std::cerr << "Error: Phantom size mismatch. Expected " << expectedSize
            << " values, but file contains " << phantomData.size()
            << " values." << std::endl;
        exit(-1);
    }

    return phantomData;
}

bool loadSinogram(const std::string& filename, std::vector<float>& sinogram, unsigned int numRays) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;
    sinogram.resize(numRays);
    in.read(reinterpret_cast<char*>(sinogram.data()), numRays * sizeof(float));
    if (!in) return false;
    return true;
}

std::vector<float> flipImageVertically(const std::vector<float>& originalData, int width, int height) {
    std::vector<float> flippedData(width * height);

    for (int y = 0; y < height; ++y) {
        // Calculate destination row index
        int flippedY = height - 1 - y;

        // Pointers to the start of the source and destination rows
        const float* srcRow = originalData.data() + (y * width);
        float* destRow = flippedData.data() + (flippedY * width);

        // Copy the entire row at once
        memcpy(destRow, srcRow, width * sizeof(float));
    }
    return flippedData;
}

/**
 * @brief Log performance metrics to a CSV file.
 * @param executionType The type of execution (e.g., "Sequential", "OpenMP").
 * @param geom The geometry parameters.
 * @param numIterations The number of iterations performed.
 * @param reconstructionTime The reconstruction time duration.
 */
void logPerformance(const std::string& executionType,
    const Geometry& geom, const int numIterations,
    const double reconstructionTime,
    const double finalErrorNorm,
    const std::string& filename) {
    std::ofstream logFile;

    // Check if the file already exists 
    std::ifstream fileExists(filename);
    bool writeHeader = !fileExists.good();
    fileExists.close();

    // Open the file in append mode
    logFile.open(filename, std::ios::out | std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open log file for writing." << std::endl;
        return;
    }

    if (writeHeader) {
        logFile << "Timestamp,ExecutionType,NumIterations,ImageWidth,ImageHeight,NumAngles,"
            "NumDetectors,ReconstructionTime_ms,FinalErrorNorm\n";
    }

    // Get the current system time for the log entry
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_tm = std::localtime(&now_time);

    // Write data to CSV file
    logFile << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") << "," << executionType << ","
        << numIterations << "," << geom.imageWidth << ","
        << geom.imageHeight << "," << geom.nAngles << "," << geom.nDetectors
        << "," << reconstructionTime << "," << finalErrorNorm << "\n";

    logFile.close();
    std::cout << "Performance metrics logged." << std::endl;
}

void saveImage(const std::string& filename,
    const std::vector<float>& imageData, unsigned int width,
    unsigned int height) {

    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "' for writing." << std::endl;
        return;
    }

    // Check if the vector size matches the provided dimensions
    if (imageData.size() != width * height) {
        std::cerr << "Error: Image data size (" << imageData.size() << ") does not match dimensions (" << width << "x" << height << ")." << std::endl;
        outFile.close();
        return;
    }

    // Write to file
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            unsigned int index = y * width + x;
            outFile << imageData[index] << " ";
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "Image data successfully saved." << std::endl;
}