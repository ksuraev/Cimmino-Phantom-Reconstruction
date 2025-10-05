/**
 * @file Utilities.cpp
 * @brief Utility functions for loading/saving data and logging performance metrics.
 */
#include "Utilities.hpp"

bool loadSparseMatrixBinary(const std::string& filename, SparseMatrix& matrix, SparseMatrixHeader& header) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // Read matrix dimensions
    inFile.read(reinterpret_cast<char*>(&header.num_rows), sizeof(u_int64_t));
    inFile.read(reinterpret_cast<char*>(&header.num_cols), sizeof(u_int64_t));
    inFile.read(reinterpret_cast<char*>(&header.num_non_zero), sizeof(u_int64_t));

    std::cout << "Sparse Projection Matrix dimensions: " << header.num_rows << "x" << header.num_cols << ", Non-zero elements: " << header.num_non_zero << std::endl;

    // Validate header values
    if (header.num_rows <= 0 || header.num_cols <= 0 || header.num_non_zero < 0) {
        std::cerr << "Error: Invalid header values." << std::endl;
        return false;
    }

    // Read rows, cols and vals
    matrix.rows.resize(header.num_rows + 1);
    inFile.read(reinterpret_cast<char*>(matrix.rows.data()), matrix.rows.size() * sizeof(int));
    matrix.cols.resize(header.num_non_zero);
    inFile.read(reinterpret_cast<char*>(matrix.cols.data()), matrix.cols.size() * sizeof(int));
    matrix.vals.resize(header.num_non_zero);
    inFile.read(reinterpret_cast<char*>(matrix.vals.data()), matrix.vals.size() * sizeof(float));

    inFile.close();
    return true;
}

std::vector<float> loadPhantom(const std::string& filename, const Geometry& geom) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open phantom file " + filename);

    std::vector<float> phantomData;
    float value;
    while (file >> value) {
        phantomData.push_back(value);
    }
    file.close();

    size_t expectedSize = geom.imageWidth * geom.imageHeight;
    if (phantomData.size() != expectedSize) throw std::runtime_error("Phantom size mismatch. Expected " + std::to_string(expectedSize) + " values, but file contains " + std::to_string(phantomData.size()) + " values.");
    return phantomData;
}

bool loadSinogram(const std::string& filename, std::vector<float>& sinogram, uint numRays) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    sinogram.clear();
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        float value;
        while (iss >> value) {
            sinogram.push_back(value);
        }
    }

    if (sinogram.size() != numRays) {
        std::cerr << "Error: Expected " << numRays << " rays, but loaded " << sinogram.size() << " values." << std::endl;
        return false;
    }

    return true;
}

std::vector<float> loadColourMapTexture(const std::string& filename) {
    std::vector<float> colourMapData;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return colourMapData;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        float r, g, b;
        if (!(ss >> r >> g >> b))
            continue;
        colourMapData.push_back(r);
        colourMapData.push_back(g);
        colourMapData.push_back(b);
        colourMapData.push_back(1.0f);
    }
    return colourMapData;
}

std::vector<float> flipImageVertically(const std::vector<float>& originalData, int width, int height) {
    std::vector<float> flippedData(originalData);
    for (size_t y = 0; y < height / 2; ++y) {
        for (size_t x = 0; x < width; ++x) {
            std::swap(flippedData[y * width + x], flippedData[(height - 1 - y) * width + x]);
        }
    }
    return flippedData;
}

void logPerformance(
    const Geometry& geom, const int numIterations,
    const double scanTime,
    const double projTime,
    const double reconTime,
    const double finalErrorNorm,
    const std::string& filename) {
    std::ofstream logFile;

    std::ifstream fileExists(filename);
    bool writeHeader = !fileExists.good();
    fileExists.close();

    // Open the file in append mode, so we don't overwrite previous results
    logFile.open(filename, std::ios::out | std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open log file for writing." << std::endl;
        return;
    }

    if (writeHeader) {
        logFile << "Timestamp,ExecutionType,NumIterations,ImageWidth,ImageHeight,NumAngles,NumDetectors,ScanTime_ms,ProjectionTime_ms,ReconstructionTime_ms,FinalErrorNorm\n";
    }

    // Get the current system time for the log entry
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_tm = std::localtime(&now_time);

    // Write the new data row to CSV file
    logFile << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") << "," << "Metal" << ","
        << numIterations << "," << geom.imageWidth << ","
        << geom.imageHeight << "," << geom.nAngles << "," << geom.nDetectors
        << "," << scanTime << "," << projTime << ","
        << reconTime << "," << finalErrorNorm << "\n";

    logFile.close();
    std::cout << "Performance metrics logged." << std::endl;
}
