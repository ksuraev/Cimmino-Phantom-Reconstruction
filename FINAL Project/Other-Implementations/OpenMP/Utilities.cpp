//
//  Utilities.cpp
//  MetalCalculations
//
//  Created by Kate Suraev on 4/9/2025.
//
#include "../include/Utilities.hpp"

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

    std::cout << "Reading " << header.num_rows + 1 << " row indices (" << (totalRays + 1) * sizeof(int) << " bytes)" << std::endl;
    std::cout << "Reading " << header.num_non_zero << " column indices (" << header.num_non_zero * sizeof(int) << " bytes)" << std::endl;
    std::cout << "Reading " << header.num_non_zero << " values (" << header.num_non_zero * sizeof(float) << " bytes)" << std::endl;

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

    std::cout << "Sparse matrix successfully loaded from '" << binFileName << "'." << std::endl;
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

/**
 * @brief Log performance metrics to a CSV file.
 * @param executionType The type of execution (e.g., "Sequential", "OpenMP").
 * @param geom The geometry parameters.
 * @param numIterations The number of iterations performed.
 * @param reconstructionTime The reconstruction time duration.
 */
void logPerformance(const std::string& executionType,
    const Geometry& geom, const int numIterations,
    const double reconstructionTime, const std::string filename) {
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
            "NumDetectors,ReconstructionTime_ms\n";
    }

    // Get the current system time for the log entry
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_tm = std::localtime(&now_time);

    // Write data to CSV file
    logFile << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") << "," << executionType << ","
        << numIterations << "," << geom.imageWidth << ","
        << geom.imageHeight << "," << geom.nAngles << "," << geom.nDetectors
        << "," << reconstructionTime << "\n";

    logFile.close();
    std::cout << "Performance metrics logged to " << filename << std::endl;
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
    std::cout << "Image data successfully saved to '" << filename << "'." << std::endl;
}

