//
//  Utilities.cpp
//  MetalCalculations
//
//  Created by Kate Suraev on 4/9/2025.
//
#include "../include/Utilities.hpp"

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

    // Read rows array
    matrix.rows.resize(header.num_rows + 1);
    inFile.read(reinterpret_cast<char*>(matrix.rows.data()), matrix.rows.size() * sizeof(int));

    // Read cols array
    matrix.cols.resize(header.num_non_zero);
    inFile.read(reinterpret_cast<char*>(matrix.cols.data()), matrix.cols.size() * sizeof(int));

    // Read vals array
    matrix.vals.resize(header.num_non_zero);
    inFile.read(reinterpret_cast<char*>(matrix.vals.data()), matrix.vals.size() * sizeof(float));

    inFile.close();
    return true;
}

std::vector<float> loadPhantom(const char* filename, const Geometry& geom) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open phantom file " << filename << std::endl;
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
        std::cerr << "Error: Phantom size mismatch. Expected " << expectedSize << " values, but file contains " << phantomData.size() << " values." << std::endl;
        exit(-1);
    }
    return phantomData;
}

std::vector<float> loadPhantomBinary(const char* filename, const Geometry& geom) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open phantom file " << filename << std::endl;
        exit(-1);
    }

    size_t expectedSize = geom.imageWidth * geom.imageHeight;
    std::vector<float> phantomData(expectedSize);

    file.read(reinterpret_cast<char*>(phantomData.data()), expectedSize * sizeof(float));
    if (!file) {
        std::cerr << "Error: Failed to read phantom file " << filename << std::endl;
        exit(-1);
    }

    return phantomData;
}

bool loadSinogram(const std::string& filename, std::vector<float>& sinogram, uint numRays) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;
    sinogram.resize(numRays);
    in.read(reinterpret_cast<char*>(sinogram.data()), numRays * sizeof(float));
    if (!in) return false;
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

void saveSinogram(const std::string& filename, MTL::Buffer* sinogramBuffer, uint numRays) {
    // Calculate the total number of float elements
    size_t size = numRays;
    size_t total_bytes = size * sizeof(float);

    // Get a pointer to the raw data in the Metal buffer
    void* buffer_contents = sinogramBuffer->contents();

    // Open a file stream for writing in binary mode
    std::ofstream outFile(filename, std::ios::binary);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "' for writing." << std::endl;
        return;
    }

    // Write the raw bytes directly from the Metal buffer's contents to the file
    outFile.write(static_cast<const char*>(buffer_contents), total_bytes);

    outFile.close();
    std::cout << "Sinogram data successfully saved to '" << filename << "'." << std::endl;
}

void saveTextureToFile(const std::string& filename, MTL::Texture* texture) {
    if (!texture) {
        std::cerr << "Error: Cannot save a null texture." << std::endl;
        return;
    }

    long width = texture->width();
    long height = texture->height();
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Texture has invalid dimensions." << std::endl;
        return;
    }

    // Create vector to hold texture data
    std::vector<float> textureVector(width * height);

    // Define the region to read (entire texture)
    MTL::Region region = MTL::Region::Make2D(0, 0, width, height);

    // Read the texture data into the vector
    texture->getBytes(textureVector.data(), width * sizeof(float), region, 0);

    // Open file for binary writing
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "' for writing." << std::endl;
        return;
    }

    outFile.write(reinterpret_cast<const char*>(textureVector.data()),
        textureVector.size() * sizeof(float));

    outFile.close();
    std::cout << "Texture data successfully saved to binary file '" << filename << "'." << std::endl;
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


void logPerformance(
    const Geometry& geom, const int numIterations,
    const std::chrono::duration<double, std::milli>& scanTime,
    const std::chrono::duration<double, std::milli>& projTime,
    const std::chrono::duration<double, std::milli>& reconTime, const std::string filename) {


    std::ofstream logFile;

    // Check if the file already exists 
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
        logFile << "Timestamp,NumIterations,ImageWidth,ImageHeight,NumAngles,NumDetectors,ScanTime_ms,ProjectionTime_ms,ReconstructionTime_ms\n";
    }

    // Get the current system time for the log entry
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_tm = std::localtime(&now_time);

    // Write the new data row to CSV file
    logFile << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") << ","
        << numIterations << "," << geom.imageWidth << ","
        << geom.imageHeight << "," << geom.nAngles << "," << geom.nDetectors
        << "," << scanTime.count() << "," << projTime.count() << ","
        << reconTime.count() << "\n";

    logFile.close();
    std::cout << "Performance metrics logged to " << filename << std::endl;
}

void logPerformance(
    const Geometry& geom, const int numIterations,
    const std::chrono::duration<double, std::milli>& reconTime) {
    const std::string filename = "performance_log.csv";
    std::ofstream logFile;

    // Check if the file already exists to determine if we need to write a
    // header
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
        logFile << "Timestamp,NumIterations,ImageWidth,ImageHeight,NumAngles,"
            "NumDetectors,ReconstructionTime_ms\n";
    }

    // Get the current system time for the log entry
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_tm = std::localtime(&now_time);

    // Write the new data row to the CSV file
    logFile << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") << ","
        << numIterations << "," << geom.imageWidth << ","
        << geom.imageHeight << "," << geom.nAngles << "," << geom.nDetectors
        << "," << reconTime.count() << "\n";

    logFile.close();
    std::cout << "Performance metrics logged to " << filename << std::endl;
}
