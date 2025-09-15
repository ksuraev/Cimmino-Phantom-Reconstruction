//
//  Utilities.hpp
//  MetalCalculations
//
//  Created by Kate Suraev on 4/9/2025.
//
#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <cstdlib>

// --- Shared Data Structures ---
struct Geometry {
    int imageWidth;
    int imageHeight;
    int nAngles;
    int nDetectors;
};

struct SparseMatrixHeader {
    uint64_t num_rows;
    uint64_t num_cols;
    uint64_t num_non_zero;
};

struct SparseMatrix {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> vals;
};

std::string getBasePath();

bool loadSparseMatrixBinary(const std::string& binFileName, SparseMatrix& matrix, SparseMatrixHeader header, size_t totalRays);

// Load sparse projection matrix from binary file
bool loadSinogram(const std::string& filename, std::vector<float>& sinogram, unsigned int numRays);

// Load phantom from file
std::vector<float> loadPhantom(const char* filename, const Geometry& geom);

// Load sinogram
bool loadSinogram(const std::string& filename, std::vector<float>& sinogram, unsigned int numRays);

void logPerformance(const std::string& executionType,
    const Geometry& geom, const int numIterations,
    const double reconstructionTime, const std::string& filename);

void saveImage(const std::string& filename,
    const std::vector<float>& imageData, unsigned int width,
    unsigned int height);

/**
* @brief Times the execution of a method and returns the duration in microseconds.
* @param methodToTime The method to be timed.
* @return The duration of the method execution in microseconds.
* Inspired by Maksym's code from lecture 14/07/2025
*/
static double timeMethod_ms(const std::function<void()>& methodToTime) {
    auto start = std::chrono::high_resolution_clock::now();
    methodToTime();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}
#endif
