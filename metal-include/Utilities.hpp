/**
 * @file Utilities.hpp
 * @brief Utility header file containing helper functions and structures for CT reconstruction.
 * Includes functions for loading and saving data, as well as logging performance metrics.
 */
#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>


 /**
  * @brief Structure to hold geometry parameters for CT reconstruction.
  */
struct Geometry {
    uint imageWidth;
    uint imageHeight;
    uint nAngles;
    uint nDetectors;
};

/**
 * @brief Structure to hold the header information of a sparse matrix.
 */
struct SparseMatrixHeader {
    u_int64_t num_rows;
    u_int64_t num_cols;
    u_int64_t num_non_zero;
};

/**
 * @brief Structure to hold a sparse matrix in CSR format.
 */
struct SparseMatrix {
    std::vector<uint32_t> rows;
    std::vector<uint32_t> cols;
    std::vector<float> vals;
};

/**
 * @brief Load sparse projection matrix from binary file for testing.
 * @param filename The name of the binary file containing the sparse matrix.
 * @param matrix The sparse matrix structure to be filled with loaded data.
 * @param header The header structure to be filled with matrix dimensions.
 * @return SparseMatrix The loaded sparse matrix.
 */
bool loadSparseMatrixBinary(const std::string& filename, SparseMatrix& matrix, SparseMatrixHeader& header);

/**
 * @brief Load phantom data from a text file into a vector, ensuring it matches the expected
 * geometry.
 * @param filename The name of the file containing the phantom data.
 * @param geom The geometry structure containing image dimensions.
 * @return The loaded phantom data as a vector of floats.
 */
std::vector<float> loadPhantom(const std::string& filename, const Geometry& geom);

/**
 * @brief Load colourmap texture from a text file into a vector.
 * @param filename The name of the file containing the colourmap data.
 * @return The loaded colourmap data as a vector of floats.
 * This is used to colour the sinogram, reconstructed image and phantom in window.
 */
std::vector<float> loadColourMapTexture(const std::string& filename);

/**
 * @brief Load sinogram data from a binary file into a vector.
 * @param filename The name of the binary file containing the sinogram data.
 * @param sinogram The vector to be filled with the loaded sinogram data.
 * @param numRays The number of rays (size of the sinogram).
 * @return True if the sinogram was successfully loaded, false otherwise.
 */
bool loadSinogram(const std::string& filename, std::vector<float>& sinogram, uint numRays);

/**
 * @brief Flip image data vertically.
 * @param original_data The original image data as a vector of floats.
 * @param width The width of the image.
 * @param height The height of the image.
 * @return A new vector containing the vertically flipped image data.
 */
std::vector<float> flipImageVertically(const std::vector<float>& originalData, int width, int height);

/**
 * @brief Log the performance of the CT reconstruction process to a CSV file.
 * @param geom The geometry structure containing geometry parameters.
 * @param numIterations The number of iterations used in the reconstruction.
 * @param projTime The time taken for the projection step in milliseconds.
 * @param scanTime The time taken for the scan step in milliseconds.
 * @param reconTime The time taken for the reconstruction step in milliseconds.
 */
void logPerformance(
    const Geometry& geom, const int numIterations,
    const double scanTime,
    const double projTime,
    const double reconTime,
    const double finalErrorNorm,
    const std::string& filename);

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

#endif  // UTILS_HPP
