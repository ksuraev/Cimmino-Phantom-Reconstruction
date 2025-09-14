//
//  Utilities.hpp
//  MetalCalculations
//
//  Created by Kate Suraev on 4/9/2025.
//
#ifndef UTILS_HPP
#define UTILS_HPP

#include <Metal/Metal.hpp>
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
    int imageWidth;
    int imageHeight;
    int nAngles;
    int nDetectors;
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
    std::vector<int> rows;
    std::vector<int> cols;
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
std::vector<float> loadPhantom(const char* filename, const Geometry& geom);

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
 * @brief Save sinogram data from a Metal buffer to a binary file for testing.
 * @param filename The name of the binary file to save the sinogram data to.
 * @param sinogramBuffer The Metal buffer containing the sinogram data.
 * @param numRays The number of rays (size of the sinogram).
 */
void saveSinogram(const std::string& filename, MTL::Buffer* sinogramBuffer, uint numRays);

/**
 * @brief Save a Metal texture to a binary file for testing.
 * @param filename The name of the binary file to save the texture data to.
 * @param texture The Metal texture to be saved.
 */
void saveTextureToFile(const std::string& filename, MTL::Texture* texture);

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
void logPerformance(const Geometry& geom, const int numIterations,
    const std::chrono::duration<double, std::milli>& projTime,
    const std::chrono::duration<double, std::milli>& scanTime,
    const std::chrono::duration<double, std::milli>& reconTime);

/**
 * @brief Log the performance of the image reconstruction to a CSV file.
 * @param geom The geometry structure containing geometry parameters.
 * @param numIterations The number of iterations used in the reconstruction.
 * @param reconTime The time taken for the reconstruction step in milliseconds.
 */
void logPerformance(const Geometry& geom, const int numIterations,
    const std::chrono::duration<double, std::milli>& reconTime);

#endif  // UTILS_HPP
