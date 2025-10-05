/**
 * @file mtlComputeEngine.hpp
 * @brief Header file for the MTLComputeEngine class, which handles CT scan simulation and image
 * reconstruction using Metal.
 */

#ifndef MTLCOMPUTEENGINE_HPP
#define MTLCOMPUTEENGINE_HPP
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "Utilities.hpp"
#include "MetalUtilities.hpp"
#include "MetalContext.hpp"
#include <bit>

class MTLComputeEngine {
public:
  /**
   * @brief Metal compute engine constructor.
   * @param context The Metal context containing the device, command queue, and library.
   * @param geom The geometry structure containing geometry parameters.
   */
  MTLComputeEngine(MetalContext& context, const Geometry& geom);

  ~MTLComputeEngine();

  /**
   * @brief Generate the projection matrix, calculate the total row weight sum,
   * and upload the matrix data to Metal buffers.
   */
  void loadProjectionMatrix(const std::string& projectionFileName);

  /**
   * @brief Perform the CT scan simulation to generate the sinogram from the phantom data.
   * @param phantomData The input phantom image data as a flat vector.
   */
  void computeSinogram(std::vector<float>& phantomData, double& scanTimeMs);

  /**
   * @brief Reconstruct the image using Cimmino's algorithm implemented in Metal.
   * @param numIterations The number of iterations to perform.
   * @param relativeErrorNorm Reference to store the norm of the final update for convergence analysis.
   * @return The time taken for the reconstruction in milliseconds.
   */
  double reconstructImage(int numIterations, double& relativeErrorNorm, const double relativeErrorThreshold = 1e-2, const int errorCheckInterval = 50);

  // Getters to access textures from render engine
  MTL::Texture* getReconstructedTexture() const { return reconstructedTexture; }
  MTL::Texture* getSinogramTexture() const { return sinogramTexture; }
  MTL::Texture* getOriginalPhantomTexture() const { return originalPhantomTexture; }

protected:
  MTL::Device* device;
  MTL::Library* defaultLibrary;
  MTL::CommandQueue* commandQueue;

  MTL::Buffer* offsetsBuffer;
  MTL::Buffer* colsBuffer;
  MTL::Buffer* valsBuffer;
  MTL::Buffer* sinogramBuffer;
  MTL::Buffer* reconstructedBuffer;
  MTL::Buffer* phantomBuffer;

  MetalUtilities* metalUtils;
  Geometry geom;

  /**
   * @brief Initialise the phantom data by flipping it vertically and calculating its norm.
   * @param phantomData The input phantom image data as a flat vector.
   */
  void initialisePhantom(std::vector<float>& phantomData);

  /**
   * @brief Find the maximum value in a Metal texture and store it in a buffer.
   * @param texture The Metal texture to search for the maximum value.
   * @param maxValBuffer The Metal buffer to store the maximum value found.
   */
  void findMaxValInTexture(MTL::Texture* texture, float& maxValue);

  /**
   * @brief Normalise a Metal texture using the maximum value stored in a buffer.
   * @param texture The Metal texture to be normalised.
   * @param maxValBuffer The Metal buffer containing the maximum value for normalisation.
   */
  void normaliseTexture(MTL::Texture* texture, float maxValue);

private:
  MTL::Texture* reconstructedTexture;
  MTL::Texture* sinogramTexture;
  MTL::Texture* originalPhantomTexture;

  uint totalRays;
  size_t totalNonZeroElements;
  float totalWeightSum;
  double phantomNorm;
};
#endif  // MTLCOMPUTEENGINE_HPP
