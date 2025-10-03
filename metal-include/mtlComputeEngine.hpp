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
#include "MetalContext.hpp"

class MTLComputeEngine {
public:
  /**
   * @brief Metal compute engine constructor.
   * @param context The Metal context containing the device, command queue, and library.
   * @param geom The geometry structure containing geometry parameters.
   */
  MTLComputeEngine(MetalContext& context, const Geometry& geom);

  /**
   * @brief Metal compute engine destructor.
   * Cleans up Metal resources.
   */
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
  void computeSinogram(std::vector<float>& phantomData);

  /**
   * @brief Reconstruct the image using Cimmino's algorithm implemented in Metal.
   * @param numIterations The number of iterations to perform.
   * @param finalUpdateNorm Reference to store the norm of the final update for convergence analysis.
   * @return The time taken for the reconstruction in milliseconds.
   */
  std::chrono::duration<double, std::milli> reconstructImage(int numIterations, double& finalUpdateNorm);

  // Getters to access textures from render engine
  MTL::Texture* getReconstructedTexture() const { return reconstructedTexture; }
  MTL::Texture* getSinogramTexture() const { return sinogramTexture; }
  MTL::Texture* getOriginalPhantomTexture() const { return originalPhantomTexture; }

private:
  /* METAL RESOURCES */
  MTL::Device* device;
  MTL::Library* defaultLibrary;
  MTL::CommandQueue* commandQueue;

  /* BUFFERS */
  MTL::Buffer* offsetsBuffer;
  MTL::Buffer* colsBuffer;
  MTL::Buffer* valsBuffer;
  MTL::Buffer* sinogramBuffer;
  MTL::Buffer* reconstructedBuffer;
  MTL::Buffer* phantomBuffer;

  /* TEXTURES */
  MTL::Texture* reconstructedTexture;
  MTL::Texture* sinogramTexture;
  MTL::Texture* originalPhantomTexture;


  Geometry geom;
  uint totalRays;
  size_t totalNonZeroElements;
  float totalWeightSum;
  double phantomNorm;
  std::string basePath = PROJECT_BASE_PATH;

  /**
   * @brief Create a compute pipeline state for a given kernel function.
   * @param function The kernel function to create the pipeline for.
   * @return The created compute pipeline state.
   */
  MTL::ComputePipelineState* createComputePipeline(MTL::Function* function);

  /**
   * @brief Create a Metal buffer with specified size, options, and optional initial data.
   * @param size The size of the buffer in bytes.
   * @param options The resource options for the buffer.
   * @param data Optional pointer to initial data to copy into the buffer.
   * @return The created Metal buffer.
   */
  MTL::Buffer* createBuffer(size_t size, MTL::ResourceOptions options, void* data);

  /**
   * @brief Create a Metal texture with specified width, height, pixel format, and usage.
   * @param width The width of the texture.
   * @param height The height of the texture.
   * @param pixelFormat The pixel format of the texture.
   * @param usage The texture usage options.
   * @return The created Metal texture.
   */
  MTL::Texture* createTexture(uint width, uint height, MTL::PixelFormat pixelFormat, MTL::TextureUsage usage);

  /**
   * @brief Calculate grid and thread group sizes and dispatch threads for a compute kernel.
   * @param encoder The compute command encoder.
   * @param pipeline The compute pipeline state.
   * @param totalElements The total number of elements to process.
   */
  void dispatchThreads(MTL::ComputeCommandEncoder* encoder, MTL::ComputePipelineState* pipeline, size_t totalElements);

  /**
   * @brief Copy data from a Metal buffer to a 2D Metal texture.
   * @param cmdBuffer The command buffer to encode the copy operation.
   * @param buffer The Metal buffer containing the data to copy.
   * @param texture The Metal texture to copy the data into.
   * @param width The width of the texture.
   * @param height The height of the texture.
   * Source: https://developer.apple.com/documentation/metal/reading-pixel-data-from-a-drawable-texture
   * Source: https://developer.apple.com/documentation/metal/mtlblitcommandencoder
   */
  void copyBufferToTexture(MTL::CommandBuffer* cmdBuffer, MTL::Buffer* buffer, MTL::Texture* texture, size_t width,
    size_t height);

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
};
#endif  // MTLCOMPUTEENGINE_HPP
