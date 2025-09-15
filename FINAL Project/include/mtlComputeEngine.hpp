//
//  mtlEngine.hpp
//  RenderPipeline
//
//  Created by Kate Suraev on 5/9/2025.
//

#ifndef MTLCOMPUTEENGINE_HPP
#define MTLCOMPUTEENGINE_HPP
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "Utilities.hpp"
#include "MetalContext.hpp"

class MTLComputeEngine {
public:
  MTLComputeEngine(MetalContext& context, const Geometry& geom);
  ~MTLComputeEngine();

  void init();
  void generateProjectionMatrix();
  void performScan(std::vector<float>& phantomData);
  std::chrono::duration<double, std::milli> reconstructImage(int numIterations, double& finalUpdateNorm);

  // Getters for device and command queue
  MTL::Texture* getReconstructedTexture() const { return reconstructedTexture; }
  MTL::Texture* getSinogramTexture() const { return sinogramTexture; }
  MTL::Texture* getOriginalPhantomTexture() const { return originalPhantomTexture; }
private:
  MTL::Device* device;
  MTL::Library* defaultLibrary;
  MTL::CommandQueue* commandQueue;

  /* BUFFERS */
  MTL::Buffer* geomBuffer;
  MTL::Buffer* rowsBuffer;
  MTL::Buffer* colsBuffer;
  MTL::Buffer* valsBuffer;
  MTL::Buffer* rowNormsSqBuffer;
  MTL::Buffer* offsetsBuffer;
  MTL::Buffer* sinogramBuffer;
  MTL::Buffer* maxValSinogramBuffer;
  MTL::Buffer* maxValReconBuffer;
  MTL::Buffer* reconstructedBuffer;
  MTL::Buffer* phantomBuffer;

  /* TEXTURES */
  // MTL::Texture* viridisTexture;
  MTL::Texture* reconstructedTexture;
  MTL::Texture* sinogramTexture;
  MTL::Texture* originalPhantomTexture;

  // State
  Geometry geom;
  uint totalRays;
  size_t totalNonZeroElements;
  std::vector<int> offsets;
  float totalWeightSum;
  MTL::ComputePipelineState* createComputePipeline(MTL::Function* function);
  MTL::Function* createKernelFn(const char* functionName);
  void findMaxValInTexture(MTL::Texture* texture, MTL::Buffer*& maxValbuffer);
  void normaliseTexture(MTL::Texture* texture,
    MTL::Buffer*& maxValBuffer);
};
#endif  // MTLCOMPUTEENGINE_HPP
