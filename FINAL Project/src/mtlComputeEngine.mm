//
//  MTLComputeEngine.mm
//  RenderPipeline
//
//  Created by Kate Suraev on 5/9/2025.
//

#include "../include/mtlComputeEngine.hpp"
/**
 * @brief Constructor for the MTLComputeEngine class.
 * @param geom The geometry parameters for the CT scan.
 */
MTLComputeEngine::MTLComputeEngine(MetalContext& context, const Geometry& geom) : geom(geom)
{
  // Set total rays - one per angle-detector pair
  totalRays = geom.nAngles * geom.nDetectors;

  // Reserve space for offsets vector
  offsets.resize(totalRays + 1, 0);

  // Initialise device, command queue and default library
  device = context.getDevice();
  commandQueue = context.getCommandQueue();
  defaultLibrary = context.getLibrary();

  // Set geometry buffer
  geomBuffer = device->newBuffer(&geom, sizeof(Geometry),
    MTL::ResourceStorageModeShared);
  if (!geomBuffer) {
    std::cerr << "Error: Failed to create geometry buffer. " << std::endl;
    std::exit(-1);
  }
}

MTLComputeEngine::~MTLComputeEngine() {
  //   geomBuffer->release();
  //   //   renderPipeline->release();
  //   reconstructedTexture->release();
  //   sinogramTexture->release();
  //   originalPhantomTexture->release();
  //   commandQueue->release();
  //   defaultLibrary->release();
  //   //   glfwDestroyWindow(glfwWindow);
  //   //   glfwTerminate();
  //   device->release();
}


/**
 * @brief Create a compute pipeline state for a given kernel function.
 * @param function The kernel function to create the pipeline for.
 * @return The created compute pipeline state.
 */
MTL::ComputePipelineState*
MTLComputeEngine::createComputePipeline(MTL::Function* function) {
  NS::Error* error = nullptr;
  MTL::ComputePipelineState* pipeline =
    device->newComputePipelineState(function, &error);
  if (!pipeline) {
    std::cerr << "Error: Failed to create pipeline state. " << pipeline->label()
      << " " << error->localizedDescription()->utf8String() << std::endl;
    std::exit(-1);
  }
  function->release();
  return pipeline;
}

/**
 * @brief Create a kernel function from the default library.
 * @param functionName The name of the kernel function to create.
 * @return The created kernel function.
 */
MTL::Function* MTLComputeEngine::createKernelFn(const char* functionName) {
  MTL::Function* fn = defaultLibrary->newFunction(NS::String::string(functionName, NS::UTF8StringEncoding));
  if (!fn) {
    std::cerr << "Failed to find kernel " << functionName << " in the library." << std::endl;
    std::exit(-1);
  }

  return fn;
}

/**
 * @brief Calculate grid and thread group sizes and dispatch threads for a
 * compute kernel.
 * @param encoder The compute command encoder.
 * @param pipeline The compute pipeline state.
 * @param totalElements The total number of elements to process.
 */
void dispatchThreads(MTL::ComputeCommandEncoder* encoder, MTL::ComputePipelineState* pipeline, size_t totalElements) {
  // Calculate thread group size
  NS::UInteger threadGroupSize = pipeline->maxTotalThreadsPerThreadgroup();

  // Check if thread group size exceeds total elements
  if (threadGroupSize > totalElements) threadGroupSize = totalElements;

  // Set grid size and dispatch threads
  MTL::Size gridSize = MTL::Size(totalElements, 1, 1);
  encoder->dispatchThreads(gridSize, MTL::Size(threadGroupSize, 1, 1));
}


/**
 * @brief Generate the projection matrix and calculate total row weights sum.
 */
void MTLComputeEngine::generateProjectionMatrix() {
  // Load projection matrix from binary file - generated using Astra Toolbox
  SparseMatrixHeader header;
  SparseMatrix matrix;
  loadSparseMatrixBinary("data/projection_256_astra.bin", matrix, header);
  totalNonZeroElements = header.num_non_zero;

  // Check sizes match expected
  if (matrix.rows.size() != (totalRays + 1)) {
    std::cerr << "Error: Offsets size does not match total rays + 1." << std::endl;
    exit(-1);
  }
  if (matrix.cols.size() != totalNonZeroElements ||
    matrix.vals.size() != totalNonZeroElements) {
    std::cerr
      << "Error: Cols or Vals size does not match total non-zero elements." << std::endl;
    exit(-1);
  }

  // Load into projection matrix CSR buffers
  offsetsBuffer = device->newBuffer(matrix.rows.data(), (matrix.rows.size()) * sizeof(int), MTL::ResourceStorageModeShared);
  colsBuffer = device->newBuffer(matrix.cols.data(), matrix.cols.size() * sizeof(int), MTL::ResourceStorageModeShared);
  valsBuffer = device->newBuffer(matrix.vals.data(), matrix.vals.size() * sizeof(float), MTL::ResourceStorageModeShared);

  // Check the buffers are not null
  if (!offsetsBuffer || !colsBuffer || !valsBuffer) {
    std::cerr << "Error: Failed to create projection matrix buffers. " << std::endl;
    exit(-1);
  }

  // Calculate row weights and total weight sum on cpu
  std::vector<float> rowWeights(totalRays, 0.0f);
  for (uint i = 0; i < totalRays; ++i) {
    double rowNormSq = 0.0;
    for (uint j = matrix.rows[i]; j < matrix.rows[i + 1]; ++j) {
      rowNormSq += static_cast<double>(matrix.vals[j]) * matrix.vals[j];
    }
    // Prevent division by zero
    if (rowNormSq < 1e-9) {
      rowNormSq = 1.0;
    }
    rowWeights[i] = static_cast<float>(rowNormSq);
  }

  // Calculate total weight sum for Cimmino's algorithm
  totalWeightSum = 0.0f;
  for (float weight : rowWeights) {
    totalWeightSum += weight;
  }
  std::cout << "Total weight sum calculated: " << totalWeightSum << std::endl;
}


void MTLComputeEngine::performScan(std::vector<float>& phantomData) {
  std::vector<float> flippedPhantomData = flipImageVertically(phantomData, geom.imageWidth, geom.imageHeight);

  // Initialise texture for phantom
  auto phantomTextureDesc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR32Float, geom.imageWidth, geom.imageHeight, false);
  phantomTextureDesc->setUsage(MTL::TextureUsageShaderRead);
  originalPhantomTexture = device->newTexture(phantomTextureDesc);
  phantomBuffer = device->newBuffer(flippedPhantomData.data(), flippedPhantomData.size() * sizeof(float), MTL::ResourceStorageModeShared);

  // Load original phantom data into texture
  MTL::Region region = MTL::Region::Make2D(0, 0, geom.imageWidth, geom.imageHeight);
  originalPhantomTexture->replaceRegion(region, 0, flippedPhantomData.data(), geom.imageWidth * sizeof(float));
  if (!originalPhantomTexture) {
    std::cerr << "Error: Failed to create phantom texture. " << std::endl;
    exit(-1);
  }

  // Initialise sinogram buffer
  sinogramBuffer = device->newBuffer(totalRays * sizeof(float), MTL::ResourceStorageModeShared);

  if (!sinogramBuffer) {
    std::cerr << "Error: Failed to create sinogram buffer. " << std::endl;
    exit(-1);
  }

  // Load perform scan kernel function
  MTL::Function* performScanFn = createKernelFn("performScan");

  auto cmdBuffer = commandQueue->commandBuffer();
  auto encoder = cmdBuffer->computeCommandEncoder();

  // Create metal scan compute pipeline
  MTL::ComputePipelineState* scanPipeline = createComputePipeline(performScanFn);

  // Compute center of image
  float imgCenterX = static_cast<float>(geom.imageWidth) / 2.0f;
  float imgCenterY = static_cast<float>(geom.imageHeight) / 2.0f;

  // Compute number of steps
  float stepSize = 2.0f;
  int numSteps = static_cast<int>(sqrt(float(geom.imageWidth * geom.imageWidth + geom.imageHeight * geom.imageHeight)) / stepSize);

  // Set arguments for scan kernel
  encoder->setComputePipelineState(scanPipeline);
  encoder->setBuffer(phantomBuffer, 0, 0);
  encoder->setBuffer(sinogramBuffer, 0, 1);
  encoder->setBuffer(geomBuffer, 0, 2);
  encoder->setBytes(&imgCenterX, sizeof(float), 3);
  encoder->setBytes(&imgCenterY, sizeof(float), 4);
  encoder->setBytes(&numSteps, sizeof(int), 5);

  // Perform scan
  dispatchThreads(encoder, scanPipeline, totalRays);
  encoder->endEncoding();

  // Copy sinogram buffer into texture
  auto sinoTextureDesc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR32Float, geom.nDetectors, geom.nAngles, false);
  sinoTextureDesc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
  sinogramTexture = device->newTexture(sinoTextureDesc);
  if (!sinogramTexture) {
    std::cerr << "failed to create sinogram texture" << std::endl;
  }

  // source
  // https://developer.apple.com/documentation/metal/reading-pixel-data-from-a-drawable-texture
  // https://developer.apple.com/documentation/metal/mtlblitcommandencoder
  auto blitEncoder = cmdBuffer->blitCommandEncoder();
  blitEncoder->copyFromBuffer(sinogramBuffer, 0, geom.nDetectors * sizeof(float), 0, MTL::Size(geom.nDetectors, geom.nAngles, 1), sinogramTexture, 0, 0, MTL::Origin(0, 0, 0));

  blitEncoder->endEncoding();

  // Commit and wait once everything is encoded
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();

  // Normalise texture to [0,1] range
  auto startNorm = std::chrono::high_resolution_clock::now();

  findMaxValInTexture(sinogramTexture, maxValSinogramBuffer);
  normaliseTexture(sinogramTexture, maxValSinogramBuffer);

  auto endNorm = std::chrono::high_resolution_clock::now();
  auto normMs = std::chrono::duration<double, std::milli>(endNorm - startNorm);
  std::cout << "Sinogram normalisation time: " << normMs.count() << " ms" << std::endl;

  // If needed, save sinogram to file
  // saveTextureToFile("sinogram_256.bin", sinogramTexture);
}

/**
 * @brief Find the maximum value in a texture using a two-pass reduction
 * approach.
 * @param texture The input texture to find the maximum value in.
 * @param maxValbuffer The output buffer to store the maximum value.
 */
void MTLComputeEngine::findMaxValInTexture(MTL::Texture* texture,
  MTL::Buffer*& maxValbuffer) {
  if (!texture) {
    std::cerr << "Input texture is null." << std::endl;
    exit(-1);
  }

  // Pass 1 pipeline - find max per thread group
  auto maxPerThreadGroupFn = createKernelFn("findMaxPerThreadgroupKernel");
  MTL::ComputePipelineState* findMaxPipeline = createComputePipeline(maxPerThreadGroupFn);

  // Pass 2 pipeline - reduce to find final max
  auto reduceMaxFn = createKernelFn("reduceMaxKernel");
  MTL::ComputePipelineState* reduceMaxPipeline = createComputePipeline(reduceMaxFn);

  auto cmdBuffer = commandQueue->commandBuffer();

  // Pass 1 setup
  MTL::Size gridSize = MTL::Size(texture->width(), texture->height(), 1);

  // Using a fixed threadgroup size of 16x16 = 256 threads
  // The kernel's shared memory array size must match this
  NS::UInteger tgWidth = 16, tgHeight = 16;
  MTL::Size threadgroupSize = MTL::Size(tgWidth, tgHeight, 1);

  // Calculate number of thread groups
  MTL::Size numThreadgroups = MTL::Size((gridSize.width + tgWidth - 1) / tgWidth, (gridSize.height + tgHeight - 1) / tgHeight, 1);
  float numTgTotal = numThreadgroups.width * numThreadgroups.height;

  // Create intermediate map buffer to hold one max value per threadgroup
  MTL::Buffer* mapBuffer = device->newBuffer(numTgTotal * sizeof(float), MTL::ResourceStorageModePrivate);

  // Encode pass 1
  auto p1Encoder = cmdBuffer->computeCommandEncoder();
  p1Encoder->setComputePipelineState(findMaxPipeline);
  p1Encoder->setTexture(sinogramTexture, 0);
  p1Encoder->setBuffer(mapBuffer, 0, 0);
  p1Encoder->setBytes(&numTgTotal, sizeof(float), 1);
  p1Encoder->dispatchThreads(gridSize, threadgroupSize);
  p1Encoder->endEncoding();

  // Pass 2 setup
  maxValbuffer = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

  // Set result to 0 before encoding
  auto blitEncoder = cmdBuffer->blitCommandEncoder();
  blitEncoder->fillBuffer(maxValbuffer, NS::Range(0, sizeof(float)), 0);
  blitEncoder->endEncoding();

  // Encode pass 2
  auto p2Encoder = cmdBuffer->computeCommandEncoder();
  p2Encoder->setComputePipelineState(reduceMaxPipeline);
  p2Encoder->setBuffer(mapBuffer, 0, 0);
  p2Encoder->setBuffer(maxValbuffer, 0, 1);

  // Dispatch one thread for each value in the map buffer
  MTL::Size reduceGridSize = MTL::Size(numTgTotal, 1, 1);
  NS::UInteger reduceTgSize = reduceMaxPipeline->maxTotalThreadsPerThreadgroup();
  if (reduceTgSize > numTgTotal) {
    reduceTgSize = numTgTotal;
  }
  MTL::Size reduceThreadgroupSize = MTL::Size(reduceTgSize, 1, 1);

  p2Encoder->dispatchThreads(reduceGridSize, reduceThreadgroupSize);
  p2Encoder->endEncoding();

  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();
}

/**
 * @brief Normalise a texture by dividing each pixel by the maximum value.
 * @param texture The input texture to be normalised.
 * @param maxValBuffer The buffer containing the maximum value.
 */
void MTLComputeEngine::normaliseTexture(MTL::Texture* texture,
  MTL::Buffer*& maxValBuffer) {
  if (!texture) {
    std::cerr << "Texture to be normalised is null." << std::endl;
    exit(-1);
  }
  if (!maxValBuffer) {
    std::cout << "Max val buffer is null." << std::endl;
    exit(-1);
  }

  // Initialise normalise kernel function
  auto normaliseFn = createKernelFn("normaliseKernel");

  // Create compute pipeline
  MTL::ComputePipelineState* normalisePipeline = createComputePipeline(normaliseFn);

  // Create command buffer and encoder
  auto cmdBuffer = commandQueue->commandBuffer();
  auto encoder = cmdBuffer->computeCommandEncoder();

  // Set pipeline and args
  encoder->setComputePipelineState(normalisePipeline);
  encoder->setTexture(texture, 0);
  encoder->setBuffer(maxValBuffer, 0, 1);

  // Set grid and threadgroup sizes
  MTL::Size textureGridSize = MTL::Size(texture->width(), texture->height(), 1);
  NS::UInteger normTgW = normalisePipeline->threadExecutionWidth();
  NS::UInteger normTgH = normalisePipeline->maxTotalThreadsPerThreadgroup() / normTgW;
  MTL::Size normThreadgroup = MTL::Size(normTgW, normTgH, 1);

  encoder->dispatchThreads(textureGridSize, normThreadgroup);
  encoder->endEncoding();

  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();
}


/**
 * @brief Perform image reconstruction using Cimmino's algorithm.
 * @param numIterations The number of iterations to perform.
 * @return The duration taken for the reconstruction process.
 */
std::chrono::duration<double, std::milli>
MTLComputeEngine::reconstructImage(int numIterations) {

  // Initialise reconstruction and update functions
  auto cimminoFn = createKernelFn("cimminosReconstruction");
  auto applyUpdateFn = createKernelFn("applyUpdate");

  // Create pipelines
  auto cimminoPipeline = createComputePipeline(cimminoFn);
  auto applyUpdatePipeline = createComputePipeline(applyUpdateFn);

  size_t width = geom.imageWidth;
  size_t height = geom.imageHeight;
  uint imageSize = width * height;

  // Create buffers
  reconstructedBuffer = device->newBuffer(imageSize * sizeof(float), MTL::ResourceStorageModeShared);
  auto updateBuffer = device->newBuffer(imageSize * sizeof(float), MTL::ResourceStorageModeShared);
  if (!updateBuffer || !reconstructedBuffer) {
    std::cerr << "Error: Failed to create reconstruction buffers. "
      << std::endl;
    exit(-1);
  }

  // Initialise reconstructed image to zero - check if memset is faster
  MTL::CommandBuffer* cmdBuffer = commandQueue->commandBuffer();
  MTL::BlitCommandEncoder* blit = cmdBuffer->blitCommandEncoder();
  blit->fillBuffer(reconstructedBuffer, NS::Range(0, imageSize * sizeof(float)), 0.0f);
  blit->endEncoding();
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();

  // Start timing reconstruction
  std::cout << "Starting reconstruction for " << numIterations << " iterations..." << std::endl;

  auto startTime = std::chrono::high_resolution_clock::now();
  cmdBuffer = commandQueue->commandBuffer();


  for (int i = 0; i < numIterations; ++i) {
    cmdBuffer = commandQueue->commandBuffer();
    // Clear the update buffer before each iteration
    // MTL::BlitCommandEncoder* blitEncoder = cmdBuffer->blitCommandEncoder();
    // blitEncoder->fillBuffer(updateBuffer,
    //   NS::Range(0, imageSize * sizeof(float)), 0);
    // blitEncoder->endEncoding();
    memset(updateBuffer->contents(), 0, updateBuffer->length());

    // Dispatch the Cimmino reconstruction kernel
    MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(cimminoPipeline);
    encoder->setBuffer(reconstructedBuffer, 0, 0);
    encoder->setBuffer(sinogramBuffer, 0, 1);
    encoder->setBuffer(offsetsBuffer, 0, 2);
    encoder->setBuffer(colsBuffer, 0, 3);
    encoder->setBuffer(valsBuffer, 0, 4);
    encoder->setBytes(&totalWeightSum, sizeof(float), 5);
    encoder->setBytes(&totalRays, sizeof(uint), 6);
    encoder->setBuffer(updateBuffer, 0, 7);
    dispatchThreads(encoder, cimminoPipeline, totalRays);
    encoder->endEncoding();
    cmdBuffer->commit();


    // Dispatch the apply update kernel to get the new image estimate
    cmdBuffer = commandQueue->commandBuffer();
    encoder = cmdBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(applyUpdatePipeline);
    encoder->setBuffer(reconstructedBuffer, 0, 0);
    encoder->setBuffer(updateBuffer, 0, 1);
    encoder->setBytes(&imageSize, sizeof(uint), 2);
    dispatchThreads(encoder, applyUpdatePipeline, imageSize);
    encoder->endEncoding();
    cmdBuffer->commit();

#ifdef DEBUG
    // Every 50 iterations compute the update norm - how much the image has changed
    // If the algorithm is working correctly this should be decreasing
    if ((i + 1) % 50 == 0) {
      std::vector<float> errorData(imageSize, 0.0f);

      // Get the reconstructed image an copy to temp vector
      float* imageData = static_cast<float*>(reconstructedBuffer->contents());
      std::vector<float> tempData(imageData, imageData + imageSize);

      // Transpose the temp matrix in place for correct orientation and comparison
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = y + 1; x < width; ++x) {
          std::swap(tempData[y * width + x], tempData[x * height + y]);
        }
      }

      // Get phantom data for comparison
      float* phantomData = static_cast<float*>(phantomBuffer->contents());

      // Compute error between phantom and reconstruction
      for (int i = 0; i < imageSize; ++i) {
        errorData[i] = (tempData[i] - phantomData[i]) * (tempData[i] - phantomData[i]);
      }

      // Compute residual error
      double sumOfSquares = 0.0;
      for (uint i = 0; i < imageSize; ++i) {
        sumOfSquares += errorData[i];
      }
      double finalUpdateNorm = sqrt(sumOfSquares);

      // Print the result
      std::cout << "Iteration " << (i + 1)
        << ": Update Norm = " << finalUpdateNorm << std::endl;
    }
#endif
  }
  cmdBuffer->waitUntilCompleted();

  // We time reconstruction loop only for direct comparison to sequential program
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> totalReconstructTime = endTime - startTime;

  std::cout << "Reconstruction time for " << numIterations << " iterations " << totalReconstructTime.count() << " ms" << std::endl;

  // Transpose the reconstructed image for correct orientation
  float* bufferData = static_cast<float*>(reconstructedBuffer->contents());

  // Transpose the matrix in place
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = y + 1; x < width; ++x) {
      std::swap(bufferData[y * width + x], bufferData[x * height + y]);
    }
  }

  // Notify device that the buffer has been modified
  reconstructedBuffer->didModifyRange(NS::Range(0, imageSize * sizeof(float)));

  // Create texture for reconstructed image
  cmdBuffer = commandQueue->commandBuffer();
  MTL::BlitCommandEncoder* copyBlit = cmdBuffer->blitCommandEncoder();
  size_t sourceBytesPerRow = width * sizeof(float);
  size_t sourceBytesPerImage = sourceBytesPerRow * height;
  MTL::TextureDescriptor* textureDesc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR32Float, width, height, false);
  textureDesc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
  reconstructedTexture = device->newTexture(textureDesc);

  if (!reconstructedTexture) {
    std::cerr << "Error: Failed to create reconstructed image texture." << std::endl;
    exit(-1);
  }

  // Copy reconstructed buffer into texture using blit encoder
  copyBlit->copyFromBuffer(reconstructedBuffer, 0, sourceBytesPerRow, sourceBytesPerImage, MTL::Size(width, height, 1), reconstructedTexture, 0, 0, MTL::Origin(0, 0, 0));
  copyBlit->endEncoding();
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();

  // findMaxValInTexture(reconstructedTexture, maxValReconBuffer);
  // normaliseTexture(reconstructedTexture, maxValReconBuffer);

  std::cout << "Reconstruction complete. Reconstructed image copied to texture." << std::endl;

  // If needed, uncomment to save reconstructed texture to file
  // saveTextureToFile("reconstructed_256.bin", reconstructedTexture);

  return totalReconstructTime;
}