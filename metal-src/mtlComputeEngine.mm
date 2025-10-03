/**
 * @file mtlComputeEngine.mm
 * @brief Implementation of the MTLComputeEngine class for Metal-based CT
 * reconstruction.
 *
 * This class handles the generation of the projection matrix,
 * performing the scan to create a sinogram, and reconstructing the image using
 * the Cimmino's algorithm. It utilises Metal compute shaders for parallel
 * processing. See mtlComputeEngine.hpp for class definition. See
 * metal-shaders/kernels.metal for Metal shader implementations.
 */

#include "../metal-include/mtlComputeEngine.hpp"

MTLComputeEngine::MTLComputeEngine(MetalContext &context, const Geometry &geom) : geom(geom) {
    // Set total rays - one per angle-detector pair
    totalRays = geom.nAngles * geom.nDetectors;

    // Initialise metal context
    device = context.getDevice();
    commandQueue = context.getCommandQueue();
    defaultLibrary = context.getLibrary();
}

MTLComputeEngine::~MTLComputeEngine() {}

MTL::ComputePipelineState *MTLComputeEngine::createComputePipeline(MTL::Function *function) {
    NS::Error *error = nullptr;
    MTL::ComputePipelineState *pipeline = device->newComputePipelineState(function, &error);
    if (!pipeline) {
        std::cerr << "Error: Failed to create pipeline state. " << pipeline->label() << " "
                  << error->localizedDescription()->utf8String() << std::endl;
        std::exit(-1);
    }
    function->release();
    return pipeline;
}

MTL::Buffer *MTLComputeEngine::createBuffer(size_t size, MTL::ResourceOptions options, void *data = nullptr) {
    MTL::Buffer *buffer = nullptr;
    if (data) {
        buffer = device->newBuffer(data, size, options);
    } else {
        buffer = device->newBuffer(size, options);
    }
    if (!buffer) {
        std::cerr << "Error: Failed to create buffer of size " << size << std::endl;
        std::exit(-1);
    }
    return buffer;
}

MTL::Texture *MTLComputeEngine::createTexture(uint width, uint height, MTL::PixelFormat pixelFormat,
                                              MTL::TextureUsage usage) {
    // Create a texture descriptor
    MTL::TextureDescriptor *textureDesc =
        MTL::TextureDescriptor::texture2DDescriptor(pixelFormat, width, height, false);
    textureDesc->setUsage(usage);

    // Create the texture
    MTL::Texture *texture = device->newTexture(textureDesc);
    if (!texture) {
        std::cerr << "Error: Failed to create texture of size " << width << "x" << height << std::endl;
        exit(-1);
    }

    return texture;
}

void MTLComputeEngine::dispatchThreads(MTL::ComputeCommandEncoder *encoder, MTL::ComputePipelineState *pipeline,
                                       size_t totalElements) {
    // Calculate thread group size
    NS::UInteger threadGroupSize = pipeline->maxTotalThreadsPerThreadgroup();

    // Check if thread group size exceeds total elements
    if (threadGroupSize > totalElements) threadGroupSize = totalElements;

    // Set grid size and dispatch threads
    MTL::Size gridSize = MTL::Size(totalElements, 1, 1);
    encoder->dispatchThreads(gridSize, MTL::Size(threadGroupSize, 1, 1));
}

void MTLComputeEngine::copyBufferToTexture(MTL::CommandBuffer *cmdBuffer, MTL::Buffer *buffer, MTL::Texture *texture,
                                           size_t width, size_t height) {
    auto blitEncoder = cmdBuffer->blitCommandEncoder();
    blitEncoder->copyFromBuffer(buffer, 0, width * sizeof(float), 0, MTL::Size(width, height, 1), texture, 0, 0,
                                MTL::Origin(0, 0, 0));
    blitEncoder->endEncoding();
}

void MTLComputeEngine::loadProjectionMatrix(const std::string &projectionFileName) {
    // Load projection matrix from binary file - generated using ASTRA Toolbox
    SparseMatrixHeader header;
    SparseMatrix matrix;
    loadSparseMatrixBinary(std::string(PROJECT_BASE_PATH) + "/metal-data/" + projectionFileName, matrix, header);

    totalNonZeroElements = header.num_non_zero;

    // Check sizes match expected
    if (matrix.rows.size() != (totalRays + 1)) {
        std::cerr << "Error: Offsets size does not match total rays + 1." << std::endl;
        exit(-1);
    }
    if (matrix.cols.size() != totalNonZeroElements || matrix.vals.size() != totalNonZeroElements) {
        std::cerr << "Error: Cols or Vals size does not match total non-zero elements." << std::endl;
        exit(-1);
    }

    // Load into projection matrix CSR metal buffers
    offsetsBuffer =
        createBuffer((matrix.rows.size()) * sizeof(int), MTL::ResourceStorageModeShared, matrix.rows.data());
    colsBuffer = createBuffer(matrix.cols.size() * sizeof(int), MTL::ResourceStorageModeShared, matrix.cols.data());
    valsBuffer = createBuffer(matrix.vals.size() * sizeof(float), MTL::ResourceStorageModeShared, matrix.vals.data());

    // Calculate total row weight sum for Cimmino's algorithm
    totalWeightSum = 0.0f;
    for (size_t i = 0; i < totalRays; ++i) {
        double rowNormSq = 0.0;
        for (size_t j = matrix.rows[i]; j < matrix.rows[i + 1]; ++j) {
            rowNormSq += static_cast<double>(matrix.vals[j]) * matrix.vals[j];
        }
        totalWeightSum += static_cast<float>(rowNormSq);
    }
}

void MTLComputeEngine::initialisePhantom(std::vector<float> &phantomData) {
    // Flip phantom data vertically for correct orientation
    std::vector<float> flippedPhantomData = flipImageVertically(phantomData, geom.imageWidth, geom.imageHeight);

    // Precompute phantom norm for convergence checking
    float phantomNormSum = 0.0f;
    for (const auto &val : flippedPhantomData) {
        phantomNormSum += val * val;
    }
    phantomNorm = (double)sqrt(phantomNormSum);

    // Create texture and buffer for phantom
    originalPhantomTexture =
        createTexture(geom.imageWidth, geom.imageHeight, MTL::PixelFormatR32Float, MTL::TextureUsageShaderRead);
    phantomBuffer = createBuffer(flippedPhantomData.size() * sizeof(float), MTL::ResourceStorageModeShared,
                                 flippedPhantomData.data());

    // Load original phantom data into texture
    MTL::Region region = MTL::Region::Make2D(0, 0, geom.imageWidth, geom.imageHeight);
    originalPhantomTexture->replaceRegion(region, 0, flippedPhantomData.data(), geom.imageWidth * sizeof(float));
    if (!originalPhantomTexture) {
        std::cerr << "Error: Failed to create phantom texture. " << std::endl;
        exit(-1);
    }
}

void MTLComputeEngine::computeSinogram(std::vector<float> &phantomData) {
    // Initialise phantom
    initialisePhantom(phantomData);

    sinogramBuffer = createBuffer(totalRays * sizeof(float), MTL::ResourceStorageModeShared);

    MTL::Function *computeSinogram = createKernelFn("computeSinogram", defaultLibrary);
    MTL::ComputePipelineState *scanPipeline = createComputePipeline(computeSinogram);

    // Encode and dispatch compute kernel
    auto cmdBuffer = commandQueue->commandBuffer();
    auto encoder = cmdBuffer->computeCommandEncoder();
    l encoder->setBuffer(phantomBuffer, 0, 0);
    encoder->setBuffer(offsetsBuffer, 0, 1);
    encoder->setBuffer(colsBuffer, 0, 2);
    encoder->setBuffer(valsBuffer, 0, 3);
    encoder->setBuffer(sinogramBuffer, 0, 4);
    encoder->setBytes(&totalRays, sizeof(uint), 5);
    encoder->setComputePipelineState(scanPipeline);

    dispatchThreads(encoder, scanPipeline, totalRays);
    encoder->endEncoding();

    // Copy sinogram buffer into texture
    sinogramTexture = createTexture(geom.nDetectors, geom.nAngles, MTL::PixelFormatR32Float,
                                    MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);

    copyBufferToTexture(cmdBuffer, sinogramBuffer, sinogramTexture, geom.nDetectors, geom.nAngles);

    // Commit and wait only once everything is encoded
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    /* Uncomment to save non-normalised sinogram buffer to .txt file */
    // std::string filePath =
    //     std::string(PROJECT_BASE_PATH) + "/metal-data/sinogram_" + std::to_string(geom.imageWidth) + ".txt";
    // saveTextureToFile(filePath, sinogramTexture);

    // Normalise sinogram texture
    float maxSinogramValue = 0.0f;
    findMaxValInTexture(sinogramTexture, maxSinogramValue);
    normaliseTexture(sinogramTexture, maxSinogramValue);

    /* Uncomment to save normalised sinogram texture to .txt file */
    // std::string normalisedFilePath =
    //     std::string(PROJECT_BASE_PATH) + "/metal-data/sinogram_" + std::to_string(geom.imageWidth) +
    //     "_norm.txt";
    // saveTextureToFile(normalisedFilePath, sinogramTexture);
}

void MTLComputeEngine::findMaxValInTexture(MTL::Texture *texture, float &maxValue) {
    if (!texture) {
        std::cerr << "Input texture is null." << std::endl;
        exit(-1);
    }

    // Pass 1 - find max per thread group
    auto maxPerThreadGroupFn = createKernelFn("findMaxPerThreadgroupKernel", defaultLibrary);
    MTL::ComputePipelineState *findMaxPipeline = createComputePipeline(maxPerThreadGroupFn);

    auto cmdBuffer = commandQueue->commandBuffer();

    // Using a fixed threadgroup size of 16x16 = 256 threads
    // The kernel's shared memory array size must match this
    MTL::Size gridSize = MTL::Size(texture->width(), texture->height(), 1);
    NS::UInteger tgWidth = 16, tgHeight = 16;
    MTL::Size threadgroupSize = MTL::Size(tgWidth, tgHeight, 1);

    // Calculate number of thread groups
    MTL::Size numThreadgroups =
        MTL::Size((gridSize.width + tgWidth - 1) / tgWidth, (gridSize.height + tgHeight - 1) / tgHeight, 1);
    float numTgTotal = numThreadgroups.width * numThreadgroups.height;

    // Create intermediate map buffer to hold one max value per threadgroup
    MTL::Buffer *maxValuesBuffer = createBuffer(numTgTotal * sizeof(float), MTL::ResourceStorageModePrivate);

    // Encode and dispatch pass 1
    auto p1Encoder = cmdBuffer->computeCommandEncoder();
    p1Encoder->setComputePipelineState(findMaxPipeline);
    p1Encoder->setTexture(sinogramTexture, 0);
    p1Encoder->setBuffer(maxValuesBuffer, 0, 0);
    p1Encoder->setBytes(&numTgTotal, sizeof(float), 1);
    p1Encoder->dispatchThreads(gridSize, threadgroupSize);
    p1Encoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    // Pass 2: Read back max values buffer to CPU and find max value
    float *maxValues = static_cast<float *>(maxValuesBuffer->contents());
    size_t numElements = numTgTotal;

    float maxVal = maxValues[0];
    for (size_t i = 1; i < numElements; ++i) {
        if (maxValues[i] > maxVal) {
            maxVal = maxValues[i];
        }
    }
    maxValue = maxVal;
}

void MTLComputeEngine::normaliseTexture(MTL::Texture *texture, float maxValue) {
    if (!texture) {
        std::cerr << "Texture to be normalised is null." << std::endl;
        exit(-1);
    }
    if (maxValue == 0.0f) {
        std::cerr << "Max value is zero, texture will not be normalised." << std::endl;
        return;
    }

    // Initialise normalise kernel
    auto normaliseFn = createKernelFn("normaliseKernel", defaultLibrary);
    MTL::ComputePipelineState *normalisePipeline = createComputePipeline(normaliseFn);

    // Encode normalise kernel
    auto cmdBuffer = commandQueue->commandBuffer();
    auto encoder = cmdBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(normalisePipeline);
    encoder->setTexture(texture, 0);
    encoder->setBytes(&maxValue, sizeof(float), 0);

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

std::chrono::duration<double, std::milli> MTLComputeEngine::reconstructImage(int numIterations,
                                                                             double &relativeErrorNorm) {
    auto startTimeTotal = std::chrono::high_resolution_clock::now();

    // Initialise kernel functions and pipelines
    auto cimminoFn = createKernelFn("cimminosReconstruction", defaultLibrary);
    auto applyUpdateFn = createKernelFn("applyUpdate", defaultLibrary);
    auto computeDifferenceFn = createKernelFn("computeRelativeDifference", defaultLibrary);
    auto cimminoPipeline = createComputePipeline(cimminoFn);
    auto applyUpdatePipeline = createComputePipeline(applyUpdateFn);
    auto computeDifferencePipeline = createComputePipeline(computeDifferenceFn);

    size_t width = geom.imageWidth;
    size_t height = geom.imageHeight;
    uint imageSize = width * height;

    // Create buffers
    reconstructedBuffer = createBuffer(imageSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto updateBuffer = createBuffer(imageSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto differenceSumBuffer = createBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    // Initialise reconstructed image to zero
    memset(reconstructedBuffer->contents(), 0, reconstructedBuffer->length());
    auto cmdBuffer = commandQueue->commandBuffer();

    std::cout << "Starting reconstruction for " << numIterations << " iterations..." << std::endl;

    // Start timing reconstruction
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        cmdBuffer = commandQueue->commandBuffer();

        // Reset update buffer to zero
        memset(updateBuffer->contents(), 0, updateBuffer->length());

        // Dispatch the Cimmino reconstruction kernel
        MTL::ComputeCommandEncoder *encoder = cmdBuffer->computeCommandEncoder();
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

        // Dispatch the apply update kernel to get the new image estimate and calculate norms
        encoder = cmdBuffer->computeCommandEncoder();
        encoder->setComputePipelineState(applyUpdatePipeline);
        encoder->setBuffer(reconstructedBuffer, 0, 0);
        encoder->setBuffer(updateBuffer, 0, 1);
        encoder->setBytes(&imageSize, sizeof(uint), 5);
        dispatchThreads(encoder, applyUpdatePipeline, imageSize);
        encoder->endEncoding();

        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();

        // Check for convergence every 50 iterations
        if ((i + 1) % 1 == 0) {
            // Compute relative error norm
            cmdBuffer = commandQueue->commandBuffer();
            encoder = cmdBuffer->computeCommandEncoder();

            encoder->setComputePipelineState(computeDifferencePipeline);
            encoder->setBuffer(reconstructedBuffer, 0, 0);
            encoder->setBuffer(phantomBuffer, 0, 1);
            encoder->setBuffer(differenceSumBuffer, 0, 2);
            dispatchThreads(encoder, computeDifferencePipeline, imageSize);
            encoder->endEncoding();
            cmdBuffer->commit();
            cmdBuffer->waitUntilCompleted();

            // Read back difference sum and compute relative error norm
            float *differenceSum = static_cast<float *>(differenceSumBuffer->contents());
            relativeErrorNorm = sqrt((double)(*differenceSum)) / phantomNorm;

            if (relativeErrorNorm < 1e-2) {
                std::cout << "Converged after " << (i + 1) << " iterations with relative error norm "
                          << relativeErrorNorm << std::endl;
                break;
            }
            memset(differenceSumBuffer->contents(), 0, sizeof(float));
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> totalReconstructTime = endTime - startTime;

    std::cout << "Reconstruction time for " << numIterations << " iterations " << totalReconstructTime.count() << " ms"
              << std::endl;

    // Copy reconstructed buffer into texture using blit encoder
    reconstructedTexture = createTexture(width, height, MTL::PixelFormatR32Float, MTL::TextureUsageShaderRead);
    cmdBuffer = commandQueue->commandBuffer();
    copyBufferToTexture(cmdBuffer, reconstructedBuffer, reconstructedTexture, width, height);

    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    auto endTimeTotal = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> totalTime = endTimeTotal - startTimeTotal;
    std::cout << "Total time including data transfers: " << totalTime.count() << " ms" << std::endl;

    // Save reconstructed texture to file
    std::string imageFileName = basePath + "/metal-data/metal_" + std::to_string(numIterations) + "_" +
                                std::to_string(geom.imageWidth) + ".txt";
    saveTextureToFile(imageFileName, reconstructedTexture);

    std::cout << "Reconstruction complete. Reconstructed image saved." << std::endl;

    return totalReconstructTime;
}
