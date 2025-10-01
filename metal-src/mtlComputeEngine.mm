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

    // Set geometry buffer
    geomBuffer = createBuffer(sizeof(Geometry), MTL::ResourceStorageModeShared, (void *)&geom);
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

void MTLComputeEngine::generateProjectionMatrix(const std::string &projectionFileName) {
    // Load projection matrix from binary file - generated using Astra Toolbox
    SparseMatrixHeader header;
    SparseMatrix matrix;
    loadSparseMatrixBinary(basePath + "/metal-data/" + projectionFileName, matrix, header);

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

void MTLComputeEngine::performScan(std::vector<float> &phantomData) {
    // Flip phantom data vertically for correct orientation
    std::vector<float> flippedPhantomData = flipImageVertically(phantomData, geom.imageWidth, geom.imageHeight);

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

    sinogramBuffer = createBuffer(totalRays * sizeof(float), MTL::ResourceStorageModeShared);

    // Load perform scan kernel function
    // MTL::Function *performScanFn = createKernelFn("performScan", defaultLibrary);
    MTL::Function *computeSinogram = createKernelFn("computeSinogram", defaultLibrary);

    auto cmdBuffer = commandQueue->commandBuffer();
    auto encoder = cmdBuffer->computeCommandEncoder();

    // Create metal scan compute pipeline
    MTL::ComputePipelineState *scanPipeline = createComputePipeline(computeSinogram);

    // Set arguments for scan kernel
    encoder->setBuffer(phantomBuffer, 0, 0);
    encoder->setBuffer(offsetsBuffer, 0, 1);
    encoder->setBuffer(colsBuffer, 0, 2);
    encoder->setBuffer(valsBuffer, 0, 3);
    encoder->setBuffer(sinogramBuffer, 0, 4);
    encoder->setBytes(&totalRays, sizeof(uint), 5);
    encoder->setComputePipelineState(scanPipeline);
    // encoder->setBuffer(phantomBuffer, 0, 0);
    // encoder->setBuffer(sinogramBuffer, 0, 1);
    // encoder->setBuffer(geomBuffer, 0, 2);
    // encoder->setBytes(&imgCenterX, sizeof(float), 3);
    // encoder->setBytes(&imgCenterY, sizeof(float), 4);
    // encoder->setBytes(&numSteps, sizeof(int), 5);

    // Perform scan
    dispatchThreads(encoder, scanPipeline, totalRays);
    encoder->endEncoding();

    // Copy sinogram buffer into texture
    sinogramTexture = createTexture(geom.nDetectors, geom.nAngles, MTL::PixelFormatR32Float,
                                    MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);

    // source
    // https://developer.apple.com/documentation/metal/reading-pixel-data-from-a-drawable-texture
    // source
    // https://developer.apple.com/documentation/metal/mtlblitcommandencoder
    auto blitEncoder = cmdBuffer->blitCommandEncoder();
    blitEncoder->copyFromBuffer(sinogramBuffer, 0, geom.nDetectors * sizeof(float), 0,
                                MTL::Size(geom.nDetectors, geom.nAngles, 1), sinogramTexture, 0, 0,
                                MTL::Origin(0, 0, 0));
    blitEncoder->endEncoding();

    // Commit and wait only once everything is encoded
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    // Normalise sinogram texture
    findMaxValInTexture(sinogramTexture, maxValSinogramBuffer);
    normaliseTexture(sinogramTexture, maxValSinogramBuffer);

    /* Uncomment to save non-normalised sinogram buffer to .txt file */
    // std::string filePath = basePath + "/metal-data/sinogram_" + std::to_string(geom.imageWidth) + ".txt";
    // saveTextureToFile(filePath, sinogramTexture);

    /* Uncomment to save normalised sinogram texture to .txt file */
    // std::string normalisedFilePath =
    //     basePath + "/metal-data/sinogram_" + std::to_string(geom.imageWidth) + "_normalised.txt";
    // saveTextureToFile(normalisedFilePath, sinogramTexture);
}

void MTLComputeEngine::findMaxValInTexture(MTL::Texture *texture, MTL::Buffer *&maxValbuffer) {
    if (!texture) {
        std::cerr << "Input texture is null." << std::endl;
        exit(-1);
    }

    // Pass 1 pipeline - find max per thread group
    auto maxPerThreadGroupFn = createKernelFn("findMaxPerThreadgroupKernel", defaultLibrary);
    MTL::ComputePipelineState *findMaxPipeline = createComputePipeline(maxPerThreadGroupFn);

    // Pass 2 pipeline - reduce to find final max
    auto reduceMaxFn = createKernelFn("reduceMaxKernel", defaultLibrary);
    MTL::ComputePipelineState *reduceMaxPipeline = createComputePipeline(reduceMaxFn);

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
    MTL::Buffer *mapBuffer = createBuffer(numTgTotal * sizeof(float), MTL::ResourceStorageModePrivate);

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

void MTLComputeEngine::normaliseTexture(MTL::Texture *texture, MTL::Buffer *&maxValBuffer) {
    if (!texture) {
        std::cerr << "Texture to be normalised is null." << std::endl;
        exit(-1);
    }
    if (!maxValBuffer) {
        std::cout << "Max val buffer is null." << std::endl;
        exit(-1);
    }

    // Initialise normalise kernel
    auto normaliseFn = createKernelFn("normaliseKernel", defaultLibrary);
    MTL::ComputePipelineState *normalisePipeline = createComputePipeline(normaliseFn);

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

std::chrono::duration<double, std::milli> MTLComputeEngine::reconstructImage(int numIterations,
                                                                             double &relativeErrorNorm) {
    // Initialise reconstruction and update functions
    auto cimminoFn = createKernelFn("cimminosReconstruction", defaultLibrary);
    auto applyUpdateFn = createKernelFn("applyUpdate", defaultLibrary);

    auto cimminoPipeline = createComputePipeline(cimminoFn);
    auto applyUpdatePipeline = createComputePipeline(applyUpdateFn);

    // auto computeDifferencePipeline = createComputePipeline(createKernelFn("computeDifference", defaultLibrary));
    size_t width = geom.imageWidth;
    size_t height = geom.imageHeight;
    uint imageSize = width * height;

    // Create buffers
    reconstructedBuffer = createBuffer(imageSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto updateBuffer = createBuffer(imageSize * sizeof(float), MTL::ResourceStorageModeShared);

    auto differenceSumBuffer = createBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    auto phantomNormBuffer = createBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    // Initialise reconstructed image to zero - check if memset is faster
    MTL::CommandBuffer *cmdBuffer = commandQueue->commandBuffer();
    MTL::BlitCommandEncoder *blit = cmdBuffer->blitCommandEncoder();
    blit->fillBuffer(reconstructedBuffer, NS::Range(0, imageSize * sizeof(float)), 0.0f);
    blit->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    std::cout << "Starting reconstruction for " << numIterations << " iterations..." << std::endl;

    // Start timing reconstruction
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        // Reset command buffer for each iteration
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
        encoder->setBuffer(phantomBuffer, 0, 2);
        encoder->setBuffer(differenceSumBuffer, 0, 3);
        encoder->setBuffer(phantomNormBuffer, 0, 4);
        encoder->setBytes(&imageSize, sizeof(uint), 5);
        dispatchThreads(encoder, applyUpdatePipeline, imageSize);
        encoder->endEncoding();

        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();

        // Check for convergence every 50 iterations
        if ((i + 1) % 50 == 0) {
            double *differenceSum = static_cast<double *>(differenceSumBuffer->contents());
            double *phantomNorm = static_cast<double *>(phantomNormBuffer->contents());

            double differenceNorm = sqrt(*differenceSum);
            double phantomNormValue = sqrt(*phantomNorm);

            relativeErrorNorm = differenceNorm / phantomNormValue;

            // Check convergence
            if (relativeErrorNorm < 1e-2) {
                std::cout << "Converged with relative error norm: " << relativeErrorNorm << std::endl;
                break;
            }

            // Reset the buffers for the next iteration
            memset(differenceSumBuffer->contents(), 0, sizeof(float));
            memset(phantomNormBuffer->contents(), 0, sizeof(float));
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> totalReconstructTime = endTime - startTime;

    std::cout << "Reconstruction time for " << numIterations << " iterations " << totalReconstructTime.count() << " ms"
              << std::endl;

    // Create texture for reconstructed image
    reconstructedTexture = createTexture(width, height, MTL::PixelFormatR32Float, MTL::TextureUsageShaderRead);

    // Copy reconstructed buffer into texture using blit encoder
    cmdBuffer = commandQueue->commandBuffer();
    MTL::BlitCommandEncoder *copyBlit = cmdBuffer->blitCommandEncoder();
    copyBlit->copyFromBuffer(reconstructedBuffer, 0, width * sizeof(float), width * sizeof(float) * height,
                             MTL::Size(width, height, 1), reconstructedTexture, 0, 0, MTL::Origin(0, 0, 0));

    copyBlit->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    std::cout << "Reconstruction complete. Reconstructed image copied to texture." << std::endl;

    // Save reconstructed texture to file
    std::string imageFileName = basePath + "/metal-data/metal_" + std::to_string(numIterations) + "_" +
                                std::to_string(geom.imageWidth) + ".txt";
    saveTextureToFile(imageFileName, reconstructedTexture);

    return totalReconstructTime;
}
