/**
 * @file MetalComputeEngine.mm
 * @brief Implementation of the MTLComputeEngine class for Metal-based CT
 * reconstruction.
 *
 * This class handles the generation of the projection matrix,
 * performing the scan to create a sinogram, and reconstructing the image using
 * the Cimmino's algorithm. It utilises Metal compute shaders for parallel
 * processing. See MetalComputeEngine.hpp for class definition. See
 * metal-shaders/kernels.metal for Metal shader implementations.
 */

#include "MetalComputeEngine.hpp"
#include "MetalUtilities.hpp"

MTLComputeEngine::MTLComputeEngine(MetalContext &context, const Geometry &geom) : geom(geom), totalRays(geom.nAngles * geom.nDetectors) {
    // Initialise metal context
    device = context.getDevice();
    commandQueue = context.getCommandQueue();
    defaultLibrary = context.getLibrary();

    metalUtils = new MetalUtilities(device, defaultLibrary, commandQueue);
}

MTLComputeEngine::~MTLComputeEngine() { delete metalUtils; }

void MTLComputeEngine::loadProjectionMatrix(const std::string &projectionFileName) {
    // Load sparse projection matrix from binary file - generated using ASTRA Toolbox
    SparseMatrixHeader header;
    SparseMatrix matrix;
    loadSparseMatrixBinary(std::string(PROJECT_BASE_PATH) + projectionFileName, matrix, header);

    totalNonZeroElements = header.num_non_zero;

    // Check sizes match expected
    if (matrix.rows.size() != (totalRays + 1))
        throw std::runtime_error("Error: Rows/offsets size does not match expected number of rays + 1.");
    if (matrix.cols.size() != totalNonZeroElements || matrix.vals.size() != totalNonZeroElements)
        throw std::runtime_error("Error: Cols or Vals size does not match total non-zero elements.");

    // Precondition values by normalising each row to unit norm
    totalWeightSum = 0.0f;
    for (size_t i = 0; i < totalRays; ++i) {
        double rowNormSq = 0.0;

        // Compute the squared norm of the row
        for (size_t j = matrix.rows[i]; j < matrix.rows[i + 1]; ++j) {
            rowNormSq += static_cast<double>(matrix.vals[j] * matrix.vals[j]);
        }

        float rowNorm = static_cast<float>(sqrt(rowNormSq));
        if (rowNorm > 0.0f) {
            // Normalise the row and accumulate the normalised weight sum
            for (size_t j = matrix.rows[i]; j < matrix.rows[i + 1]; ++j) {
                matrix.vals[j] /= rowNorm;
            }
            totalWeightSum += 1.0f;  // Each normalised row has unit norm
        }
    }

    // Load into projection matrix CSR metal buffers
    offsetsBuffer = metalUtils->createBuffer((matrix.rows.size()) * sizeof(int), MTL::ResourceStorageModeShared, matrix.rows.data());
    colsBuffer = metalUtils->createBuffer(matrix.cols.size() * sizeof(int), MTL::ResourceStorageModeShared, matrix.cols.data());
    valsBuffer = metalUtils->createBuffer(matrix.vals.size() * sizeof(float), MTL::ResourceStorageModeShared, matrix.vals.data());
}

void MTLComputeEngine::initialisePhantom(std::vector<float> &phantomData) {
    // Flip phantom data vertically for correct orientation
    std::vector<float> flippedPhantomData = flipImageVertically(phantomData, geom.imageWidth, geom.imageHeight);

    // Precompute phantom norm for convergence checking
    float phantomNormSum = 0.0f;
    for (const auto &val : flippedPhantomData) {
        phantomNormSum += val * val;
    }
    phantomNorm = static_cast<double>(sqrt(phantomNormSum));

    // Create texture and buffer for phantom
    originalPhantomTexture =
        metalUtils->createTexture(geom.imageWidth, geom.imageHeight, MTL::PixelFormatR32Float, MTL::TextureUsageShaderRead);
    phantomBuffer =
        metalUtils->createBuffer(flippedPhantomData.size() * sizeof(float), MTL::ResourceStorageModeShared, flippedPhantomData.data());

    // Load original phantom data into texture
    MTL::Region region = MTL::Region::Make2D(0, 0, geom.imageWidth, geom.imageHeight);
    originalPhantomTexture->replaceRegion(region, 0, flippedPhantomData.data(), geom.imageWidth * sizeof(float));
    if (!originalPhantomTexture) throw std::runtime_error("Failed to create phantom texture.");
}

void MTLComputeEngine::computeSinogram(std::vector<float> &phantomData, double &scanTimeMs) {
    // Initialise phantom - flip vertically, load into buffer and texture, compute norm
    initialisePhantom(phantomData);

    // Start timing scan
    auto startTime = std::chrono::high_resolution_clock::now();
    sinogramBuffer = metalUtils->createBuffer(totalRays * sizeof(float), MTL::ResourceStorageModeShared);

    MTL::Function *computeSinogram = metalUtils->createKernelFn("computeSinogram", defaultLibrary);
    MTL::ComputePipelineState *scanPipeline = metalUtils->createComputePipeline(computeSinogram);

    // Encode and dispatch compute kernel
    auto cmdBuffer = commandQueue->commandBuffer();
    auto encoder = cmdBuffer->computeCommandEncoder();
    encoder->setBuffer(phantomBuffer, 0, 0);
    encoder->setBuffer(offsetsBuffer, 0, 1);
    encoder->setBuffer(colsBuffer, 0, 2);
    encoder->setBuffer(valsBuffer, 0, 3);
    encoder->setBuffer(sinogramBuffer, 0, 4);
    encoder->setBytes(&totalRays, sizeof(uint), 5);
    encoder->setComputePipelineState(scanPipeline);

    metalUtils->dispatchThreads(encoder, scanPipeline, totalRays);
    encoder->endEncoding();

    // Copy sinogram buffer into texture
    sinogramTexture = metalUtils->createTexture(geom.nDetectors, geom.nAngles, MTL::PixelFormatR32Float,
                                                MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    metalUtils->copyBufferToTexture(cmdBuffer, sinogramBuffer, sinogramTexture, geom.nDetectors, geom.nAngles);

    // Commit and wait only once everything is encoded
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    auto endTime = std::chrono::high_resolution_clock::now();
    scanTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    std::cout << "Scan time: " << scanTimeMs << " ms" << std::endl;

    /* Uncomment to save non-normalised sinogram buffer to .txt file */
    // std::string filePath =
    //     std::string(PROJECT_BASE_PATH) + "/metal-data/sinogram_" + std::to_string(geom.imageWidth) + ".txt";
    // metalUtils->saveTextureToFile(filePath, sinogramTexture);

    // Normalise sinogram texture
    float maxSinogramValue = 0.0f;
    findMaxValInTexture(sinogramTexture, maxSinogramValue);
    normaliseTexture(sinogramTexture, maxSinogramValue);
}

void MTLComputeEngine::findMaxValInTexture(MTL::Texture *texture, float &maxValue) {
    if (!texture) throw std::runtime_error("Texture to find max value in is null.");

    // Find max value in texture using atomic operations in Metal
    auto maxPerThreadGroupFn = metalUtils->createKernelFn("findMaxInTexture", defaultLibrary);
    MTL::ComputePipelineState *findMaxPipeline = metalUtils->createComputePipeline(maxPerThreadGroupFn);

    auto cmdBuffer = commandQueue->commandBuffer();

    // Query GPU for thread execution parameters
    NS::UInteger maxThreadsPerThreadgroup = findMaxPipeline->maxTotalThreadsPerThreadgroup();
    NS::UInteger threadExecutionWidth = findMaxPipeline->threadExecutionWidth();

    // Dynamically calculate 2D threadgroup size
    NS::UInteger tgWidth = std::min(threadExecutionWidth, texture->width());
    NS::UInteger tgHeight = std::min(maxThreadsPerThreadgroup / tgWidth, texture->height());
    MTL::Size threadgroupSize = MTL::Size(tgWidth, tgHeight, 1);

    MTL::Size gridSize = MTL::Size(texture->width(), texture->height(), 1);
    MTL::Size numThreadgroups = MTL::Size((gridSize.width + threadgroupSize.width - 1) / threadgroupSize.width,
                                          (gridSize.height + threadgroupSize.height - 1) / threadgroupSize.height, 1);

    // Buffer to store max value updated atomically
    MTL::Buffer *maxValuesBuffer = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    uint *maxValuePtr = static_cast<uint *>(maxValuesBuffer->contents());
    *maxValuePtr = 0;

    // Encode and dispatch
    auto p1Encoder = cmdBuffer->computeCommandEncoder();
    p1Encoder->setComputePipelineState(findMaxPipeline);
    p1Encoder->setTexture(texture, 0);
    p1Encoder->setBuffer(maxValuesBuffer, 0, 0);
    p1Encoder->dispatchThreads(gridSize, threadgroupSize);
    p1Encoder->endEncoding();
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    // Read back max value from buffer
    uint maxValueUint = *maxValuePtr;
    float maxFloatValue = std::bit_cast<float>(maxValueUint);
    maxValue = maxFloatValue;
}

void MTLComputeEngine::normaliseTexture(MTL::Texture *texture, float maxValue) {
    if (!texture) throw std::runtime_error("Texture to be normalised is null.");
    if (maxValue == 0.0f) {
        std::cerr << "Max value is zero, texture will not be normalised." << std::endl;
        return;
    }

    // Initialise normalise kernel
    auto normaliseFn = metalUtils->createKernelFn("normaliseKernel", defaultLibrary);
    MTL::ComputePipelineState *normalisePipeline = metalUtils->createComputePipeline(normaliseFn);

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

double MTLComputeEngine::reconstructImage(int numIterations, double &relativeErrorNorm, const float relaxationParameter,
                                          const double relativeErrorThreshold, const int errorCheckInterval) {
    auto startTimeTotal = std::chrono::high_resolution_clock::now();

    // Initialise kernel functions and pipelines
    auto cimminoFn = metalUtils->createKernelFn("cimminosReconstruction", defaultLibrary);
    auto applyUpdateFn = metalUtils->createKernelFn("applyUpdate", defaultLibrary);
    auto computeDifferenceFn = metalUtils->createKernelFn("computeRelativeDifference", defaultLibrary);
    auto cimminoPipeline = metalUtils->createComputePipeline(cimminoFn);
    auto applyUpdatePipeline = metalUtils->createComputePipeline(applyUpdateFn);
    auto computeDifferencePipeline = metalUtils->createComputePipeline(computeDifferenceFn);

    const size_t width = geom.imageWidth;
    const size_t height = geom.imageHeight;
    const uint imageSize = width * height;

    // Create buffers
    reconstructedBuffer = metalUtils->createBuffer(imageSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto updateBuffer = metalUtils->createBuffer(imageSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto differenceSumBuffer = metalUtils->createBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    // Initialise reconstructed image to zero
    memset(reconstructedBuffer->contents(), 0, reconstructedBuffer->length());
    auto cmdBuffer = commandQueue->commandBuffer();

    std::cout << "Starting reconstruction for " << numIterations << " iterations..." << std::endl;

    // Start timing reconstruction
    auto startTime = std::chrono::high_resolution_clock::now();

    // Compute once on CPU
    float relaxationFactor = relaxationParameter * (2.0f / totalWeightSum);

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
        encoder->setBytes(&relaxationFactor, sizeof(float), 8);
        metalUtils->dispatchThreads(encoder, cimminoPipeline, totalRays);
        encoder->endEncoding();

        // Dispatch the apply update kernel to get the new image estimate and calculate norms
        encoder = cmdBuffer->computeCommandEncoder();
        encoder->setComputePipelineState(applyUpdatePipeline);
        encoder->setBuffer(reconstructedBuffer, 0, 0);
        encoder->setBuffer(updateBuffer, 0, 1);
        encoder->setBytes(&imageSize, sizeof(uint), 2);
        metalUtils->dispatchThreads(encoder, applyUpdatePipeline, imageSize);
        encoder->endEncoding();

        cmdBuffer->commit();
        cmdBuffer->waitUntilCompleted();

        // Check for convergence every 50 iterations
        if ((i + 1) % errorCheckInterval == 0) {
            cmdBuffer = commandQueue->commandBuffer();
            encoder = cmdBuffer->computeCommandEncoder();

            // Compute difference between current reconstruction and original phantom
            encoder->setComputePipelineState(computeDifferencePipeline);
            encoder->setBuffer(reconstructedBuffer, 0, 0);
            encoder->setBuffer(phantomBuffer, 0, 1);
            encoder->setBuffer(differenceSumBuffer, 0, 2);
            metalUtils->dispatchThreads(encoder, computeDifferencePipeline, imageSize);
            encoder->endEncoding();
            cmdBuffer->commit();
            cmdBuffer->waitUntilCompleted();

            // Read back difference sum and compute relative error norm
            auto *differenceSum = static_cast<float *>(differenceSumBuffer->contents());
            relativeErrorNorm = sqrt(static_cast<double>(*differenceSum)) / phantomNorm;

            if (relativeErrorNorm < relativeErrorThreshold) {
                std::cout << "Converged after " << (i + 1) << " iterations with relative error norm " << relativeErrorNorm << std::endl;
                break;
            }
            memset(differenceSumBuffer->contents(), 0, sizeof(float));
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> totalReconstructTime = endTime - startTime;

    std::cout << "Reconstruction time for " << numIterations << " iterations " << totalReconstructTime.count() << " ms" << std::endl;

    // Copy reconstructed buffer into texture using blit encoder
    reconstructedTexture = metalUtils->createTexture(width, height, MTL::PixelFormatR32Float, MTL::TextureUsageShaderRead);
    cmdBuffer = commandQueue->commandBuffer();
    metalUtils->copyBufferToTexture(cmdBuffer, reconstructedBuffer, reconstructedTexture, width, height);
    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    // Save reconstructed texture to file
    std::string imageFileName = std::string(PROJECT_BASE_PATH) + "/metal-data/metal_" + std::to_string(numIterations) + "_" +
                                std::to_string(geom.imageWidth) + ".txt";
    metalUtils->saveTextureToFile(imageFileName, reconstructedTexture);

    auto endTimeTotal = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> totalTime = endTimeTotal - startTimeTotal;
    std::cout << "Total time including data transfers: " << totalTime.count() << " ms" << std::endl;

    return totalReconstructTime.count();
}
