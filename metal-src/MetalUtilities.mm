// Implementation of Metal utility functions for setting up and managing Metal resources.

#include "MetalUtilities.hpp"

MetalUtilities::MetalUtilities(MTL::Device* device, MTL::Library* library, MTL::CommandQueue* commandQueue)
    : device(device), defaultLibrary(library), commandQueue(commandQueue) {
    if (!device || !defaultLibrary || !commandQueue) throw std::runtime_error("Invalid Metal context provided to MetalUtilities.");
}

MTL::Function* MetalUtilities::createKernelFn(const char* functionName, MTL::Library* library) {
    MTL::Function* fn = library->newFunction(NS::String::string(functionName, NS::UTF8StringEncoding));
    if (!fn) throw std::runtime_error("Failed to create kernel function " + std::string(functionName));
    return fn;
}

MTL::ComputePipelineState* MetalUtilities::createComputePipeline(MTL::Function* function) {
    NS::Error* error = nullptr;
    MTL::ComputePipelineState* pipeline = device->newComputePipelineState(function, &error);
    if (!pipeline)
        throw std::runtime_error("Error: Failed to create pipeline state. " +
                                 (pipeline->label() ? std::string(pipeline->label()->utf8String()) : "Unnamed Pipeline") + " " +
                                 (error ? std::string(error->localizedDescription()->utf8String()) : "Unknown Error"));
    function->release();
    return pipeline;
}

MTL::Buffer* MetalUtilities::createBuffer(size_t size, MTL::ResourceOptions options, void* data) {
    MTL::Buffer* buffer = nullptr;
    if (data)
        buffer = device->newBuffer(data, size, options);
    else
        buffer = device->newBuffer(size, options);

    if (!buffer) throw std::runtime_error("Error: Failed to create buffer of size " + std::to_string(size));
    return buffer;
}

MTL::Texture* MetalUtilities::createTexture(uint width, uint height, MTL::PixelFormat pixelFormat, MTL::TextureUsage usage) {
    // Create a texture descriptor
    MTL::TextureDescriptor* textureDesc = MTL::TextureDescriptor::texture2DDescriptor(pixelFormat, width, height, false);
    textureDesc->setUsage(usage);

    // Create the texture
    MTL::Texture* texture = device->newTexture(textureDesc);
    if (!texture) throw std::runtime_error("Error: Failed to create texture.");
    return texture;
}

void MetalUtilities::dispatchThreads(MTL::ComputeCommandEncoder* encoder, MTL::ComputePipelineState* pipeline, size_t totalElements) {
    // Calculate thread group size
    NS::UInteger threadGroupSize = pipeline->maxTotalThreadsPerThreadgroup();

    // Check if thread group size exceeds total elements
    if (threadGroupSize > totalElements) threadGroupSize = totalElements;

    // Set grid size and dispatch threads
    MTL::Size gridSize = MTL::Size(totalElements, 1, 1);
    encoder->dispatchThreads(gridSize, MTL::Size(threadGroupSize, 1, 1));
}

void MetalUtilities::copyBufferToTexture(
    MTL::CommandBuffer* cmdBuffer, MTL::Buffer* buffer, MTL::Texture* texture, size_t width, size_t height) {
    auto blitEncoder = cmdBuffer->blitCommandEncoder();
    blitEncoder->copyFromBuffer(buffer, 0, width * sizeof(float), 0, MTL::Size(width, height, 1), texture, 0, 0, MTL::Origin(0, 0, 0));
    blitEncoder->endEncoding();
}

void MetalUtilities::saveTextureToFile(const std::string& filename, MTL::Texture* texture) {
    if (!texture) {
        std::cerr << "Error: Cannot save a null texture." << std::endl;
        return;
    }

    long width = texture->width();
    long height = texture->height();
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Texture has invalid dimensions." << std::endl;
        return;
    }

    std::vector<float> textureVector(width * height);

    // Define the region to read (entire texture)
    MTL::Region region = MTL::Region::Make2D(0, 0, width, height);
    texture->getBytes(textureVector.data(), width * sizeof(float), region, 0);

    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "' for writing." << std::endl;
        return;
    }

    for (long y = 0; y < height; ++y) {
        for (long x = 0; x < width; ++x) {
            outFile << textureVector[y * width + x];
            if (x < width - 1) {
                outFile << " ";
            }
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "Texture data successfully saved to text file." << std::endl;
}