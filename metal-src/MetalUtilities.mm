#include "../metal-include/MetalUtilities.hpp"

MetalUtilities::MetalUtilities(MTL::Device* device, MTL::Library* library, MTL::CommandQueue* commandQueue)
    : device(device), defaultLibrary(library), commandQueue(commandQueue) {
    if (!device || !defaultLibrary || !commandQueue) {
        throw std::runtime_error("Invalid Metal context provided to MetalUtilities.");
    }
}

MTL::Function* MetalUtilities::createKernelFn(const char* functionName, MTL::Library* library) {
    MTL::Function* fn = library->newFunction(NS::String::string(functionName, NS::UTF8StringEncoding));
    if (!fn) {
        std::cerr << "Failed to find kernel " << functionName << " in the library." << std::endl;
        std::exit(-1);
    }
    return fn;
}

MTL::ComputePipelineState* MetalUtilities::createComputePipeline(MTL::Function* function) {
    NS::Error* error = nullptr;
    MTL::ComputePipelineState* pipeline = device->newComputePipelineState(function, &error);
    if (!pipeline) {
        std::cerr << "Error: Failed to create pipeline state. " << pipeline->label() << " "
                  << error->localizedDescription()->utf8String() << std::endl;
        std::exit(-1);
    }
    function->release();
    return pipeline;
}

MTL::Buffer* MetalUtilities::createBuffer(size_t size, MTL::ResourceOptions options, void* data) {
    MTL::Buffer* buffer = nullptr;
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

MTL::Texture* MetalUtilities::createTexture(uint width, uint height, MTL::PixelFormat pixelFormat,
                                            MTL::TextureUsage usage) {
    // Create a texture descriptor
    MTL::TextureDescriptor* textureDesc =
        MTL::TextureDescriptor::texture2DDescriptor(pixelFormat, width, height, false);
    textureDesc->setUsage(usage);

    // Create the texture
    MTL::Texture* texture = device->newTexture(textureDesc);
    if (!texture) {
        std::cerr << "Error: Failed to create texture of size " << width << "x" << height << std::endl;
        exit(-1);
    }

    return texture;
}

void MetalUtilities::dispatchThreads(MTL::ComputeCommandEncoder* encoder, MTL::ComputePipelineState* pipeline,
                                     size_t totalElements) {
    // Calculate thread group size
    NS::UInteger threadGroupSize = pipeline->maxTotalThreadsPerThreadgroup();

    // Check if thread group size exceeds total elements
    if (threadGroupSize > totalElements) threadGroupSize = totalElements;

    // Set grid size and dispatch threads
    MTL::Size gridSize = MTL::Size(totalElements, 1, 1);
    encoder->dispatchThreads(gridSize, MTL::Size(threadGroupSize, 1, 1));
}

void MetalUtilities::copyBufferToTexture(MTL::CommandBuffer* cmdBuffer, MTL::Buffer* buffer, MTL::Texture* texture,
                                         size_t width, size_t height) {
    auto blitEncoder = cmdBuffer->blitCommandEncoder();
    blitEncoder->copyFromBuffer(buffer, 0, width * sizeof(float), 0, MTL::Size(width, height, 1), texture, 0, 0,
                                MTL::Origin(0, 0, 0));
    blitEncoder->endEncoding();
}