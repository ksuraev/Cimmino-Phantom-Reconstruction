#ifndef METAL_UTILITIES_HPP
#define METAL_UTILITIES_HPP
#include <Metal/Metal.hpp>
#include <string>
#include <iostream>

class MetalUtilities {
public:
    MetalUtilities(MTL::Device* device, MTL::Library* library, MTL::CommandQueue* commandQueue);
    /**
     * @brief Create a Metal kernel function from the library.
     * @param functionName The name of the kernel function to create.
     * @return A pointer to the created Metal function.
     * Exits the program if the function cannot be found.
     */
    MTL::Function* createKernelFn(const char* functionName, MTL::Library* library);

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
    MTL::Buffer* createBuffer(size_t size, MTL::ResourceOptions options, void* data = nullptr);

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
     * @brief Calculate grid and thread group sizes and dispatch threads for a compute kernel.
     * @param encoder The compute command encoder.
     * @param pipeline The compute pipeline state.
     * @param totalElements The total number of elements to process.
     */
    void dispatchThreads(MTL::ComputeCommandEncoder* encoder, MTL::ComputePipelineState* pipeline, size_t totalElements);

    /**
     * @brief Save a Metal texture to a binary file for testing.
     * @param filename The name of the binary file to save the texture data to.
     * @param texture The Metal texture to be saved.
     */
    void saveTextureToFile(const std::string& filename, MTL::Texture* texture);

private:
    MTL::Device* device;
    MTL::Library* defaultLibrary;
    MTL::CommandQueue* commandQueue;
};
#endif // METAL_UTILITIES_HPP