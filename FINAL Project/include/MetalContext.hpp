/**
 * @file MetalContext.hpp
 * @brief Metal context class for managing Metal device, command queue, and library.
 *
 * This class encapsulates the initialisation and management of a Metal device,
 * command queue, and library. It provides methods to access these components.
 */
#ifndef METALCONTEXT_HPP
#define METALCONTEXT_HPP
#include <Metal/Metal.hpp>
#include <iostream>

class MetalContext {
public:
    /**
     * @brief Metal context constructor.
     * Initialises the Metal device, command queue, and loads the default library.
     */
    MetalContext() {
        device = MTL::CreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Error: Failed to create device." << std::endl;
            std::exit(-1);
        }
        commandQueue = device->newCommandQueue();
        if (!commandQueue) {
            std::cerr << "Error: Failed to create command queue." << std::endl;
            std::exit(-1);
        }

        const std::string basePath = PROJECT_BASE_PATH;

        // Load precompiled metallib from file
        std::string libraryPath = basePath + "/build/metallibrary.metallib";

        NS::String* filePath = NS::String::string(libraryPath.c_str(), NS::UTF8StringEncoding);
        NS::Error* error;
        library = device->newLibrary(filePath, &error);
        if (error) {
            std::cerr << "Error: Failed to load metallib from file: " << error->localizedDescription()->utf8String() << std::endl;
            std::exit(-1);
        }
    }

    ~MetalContext() {
        commandQueue->release();
        library->release();
        device->release();
    }

    /* Getters */
    MTL::Device* getDevice() const { return device; }
    MTL::CommandQueue* getCommandQueue() const { return commandQueue; }
    MTL::Library* getLibrary() const { return library; }

private:
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::Library* library;
};
#endif  // METALCONTEXT_HPP