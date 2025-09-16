/**
 * @file mtlRenderEngine.hpp
 * @brief Header file for the MTLRenderEngine class, which handles rendering using Metal.
 * This class handles the setup of a GLFW window with a CAMetalLayer,
 * creation of the Metal render pipeline, and rendering of textures.
 */
#ifndef MTLRENDERENGINE_HPP
#define MTLRENDERENGINE_HPP

 // Using GLFW for window management with Metal-cpp
#ifdef __OBJC__
#define GLFW_INCLUDE_NONE
#import <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#import <GLFW/glfw3native.h>
#endif

#include <QuartzCore/CAMetalLayer.h>
#include <QuartzCore/QuartzCore.hpp>
#include "MetalContext.hpp"

#include "Utilities.hpp"

class MTLRenderEngine {
public:
    /**
     * @brief Metal render engine constructor.
     * @param context The Metal context containing the device, command queue, and library.
     */
    MTLRenderEngine(MetalContext& context);

    /**
     * @brief Metal render engine destructor.
     * Cleans up Metal resources and GLFW window.
     */
    ~MTLRenderEngine();

    /**
     * @brief Create the Metal render pipeline.
     */
    void createRenderPipeline();

    /**
     * @brief Render loop to display textures.
     * This function handles the rendering loop, showing the sinogram,
     * reconstructed image, and original phantom textures. Use left/right arrow keys
     * to switch between textures.
     */
    void render();

    /**
     * @brief GLFW key callback to handle user input for switching textures.
     * @param window The GLFW window receiving the event.
     * @param key The keyboard key that was pressed or released.
     * @param scancode The system-specific scancode of the key.
     * @param action GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT.
     * @param mods Bit field describing which modifier keys were held down.
     */
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    /**
     * @brief Set the reconstructed texture from the compute engine.
     * @param texture The Metal texture containing the reconstructed image.
     */
    void setReconstructedTexture(MTL::Texture* texture) { reconstructedTexture = texture; }

    /**
     * @brief Set the sinogram texture from the compute engine.
     * @param texture The Metal texture containing the sinogram data.
     */
    void setSinogramTexture(MTL::Texture* texture) { sinogramTexture = texture; }

    /**
     * @brief Set the original phantom texture from the compute engine.
     * @param texture The Metal texture containing the original phantom image.
     */
    void setOriginalPhantomTexture(MTL::Texture* texture) { originalPhantomTexture = texture; }

private:
    MTL::Device* device;
    MTL::Library* library;
    MTL::CommandQueue* commandQueue;

    /* WINDOW & RENDER PIPELINE */
    GLFWwindow* glfwWindow;
    NSWindow* metalWindow;
    CAMetalLayer* metalLayer;
    MTL::RenderPipelineState* renderPipeline;

    /* TEXTURES */
    MTL::Texture* colourMapTexture;
    MTL::Texture* reconstructedTexture;
    MTL::Texture* sinogramTexture;
    MTL::Texture* originalPhantomTexture;

    // Current texture being displayed (0: sinogram, 1: reconstructed, 2: original phantom)
    int currentTextureIndex = 0;

    /**
     * @brief Initialise a GLFW window with a CAMetalLayer for Metal rendering.
     * This function sets up a GLFW window, configures it for Metal rendering,
     * and attaches a CAMetalLayer to the window's content view.
     * Source https://metaltutorial.com/Lesson%201%3A%20Hello%20Metal/1.%20Hello%20Window/
     */
    void initWindow();

    /**
     * @brief GLFW framebuffer size callback to handle window resizing.
     * @param window The GLFW window that was resized.
     * @param width The new width of the framebuffer.
     * @param height The new height of the framebuffer.
     * Source https://metaltutorial.com/Lesson%201%3A%20Hello%20Metal/3.%20Textures/
     */
    static void frameBufferSizeCallback(GLFWwindow* window, int width, int height);

    /**
     * @brief Resize the Metal framebuffer to match the new window dimensions.
     * @param width The new width of the framebuffer.
     * @param height The new height of the framebuffer.
     */
    void resizeFrameBuffer(int width, int height);

    /**
     * @brief Get the next drawable from the CAMetalLayer.
     * @param nativeLayer The CAMetalLayer instance.
     * @return A pointer to the next CA::MetalDrawable.
     * This function bridges between C++ and Objective-C to call the native method.
     */
    CA::MetalDrawable* getNextDrawable(CAMetalLayer* nativeLayer);

    // Move to MetalContext or a utility class?
    MTL::Function* createKernelFn(const char* functionName);
};
#endif  // MTLRENDERENGINE_HPP