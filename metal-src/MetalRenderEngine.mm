/**
 * @file MetalRenderEngine.mm
 * @brief Implementation of the MTLRenderEngine class for rendering using Metal and GLFW.
 *
 * This class handles the setup of a GLFW window with a CAMetalLayer,
 * creation of the Metal render pipeline, and rendering of textures.
 */
#include "../metal-include/MetalRenderEngine.hpp"

MTLRenderEngine::MTLRenderEngine(MetalContext &context) {
    // Initialise Metal objects
    device = context.getDevice();
    commandQueue = context.getCommandQueue();
    library = context.getLibrary();

    // Initialise GLFW window with CAMetalLayer
    initWindow();

    // Create the render pipeline
    createRenderPipeline();
}

MTLRenderEngine::~MTLRenderEngine() {
    glfwDestroyWindow(glfwWindow);
    glfwTerminate();
}

void MTLRenderEngine::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindow = glfwCreateWindow(512, 512, "Cimmino's Phantom Reconstruction", NULL, NULL);
    if (!glfwWindow) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetWindowUserPointer(glfwWindow, this);
    glfwSetFramebufferSizeCallback(glfwWindow, frameBufferSizeCallback);
    glfwSetKeyCallback(glfwWindow, MTLRenderEngine::keyCallback);
    metalWindow = glfwGetCocoaWindow(glfwWindow);
    metalLayer = [CAMetalLayer layer];
    metalLayer.device = (__bridge id<MTLDevice>)device;
    metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    metalWindow.contentView.layer = metalLayer;
    metalWindow.contentView.wantsLayer = YES;
}

void MTLRenderEngine::frameBufferSizeCallback(GLFWwindow *window, int width, int height) {
    MTLRenderEngine *engine = (MTLRenderEngine *)glfwGetWindowUserPointer(window);
    engine->resizeFrameBuffer(width, height);
}

void MTLRenderEngine::resizeFrameBuffer(int width, int height) { metalLayer.drawableSize = CGSizeMake(width, height); }

CA::MetalDrawable *MTLRenderEngine::getNextDrawable(CAMetalLayer *nativeLayer) {
    // Call the native Objective-C method
    id<CAMetalDrawable> nativeDrawable = [nativeLayer nextDrawable];
    // Cast the result back to the C++ wrapper type
    return (__bridge CA::MetalDrawable *)nativeDrawable;
}

void MTLRenderEngine::createRenderPipeline() {
    NS::Error *error = nullptr;

    // Source for magma cmap https://github.com/BIDS/colormap/blob/master/colormaps.py
    std::string basePath = PROJECT_BASE_PATH;
    auto colourMapData = loadColourMapTexture(basePath + "/metal-data/magma.txt");

    // Create colour map texture
    long numColors = colourMapData.size() / 4;
    MTL::TextureDescriptor *texDesc =
        MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA32Float, numColors, 1, false);
    colourMapTexture = device->newTexture(texDesc);
    colourMapTexture->replaceRegion(MTL::Region(0, 0, numColors, 1), 0, colourMapData.data(),
                                    numColors * sizeof(float) * 4);

    // Extract vertex and fragment kernel functions from library
    MTL::Function *vertexFn = createKernelFn("vertex_main", library);
    MTL::Function *fragmentFn = createKernelFn("fragment_main", library);

    // Set render pipeline descriptor with functions and pixel format
    MTL::RenderPipelineDescriptor *renderDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    renderDesc->setVertexFunction(vertexFn);
    renderDesc->setFragmentFunction(fragmentFn);
    renderDesc->colorAttachments()->object(0)->setPixelFormat(static_cast<MTL::PixelFormat>(metalLayer.pixelFormat));

    if (!renderDesc) {
        std::cerr << "Error: Failed to create render pipeline descriptor: "
                  << error->localizedDescription()->utf8String() << std::endl;
        exit(-1);
    }

    // Create render pipeline
    renderPipeline = device->newRenderPipelineState(renderDesc, &error);
    if (!renderPipeline) {
        std::cerr << "Error: Failed to create render pipeline state: " << error->localizedDescription()->utf8String()
                  << std::endl;
        exit(-1);
    }
}

// Callback to handle key presses for switching textures in window
void MTLRenderEngine::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    // Retrieve the MTLRenderEngine instance from the user pointer
    MTLRenderEngine *engine = static_cast<MTLRenderEngine *>(glfwGetWindowUserPointer(window));

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
            case GLFW_KEY_RIGHT:
                engine->currentTextureIndex = (engine->currentTextureIndex + 1) % 3;  // Move forward
                break;
            case GLFW_KEY_LEFT:
                engine->currentTextureIndex = (engine->currentTextureIndex + 2) % 3;  // Move backward
                break;
        }
    }
}

void MTLRenderEngine::render() {
    while (!glfwWindowShouldClose(glfwWindow)) {
        NS::AutoreleasePool *framePool = NS::AutoreleasePool::alloc()->init();

        glfwPollEvents();

        CA::MetalDrawable *drawable = getNextDrawable(reinterpret_cast<CAMetalLayer *>(metalLayer));
        if (!drawable) {
            framePool->release();  // Drain the pool before continuing
            continue;
        }

        MTL::Texture *textureToDisplay = nullptr;

        // Set the current texture based on the index - use arrow keys to change
        switch (currentTextureIndex) {
            case 0:
                textureToDisplay = sinogramTexture;
                break;
            case 1:
                textureToDisplay = reconstructedTexture;
                break;
            case 2:
                textureToDisplay = originalPhantomTexture;
                break;
        }

        // Set render pass descriptor and configure colour attachment
        auto renderPassDesc = MTL::RenderPassDescriptor::renderPassDescriptor();
        auto colourAttachment = renderPassDesc->colorAttachments()->object(0);
        colourAttachment->setTexture(drawable->texture());
        colourAttachment->setLoadAction(MTL::LoadActionClear);
        colourAttachment->setClearColor(MTL::ClearColor(0.1, 0.1, 0.2, 1.0));
        colourAttachment->setStoreAction(MTL::StoreActionStore);

        auto cmdBuffer = commandQueue->commandBuffer();
        auto encoder = cmdBuffer->renderCommandEncoder(renderPassDesc);

        // Set pipeline state, textures and draw
        encoder->setRenderPipelineState(renderPipeline);
        encoder->setFragmentTexture(textureToDisplay, 0);
        encoder->setFragmentTexture(colourMapTexture, 1);
        encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
        encoder->endEncoding();

        cmdBuffer->presentDrawable(drawable);
        cmdBuffer->commit();

        framePool->release();
    }
}
