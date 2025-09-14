#include "../include/mtlRenderEngine.hpp"

MTLRenderEngine::MTLRenderEngine(MetalContext& context)
{
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


void MTLRenderEngine::frameBufferSizeCallback(GLFWwindow* window, int width,
  int height) {
  MTLRenderEngine* engine = (MTLRenderEngine*)glfwGetWindowUserPointer(window);
  engine->resizeFrameBuffer(width, height);
}

void MTLRenderEngine::resizeFrameBuffer(int width, int height) {
  metalLayer.drawableSize = CGSizeMake(width, height);
}


CA::MetalDrawable* MTLRenderEngine::getNextDrawable(CAMetalLayer* nativeLayer) {
  // Call the native Objective-C method
  id<CAMetalDrawable> nativeDrawable = [nativeLayer nextDrawable];
  // Cast the result back to the C++ wrapper type
  return (__bridge CA::MetalDrawable*)nativeDrawable;
}

/**
 * @brief Create a kernel function from the default library.
 * @param functionName The name of the kernel function to create.
 * @return The created kernel function.
 */
MTL::Function* MTLRenderEngine::createKernelFn(const char* functionName) {
  MTL::Function* fn = library->newFunction(
    NS::String::string(functionName, NS::UTF8StringEncoding));
  if (!fn) {
    std::cerr << "Failed to find kernel " << functionName << " in the library."
      << std::endl;
    std::exit(-1);
  }
  return fn;
}

/**
 * @brief Create the render pipeline state for rendering.
 * This function sets up the vertex and fragment shaders, configures the
 * render pipeline descriptor, and creates the render pipeline state object.
 */
void MTLRenderEngine::createRenderPipeline() {
  NS::Error* error = nullptr;

  // Source for colour values https://github.com/BIDS/colormap/blob/master/colormaps.py
  auto colourMapData = loadColourMapTexture("../data/magma.txt");

  long numColors = colourMapData.size() / 4;

  MTL::TextureDescriptor* texDesc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA32Float, numColors, 1, false);

  colourMapTexture = device->newTexture(texDesc);
  colourMapTexture->replaceRegion(MTL::Region(0, 0, numColors, 1), 0, colourMapData.data(), numColors * sizeof(float) * 4);

  // Extract vertex and fragment kernel functions
  MTL::Function* vertexFn = createKernelFn("vertex_main");
  MTL::Function* fragmentFn = createKernelFn("fragment_main");

  // Set render pipeline descriptor with functions and pixel format
  MTL::RenderPipelineDescriptor* renderDesc = MTL::RenderPipelineDescriptor::alloc()->init();
  renderDesc->setVertexFunction(vertexFn);
  renderDesc->setFragmentFunction(fragmentFn);
  renderDesc->colorAttachments()->object(0)->setPixelFormat(static_cast<MTL::PixelFormat>(metalLayer.pixelFormat));

  if (!renderDesc) {
    std::cerr << "Error: Failed to create render pipeline descriptor: " << error->localizedDescription()->utf8String() << std::endl;
    exit(-1);
  }

  // Create render pipeline
  renderPipeline = device->newRenderPipelineState(renderDesc, &error);
  if (!renderPipeline) {
    std::cerr << "Error: Failed to create render pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
    exit(-1);
  }
}

int currentTextureIndex = 0;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS || action == GLFW_REPEAT) {
    switch (key) {
    case GLFW_KEY_RIGHT:
      currentTextureIndex = (currentTextureIndex + 1) % 3; // Move forward
      break;
    case GLFW_KEY_LEFT:
      currentTextureIndex = (currentTextureIndex + 2) % 3; // Move backward
      break;
    }
  }
}

/**
 * @brief Render loop to display textures.
 * This function handles the rendering loop, showing the sinogram,
 * reconstructed image, and original phantom textures. Use left/right arrow keys
 * to switch between textures.
 */
void MTLRenderEngine::render() {
  while (!glfwWindowShouldClose(glfwWindow)) {
    NS::AutoreleasePool* framePool = NS::AutoreleasePool::alloc()->init();

    glfwPollEvents();

    CA::MetalDrawable* drawable = getNextDrawable(reinterpret_cast<CAMetalLayer*>(metalLayer));
    if (!drawable) {
      framePool->release(); // Drain the pool before continuing
      continue;
    }

    MTL::Texture* textureToDisplay = nullptr;

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

void MTLRenderEngine::initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindow = glfwCreateWindow(400, 400, "Cimmino's Phantom Reconstruction", NULL, NULL);
  if (!glfwWindow) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwSetWindowUserPointer(glfwWindow, this);
  glfwSetFramebufferSizeCallback(glfwWindow, frameBufferSizeCallback);
  glfwSetKeyCallback(glfwWindow, keyCallback);
  metalWindow = glfwGetCocoaWindow(glfwWindow);
  metalLayer = [CAMetalLayer layer];
  metalLayer.device = (__bridge id<MTLDevice>)device;
  metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
  metalWindow.contentView.layer = metalLayer;
  metalWindow.contentView.wantsLayer = YES;
}