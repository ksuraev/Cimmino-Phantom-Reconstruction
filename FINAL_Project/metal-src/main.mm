// TODO: Add timer function to utilities, add error handling where possible, add
// comments, readme file, doc, remove releases Add python code to generate
// phantom data cmake --build build --config Release
// ./build/MetalCpp
// rm -rf build
// cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
// cmake --build build --clean-first
#include "../metal-include/mtlComputeEngine.hpp"
#include "../metal-include/mtlRenderEngine.hpp"

#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define NUM_ANGLES 90  // Number of projection angles

// Name of files in data directory
const std::string PROJECTION_MATRIX_FILE = "projection_256_astra.bin";
const std::string PHANTOM_FILE = "phantom_256.txt";
const std::string LOG_FILE = "metal_performance_log.csv";

int main(int argc, char **argv) {
    if (IMAGE_WIDTH != IMAGE_HEIGHT) {
        std::cerr << "Image width and height must be the same." << std::endl;
        return -1;
    }
    // Default value
    int numIterations = 1000;

    if (argc > 1) {
        numIterations = std::atoi(argv[1]);  // Convert the first argument to an integer
    }

    try {
        // Create an autorelease pool for memory management
        NS::AutoreleasePool *pPool = NS::AutoreleasePool::alloc()->init();

        // Set geometry parameters for scanner
        uint numDetectors = std::ceil(2 * std::sqrt(2) * IMAGE_WIDTH);
        Geometry geom = {IMAGE_WIDTH, IMAGE_HEIGHT, NUM_ANGLES, numDetectors};

        // Initialise Metal context - device, command queue, library
        MetalContext context = MetalContext();

        // Create compute and render engines
        MTLComputeEngine mtlComputeEngine = MTLComputeEngine(context, geom);
        MTLRenderEngine mtlRenderEngine = MTLRenderEngine(context);

        // Generate projection matrix
        double projectionTime =
            timeMethod_ms([&]() { mtlComputeEngine.generateProjectionMatrix(PROJECTION_MATRIX_FILE); });

        // Generate sinogram for loaded phantom
        std::string basePath = PROJECT_BASE_PATH;
        std::vector<float> phantomData = loadPhantom(basePath + "/metal-data/" + PHANTOM_FILE, geom);
        double scanTime = timeMethod_ms([&]() { mtlComputeEngine.performScan(phantomData); });

        double finalErrorNorm = 0.0;
        // Perform cimmino reconstruction
        auto totalReconstructTime = mtlComputeEngine.reconstructImage(numIterations, finalErrorNorm);

        // Get textures from metal compute engine and set in render engine
        mtlRenderEngine.setSinogramTexture(mtlComputeEngine.getSinogramTexture());
        mtlRenderEngine.setReconstructedTexture(mtlComputeEngine.getReconstructedTexture());
        mtlRenderEngine.setOriginalPhantomTexture(mtlComputeEngine.getOriginalPhantomTexture());

        // Open window and display sinogram, reconstructed image and original
        // phantom
        mtlRenderEngine.render();

        // Log performance metrics
        logPerformance(geom, numIterations, projectionTime, scanTime, totalReconstructTime, finalErrorNorm,
                       basePath + "/metal-logs/" + LOG_FILE);

        // Release the autorelease pool
        pPool->release();
    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    }

    return 0;
}
