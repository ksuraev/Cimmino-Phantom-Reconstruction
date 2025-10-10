#include "MetalComputeEngine.hpp"
#include "MetalRenderEngine.hpp"

constexpr int IMAGE_WIDTH = 256;
constexpr int IMAGE_HEIGHT = 256;
constexpr int NUM_ANGLES = 360;

constexpr const char PROJECTION_MATRIX_FILE[] = "/metal-data/projection_256.bin";
constexpr const char PHANTOM_FILE[] = "/metal-data/phantom_256.txt";
constexpr const char LOG_FILE[] = "/metal-logs/metal_performance_log.csv";

constexpr float RELAXATION_FACTOR = 350.0;

int main(int argc, char **argv) {
    if (IMAGE_WIDTH != IMAGE_HEIGHT) {
        std::cerr << "Image width and height must be equal." << std::endl;
        return -1;
    }

    int numIterations = 100;
    if (argc > 1) numIterations = std::atoi(argv[1]);

    try {
        NS::AutoreleasePool *pPool = NS::AutoreleasePool::alloc()->init();

        // Set geometry parameters for scanner
        uint numDetectors = std::ceil(2 * std::sqrt(2) * IMAGE_WIDTH);
        Geometry geom = {IMAGE_WIDTH, IMAGE_HEIGHT, NUM_ANGLES, numDetectors};

        MetalContext context = MetalContext();

        MTLComputeEngine mtlComputeEngine = MTLComputeEngine(context, geom);

        // Load projection matrix from .bin file and compute total weight sum
        double projectionTime = timeMethod_ms([&]() { mtlComputeEngine.loadProjectionMatrix(PROJECTION_MATRIX_FILE); });

        // Compute sinogram for the phantom
        std::vector<float> phantomData = loadPhantom(std::string(PROJECT_BASE_PATH) + PHANTOM_FILE, geom);
        double scanTime = 0.0;
        mtlComputeEngine.computeSinogram(phantomData, scanTime);

        // Perform Cimmino's reconstruction
        double finalErrorNorm = 0.0;
        auto totalReconstructTime = mtlComputeEngine.reconstructImage(numIterations, finalErrorNorm, RELAXATION_FACTOR);

        // MTLRenderEngine mtlRenderEngine = MTLRenderEngine(context);

        // // Get textures from metal compute engine and render with metal render engine
        // mtlRenderEngine.setSinogramTexture(mtlComputeEngine.getSinogramTexture());
        // mtlRenderEngine.setReconstructedTexture(mtlComputeEngine.getReconstructedTexture());
        // mtlRenderEngine.setOriginalPhantomTexture(mtlComputeEngine.getOriginalPhantomTexture());
        // mtlRenderEngine.render();

        // logPerformance(geom, numIterations, projectionTime, scanTime, totalReconstructTime, finalErrorNorm,
        //                std::string(PROJECT_BASE_PATH) + LOG_FILE);
        logExperiment("Model-11", geom, numIterations, RELAXATION_FACTOR, finalErrorNorm, totalReconstructTime,
            std::string(PROJECT_BASE_PATH) + "/metal-logs/other_phantom_experiment.csv");

        pPool->release();
    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    }
    return 0;
}
