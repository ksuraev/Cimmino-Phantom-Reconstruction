#include "../metal-include/mtlComputeEngine.hpp"
#include "../metal-include/mtlRenderEngine.hpp"

constexpr int IMAGE_WIDTH = 256;
constexpr int IMAGE_HEIGHT = 256;
constexpr int NUM_ANGLES = 90;

constexpr const char PROJECTION_MATRIX_FILE[] = "projection_256.bin";
constexpr const char PHANTOM_FILE[] = "phantom_256.txt";
constexpr const char LOG_FILE[] = "metal_performance_log.csv";
constexpr const char SINOGRAM_TEST_FILE[] = "sinogram_1024_test.txt";

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
        std::vector<float> phantomData =
            loadPhantom(std::string(PROJECT_BASE_PATH) + "/metal-data/" + PHANTOM_FILE, geom);
        double scanTime = timeMethod_ms([&]() { mtlComputeEngine.computeSinogram(phantomData); });

        // // Test sinogram normalisation
        // std::vector<float> testSinogram;
        // mtlComputeEngine.testSinogramNormalisation(std::string(PROJECT_BASE_PATH) + "/metal-data/" +
        // SINOGRAM_TEST_FILE,
        //                                            testSinogram, 90, 2897);

        // Perform Cimmino's reconstruction
        double finalErrorNorm = 0.0;
        auto totalReconstructTime = mtlComputeEngine.reconstructImage(numIterations, finalErrorNorm);

        MTLRenderEngine mtlRenderEngine = MTLRenderEngine(context);

        // Get textures from metal compute engine and render with metal render engine
        mtlRenderEngine.setSinogramTexture(mtlComputeEngine.getSinogramTexture());
        mtlRenderEngine.setReconstructedTexture(mtlComputeEngine.getReconstructedTexture());
        mtlRenderEngine.setOriginalPhantomTexture(mtlComputeEngine.getOriginalPhantomTexture());
        mtlRenderEngine.render();

        logPerformance(geom, numIterations, projectionTime, scanTime, totalReconstructTime, finalErrorNorm,
                       std::string(PROJECT_BASE_PATH) + "/metal-logs/" + LOG_FILE);

        pPool->release();
    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    }
    return 0;
}
