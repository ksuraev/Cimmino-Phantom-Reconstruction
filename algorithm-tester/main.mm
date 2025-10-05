// Used to test the algorithm accuracy with a known solution
// Using a small 32x32 image and 90 angles for quick testing
// Solution stored in solution_*.txt files with * being the iteration number

#include "AlgorithmTester.hpp"

constexpr int IMAGE_SIZE = 32;
constexpr int NUM_ANGLES = 90;

int main(int argc, char **argv) {
    int numIterations = 10;
    if (argc > 1) numIterations = std::atoi(argv[1]);
    try {
        NS::AutoreleasePool *pPool = NS::AutoreleasePool::alloc()->init();

        // Set geometry parameters for scanner
        uint numDetectors = std::ceil(2 * std::sqrt(2) * IMAGE_SIZE);
        Geometry geom = {IMAGE_SIZE, IMAGE_SIZE, NUM_ANGLES, numDetectors};

        MetalContext context = MetalContext();

        AlgorithmTester AlgorithmTester(context, geom);

        double finalErrorNorm = 0.0;
        AlgorithmTester.testReconstruction(geom, numIterations, finalErrorNorm);

        pPool->release();
    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    }
    return 0;
}
