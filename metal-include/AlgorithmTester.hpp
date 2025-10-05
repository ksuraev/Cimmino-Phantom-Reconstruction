// For testing reconstruction algorithm accuracy with known solution
#ifndef ALGORITHM_TESTER_HPP
#define ALGORITHM_TESTER_HPP
#include "MetalComputeEngine.hpp"

class AlgorithmTester : public MTLComputeEngine {
public:
    AlgorithmTester(MetalContext& context, const Geometry& geom);

    void testReconstruction(Geometry& geom, int numIterations, double& finalErrorNorm);

    void logPerformance(const std::string& logFilePath, double reconstructionTime);
private:
    uint imageSize;
    uint nIterations;

    void generateTestProjector(uint nAngles, uint nDetectors, uint imageSize);
    void generateTestPhantom(uint size);
    void generateTestSinogram(uint nAngles, uint nDetectors);
    bool checkAccuracy(int numIterations);

};
#endif // ALGORITHM_TESTER_HPP