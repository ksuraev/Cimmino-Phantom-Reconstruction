#ifndef NORMALISATIONPROFILER_HPP
#define NORMALISATIONPROFILER_HPP

#include "MetalComputeEngine.hpp"

class NormalisationProfiler : public MTLComputeEngine {
public:
    NormalisationProfiler(MetalContext& context, const Geometry& geom);

    double normaliseSinogram(const std::string& fileName, uint nAngles, uint nDetectors);

    void logPerformance(const std::string& logFilePath, double normalisationTime);
};
#endif // NORMALISATIONPROFILER_HPP