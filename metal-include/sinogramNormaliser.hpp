#ifndef SINOGRAMNORMALISER_HPP
#define SINOGRAMNORMALISER_HPP

#include "mtlComputeEngine.hpp"

class SinogramNormaliser : public MTLComputeEngine {
public:
    SinogramNormaliser(MetalContext& context, const Geometry& geom);

    double normaliseSinogram(const std::string& fileName, uint nAngles, uint nDetectors);
};
#endif // SINOGRAMNORMALISER_HPP