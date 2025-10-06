// Definition of AlgorithmTester class for testing reconstruction and sinogram computation algorithms using Metal.
#ifndef ALGORITHM_TESTER_HPP
#define ALGORITHM_TESTER_HPP
#include "MetalComputeEngine.hpp"

class AlgorithmTester : public MTLComputeEngine {
public:
    /**
     * @brief AlgorithmTester constructor.
     * @param context The Metal context containing the device, command queue, and library.
     * @param geom The geometry structure containing geometry parameters.
     */
    AlgorithmTester(MetalContext& context, const Geometry& geom);

    /**
     * @brief Test the reconstruction algorithm by loading a small projection matrix,generating a test phantom, test sinogram,
     * and performing reconstruction. The accuracy of the reconstruction is checked against a known solution.
     * @param geom The geometry structure containing geometry parameters.
     * @param numIterations The number of iterations to perform for reconstruction.
     * @param finalErrorNorm Reference to store the norm of the final update for convergence analysis.
     */
    void testReconstruction(Geometry& geom, uint numIterations, double& finalErrorNorm);

    /**
     * @brief Test the sinogram computation by by loading a small projection matrix, generating a test phantom,
     * and computing the sinogram.
     * The accuracy of the computed sinogram is checked against a known solution.
     * @param geom The geometry structure containing geometry parameters.
     */
    void testSinogramComputation(Geometry& geom);
private:
    /**
     * @brief Load a test projection matrix from file matching image size and upload to Metal buffers.
     * @param geom The geometry structure containing geometry parameters.
     */
    void generateTestProjector(Geometry& geom);

    /**
     * @brief Generate a simple test phantom of all ones and initialise for Metal.
     * @param imageSize The size of the phantom image (imageSize x imageSize).
     */
    void generateTestPhantom(uint imageSize);

    /**
     * @brief Generate a simple test sinogram of all ones and initialise for Metal.
     * @param nAngles The number of projection angles.
     * @param nDetectors The number of detectors per angle.
     */
    void generateTestSinogram(uint nAngles, uint nDetectors);

    /**
     * @brief Check the accuracy of the computed sinogram against a known solution.
     * @param imageSize The size of the phantom image (imageSize x imageSize).
     * @return True if the sinogram is accurate within a tolerance, false otherwise.
     */
    bool checkSinogramAccuracy(uint imageSize);

    /**
     * @brief Check the accuracy of the reconstructed image against a known solution.
     * @param numIterations The number of iterations performed for reconstruction.
     * @return True if the reconstruction is accurate within a tolerance, false otherwise.
     */
    bool checkReconstructionAccuracy(uint numIterations);
};
#endif // ALGORITHM_TESTER_HPP