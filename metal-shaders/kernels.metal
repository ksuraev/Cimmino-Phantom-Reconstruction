// Metal Shading Language (MSL) code for GPU image reconstruction and processing
#include <metal_stdlib>
using namespace metal;

// Geometry structure to hold imaging/scanner parameters
struct Geometry {
    uint imageWidth;
    uint imageHeight;
    uint nAngles;
    uint nDetectors;
};

// Calculate the sinogram by projecting the phantom image to simulate the projection of rays through the image/phantom
kernel void computeSinogram(const device float* phantom [[buffer(0)]], const device int* offsetsBuffer_A [[buffer(1)]],
                            const device int* colsBuffer_A [[buffer(2)]], const device float* valsBuffer_A [[buffer(3)]],
                            device float* sinogram [[buffer(4)]], constant uint& numRays [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    uint rayIndex = gid;
    if (rayIndex >= numRays) return;

    int rowStart = offsetsBuffer_A[rayIndex];
    int rowEnd = offsetsBuffer_A[rayIndex + 1];

    float dotProduct = 0.0f;

    // Vectorised loop processing 4 elements at a time
    int i = 0;
    for (; i + 3 < rowEnd - rowStart; i += 4) {
        int base = rowStart + i;

        float4 phantomVals = float4(phantom[colsBuffer_A[base]], phantom[colsBuffer_A[base + 1]], phantom[colsBuffer_A[base + 2]],
                                    phantom[colsBuffer_A[base + 3]]);
        float4 coeffs = float4(valsBuffer_A[base], valsBuffer_A[base + 1], valsBuffer_A[base + 2], valsBuffer_A[base + 3]);

        dotProduct += dot(phantomVals, coeffs);
    }

    // Remainder loop for leftover elements
    for (; i < rowEnd - rowStart; ++i) {
        int idx = rowStart + i;
        dotProduct += valsBuffer_A[idx] * phantom[colsBuffer_A[idx]];
    }

    sinogram[rayIndex] = dotProduct;
}

kernel void findMaxInTexture(texture2d<float, access::read> inputTexture [[texture(0)]], device atomic_uint* maxValue [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= inputTexture.get_width() || gid.y >= inputTexture.get_height()) return;

    // Read the pixel value from the texture
    float pixelValue = inputTexture.read(gid).r;

    // Convert the float value to uint for atomic operations
    uint intPixelValue = as_type<uint>(pixelValue);

    // Atomically update the maximum value
    atomic_fetch_max_explicit(maxValue, intPixelValue, memory_order_relaxed);
}

// Normalise the texture by dividing each pixel by the global maximum
kernel void normaliseKernel(texture2d<float, access::read_write> inputTexture [[texture(0)]], constant float& maxValue [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= inputTexture.get_width() || gid.y >= inputTexture.get_height()) return;

    // Read the current pixel value
    float currentVal = inputTexture.read(gid).r;

    // Calculate the normalised value and write it back to the texture
    if (maxValue > 0.0f) {
        float normalisedVal = currentVal / maxValue;
        inputTexture.write(float4(normalisedVal, normalisedVal, normalisedVal, 1.0), gid);
    } else {
        inputTexture.write(float4(0.0, 0.0, 0.0, 1.0), gid);  // Handle maxValue == 0
    }
}

// Pass 1 of Cimmino's algorithm:
kernel void cimminosReconstruction(const device float* reconstructedBuffer [[buffer(0)]],  // The current reconstruction x^k
                                   const device float* sinogramBuffer_b [[buffer(1)]], const device int* offsetsBuffer_A [[buffer(2)]],
                                   const device int* colsBuffer_A [[buffer(3)]], const device float* valsBuffer_A [[buffer(4)]],
                                   constant float& totalWeightSum [[buffer(5)]], constant uint& numRays [[buffer(6)]],
                                   device atomic_float* updateBuffer [[buffer(7)]],  // The next reconstruction x^(k+1)
                                   uint gid [[thread_position_in_grid]]) {
    uint rayIndex = gid;
    if (rayIndex >= numRays) return;

    // Get row start and end for this ray
    int rowStart = offsetsBuffer_A[rayIndex];
    int rowEnd = offsetsBuffer_A[rayIndex + 1];

    // Compute the dot product for this ray
    float dotProduct = 0.0f;
    for (int i = rowStart; i < rowEnd; ++i) {
        dotProduct += valsBuffer_A[i] * reconstructedBuffer[colsBuffer_A[i]];
    }

    // Compute residual and scaling factor
    float b_i = sinogramBuffer_b[rayIndex];
    float residual = b_i - dotProduct;
    float scalar = 2.0f * (1.0f / totalWeightSum) * residual;

    // Back Project - Add this ray's contribution to the update buffer
    for (int i = rowStart; i < rowEnd; ++i) {
        int pixelIndex = colsBuffer_A[i];
        float weight = valsBuffer_A[i];
        float contribution = scalar * weight;

        // Atomically add the contribution to prevent race conditions
        atomic_fetch_add_explicit(&updateBuffer[pixelIndex], contribution, memory_order_relaxed);
    }
}

// Pass 2 of Cimmino's algorithm: apply the update to the reconstructed image
kernel void applyUpdate(device float* reconstructedBuffer [[buffer(0)]], const device float* updateBuffer [[buffer(1)]],
                        constant uint& numPixels [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= numPixels) return;
    reconstructedBuffer[gid] += updateBuffer[gid];
}

// Compute the relative difference between the reconstruction and the phantom
kernel void computeRelativeDifference(const device float* reconstructedBuffer [[buffer(0)]],
                                      const device float* phantomBuffer [[buffer(1)]],
                                      device atomic_float* differenceSumBuffer [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    float currentValue = reconstructedBuffer[gid];
    float phantomValue = phantomBuffer[gid];
    float difference = currentValue - phantomValue;

    // Accumulate the squared difference for the numerator
    atomic_fetch_add_explicit(differenceSumBuffer, difference * difference, memory_order_relaxed);
}

// Vertex and Fragment shaders for rendering the texture with a colourmap
// Source https://metaltutorial.com/Lesson%201%3A%20Hello%20Metal/3.%20Textures/
struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut vertex_main(uint vid [[vertex_id]]) {
    VertexOut out;
    float4 positions[4] = {float4(-1, -1, 0, 1), float4(1, -1, 0, 1), float4(-1, 1, 0, 1), float4(1, 1, 0, 1)};
    float2 texCoords[4] = {float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1)};
    out.position = positions[vid];
    out.texCoord = texCoords[vid];
    return out;
}

// A fragment shader that samples from the image texture and maps it to a colourmap
// Source https://metaltutorial.com/Lesson%201%3A%20Hello%20Metal/3.%20Textures/
fragment float4 fragment_main(VertexOut in [[stage_in]], texture2d<float> imageTexture [[texture(0)]],
                              texture2d<float> colourMapTexture [[texture(1)]]) {
    // Sample the image texture
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);

    float value = imageTexture.sample(s, in.texCoord).r;

    // Map single channel to colourmap texture
    float2 colourMapUV = float2(value, 0.5);
    float4 colour = colourMapTexture.sample(s, colourMapUV);
    return colour;
}