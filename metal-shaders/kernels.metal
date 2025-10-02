// Metal Shading Language (MSL) code for GPU image reconstruction and processing
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Geometry structure to hold imaging/scanner parameters
struct Geometry {
  uint imageWidth;
  uint imageHeight;
  uint nAngles;
  uint nDetectors;
};


// Calculate the sinogram by projecting the phantom image
// Simulate the projection of rays through the image/phantom
kernel void computeSinogram(const device float* phantom [[buffer(0)]],
                            const device int* offsetsBuffer_A [[buffer(1)]],
                            const device int* colsBuffer_A [[buffer(2)]],
                            const device float* valsBuffer_A [[buffer(3)]],
                            device float* sinogram [[buffer(4)]],
                            constant uint& numRays [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    // Each thread computes one row of the sinogram
    uint rayIndex = gid;
    if (rayIndex >= numRays) return;

    // Get row start and end for this ray
    int rowStart = offsetsBuffer_A[rayIndex];
    int rowEnd = offsetsBuffer_A[rayIndex + 1];

    // Compute the dot product for this ray
    float dotProduct = 0.0f;
    for (int i = rowStart; i < rowEnd; ++i) {
        dotProduct += valsBuffer_A[i] * phantom[colsBuffer_A[i]];
    }

    // Store the result in the sinogram buffer
    sinogram[rayIndex] = dotProduct;
}

// Source https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
kernel void findMaxPerThreadgroupKernel(texture2d<float, access::read> input_texture [[texture(0)]],
                                        device float* map_buffer [[buffer(0)]],
                                        constant uint& threadgroup_size [[buffer(1)]],
                                        uint2 t_pos_in_grid [[thread_position_in_grid]],
                                        uint t_idx_in_tg [[thread_index_in_threadgroup]],
                                        uint simd_lane_id [[thread_index_in_simdgroup]],
                                        uint simdgroup_size [[threads_per_simdgroup]],
                                        uint simd_group_id [[simdgroup_index_in_threadgroup]],
                                        uint2 tg_pos_in_grid [[threadgroup_position_in_grid]],
                                        uint2 tg_per_grid [[threadgroups_per_grid]]) {
  // Threadgroup shared memory                                          
  threadgroup float local_max_values[256];  

  // Load pixel value
  float pixelValue = 0.0f;
  if (t_pos_in_grid.x < input_texture.get_width() && t_pos_in_grid.y < input_texture.get_height()) {
    pixelValue = input_texture.read(t_pos_in_grid).r;
  }

  // Per-SIMD partial reduction
  float val = pixelValue;
  for (uint offset = simdgroup_size / 2; offset > 0; offset /= 2) {
    val = max(val, simd_shuffle_down(val, offset));
  }

  // Write one value per SIMD group to threadgroup memory
  if (simd_lane_id == 0) {
    local_max_values[simd_group_id] = val;
  }

  // Sync within threadgroup
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduce remaining SIMD-group maxes in threadgroup memory
  uint simdgroup_count = threadgroup_size / simdgroup_size;
  float thread_group_max = 0.0f;
  if (t_idx_in_tg < simdgroup_count) {
    thread_group_max = local_max_values[t_idx_in_tg];
    for (uint offset = simdgroup_count / 2; offset > 0; offset /= 2) {
      if (t_idx_in_tg < offset) {
        thread_group_max = max(thread_group_max, local_max_values[t_idx_in_tg + offset]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // Write final threadgroup max to output
  if (t_idx_in_tg == 0) {
    uint tg_idx = tg_pos_in_grid.y * tg_per_grid.x + tg_pos_in_grid.x;
    map_buffer[tg_idx] = thread_group_max;
  }
}

// Atomic helper function
// Source https://stackoverflow.com/questions/36663645/finding-the-minimum-and-maximum-value-within-a-metal-texture
static void atomic_uint_exchange_if_greater_than(volatile device atomic_uint* current,
                                                 uint candidate) {
  uint currentVal;
  do {
    currentVal = *reinterpret_cast<volatile device uint*>(current);
  } while (candidate > currentVal && !atomic_compare_exchange_weak_explicit(current, &currentVal, candidate, memory_order_relaxed, memory_order_relaxed));
}

// Pass 2 - reduce the local max values to find the global max
kernel void reduceMaxKernel(const device float* mapBuffer [[buffer(0)]],
                            device atomic_uint* resultBuffer [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
  float localMax = mapBuffer[gid];

  // Use the helper function to update global max
  atomic_uint_exchange_if_greater_than(resultBuffer, as_type<uint>(localMax));
}

// Normalise the texture by dividing each pixel by the global maximum
kernel void normaliseKernel(texture2d<float, access::read_write> inputTexture [[texture(0)]],
                            const device float* maxValueBuffer [[buffer(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= inputTexture.get_width() || gid.y >= inputTexture.get_height()) return;

  // Read the current pixel value
  float currentVal = inputTexture.read(gid).r;

  // Read the global maximum value
  float maxVal = maxValueBuffer[0];

  // Calculate the normalised value and write it back to the texture
    if (maxVal > 0.0f) {
        float normalisedVal = currentVal / maxVal;
        inputTexture.write(float4(normalisedVal, normalisedVal, normalisedVal, 1.0), gid);
    } else {
        inputTexture.write(float4(0.0, 0.0, 0.0, 1.0), gid); // Handle maxVal == 0
    }
}

// Pass 1 of Cimmino's algorithm:
kernel void cimminosReconstruction(const device float* reconstructedBuffer [[buffer(0)]], // The current reconstruction x^k
                                   const device float* sinogramBuffer_b [[buffer(1)]],
                                   const device int* offsetsBuffer_A [[buffer(2)]],
                                   const device int* colsBuffer_A [[buffer(3)]],
                                   const device float* valsBuffer_A [[buffer(4)]],
                                   constant float& totalWeightSum [[buffer(5)]],
                                   constant uint& numRays [[buffer(6)]],
                                   device atomic_float* updateBuffer [[buffer(7)]], // The next reconstruction x^(k+1)
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

  // Get component of b at ray index
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
kernel void applyUpdate(device float* reconstructedBuffer [[buffer(0)]],
                        const device float* updateBuffer [[buffer(1)]],
                        constant uint& numPixels [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= numPixels) return;

  // Apply the update to the reconstructed buffer
  reconstructedBuffer[gid] += updateBuffer[gid];
}

kernel void computeRelativeDifference(const device float* reconstructedBuffer [[buffer(0)]],
                                    const device float* phantomBuffer [[buffer(1)]],
                                    device atomic_float* differenceSumBuffer [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
  // Compute the difference between the reconstruction and the phantom
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
  float4 positions[4] = {float4(-1, -1, 0, 1), float4(1, -1, 0, 1), float4(-1, 1, 0, 1),
                         float4(1, 1, 0, 1)};
  float2 texCoords[4] = {float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1)};
  out.position = positions[vid];
  out.texCoord = texCoords[vid];
  return out;
}

// A fragment shader that samples from the image texture and maps it to a colourmap
// Source https://metaltutorial.com/Lesson%201%3A%20Hello%20Metal/3.%20Textures/
fragment float4 fragment_main(VertexOut in [[stage_in]],
                              texture2d<float> imageTexture [[texture(0)]],
                              texture2d<float> colourMapTexture [[texture(1)]]) {
  // Sample the image texture
  constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);

  float value = imageTexture.sample(s, in.texCoord).r;  // 0..1

  // Map single channel to colourmap texture
  float2 colourMapUV = float2(value, 0.5); 
  float4 colour = colourMapTexture.sample(s, colourMapUV);
  return colour;
}