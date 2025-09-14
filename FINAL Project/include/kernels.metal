// xcrun -sdk macosx metal -o metallibrary.ir -c kernels.metal
// xcrun -sdk macosx metallib -o metallibrary.metallib metallibrary.ir
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Geometry structure to hold imaging parameters
struct Geometry {
    int imageWidth;
    int imageHeight;
    int nAngles;
    int nDetectors;
};


// Pass 1 of Cimmino's algorithm: 
kernel void cimminosReconstruction(
    const device float* reconstructedBuffer    [[buffer(0)]],
    const device float* sinogramBuffer_b       [[buffer(1)]],
    const device int* offsetsBuffer_A     [[buffer(2)]],
    const device int* colsBuffer_A     [[buffer(3)]],
    const device float* valsBuffer_A     [[buffer(4)]],
    constant float& totalWeightSum       [[buffer(5)]],
    constant uint& numRays              [[buffer(6)]],
    device atomic_float* updateBuffer  [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint rayIndex = gid;
    if (rayIndex >= numRays) return;

    // Calculate the dot product <a_i, x>
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

    float scalar = 3.0f * (1.0f / totalWeightSum) * residual;

    // Back Project - Add this ray's contribution to the update buffer
    for (int i = rowStart; i < rowEnd; ++i) {
        int   pixelIndex = colsBuffer_A[i];
        float weight      = valsBuffer_A[i];
        float contribution = scalar * weight;

        // Atomically add the contribution to prevent race conditions
        atomic_fetch_add_explicit(&updateBuffer[pixelIndex], contribution, memory_order_relaxed);
    }
}

kernel void applyUpdate(
    device float* reconstructedBuffer               [[buffer(0)]],
    const device float* updateBuffer    [[buffer(1)]],
    constant uint& numPixels      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < numPixels) {
        reconstructedBuffer[gid] += updateBuffer[gid];
    }
}

// Calculate the squared difference between the reconstructed image and the phantom image
kernel void calculateSquaredDifference(
    const device float* reconstructedBuffer, 
    const device float* phantomBuffer,     
    device float* squaredDifferenceBuffer,  
    constant uint&       numPixels,
    uint gid [[thread_position_in_grid]])
{
    float difference = reconstructedBuffer[gid] - phantomBuffer[gid];
    squaredDifferenceBuffer[gid] = difference * difference;
}

// Vertex and Fragment shaders for rendering the texture with a colormap
struct VertexOut { float4 position [[position]]; float2 texCoord; };
vertex VertexOut vertex_main(uint vid [[vertex_id]]) {
    VertexOut out;
    float4 positions[4] = { float4(-1, -1, 0, 1), float4(1, -1, 0, 1), float4(-1, 1, 0, 1), float4(1, 1, 0, 1) };
    float2 texCoords[4] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1) };
    out.position = positions[vid];
    out.texCoord = texCoords[vid];
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]], texture2d<float> imageTexture [[texture(0)]], texture2d<float> viridisTexture [[texture(1)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);

    float value = imageTexture.sample(s, in.texCoord).r; // 0..1

    // Map single channel to colourmap texture
    float2 viridisUV = float2(value, 0.5); // y=0.5 for single-row 2D colormap
    float4 colour = viridisTexture.sample(s, viridisUV);
    return colour;
}

// Calculate the sinogram from the phantom image
// Simulate the projection of rays through the image/phantom
kernel void performScan(
                        device float* phantomBuffer [[buffer(0)]],
                        device float* sinogramBuffer [[buffer(1)]],
                        constant Geometry& geom [[buffer(2)]],
                        constant float& center_x [[buffer(3)]],
                        constant float& center_y [[buffer(4)]],
                        constant int& num_steps [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]
)
{
    // Each thread finds the detector and angle it is responsible for
    uint angle_idx = gid / geom.nDetectors;
    uint detector_idx = gid % geom.nDetectors;

    // Bounds check
    if (angle_idx >= geom.nAngles || detector_idx >= geom.nDetectors) return;

    // Calculate the ray direction and position
    float angle = M_PI_F * angle_idx / geom.nAngles;
    float dir_x = cos(angle), dir_y = sin(angle);
    float perp_dir_x = -dir_y, perp_dir_y = dir_x;
    float detector_pos = detector_idx - (geom.nDetectors / 2.0f);
    
    // Perform ray marching
    float accumulator = 0.0f;
    float step_size = 2.0f;
    
    float start_x = center_x + detector_pos * perp_dir_x;
    float start_y = center_y + detector_pos * perp_dir_y;

    for (int i = -num_steps / 2; i < num_steps / 2; ++i) {
        float t = i * step_size;
        int current_x = static_cast<uint>(start_x + t * dir_x);
        int current_y = static_cast<uint>(start_y + t * dir_y);
        
        // Check bounds before accessing the buffer
        if (current_x < geom.imageWidth && current_y < geom.imageHeight) {
            uint index = current_y * geom.imageWidth + current_x;
            accumulator += phantomBuffer[index];
        }
    }
    
    float finalValue = accumulator * step_size;
    sinogramBuffer[gid] = finalValue;
}
// kernel void performScan(
//                         texture2d<float, access::read> phantomTexture [[texture(0)]],
//                         device float* sinogramBuffer [[buffer(0)]],
//                         constant Geometry& geom [[buffer(1)]],
//                         uint gid [[thread_position_in_grid]]
// )
// {
//     // Each thread finds the detector and angle it is responsible for
//     uint angle_idx = gid / geom.nDetectors;
//     uint detector_idx = gid % geom.nDetectors;
//     if (angle_idx >= geom.nAngles || detector_idx >= geom.nDetectors) return;

//     // Calculate the ray angle and direction
//     float angle = M_PI_F * angle_idx / geom.nAngles;
//     float dir_x = cos(angle), dir_y = sin(angle);

//     // Calculate the perpendicular direction and starting point
//     float perp_dir_x = -dir_y, perp_dir_y = dir_x;
//     float detector_pos = detector_idx - (geom.nDetectors / 2.0f);
//     float center_x = geom.imageWidth / 2.0f, center_y = geom.imageHeight / 2.0f;
    
//     // Perform ray marching
//     float accumulator = 0.0f;
//     float step_size = 2.0f;
//     int num_steps = static_cast<int>(sqrt(float(geom.imageWidth * geom.imageWidth + geom.imageHeight * geom.imageHeight)) / step_size);
    
//     float start_x = center_x + detector_pos * perp_dir_x;
//     float start_y = center_y + detector_pos * perp_dir_y;

//     for (int i = -num_steps / 2; i < num_steps / 2; ++i) {
//         float t = i * step_size;
//         int current_x = static_cast<uint>(start_x + t * dir_x);
//         int current_y = static_cast<uint>(start_y + t * dir_y);
        
//         // Check bounds before accessing the buffer
//         if (current_x < geom.imageWidth && current_y < geom.imageHeight) {
//             float value = phantomTexture.read(uint2(current_x, current_y)).r;
//         accumulator += value;
//         }
//     }
    
//     float finalValue = accumulator * step_size;
//     sinogramBuffer[gid] = finalValue;
// }
        
/** Normalisation kernels **/
// Pass 1: Find the maximum value within each threadgroup
// https://stackoverflow.com/questions/36663645/finding-the-minimum-and-maximum-value-within-a-metal-texture
// kernel void findMaxPerThreadgroupKernel(
//     texture2d<float, access::read> input_texture [[texture(0)]],
//     device float* map_buffer [[buffer(0)]], // Intermediate buffer
//     uint2 t_pos_in_grid [[thread_position_in_grid]],
//     uint t_idx_in_tg [[thread_index_in_threadgroup]],
//     uint2 tg_pos_in_grid [[threadgroup_position_in_grid]],
//     uint2 tg_per_grid [[threadgroups_per_grid]]
// ) {
//     // Shared memory for local maximum values
//     threadgroup float localMaxValues[256];

//     float pixelValue = 0.0f;
//     if (t_pos_in_grid.x < input_texture.get_width() && t_pos_in_grid.y < input_texture.get_height()) {
//         pixelValue = input_texture.read(t_pos_in_grid).r;
//     }
//     localMaxValues[t_idx_in_tg] = pixelValue;

//     threadgroup_barrier(mem_flags::mem_threadgroup);

//     for (uint stride = 128; stride > 0; stride /= 2) {
//         if (t_idx_in_tg < stride) {
//             localMaxValues[t_idx_in_tg] = max(localMaxValues[t_idx_in_tg], localMaxValues[t_idx_in_tg + stride]);
//         }
//         threadgroup_barrier(mem_flags::mem_threadgroup);
//     }

//     if (t_idx_in_tg == 0) {
//         uint tg_idx = tg_pos_in_grid.y * tg_per_grid.x + tg_pos_in_grid.x;
//         map_buffer[tg_idx] = localMaxValues[0];
//     }
// }

// Source https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
kernel void findMaxPerThreadgroupKernel(
    texture2d<float, access::read> input_texture [[texture(0)]],
    device float* map_buffer [[buffer(0)]],    
    constant uint& threadgroup_size [[buffer(1)]],
    uint2 t_pos_in_grid [[thread_position_in_grid]],
    uint t_idx_in_tg [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_size [[threads_per_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint2 tg_pos_in_grid [[threadgroup_position_in_grid]],
    uint2 tg_per_grid [[threadgroups_per_grid]],
)
{
    threadgroup float local_max_values[256]; // Threadgroup shared memory

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
static void atomic_uint_exchange_if_greater_than(
    volatile device atomic_uint *current,
    uint candidate
) {
    uint currentVal;
    do {
        currentVal = *reinterpret_cast<volatile device uint*>(current);
    } while (candidate > currentVal && !atomic_compare_exchange_weak_explicit(current, &currentVal, candidate, memory_order_relaxed, memory_order_relaxed));
}

// Pass 2 - reduce the local max values to find the global max
kernel void reduceMaxKernel(
    const device float* mapBuffer [[buffer(0)]],
    device atomic_uint* resultBuffer [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    // Read the local max found by one of threadgroups in pass 1
    float localMax = mapBuffer[gid];
    
    // Use the helper function to update global max
    atomic_uint_exchange_if_greater_than(resultBuffer, as_type<uint>(localMax));
}

kernel void normaliseKernel(
    texture2d<float, access::read_write> inputTexture [[texture(0)]],
    const device atomic_uint* maxValueBuffer [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Check if the thread is within the bounds of the texture
    if (gid.x >= inputTexture.get_width() || gid.y >= inputTexture.get_height()) {
        return;
    }

    // Read the current pixel value
    float currentVal = inputTexture.read(gid).r;
    
    // Read the global maximum value
    float maxVal = as_type<float>(atomic_load_explicit(maxValueBuffer, memory_order_relaxed));
    
    // Calculate the normalised value and write it back to the texture
    if (maxVal > 1e-6) {
        float normalisedVal = currentVal / maxVal;
        inputTexture.write(float4(normalisedVal, 0.0, 0.0, 1.0), gid);
    }
}

// Alternative reduction kernel using simdgroup operations - to test

// kernel void
// reduce(const device int *input [[buffer(0)]],
// device atomic_int *output [[buffer(1)]],
// threadgroup int *ldata [[threadgroup(0)]],
// uint gid [[thread_position_in_grid]],
// uint lid [[thread_position_in_threadgroup]],
// uint lsize [[threads_per_threadgroup]],
// uint simd_size [[threads_per_simdgroup]],
// uint simd_lane_id [[thread_index_in_simdgroup]],
// uint simd_group_id [[simdgroup_index_in_threadgroup]])
// {
// // Perform the first level of reduction.
// // Read from device memory, write to threadgroup memory.
// int val = input[gid] + input[gid + lsize];
// for (uint s=lsize/simd_size; s>simd_size; s/=simd_size)
// {
// // Perform per-SIMD partial reduction.
// for (uint offset=simd_size/2; offset>0; offset/=2)
// val += simd_shuffle_down(val, offset);
// // Write per-SIMD partial reduction value to threadgroup memory.
// if (simd_lane_id == 0)
// ldata[simd_group_id] = val;
// // Wait for all partial reductions to complete.
// threadgroup_barrier(mem_flags::mem_threadgroup);
// val = (lid < s) ? ldata[lid] : 0;
// }
// // Perform final per-SIMD partial reduction to calculate
// // the threadgroup partial reduction result.
// for (uint offset=simd_size/2; offset>0; offset/=2)
// val += simd_shuffle_down(val, offset);
// // Atomically update the reduction result.
// if (lid == 0)
// atomic_fetch_add_explicit(output, val,
// memory_order_relaxed);
// }





// Simple swap function for Metal
// template<typename T>
// void swap(thread T& a, thread T& b) {
//     T temp = a;
//     a = b;
//     b = temp;
// }