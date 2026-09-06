#include "metalign/gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace metalign {
namespace {

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

__global__ void log_response_device(const float* image, int width, int height,
                                    float normalization, float* output) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;
    const int index = y * width + x;
    if (x == 0 || y == 0 || x + 1 == width || y + 1 == height) {
        output[index] = 0.0F;
        return;
    }
    const float center = image[index];
    output[index] = (4.0F * center - image[index - 1] - image[index + 1] -
                     image[index - width] - image[index + width]) * normalization;
}

__global__ void gaussian_row_device(const float* input, int width, int height,
                                    const float* kernel, int radius, float* output) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;
    float value = 0.0F;
    for (int tap = -radius; tap <= radius; ++tap) {
        const int sx = max(0, min(width - 1, x + tap));
        value = fmaf(input[y * width + sx], kernel[abs(tap)], value);
    }
    output[y * width + x] = value;
}

__global__ void gaussian_column_device(const float* input, int width, int height,
                                       const float* kernel, int radius, float* output) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;
    float value = 0.0F;
    for (int tap = -radius; tap <= radius; ++tap) {
        const int sy = max(0, min(height - 1, y + tap));
        value = fmaf(input[sy * width + x], kernel[abs(tap)], value);
    }
    output[y * width + x] = value;
}

__global__ void rgb_to_grayscale_device(const std::uint8_t* rgb, std::size_t pixels,
                                        float* output) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    if (index >= pixels) return;
    constexpr float red = 0.29899999499320984F;
    constexpr float green = 0.5870000123977661F;
    constexpr float blue = 0.11400000005960464F;
    const float r = static_cast<float>(rgb[index * 3U]);
    const float g = static_cast<float>(rgb[index * 3U + 1U]);
    const float b = static_cast<float>(rgb[index * 3U + 2U]);
    const float luminance = fmaf(b, blue, fmaf(r, red, g * green));
    const std::uint8_t code = static_cast<std::uint8_t>(
        static_cast<std::uint32_t>(luminance) & 0xffU);
    output[index] = static_cast<float>(code) / 255.0F;
}

__global__ void integer_decimate_device(const float* input, int input_width,
                                        int output_width, int output_height,
                                        int factor, float* output) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= output_width || y >= output_height) return;
    output[y * output_width + x] =
        input[(y * factor) * input_width + x * factor];
}

__global__ void upsample_highest_device(const float* input,
                                        int input_width,
                                        int input_height,
                                        int output_width,
                                        float* output) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= input_width || y >= input_height) return;
    const float top_left = input[y * input_width + x];
    const int output_x = x * 2;
    const int output_y = y * 2;
    output[output_y * output_width + output_x] = top_left;
    if (y + 1 < input_height) {
        const float bottom_left = input[(y + 1) * input_width + x];
        output[(output_y + 1) * output_width + output_x] =
            (top_left + bottom_left) * 0.5F;
    }
    if (x + 1 < input_width) {
        const float top_right = input[y * input_width + x + 1];
        output[output_y * output_width + output_x + 1] =
            (top_left + top_right) * 0.5F;
    }
    if (x + 1 < input_width && y + 1 < input_height) {
        const float top_right = input[y * input_width + x + 1];
        const float bottom_left = input[(y + 1) * input_width + x];
        const float bottom_right = input[(y + 1) * input_width + x + 1];
        output[(output_y + 1) * output_width + output_x + 1] =
            (((top_left + top_right) + bottom_left) + bottom_right) * 0.25F;
    }
}

// Clean-room CUDA reconstruction of
// gpu::log_locate_points_device(float const*, float const*, KeyPoint*, ...).
// The launch geometry, shared layout, arithmetic association and 36-byte ABI
// below come from the target sm_35 PTX plus runtime argument/output captures.
// A 16x16 block emits from its central 14x14 tile; blockIdx.z selects one of
// the three internal scale extrema and the three LoG planes are made in one
// pass directly from five Gaussian planes.
__global__ void log_locate_points_device(
    const float* gaussian, const float* sigma, GpuExtremum* output,
    std::uint32_t* counter, std::uint32_t capacity, int width, int height,
    std::uint64_t intervals, float minimum, float maximum, int octave,
    float sigma0, float border_factor, int row_offset) {
    extern __shared__ float buffer[];
    const int level0 = static_cast<int>(blockIdx.z);
    const int center_level = level0 + 1;
    const int tile_width = static_cast<int>(blockDim.x) - 2;
    const int tile_height = static_cast<int>(blockDim.y) - 2;
    const int x = tile_width * static_cast<int>(blockIdx.x) +
                  static_cast<int>(threadIdx.x);
    const int y = tile_height * static_cast<int>(blockIdx.y) +
                  static_cast<int>(threadIdx.y) + row_offset;
    const int lane = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    const int plane_stride = static_cast<int>(blockDim.x * blockDim.y);
    const int clamped_x = max(1, min(x, width - 2));
    const int clamped_y = max(1, min(y, height - 2));
    const std::size_t pixels = static_cast<std::size_t>(width) * height;

#pragma unroll
    for (int plane = 0; plane < 3; ++plane) {
        const int level = level0 + plane;
        const float* image = gaussian + static_cast<std::size_t>(level) * pixels;
        const int index = clamped_y * width + clamped_x;
        const float center = image[index];
        float response = center * 4.0F;
        response = response - image[index - 1];
        response = response - image[index + 1];
        response = response - image[index - width];
        response = response - image[index + width];
        const float level_sigma = sigma[level];
        buffer[plane * plane_stride + lane] =
            (level_sigma * level_sigma) * response;
    }
    __syncthreads();

    if (x >= width - 2 || y >= height - 2 || threadIdx.x == 0 ||
        threadIdx.x + 1 >= blockDim.x || threadIdx.y == 0 ||
        threadIdx.y + 1 >= blockDim.y)
        return;

    const float center = buffer[plane_stride + lane];
    const float sign = copysignf(1.0F, center);
    const float magnitude = center * sign;
    const float interval_count = static_cast<float>(intervals);
    if (magnitude <= (minimum * 0.5F) / interval_count) return;

#pragma unroll
    for (int dl = -1; dl <= 1; ++dl) {
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                if (dl == 0 && dy == 0 && dx == 0) continue;
                const float other = buffer[(dl + 1) * plane_stride + lane +
                                           dy * static_cast<int>(blockDim.x) + dx];
                if (magnitude <= sign * other) return;
            }
        }
    }

    const float left = buffer[plane_stride + lane - 1];
    const float right = buffer[plane_stride + lane + 1];
    const float up = buffer[plane_stride + lane - static_cast<int>(blockDim.x)];
    const float down = buffer[plane_stride + lane + static_cast<int>(blockDim.x)];
    const float previous = buffer[lane];
    const float next = buffer[2 * plane_stride + lane];
    const float gx = (right - left) * 0.5F;
    const float gy = (down - up) * 0.5F;
    const float gz = (next - previous) * 0.5F;
    const float twice_center = center + center;
    const float hxx = (right + left) - twice_center;
    const float hyy = (down + up) - twice_center;
    const float hzz = (next + previous) - twice_center;

    const int row = static_cast<int>(blockDim.x);
    const float hxy = (((buffer[plane_stride + lane + row + 1] -
                         buffer[plane_stride + lane + row - 1]) -
                        buffer[plane_stride + lane - row + 1]) +
                       buffer[plane_stride + lane - row - 1]) * 0.25F;
    const float hxz = (((buffer[2 * plane_stride + lane + 1] -
                         buffer[2 * plane_stride + lane - 1]) -
                        buffer[lane + 1]) + buffer[lane - 1]) * 0.25F;
    const float hyz = (((buffer[2 * plane_stride + lane + row] -
                         buffer[2 * plane_stride + lane - row]) -
                        buffer[lane + row]) + buffer[lane - row]) * 0.25F;

    const float cofactor_xx = hyy * hzz - hyz * hyz;
    const float cofactor_xy_neg = hzz * hxy - hxz * hyz;
    float determinant = hxx * cofactor_xx - hxy * cofactor_xy_neg;
    const float cofactor_xz = hxy * hyz - hyy * hxz;
    determinant = fmaf(hxz, cofactor_xz, determinant);
    if (determinant == 0.0F) return;
    const float inverse_determinant = __frcp_rn(determinant);

    const float xy_neg = hxz * hyz - hzz * hxy;
    float numerator_x = gy * xy_neg;
    numerator_x = fmaf(gx, cofactor_xx, numerator_x);
    numerator_x = fmaf(gz, cofactor_xz, numerator_x);
    const float solve_x = numerator_x * inverse_determinant;
    const float offset_x = -solve_x;

    numerator_x = gx * xy_neg;
    const float cofactor_yy = hxx * hzz - hxz * hxz;
    numerator_x = fmaf(gy, cofactor_yy, numerator_x);
    const float cofactor_yz = hxy * hxz - hyz * hxx;
    numerator_x = fmaf(gz, cofactor_yz, numerator_x);
    const float solve_y = numerator_x * inverse_determinant;
    const float offset_y = -solve_y;

    numerator_x = gy * cofactor_yz;
    numerator_x = fmaf(gx, cofactor_xz, numerator_x);
    const float cofactor_zz = hxx * hyy - hxy * hxy;
    numerator_x = fmaf(gz, cofactor_zz, numerator_x);
    const float solve_z = numerator_x * inverse_determinant;
    const float offset_z = -solve_z;

    if (fabsf(offset_x) > 1.0F || fabsf(offset_y) > 1.0F ||
        fabsf(offset_z) > 1.0F)
        return;

    float refined_response = gy * offset_y;
    refined_response = fmaf(gx, offset_x, refined_response);
    refined_response = fmaf(gz, offset_z, refined_response);
    refined_response = fmaf(refined_response, 0.5F, center);
    const float absolute_response = fabsf(refined_response);
    const float determinant_xy = hxx * hyy - hxy * hxy;
    if (absolute_response < minimum / interval_count || determinant_xy <= 0.0F)
        return;
    const float trace = hxx + hyy;
    if ((trace * trace) / determinant_xy >=
        ((maximum + 1.0F) * (maximum + 1.0F)) / maximum)
        return;

    const float refined_x = (static_cast<float>(x) - solve_x) + 0.5F;
    const float refined_y = (static_cast<float>(y) - solve_y) + 0.5F;
    const float exponent =
        (static_cast<float>(center_level) - solve_z) / interval_count;
    const float scale = powf(2.0F, exponent) * sigma0;
    const float border = scale * border_factor;
    if (refined_x < border || refined_y < border ||
        refined_x > static_cast<float>(width) - border ||
        refined_y > static_cast<float>(height) - border)
        return;

    const std::uint32_t output_index = atomicInc(counter, 0xffffffffU);
    if (output_index >= capacity) return;
    GpuExtremum& point = output[output_index];
    point.x = refined_x;
    point.y = refined_y;
    // point.z remains zero after the host-side cudaMemset.
    point.scale = scale;
    point.sign_or_orientation = -1.0F;
    point.response = interval_count * absolute_response;
    point.octave = static_cast<std::uint32_t>(octave);
    point.level = static_cast<std::uint32_t>(center_level);
    point.flag = refined_response > 0.0F ? 1U : 0U;
}

// CUDA 10.1 expf expansion embedded in the target sm_35 orientation PTX.
// CUDA 12 emits a different approximation whose few-ULP weight changes can
// move a refined orientation across an MLDB comparison boundary.
__device__ __forceinline__ float target_expf_legacy(float exponent) {
    const float scaled = exponent * __uint_as_float(0x3FB8AA3BU);
    float whole = 0.0F;
    asm("cvt.rzi.f32.f32 %0, %1;" : "=f"(whole) : "f"(scaled));
    float reduced = fmaf(whole, __uint_as_float(0xBF317200U), exponent);
    reduced = fmaf(whole, __uint_as_float(0xB5BFBE8EU), reduced);
    const float fraction = reduced * __uint_as_float(0x3FB8AA3BU);
    float fraction_power = 0.0F;
    float whole_power = 0.0F;
    asm("ex2.approx.ftz.f32 %0, %1;"
        : "=f"(fraction_power) : "f"(fraction));
    asm("ex2.approx.f32 %0, %1;"
        : "=f"(whole_power) : "f"(whole + 0.0F));
    float result = fraction_power * whole_power;
    if (exponent < -105.0F) result = 0.0F;
    if (exponent > 105.0F) result = __uint_as_float(0x7F800000U);
    return result;
}

__global__ void orientation_hist_device(
    const float* image, int width, int height,
    const FeaturePrimitive* points, int count,
    float* output, std::uint32_t* output_counts) {
    const int point_index = static_cast<int>(blockIdx.x);
    if (point_index >= count) return;
    // Target sm_35 PTX uses the lower 36 floats for atomic accumulation/local
    // maxima and the upper 36 for its directly convolved histogram.
    __shared__ float histogram[72];
    for (int bin = static_cast<int>(threadIdx.x); bin < 72; bin += blockDim.x)
        histogram[bin] = 0.0F;
    __syncthreads();
    const FeaturePrimitive point = points[point_index];
    const float sigma = 1.5F * point.scale;
    const int radius = static_cast<int>(fmaf(sigma, 3.0F, 0.5F));
    const int center_x = static_cast<int>(point.x);
    const int center_y = static_cast<int>(point.y);
    const int side = 2 * radius + 1;
    const int sample_count = side * side;
    const float weight_denominator = sigma * (sigma + sigma);
    for (int sample = static_cast<int>(threadIdx.x); sample < sample_count;
         sample += static_cast<int>(blockDim.x)) {
        const int dx = sample % side - radius;
        const int dy = sample / side - radius;
        const int squared_radius = dx * dx + dy * dy;
        if (squared_radius > radius * radius) continue;
        const int px = min(max(center_x + dx, 1), width - 2);
        const int py = min(max(center_y + dy, 1), height - 2);
        const float gx = image[py * width + px + 1] -
                         image[py * width + px - 1];
        const float gy = image[(py - 1) * width + px] -
                         image[(py + 1) * width + px];
        const float magnitude = sqrtf(fmaf(gx, gx, gy * gy));
        const float angle = atan2f(gy, gx);
        unsigned bin = static_cast<unsigned>(
            ((angle + 3.1415927410125732F) * 36.0F) /
                6.2831854820251465F +
            0.5F);
        if (bin > 35U) bin = 0U;
        const float exponent = -static_cast<float>(squared_radius) /
                               weight_denominator;
        atomicAdd(&histogram[bin], magnitude * target_expf_legacy(exponent));
    }
    __syncthreads();
    const int thread = static_cast<int>(threadIdx.x);
    if (thread < 36) {
        const float adjacent = histogram[(thread + 35) % 36] +
                               histogram[(thread + 1) % 36];
        const float adjacent_weighted = adjacent * 4.0F;
        const float center_weighted =
            fmaf(histogram[thread], 6.0F, adjacent_weighted);
        const float outer = histogram[(thread + 34) % 36] +
                            histogram[(thread + 2) % 36];
        histogram[thread + 36] = center_weighted + outer;
    }
    __syncthreads();
    if (thread < 36) {
        const float left = histogram[(thread + 35) % 36 + 36];
        const float center = histogram[thread + 36];
        const float right = histogram[(thread + 1) % 36 + 36];
        histogram[thread] = center > left && center > right ? center : 0.0F;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    float maximum = histogram[0];
    for (int bin = 1; bin < 36; ++bin)
        if (histogram[bin] > maximum) maximum = histogram[bin];
    const float threshold = maximum * 0.8F;
    std::uint32_t produced = 0;
    for (int bin = 0; bin < 36 && produced < 10U; ++bin) {
        const float peak = histogram[bin];
        if (peak < threshold || peak == 0.0F) continue;
        const float left = histogram[(bin + 35) % 36 + 36];
        const float right = histogram[(bin + 1) % 36 + 36];
        const float numerator = (left - right) * 0.5F;
        const float denominator = fmaf(peak, -2.0F, left + right);
        float refined = static_cast<float>(bin) + numerator / denominator;
        if (refined < 0.0F) refined += 36.0F;
        else if (!(refined < 36.0F)) refined -= 36.0F;
        output[static_cast<std::size_t>(point_index) * 10U + produced++] =
            ((refined + refined) * 3.1415927410125732F) / 36.0F -
            3.1415927410125732F;
    }
    output_counts[point_index] = produced;
}

__device__ __forceinline__ float target_sincos_component(
    float reduced, int quadrant) {
    const float squared = reduced * reduced;
    float coefficient = 0.0F;
    if ((quadrant & 1) != 0) {
        coefficient = fmaf(__uint_as_float(0x37CCF5CEU), squared,
                           __uint_as_float(0xBAB6061AU));
        coefficient = fmaf(coefficient, squared,
                           __uint_as_float(0x3D2AAAA5U));
        coefficient = fmaf(coefficient, squared, -0.5F);
        coefficient = fmaf(coefficient, squared, 1.0F);
    } else {
        coefficient = fmaf(__uint_as_float(0xB94CA1F9U), squared,
                           __uint_as_float(0x3C08839EU));
        coefficient = fmaf(coefficient, squared,
                           __uint_as_float(0xBE2AAAA3U));
        coefficient = fmaf(coefficient, squared, 0.0F);
        coefficient = fmaf(coefficient, reduced, reduced);
    }
    return (quadrant & 2) != 0 ? -coefficient : coefficient;
}

// CUDA 10.1's sincosf expansion embedded in the target sm_35 PTX.  MLDB
// angles are finite and lie in [-pi, pi], so the PTX's large-argument Payne-
// Hanek branch is unreachable here; the directly observed fast branch is
// reproduced with its exact binary32 constants and FMA order.
__device__ __forceinline__ void target_sincos_legacy(
    float angle, float* sine, float* cosine) {
    const int quadrant = __float2int_rn(
        angle * __uint_as_float(0x3F22F983U));
    const float negative_quadrant = -static_cast<float>(quadrant);
    float reduced = fmaf(negative_quadrant, __uint_as_float(0x3FC90FDAU),
                         angle);
    reduced = fmaf(negative_quadrant, __uint_as_float(0x33A22168U), reduced);
    reduced = fmaf(negative_quadrant, __uint_as_float(0x27C234C5U), reduced);
    *sine = target_sincos_component(reduced, quadrant);
    *cosine = target_sincos_component(reduced, quadrant + 1);
}

__device__ __forceinline__ float mldb_integral_cell(
    const float* integral, int cell, int grid, int cell_size) {
    const int cell_row = cell / grid;
    const int cell_column = cell % grid;
    const int bottom = (cell_row + 1) * cell_size - 1;
    const int right = (cell_column + 1) * cell_size - 1;
    float value = integral[bottom * 21 + right];
    if (cell_column != 0)
        value -= integral[bottom * 21 + cell_column * cell_size - 1];
    if (cell_row != 0)
        value -= integral[(cell_row * cell_size - 1) * 21 + right];
    if (cell_column != 0 && cell_row != 0)
        value += integral[(cell_row * cell_size - 1) * 21 +
                          cell_column * cell_size - 1];
    return value;
}

// Target sm_35 topology: one 64-thread block per oriented keypoint builds a
// 21x21x3 patch, performs row then column prefix sums in shared memory, and
// emits 90 cell values (5/9/16 cells times three channels).
__global__ void mldb_extract_values_device_recovered(
    const float* image, int width, int height,
    const FeaturePrimitive* points, int count, float* values) {
    const int point_index = static_cast<int>(blockIdx.x);
    if (point_index >= count) return;
    __shared__ float patch[3][21 * 21];
    const FeaturePrimitive point = points[point_index];
    float sine = 0.0F;
    float cosine = 0.0F;
    target_sincos_legacy(point.orientation, &sine, &cosine);
    const float sampling_scale = point.scale * 1.1F;
    const int border = static_cast<int>(point.scale + 0.5F);
    const int maximum_x = width - 1 - border;
    const int maximum_y = height - 1 - border;
    for (int sample = static_cast<int>(threadIdx.x); sample < 21 * 21;
         sample += static_cast<int>(blockDim.x)) {
        const int column = sample % 21 - 10;
        const int row = sample / 21 - 10;
        const float rotated_x = static_cast<float>(column) * cosine -
                                static_cast<float>(row) * sine;
        const float rotated_y = fmaf(
            sine, static_cast<float>(column),
            cosine * static_cast<float>(row));
        int px = static_cast<int>(fmaf(sampling_scale, rotated_x, point.x));
        int py = static_cast<int>(fmaf(sampling_scale, rotated_y, point.y));
        px = max(border, min(px, maximum_x));
        py = max(border, min(py, maximum_y));
        const int source = py * width + px;
        const float intensity = image[source];
        const float dx = image[source + border] - image[source - border];
        const float dy = image[source + border * width] -
                         image[source - border * width];
        patch[0][sample] = intensity;
        patch[1][sample] = fmaf(cosine, dx, sine * dy);
        patch[2][sample] = cosine * dy - sine * dx;
    }
    __syncthreads();

    const int thread = static_cast<int>(threadIdx.x);
    if (thread < 63) {
        const int channel = thread % 3;
        const int row = thread / 3;
        float* line = &patch[channel][row * 21];
        for (int column = 1; column < 21; ++column)
            line[column] = line[column - 1] + line[column];
    }
    __syncthreads();
    if (thread < 63) {
        const int channel = thread % 3;
        const int column = thread / 3;
        float* plane = patch[channel];
        for (int row = 1; row < 21; ++row)
            plane[row * 21 + column] =
                plane[(row - 1) * 21 + column] + plane[row * 21 + column];
    }
    __syncthreads();

    if (thread < 63) {
        constexpr int subset0[5]{0, 2, 4, 6, 8};
        constexpr int subset1[9]{1, 3, 5, 9, 12, 15, 19, 21, 23};
        constexpr int subset2[16]{0, 2, 4, 6, 7, 8, 10, 11,
                                  13, 14, 16, 17, 18, 20, 22, 24};
        const int channel = thread % 3;
        const int slot = thread / 3;
        float* output = values + static_cast<std::size_t>(point_index) * 90U;
        if (slot < 5)
            output[slot * 3 + channel] =
                mldb_integral_cell(patch[channel], subset0[slot], 3, 7);
        if (slot < 9)
            output[15 + slot * 3 + channel] =
                mldb_integral_cell(patch[channel], subset1[slot], 5, 4);
        if (slot < 16)
            output[42 + slot * 3 + channel] =
                mldb_integral_cell(patch[channel], subset2[slot], 5, 4);
    }
}

__device__ __forceinline__ void mldb_pair_for_bit(
    int bit, int& base, int& channel, int& first, int& second) {
    int value_count = 5;
    int pairs_per_channel = 10;
    base = 0;
    if (bit >= 30) {
        bit -= 30;
        value_count = 9;
        pairs_per_channel = 36;
        base = 15;
        if (bit >= 108) {
            bit -= 108;
            value_count = 16;
            pairs_per_channel = 120;
            base = 42;
        }
    }
    channel = bit / pairs_per_channel;
    int pair = bit - channel * pairs_per_channel;
    first = 0;
    while (pair >= value_count - first - 1) {
        pair -= value_count - first - 1;
        ++first;
    }
    second = first + 1 + pair;
}

__global__ void mldb_compare_values_device_recovered(
    const float* values, int count, std::uint8_t* descriptors) {
    const int point_index = static_cast<int>(blockIdx.x);
    const int byte = static_cast<int>(threadIdx.x);
    if (point_index >= count || byte >= 64) return;
    std::uint8_t output = 0;
    if (byte < 63) {
        const float* point_values = values +
            static_cast<std::size_t>(point_index) * 90U;
        for (int lane = 0; lane < 8; ++lane) {
            const int bit = byte * 8 + lane;
            if (bit >= 498) break;
            int base = 0;
            int channel = 0;
            int first = 0;
            int second = 0;
            mldb_pair_for_bit(bit, base, channel, first, second);
            if (point_values[base + first * 3 + channel] >
                point_values[base + second * 3 + channel])
                output |= static_cast<std::uint8_t>(1U << lane);
        }
    }
    descriptors[static_cast<std::size_t>(point_index) * kDescriptorSize + byte] =
        output;
}

// Recovered target specialization:
// gpu::matchUnrolledCached<16, 16, HammingDist, int>
// grid=(ceil(query_count/16),1,1), block=(16,16,1), shared footprint=2048.
// Each x lane owns one candidate from every 16-row target tile; each y lane
// owns one query.  Reduction deliberately chooses the right-hand best on an
// exact tie, matching the PTX setp.lt/else sequence.  A tie necessarily fails
// Lowe's strict ratio test, but retaining the ordering makes the kernel exact.
__global__ void match_unrolled_cached_16x16_hamming_i32(
    const std::uint32_t* queries, int query_count,
    const std::uint32_t* targets, int target_count, int dimension,
    std::int32_t* output, float ratio, int output_offset) {
    __shared__ std::uint32_t query_tile[16][16];
    __shared__ std::uint32_t target_tile[16][16];

    const int lane = static_cast<int>(threadIdx.x);
    const int query_lane = static_cast<int>(threadIdx.y);
    const int query = static_cast<int>(blockIdx.x) * 16 + query_lane;
    const int clamped_query = max(0, min(query, query_count - 1));
    query_tile[query_lane][lane] = lane < dimension
        ? queries[static_cast<std::size_t>(clamped_query) * dimension + lane]
        : 0U;

    float best = FLT_MAX;
    float second = FLT_MAX;
    int best_index = -1;
    int second_index = -1;
    for (int base = 0; base < target_count; base += 16) {
        // Target PTX uses y as the target row loader and x as the component,
        // then stores transposed.  This makes each half-warp issue contiguous
        // 64-byte descriptor loads instead of striding by 64 bytes.
        const int loaded_target = base + query_lane;
        target_tile[lane][query_lane] =
            loaded_target < target_count && lane < dimension
            ? targets[static_cast<std::size_t>(loaded_target) * dimension + lane]
            : 0U;
        __syncthreads();

        const int target = base + lane;
        if (target < target_count) {
            unsigned distance = 0;
#pragma unroll
            for (int component = 0; component < 16; ++component)
                distance += static_cast<unsigned>(__popc(
                    query_tile[query_lane][component] ^
                    target_tile[component][lane]));
            const float value = static_cast<float>(distance);
            if (value < best) {
                second = best;
                second_index = best_index;
                best = value;
                best_index = target;
            } else if (value < second) {
                second = value;
                second_index = target;
            }
        }
        __syncthreads();
    }

    constexpr unsigned full_mask = 0xffffffffU;
    for (int offset = 8; offset != 0; offset >>= 1) {
        const float right_best = __shfl_down_sync(full_mask, best, offset, 16);
        const float right_second = __shfl_down_sync(full_mask, second, offset, 16);
        const int right_best_index =
            __shfl_down_sync(full_mask, best_index, offset, 16);
        const int right_second_index =
            __shfl_down_sync(full_mask, second_index, offset, 16);
        if (best < right_best) {
            if (right_best < second) {
                second = right_best;
                second_index = right_best_index;
            }
        } else {
            const float left_best = best;
            const int left_best_index = best_index;
            best = right_best;
            best_index = right_best_index;
            if (right_second < left_best) {
                second = right_second;
                second_index = right_second_index;
            } else {
                second = left_best;
                second_index = left_best_index;
            }
        }
    }

    if (lane == 0 && query >= 0 && query < query_count)
        output[query] = best < second * ratio ? best_index + output_offset : -1;
}

std::vector<std::uint32_t> flatten(const std::vector<Keypoint>& keypoints) {
    static_assert(kDescriptorSize == 16 * sizeof(std::uint32_t));
    std::vector<std::uint32_t> result(keypoints.size() * 16);
    for (std::size_t i = 0; i < keypoints.size(); ++i)
        std::memcpy(result.data() + i * 16,
                    keypoints[i].descriptor.data(), kDescriptorSize);
    return result;
}

unsigned hamming_distance(const Keypoint& first, const Keypoint& second) {
    unsigned result = 0;
    for (std::size_t i = 0; i < kDescriptorSize; i += sizeof(std::uint64_t)) {
        std::uint64_t a = 0;
        std::uint64_t b = 0;
        std::memcpy(&a, first.descriptor.data() + i, sizeof(a));
        std::memcpy(&b, second.descriptor.data() + i, sizeof(b));
        result += static_cast<unsigned>(__builtin_popcountll(a ^ b));
    }
    return result;
}

template <class T>
void ensure_cuda_buffer(T*& buffer, std::size_t& capacity, std::size_t count,
                        const char* operation) {
    if (count <= capacity) return;
    if (buffer) check_cuda(cudaFree(buffer), "cudaFree grow buffer");
    buffer = nullptr;
    capacity = 0;
    check_cuda(cudaMalloc(&buffer, count * sizeof(T)), operation);
    capacity = count;
}

class CudaAccelerator final : public DescriptorAccelerator {
public:
    explicit CudaAccelerator(int device) : device_(device) {
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        name_ = properties.name;
    }

    ~CudaAccelerator() override {
        cudaSetDevice(device_);
        if (output_) cudaFree(output_);
        if (targets_) cudaFree(targets_);
        if (queries_) cudaFree(queries_);
        if (feature_values_) cudaFree(feature_values_);
        if (feature_descriptors_) cudaFree(feature_descriptors_);
        if (feature_counts_) cudaFree(feature_counts_);
        if (feature_orientations_) cudaFree(feature_orientations_);
        if (feature_points_) cudaFree(feature_points_);
        if (feature_output_) cudaFree(feature_output_);
        if (blur_output_) cudaFree(blur_output_);
        if (gaussian_kernel_) cudaFree(gaussian_kernel_);
        if (rgb_input_) cudaFree(rgb_input_);
        if (feature_image_) cudaFree(feature_image_);
        if (extrema_gaussian_) cudaFree(extrema_gaussian_);
        if (extrema_sigma_) cudaFree(extrema_sigma_);
        if (extrema_output_) cudaFree(extrema_output_);
        if (extrema_counter_) cudaFree(extrema_counter_);
        if (resident_gray_) cudaFree(resident_gray_);
        for (float* octave : resident_octaves_)
            if (octave) cudaFree(octave);
    }

    std::string backend_name() const override { return "cuda-target"; }
    std::string device_name() const override { return name_; }
    bool supports_feature_extraction() const override { return true; }
    bool supports_extrema_detection() const override { return true; }
    bool supports_device_gaussian() const override { return true; }
    bool supports_device_grayscale() const override { return true; }
    bool supports_resident_feature_pipeline() const override { return true; }

    std::vector<ResidentFeatureOctave> begin_resident_feature_image(
        const Image& image, int downscale) override {
        mutex_.lock();
        try {
            if (resident_active_)
                throw std::runtime_error("CUDA resident feature session already active");
            if (downscale != 0 && downscale != 1 && downscale != 2 &&
                downscale != 4 && downscale != 8)
                throw std::runtime_error("CUDA resident feature downscale is invalid");
            check_cuda(cudaSetDevice(device_), "cudaSetDevice");
            resident_owner_ = std::this_thread::get_id();
            resident_active_ = true;
            resident_h2d_bytes_ = 0;
            resident_d2h_bytes_ = 0;
            resident_octave_count_ = 0;

            const std::size_t full_pixels = image.width * image.height;
            ensure_cuda_buffer(resident_gray_, resident_gray_capacity_, full_pixels,
                               "cudaMalloc resident grayscale");
            if (image.rgb.size() == full_pixels * 3U) {
                ensure_cuda_buffer(rgb_input_, rgb_input_capacity_, image.rgb.size(),
                                   "cudaMalloc resident RGB input");
                check_cuda(cudaMemcpy(rgb_input_, image.rgb.data(), image.rgb.size(),
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy resident RGB input");
                resident_h2d_bytes_ += image.rgb.size();
                constexpr unsigned block = 256;
                rgb_to_grayscale_device<<<
                    (static_cast<unsigned>(full_pixels) + block - 1U) / block, block>>>(
                        rgb_input_, full_pixels, resident_gray_);
                check_cuda(cudaGetLastError(), "resident RGB grayscale launch");
            } else if (image.gray.size() == full_pixels) {
                check_cuda(cudaMemcpy(resident_gray_, image.gray.data(),
                                      full_pixels * sizeof(float), cudaMemcpyHostToDevice),
                           "cudaMemcpy resident grayscale input");
                resident_h2d_bytes_ += full_pixels * sizeof(float);
            } else {
                throw std::runtime_error("CUDA resident feature image has no pixels");
            }

            std::size_t width = downscale == 0
                ? image.width * 2U - 1U
                : (image.width + static_cast<std::size_t>(downscale) - 1U) /
                      static_cast<std::size_t>(downscale);
            std::size_t height = downscale == 0
                ? image.height * 2U - 1U
                : (image.height + static_cast<std::size_t>(downscale) - 1U) /
                      static_cast<std::size_t>(downscale);
            ensure_cuda_buffer(blur_output_, blur_output_capacity_, width * height,
                               "cudaMalloc resident decimation scratch");
            const float* initial_input = resident_gray_;
            if (downscale == 0) {
                launch_upsample_highest(resident_gray_, image.width, image.height,
                                        width, blur_output_);
                initial_input = blur_output_;
            } else if (downscale > 1) {
                launch_decimate(resident_gray_, image.width, width, height, downscale,
                                blur_output_);
                initial_input = blur_output_;
            }

            std::vector<ResidentFeatureOctave> result;
            result.reserve(resident_octaves_.size());
            for (int octave = 0; octave < static_cast<int>(resident_octaves_.size()) &&
                                 width >= 32U && height >= 32U; ++octave) {
                const std::size_t pixels = width * height;
                ensure_cuda_buffer(
                    resident_octaves_[static_cast<std::size_t>(octave)],
                    resident_octave_capacities_[static_cast<std::size_t>(octave)],
                    pixels * 5U, "cudaMalloc resident Gaussian octave");
                float* planes = resident_octaves_[static_cast<std::size_t>(octave)];
                resident_widths_[static_cast<std::size_t>(octave)] = width;
                resident_heights_[static_cast<std::size_t>(octave)] = height;
                if (octave == 0) {
                    constexpr float sigma0 = 1.6F;
                    const float inherited_sigma = downscale == 0 ? 1.0F : 0.5F;
                    const double sigma0_double = static_cast<double>(sigma0);
                    launch_gaussian(
                        initial_input, width, height,
                        std::sqrt(sigma0_double * sigma0_double -
                                  static_cast<double>(inherited_sigma * inherited_sigma)),
                        planes);
                }
                constexpr float scale_step = 1.2599210498948732F;
                float previous_sigma = 1.600000023841858F;
                for (int level = 1; level < 5; ++level) {
                    const float sigma = previous_sigma * scale_step;
                    launch_gaussian(
                        planes + static_cast<std::size_t>(level - 1) * pixels,
                        width, height,
                        std::sqrt(sigma * sigma - previous_sigma * previous_sigma),
                        planes + static_cast<std::size_t>(level) * pixels);
                    previous_sigma = sigma;
                }
                ResidentFeatureOctave host;
                host.width = width;
                host.height = height;
                host.extrema = locate_resident_extrema(planes, width, height, octave);
                result.push_back(std::move(host));
                ++resident_octave_count_;

                const std::size_t next_width = (width + 1U) / 2U;
                const std::size_t next_height = (height + 1U) / 2U;
                if (octave + 1 < static_cast<int>(resident_octaves_.size()) &&
                    next_width >= 32U && next_height >= 32U) {
                    ensure_cuda_buffer(
                        resident_octaves_[static_cast<std::size_t>(octave + 1)],
                        resident_octave_capacities_[static_cast<std::size_t>(octave + 1)],
                        next_width * next_height * 5U,
                        "cudaMalloc resident Gaussian octave");
                    launch_decimate(planes + 3U * pixels, width, next_width,
                                    next_height, 2,
                                    resident_octaves_[static_cast<std::size_t>(octave + 1)]);
                }
                width = next_width;
                height = next_height;
            }
            return result;
        } catch (...) {
            resident_active_ = false;
            resident_owner_ = {};
            mutex_.unlock();
            throw;
        }
    }

    std::vector<std::vector<float>> resident_orientation_peaks(
        int octave, int level,
        std::span<const FeaturePrimitive> points) override {
        validate_resident_layer(octave, level);
        std::vector<std::vector<float>> result(points.size());
        if (points.empty()) return result;
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        ensure_cuda_buffer(feature_points_, feature_point_capacity_, points.size(),
                           "cudaMalloc resident feature points");
        ensure_cuda_buffer(feature_orientations_, feature_orientation_capacity_,
                           points.size() * 10U, "cudaMalloc resident orientations");
        ensure_cuda_buffer(feature_counts_, feature_count_capacity_, points.size(),
                           "cudaMalloc resident orientation counts");
        check_cuda(cudaMemcpy(feature_points_, points.data(), points.size_bytes(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy resident orientation points");
        resident_h2d_bytes_ += points.size_bytes();
        const std::size_t slot = static_cast<std::size_t>(octave);
        const std::size_t pixels = resident_widths_[slot] * resident_heights_[slot];
        const float* plane = resident_octaves_[slot] +
            static_cast<std::size_t>(level) * pixels;
        orientation_hist_device<<<static_cast<unsigned>(points.size()), 128>>>(
            plane, static_cast<int>(resident_widths_[slot]),
            static_cast<int>(resident_heights_[slot]), feature_points_,
            static_cast<int>(points.size()), feature_orientations_, feature_counts_);
        check_cuda(cudaGetLastError(), "resident orientation histogram launch");
        std::vector<float> values(points.size() * 10U);
        std::vector<std::uint32_t> counts(points.size());
        check_cuda(cudaMemcpy(values.data(), feature_orientations_,
                              values.size() * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy resident orientations");
        check_cuda(cudaMemcpy(counts.data(), feature_counts_,
                              counts.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy resident orientation counts");
        resident_d2h_bytes_ += values.size() * sizeof(float) +
                               counts.size() * sizeof(std::uint32_t);
        for (std::size_t index = 0; index < points.size(); ++index)
            result[index].assign(values.begin() + static_cast<std::ptrdiff_t>(index * 10U),
                                 values.begin() + static_cast<std::ptrdiff_t>(
                                     index * 10U + counts[index]));
        return result;
    }

    std::vector<Descriptor> resident_mldb_descriptors(
        int octave, int level,
        std::span<const FeaturePrimitive> points) override {
        validate_resident_layer(octave, level);
        std::vector<Descriptor> result(points.size());
        if (points.empty()) return result;
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        ensure_cuda_buffer(feature_points_, feature_point_capacity_, points.size(),
                           "cudaMalloc resident feature points");
        ensure_cuda_buffer(feature_descriptors_, feature_descriptor_capacity_,
                           points.size() * kDescriptorSize,
                           "cudaMalloc resident feature descriptors");
        ensure_cuda_buffer(feature_values_, feature_value_capacity_,
                           points.size() * 90U,
                           "cudaMalloc resident MLDB values");
        check_cuda(cudaMemcpy(feature_points_, points.data(), points.size_bytes(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy resident descriptor points");
        resident_h2d_bytes_ += points.size_bytes();
        const std::size_t slot = static_cast<std::size_t>(octave);
        const std::size_t pixels = resident_widths_[slot] * resident_heights_[slot];
        const float* plane = resident_octaves_[slot] +
            static_cast<std::size_t>(level) * pixels;
        constexpr unsigned block = 64;
        mldb_extract_values_device_recovered<<<
            static_cast<unsigned>(points.size()), block>>>(
            plane, static_cast<int>(resident_widths_[slot]),
            static_cast<int>(resident_heights_[slot]), feature_points_,
            static_cast<int>(points.size()), feature_values_);
        check_cuda(cudaGetLastError(), "resident MLDB extract-values launch");
        mldb_compare_values_device_recovered<<<
            static_cast<unsigned>(points.size()), block>>>(
            feature_values_, static_cast<int>(points.size()), feature_descriptors_);
        check_cuda(cudaGetLastError(), "resident MLDB compare-values launch");
        check_cuda(cudaMemcpy(result.data(), feature_descriptors_,
                              result.size() * sizeof(Descriptor), cudaMemcpyDeviceToHost),
                   "cudaMemcpy resident feature descriptors");
        resident_d2h_bytes_ += result.size() * sizeof(Descriptor);
        return result;
    }

    Image resident_feature_level(int octave, int level) override {
        validate_resident_layer(octave, level);
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        const std::size_t slot = static_cast<std::size_t>(octave);
        const std::size_t pixels = resident_widths_[slot] * resident_heights_[slot];
        Image result;
        result.width = resident_widths_[slot];
        result.height = resident_heights_[slot];
        result.gray.resize(pixels);
        check_cuda(cudaMemcpy(
            result.gray.data(),
            resident_octaves_[slot] + static_cast<std::size_t>(level) * pixels,
            pixels * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy resident Gaussian diagnostic");
        resident_d2h_bytes_ += pixels * sizeof(float);
        return result;
    }

    void end_resident_feature_image() override {
        if (!resident_active_ || resident_owner_ != std::this_thread::get_id())
            throw std::runtime_error("CUDA resident feature session owner mismatch");
        try {
            check_cuda(cudaSetDevice(device_), "cudaSetDevice");
            check_cuda(cudaDeviceSynchronize(), "CUDA resident feature synchronize");
            if (const char* value = std::getenv("METALIGN_GPU_TRANSFER_STATS");
                value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0) {
                std::fprintf(stderr,
                             "GPU transfer stats backend=cuda device=%d octaves=%zu "
                             "h2d_bytes=%zu d2h_bytes=%zu\n",
                             device_, resident_octave_count_, resident_h2d_bytes_,
                             resident_d2h_bytes_);
            }
        } catch (...) {
            resident_active_ = false;
            resident_owner_ = {};
            mutex_.unlock();
            throw;
        }
        resident_active_ = false;
        resident_owner_ = {};
        mutex_.unlock();
    }

    Image grayscale(const Image& image) override {
        if (image.rgb.size() != image.width * image.height * 3U) return image;
        std::lock_guard lock(mutex_);
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        const std::size_t pixels = image.width * image.height;
        ensure_cuda_buffer(rgb_input_, rgb_input_capacity_, image.rgb.size(),
                           "cudaMalloc RGB input");
        ensure_cuda_buffer(feature_output_, feature_output_capacity_, pixels,
                           "cudaMalloc grayscale output");
        check_cuda(cudaMemcpy(rgb_input_, image.rgb.data(), image.rgb.size(),
                              cudaMemcpyHostToDevice), "cudaMemcpy RGB input");
        constexpr unsigned block = 256;
        rgb_to_grayscale_device<<<
            (static_cast<unsigned>(pixels) + block - 1U) / block, block>>>(
                rgb_input_, pixels, feature_output_);
        check_cuda(cudaGetLastError(), "RGB grayscale launch");
        Image result = image;
        result.gray.resize(pixels);
        check_cuda(cudaMemcpy(result.gray.data(), feature_output_,
                              pixels * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy grayscale output");
        return result;
    }

    Image gaussian_blur(const Image& image, double sigma) override {
        if (image.empty() || sigma <= 0.0) return image;
        std::lock_guard lock(mutex_);
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        const int radius = std::max(1, static_cast<int>(4.0 * sigma + 1.0));
        std::vector<float> kernel(static_cast<std::size_t>(radius + 1));
        const float sigma_squared = static_cast<float>(sigma * sigma);
        const double exponent = -0.5 / static_cast<double>(sigma_squared);
        double sum = -1.0;
        for (int index = 0; index <= radius; ++index) {
            const float value = static_cast<float>(std::exp(exponent * index * index));
            kernel[static_cast<std::size_t>(index)] = value;
            sum += static_cast<double>(value + value);
        }
        for (float& value : kernel)
            value = static_cast<float>(static_cast<double>(value) / sum);

        ensure_cuda_buffer(feature_image_, feature_image_capacity_, image.gray.size(),
                           "cudaMalloc Gaussian input");
        ensure_cuda_buffer(feature_output_, feature_output_capacity_, image.gray.size(),
                           "cudaMalloc Gaussian row output");
        ensure_cuda_buffer(blur_output_, blur_output_capacity_, image.gray.size(),
                           "cudaMalloc Gaussian column output");
        ensure_cuda_buffer(gaussian_kernel_, gaussian_kernel_capacity_, kernel.size(),
                           "cudaMalloc Gaussian kernel");
        check_cuda(cudaMemcpy(feature_image_, image.gray.data(),
                              image.gray.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy Gaussian input");
        check_cuda(cudaMemcpy(gaussian_kernel_, kernel.data(),
                              kernel.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy Gaussian kernel");
        const dim3 block(16, 16, 1);
        const dim3 grid((static_cast<unsigned>(image.width) + 15U) / 16U,
                        (static_cast<unsigned>(image.height) + 15U) / 16U, 1);
        gaussian_row_device<<<grid, block>>>(
            feature_image_, static_cast<int>(image.width),
            static_cast<int>(image.height), gaussian_kernel_, radius, feature_output_);
        check_cuda(cudaGetLastError(), "Gaussian row launch");
        gaussian_column_device<<<grid, block>>>(
            feature_output_, static_cast<int>(image.width),
            static_cast<int>(image.height), gaussian_kernel_, radius, blur_output_);
        check_cuda(cudaGetLastError(), "Gaussian column launch");
        Image result = image;
        result.gray.resize(image.gray.size());
        check_cuda(cudaMemcpy(result.gray.data(), blur_output_,
                              result.gray.size() * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy Gaussian output");
        return result;
    }

    std::vector<GpuExtremum> locate_extrema(
        std::span<const Image> gaussian_levels, int octave) override {
        std::lock_guard lock(mutex_);
        if (gaussian_levels.size() != 5)
            throw std::runtime_error("CUDA extrema producer requires five Gaussian levels");
        const std::size_t width = gaussian_levels.front().width;
        const std::size_t height = gaussian_levels.front().height;
        const std::size_t pixels = width * height;
        for (const Image& image : gaussian_levels)
            if (image.width != width || image.height != height ||
                image.gray.size() != pixels)
                throw std::runtime_error("CUDA extrema Gaussian level shape mismatch");
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        ensure_cuda_buffer(extrema_gaussian_, extrema_gaussian_capacity_, pixels * 5U,
                           "cudaMalloc extrema Gaussian pyramid");
        ensure_cuda_buffer(extrema_sigma_, extrema_sigma_capacity_, 5U,
                           "cudaMalloc extrema sigma table");
        constexpr std::size_t capacity = 1'100'000U;
        ensure_cuda_buffer(extrema_output_, extrema_output_capacity_, capacity,
                           "cudaMalloc extrema output");
        ensure_cuda_buffer(extrema_counter_, extrema_counter_capacity_, 1U,
                           "cudaMalloc extrema counter");
        for (std::size_t level = 0; level < gaussian_levels.size(); ++level)
            check_cuda(cudaMemcpy(extrema_gaussian_ + level * pixels,
                                  gaussian_levels[level].gray.data(),
                                  pixels * sizeof(float), cudaMemcpyHostToDevice),
                       "cudaMemcpy extrema Gaussian level");
        const float sigma[5]{1.600000023841858F, 2.015873670578003F,
                             2.539841651916504F, 3.200000047683716F,
                             4.031747341156006F};
        check_cuda(cudaMemcpy(extrema_sigma_, sigma, sizeof(sigma),
                              cudaMemcpyHostToDevice), "cudaMemcpy extrema sigma");
        check_cuda(cudaMemset(extrema_output_, 0, capacity * sizeof(GpuExtremum)),
                   "cudaMemset extrema output");
        check_cuda(cudaMemset(extrema_counter_, 0, sizeof(std::uint32_t)),
                   "cudaMemset extrema counter");

        const unsigned grid_x = (static_cast<unsigned>(width - 2U) + 13U) / 14U;
        const std::size_t active_pixels = width * (height - 2U);
        const unsigned stripes = static_cast<unsigned>(
            std::max<std::size_t>(1U, (active_pixels + 699'999U) / 700'000U));
        const unsigned stripe_grid_y =
            (static_cast<unsigned>(height - 2U) + stripes * 14U - 1U) /
            (stripes * 14U);
        const dim3 block(16, 16, 1);
        for (unsigned row_offset = 0; row_offset < height - 2U;
             row_offset += stripe_grid_y * 14U) {
            const unsigned rows = static_cast<unsigned>(height - 2U) - row_offset;
            const unsigned grid_y = std::min(stripe_grid_y, (rows + 13U) / 14U);
            const dim3 grid(grid_x, grid_y, 3);
            log_locate_points_device<<<grid, block, 3U * 16U * 16U * sizeof(float)>>>(
                extrema_gaussian_, extrema_sigma_, extrema_output_, extrema_counter_,
                static_cast<std::uint32_t>(capacity), static_cast<int>(width),
                static_cast<int>(height), 3U, 0.0F, 10.0F, octave,
                1.600000023841858F, 7.0F, static_cast<int>(row_offset));
            check_cuda(cudaGetLastError(), "log locate points launch");
        }
        std::uint32_t count = 0;
        check_cuda(cudaMemcpy(&count, extrema_counter_, sizeof(count),
                              cudaMemcpyDeviceToHost), "cudaMemcpy extrema count");
        if (count > capacity)
            throw std::runtime_error("CUDA extrema output exceeded target capacity");
        std::vector<GpuExtremum> result(count);
        check_cuda(cudaMemcpy(result.data(), extrema_output_,
                              result.size() * sizeof(GpuExtremum),
                              cudaMemcpyDeviceToHost), "cudaMemcpy extrema output");
        return result;
    }

    Image laplacian_response(const Image& image, float sigma) override {
        std::lock_guard lock(mutex_);
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        ensure_cuda_buffer(feature_image_, feature_image_capacity_, image.gray.size(),
                           "cudaMalloc feature image");
        ensure_cuda_buffer(feature_output_, feature_output_capacity_, image.gray.size(),
                           "cudaMalloc LoG output");
        check_cuda(cudaMemcpy(feature_image_, image.gray.data(),
                              image.gray.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "cudaMemcpy feature image");
        const dim3 block(16, 16, 1);
        const dim3 grid((static_cast<unsigned>(image.width) + 15U) / 16U,
                        (static_cast<unsigned>(image.height) + 15U) / 16U, 1);
        log_response_device<<<grid, block>>>(
            feature_image_, static_cast<int>(image.width),
            static_cast<int>(image.height), sigma * sigma, feature_output_);
        check_cuda(cudaGetLastError(), "log response launch");
        Image result;
        result.width = image.width;
        result.height = image.height;
        result.gray.resize(image.gray.size());
        check_cuda(cudaMemcpy(result.gray.data(), feature_output_,
                              result.gray.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "cudaMemcpy LoG output");
        return result;
    }

    std::vector<std::vector<float>> orientation_peaks(
        const Image& image, std::span<const FeaturePrimitive> points) override {
        std::lock_guard lock(mutex_);
        std::vector<std::vector<float>> result(points.size());
        if (points.empty()) return result;
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        ensure_cuda_buffer(feature_image_, feature_image_capacity_, image.gray.size(),
                           "cudaMalloc feature image");
        ensure_cuda_buffer(feature_points_, feature_point_capacity_, points.size(),
                           "cudaMalloc feature points");
        ensure_cuda_buffer(feature_orientations_, feature_orientation_capacity_,
                           points.size() * 10U, "cudaMalloc orientations");
        ensure_cuda_buffer(feature_counts_, feature_count_capacity_, points.size(),
                           "cudaMalloc orientation counts");
        check_cuda(cudaMemcpy(feature_image_, image.gray.data(),
                              image.gray.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy feature image");
        check_cuda(cudaMemcpy(feature_points_, points.data(),
                              points.size_bytes(), cudaMemcpyHostToDevice),
                   "cudaMemcpy feature points");
        orientation_hist_device<<<static_cast<unsigned>(points.size()), 128>>>(
            feature_image_, static_cast<int>(image.width),
            static_cast<int>(image.height), feature_points_,
            static_cast<int>(points.size()), feature_orientations_, feature_counts_);
        check_cuda(cudaGetLastError(), "orientation histogram launch");
        std::vector<float> values(points.size() * 10U);
        std::vector<std::uint32_t> counts(points.size());
        check_cuda(cudaMemcpy(values.data(), feature_orientations_,
                              values.size() * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy orientations");
        check_cuda(cudaMemcpy(counts.data(), feature_counts_,
                              counts.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy orientation counts");
        for (std::size_t index = 0; index < points.size(); ++index)
            result[index].assign(values.begin() + static_cast<std::ptrdiff_t>(index * 10U),
                                 values.begin() + static_cast<std::ptrdiff_t>(
                                     index * 10U + counts[index]));
        return result;
    }

    std::vector<Descriptor> mldb_descriptors(
        const Image& image, std::span<const FeaturePrimitive> points) override {
        std::lock_guard lock(mutex_);
        std::vector<Descriptor> result(points.size());
        if (points.empty()) return result;
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        ensure_cuda_buffer(feature_image_, feature_image_capacity_, image.gray.size(),
                           "cudaMalloc feature image");
        ensure_cuda_buffer(feature_points_, feature_point_capacity_, points.size(),
                           "cudaMalloc feature points");
        ensure_cuda_buffer(feature_descriptors_, feature_descriptor_capacity_,
                           points.size() * kDescriptorSize,
                           "cudaMalloc feature descriptors");
        ensure_cuda_buffer(feature_values_, feature_value_capacity_,
                           points.size() * 90U,
                           "cudaMalloc MLDB values");
        check_cuda(cudaMemcpy(feature_image_, image.gray.data(),
                              image.gray.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy feature image");
        check_cuda(cudaMemcpy(feature_points_, points.data(), points.size_bytes(),
                              cudaMemcpyHostToDevice), "cudaMemcpy feature points");
        constexpr unsigned block = 64;
        mldb_extract_values_device_recovered<<<
            static_cast<unsigned>(points.size()), block>>>(
            feature_image_, static_cast<int>(image.width),
            static_cast<int>(image.height), feature_points_,
            static_cast<int>(points.size()), feature_values_);
        check_cuda(cudaGetLastError(), "MLDB extract-values launch");
        mldb_compare_values_device_recovered<<<
            static_cast<unsigned>(points.size()), block>>>(
            feature_values_, static_cast<int>(points.size()), feature_descriptors_);
        check_cuda(cudaGetLastError(), "MLDB compare-values launch");
        check_cuda(cudaMemcpy(result.data(), feature_descriptors_,
                              result.size() * sizeof(Descriptor), cudaMemcpyDeviceToHost),
                   "cudaMemcpy feature descriptors");
        return result;
    }

    std::vector<RatioMatchResult> ratio_matches(
        const std::vector<Keypoint>& queries,
        const std::vector<Keypoint>& targets,
        float ratio) override {
        const RatioMatchBatch batch{&queries, &targets};
        auto result = ratio_match_batches(std::span<const RatioMatchBatch>(&batch, 1), ratio);
        return std::move(result.front());
    }

    std::vector<std::vector<RatioMatchResult>> ratio_match_batches(
        std::span<const RatioMatchBatch> batches, float ratio) override {
        std::lock_guard lock(mutex_);
        std::vector<std::vector<RatioMatchResult>> result(batches.size());
        std::unordered_map<const std::vector<Keypoint>*, std::size_t> offsets;
        std::vector<std::uint32_t> packed;
        auto append_once = [&](const std::vector<Keypoint>* keypoints) {
            if (!keypoints) throw std::runtime_error("invalid CUDA descriptor match batch");
            const auto found = offsets.find(keypoints);
            if (found != offsets.end()) return found->second;
            const std::size_t offset = packed.size() / 16U;
            offsets.emplace(keypoints, offset);
            const std::vector<std::uint32_t> words = flatten(*keypoints);
            packed.insert(packed.end(), words.begin(), words.end());
            return offset;
        };
        struct Launch {
            std::size_t query_offset = 0;
            std::size_t target_offset = 0;
            std::size_t output_offset = 0;
        };
        std::vector<Launch> launches(batches.size());
        std::size_t output_count = 0;
        for (std::size_t index = 0; index < batches.size(); ++index) {
            const RatioMatchBatch& batch = batches[index];
            if (!batch.queries || !batch.targets)
                throw std::runtime_error("invalid CUDA descriptor match batch");
            result[index].assign(batch.queries->size(), {-1, 0.0});
            launches[index] = {
                append_once(batch.queries), append_once(batch.targets), output_count};
            output_count += batch.queries->size();
        }
        if (output_count == 0) return result;
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        ensure_cuda_buffer(queries_, query_capacity_, packed.size(),
                           "cudaMalloc packed descriptors");
        ensure_cuda_buffer(output_, output_capacity_, output_count,
                           "cudaMalloc output");
        check_cuda(cudaMemset(output_, 0xff, output_count * sizeof(std::int32_t)),
                   "cudaMemset batched output");
        if (!packed.empty()) {
            check_cuda(cudaMemcpy(queries_, packed.data(),
                                  packed.size() * sizeof(std::uint32_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy packed descriptors");
        }
        const dim3 block(16, 16, 1);
        for (std::size_t index = 0; index < batches.size(); ++index) {
            const RatioMatchBatch& batch = batches[index];
            if (batch.queries->empty() || batch.targets->size() < 2) continue;
            const Launch& launch = launches[index];
            const dim3 grid(
                (static_cast<unsigned>(batch.queries->size()) + 15U) / 16U, 1, 1);
            match_unrolled_cached_16x16_hamming_i32<<<grid, block>>>(
                queries_ + launch.query_offset * 16U,
                static_cast<int>(batch.queries->size()),
                queries_ + launch.target_offset * 16U,
                static_cast<int>(batch.targets->size()), 16,
                output_ + launch.output_offset, ratio, 0);
            check_cuda(cudaGetLastError(), "batched matchUnrolledCached launch");
        }
        std::vector<std::int32_t> host_output(output_count, -1);
        check_cuda(cudaMemcpy(host_output.data(), output_,
                              host_output.size() * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost), "cudaMemcpy output");
        for (std::size_t index = 0; index < batches.size(); ++index) {
            const RatioMatchBatch& batch = batches[index];
            const Launch& launch = launches[index];
            for (std::size_t query = 0; query < result[index].size(); ++query) {
                const std::int32_t target = host_output[launch.output_offset + query];
                if (target < 0) continue;
                result[index][query] = {target, static_cast<double>(hamming_distance(
                    (*batch.queries)[query],
                    (*batch.targets)[static_cast<std::size_t>(target)]))};
            }
        }
        return result;
    }

private:
    void launch_upsample_highest(const float* input, std::size_t input_width,
                                 std::size_t input_height,
                                 std::size_t output_width, float* output) {
        const dim3 block(16, 16, 1);
        const dim3 grid((static_cast<unsigned>(input_width) + 15U) / 16U,
                        (static_cast<unsigned>(input_height) + 15U) / 16U, 1);
        upsample_highest_device<<<grid, block>>>(
            input, static_cast<int>(input_width), static_cast<int>(input_height),
            static_cast<int>(output_width), output);
        check_cuda(cudaGetLastError(), "resident Highest upsample launch");
    }

    void launch_decimate(const float* input, std::size_t input_width,
                         std::size_t output_width, std::size_t output_height,
                         int factor, float* output) {
        const dim3 block(16, 16, 1);
        const dim3 grid((static_cast<unsigned>(output_width) + 15U) / 16U,
                        (static_cast<unsigned>(output_height) + 15U) / 16U, 1);
        integer_decimate_device<<<grid, block>>>(
            input, static_cast<int>(input_width), static_cast<int>(output_width),
            static_cast<int>(output_height), factor, output);
        check_cuda(cudaGetLastError(), "resident integer decimation launch");
    }

    void launch_gaussian(const float* input, std::size_t width,
                         std::size_t height, double sigma, float* output) {
        const int radius = std::max(1, static_cast<int>(4.0 * sigma + 1.0));
        std::vector<float> kernel(static_cast<std::size_t>(radius + 1));
        const float sigma_squared = static_cast<float>(sigma * sigma);
        const double exponent = -0.5 / static_cast<double>(sigma_squared);
        double sum = -1.0;
        for (int index = 0; index <= radius; ++index) {
            const float value = static_cast<float>(std::exp(exponent * index * index));
            kernel[static_cast<std::size_t>(index)] = value;
            sum += static_cast<double>(value + value);
        }
        for (float& value : kernel)
            value = static_cast<float>(static_cast<double>(value) / sum);
        ensure_cuda_buffer(feature_output_, feature_output_capacity_, width * height,
                           "cudaMalloc resident Gaussian scratch");
        ensure_cuda_buffer(gaussian_kernel_, gaussian_kernel_capacity_, kernel.size(),
                           "cudaMalloc resident Gaussian kernel");
        check_cuda(cudaMemcpy(gaussian_kernel_, kernel.data(),
                              kernel.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy resident Gaussian kernel");
        resident_h2d_bytes_ += kernel.size() * sizeof(float);
        const dim3 block(16, 16, 1);
        const dim3 grid((static_cast<unsigned>(width) + 15U) / 16U,
                        (static_cast<unsigned>(height) + 15U) / 16U, 1);
        gaussian_row_device<<<grid, block>>>(
            input, static_cast<int>(width), static_cast<int>(height),
            gaussian_kernel_, radius, feature_output_);
        check_cuda(cudaGetLastError(), "resident Gaussian row launch");
        gaussian_column_device<<<grid, block>>>(
            feature_output_, static_cast<int>(width), static_cast<int>(height),
            gaussian_kernel_, radius, output);
        check_cuda(cudaGetLastError(), "resident Gaussian column launch");
    }

    std::vector<GpuExtremum> locate_resident_extrema(
        const float* planes, std::size_t width, std::size_t height, int octave) {
        ensure_cuda_buffer(extrema_sigma_, extrema_sigma_capacity_, 5U,
                           "cudaMalloc resident extrema sigma table");
        constexpr std::size_t capacity = 1'100'000U;
        ensure_cuda_buffer(extrema_output_, extrema_output_capacity_, capacity,
                           "cudaMalloc resident extrema output");
        ensure_cuda_buffer(extrema_counter_, extrema_counter_capacity_, 1U,
                           "cudaMalloc resident extrema counter");
        const float sigma[5]{1.600000023841858F, 2.015873670578003F,
                             2.539841651916504F, 3.200000047683716F,
                             4.031747341156006F};
        check_cuda(cudaMemcpy(extrema_sigma_, sigma, sizeof(sigma),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy resident extrema sigma");
        resident_h2d_bytes_ += sizeof(sigma);
        check_cuda(cudaMemset(extrema_output_, 0, capacity * sizeof(GpuExtremum)),
                   "cudaMemset resident extrema output");
        check_cuda(cudaMemset(extrema_counter_, 0, sizeof(std::uint32_t)),
                   "cudaMemset resident extrema counter");
        const unsigned grid_x = (static_cast<unsigned>(width - 2U) + 13U) / 14U;
        const std::size_t active_pixels = width * (height - 2U);
        const unsigned stripes = static_cast<unsigned>(
            std::max<std::size_t>(1U, (active_pixels + 699'999U) / 700'000U));
        const unsigned stripe_grid_y =
            (static_cast<unsigned>(height - 2U) + stripes * 14U - 1U) /
            (stripes * 14U);
        const dim3 block(16, 16, 1);
        for (unsigned row_offset = 0; row_offset < height - 2U;
             row_offset += stripe_grid_y * 14U) {
            const unsigned rows = static_cast<unsigned>(height - 2U) - row_offset;
            const unsigned grid_y = std::min(stripe_grid_y, (rows + 13U) / 14U);
            const dim3 grid(grid_x, grid_y, 3);
            log_locate_points_device<<<grid, block,
                3U * 16U * 16U * sizeof(float)>>>(
                    planes, extrema_sigma_, extrema_output_, extrema_counter_,
                    static_cast<std::uint32_t>(capacity), static_cast<int>(width),
                    static_cast<int>(height), 3U, 0.0F, 10.0F, octave,
                    1.600000023841858F, 7.0F, static_cast<int>(row_offset));
            check_cuda(cudaGetLastError(), "resident log locate points launch");
        }
        std::uint32_t count = 0;
        check_cuda(cudaMemcpy(&count, extrema_counter_, sizeof(count),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy resident extrema count");
        resident_d2h_bytes_ += sizeof(count);
        if (count > capacity)
            throw std::runtime_error("CUDA resident extrema output exceeded capacity");
        std::vector<GpuExtremum> result(count);
        if (!result.empty()) {
            check_cuda(cudaMemcpy(result.data(), extrema_output_,
                                  result.size() * sizeof(GpuExtremum),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy resident extrema output");
            resident_d2h_bytes_ += result.size() * sizeof(GpuExtremum);
        }
        return result;
    }

    void validate_resident_layer(int octave, int level) const {
        if (!resident_active_ || resident_owner_ != std::this_thread::get_id())
            throw std::runtime_error("CUDA resident feature session owner mismatch");
        if (octave < 0 || octave >= static_cast<int>(resident_octave_count_) ||
            level < 0 || level >= 5)
            throw std::runtime_error("CUDA resident feature layer is out of range");
    }

    int device_ = 0;
    std::string name_;
    std::uint32_t* queries_ = nullptr;
    std::uint32_t* targets_ = nullptr;
    std::int32_t* output_ = nullptr;
    std::size_t query_capacity_ = 0;
    std::size_t target_capacity_ = 0;
    std::size_t output_capacity_ = 0;
    std::mutex mutex_;
    float* feature_image_ = nullptr;
    float* feature_output_ = nullptr;
    float* blur_output_ = nullptr;
    float* gaussian_kernel_ = nullptr;
    std::uint8_t* rgb_input_ = nullptr;
    FeaturePrimitive* feature_points_ = nullptr;
    float* feature_orientations_ = nullptr;
    std::uint32_t* feature_counts_ = nullptr;
    float* feature_values_ = nullptr;
    std::uint8_t* feature_descriptors_ = nullptr;
    std::size_t feature_image_capacity_ = 0;
    std::size_t feature_output_capacity_ = 0;
    std::size_t blur_output_capacity_ = 0;
    std::size_t gaussian_kernel_capacity_ = 0;
    std::size_t rgb_input_capacity_ = 0;
    std::size_t feature_point_capacity_ = 0;
    std::size_t feature_orientation_capacity_ = 0;
    std::size_t feature_count_capacity_ = 0;
    std::size_t feature_value_capacity_ = 0;
    std::size_t feature_descriptor_capacity_ = 0;
    float* extrema_gaussian_ = nullptr;
    float* extrema_sigma_ = nullptr;
    GpuExtremum* extrema_output_ = nullptr;
    std::uint32_t* extrema_counter_ = nullptr;
    std::size_t extrema_gaussian_capacity_ = 0;
    std::size_t extrema_sigma_capacity_ = 0;
    std::size_t extrema_output_capacity_ = 0;
    std::size_t extrema_counter_capacity_ = 0;
    float* resident_gray_ = nullptr;
    std::size_t resident_gray_capacity_ = 0;
    std::array<float*, 6> resident_octaves_{};
    std::array<std::size_t, 6> resident_octave_capacities_{};
    std::array<std::size_t, 6> resident_widths_{};
    std::array<std::size_t, 6> resident_heights_{};
    std::size_t resident_octave_count_ = 0;
    std::size_t resident_h2d_bytes_ = 0;
    std::size_t resident_d2h_bytes_ = 0;
    bool resident_active_ = false;
    std::thread::id resident_owner_{};
};

}  // namespace

std::vector<GpuDeviceInfo> enumerate_cuda_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return {};
    std::vector<GpuDeviceInfo> result;
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, index) == cudaSuccess)
            result.push_back({"cuda", index, properties.name,
                              static_cast<std::uint64_t>(properties.totalGlobalMem)});
    }
    return result;
}

std::unique_ptr<DescriptorAccelerator> create_cuda_accelerator(int device_index) {
    return std::make_unique<CudaAccelerator>(device_index);
}

}  // namespace metalign
