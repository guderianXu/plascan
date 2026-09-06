#define CL_TARGET_OPENCL_VERSION 120
#include "metalign/gpu.hpp"

#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace metalign {
namespace {

constexpr const char* kernel_source = R"CLC(
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void match_unrolled_cached_16x16_hamming_i32(
        __global const uint* queries, const int query_count, const int query_offset,
        __global const uint* targets, const int target_count, const int target_offset,
        const int dimension, __global int* output,
        const float ratio, const int output_offset) {
    __local uint query_tile[16][16];
    __local uint target_tile[16][16];
    __local float best_values[16][16];
    __local float second_values[16][16];
    __local int best_indices[16][16];
    __local int second_indices[16][16];

    const int lane = (int)get_local_id(0);
    const int query_lane = (int)get_local_id(1);
    const int query = (int)get_group_id(0) * 16 + query_lane;
    const int clamped_query = max(0, min(query, query_count - 1));
    query_tile[query_lane][lane] = lane < dimension
        ? queries[(query_offset + clamped_query) * dimension + lane] : 0U;
    float best = FLT_MAX;
    float second = FLT_MAX;
    int best_index = -1;
    int second_index = -1;
    for (int base = 0; base < target_count; base += 16) {
        const int loaded_target = base + query_lane;
        target_tile[lane][query_lane] =
            loaded_target < target_count && lane < dimension
            ? targets[(target_offset + loaded_target) * dimension + lane] : 0U;
        barrier(CLK_LOCAL_MEM_FENCE);
        const int target = base + lane;
        if (target < target_count) {
            uint distance = 0;
            for (int component = 0; component < 16; ++component)
                distance += popcount(query_tile[query_lane][component] ^
                                     target_tile[component][lane]);
            const float value = (float)distance;
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
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    best_values[query_lane][lane] = best;
    second_values[query_lane][lane] = second;
    best_indices[query_lane][lane] = best_index;
    second_indices[query_lane][lane] = second_index;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane == 0 && query < query_count) {
        float reduced_best = best_values[query_lane][0];
        float reduced_second = second_values[query_lane][0];
        int reduced_best_index = best_indices[query_lane][0];
        for (int right = 1; right < 16; ++right) {
            const float right_best = best_values[query_lane][right];
            const float right_second = second_values[query_lane][right];
            const int right_best_index = best_indices[query_lane][right];
            if (reduced_best < right_best) {
                if (right_best < reduced_second)
                    reduced_second = right_best;
            } else {
                const float left_best = reduced_best;
                reduced_best = right_best;
                reduced_best_index = right_best_index;
                reduced_second = right_second < left_best
                    ? right_second : left_best;
            }
        }
        output[output_offset + query] = reduced_best < reduced_second * ratio
            ? reduced_best_index : -1;
    }
}

typedef struct {
    float x;
    float y;
    float scale;
    float orientation;
} FeaturePrimitive;

__kernel void log_response_device(
        __global const float* image, const int width, const int height,
        const float normalization, __global float* output) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const int index = y * width + x;
    if (x == 0 || y == 0 || x + 1 == width || y + 1 == height) {
        output[index] = 0.0f;
        return;
    }
    output[index] = (4.0f * image[index] - image[index - 1] - image[index + 1] -
                     image[index - width] - image[index + width]) * normalization;
}

__kernel void gaussian_row_device(
        __global const float* input, const int width, const int height,
        __global const float* weights, const int radius, __global float* output) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    float value = 0.0f;
    for (int tap = -radius; tap <= radius; ++tap) {
        const int sx = max(0, min(width - 1, x + tap));
        value = fma(input[y * width + sx], weights[abs(tap)], value);
    }
    output[y * width + x] = value;
}

__kernel void gaussian_column_device(
        __global const float* input, const int width, const int height,
        __global const float* weights, const int radius, __global float* output) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    float value = 0.0f;
    for (int tap = -radius; tap <= radius; ++tap) {
        const int sy = max(0, min(height - 1, y + tap));
        value = fma(input[sy * width + x], weights[abs(tap)], value);
    }
    output[y * width + x] = value;
}

__kernel void rgb_to_grayscale_device(
        __global const uchar* rgb, const ulong pixels, __global float* output) {
    const ulong index = (ulong)get_global_id(0);
    if (index >= pixels) return;
    const float r = (float)rgb[index * 3UL];
    const float g = (float)rgb[index * 3UL + 1UL];
    const float b = (float)rgb[index * 3UL + 2UL];
    const float luminance = fma(b, as_float((uint)0x3de978d5U),
                                fma(r, as_float((uint)0x3e991687U),
                                    g * as_float((uint)0x3f1645a2U)));
    const uchar code = convert_uchar_rtz(luminance);
    output[index] = (float)code / 255.0f;
}

__kernel void integer_decimate_device(
        __global const float* input, const int input_width,
        const int output_width, const int output_height, const int factor,
        __global float* output) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= output_width || y >= output_height) return;
    output[y * output_width + x] =
        input[(y * factor) * input_width + x * factor];
}

__kernel void upsample_highest_device(
        __global const float* input, const int input_width,
        const int input_height, const int output_width,
        __global float* output) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= input_width || y >= input_height) return;
    const float top_left = input[y * input_width + x];
    const int output_x = x * 2;
    const int output_y = y * 2;
    output[output_y * output_width + output_x] = top_left;
    if (y + 1 < input_height) {
        const float bottom_left = input[(y + 1) * input_width + x];
        output[(output_y + 1) * output_width + output_x] =
            (top_left + bottom_left) * 0.5f;
    }
    if (x + 1 < input_width) {
        const float top_right = input[y * input_width + x + 1];
        output[output_y * output_width + output_x + 1] =
            (top_left + top_right) * 0.5f;
    }
    if (x + 1 < input_width && y + 1 < input_height) {
        const float top_right = input[y * input_width + x + 1];
        const float bottom_left = input[(y + 1) * input_width + x];
        const float bottom_right = input[(y + 1) * input_width + x + 1];
        output[(output_y + 1) * output_width + output_x + 1] =
            (((top_left + top_right) + bottom_left) + bottom_right) * 0.25f;
    }
}

typedef struct {
    float x, y, z, scale, sign_or_orientation, response;
    uint octave, level, flag;
} GpuExtremum;

// OpenCL form of the captured target locatePoints ABI.  Argument 15 is the
// observed 3072-byte local buffer.  As in CUDA, global sizes are rounded from
// 14x14 effective tiles while the required local size remains 16x16x1.
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void locate_extrema_device(
        __global const float* gaussian, __global const float* sigma,
        __global GpuExtremum* output, volatile __global uint* counter,
        const uint capacity, const int width, const int height,
        const ulong intervals, const float minimum, const float maximum,
        const int octave, const float sigma0, const float border_factor,
        const int row_offset, __local float* buffer) {
    const int level0 = (int)get_group_id(2);
    const int center_level = level0 + 1;
    const int x = 14 * (int)get_group_id(0) + (int)get_local_id(0);
    const int y = 14 * (int)get_group_id(1) + (int)get_local_id(1) + row_offset;
    const int lane = (int)get_local_id(1) * 16 + (int)get_local_id(0);
    const int pixels = width * height;
    const int clamped_x = max(1, min(x, width - 2));
    const int clamped_y = max(1, min(y, height - 2));
    for (int plane = 0; plane < 3; ++plane) {
        const int level = level0 + plane;
        __global const float* image = gaussian + level * pixels;
        const int index = clamped_y * width + clamped_x;
        float value = image[index] * 4.0f;
        value = value - image[index - 1];
        value = value - image[index + 1];
        value = value - image[index - width];
        value = value - image[index + width];
        buffer[plane * 256 + lane] = (sigma[level] * sigma[level]) * value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x >= width - 2 || y >= height - 2 || get_local_id(0) == 0 ||
        get_local_id(0) == 15 || get_local_id(1) == 0 || get_local_id(1) == 15)
        return;

    const float center = buffer[256 + lane];
    const float sign = copysign(1.0f, center);
    const float magnitude = center * sign;
    const float interval_count = (float)intervals;
    if (magnitude <= (minimum * 0.5f) / interval_count) return;
    for (int dl = -1; dl <= 1; ++dl)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dl == 0 && dy == 0 && dx == 0) continue;
                if (magnitude <= sign * buffer[(dl + 1) * 256 + lane + dy * 16 + dx])
                    return;
            }

    const float left = buffer[256 + lane - 1];
    const float right = buffer[256 + lane + 1];
    const float up = buffer[256 + lane - 16];
    const float down = buffer[256 + lane + 16];
    const float previous = buffer[lane];
    const float next = buffer[512 + lane];
    const float gx = (right - left) * 0.5f;
    const float gy = (down - up) * 0.5f;
    const float gz = (next - previous) * 0.5f;
    const float twice_center = center + center;
    const float hxx = (right + left) - twice_center;
    const float hyy = (down + up) - twice_center;
    const float hzz = (next + previous) - twice_center;
    const float hxy = (((buffer[256 + lane + 17] - buffer[256 + lane + 15]) -
                         buffer[256 + lane - 15]) + buffer[256 + lane - 17]) * 0.25f;
    const float hxz = (((buffer[512 + lane + 1] - buffer[512 + lane - 1]) -
                         buffer[lane + 1]) + buffer[lane - 1]) * 0.25f;
    const float hyz = (((buffer[512 + lane + 16] - buffer[512 + lane - 16]) -
                         buffer[lane + 16]) + buffer[lane - 16]) * 0.25f;
    const float cofactor_xx = hyy * hzz - hyz * hyz;
    const float cofactor_xy_neg = hzz * hxy - hxz * hyz;
    float determinant = hxx * cofactor_xx - hxy * cofactor_xy_neg;
    const float cofactor_xz = hxy * hyz - hyy * hxz;
    determinant = fma(hxz, cofactor_xz, determinant);
    if (determinant == 0.0f) return;
    const float inverse = 1.0f / determinant;
    const float xy_neg = hxz * hyz - hzz * hxy;
    float term = gy * xy_neg;
    term = fma(gx, cofactor_xx, term);
    term = fma(gz, cofactor_xz, term);
    const float solve_x = term * inverse;
    term = gx * xy_neg;
    term = fma(gy, hxx * hzz - hxz * hxz, term);
    const float cofactor_yz = hxy * hxz - hyz * hxx;
    term = fma(gz, cofactor_yz, term);
    const float solve_y = term * inverse;
    term = gy * cofactor_yz;
    term = fma(gx, cofactor_xz, term);
    const float cofactor_zz = hxx * hyy - hxy * hxy;
    term = fma(gz, cofactor_zz, term);
    const float solve_z = term * inverse;
    const float offset_x = -solve_x, offset_y = -solve_y, offset_z = -solve_z;
    if (fabs(offset_x) > 1.0f || fabs(offset_y) > 1.0f || fabs(offset_z) > 1.0f)
        return;
    float refined = gy * offset_y;
    refined = fma(gx, offset_x, refined);
    refined = fma(gz, offset_z, refined);
    refined = fma(refined, 0.5f, center);
    const float absolute_response = fabs(refined);
    const float determinant_xy = hxx * hyy - hxy * hxy;
    if (absolute_response < minimum / interval_count || determinant_xy <= 0.0f)
        return;
    const float trace = hxx + hyy;
    if ((trace * trace) / determinant_xy >=
        ((maximum + 1.0f) * (maximum + 1.0f)) / maximum) return;
    const float refined_x = ((float)x - solve_x) + 0.5f;
    const float refined_y = ((float)y - solve_y) + 0.5f;
    const float scale = pow(2.0f, ((float)center_level - solve_z) /
                            interval_count) * sigma0;
    const float border = scale * border_factor;
    if (refined_x < border || refined_y < border ||
        refined_x > (float)width - border || refined_y > (float)height - border)
        return;
    const uint index = atomic_inc(counter);
    if (index >= capacity) return;
    output[index].x = refined_x;
    output[index].y = refined_y;
    output[index].scale = scale;
    output[index].sign_or_orientation = -1.0f;
    output[index].response = interval_count * absolute_response;
    output[index].octave = (uint)octave;
    output[index].level = (uint)center_level;
    output[index].flag = refined > 0.0f ? 1U : 0U;
}

__kernel void orientation_peaks_device(
        __global const float* image, const int width, const int height,
        __global const FeaturePrimitive* points, const int count,
        __global float* output, __global uint* output_counts) {
    const int point_index = (int)get_global_id(0);
    if (point_index >= count) return;
    const FeaturePrimitive point = points[point_index];
    float histogram[36];
    for (int bin = 0; bin < 36; ++bin) histogram[bin] = 0.0f;
    const float sigma = 1.5f * point.scale;
    const int radius = min(20, (int)(3.0f * sigma + 0.5f));
    const int center_x = (int)point.x;
    const int center_y = (int)point.y;
    const int valid = center_x > radius && center_y > radius &&
        center_x + radius + 1 < width && center_y + radius + 1 < height;
    if (valid) {
        const float inverse_weight_variance = 1.0f / (sigma * (sigma + sigma));
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy > radius * radius) continue;
                const int px = center_x + dx;
                const int py = center_y + dy;
                const float gx = image[py * width + px + 1] - image[py * width + px - 1];
                const float gy = image[(py - 1) * width + px] - image[(py + 1) * width + px];
                const float magnitude = sqrt(gx * gx + gy * gy);
                const float angle = atan2(gy, gx) + 3.1415927410125732f;
                uint bin = (uint)(angle * 5.729578f + 0.5f);
                if (bin >= 36U) bin = 0U;
                const float wx = exp((float)(dx * -dx) * inverse_weight_variance);
                const float wy = exp((float)(dy * -dy) * inverse_weight_variance);
                histogram[bin] += wx * wy * magnitude;
            }
        }
    }
    uint produced = 0;
    if (valid) {
        float temporary[36];
        for (int pass = 0; pass < 2; ++pass) {
            for (int bin = 0; bin < 36; ++bin) temporary[bin] = histogram[bin];
            for (int bin = 0; bin < 36; ++bin)
                histogram[bin] = temporary[(bin + 35) % 36] * 0.25f +
                    temporary[bin] * 0.5f + temporary[(bin + 1) % 36] * 0.25f;
        }
        float maximum = histogram[0];
        for (int bin = 1; bin < 36; ++bin) maximum = fmax(maximum, histogram[bin]);
        const float threshold = maximum * 0.8f;
        for (int bin = 0; bin < 36 && produced < 10U; ++bin) {
            const float left = histogram[(bin + 35) % 36];
            const float center = histogram[bin];
            const float right = histogram[(bin + 1) % 36];
            if (center <= left || center <= right || center < threshold) continue;
            float refined = (float)bin + 0.5f * (left - right) /
                (left + right - 2.0f * center);
            if (refined < 0.0f) refined += 36.0f;
            if (refined >= 36.0f) refined -= 36.0f;
            output[point_index * 10 + produced++] =
                refined * 0.17453292f - 3.1415927410125732f;
        }
    }
    output_counts[point_index] = produced;
}

void mldb_compare_cells(
        float values[16][3], const int value_count,
        __global uchar* descriptor, int* bit) {
    for (int channel = 0; channel < 3; ++channel)
        for (int first = 0; first < value_count; ++first)
            for (int second = first + 1; second < value_count; ++second, ++(*bit))
                if (values[first][channel] > values[second][channel])
                    descriptor[*bit >> 3] |= (uchar)(1U << (*bit & 7));
}

void mldb_describe_level(
        __global const float* image, const int width, const int height,
        const FeaturePrimitive point, const int grid,
        const int selected[16], const int selected_count,
        __global uchar* descriptor, int* bit) {
    const int radius = 10;
    const int cell_size = (grid + 2 * radius - 1) / grid;
    const int border = (int)(point.scale + 0.5f);
    const int maximum_x = width - 1 - border;
    const int maximum_y = height - 1 - border;
    const float sine = sin(point.orientation);
    const float cosine = cos(point.orientation);
    const float sampling_scale = 1.1f * point.scale;
    float values[16][3];
    for (int index = 0; index < 16; ++index)
        for (int channel = 0; channel < 3; ++channel) values[index][channel] = 0.0f;
    for (int index = 0; index < selected_count; ++index) {
        const int cell = selected[index];
        const int row_start = cell_size * (cell / grid) - radius;
        const int column_start = cell_size * (cell % grid) - radius;
        for (int row = row_start; row < row_start + cell_size; ++row) {
            for (int column = column_start; column < column_start + cell_size; ++column) {
                int py = (int)(((float)column * sine + (float)row * cosine) *
                               sampling_scale + point.y);
                int px = (int)(((float)column * cosine - (float)row * sine) *
                               sampling_scale + point.x);
                py = max(border, min(py, maximum_y));
                px = max(border, min(px, maximum_x));
                const float center = image[py * width + px];
                const float dx = image[py * width + px + border] -
                                 image[py * width + px - border];
                const float dy = image[(py + border) * width + px] -
                                 image[(py - border) * width + px];
                values[index][0] += center;
                values[index][1] += cosine * dx + sine * dy;
                values[index][2] += cosine * dy - sine * dx;
            }
        }
    }
    mldb_compare_cells(values, selected_count, descriptor, bit);
}

__kernel void mldb_descriptor_device(
        __global const float* image, const int width, const int height,
        __global const FeaturePrimitive* points, const int count,
        __global uchar* descriptors) {
    const int point_index = (int)get_global_id(0);
    if (point_index >= count) return;
    __global uchar* descriptor = descriptors + point_index * 64;
    for (int byte = 0; byte < 64; ++byte) descriptor[byte] = 0;
    const int level0[16] = {0,2,4,6,8,0,0,0,0,0,0,0,0,0,0,0};
    const int level1[16] = {1,3,5,9,12,15,19,21,23,0,0,0,0,0,0,0};
    const int level2[16] = {0,2,4,6,7,8,10,11,13,14,16,17,18,20,22,24};
    int bit = 0;
    const FeaturePrimitive point = points[point_index];
    mldb_describe_level(image, width, height, point, 3, level0, 5, descriptor, &bit);
    mldb_describe_level(image, width, height, point, 5, level1, 9, descriptor, &bit);
    mldb_describe_level(image, width, height, point, 5, level2, 16, descriptor, &bit);
}
)CLC";

void check_cl(cl_int error, const char* operation) {
    if (error != CL_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with OpenCL error " +
                                 std::to_string(error));
}

std::vector<cl_device_id> opencl_gpu_devices() {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS) return {};
    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    std::vector<cl_device_id> devices;
    for (cl_platform_id platform : platforms) {
        cl_uint count = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count) != CL_SUCCESS)
            continue;
        std::vector<cl_device_id> platform_devices(count);
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count,
                           platform_devices.data(), nullptr) == CL_SUCCESS)
            devices.insert(devices.end(), platform_devices.begin(), platform_devices.end());
    }
    return devices;
}

std::string device_string(cl_device_id device, cl_device_info field) {
    std::size_t size = 0;
    clGetDeviceInfo(device, field, 0, nullptr, &size);
    std::string result(size, '\0');
    if (size != 0) clGetDeviceInfo(device, field, size, result.data(), nullptr);
    while (!result.empty() && result.back() == '\0') result.pop_back();
    return result;
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

class OpenClAccelerator final : public DescriptorAccelerator {
public:
    explicit OpenClAccelerator(int index) {
        const auto devices = opencl_gpu_devices();
        if (index < 0 || static_cast<std::size_t>(index) >= devices.size())
            throw std::runtime_error("invalid OpenCL GPU device index");
        device_ = devices[static_cast<std::size_t>(index)];
        name_ = device_string(device_, CL_DEVICE_NAME);
        cl_int error = CL_SUCCESS;
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
        check_cl(error, "clCreateContext");
        queue_ = clCreateCommandQueue(context_, device_, 0, &error);
        check_cl(error, "clCreateCommandQueue");
        const std::size_t length = std::char_traits<char>::length(kernel_source);
        const char* source = kernel_source;
        program_ = clCreateProgramWithSource(context_, 1, &source, &length, &error);
        check_cl(error, "clCreateProgramWithSource");
        error = clBuildProgram(program_, 1, &device_, "", nullptr, nullptr);
        if (error != CL_SUCCESS) {
            std::size_t log_size = 0;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG,
                                  log.size(), log.data(), nullptr);
            throw std::runtime_error("OpenCL program build failed: " + log);
        }
        kernel_ = clCreateKernel(program_,
            "match_unrolled_cached_16x16_hamming_i32", &error);
        check_cl(error, "clCreateKernel");
        log_kernel_ = clCreateKernel(program_, "log_response_device", &error);
        check_cl(error, "clCreateKernel log");
        orientation_kernel_ = clCreateKernel(program_, "orientation_peaks_device", &error);
        check_cl(error, "clCreateKernel orientation");
        mldb_kernel_ = clCreateKernel(program_, "mldb_descriptor_device", &error);
        check_cl(error, "clCreateKernel MLDB");
        extrema_kernel_ = clCreateKernel(program_, "locate_extrema_device", &error);
        check_cl(error, "clCreateKernel extrema");
        gaussian_row_kernel_ = clCreateKernel(program_, "gaussian_row_device", &error);
        check_cl(error, "clCreateKernel Gaussian row");
        gaussian_column_kernel_ = clCreateKernel(program_, "gaussian_column_device", &error);
        check_cl(error, "clCreateKernel Gaussian column");
        grayscale_kernel_ = clCreateKernel(program_, "rgb_to_grayscale_device", &error);
        check_cl(error, "clCreateKernel grayscale");
        decimate_kernel_ = clCreateKernel(program_, "integer_decimate_device", &error);
        check_cl(error, "clCreateKernel integer decimation");
        upsample_kernel_ = clCreateKernel(program_, "upsample_highest_device", &error);
        check_cl(error, "clCreateKernel Highest upsample");
    }

    ~OpenClAccelerator() override {
        if (feature_descriptors_) clReleaseMemObject(feature_descriptors_);
        if (feature_counts_) clReleaseMemObject(feature_counts_);
        if (feature_orientations_) clReleaseMemObject(feature_orientations_);
        if (feature_points_) clReleaseMemObject(feature_points_);
        if (feature_output_) clReleaseMemObject(feature_output_);
        if (blur_output_) clReleaseMemObject(blur_output_);
        if (gaussian_kernel_buffer_) clReleaseMemObject(gaussian_kernel_buffer_);
        if (rgb_input_) clReleaseMemObject(rgb_input_);
        if (feature_image_) clReleaseMemObject(feature_image_);
        if (extrema_counter_) clReleaseMemObject(extrema_counter_);
        if (extrema_output_) clReleaseMemObject(extrema_output_);
        if (extrema_sigma_) clReleaseMemObject(extrema_sigma_);
        if (extrema_gaussian_) clReleaseMemObject(extrema_gaussian_);
        if (resident_gray_) clReleaseMemObject(resident_gray_);
        for (auto& octave : resident_levels_)
            for (cl_mem level : octave)
                if (level) clReleaseMemObject(level);
        if (output_) clReleaseMemObject(output_);
        if (targets_) clReleaseMemObject(targets_);
        if (queries_) clReleaseMemObject(queries_);
        if (kernel_) clReleaseKernel(kernel_);
        if (mldb_kernel_) clReleaseKernel(mldb_kernel_);
        if (orientation_kernel_) clReleaseKernel(orientation_kernel_);
        if (log_kernel_) clReleaseKernel(log_kernel_);
        if (extrema_kernel_) clReleaseKernel(extrema_kernel_);
        if (gaussian_column_kernel_) clReleaseKernel(gaussian_column_kernel_);
        if (gaussian_row_kernel_) clReleaseKernel(gaussian_row_kernel_);
        if (grayscale_kernel_) clReleaseKernel(grayscale_kernel_);
        if (decimate_kernel_) clReleaseKernel(decimate_kernel_);
        if (upsample_kernel_) clReleaseKernel(upsample_kernel_);
        if (program_) clReleaseProgram(program_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }

    std::string backend_name() const override { return "opencl-target"; }
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
                throw std::runtime_error("OpenCL resident feature session already active");
            if (downscale != 0 && downscale != 1 && downscale != 2 &&
                downscale != 4 && downscale != 8)
                throw std::runtime_error("OpenCL resident feature downscale is invalid");
            resident_owner_ = std::this_thread::get_id();
            resident_active_ = true;
            resident_h2d_bytes_ = 0;
            resident_d2h_bytes_ = 0;
            resident_octave_count_ = 0;
            const std::size_t full_pixels = image.width * image.height;
            ensure_buffer(resident_gray_, resident_gray_capacity_,
                          full_pixels * sizeof(float), CL_MEM_READ_WRITE,
                          "clCreateBuffer resident grayscale");
            if (image.rgb.size() == full_pixels * 3U) {
                ensure_buffer(rgb_input_, rgb_input_capacity_, image.rgb.size(),
                              CL_MEM_READ_ONLY, "clCreateBuffer resident RGB input");
                check_cl(clEnqueueWriteBuffer(queue_, rgb_input_, CL_FALSE, 0,
                                             image.rgb.size(), image.rgb.data(),
                                             0, nullptr, nullptr),
                         "clEnqueueWriteBuffer resident RGB input");
                resident_h2d_bytes_ += image.rgb.size();
                const cl_ulong pixels = static_cast<cl_ulong>(full_pixels);
                check_cl(clSetKernelArg(grayscale_kernel_, 0, sizeof(rgb_input_),
                                       &rgb_input_),
                         "clSetKernelArg resident grayscale 0");
                check_cl(clSetKernelArg(grayscale_kernel_, 1, sizeof(pixels), &pixels),
                         "clSetKernelArg resident grayscale 1");
                check_cl(clSetKernelArg(grayscale_kernel_, 2, sizeof(resident_gray_),
                                       &resident_gray_),
                         "clSetKernelArg resident grayscale 2");
                constexpr std::size_t local = 256;
                const std::size_t global =
                    ((full_pixels + local - 1U) / local) * local;
                check_cl(clEnqueueNDRangeKernel(queue_, grayscale_kernel_, 1, nullptr,
                                               &global, &local, 0, nullptr, nullptr),
                         "clEnqueueNDRangeKernel resident grayscale");
            } else if (image.gray.size() == full_pixels) {
                check_cl(clEnqueueWriteBuffer(queue_, resident_gray_, CL_FALSE, 0,
                                             full_pixels * sizeof(float),
                                             image.gray.data(), 0, nullptr, nullptr),
                         "clEnqueueWriteBuffer resident grayscale input");
                resident_h2d_bytes_ += full_pixels * sizeof(float);
            } else {
                throw std::runtime_error("OpenCL resident feature image has no pixels");
            }

            std::size_t width = downscale == 0
                ? image.width * 2U - 1U
                : (image.width + static_cast<std::size_t>(downscale) - 1U) /
                      static_cast<std::size_t>(downscale);
            std::size_t height = downscale == 0
                ? image.height * 2U - 1U
                : (image.height + static_cast<std::size_t>(downscale) - 1U) /
                      static_cast<std::size_t>(downscale);
            ensure_buffer(blur_output_, blur_output_capacity_,
                          width * height * sizeof(float), CL_MEM_READ_WRITE,
                          "clCreateBuffer resident decimation scratch");
            cl_mem initial_input = resident_gray_;
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
            result.reserve(resident_levels_.size());
            for (int octave = 0; octave < static_cast<int>(resident_levels_.size()) &&
                                 width >= 32U && height >= 32U; ++octave) {
                const std::size_t slot = static_cast<std::size_t>(octave);
                const std::size_t bytes = width * height * sizeof(float);
                for (int level = 0; level < 5; ++level)
                    ensure_buffer(resident_levels_[slot][static_cast<std::size_t>(level)],
                                  resident_level_capacities_[slot]
                                      [static_cast<std::size_t>(level)],
                                  bytes, CL_MEM_READ_WRITE,
                                  "clCreateBuffer resident Gaussian level");
                resident_widths_[slot] = width;
                resident_heights_[slot] = height;
                if (octave == 0) {
                    constexpr float sigma0 = 1.6F;
                    const float inherited_sigma = downscale == 0 ? 1.0F : 0.5F;
                    const double sigma0_double = static_cast<double>(sigma0);
                    launch_gaussian(
                        initial_input, width, height,
                        std::sqrt(sigma0_double * sigma0_double -
                                  static_cast<double>(inherited_sigma * inherited_sigma)),
                        resident_levels_[slot][0]);
                }
                constexpr float scale_step = 1.2599210498948732F;
                float previous_sigma = 1.600000023841858F;
                for (int level = 1; level < 5; ++level) {
                    const float sigma = previous_sigma * scale_step;
                    launch_gaussian(
                        resident_levels_[slot][static_cast<std::size_t>(level - 1)],
                        width, height,
                        std::sqrt(sigma * sigma - previous_sigma * previous_sigma),
                        resident_levels_[slot][static_cast<std::size_t>(level)]);
                    previous_sigma = sigma;
                }
                ResidentFeatureOctave host;
                host.width = width;
                host.height = height;
                host.extrema = locate_resident_extrema(
                    slot, width, height, octave);
                result.push_back(std::move(host));
                ++resident_octave_count_;

                const std::size_t next_width = (width + 1U) / 2U;
                const std::size_t next_height = (height + 1U) / 2U;
                if (octave + 1 < static_cast<int>(resident_levels_.size()) &&
                    next_width >= 32U && next_height >= 32U) {
                    const std::size_t next_slot = slot + 1U;
                    ensure_buffer(resident_levels_[next_slot][0],
                                  resident_level_capacities_[next_slot][0],
                                  next_width * next_height * sizeof(float),
                                  CL_MEM_READ_WRITE,
                                  "clCreateBuffer resident Gaussian level");
                    launch_decimate(resident_levels_[slot][3], width, next_width,
                                    next_height, 2, resident_levels_[next_slot][0]);
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
        const std::size_t slot = static_cast<std::size_t>(octave);
        ensure_resident_point_buffers(points, true, false);
        const cl_int width = static_cast<cl_int>(resident_widths_[slot]);
        const cl_int height = static_cast<cl_int>(resident_heights_[slot]);
        const cl_int count = static_cast<cl_int>(points.size());
        cl_mem image = resident_levels_[slot][static_cast<std::size_t>(level)];
        check_cl(clSetKernelArg(orientation_kernel_, 0, sizeof(image), &image),
                 "clSetKernelArg resident orientation 0");
        check_cl(clSetKernelArg(orientation_kernel_, 1, sizeof(width), &width),
                 "clSetKernelArg resident orientation 1");
        check_cl(clSetKernelArg(orientation_kernel_, 2, sizeof(height), &height),
                 "clSetKernelArg resident orientation 2");
        check_cl(clSetKernelArg(orientation_kernel_, 3, sizeof(feature_points_),
                               &feature_points_),
                 "clSetKernelArg resident orientation 3");
        check_cl(clSetKernelArg(orientation_kernel_, 4, sizeof(count), &count),
                 "clSetKernelArg resident orientation 4");
        check_cl(clSetKernelArg(orientation_kernel_, 5, sizeof(feature_orientations_),
                               &feature_orientations_),
                 "clSetKernelArg resident orientation 5");
        check_cl(clSetKernelArg(orientation_kernel_, 6, sizeof(feature_counts_),
                               &feature_counts_),
                 "clSetKernelArg resident orientation 6");
        const std::size_t global = points.size();
        check_cl(clEnqueueNDRangeKernel(queue_, orientation_kernel_, 1, nullptr,
                                       &global, nullptr, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel resident orientation");
        std::vector<float> values(points.size() * 10U);
        std::vector<cl_uint> counts(points.size());
        check_cl(clEnqueueReadBuffer(queue_, feature_orientations_, CL_TRUE, 0,
                                    values.size() * sizeof(float), values.data(),
                                    0, nullptr, nullptr),
                 "clEnqueueReadBuffer resident orientations");
        check_cl(clEnqueueReadBuffer(queue_, feature_counts_, CL_TRUE, 0,
                                    counts.size() * sizeof(cl_uint), counts.data(),
                                    0, nullptr, nullptr),
                 "clEnqueueReadBuffer resident orientation counts");
        resident_d2h_bytes_ += values.size() * sizeof(float) +
                               counts.size() * sizeof(cl_uint);
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
        const std::size_t slot = static_cast<std::size_t>(octave);
        ensure_resident_point_buffers(points, false, true);
        const cl_int width = static_cast<cl_int>(resident_widths_[slot]);
        const cl_int height = static_cast<cl_int>(resident_heights_[slot]);
        const cl_int count = static_cast<cl_int>(points.size());
        cl_mem image = resident_levels_[slot][static_cast<std::size_t>(level)];
        check_cl(clSetKernelArg(mldb_kernel_, 0, sizeof(image), &image),
                 "clSetKernelArg resident MLDB 0");
        check_cl(clSetKernelArg(mldb_kernel_, 1, sizeof(width), &width),
                 "clSetKernelArg resident MLDB 1");
        check_cl(clSetKernelArg(mldb_kernel_, 2, sizeof(height), &height),
                 "clSetKernelArg resident MLDB 2");
        check_cl(clSetKernelArg(mldb_kernel_, 3, sizeof(feature_points_),
                               &feature_points_),
                 "clSetKernelArg resident MLDB 3");
        check_cl(clSetKernelArg(mldb_kernel_, 4, sizeof(count), &count),
                 "clSetKernelArg resident MLDB 4");
        check_cl(clSetKernelArg(mldb_kernel_, 5, sizeof(feature_descriptors_),
                               &feature_descriptors_),
                 "clSetKernelArg resident MLDB 5");
        const std::size_t global = points.size();
        check_cl(clEnqueueNDRangeKernel(queue_, mldb_kernel_, 1, nullptr, &global,
                                       nullptr, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel resident MLDB");
        check_cl(clEnqueueReadBuffer(queue_, feature_descriptors_, CL_TRUE, 0,
                                    result.size() * sizeof(Descriptor), result.data(),
                                    0, nullptr, nullptr),
                 "clEnqueueReadBuffer resident MLDB");
        resident_d2h_bytes_ += result.size() * sizeof(Descriptor);
        return result;
    }

    Image resident_feature_level(int octave, int level) override {
        validate_resident_layer(octave, level);
        const std::size_t slot = static_cast<std::size_t>(octave);
        const std::size_t pixels = resident_widths_[slot] * resident_heights_[slot];
        Image result;
        result.width = resident_widths_[slot];
        result.height = resident_heights_[slot];
        result.gray.resize(pixels);
        check_cl(clEnqueueReadBuffer(
            queue_, resident_levels_[slot][static_cast<std::size_t>(level)], CL_TRUE,
            0, pixels * sizeof(float), result.gray.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer resident Gaussian diagnostic");
        resident_d2h_bytes_ += pixels * sizeof(float);
        return result;
    }

    void end_resident_feature_image() override {
        if (!resident_active_ || resident_owner_ != std::this_thread::get_id())
            throw std::runtime_error("OpenCL resident feature session owner mismatch");
        try {
            check_cl(clFinish(queue_), "OpenCL resident feature finish");
            if (const char* value = std::getenv("METALIGN_GPU_TRANSFER_STATS");
                value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0) {
                std::fprintf(stderr,
                             "GPU transfer stats backend=opencl octaves=%zu "
                             "h2d_bytes=%zu d2h_bytes=%zu\n",
                             resident_octave_count_, resident_h2d_bytes_,
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
        const std::size_t pixels_size = image.width * image.height;
        ensure_buffer(rgb_input_, rgb_input_capacity_, image.rgb.size(),
                      CL_MEM_READ_ONLY, "clCreateBuffer RGB input");
        ensure_buffer(feature_output_, feature_output_capacity_,
                      pixels_size * sizeof(float), CL_MEM_READ_WRITE,
                      "clCreateBuffer grayscale output");
        check_cl(clEnqueueWriteBuffer(queue_, rgb_input_, CL_FALSE, 0,
                                     image.rgb.size(), image.rgb.data(),
                                     0, nullptr, nullptr),
                 "clEnqueueWriteBuffer RGB input");
        const cl_ulong pixels = static_cast<cl_ulong>(pixels_size);
        check_cl(clSetKernelArg(grayscale_kernel_, 0, sizeof(rgb_input_), &rgb_input_),
                 "clSetKernelArg grayscale 0");
        check_cl(clSetKernelArg(grayscale_kernel_, 1, sizeof(pixels), &pixels),
                 "clSetKernelArg grayscale 1");
        check_cl(clSetKernelArg(grayscale_kernel_, 2, sizeof(feature_output_), &feature_output_),
                 "clSetKernelArg grayscale 2");
        constexpr std::size_t local = 256;
        const std::size_t global = ((pixels_size + local - 1U) / local) * local;
        check_cl(clEnqueueNDRangeKernel(queue_, grayscale_kernel_, 1, nullptr,
                                       &global, &local, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel grayscale");
        Image result = image;
        result.gray.resize(pixels_size);
        check_cl(clEnqueueReadBuffer(queue_, feature_output_, CL_TRUE, 0,
                                    pixels_size * sizeof(float), result.gray.data(),
                                    0, nullptr, nullptr),
                 "clEnqueueReadBuffer grayscale output");
        return result;
    }

    Image gaussian_blur(const Image& image, double sigma) override {
        if (image.empty() || sigma <= 0.0) return image;
        std::lock_guard lock(mutex_);
        const int radius_value = std::max(1, static_cast<int>(4.0 * sigma + 1.0));
        std::vector<float> kernel(static_cast<std::size_t>(radius_value + 1));
        const float sigma_squared = static_cast<float>(sigma * sigma);
        const double exponent = -0.5 / static_cast<double>(sigma_squared);
        double sum = -1.0;
        for (int index = 0; index <= radius_value; ++index) {
            const float value = static_cast<float>(std::exp(exponent * index * index));
            kernel[static_cast<std::size_t>(index)] = value;
            sum += static_cast<double>(value + value);
        }
        for (float& value : kernel)
            value = static_cast<float>(static_cast<double>(value) / sum);

        const std::size_t image_bytes = image.gray.size() * sizeof(float);
        ensure_buffer(feature_image_, feature_image_capacity_, image_bytes,
                      CL_MEM_READ_ONLY, "clCreateBuffer Gaussian input");
        ensure_buffer(feature_output_, feature_output_capacity_, image_bytes,
                      CL_MEM_READ_WRITE, "clCreateBuffer Gaussian row output");
        ensure_buffer(blur_output_, blur_output_capacity_, image_bytes,
                      CL_MEM_WRITE_ONLY, "clCreateBuffer Gaussian column output");
        ensure_buffer(gaussian_kernel_buffer_, gaussian_kernel_capacity_,
                      kernel.size() * sizeof(float), CL_MEM_READ_ONLY,
                      "clCreateBuffer Gaussian kernel");
        check_cl(clEnqueueWriteBuffer(queue_, feature_image_, CL_FALSE, 0, image_bytes,
                                     image.gray.data(), 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer Gaussian input");
        check_cl(clEnqueueWriteBuffer(queue_, gaussian_kernel_buffer_, CL_FALSE, 0,
                                     kernel.size() * sizeof(float), kernel.data(),
                                     0, nullptr, nullptr),
                 "clEnqueueWriteBuffer Gaussian kernel");
        const cl_int width = static_cast<cl_int>(image.width);
        const cl_int height = static_cast<cl_int>(image.height);
        const cl_int radius = static_cast<cl_int>(radius_value);
        const std::size_t local[2]{16, 16};
        const std::size_t global[2]{((image.width + 15U) / 16U) * 16U,
                                    ((image.height + 15U) / 16U) * 16U};
        check_cl(clSetKernelArg(gaussian_row_kernel_, 0, sizeof(feature_image_), &feature_image_), "clSetKernelArg Gaussian row 0");
        check_cl(clSetKernelArg(gaussian_row_kernel_, 1, sizeof(width), &width), "clSetKernelArg Gaussian row 1");
        check_cl(clSetKernelArg(gaussian_row_kernel_, 2, sizeof(height), &height), "clSetKernelArg Gaussian row 2");
        check_cl(clSetKernelArg(gaussian_row_kernel_, 3, sizeof(gaussian_kernel_buffer_), &gaussian_kernel_buffer_), "clSetKernelArg Gaussian row 3");
        check_cl(clSetKernelArg(gaussian_row_kernel_, 4, sizeof(radius), &radius), "clSetKernelArg Gaussian row 4");
        check_cl(clSetKernelArg(gaussian_row_kernel_, 5, sizeof(feature_output_), &feature_output_), "clSetKernelArg Gaussian row 5");
        check_cl(clEnqueueNDRangeKernel(queue_, gaussian_row_kernel_, 2, nullptr,
                                       global, local, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel Gaussian row");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 0, sizeof(feature_output_), &feature_output_), "clSetKernelArg Gaussian column 0");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 1, sizeof(width), &width), "clSetKernelArg Gaussian column 1");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 2, sizeof(height), &height), "clSetKernelArg Gaussian column 2");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 3, sizeof(gaussian_kernel_buffer_), &gaussian_kernel_buffer_), "clSetKernelArg Gaussian column 3");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 4, sizeof(radius), &radius), "clSetKernelArg Gaussian column 4");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 5, sizeof(blur_output_), &blur_output_), "clSetKernelArg Gaussian column 5");
        check_cl(clEnqueueNDRangeKernel(queue_, gaussian_column_kernel_, 2, nullptr,
                                       global, local, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel Gaussian column");
        Image result = image;
        result.gray.resize(image.gray.size());
        check_cl(clEnqueueReadBuffer(queue_, blur_output_, CL_TRUE, 0, image_bytes,
                                    result.gray.data(), 0, nullptr, nullptr),
                 "clEnqueueReadBuffer Gaussian output");
        return result;
    }

    std::vector<GpuExtremum> locate_extrema(
        std::span<const Image> gaussian_levels, int octave) override {
        std::lock_guard lock(mutex_);
        if (gaussian_levels.size() != 5)
            throw std::runtime_error("OpenCL extrema producer requires five Gaussian levels");
        const std::size_t width_size = gaussian_levels.front().width;
        const std::size_t height_size = gaussian_levels.front().height;
        const std::size_t pixels = width_size * height_size;
        for (const Image& image : gaussian_levels)
            if (image.width != width_size || image.height != height_size ||
                image.gray.size() != pixels)
                throw std::runtime_error("OpenCL extrema Gaussian level shape mismatch");
        constexpr std::size_t capacity = 1'100'000U;
        ensure_buffer(extrema_gaussian_, extrema_gaussian_capacity_,
                      pixels * 5U * sizeof(float), CL_MEM_READ_ONLY,
                      "clCreateBuffer extrema Gaussian");
        ensure_buffer(extrema_sigma_, extrema_sigma_capacity_, 5U * sizeof(float),
                      CL_MEM_READ_ONLY, "clCreateBuffer extrema sigma");
        ensure_buffer(extrema_output_, extrema_output_capacity_,
                      capacity * sizeof(GpuExtremum), CL_MEM_READ_WRITE,
                      "clCreateBuffer extrema output");
        ensure_buffer(extrema_counter_, extrema_counter_capacity_, sizeof(cl_uint),
                      CL_MEM_READ_WRITE, "clCreateBuffer extrema counter");
        for (std::size_t level = 0; level < gaussian_levels.size(); ++level)
            check_cl(clEnqueueWriteBuffer(queue_, extrema_gaussian_, CL_FALSE,
                         level * pixels * sizeof(float), pixels * sizeof(float),
                         gaussian_levels[level].gray.data(), 0, nullptr, nullptr),
                     "clEnqueueWriteBuffer extrema Gaussian");
        const float sigma[5]{1.600000023841858F, 2.015873670578003F,
                             2.539841651916504F, 3.200000047683716F,
                             4.031747341156006F};
        const cl_uint zero = 0;
        check_cl(clEnqueueWriteBuffer(queue_, extrema_sigma_, CL_FALSE, 0,
                                     sizeof(sigma), sigma, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer extrema sigma");
        check_cl(clEnqueueFillBuffer(queue_, extrema_output_, &zero, sizeof(zero), 0,
                                    capacity * sizeof(GpuExtremum), 0, nullptr, nullptr),
                 "clEnqueueFillBuffer extrema output");
        check_cl(clEnqueueWriteBuffer(queue_, extrema_counter_, CL_FALSE, 0,
                                     sizeof(zero), &zero, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer extrema counter");
        const cl_uint capacity_value = static_cast<cl_uint>(capacity);
        const cl_int width = static_cast<cl_int>(width_size);
        const cl_int height = static_cast<cl_int>(height_size);
        const cl_ulong intervals = 3;
        const float minimum = 0.0F, maximum = 10.0F;
        const cl_int octave_value = octave;
        const float sigma0 = 1.600000023841858F, border = 7.0F;
        check_cl(clSetKernelArg(extrema_kernel_, 0, sizeof(extrema_gaussian_), &extrema_gaussian_), "clSetKernelArg extrema 0");
        check_cl(clSetKernelArg(extrema_kernel_, 1, sizeof(extrema_sigma_), &extrema_sigma_), "clSetKernelArg extrema 1");
        check_cl(clSetKernelArg(extrema_kernel_, 2, sizeof(extrema_output_), &extrema_output_), "clSetKernelArg extrema 2");
        check_cl(clSetKernelArg(extrema_kernel_, 3, sizeof(extrema_counter_), &extrema_counter_), "clSetKernelArg extrema 3");
        check_cl(clSetKernelArg(extrema_kernel_, 4, sizeof(capacity_value), &capacity_value), "clSetKernelArg extrema 4");
        check_cl(clSetKernelArg(extrema_kernel_, 5, sizeof(width), &width), "clSetKernelArg extrema 5");
        check_cl(clSetKernelArg(extrema_kernel_, 6, sizeof(height), &height), "clSetKernelArg extrema 6");
        check_cl(clSetKernelArg(extrema_kernel_, 7, sizeof(intervals), &intervals), "clSetKernelArg extrema 7");
        check_cl(clSetKernelArg(extrema_kernel_, 8, sizeof(minimum), &minimum), "clSetKernelArg extrema 8");
        check_cl(clSetKernelArg(extrema_kernel_, 9, sizeof(maximum), &maximum), "clSetKernelArg extrema 9");
        check_cl(clSetKernelArg(extrema_kernel_, 10, sizeof(octave_value), &octave_value), "clSetKernelArg extrema 10");
        check_cl(clSetKernelArg(extrema_kernel_, 11, sizeof(sigma0), &sigma0), "clSetKernelArg extrema 11");
        check_cl(clSetKernelArg(extrema_kernel_, 12, sizeof(border), &border), "clSetKernelArg extrema 12");
        check_cl(clSetKernelArg(extrema_kernel_, 14, 3U * 16U * 16U * sizeof(float), nullptr), "clSetKernelArg extrema local");
        const std::size_t active_pixels = width_size * (height_size - 2U);
        const std::size_t stripes = std::max<std::size_t>(
            1U, (active_pixels + 699'999U) / 700'000U);
        const std::size_t stripe_blocks =
            (height_size - 2U + stripes * 14U - 1U) / (stripes * 14U);
        const std::size_t local[3]{16, 16, 1};
        const std::size_t global_x = ((width_size - 2U + 13U) / 14U) * 16U;
        for (std::size_t row_offset = 0; row_offset < height_size - 2U;
             row_offset += stripe_blocks * 14U) {
            const std::size_t rows = height_size - 2U - row_offset;
            const std::size_t block_rows = std::min(stripe_blocks, (rows + 13U) / 14U);
            const cl_int offset = static_cast<cl_int>(row_offset);
            check_cl(clSetKernelArg(extrema_kernel_, 13, sizeof(offset), &offset), "clSetKernelArg extrema 13");
            const std::size_t global[3]{global_x, block_rows * 16U, 3};
            check_cl(clEnqueueNDRangeKernel(queue_, extrema_kernel_, 3, nullptr,
                                            global, local, 0, nullptr, nullptr),
                     "clEnqueueNDRangeKernel extrema");
        }
        cl_uint count = 0;
        check_cl(clEnqueueReadBuffer(queue_, extrema_counter_, CL_TRUE, 0,
                                    sizeof(count), &count, 0, nullptr, nullptr),
                 "clEnqueueReadBuffer extrema count");
        if (count > capacity) throw std::runtime_error("OpenCL extrema capacity exceeded");
        std::vector<GpuExtremum> result(count);
        if (!result.empty()) {
            check_cl(clEnqueueReadBuffer(queue_, extrema_output_, CL_TRUE, 0,
                                        result.size() * sizeof(GpuExtremum), result.data(),
                                        0, nullptr, nullptr),
                     "clEnqueueReadBuffer extrema output");
        }
        return result;
    }

    Image laplacian_response(const Image& image, float sigma) override {
        std::lock_guard lock(mutex_);
        ensure_buffer(feature_image_, feature_image_capacity_,
                      image.gray.size() * sizeof(float), CL_MEM_READ_ONLY,
                      "clCreateBuffer feature image");
        ensure_buffer(feature_output_, feature_output_capacity_,
                      image.gray.size() * sizeof(float), CL_MEM_WRITE_ONLY,
                      "clCreateBuffer feature output");
        check_cl(clEnqueueWriteBuffer(queue_, feature_image_, CL_FALSE, 0,
                    image.gray.size() * sizeof(float), image.gray.data(), 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer feature image");
        const cl_int width = static_cast<cl_int>(image.width);
        const cl_int height = static_cast<cl_int>(image.height);
        const float normalization = sigma * sigma;
        check_cl(clSetKernelArg(log_kernel_, 0, sizeof(feature_image_), &feature_image_), "clSetKernelArg log 0");
        check_cl(clSetKernelArg(log_kernel_, 1, sizeof(width), &width), "clSetKernelArg log 1");
        check_cl(clSetKernelArg(log_kernel_, 2, sizeof(height), &height), "clSetKernelArg log 2");
        check_cl(clSetKernelArg(log_kernel_, 3, sizeof(normalization), &normalization), "clSetKernelArg log 3");
        check_cl(clSetKernelArg(log_kernel_, 4, sizeof(feature_output_), &feature_output_), "clSetKernelArg log 4");
        const std::size_t local[2]{16, 16};
        const std::size_t global[2]{((image.width + 15U) / 16U) * 16U,
                                    ((image.height + 15U) / 16U) * 16U};
        check_cl(clEnqueueNDRangeKernel(queue_, log_kernel_, 2, nullptr, global, local,
                                        0, nullptr, nullptr), "clEnqueueNDRangeKernel log");
        Image result;
        result.width = image.width;
        result.height = image.height;
        result.gray.resize(image.gray.size());
        check_cl(clEnqueueReadBuffer(queue_, feature_output_, CL_TRUE, 0,
                    result.gray.size() * sizeof(float), result.gray.data(), 0, nullptr, nullptr),
                 "clEnqueueReadBuffer log");
        return result;
    }

    std::vector<std::vector<float>> orientation_peaks(
        const Image& image, std::span<const FeaturePrimitive> points) override {
        std::lock_guard lock(mutex_);
        std::vector<std::vector<float>> result(points.size());
        if (points.empty()) return result;
        ensure_feature_buffers(image, points, true, false);
        const cl_int width = static_cast<cl_int>(image.width);
        const cl_int height = static_cast<cl_int>(image.height);
        const cl_int count = static_cast<cl_int>(points.size());
        check_cl(clSetKernelArg(orientation_kernel_, 0, sizeof(feature_image_), &feature_image_), "clSetKernelArg orientation 0");
        check_cl(clSetKernelArg(orientation_kernel_, 1, sizeof(width), &width), "clSetKernelArg orientation 1");
        check_cl(clSetKernelArg(orientation_kernel_, 2, sizeof(height), &height), "clSetKernelArg orientation 2");
        check_cl(clSetKernelArg(orientation_kernel_, 3, sizeof(feature_points_), &feature_points_), "clSetKernelArg orientation 3");
        check_cl(clSetKernelArg(orientation_kernel_, 4, sizeof(count), &count), "clSetKernelArg orientation 4");
        check_cl(clSetKernelArg(orientation_kernel_, 5, sizeof(feature_orientations_), &feature_orientations_), "clSetKernelArg orientation 5");
        check_cl(clSetKernelArg(orientation_kernel_, 6, sizeof(feature_counts_), &feature_counts_), "clSetKernelArg orientation 6");
        const std::size_t global = points.size();
        check_cl(clEnqueueNDRangeKernel(queue_, orientation_kernel_, 1, nullptr, &global,
                                        nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel orientation");
        std::vector<float> values(points.size() * 10U);
        std::vector<cl_uint> counts(points.size());
        check_cl(clEnqueueReadBuffer(queue_, feature_orientations_, CL_TRUE, 0,
                    values.size() * sizeof(float), values.data(), 0, nullptr, nullptr),
                 "clEnqueueReadBuffer orientations");
        check_cl(clEnqueueReadBuffer(queue_, feature_counts_, CL_TRUE, 0,
                    counts.size() * sizeof(cl_uint), counts.data(), 0, nullptr, nullptr),
                 "clEnqueueReadBuffer orientation counts");
        for (std::size_t index = 0; index < points.size(); ++index)
            result[index].assign(values.begin() + static_cast<std::ptrdiff_t>(index * 10U),
                values.begin() + static_cast<std::ptrdiff_t>(index * 10U + counts[index]));
        return result;
    }

    std::vector<Descriptor> mldb_descriptors(
        const Image& image, std::span<const FeaturePrimitive> points) override {
        std::lock_guard lock(mutex_);
        std::vector<Descriptor> result(points.size());
        if (points.empty()) return result;
        ensure_feature_buffers(image, points, false, true);
        const cl_int width = static_cast<cl_int>(image.width);
        const cl_int height = static_cast<cl_int>(image.height);
        const cl_int count = static_cast<cl_int>(points.size());
        check_cl(clSetKernelArg(mldb_kernel_, 0, sizeof(feature_image_), &feature_image_), "clSetKernelArg MLDB 0");
        check_cl(clSetKernelArg(mldb_kernel_, 1, sizeof(width), &width), "clSetKernelArg MLDB 1");
        check_cl(clSetKernelArg(mldb_kernel_, 2, sizeof(height), &height), "clSetKernelArg MLDB 2");
        check_cl(clSetKernelArg(mldb_kernel_, 3, sizeof(feature_points_), &feature_points_), "clSetKernelArg MLDB 3");
        check_cl(clSetKernelArg(mldb_kernel_, 4, sizeof(count), &count), "clSetKernelArg MLDB 4");
        check_cl(clSetKernelArg(mldb_kernel_, 5, sizeof(feature_descriptors_), &feature_descriptors_), "clSetKernelArg MLDB 5");
        const std::size_t global = points.size();
        check_cl(clEnqueueNDRangeKernel(queue_, mldb_kernel_, 1, nullptr, &global,
                                        nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel MLDB");
        check_cl(clEnqueueReadBuffer(queue_, feature_descriptors_, CL_TRUE, 0,
                    result.size() * sizeof(Descriptor), result.data(), 0, nullptr, nullptr),
                 "clEnqueueReadBuffer MLDB");
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
        std::unordered_map<const std::vector<Keypoint>*, std::size_t> query_offsets;
        std::unordered_map<const std::vector<Keypoint>*, std::size_t> target_offsets;
        std::vector<std::uint32_t> query_data;
        std::vector<std::uint32_t> target_data;
        auto append_once = [](const std::vector<Keypoint>* keypoints,
                              auto& offsets, auto& packed) {
            if (!keypoints) throw std::runtime_error("invalid OpenCL descriptor match batch");
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
                throw std::runtime_error("invalid OpenCL descriptor match batch");
            result[index].assign(batch.queries->size(), {-1, 0.0});
            launches[index] = {
                append_once(batch.queries, query_offsets, query_data),
                append_once(batch.targets, target_offsets, target_data), output_count};
            output_count += batch.queries->size();
        }
        if (output_count == 0) return result;
        ensure_buffer(queries_, query_capacity_, query_data.size() * sizeof(std::uint32_t),
                      CL_MEM_READ_ONLY, "clCreateBuffer queries");
        ensure_buffer(targets_, target_capacity_, target_data.size() * sizeof(std::uint32_t),
                      CL_MEM_READ_ONLY, "clCreateBuffer targets");
        ensure_buffer(output_, output_capacity_, output_count * sizeof(cl_int),
                      CL_MEM_WRITE_ONLY, "clCreateBuffer output");
        if (!query_data.empty()) {
            check_cl(clEnqueueWriteBuffer(queue_, queries_, CL_FALSE, 0,
                                          query_data.size() * sizeof(std::uint32_t),
                                          query_data.data(), 0, nullptr, nullptr),
                     "clEnqueueWriteBuffer packed queries");
        }
        if (!target_data.empty()) {
            check_cl(clEnqueueWriteBuffer(queue_, targets_, CL_FALSE, 0,
                                          target_data.size() * sizeof(std::uint32_t),
                                          target_data.data(), 0, nullptr, nullptr),
                     "clEnqueueWriteBuffer packed targets");
        }
        const cl_int minus_one = -1;
        check_cl(clEnqueueFillBuffer(queue_, output_, &minus_one, sizeof(minus_one), 0,
                                    output_count * sizeof(cl_int), 0, nullptr, nullptr),
                 "clEnqueueFillBuffer batched output");
        const cl_int dimension = 16;
        const std::size_t local[2] = {16, 16};
        for (std::size_t index = 0; index < batches.size(); ++index) {
            const RatioMatchBatch& batch = batches[index];
            if (batch.queries->empty() || batch.targets->size() < 2) continue;
            const cl_int query_count = static_cast<cl_int>(batch.queries->size());
            const cl_int target_count = static_cast<cl_int>(batch.targets->size());
            const cl_int query_offset = static_cast<cl_int>(launches[index].query_offset);
            const cl_int target_offset = static_cast<cl_int>(launches[index].target_offset);
            const cl_int output_offset = static_cast<cl_int>(launches[index].output_offset);
            check_cl(clSetKernelArg(kernel_, 0, sizeof(queries_), &queries_), "clSetKernelArg 0");
            check_cl(clSetKernelArg(kernel_, 1, sizeof(query_count), &query_count), "clSetKernelArg 1");
            check_cl(clSetKernelArg(kernel_, 2, sizeof(query_offset), &query_offset), "clSetKernelArg 2");
            check_cl(clSetKernelArg(kernel_, 3, sizeof(targets_), &targets_), "clSetKernelArg 3");
            check_cl(clSetKernelArg(kernel_, 4, sizeof(target_count), &target_count), "clSetKernelArg 4");
            check_cl(clSetKernelArg(kernel_, 5, sizeof(target_offset), &target_offset), "clSetKernelArg 5");
            check_cl(clSetKernelArg(kernel_, 6, sizeof(dimension), &dimension), "clSetKernelArg 6");
            check_cl(clSetKernelArg(kernel_, 7, sizeof(output_), &output_), "clSetKernelArg 7");
            check_cl(clSetKernelArg(kernel_, 8, sizeof(ratio), &ratio), "clSetKernelArg 8");
            check_cl(clSetKernelArg(kernel_, 9, sizeof(output_offset), &output_offset), "clSetKernelArg 9");
            const std::size_t global[2] = {
                ((batch.queries->size() + 15) / 16) * 16, 16};
            check_cl(clEnqueueNDRangeKernel(queue_, kernel_, 2, nullptr, global, local,
                                            0, nullptr, nullptr),
                     "clEnqueueNDRangeKernel batched match");
        }
        std::vector<cl_int> host_output(output_count);
        check_cl(clEnqueueReadBuffer(queue_, output_, CL_TRUE, 0,
                                     host_output.size() * sizeof(cl_int), host_output.data(),
                                     0, nullptr, nullptr), "clEnqueueReadBuffer output");
        for (std::size_t index = 0; index < batches.size(); ++index) {
            const RatioMatchBatch& batch = batches[index];
            for (std::size_t query = 0; query < result[index].size(); ++query) {
                const std::int32_t target =
                    host_output[launches[index].output_offset + query];
                if (target < 0) continue;
                result[index][query] = {target, static_cast<double>(hamming_distance(
                    (*batch.queries)[query],
                    (*batch.targets)[static_cast<std::size_t>(target)]))};
            }
        }
        return result;
    }

private:
    void launch_upsample_highest(cl_mem input, std::size_t input_width,
                                 std::size_t input_height,
                                 std::size_t output_width, cl_mem output) {
        const cl_int input_width_value = static_cast<cl_int>(input_width);
        const cl_int input_height_value = static_cast<cl_int>(input_height);
        const cl_int output_width_value = static_cast<cl_int>(output_width);
        check_cl(clSetKernelArg(upsample_kernel_, 0, sizeof(input), &input),
                 "clSetKernelArg Highest upsample 0");
        check_cl(clSetKernelArg(upsample_kernel_, 1, sizeof(input_width_value),
                               &input_width_value),
                 "clSetKernelArg Highest upsample 1");
        check_cl(clSetKernelArg(upsample_kernel_, 2, sizeof(input_height_value),
                               &input_height_value),
                 "clSetKernelArg Highest upsample 2");
        check_cl(clSetKernelArg(upsample_kernel_, 3, sizeof(output_width_value),
                               &output_width_value),
                 "clSetKernelArg Highest upsample 3");
        check_cl(clSetKernelArg(upsample_kernel_, 4, sizeof(output), &output),
                 "clSetKernelArg Highest upsample 4");
        const std::size_t local[2]{16, 16};
        const std::size_t global[2]{((input_width + 15U) / 16U) * 16U,
                                    ((input_height + 15U) / 16U) * 16U};
        check_cl(clEnqueueNDRangeKernel(queue_, upsample_kernel_, 2, nullptr,
                                       global, local, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel Highest upsample");
    }

    void launch_decimate(cl_mem input, std::size_t input_width,
                         std::size_t output_width, std::size_t output_height,
                         int factor, cl_mem output) {
        const cl_int input_width_value = static_cast<cl_int>(input_width);
        const cl_int output_width_value = static_cast<cl_int>(output_width);
        const cl_int output_height_value = static_cast<cl_int>(output_height);
        const cl_int factor_value = static_cast<cl_int>(factor);
        check_cl(clSetKernelArg(decimate_kernel_, 0, sizeof(input), &input),
                 "clSetKernelArg resident decimate 0");
        check_cl(clSetKernelArg(decimate_kernel_, 1, sizeof(input_width_value),
                               &input_width_value),
                 "clSetKernelArg resident decimate 1");
        check_cl(clSetKernelArg(decimate_kernel_, 2, sizeof(output_width_value),
                               &output_width_value),
                 "clSetKernelArg resident decimate 2");
        check_cl(clSetKernelArg(decimate_kernel_, 3, sizeof(output_height_value),
                               &output_height_value),
                 "clSetKernelArg resident decimate 3");
        check_cl(clSetKernelArg(decimate_kernel_, 4, sizeof(factor_value),
                               &factor_value),
                 "clSetKernelArg resident decimate 4");
        check_cl(clSetKernelArg(decimate_kernel_, 5, sizeof(output), &output),
                 "clSetKernelArg resident decimate 5");
        const std::size_t local[2]{16, 16};
        const std::size_t global[2]{((output_width + 15U) / 16U) * 16U,
                                    ((output_height + 15U) / 16U) * 16U};
        check_cl(clEnqueueNDRangeKernel(queue_, decimate_kernel_, 2, nullptr,
                                       global, local, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel resident decimate");
    }

    void launch_gaussian(cl_mem input, std::size_t width, std::size_t height,
                         double sigma, cl_mem output) {
        const int radius_value = std::max(1, static_cast<int>(4.0 * sigma + 1.0));
        std::vector<float> kernel(static_cast<std::size_t>(radius_value + 1));
        const float sigma_squared = static_cast<float>(sigma * sigma);
        const double exponent = -0.5 / static_cast<double>(sigma_squared);
        double sum = -1.0;
        for (int index = 0; index <= radius_value; ++index) {
            const float value = static_cast<float>(std::exp(exponent * index * index));
            kernel[static_cast<std::size_t>(index)] = value;
            sum += static_cast<double>(value + value);
        }
        for (float& value : kernel)
            value = static_cast<float>(static_cast<double>(value) / sum);
        const std::size_t bytes = width * height * sizeof(float);
        ensure_buffer(feature_output_, feature_output_capacity_, bytes,
                      CL_MEM_READ_WRITE, "clCreateBuffer resident Gaussian scratch");
        ensure_buffer(gaussian_kernel_buffer_, gaussian_kernel_capacity_,
                      kernel.size() * sizeof(float), CL_MEM_READ_ONLY,
                      "clCreateBuffer resident Gaussian kernel");
        // The half-kernel is method-local; Intel's ICD may defer a
        // nonblocking host copy past this function return.
        check_cl(clEnqueueWriteBuffer(queue_, gaussian_kernel_buffer_, CL_TRUE, 0,
                                     kernel.size() * sizeof(float), kernel.data(),
                                     0, nullptr, nullptr),
                 "clEnqueueWriteBuffer resident Gaussian kernel");
        resident_h2d_bytes_ += kernel.size() * sizeof(float);
        const cl_int width_value = static_cast<cl_int>(width);
        const cl_int height_value = static_cast<cl_int>(height);
        const cl_int radius = static_cast<cl_int>(radius_value);
        const std::size_t local[2]{16, 16};
        const std::size_t global[2]{((width + 15U) / 16U) * 16U,
                                    ((height + 15U) / 16U) * 16U};
        check_cl(clSetKernelArg(gaussian_row_kernel_, 0, sizeof(input), &input),
                 "clSetKernelArg resident Gaussian row 0");
        check_cl(clSetKernelArg(gaussian_row_kernel_, 1, sizeof(width_value), &width_value),
                 "clSetKernelArg resident Gaussian row 1");
        check_cl(clSetKernelArg(gaussian_row_kernel_, 2, sizeof(height_value), &height_value),
                 "clSetKernelArg resident Gaussian row 2");
        check_cl(clSetKernelArg(gaussian_row_kernel_, 3, sizeof(gaussian_kernel_buffer_),
                               &gaussian_kernel_buffer_),
                 "clSetKernelArg resident Gaussian row 3");
        check_cl(clSetKernelArg(gaussian_row_kernel_, 4, sizeof(radius), &radius),
                 "clSetKernelArg resident Gaussian row 4");
        check_cl(clSetKernelArg(gaussian_row_kernel_, 5, sizeof(feature_output_),
                               &feature_output_),
                 "clSetKernelArg resident Gaussian row 5");
        check_cl(clEnqueueNDRangeKernel(queue_, gaussian_row_kernel_, 2, nullptr,
                                       global, local, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel resident Gaussian row");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 0, sizeof(feature_output_),
                               &feature_output_),
                 "clSetKernelArg resident Gaussian column 0");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 1, sizeof(width_value), &width_value),
                 "clSetKernelArg resident Gaussian column 1");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 2, sizeof(height_value), &height_value),
                 "clSetKernelArg resident Gaussian column 2");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 3, sizeof(gaussian_kernel_buffer_),
                               &gaussian_kernel_buffer_),
                 "clSetKernelArg resident Gaussian column 3");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 4, sizeof(radius), &radius),
                 "clSetKernelArg resident Gaussian column 4");
        check_cl(clSetKernelArg(gaussian_column_kernel_, 5, sizeof(output), &output),
                 "clSetKernelArg resident Gaussian column 5");
        check_cl(clEnqueueNDRangeKernel(queue_, gaussian_column_kernel_, 2, nullptr,
                                       global, local, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel resident Gaussian column");
    }

    std::vector<GpuExtremum> locate_resident_extrema(
        std::size_t octave_slot, std::size_t width_size,
        std::size_t height_size, int octave) {
        const std::size_t pixels = width_size * height_size;
        constexpr std::size_t capacity = 1'100'000U;
        ensure_buffer(extrema_gaussian_, extrema_gaussian_capacity_,
                      pixels * 5U * sizeof(float), CL_MEM_READ_ONLY,
                      "clCreateBuffer resident extrema Gaussian");
        ensure_buffer(extrema_sigma_, extrema_sigma_capacity_, 5U * sizeof(float),
                      CL_MEM_READ_ONLY, "clCreateBuffer resident extrema sigma");
        ensure_buffer(extrema_output_, extrema_output_capacity_,
                      capacity * sizeof(GpuExtremum), CL_MEM_READ_WRITE,
                      "clCreateBuffer resident extrema output");
        ensure_buffer(extrema_counter_, extrema_counter_capacity_, sizeof(cl_uint),
                      CL_MEM_READ_WRITE, "clCreateBuffer resident extrema counter");
        for (std::size_t level = 0; level < 5U; ++level)
            check_cl(clEnqueueCopyBuffer(
                         queue_, resident_levels_[octave_slot][level], extrema_gaussian_,
                         0, level * pixels * sizeof(float), pixels * sizeof(float),
                         0, nullptr, nullptr),
                     "clEnqueueCopyBuffer resident extrema Gaussian");
        const float sigma[5]{1.600000023841858F, 2.015873670578003F,
                             2.539841651916504F, 3.200000047683716F,
                             4.031747341156006F};
        const cl_uint zero = 0;
        check_cl(clEnqueueWriteBuffer(queue_, extrema_sigma_, CL_FALSE, 0,
                                     sizeof(sigma), sigma, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer resident extrema sigma");
        resident_h2d_bytes_ += sizeof(sigma);
        check_cl(clEnqueueFillBuffer(queue_, extrema_output_, &zero, sizeof(zero), 0,
                                    capacity * sizeof(GpuExtremum), 0, nullptr, nullptr),
                 "clEnqueueFillBuffer resident extrema output");
        check_cl(clEnqueueWriteBuffer(queue_, extrema_counter_, CL_FALSE, 0,
                                     sizeof(zero), &zero, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer resident extrema counter");
        const cl_uint capacity_value = static_cast<cl_uint>(capacity);
        const cl_int width = static_cast<cl_int>(width_size);
        const cl_int height = static_cast<cl_int>(height_size);
        const cl_ulong intervals = 3;
        const float minimum = 0.0F, maximum = 10.0F;
        const float sigma0 = 1.600000023841858F, border = 7.0F;
        check_cl(clSetKernelArg(extrema_kernel_, 0, sizeof(extrema_gaussian_),
                               &extrema_gaussian_), "clSetKernelArg resident extrema 0");
        check_cl(clSetKernelArg(extrema_kernel_, 1, sizeof(extrema_sigma_),
                               &extrema_sigma_), "clSetKernelArg resident extrema 1");
        check_cl(clSetKernelArg(extrema_kernel_, 2, sizeof(extrema_output_),
                               &extrema_output_), "clSetKernelArg resident extrema 2");
        check_cl(clSetKernelArg(extrema_kernel_, 3, sizeof(extrema_counter_),
                               &extrema_counter_), "clSetKernelArg resident extrema 3");
        check_cl(clSetKernelArg(extrema_kernel_, 4, sizeof(capacity_value),
                               &capacity_value), "clSetKernelArg resident extrema 4");
        check_cl(clSetKernelArg(extrema_kernel_, 5, sizeof(width), &width),
                 "clSetKernelArg resident extrema 5");
        check_cl(clSetKernelArg(extrema_kernel_, 6, sizeof(height), &height),
                 "clSetKernelArg resident extrema 6");
        check_cl(clSetKernelArg(extrema_kernel_, 7, sizeof(intervals), &intervals),
                 "clSetKernelArg resident extrema 7");
        check_cl(clSetKernelArg(extrema_kernel_, 8, sizeof(minimum), &minimum),
                 "clSetKernelArg resident extrema 8");
        check_cl(clSetKernelArg(extrema_kernel_, 9, sizeof(maximum), &maximum),
                 "clSetKernelArg resident extrema 9");
        check_cl(clSetKernelArg(extrema_kernel_, 10, sizeof(octave), &octave),
                 "clSetKernelArg resident extrema 10");
        check_cl(clSetKernelArg(extrema_kernel_, 11, sizeof(sigma0), &sigma0),
                 "clSetKernelArg resident extrema 11");
        check_cl(clSetKernelArg(extrema_kernel_, 12, sizeof(border), &border),
                 "clSetKernelArg resident extrema 12");
        check_cl(clSetKernelArg(extrema_kernel_, 14,
                               3U * 16U * 16U * sizeof(float), nullptr),
                 "clSetKernelArg resident extrema local");
        const std::size_t active_pixels = width_size * (height_size - 2U);
        const std::size_t stripes = std::max<std::size_t>(
            1U, (active_pixels + 699'999U) / 700'000U);
        const std::size_t stripe_blocks =
            (height_size - 2U + stripes * 14U - 1U) / (stripes * 14U);
        const std::size_t local[3]{16, 16, 1};
        const std::size_t global_x = ((width_size - 2U + 13U) / 14U) * 16U;
        for (std::size_t row_offset = 0; row_offset < height_size - 2U;
             row_offset += stripe_blocks * 14U) {
            const std::size_t rows = height_size - 2U - row_offset;
            const std::size_t block_rows =
                std::min(stripe_blocks, (rows + 13U) / 14U);
            const cl_int offset = static_cast<cl_int>(row_offset);
            check_cl(clSetKernelArg(extrema_kernel_, 13, sizeof(offset), &offset),
                     "clSetKernelArg resident extrema 13");
            const std::size_t global[3]{global_x, block_rows * 16U, 3};
            check_cl(clEnqueueNDRangeKernel(queue_, extrema_kernel_, 3, nullptr,
                                           global, local, 0, nullptr, nullptr),
                     "clEnqueueNDRangeKernel resident extrema");
        }
        cl_uint count = 0;
        check_cl(clEnqueueReadBuffer(queue_, extrema_counter_, CL_TRUE, 0,
                                    sizeof(count), &count, 0, nullptr, nullptr),
                 "clEnqueueReadBuffer resident extrema count");
        resident_d2h_bytes_ += sizeof(count);
        if (count > capacity)
            throw std::runtime_error(
                "OpenCL resident extrema capacity exceeded: count=" +
                std::to_string(count) + ", octave=" + std::to_string(octave));
        std::vector<GpuExtremum> result(count);
        if (!result.empty()) {
            check_cl(clEnqueueReadBuffer(queue_, extrema_output_, CL_TRUE, 0,
                                        result.size() * sizeof(GpuExtremum), result.data(),
                                        0, nullptr, nullptr),
                     "clEnqueueReadBuffer resident extrema output");
            resident_d2h_bytes_ += result.size() * sizeof(GpuExtremum);
        }
        return result;
    }

    void ensure_resident_point_buffers(
        std::span<const FeaturePrimitive> points,
        bool orientations, bool descriptors) {
        ensure_buffer(feature_points_, feature_point_capacity_, points.size_bytes(),
                      CL_MEM_READ_ONLY, "clCreateBuffer resident feature points");
        if (orientations) {
            ensure_buffer(feature_orientations_, feature_orientation_capacity_,
                          points.size() * 10U * sizeof(float), CL_MEM_WRITE_ONLY,
                          "clCreateBuffer resident orientations");
            ensure_buffer(feature_counts_, feature_count_capacity_,
                          points.size() * sizeof(cl_uint), CL_MEM_WRITE_ONLY,
                          "clCreateBuffer resident orientation counts");
        }
        if (descriptors)
            ensure_buffer(feature_descriptors_, feature_descriptor_capacity_,
                          points.size() * sizeof(Descriptor), CL_MEM_WRITE_ONLY,
                          "clCreateBuffer resident MLDB");
        check_cl(clEnqueueWriteBuffer(queue_, feature_points_, CL_FALSE, 0,
                                     points.size_bytes(), points.data(),
                                     0, nullptr, nullptr),
                 "clEnqueueWriteBuffer resident feature points");
        resident_h2d_bytes_ += points.size_bytes();
    }

    void validate_resident_layer(int octave, int level) const {
        if (!resident_active_ || resident_owner_ != std::this_thread::get_id())
            throw std::runtime_error("OpenCL resident feature session owner mismatch");
        if (octave < 0 || octave >= static_cast<int>(resident_octave_count_) ||
            level < 0 || level >= 5)
            throw std::runtime_error("OpenCL resident feature layer is out of range");
    }

    void ensure_feature_buffers(const Image& image,
                                std::span<const FeaturePrimitive> points,
                                bool orientations, bool descriptors) {
        ensure_buffer(feature_image_, feature_image_capacity_,
                      image.gray.size() * sizeof(float), CL_MEM_READ_ONLY,
                      "clCreateBuffer feature image");
        ensure_buffer(feature_points_, feature_point_capacity_, points.size_bytes(),
                      CL_MEM_READ_ONLY, "clCreateBuffer feature points");
        if (orientations) {
            ensure_buffer(feature_orientations_, feature_orientation_capacity_,
                          points.size() * 10U * sizeof(float), CL_MEM_WRITE_ONLY,
                          "clCreateBuffer orientations");
            ensure_buffer(feature_counts_, feature_count_capacity_,
                          points.size() * sizeof(cl_uint), CL_MEM_WRITE_ONLY,
                          "clCreateBuffer orientation counts");
        }
        if (descriptors)
            ensure_buffer(feature_descriptors_, feature_descriptor_capacity_,
                          points.size() * sizeof(Descriptor), CL_MEM_WRITE_ONLY,
                          "clCreateBuffer MLDB");
        check_cl(clEnqueueWriteBuffer(queue_, feature_image_, CL_FALSE, 0,
                    image.gray.size() * sizeof(float), image.gray.data(), 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer feature image");
        check_cl(clEnqueueWriteBuffer(queue_, feature_points_, CL_FALSE, 0,
                    points.size_bytes(), points.data(), 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer feature points");
    }
    void ensure_buffer(cl_mem& buffer, std::size_t& capacity, std::size_t bytes,
                       cl_mem_flags flags, const char* operation) {
        if (bytes <= capacity) return;
        if (buffer) check_cl(clReleaseMemObject(buffer), "clReleaseMemObject grow buffer");
        buffer = nullptr;
        capacity = 0;
        cl_int error = CL_SUCCESS;
        buffer = clCreateBuffer(context_, flags, bytes, nullptr, &error);
        check_cl(error, operation);
        capacity = bytes;
    }

    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
    cl_kernel log_kernel_ = nullptr;
    cl_kernel orientation_kernel_ = nullptr;
    cl_kernel mldb_kernel_ = nullptr;
    cl_kernel extrema_kernel_ = nullptr;
    cl_kernel gaussian_row_kernel_ = nullptr;
    cl_kernel gaussian_column_kernel_ = nullptr;
    cl_kernel grayscale_kernel_ = nullptr;
    cl_kernel decimate_kernel_ = nullptr;
    cl_kernel upsample_kernel_ = nullptr;
    cl_mem queries_ = nullptr;
    cl_mem targets_ = nullptr;
    cl_mem output_ = nullptr;
    std::size_t query_capacity_ = 0;
    std::size_t target_capacity_ = 0;
    std::size_t output_capacity_ = 0;
    cl_mem feature_image_ = nullptr;
    cl_mem feature_output_ = nullptr;
    cl_mem blur_output_ = nullptr;
    cl_mem gaussian_kernel_buffer_ = nullptr;
    cl_mem rgb_input_ = nullptr;
    cl_mem feature_points_ = nullptr;
    cl_mem feature_orientations_ = nullptr;
    cl_mem feature_counts_ = nullptr;
    cl_mem feature_descriptors_ = nullptr;
    std::size_t feature_image_capacity_ = 0;
    std::size_t feature_output_capacity_ = 0;
    std::size_t blur_output_capacity_ = 0;
    std::size_t gaussian_kernel_capacity_ = 0;
    std::size_t rgb_input_capacity_ = 0;
    std::size_t feature_point_capacity_ = 0;
    std::size_t feature_orientation_capacity_ = 0;
    std::size_t feature_count_capacity_ = 0;
    std::size_t feature_descriptor_capacity_ = 0;
    cl_mem extrema_gaussian_ = nullptr;
    cl_mem extrema_sigma_ = nullptr;
    cl_mem extrema_output_ = nullptr;
    cl_mem extrema_counter_ = nullptr;
    std::size_t extrema_gaussian_capacity_ = 0;
    std::size_t extrema_sigma_capacity_ = 0;
    std::size_t extrema_output_capacity_ = 0;
    std::size_t extrema_counter_capacity_ = 0;
    cl_mem resident_gray_ = nullptr;
    std::size_t resident_gray_capacity_ = 0;
    std::array<std::array<cl_mem, 5>, 6> resident_levels_{};
    std::array<std::array<std::size_t, 5>, 6> resident_level_capacities_{};
    std::array<std::size_t, 6> resident_widths_{};
    std::array<std::size_t, 6> resident_heights_{};
    std::size_t resident_octave_count_ = 0;
    std::size_t resident_h2d_bytes_ = 0;
    std::size_t resident_d2h_bytes_ = 0;
    bool resident_active_ = false;
    std::thread::id resident_owner_{};
    std::mutex mutex_;
    std::string name_;
};

}  // namespace

std::vector<GpuDeviceInfo> enumerate_opencl_devices() {
    const auto devices = opencl_gpu_devices();
    std::vector<GpuDeviceInfo> result;
    for (std::size_t index = 0; index < devices.size(); ++index) {
        cl_ulong memory = 0;
        clGetDeviceInfo(devices[index], CL_DEVICE_GLOBAL_MEM_SIZE,
                        sizeof(memory), &memory, nullptr);
        result.push_back({"opencl", static_cast<int>(index),
                          device_string(devices[index], CL_DEVICE_NAME), memory});
    }
    return result;
}

std::unique_ptr<DescriptorAccelerator> create_opencl_accelerator(int device_index) {
    return std::make_unique<OpenClAccelerator>(device_index);
}

}  // namespace metalign
