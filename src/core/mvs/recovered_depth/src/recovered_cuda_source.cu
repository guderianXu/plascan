#include "recovered_cuda_source.hpp"

#include "metmodel/patchmatch.hpp"
#include "metmodel/octree_prepare.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <bit>
#include <cfloat>

namespace metmodel
{
    namespace
    {

        // Source reconstruction of cuda::filterBasedOnVoting.  Keep the block-local
        // reduction and unsigned atomic additions: these are observable parts of the
        // recovered PTX contract, not an optimization choice.
        __global__ void depth_voting_finalize_kernel(float* depth,
                                                     const std::int32_t* votes,
                                                     std::uint32_t width,
                                                     std::uint32_t height,
                                                     std::uint32_t* counter_empty,
                                                     std::uint32_t* counter_bad,
                                                     std::uint32_t* counter_normal,
                                                     std::uint32_t* counter_good,
                                                     std::uint32_t pixel_offset)
        {
            __shared__ std::uint32_t local_empty;
            __shared__ std::uint32_t local_bad;
            __shared__ std::uint32_t local_normal;
            __shared__ std::uint32_t local_good;
            if (threadIdx.x == 0U)
            {
                local_empty = 0U;
                local_bad = 0U;
                local_normal = 0U;
                local_good = 0U;
            }
            __syncthreads();

            const std::uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x + pixel_offset;
            const std::uint32_t y = linear / width;
            if (y < height)
            {
                const std::uint32_t x = linear % width;
                const std::uint32_t index = y * width + x;
                if (depth[index] == 0.0F)
                {
                    atomicAdd(&local_empty, 1U);
                }
                else if (votes[index] < 1)
                {
                    depth[index] = 0.0F;
                    atomicAdd(&local_bad, 1U);
                }
                else if (votes[index] < 200)
                {
                    atomicAdd(&local_normal, 1U);
                }
                else
                {
                    atomicAdd(&local_good, 1U);
                }
            }

            __syncthreads();
            if (threadIdx.x == 0U)
            {
                atomicAdd(counter_empty, local_empty);
                atomicAdd(counter_bad, local_bad);
                atomicAdd(counter_normal, local_normal);
                atomicAdd(counter_good, local_good);
            }
        }

        __global__
            __launch_bounds__(128,
                              1) void patchmatch_copy_inlier_masks_kernel(std::uint8_t* neighbor_inlier_masks,
                                                                          std::uint32_t width,
                                                                          std::uint32_t height,
                                                                          std::uint32_t hypotheses_per_pixel,
                                                                          std::uint32_t neighbor_count,
                                                                          const std::uint8_t* temporary_inlier_masks,
                                                                          const std::uint8_t* winner,
                                                                          std::uint32_t is_checkboard,
                                                                          std::uint32_t checkboard_step,
                                                                          std::uint32_t only_each_fourth_pixel,
                                                                          std::uint32_t pixel_offset)
        {
            constexpr std::uint32_t capacity = 128U * 1024U;
            const std::uint32_t temporary_index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t global_index = pixel_offset + temporary_index;
            std::uint32_t x = 0U;
            std::uint32_t y = 0U;
            if (is_checkboard != 0U)
            {
                if (only_each_fourth_pixel == 0U)
                {
                    const std::uint32_t width_half = (width + 1U) / 2U;
                    y = global_index / width_half;
                    x = ((y % 2U) + checkboard_step) % 2U + 2U * (global_index % width_half);
                }
                else
                {
                    const std::uint32_t width_part = (width + 3U) / 4U;
                    y = 1U + 2U * global_index / width_part;
                    x = 1U + (((y / 2U) % 2U + checkboard_step) % 2U) * 2U + 4U * (global_index % width_part);
                }
            }
            else if (only_each_fourth_pixel == 0U)
            {
                y = global_index / width;
                x = global_index % width;
            }
            else
            {
                const std::uint32_t width_half = (width + 1U) / 2U;
                y = 1U + 2U * (global_index / width_half);
                x = 1U + 2U * (global_index % width_half);
            }
            if (x >= width || y >= height)
                return;

            const std::uint8_t best = winner[temporary_index];
            const std::uint32_t pixel = y * width + x;
            for (std::uint32_t neighbor = 0U; neighbor < neighbor_count; neighbor += 8U)
            {
                const std::uint32_t group = neighbor / 8U;
                std::uint8_t value = 0U;
                if (best != 255U)
                {
                    value = temporary_inlier_masks[group * capacity * hypotheses_per_pixel +
                                                   static_cast<std::uint32_t>(best) * capacity + temporary_index];
                }
                neighbor_inlier_masks[group * width * height + pixel] = value;
            }
        }

        struct RecoveredFloat3
        {
            float x;
            float y;
            float z;
        };

        __device__ __forceinline__ bool recovered_calibration_unproject3(const DepthVotingCalibrationCu& calibration,
                                                                         float projection_x,
                                                                         float projection_y,
                                                                         float depth,
                                                                         RecoveredFloat3& output,
                                                                         std::uint32_t normalization_order = 0U);

        __device__ __forceinline__ bool recovered_calibration_project3(const DepthVotingCalibrationCu& calibration,
                                                                       const RecoveredFloat3& point,
                                                                       RecoveredFloat3& projection);

        template <class T>
        __device__ __forceinline__ T recovered_opaque_load(const DepthVotingCalibrationCu& calibration,
                                                           std::uint32_t offset)
        {
            return *reinterpret_cast<const T*>(reinterpret_cast<const std::uint8_t*>(&calibration) + offset);
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_depth_radius_unproject_type1(
            const DepthVotingCalibrationCu& calibration, float projection_x, float projection_y, float depth)
        {
            const float* values = reinterpret_cast<const float*>(&calibration);
            const std::int32_t width = recovered_opaque_load<std::int32_t>(calibration, 84U);
            const std::int32_t height = recovered_opaque_load<std::int32_t>(calibration, 88U);
            const float focal = values[23U];
            const float cx = values[24U];
            const float cy = values[25U];
            const float b1 = values[26U];
            const float b2 = values[27U];

            // The recovered sm_30 PTX leaves these multiply/subtract pairs
            // contractible.  Its observed sm_89 JIT SASS uses FFMA, so preserve that
            // device-visible rounding boundary explicitly in the source path.
            float y = __fmaf_rn(__int2float_rn(height), -0.5F, projection_y);
            y = __fsub_rn(y, cy);
            y = __fdiv_rn(y, focal);
            float x = __fmaf_rn(__int2float_rn(width), -0.5F, projection_x);
            x = __fsub_rn(x, cx);
            x = __fmaf_rn(-y, b2, x);
            x = __fdiv_rn(x, __fadd_rn(focal, b1));

            if (recovered_opaque_load<std::uint32_t>(calibration, 112U) != 0U &&
                (values[0U] != 0.0F || values[1U] != 0.0F || values[2U] != 0.0F || values[3U] != 0.0F ||
                 values[4U] != 0.0F || values[5U] != 0.0F || values[6U] != 0.0F || values[7U] != 0.0F))
            {
                const float x0 = x;
                const float y0 = y;
                const float k1 = values[0U];
                const float k2 = values[1U];
                const float k3 = values[2U];
                const float k4 = values[3U];
                const float p1 = values[4U];
                const float p2 = values[5U];
                const float p3 = values[6U];
                const float p4 = values[7U];
                const float r2max = values[8U];
#pragma unroll
                for (std::uint32_t iteration = 0U; iteration < 5U; ++iteration)
                {
                    const float x2 = __fmul_rn(x, x);
                    const float y2 = __fmul_rn(y, y);
                    float r2 = __fadd_rn(y2, x2);
                    float norm2 = 1.0F;
                    if (r2 > r2max)
                    {
                        norm2 = __fdiv_rn(r2max, r2);
                        r2 = r2max;
                    }
                    const float r4 = __fmul_rn(r2, r2);
                    float denominator = __fmaf_rn(k1, r2, 1.0F);
                    denominator = __fmaf_rn(k2, r4, denominator);
                    denominator = __fmaf_rn(__fmul_rn(k3, r2), r4, denominator);
                    denominator = __fmaf_rn(r4, __fmul_rn(k4, r4), denominator);
                    const float inverse = __frcp_rn(denominator);
                    const float pdist = __fmaf_rn(p4, r4, __fmaf_rn(p3, r2, 1.0F));
                    const float scale = __fmul_rn(norm2, pdist);
                    const float twice_p2_x = __fmul_rn(__fadd_rn(p2, p2), x);
                    const float three_x2_plus_y2 = __fmaf_rn(x, __fmul_rn(x, 3.0F), y2);
                    const float delta_x_inner = __fmaf_rn(y, twice_p2_x, __fmul_rn(p1, three_x2_plus_y2));
                    const float three_y2_plus_x2 = __fmaf_rn(y, __fmul_rn(y, 3.0F), x2);
                    const float twice_p1_x = __fmul_rn(__fadd_rn(p1, p1), x);
                    const float delta_y_inner = __fmaf_rn(y, twice_p1_x, __fmul_rn(p2, three_y2_plus_x2));
                    x = __fmul_rn(inverse, __fmaf_rn(-scale, delta_x_inner, x0));
                    y = __fmul_rn(inverse, __fmaf_rn(-scale, delta_y_inner, y0));
                }
            }
            return {
                __fmul_rn(x, depth),
                __fmul_rn(y, depth),
                depth,
            };
        }

        __device__ __forceinline__ bool recovered_undistort_project_type1(const DepthVotingCalibrationCu& calibration,
                                                                          const RecoveredFloat3& point,
                                                                          float& projection_x,
                                                                          float& projection_y)
        {
            if (point.z <= 0.0F)
                return false;
            const float* values = reinterpret_cast<const float*>(&calibration);
            const float inverse_z = __fdiv_rn(1.0F, point.z);
            float x = __fmul_rn(point.x, inverse_z);
            float y = __fmul_rn(point.y, inverse_z);
            if (recovered_opaque_load<std::uint32_t>(calibration, 112U) != 0U)
            {
                const float x2 = __fmul_rn(x, x);
                const float y2 = __fmul_rn(y, y);
                float r2 = __fadd_rn(x2, y2);
                float norm2 = 1.0F;
                if (r2 > values[8U])
                {
                    norm2 = __fdiv_rn(values[8U], r2);
                    r2 = values[8U];
                }
                const float r4 = __fmul_rn(r2, r2);
                float radial = __fmaf_rn(values[0U], r2, __fmul_rn(values[1U], r4));
                radial = __fmaf_rn(__fmul_rn(values[2U], r2), r4, radial);
                radial = __fmaf_rn(r4, __fmul_rn(values[3U], r4), radial);
                const float pdist = __fmaf_rn(values[7U], r4, __fmaf_rn(values[6U], r2, 1.0F));
                const float dx_inner = __fmaf_rn(__fmul_rn(__fadd_rn(values[5U], values[5U]), x),
                                                 y,
                                                 __fmul_rn(values[4U], __fmaf_rn(x, __fmul_rn(3.0F, x), y2)));
                const float dy_inner = __fmaf_rn(__fmul_rn(__fadd_rn(values[4U], values[4U]), x),
                                                 y,
                                                 __fmul_rn(values[5U], __fmaf_rn(y, __fmul_rn(3.0F, y), x2)));
                const float dx = __fmaf_rn(norm2, __fmul_rn(dx_inner, pdist), __fmul_rn(x, radial));
                const float dy = __fmaf_rn(norm2, __fmul_rn(dy_inner, pdist), __fmul_rn(y, radial));
                x = __fadd_rn(x, dx);
                y = __fadd_rn(y, dy);
            }
            float px_term = __fmaf_rn(values[23U], x, __fmul_rn(values[26U], x));
            px_term = __fmaf_rn(values[27U], y, px_term);
            const float px_center =
                __fmaf_rn(__uint2float_rn(recovered_opaque_load<std::uint32_t>(calibration, 84U)), 0.5F, values[24U]);
            const float px = __fadd_rn(px_center, px_term);
            const float py_center =
                __fmaf_rn(__uint2float_rn(recovered_opaque_load<std::uint32_t>(calibration, 88U)), 0.5F, values[25U]);
            const float py = __fmaf_rn(values[23U], y, py_center);
            projection_x = px;
            projection_y = py;
            return true;
        }

        template <typename Source> __device__ __forceinline__ float recovered_undistort_sample(Source value)
        {
            if constexpr (std::is_same_v<Source, float>)
                return value;
            else
                return __uint2float_rn(static_cast<std::uint32_t>(value));
        }

        template <typename Result> __device__ __forceinline__ Result recovered_undistort_result(float value)
        {
            if constexpr (std::is_same_v<Result, float>)
                return value;
            else
                return static_cast<Result>(static_cast<std::uint32_t>(__fadd_rn(value, 0.5F)));
        }

        template <typename Source, typename Result>
        __global__ __launch_bounds__(128,
                                     1) void patchmatch_undistort_kernel(const Source* source,
                                                                         const std::uint8_t* source_mask,
                                                                         Result* result,
                                                                         std::uint8_t* result_mask,
                                                                         DepthVotingCalibrationCu source_calibration,
                                                                         DepthVotingCalibrationCu target_calibration,
                                                                         std::uint32_t width,
                                                                         std::uint32_t height,
                                                                         std::uint32_t channels,
                                                                         std::uint32_t with_mask,
                                                                         std::uint32_t pixel_offset)
        {
            const std::uint32_t logical = pixel_offset + blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t x = logical % width;
            const std::uint32_t y = logical / width;
            if (x >= width || y >= height)
                return;
            const RecoveredFloat3 ray = recovered_depth_radius_unproject_type1(
                target_calibration, __fadd_rn(__uint2float_rn(x), 0.5F), __fadd_rn(__uint2float_rn(y), 0.5F), 1.0F);
            float px = 0.0F;
            float py = 0.0F;
            const std::uint32_t output_pixel = y * width + x;
            if (!recovered_undistort_project_type1(source_calibration, ray, px, py) || px < 0.0F || py < 0.0F ||
                px >= __uint2float_rn(width) || py >= __uint2float_rn(height))
            {
                for (std::uint32_t channel = 0; channel < channels; ++channel)
                    result[output_pixel * channels + channel] = Result{};
                result_mask[output_pixel] = 255U;
                return;
            }
            const float sample_x = __fsub_rn(px, 0.5F);
            const float sample_y = __fsub_rn(py, 0.5F);
            const std::int32_t ix = __float2int_rd(sample_x);
            const std::int32_t iy = __float2int_rd(sample_y);
            const float dx = __fsub_rn(sample_x, __int2float_rn(ix));
            const float dy = __fsub_rn(sample_y, __int2float_rn(iy));
            const std::uint32_t x0 = static_cast<std::uint32_t>(max(0, min(ix, static_cast<std::int32_t>(width) - 1)));
            const std::uint32_t x1 =
                static_cast<std::uint32_t>(max(0, min(ix + 1, static_cast<std::int32_t>(width) - 1)));
            const std::uint32_t y0 = static_cast<std::uint32_t>(max(0, min(iy, static_cast<std::int32_t>(height) - 1)));
            const std::uint32_t y1 =
                static_cast<std::uint32_t>(max(0, min(iy + 1, static_cast<std::int32_t>(height) - 1)));
            const std::uint32_t ll_pixel = y0 * width + x0;
            const std::uint32_t lr_pixel = y0 * width + x1;
            const std::uint32_t ul_pixel = y1 * width + x0;
            const std::uint32_t ur_pixel = y1 * width + x1;
            const float llm = with_mask == 0U || source_mask[ll_pixel] != 0U ? 1.0F : 0.0F;
            const float lrm = with_mask == 0U || source_mask[lr_pixel] != 0U ? 1.0F : 0.0F;
            const float ulm = with_mask == 0U || source_mask[ul_pixel] != 0U ? 1.0F : 0.0F;
            const float urm = with_mask == 0U || source_mask[ur_pixel] != 0U ? 1.0F : 0.0F;
            const float one_minus_dx = __fsub_rn(1.0F, dx);
            const float one_minus_dy = __fsub_rn(1.0F, dy);
            // Preserve the target compiler's bilinear operation tree.  In
            // particular, the vertical blends are fused while each horizontal
            // weight is rounded before it enters the FMA.
            for (std::uint32_t channel = 0; channel < channels; ++channel)
            {
                const std::uint32_t ll = ll_pixel * channels + channel;
                const std::uint32_t lr = lr_pixel * channels + channel;
                const std::uint32_t ul = ul_pixel * channels + channel;
                const std::uint32_t ur = ur_pixel * channels + channel;
                const float ll_value = __fmul_rn(one_minus_dx, __fmul_rn(llm, recovered_undistort_sample(source[ll])));
                const float lr_value = __fmul_rn(dx, __fmul_rn(lrm, recovered_undistort_sample(source[lr])));
                float interpolated = __fmaf_rn(one_minus_dy, ll_value, __fmul_rn(one_minus_dy, lr_value));
                const float ul_value = __fmul_rn(one_minus_dx, __fmul_rn(ulm, recovered_undistort_sample(source[ul])));
                interpolated = __fmaf_rn(dy, ul_value, interpolated);
                const float ur_value = __fmul_rn(dx, __fmul_rn(urm, recovered_undistort_sample(source[ur])));
                interpolated = __fmaf_rn(dy, ur_value, interpolated);
                result[output_pixel * channels + channel] = recovered_undistort_result<Result>(interpolated);
            }
            result_mask[output_pixel] = __fadd_rn(__fadd_rn(llm, lrm), __fadd_rn(ulm, urm)) > 0.0F ? 0U : 255U;
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_ooc_camera_to_world(
            const OocHistogramCameraExteriorTransformCu& exterior, const RecoveredFloat3& point)
        {
            const float* m = reinterpret_cast<const float*>(&exterior);
            return {__fadd_rn(__fmaf_rn(m[0], point.x, __fmul_rn(m[1], point.y)), __fmaf_rn(m[2], point.z, m[12])),
                    __fadd_rn(__fmaf_rn(m[4], point.x, __fmul_rn(m[5], point.y)), __fmaf_rn(m[6], point.z, m[13])),
                    __fadd_rn(__fmaf_rn(m[8], point.x, __fmul_rn(m[9], point.y)), __fmaf_rn(m[10], point.z, m[14]))};
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_ooc_world_to_camera(
            const OocHistogramCameraExteriorTransformCu& exterior, const RecoveredFloat3& point)
        {
            const float* m = reinterpret_cast<const float*>(&exterior);
            const float dx = __fsub_rn(point.x, m[12]);
            const float dy = __fsub_rn(point.y, m[13]);
            const float dz = __fsub_rn(point.z, m[14]);
            const float w = m[15];
            return {__fdiv_rn(__fmaf_rn(m[8], dz, __fmaf_rn(m[0], dx, __fmul_rn(m[4], dy))), w),
                    __fdiv_rn(__fmaf_rn(m[9], dz, __fmaf_rn(m[1], dx, __fmul_rn(m[5], dy))), w),
                    __fdiv_rn(__fmaf_rn(m[10], dz, __fmaf_rn(m[2], dx, __fmul_rn(m[6], dy))), w)};
        }

        __device__ __forceinline__ float recovered_ooc_distance_squared(const RecoveredFloat3& left,
                                                                        const RecoveredFloat3& right)
        {
            const float dx = __fsub_rn(left.x, right.x);
            const float dy = __fsub_rn(left.y, right.y);
            const float dz = __fsub_rn(left.z, right.z);
            return __fmaf_rn(dz, dz, __fmaf_rn(dx, dx, __fmul_rn(dy, dy)));
        }

        __device__ __forceinline__ void
        recovered_ooc_histogram_add(OocHistogramVoxel& voxel, float residual, float raw_weight)
        {
            float position = __fadd_rn(1.0F, residual);
            position = __fmul_rn(position, 0.5F);
            position = __fmul_rn(position, 9.0F);
            position = __fadd_rn(position, 0.5F);
            int selected = max(0, min(9, __float2int_rz(position)));
            if (!(raw_weight > 1.0F))
            {
                std::uint8_t& bin = *(reinterpret_cast<std::uint8_t*>(&voxel) + 12 + selected);
                if (bin != 255U)
                {
                    const int combined = __float2int_rz(__fadd_rn(__uint2float_rn(bin), raw_weight));
                    bin = static_cast<std::uint8_t>(max(0, min(254, combined)));
                }
                return;
            }
            int total = raw_weight < 254.0F ? static_cast<std::uint8_t>(__float2int_rz(raw_weight)) : 254;
            int lower = selected;
            int upper = selected + 1;
            const auto center = [](int index)
            {
                const float divided = __fdiv_rn(__int2float_rn(index), 9.0F);
                return __fsub_rn(__fadd_rn(divided, divided), 1.0F);
            };
            if (selected != 0 && (selected == 9 || residual < center(selected)))
            {
                lower = selected - 1;
                upper = selected;
            }
            const float upper_center = center(upper);
            const float lower_center = center(lower);
            float lower_weight = __fdiv_rn(__fsub_rn(upper_center, residual), __fsub_rn(upper_center, lower_center));
            lower_weight = __fmul_rn(lower_weight, __int2float_rn(total));
            const int lower_increment = __float2int_rn(lower_weight);
            const int increments[2]{lower_increment, total - lower_increment};
            const int indices[2]{lower, upper};
#pragma unroll
            for (int i = 0; i != 2; ++i)
            {
                std::uint8_t& bin = *(reinterpret_cast<std::uint8_t*>(&voxel) + 12 + indices[i]);
                if (bin != 255U)
                    bin = static_cast<std::uint8_t>(min(254, int(bin) + increments[i]));
            }
        }

        __global__ __launch_bounds__(128,
                                     1) void ooc_histogram_mode0_kernel(OocHistogramVoxel* voxels,
                                                                        const float* pyramid,
                                                                        std::uint32_t pyramid_levels,
                                                                        std::uint32_t mode_b,
                                                                        OocHistogramCalibrationCu calibration,
                                                                        OocHistogramCameraExteriorTransformCu exterior,
                                                                        float threshold_a,
                                                                        float threshold_b,
                                                                        std::uint32_t begin,
                                                                        std::uint32_t end)
        {
            const std::uint32_t index = begin + blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= end)
                return;
            OocHistogramVoxel& voxel = voxels[index];
            const float* c = reinterpret_cast<const float*>(&calibration);
            const std::uint32_t width0 =
                *reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(&calibration) + 84U);
            const std::uint32_t height0 =
                *reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(&calibration) + 88U);
            const float* voxel_floats = reinterpret_cast<const float*>(&voxel);
            const std::uint8_t* voxel_bytes = reinterpret_cast<const std::uint8_t*>(&voxel);
            const RecoveredFloat3 position{voxel_floats[0], voxel_floats[1], voxel_floats[2]};
            const RecoveredFloat3 camera = recovered_ooc_world_to_camera(exterior, position);
            if (!(camera.z > 0.0F))
                return;
            const float inv_z = __fdiv_rn(1.0F, camera.z);
            const float nx = __fmul_rn(camera.x, inv_z);
            const float ny = __fmul_rn(camera.y, inv_z);
            const float px_term = __fmaf_rn(c[23], nx, __fmul_rn(c[26], nx));
            const float px = __fadd_rn(__fmaf_rn(c[27], ny, px_term), __fmaf_rn(__uint2float_rn(width0), 0.5F, c[24]));
            const float py = __fmaf_rn(c[23], ny, __fmaf_rn(__uint2float_rn(height0), 0.5F, c[25]));
            if (px < 0.0F || py < 0.0F || px >= __uint2float_rn(width0) || py >= __uint2float_rn(height0))
                return;

            const float projected_depth = camera.z;
            const RecoveredFloat3 horizontal_camera =
                recovered_depth_radius_unproject_type1(*reinterpret_cast<const DepthVotingCalibrationCu*>(&calibration),
                                                       __fadd_rn(px, 0.5F),
                                                       py,
                                                       projected_depth);
            const RecoveredFloat3 vertical_camera =
                recovered_depth_radius_unproject_type1(*reinterpret_cast<const DepthVotingCalibrationCu*>(&calibration),
                                                       px,
                                                       __fadd_rn(py, 0.5F),
                                                       projected_depth);
            const float horizontal_distance = __fsqrt_rn(
                recovered_ooc_distance_squared(recovered_ooc_camera_to_world(exterior, horizontal_camera), position));
            const float vertical_distance = __fsqrt_rn(
                recovered_ooc_distance_squared(recovered_ooc_camera_to_world(exterior, vertical_camera), position));
            float geometry = fmaxf(horizontal_distance, vertical_distance);
            if (mode_b == 1U)
                geometry = __fmul_rn(geometry, __uint_as_float(0x3fb50481U));

            const std::uint32_t scale_code = voxel_bytes[26];
            const float support_scale =
                scale_code == 32U ? __uint_as_float(0x2f800000U) : __fdiv_rn(1.0F, __uint2float_rn(1U << scale_code));
            const float reference_depth = __half2float(*reinterpret_cast<const __half*>(voxel_bytes + 22));
            float vote_factor = __fdiv_rn(__uint2float_rn(voxel_bytes[27]), __uint_as_float(0x437fe666U));
            vote_factor = __fmaf_rn(vote_factor, 0.75F, 0.75F);
            float level_threshold = __fmul_rn(threshold_a, 0.5F);
            level_threshold = __fmul_rn(level_threshold, support_scale);
            level_threshold = __fmul_rn(level_threshold, vote_factor);
            level_threshold = __fmul_rn(reference_depth, level_threshold);

            std::int32_t width = static_cast<std::int32_t>(width0);
            std::int32_t height = static_cast<std::int32_t>(height0);
            std::uint32_t pyramid_scale = 1U;
            std::uint32_t base = 0U;
            float selected_geometry = geometry;
            float selected_x = px;
            float selected_y = py;
            if (pyramid_levels > 1U && level_threshold > __fmul_rn(selected_geometry, 4.0F))
            {
                for (std::uint32_t level = 1U; level < pyramid_levels; ++level)
                {
                    selected_geometry = __fadd_rn(selected_geometry, selected_geometry);
                    selected_x = __fmul_rn(selected_x, 0.5F);
                    selected_y = __fmul_rn(selected_y, 0.5F);
                    base += 2U * static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height);
                    width /= 2;
                    height /= 2;
                    pyramid_scale += pyramid_scale;
                    if (level + 1U == pyramid_levels || !(level_threshold > __fmul_rn(selected_geometry, 4.0F)))
                        break;
                }
            }

            const float sx = __fsub_rn(selected_x, 0.5F);
            const float sy = __fsub_rn(selected_y, 0.5F);
            const int ix = __float2int_rz(sx);
            const int iy = __float2int_rz(sy);
            const float fx = __fsub_rn(sx, __int2float_rn(ix));
            const float fy = __fsub_rn(sy, __int2float_rn(iy));
            const int x0 = max(0, min(width - 1, ix));
            const int x1 = max(0, min(width - 1, ix + 1));
            const int y0 = max(0, min(height - 1, iy));
            const int y1 = max(0, min(height - 1, iy + 1));
            const int pixels[4]{x0 + width * y0, x1 + width * y0, x0 + width * y1, x1 + width * y1};
            const float weights[4]{__fmul_rn(__fsub_rn(1.0F, fy), __fsub_rn(1.0F, fx)),
                                   __fmul_rn(__fsub_rn(1.0F, fy), fx),
                                   __fmul_rn(__fsub_rn(1.0F, fx), fy),
                                   __fmul_rn(fy, fx)};
            float depth_sum = 0.0F, auxiliary_sum = 0.0F, valid_weight = 0.0F;
#pragma unroll
            for (int corner = 0; corner != 4; ++corner)
            {
                const std::uint32_t offset = base + 2U * static_cast<std::uint32_t>(pixels[corner]);
                const float depth = pyramid[offset];
                if (depth == 0.0F || __float_as_uint(depth) == 0xd3800000U)
                    continue;
                depth_sum = __fmaf_rn(depth, weights[corner], depth_sum);
                valid_weight = __fadd_rn(valid_weight, weights[corner]);
                auxiliary_sum = __fmaf_rn(pyramid[offset + 1U], weights[corner], auxiliary_sum);
            }
            if (!(valid_weight > 0.0F))
                return;
            float auxiliary = __fdiv_rn(auxiliary_sum, valid_weight);
            float residual = __fdiv_rn(depth_sum, valid_weight);
            auxiliary = __fdiv_rn(auxiliary, __uint2float_rn(pyramid_scale));
            residual = __fsub_rn(residual, projected_depth);
            auxiliary = __fmul_rn(auxiliary, threshold_b);
            const float one_and_half_auxiliary = __fmul_rn(1.5F, auxiliary);
            float negative_limit = __fmul_rn(8.0F, level_threshold);
            negative_limit = __fadd_rn(negative_limit, one_and_half_auxiliary);
            if (-negative_limit > residual)
                return;
            if (selected_geometry > __fmul_rn(__uint_as_float(0x3fd9999aU), level_threshold))
                return;
            float raw_weight = __uint2float_rn(mode_b);
            const float attenuation_start = __fmul_rn(level_threshold, __uint_as_float(0x3fa66666U));
            if (attenuation_start < selected_geometry)
            {
                float ramp = __fdiv_rn(selected_geometry, level_threshold);
                ramp = __fsub_rn(ramp, __uint_as_float(0x3fa66666U));
                ramp = __fdiv_rn(ramp, __uint_as_float(0x3eccccd0U));
                raw_weight = __fmul_rn(raw_weight, __fsub_rn(1.0F, ramp));
            }
            float normalized = __fdiv_rn(residual, auxiliary);
            normalized = 1.0F > normalized ? fmaxf(normalized, -1.0F) : 1.0F;
            if (residual > -one_and_half_auxiliary)
            {
                raw_weight = __fmul_rn(raw_weight, pyramid_scale > 3U ? 4.0F : __uint2float_rn(pyramid_scale));
                raw_weight = __fmul_rn(raw_weight, 3.0F);
            }
            recovered_ooc_histogram_add(voxel, normalized, raw_weight);
        }

        __device__ __forceinline__ RecoveredFloat3
        recovered_depth_radius_transform(const DepthVotingMatrix4x4f& transform, const RecoveredFloat3& point)
        {
            const float* matrix = reinterpret_cast<const float*>(&transform);
            float values[4];
#pragma unroll
            for (std::uint32_t row = 0U; row < 4U; ++row)
            {
                const std::uint32_t base = row * 4U;
                float value = __fmul_rn(matrix[base + 1U], point.y);
                value = __fmaf_rn(matrix[base + 0U], point.x, value);
                value = __fmaf_rn(matrix[base + 2U], point.z, value);
                values[row] = __fadd_rn(matrix[base + 3U], value);
            }
            return {
                __fdiv_rn(values[0], values[3]),
                __fdiv_rn(values[1], values[3]),
                __fdiv_rn(values[2], values[3]),
            };
        }

        // The target's inlined vertical-neighbor branch has a distinct PTX reduction
        // order from the center/horizontal/orthogonal branches: x*m0 is rounded first,
        // then y*m1 is fused.  This is visible in the recovered BB0_1846 path and is
        // required for bit-exact cancellation after the world transform.
        __device__ __forceinline__ RecoveredFloat3 recovered_depth_radius_transform_vertical_neighbor(
            const DepthVotingMatrix4x4f& transform, const RecoveredFloat3& point)
        {
            const float* matrix = reinterpret_cast<const float*>(&transform);
            float values[4];
#pragma unroll
            for (std::uint32_t row = 0U; row < 4U; ++row)
            {
                const std::uint32_t base = row * 4U;
                float value = __fmul_rn(matrix[base + 0U], point.x);
                value = __fmaf_rn(matrix[base + 1U], point.y, value);
                value = __fmaf_rn(matrix[base + 2U], point.z, value);
                values[row] = __fadd_rn(matrix[base + 3U], value);
            }
            return {
                __fdiv_rn(values[0], values[3]),
                __fdiv_rn(values[1], values[3]),
                __fdiv_rn(values[2], values[3]),
            };
        }

        __device__ __forceinline__ float recovered_depth_radius_squared_distance(const RecoveredFloat3& left,
                                                                                 const RecoveredFloat3& right)
        {
            const float dx = __fsub_rn(left.x, right.x);
            const float dy = __fsub_rn(left.y, right.y);
            const float dz = __fsub_rn(left.z, right.z);
            return __fmaf_rn(dz, dz, __fmaf_rn(dx, dx, __fmul_rn(dy, dy)));
        }

        __global__ __launch_bounds__(128,
                                     1) void depth_radius_estimate_type1_kernel(const float* depth,
                                                                                float* radius,
                                                                                std::uint32_t level_offset,
                                                                                DepthVotingCalibrationCu calibration,
                                                                                DepthVotingMatrix4x4f transform,
                                                                                std::uint32_t kernel_offset)
        {
            const std::uint32_t logical = kernel_offset + blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t width = recovered_opaque_load<std::uint32_t>(calibration, 84U);
            const std::uint32_t height = recovered_opaque_load<std::uint32_t>(calibration, 88U);
            const std::uint32_t x = logical % width;
            const std::uint32_t y = logical / width;
            if (x >= width || y >= height)
                return;

            const std::uint32_t index = level_offset + y * width + x;
            radius[index] = 0.0F;
            const float center_depth = depth[index];
            if (center_depth == 0.0F)
                return;
            RecoveredFloat3 center_camera{};
            if (!recovered_calibration_unproject3(calibration,
                                                  __fadd_rn(__uint2float_rn(x), 0.5F),
                                                  __fadd_rn(__uint2float_rn(y), 0.5F),
                                                  center_depth,
                                                  center_camera,
                                                  5U))
                return;
            const RecoveredFloat3 center = recovered_depth_radius_transform(transform, center_camera);

            float radius_x = 0.0F;
            if (x > 0U)
            {
                const float neighbor_depth = depth[index - 1U];
                if (neighbor_depth != 0.0F)
                {
                    RecoveredFloat3 neighbor_camera{};
                    if (recovered_calibration_unproject3(calibration,
                                                         __fadd_rn(__uint2float_rn(x - 1U), 0.5F),
                                                         __fadd_rn(__uint2float_rn(y), 0.5F),
                                                         neighbor_depth,
                                                         neighbor_camera,
                                                         1U))
                    {
                        const RecoveredFloat3 neighbor = recovered_depth_radius_transform(transform, neighbor_camera);
                        radius_x = recovered_depth_radius_squared_distance(neighbor, center);
                    }
                }
            }
            if (x + 1U < width)
            {
                const float neighbor_depth = depth[index + 1U];
                if (neighbor_depth != 0.0F)
                {
                    RecoveredFloat3 neighbor_camera{};
                    if (recovered_calibration_unproject3(calibration,
                                                         __fadd_rn(__uint2float_rn(x + 1U), 0.5F),
                                                         __fadd_rn(__uint2float_rn(y), 0.5F),
                                                         neighbor_depth,
                                                         neighbor_camera,
                                                         1U))
                    {
                        const RecoveredFloat3 neighbor = recovered_depth_radius_transform(transform, neighbor_camera);
                        const float candidate = recovered_depth_radius_squared_distance(neighbor, center);
                        if (radius_x == 0.0F || candidate > radius_x)
                            radius_x = candidate;
                    }
                }
            }
            RecoveredFloat3 ortho_x_camera{};
            if (!recovered_calibration_unproject3(calibration,
                                                  __fadd_rn(__fadd_rn(__uint2float_rn(x), 1.0F), 0.5F),
                                                  __fadd_rn(__uint2float_rn(y), 0.5F),
                                                  center_depth,
                                                  ortho_x_camera,
                                                  3U))
                return;
            const RecoveredFloat3 ortho_x = recovered_depth_radius_transform(transform, ortho_x_camera);
            const float ortho_radius_x = recovered_depth_radius_squared_distance(ortho_x, center);
            if (radius_x == 0.0F)
            {
                radius_x = ortho_radius_x;
            }
            else
            {
                const float capped = __fmul_rn(10.0F, ortho_radius_x);
                if (capped < radius_x)
                    radius_x = capped;
            }

            float radius_y = 0.0F;
            if (y > 0U)
            {
                const float neighbor_depth = depth[index - width];
                if (neighbor_depth != 0.0F)
                {
                    RecoveredFloat3 neighbor_camera{};
                    if (recovered_calibration_unproject3(calibration,
                                                         __fadd_rn(__uint2float_rn(x), 0.5F),
                                                         __fadd_rn(__uint2float_rn(y - 1U), 0.5F),
                                                         neighbor_depth,
                                                         neighbor_camera,
                                                         2U))
                    {
                        const RecoveredFloat3 neighbor =
                            recovered_depth_radius_transform_vertical_neighbor(transform, neighbor_camera);
                        radius_y = recovered_depth_radius_squared_distance(neighbor, center);
                    }
                }
            }
            if (y + 1U < height)
            {
                const float neighbor_depth = depth[index + width];
                if (neighbor_depth != 0.0F)
                {
                    RecoveredFloat3 neighbor_camera{};
                    if (recovered_calibration_unproject3(calibration,
                                                         __fadd_rn(__uint2float_rn(x), 0.5F),
                                                         __fadd_rn(__uint2float_rn(y + 1U), 0.5F),
                                                         neighbor_depth,
                                                         neighbor_camera,
                                                         2U))
                    {
                        const RecoveredFloat3 neighbor =
                            recovered_depth_radius_transform_vertical_neighbor(transform, neighbor_camera);
                        const float candidate = recovered_depth_radius_squared_distance(neighbor, center);
                        if (radius_y == 0.0F || candidate > radius_y)
                            radius_y = candidate;
                    }
                }
            }
            RecoveredFloat3 ortho_y_camera{};
            if (!recovered_calibration_unproject3(calibration,
                                                  __fadd_rn(__uint2float_rn(x), 0.5F),
                                                  __fadd_rn(__fadd_rn(__uint2float_rn(y), 1.0F), 0.5F),
                                                  center_depth,
                                                  ortho_y_camera,
                                                  4U))
                return;
            const RecoveredFloat3 ortho_y = recovered_depth_radius_transform(transform, ortho_y_camera);
            const float ortho_radius_y = recovered_depth_radius_squared_distance(ortho_y, center);
            if (radius_y == 0.0F)
            {
                radius_y = ortho_radius_y;
            }
            else
            {
                const float capped = __fmul_rn(10.0F, ortho_radius_y);
                if (capped < radius_y)
                    radius_y = capped;
            }

            const float maximum = radius_x > radius_y ? radius_x : radius_y;
            radius[index] = __fsqrt_rn(maximum);
        }

        __device__ __forceinline__ bool recovered_voting_project_type1(const DepthVotingCalibrationCu& calibration,
                                                                       const RecoveredFloat3& point,
                                                                       RecoveredFloat3& projection)
        {
            if (point.z < 0.0000001F || point.z <= 0.0F)
                return false;
            const float* values = reinterpret_cast<const float*>(&calibration);
            const float inverse_z = __frcp_rn(point.z);
            float x = __fmul_rn(point.x, inverse_z);
            float y = __fmul_rn(point.y, inverse_z);
            if (recovered_opaque_load<std::uint32_t>(calibration, 112U) != 0U)
            {
                const float x2 = __fmul_rn(x, x);
                const float y2 = __fmul_rn(y, y);
                float r2 = __fadd_rn(x2, y2);
                float norm2 = 1.0F;
                if (r2 > values[8U])
                {
                    norm2 = __fdiv_rn(values[8U], r2);
                    r2 = values[8U];
                }
                const float r4 = __fmul_rn(r2, r2);
                float radial = __fmaf_rn(values[0U], r2, __fmul_rn(values[1U], r4));
                radial = __fmaf_rn(__fmul_rn(values[2U], r2), r4, radial);
                radial = __fmaf_rn(r4, __fmul_rn(values[3U], r4), radial);
                const float pdist = __fmaf_rn(values[7U], r4, __fmaf_rn(values[6U], r2, 1.0F));
                const float dx_inner = __fmaf_rn(__fmul_rn(__fadd_rn(values[5U], values[5U]), x),
                                                 y,
                                                 __fmul_rn(values[4U], __fmaf_rn(x, __fmul_rn(3.0F, x), y2)));
                const float dy_inner = __fmaf_rn(__fmul_rn(__fadd_rn(values[4U], values[4U]), x),
                                                 y,
                                                 __fmul_rn(values[5U], __fmaf_rn(y, __fmul_rn(3.0F, y), x2)));
                x = __fadd_rn(x, __fmaf_rn(norm2, __fmul_rn(dx_inner, pdist), __fmul_rn(x, radial)));
                y = __fadd_rn(y, __fmaf_rn(norm2, __fmul_rn(dy_inner, pdist), __fmul_rn(y, radial)));
            }
            const float focal = values[23U];
            const float cx = values[24U];
            const float cy = values[25U];
            const float b1 = values[26U];
            const float b2 = values[27U];
            const float width_center =
                __fmaf_rn(__int2float_rn(recovered_opaque_load<std::int32_t>(calibration, 84U)), 0.5F, cx);
            const float height_center =
                __fmaf_rn(__int2float_rn(recovered_opaque_load<std::int32_t>(calibration, 88U)), 0.5F, cy);
            float projected_x = __fmul_rn(b1, x);
            projected_x = __fmaf_rn(focal, x, projected_x);
            projected_x = __fmaf_rn(b2, y, projected_x);
            projected_x = __fadd_rn(width_center, projected_x);
            const float projected_y = __fmaf_rn(focal, y, height_center);
            projection = {projected_x, projected_y, point.z};
            return true;
        }

        __global__
            __launch_bounds__(128,
                              1) void depth_neighbor_votes_type1_kernel(std::int32_t* votes,
                                                                        const float* reference_depth,
                                                                        const float* reference_radius,
                                                                        const std::uint8_t* neighbor_inlier_mask,
                                                                        const float* neighbor_depth_levels,
                                                                        const float* neighbor_radius_levels,
                                                                        std::uint32_t reference_level,
                                                                        DepthVotingCalibrationCu reference_calibration,
                                                                        DepthVotingMatrix4x4f reference_to_neighbor,
                                                                        std::uint32_t neighbor_levels,
                                                                        DepthVotingCalibrationCu neighbor_calibration,
                                                                        std::uint32_t* counter_inlier_supports,
                                                                        std::uint32_t* counter_inlier_intersects,
                                                                        std::uint32_t* counter_inlier_does_not_reach,
                                                                        std::uint32_t* counter_inlier_no_depth,
                                                                        std::uint32_t* counter_outlier_supports,
                                                                        std::uint32_t* counter_outlier_intersects,
                                                                        std::uint32_t* counter_outlier_does_not_reach,
                                                                        std::uint32_t kernel_offset)
        {
            __shared__ std::uint32_t local_counters[7];
            if (threadIdx.x < 7U)
                local_counters[threadIdx.x] = 0U;
            __syncthreads();

            const std::uint32_t logical = kernel_offset + blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t width = recovered_opaque_load<std::uint32_t>(reference_calibration, 84U);
            const std::uint32_t height = recovered_opaque_load<std::uint32_t>(reference_calibration, 88U);
            const std::uint32_t x = logical % width;
            const std::uint32_t y = logical / width;
            if (x < width && y < height)
            {
                const std::uint32_t index = y * width + x;
                const float depth = reference_depth[index];
                if (depth != 0.0F)
                {
                    RecoveredFloat3 point{};
                    const bool unprojected = recovered_calibration_unproject3(reference_calibration,
                                                                              __fadd_rn(__uint2float_rn(x), 0.5F),
                                                                              __fadd_rn(__uint2float_rn(y), 0.5F),
                                                                              depth,
                                                                              point,
                                                                              1U);
                    if (unprojected)
                        point = recovered_depth_radius_transform_vertical_neighbor(reference_to_neighbor, point);
                    RecoveredFloat3 neighbor_projection{};
                    if (unprojected && recovered_calibration_project3(neighbor_calibration, point, neighbor_projection))
                    {
                        const float ref_radius = reference_radius[index];
                        const bool inlier = neighbor_inlier_mask[index] != 0U;
                        float neighbor_depth = 0.0F;
                        float neighbor_depth_of_point = 3.402823466e+38F;
                        float neighbor_radius = 0.0F;
                        std::uint32_t level_offset = 0U;
                        const std::uint32_t neighbor_width =
                            recovered_opaque_load<std::uint32_t>(neighbor_calibration, 84U);
                        const std::uint32_t neighbor_height =
                            recovered_opaque_load<std::uint32_t>(neighbor_calibration, 88U);
                        for (std::uint32_t level = 0U; level < neighbor_levels; ++level)
                        {
                            const std::uint32_t downscale = 1U << level;
                            const std::uint32_t level_width = (neighbor_width + downscale - 1U) / downscale;
                            const std::uint32_t level_height = (neighbor_height + downscale - 1U) / downscale;
                            const std::int32_t nx =
                                __float2int_rz(__fdiv_rn(neighbor_projection.x, __uint2float_rn(downscale)));
                            const std::int32_t ny =
                                __float2int_rz(__fdiv_rn(neighbor_projection.y, __uint2float_rn(downscale)));
                            if (nx < 0 || ny < 0 || nx >= static_cast<std::int32_t>(level_width) ||
                                ny >= static_cast<std::int32_t>(level_height))
                                break;
                            const std::uint32_t level_index = level_offset +
                                                              static_cast<std::uint32_t>(ny) * level_width +
                                                              static_cast<std::uint32_t>(nx);
                            const float candidate_depth = neighbor_depth_levels[level_index];
                            float candidate_radius = neighbor_radius_levels[level_index];
                            level_offset += level_width * level_height;
                            if (candidate_depth == 0.0F)
                                continue;
                            if (level > reference_level)
                                candidate_radius =
                                    __fdiv_rn(candidate_radius, __uint2float_rn(1U << (level - reference_level)));
                            neighbor_depth_of_point = neighbor_projection.z;
                            neighbor_depth = candidate_depth;
                            neighbor_radius = candidate_radius;
                            break;
                        }
                        const float support_distance =
                            neighbor_depth != 0.0F ? __fmul_rn(fmaxf(ref_radius, neighbor_radius), 2.0F) : 0.0F;
                        std::int32_t chosen_vote = 0;
                        std::uint32_t counter = 7U;
                        if (neighbor_depth == 0.0F)
                        {
                            if (inlier)
                            {
                                chosen_vote = -50;
                                counter = 3U;
                            }
                        }
                        else if (neighbor_depth > __fadd_rn(neighbor_depth_of_point, support_distance))
                        {
                            if (inlier)
                            {
                                chosen_vote = -50;
                                counter = 1U;
                            }
                            else
                            {
                                chosen_vote = -15;
                                counter = 5U;
                            }
                        }
                        else if (neighbor_depth > __fsub_rn(neighbor_depth_of_point, support_distance))
                        {
                            if (inlier)
                            {
                                chosen_vote = 100;
                                counter = 0U;
                            }
                            else
                            {
                                chosen_vote = 15;
                                counter = 4U;
                            }
                        }
                        else
                        {
                            if (inlier)
                            {
                                chosen_vote = -50;
                                counter = 2U;
                            }
                            else
                            {
                                chosen_vote = -5;
                                counter = 6U;
                            }
                        }
                        if (counter < 7U)
                            atomicAdd(&local_counters[counter], 1U);
                        if (chosen_vote != 0)
                            votes[index] += chosen_vote;
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0U)
            {
                atomicAdd(counter_inlier_supports, local_counters[0]);
                atomicAdd(counter_inlier_intersects, local_counters[1]);
                atomicAdd(counter_inlier_does_not_reach, local_counters[2]);
                atomicAdd(counter_inlier_no_depth, local_counters[3]);
                atomicAdd(counter_outlier_supports, local_counters[4]);
                atomicAdd(counter_outlier_intersects, local_counters[5]);
                atomicAdd(counter_outlier_does_not_reach, local_counters[6]);
            }
        }

        __global__ __launch_bounds__(128, 1) void depth_neighbor_occlusion_votes_type1_kernel(
            std::int32_t* votes,
            const float* reference_depth,
            const float* reference_radius,
            const std::uint8_t* neighbor_inlier_mask,
            std::uint32_t neighbor_level,
            const float* neighbor_depth_levels,
            const float* neighbor_radius_levels,
            std::uint32_t reference_level,
            DepthVotingCalibrationCu reference_calibration,
            DepthVotingMatrix4x4f neighbor_to_reference,
            DepthVotingCalibrationCu neighbor_calibration,
            std::uint32_t* counter_inlier_occludes,
            std::uint32_t* counter_outlier_occludes,
            std::uint32_t kernel_offset)
        {
            __shared__ std::uint32_t local_inlier;
            __shared__ std::uint32_t local_outlier;
            if (threadIdx.x == 0U)
            {
                local_inlier = 0U;
                local_outlier = 0U;
            }
            __syncthreads();

            const std::uint32_t neighbor_width = recovered_opaque_load<std::uint32_t>(neighbor_calibration, 84U);
            const std::uint32_t neighbor_height = recovered_opaque_load<std::uint32_t>(neighbor_calibration, 88U);
            std::uint32_t level_offset = 0U;
            std::uint32_t level_width = 0U;
            std::uint32_t level_height = 0U;
            std::uint32_t downscale = 1U;
            for (std::uint32_t level = 0U; level <= neighbor_level; ++level)
            {
                downscale = 1U << level;
                level_width = (neighbor_width + downscale - 1U) / downscale;
                level_height = (neighbor_height + downscale - 1U) / downscale;
                if (level < neighbor_level)
                    level_offset += level_width * level_height;
            }
            const std::uint32_t logical = kernel_offset + blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t nx = logical % level_width;
            const std::uint32_t ny = logical / level_width;
            if (nx < level_width && ny < level_height)
            {
                const std::uint32_t neighbor_index = level_offset + ny * level_width + nx;
                const float neighbor_depth = neighbor_depth_levels[neighbor_index];
                const float neighbor_radius = neighbor_radius_levels[neighbor_index];
                if (neighbor_depth != 0.0F)
                {
                    const float projection_x =
                        __fmul_rn(__uint2float_rn(downscale), __fadd_rn(__uint2float_rn(nx), 0.5F));
                    const float projection_y =
                        __fmul_rn(__uint2float_rn(downscale), __fadd_rn(__uint2float_rn(ny), 0.5F));
                    RecoveredFloat3 point{};
                    const bool unprojected = recovered_calibration_unproject3(
                        neighbor_calibration, projection_x, projection_y, neighbor_depth, point, 1U);
                    if (unprojected)
                        point = recovered_depth_radius_transform_vertical_neighbor(neighbor_to_reference, point);
                    RecoveredFloat3 reference_projection{};
                    if (unprojected &&
                        recovered_calibration_project3(reference_calibration, point, reference_projection))
                    {
                        const std::int32_t x0 = __float2int_rz(__fsub_rn(reference_projection.x, 0.5F));
                        const std::int32_t y0 = __float2int_rz(__fsub_rn(reference_projection.y, 0.5F));
                        const std::int32_t width = recovered_opaque_load<std::int32_t>(reference_calibration, 84U);
                        const std::int32_t height = recovered_opaque_load<std::int32_t>(reference_calibration, 88U);
                        if (x0 >= 0 && y0 >= 0 && x0 < width && y0 < height)
                        {
                            const std::int32_t y_end = min(y0 + 2, height);
                            const std::int32_t x_end = min(x0 + 2, width);
                            for (std::int32_t y = y0; y < y_end; ++y)
                            {
                                for (std::int32_t x = x0; x < x_end; ++x)
                                {
                                    const std::uint32_t index = static_cast<std::uint32_t>(y * width + x);
                                    const float ref_depth = reference_depth[index];
                                    if (ref_depth == 0.0F)
                                        continue;
                                    const bool inlier = neighbor_inlier_mask[index] != 0U;
                                    const float support_radius = fmaxf(reference_radius[index], neighbor_radius);
                                    const float support_threshold =
                                        __fmaf_rn(support_radius, 2.0F, reference_projection.z);
                                    if (ref_depth > support_threshold)
                                    {
                                        atomicAdd(votes + index, inlier ? -12 : -4);
                                        atomicAdd(inlier ? &local_inlier : &local_outlier, 1U);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0U)
            {
                atomicAdd(counter_inlier_occludes, local_inlier);
                atomicAdd(counter_outlier_occludes, local_outlier);
            }
            (void)reference_level;
        }

        __device__ __forceinline__ RecoveredFloat3 unpack_recovered_normal(const std::uint8_t* packed);

        __global__ __launch_bounds__(128, 1) void patchmatch_rotate_normals_kernel(float* normals,
                                                                                   PatchMatchCamera camera)
        {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t base = 3U * index;
            const float x = normals[base + 0U];
            const float y = normals[base + 1U];
            const float z = normals[base + 2U];
            const float* transform =
                reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(&camera) + 208U);
            const auto row = [&](std::uint32_t row_index)
            {
                const std::uint32_t offset = 4U * row_index;
                float value = __fmul_rn(y, transform[offset + 1U]);
                value = __fmaf_rn(x, transform[offset + 0U], value);
                value = __fmaf_rn(z, transform[offset + 2U], value);
                return __fadd_rn(transform[offset + 3U], value);
            };
            const float transformed_x = row(0U);
            const float transformed_y = row(1U);
            const float transformed_z = row(2U);
            const float transformed_w = row(3U);
            normals[base + 0U] = __fdiv_rn(transformed_x, transformed_w);
            normals[base + 1U] = __fdiv_rn(transformed_y, transformed_w);
            normals[base + 2U] = __fdiv_rn(transformed_z, transformed_w);
        }

        __device__ __forceinline__ void recovered_insert_sorted_cost(float value, float* values, std::uint32_t& count)
        {
            std::uint32_t insertion = 0U;
            while (insertion < count && values[insertion] < value)
                ++insertion;
            if (insertion < count)
            {
                std::uint32_t last = count;
                if (count < 10U)
                {
                    ++count;
                }
                else
                {
                    --last;
                }
                while (last > insertion)
                {
                    values[last] = values[last - 1U];
                    --last;
                }
            }
            else if (count < 10U)
            {
                ++count;
            }
            if (insertion < 10U)
                values[insertion] = value;
        }

        __global__ __launch_bounds__(128, 1) void patchmatch_average_costs_kernel(const float* per_neighbor_cost,
                                                                                  float* average_cost,
                                                                                  std::uint8_t* auxiliary,
                                                                                  std::uint32_t width_original,
                                                                                  std::uint32_t height_original,
                                                                                  std::uint32_t depth_downscale,
                                                                                  std::uint32_t hypotheses_per_pixel,
                                                                                  std::uint32_t neighbor_count,
                                                                                  std::uint32_t is_checkboard,
                                                                                  std::uint32_t checkboard_step,
                                                                                  std::uint32_t only_each_fourth_pixel,
                                                                                  std::uint32_t pixel_offset)
        {
            constexpr std::uint32_t capacity = 128U * 1024U;
            const std::uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t hypothesis = global_id % hypotheses_per_pixel;
            const std::uint32_t temporary_index = global_id / hypotheses_per_pixel;
            const std::uint32_t global_index = pixel_offset + temporary_index;
            const std::uint32_t width = (width_original + depth_downscale - 1U) / depth_downscale;
            const std::uint32_t height = (height_original + depth_downscale - 1U) / depth_downscale;
            std::uint32_t x = 0U;
            std::uint32_t y = 0U;
            if (is_checkboard != 0U)
            {
                if (only_each_fourth_pixel == 0U)
                {
                    const std::uint32_t width_half = (width + 1U) / 2U;
                    y = global_index / width_half;
                    x = ((y % 2U) + checkboard_step) % 2U + 2U * (global_index % width_half);
                }
                else
                {
                    const std::uint32_t width_part = (width + 3U) / 4U;
                    y = 1U + 2U * global_index / width_part;
                    x = 1U + (((y / 2U) % 2U + checkboard_step) % 2U) * 2U + 4U * (global_index % width_part);
                }
            }
            else if (only_each_fourth_pixel == 0U)
            {
                y = global_index / width;
                x = global_index % width;
            }
            else
            {
                const std::uint32_t width_half = (width + 1U) / 2U;
                y = 1U + 2U * (global_index / width_half);
                x = 1U + 2U * (global_index % width_half);
            }
            if (x >= width || y >= height)
                return;

            float sorted_costs[10];
            std::uint32_t cost_count = 0U;
            for (std::uint32_t neighbor = 0U; neighbor < neighbor_count; ++neighbor)
            {
                const float value = per_neighbor_cost[neighbor * capacity * hypotheses_per_pixel +
                                                      temporary_index * hypotheses_per_pixel + hypothesis];
                if (value == -1.0F)
                    continue;
                recovered_insert_sorted_cost(value, sorted_costs, cost_count);
            }

            const std::uint32_t average_index = hypothesis * capacity + temporary_index;
            if (cost_count == 0U)
            {
                average_cost[average_index] = -1.0F;
                for (std::uint32_t neighbor = 0U; neighbor < neighbor_count; neighbor += 8U)
                {
                    auxiliary[(neighbor / 8U) * capacity * hypotheses_per_pixel + average_index] = 0U;
                }
                return;
            }

            const float best = sorted_costs[0];
            const float scaled_threshold = __fmul_rn(best, 1.15F);
            const float threshold = scaled_threshold > 0.1F ? scaled_threshold : 0.1F;
            float sum = best;
            std::uint32_t good_count = 1U;
            while (good_count < cost_count)
            {
                const float value = sorted_costs[good_count];
                if (value > threshold)
                    break;
                sum = __fadd_rn(sum, value);
                ++good_count;
            }
            float result = __fdiv_rn(sum, __uint2float_rn(good_count));
            if (result < 0.075F)
            {
                // The old libdevice expansion of pow(min(10,n), 1) is exact for
                // n=1..6 and 8..10, but returns the next float above 7 for n=7.
                // This one-ULP artifact is visible in target iteration-8/final costs.
                const float divisor = good_count == 7U ? __int_as_float(0x40e00001) : __uint2float_rn(good_count);
                result = __fdiv_rn(result, divisor);
            }
            average_cost[average_index] = result;

            std::uint8_t mask = 0U;
            std::uint32_t bit = 0U;
            for (std::uint32_t neighbor = 0U; neighbor < neighbor_count; ++neighbor, ++bit)
            {
                const float value = per_neighbor_cost[neighbor * capacity * hypotheses_per_pixel +
                                                      temporary_index * hypotheses_per_pixel + hypothesis];
                const bool inlier = value != -1.0F && value <= threshold;
                mask = static_cast<std::uint8_t>(mask | (static_cast<std::uint32_t>(inlier) << bit));
                if (bit == 7U || neighbor + 1U == neighbor_count)
                {
                    auxiliary[(neighbor / 8U) * capacity * hypotheses_per_pixel + average_index] = mask;
                    mask = 0U;
                    bit = static_cast<std::uint32_t>(-1);
                }
            }
        }

        __device__ __forceinline__ std::uint8_t recovered_pack_wta_component(float value)
        {
            value = value < 1.0F ? value : 1.0F;
            if (value < -1.0F)
                return 0U;
            const float shifted = __fadd_rn(value, 1.0F);
            const float scaled = __fmul_rn(shifted, 255.0F);
            return static_cast<std::uint8_t>(__fmaf_rn(scaled, 0.5F, 0.5F));
        }

        __global__ __launch_bounds__(128, 1) void patchmatch_wta_kernel(float* depth,
                                                                        std::uint8_t* normal,
                                                                        float* cost,
                                                                        std::uint32_t width_original,
                                                                        std::uint32_t height_original,
                                                                        std::uint32_t depth_downscale,
                                                                        std::uint32_t hypotheses_per_pixel,
                                                                        const float* candidate_depth,
                                                                        const float* candidate_normal,
                                                                        const float* average_cost,
                                                                        std::uint8_t* winner,
                                                                        std::uint32_t is_checkboard,
                                                                        std::uint32_t checkboard_step,
                                                                        std::uint32_t only_each_fourth_pixel,
                                                                        std::uint32_t pixel_offset)
        {
            constexpr std::uint32_t capacity = 128U * 1024U;
            const std::uint32_t temporary_index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t global_index = pixel_offset + temporary_index;
            const std::uint32_t width = (width_original + depth_downscale - 1U) / depth_downscale;
            const std::uint32_t height = (height_original + depth_downscale - 1U) / depth_downscale;
            std::uint32_t x = 0U;
            std::uint32_t y = 0U;
            if (is_checkboard != 0U)
            {
                if (only_each_fourth_pixel == 0U)
                {
                    const std::uint32_t width_half = (width + 1U) / 2U;
                    y = global_index / width_half;
                    x = ((y % 2U) + checkboard_step) % 2U + 2U * (global_index % width_half);
                }
                else
                {
                    const std::uint32_t width_part = (width + 3U) / 4U;
                    y = 1U + 2U * global_index / width_part;
                    x = 1U + (((y / 2U) % 2U + checkboard_step) % 2U) * 2U + 4U * (global_index % width_part);
                }
            }
            else if (only_each_fourth_pixel == 0U)
            {
                y = global_index / width;
                x = global_index % width;
            }
            else
            {
                const std::uint32_t width_half = (width + 1U) / 2U;
                y = 1U + 2U * (global_index / width_half);
                x = 1U + 2U * (global_index % width_half);
            }
            if (x >= width || y >= height)
                return;

            const std::uint32_t pixel = y * width + x;
            float best_cost = cost[pixel];
            std::int32_t best_hypothesis = -1;
            for (std::uint32_t hypothesis = 0U; hypothesis < hypotheses_per_pixel; ++hypothesis)
            {
                const float candidate = average_cost[hypothesis * capacity + temporary_index];
                if (candidate != -1.0F && (best_cost == -1.0F || candidate < best_cost))
                {
                    best_cost = candidate;
                    best_hypothesis = static_cast<std::int32_t>(hypothesis);
                }
            }
            if (best_hypothesis < 0)
            {
                winner[temporary_index] = 255U;
                return;
            }

            const std::uint32_t hypothesis = static_cast<std::uint32_t>(best_hypothesis);
            winner[temporary_index] = static_cast<std::uint8_t>(hypothesis);
            const std::uint32_t candidate_index = hypothesis * capacity + temporary_index;
            depth[pixel] = candidate_depth[candidate_index];
            const std::uint32_t normal_source = 3U * candidate_index;
            const std::uint32_t normal_destination = 3U * pixel;
            normal[normal_destination + 0U] = recovered_pack_wta_component(candidate_normal[normal_source + 0U]);
            normal[normal_destination + 1U] = recovered_pack_wta_component(candidate_normal[normal_source + 1U]);
            normal[normal_destination + 2U] = recovered_pack_wta_component(candidate_normal[normal_source + 2U]);
            cost[pixel] = best_cost;
        }

        __device__ __forceinline__ float recovered_bilateral_exp(float value)
        {
            // Reproduce the old libdevice expf range reduction embedded in the
            // recovered sm_30 payload.  A direct CUDA-12 expf/__expf changes the
            // accumulated depth by one or more ULPs on real target inputs.
            const float log2e = __int_as_float(0x3fb8aa3b);
            const float negative_ln2_hi = __int_as_float(0xbf317200);
            const float negative_ln2_lo = __int_as_float(0xb5bfbe8e);
            if (value < -105.0F)
                return 0.0F;
            if (value > 105.0F)
                return __int_as_float(0x7f800000);
            const float scaled = __fmul_rn(value, log2e);
            const float whole = truncf(scaled);
            float remainder = __fmaf_rn(whole, negative_ln2_hi, value);
            remainder = __fmaf_rn(whole, negative_ln2_lo, remainder);
            const float fractional = exp2f(__fmul_rn(remainder, log2e));
            const float integral = exp2f(whole);
            return __fmul_rn(fractional, integral);
        }

        __device__ __forceinline__ float recovered_warp_exp_from_c0(float c0)
        {
            const float log2e = __int_as_float(0x3fb8aa3b);
            const float value = -c0;
            if (c0 < -105.0F)
                return __int_as_float(0x7f800000);
            if (c0 > 105.0F)
                return 0.0F;
            const float whole = truncf(__fmul_rn(c0, -log2e));
            float remainder = __fmaf_rn(whole, __int_as_float(0xbf317200), value);
            remainder = __fmaf_rn(whole, __int_as_float(0xb5bfbe8e), remainder);
            return __fmul_rn(__expf(remainder), exp2f(whole));
        }

        __device__ __forceinline__ std::uint8_t recovered_pack_bilateral_component(float value)
        {
            value = value < 1.0F ? value : 1.0F;
            if (value < -1.0F)
                return 0U;
            const float shifted = __fadd_rn(value, 1.0F);
            const float scaled = __fmul_rn(shifted, 255.0F);
            return static_cast<std::uint8_t>(__fmaf_rn(scaled, 0.5F, 0.5F));
        }

        __device__ __forceinline__ float recovered_bilateral_intensity(std::uint8_t value)
        {
            return __fdiv_rn(__uint2float_rn(value), 255.0F);
        }

        __device__ __forceinline__ float recovered_bilateral_intensity(std::uint16_t value)
        {
            return __fdiv_rn(__uint2float_rn(value), 65535.0F);
        }

        __device__ __forceinline__ float recovered_bilateral_intensity(float value)
        {
            return value;
        }

        // Source reconstruction of the three image-type specializations of
        // cuda::pm_bilateral_depth_map_filtering_r3.  The 7x7 traversal and all
        // accumulations retain the target's scalar instruction order.
        template <class Image>
        __global__ __launch_bounds__(128, 1) void patchmatch_bilateral_kernel(const float* depth,
                                                                              const std::uint8_t* normal,
                                                                              const Image* image,
                                                                              float* filtered_depth,
                                                                              std::uint8_t* filtered_normal,
                                                                              std::uint32_t width,
                                                                              std::uint32_t height,
                                                                              float sigma_d,
                                                                              float sigma_r,
                                                                              std::uint32_t pixel_offset)
        {
            const std::uint32_t global_index = pixel_offset + blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t y = global_index / width;
            if (y >= height)
                return;
            const std::uint32_t x = global_index % width;
            const std::uint32_t center_index = y * width + x;
            const float inverse_range_variance = __frcp_rn(__fmul_rn(__fadd_rn(sigma_r, sigma_r), sigma_r));
            const float center_intensity = recovered_bilateral_intensity(image[center_index]);
            const float central_depth = depth[center_index];
            const float spatial_denominator = __fmul_rn(__fadd_rn(sigma_d, sigma_d), sigma_d);
            float depth_sum = 0.0F;
            RecoveredFloat3 normal_sum{0.0F, 0.0F, 0.0F};
            float weight_sum = 0.0F;
            std::uint32_t samples = 0U;

            for (std::int32_t dy = -3; dy <= 3; ++dy)
            {
                for (std::int32_t dx = -3; dx <= 3; ++dx)
                {
                    const std::int32_t neighbor_x = static_cast<std::int32_t>(x) + dx;
                    const std::int32_t neighbor_y = static_cast<std::int32_t>(y) + dy;
                    if (neighbor_x < 0 || neighbor_y < 0 || neighbor_x >= static_cast<std::int32_t>(width) ||
                        neighbor_y >= static_cast<std::int32_t>(height))
                        continue;
                    const std::uint32_t neighbor_index =
                        static_cast<std::uint32_t>(neighbor_y) * width + static_cast<std::uint32_t>(neighbor_x);
                    const float neighbor_depth = depth[neighbor_index];
                    if (neighbor_depth == 0.0F)
                        continue;
                    const float neighbor_intensity = recovered_bilateral_intensity(image[neighbor_index]);
                    const float intensity_difference = __fsub_rn(neighbor_intensity, center_intensity);
                    const float intensity_squared = __fmaf_rn(intensity_difference, intensity_difference, 0.0F);
                    const float range_argument = __fmul_rn(inverse_range_variance, intensity_squared);
                    const float range_weight = recovered_bilateral_exp(-range_argument);
                    const std::int32_t distance_squared = dx * dx + dy * dy;
                    const float spatial_argument = __fdiv_rn(-__int2float_rn(distance_squared), spatial_denominator);
                    const float spatial_weight = recovered_bilateral_exp(spatial_argument);
                    const float weight = __fmul_rn(range_weight, spatial_weight);
                    const RecoveredFloat3 unpacked = unpack_recovered_normal(normal + 3U * neighbor_index);
                    depth_sum = __fmaf_rn(neighbor_depth, weight, depth_sum);
                    normal_sum.x = __fmaf_rn(weight, unpacked.x, normal_sum.x);
                    normal_sum.y = __fmaf_rn(weight, unpacked.y, normal_sum.y);
                    normal_sum.z = __fmaf_rn(weight, unpacked.z, normal_sum.z);
                    weight_sum = __fadd_rn(weight_sum, weight);
                    ++samples;
                }
            }

            float result_depth = 0.0F;
            RecoveredFloat3 result_normal{0.0F, 0.0F, 0.0F};
            if (weight_sum > 0.0F && (central_depth != 0.0F || samples >= 15U))
            {
                result_depth = __fdiv_rn(depth_sum, weight_sum);
                const float squared =
                    __fmaf_rn(normal_sum.z,
                              normal_sum.z,
                              __fmaf_rn(normal_sum.x, normal_sum.x, __fmul_rn(normal_sum.y, normal_sum.y)));
                const float length = sqrtf(squared);
                result_normal.x = __fdiv_rn(normal_sum.x, length);
                result_normal.y = __fdiv_rn(normal_sum.y, length);
                result_normal.z = __fdiv_rn(normal_sum.z, length);
            }
            if (result_depth != 0.0F)
            {
                const float squared =
                    __fmaf_rn(result_normal.z,
                              result_normal.z,
                              __fmaf_rn(result_normal.x, result_normal.x, __fmul_rn(result_normal.y, result_normal.y)));
                const float length = sqrtf(squared);
                if (!(length >= 0.9F && length <= 1.1F))
                {
                    result_depth = 0.0F;
                    result_normal = {0.0F, 0.0F, 0.0F};
                }
            }
            filtered_depth[center_index] = result_depth;
            filtered_normal[3U * center_index + 0U] = recovered_pack_bilateral_component(result_normal.x);
            filtered_normal[3U * center_index + 1U] = recovered_pack_bilateral_component(result_normal.y);
            filtered_normal[3U * center_index + 2U] = recovered_pack_bilateral_component(result_normal.z);
        }

        __device__ __forceinline__ RecoveredFloat3 unpack_recovered_normal(const std::uint8_t* packed)
        {
            RecoveredFloat3 result;
            result.x =
                __fadd_rn(__fdiv_rn(__fadd_rn(__uint2float_rn(packed[0]), __uint2float_rn(packed[0])), 255.0F), -1.0F);
            result.y =
                __fadd_rn(__fdiv_rn(__fadd_rn(__uint2float_rn(packed[1]), __uint2float_rn(packed[1])), 255.0F), -1.0F);
            result.z =
                __fadd_rn(__fdiv_rn(__fadd_rn(__uint2float_rn(packed[2]), __uint2float_rn(packed[2])), 255.0F), -1.0F);
            const float squared =
                __fmaf_rn(result.z, result.z, __fmaf_rn(result.x, result.x, __fmul_rn(result.y, result.y)));
            const float length = sqrtf(squared);
            result.x = __fdiv_rn(result.x, length);
            result.y = __fdiv_rn(result.y, length);
            result.z = __fdiv_rn(result.z, length);
            return result;
        }

        __device__ __forceinline__ float recovered_squared_distance(const RecoveredFloat3& left,
                                                                    const RecoveredFloat3& right);
        __device__ __forceinline__ RecoveredFloat3 recovered_subtract(const RecoveredFloat3& left,
                                                                      const RecoveredFloat3& right);

        __device__ __forceinline__ RecoveredFloat3 recovered_final_unproject_perspective(
            const PatchMatchCamera& camera, float sample_x, float sample_y, float depth, std::uint32_t downscale)
        {
            float local_y = __fmaf_rn(__uint2float_rn(downscale), sample_y, -camera.cy);
            local_y = __fmaf_rn(__uint2float_rn(camera.height_original), -0.5F, local_y);
            local_y = __fdiv_rn(local_y, camera.f);
            float local_x = __fmaf_rn(__uint2float_rn(downscale), sample_x, -camera.cx);
            local_x = __fmaf_rn(__uint2float_rn(camera.width_original), -0.5F, local_x);
            local_x = __fmaf_rn(-local_y, camera.b2, local_x);
            local_x = __fdiv_rn(local_x, __fadd_rn(camera.f, camera.b1));

            const float x = __fmul_rn(local_x, depth);
            const float y = __fmul_rn(local_y, depth);
            const float z = depth;
            const float* transform =
                reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(&camera) + 208U);
            float transformed[4];
#pragma unroll
            for (std::uint32_t row = 0U; row < 4U; ++row)
            {
                const std::uint32_t base = 4U * row;
                float value = __fmul_rn(y, transform[base + 1U]);
                value = __fmaf_rn(x, transform[base + 0U], value);
                value = __fmaf_rn(z, transform[base + 2U], value);
                transformed[row] = __fadd_rn(transform[base + 3U], value);
            }
            return {
                __fdiv_rn(transformed[0], transformed[3]),
                __fdiv_rn(transformed[1], transformed[3]),
                __fdiv_rn(transformed[2], transformed[3]),
            };
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_final_transform_point(const PatchMatchCamera& camera,
                                                                                   const RecoveredFloat3& local)
        {
            const float* transform =
                reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(&camera) + 208U);
            float transformed[4];
#pragma unroll
            for (std::uint32_t row = 0U; row < 4U; ++row)
            {
                const std::uint32_t base = 4U * row;
                float value = __fmul_rn(local.y, transform[base + 1U]);
                value = __fmaf_rn(local.x, transform[base + 0U], value);
                value = __fmaf_rn(local.z, transform[base + 2U], value);
                transformed[row] = __fadd_rn(transform[base + 3U], value);
            }
            return {
                __fdiv_rn(transformed[0], transformed[3]),
                __fdiv_rn(transformed[1], transformed[3]),
                __fdiv_rn(transformed[2], transformed[3]),
            };
        }

        __device__ __forceinline__ void
        recovered_final_zero_candidates(float* candidate_depth, float* candidate_normal, std::uint32_t temporary_index)
        {
            constexpr std::uint32_t capacity = 128U * 1024U;
#pragma unroll
            for (std::uint32_t hypothesis = 0U; hypothesis < 8U; ++hypothesis)
            {
                candidate_depth[hypothesis * capacity + temporary_index] = 0.0F;
                const std::uint32_t normal_index = 3U * (hypothesis * capacity + temporary_index);
                candidate_normal[normal_index + 0U] = 0.0F;
                candidate_normal[normal_index + 1U] = 0.0F;
                candidate_normal[normal_index + 2U] = 0.0F;
            }
        }

        // CUDA 6-era libdevice expansions recovered from the target sm_30 PTX.  The
        // current CUDA toolkit's sinf/cosf/tanf/asinf implementations differ by a few
        // ULP, which is enough to change propagated depth bits.  Keep every rounding
        // point and coefficient explicit, including the integer Payne-Hanek reducer
        // used beyond the fast branch (|x| > 0x1.9c8fp+16).
        struct RecoveredLegacySinCos
        {
            float sine;
            float cosine;
        };

        __device__ __forceinline__ float recovered_legacy_reduce_trig(float value, int& quadrant)
        {
            const float absolute = fabsf(value);
            if (absolute == __int_as_float(0x7f800000))
            {
                value = __fmul_rn(value, 0.0F);
            }
            quadrant = __float2int_rn(__fmul_rn(value, __int_as_float(0x3f22f983)));
            const float negative_quadrant = -__int2float_rn(quadrant);
            float reduced = __fmaf_rn(negative_quadrant, __int_as_float(0x3fc90fda), value);
            reduced = __fmaf_rn(negative_quadrant, __int_as_float(0x33a22168), reduced);
            reduced = __fmaf_rn(negative_quadrant, __int_as_float(0x27c234c5), reduced);
            if (absolute <= __int_as_float(0x47ce4780) || absolute == __int_as_float(0x7f800000))
                return reduced;

            // Exact integer Payne-Hanek branch emitted by the target CUDA-6 libdevice.
            // The six words are 2/pi in the same least-significant-word-first order as
            // __cudart_i2opi_f in the recovered PTX.
            constexpr std::uint32_t two_over_pi[6] = {
                0x3c439041U, 0xdb629599U, 0xf534ddc0U, 0xfc2757d1U, 0x4e441529U, 0xa2f9836eU};
            const std::uint32_t value_bits = static_cast<std::uint32_t>(__float_as_int(value));
            const std::uint32_t exponent = (value_bits >> 23U) & 0xffU;
            const std::uint32_t mantissa = (value_bits << 8U) | 0x80000000U;
            std::uint32_t product[7];
            std::uint32_t carry = 0U;
#pragma unroll
            for (std::uint32_t index = 0U; index < 6U; ++index)
            {
                const std::uint32_t word = two_over_pi[index];
                const std::uint32_t low = word * mantissa;
                const std::uint32_t with_carry = low + carry;
                const std::uint32_t overflow = with_carry < low ? 1U : 0U;
                product[index] = with_carry;
                carry = __umulhi(word, mantissa) + overflow;
            }
            product[6] = carry;

            const std::int32_t exponent_offset = static_cast<std::int32_t>(exponent) - 128;
            const std::int32_t word_index = 6 - (exponent_offset >> 5);
            std::uint32_t high = product[word_index];
            std::uint32_t middle = product[word_index - 1];
            const std::uint32_t bit_shift = static_cast<std::uint32_t>(exponent_offset) & 31U;
            if (bit_shift != 0U)
            {
                high = (middle >> (32U - bit_shift)) + (high << bit_shift);
                middle = (product[word_index - 2] >> (32U - bit_shift)) + (middle << bit_shift);
            }
            std::uint32_t fixed_high = (middle >> 30U) + (high << 2U);
            const std::uint32_t fixed_low = middle << 2U;
            const std::uint32_t round_up = fixed_high >> 31U;
            const std::uint32_t quadrant_magnitude = round_up + (high >> 30U);
            const std::uint32_t sign = value_bits & 0x80000000U;
            quadrant = sign == 0U ? static_cast<int>(quadrant_magnitude) : -static_cast<int>(quadrant_magnitude);

            std::uint32_t fraction_low;
            std::uint32_t fraction_sign;
            if (round_up != 0U)
            {
                fixed_high = (fixed_low == 0U ? 1U : 0U) + ~fixed_high;
                fraction_low = 0U - fixed_low;
                fraction_sign = sign ^ 0x80000000U;
            }
            else
            {
                fraction_low = fixed_low;
                fraction_sign = sign;
            }

            std::uint32_t leading = static_cast<std::uint32_t>(__clz(fixed_high));
            std::uint32_t normalized = fixed_high;
            if (leading != 0U)
            {
                normalized = (fixed_high << leading) + (fraction_low >> (32U - leading));
            }
            constexpr std::uint32_t pi_over_two_fixed = 0xc90fdaa2U;
            std::uint32_t scaled = __umulhi(normalized, pi_over_two_fixed);
            // Target PTX uses `setp.lt.s32 scaled, 1`, not an unsigned compare.
            // Values with the high bit set therefore already have the required
            // normalization and must not be doubled a second time.
            if (static_cast<std::int32_t>(scaled) >= 1)
            {
                scaled = ((normalized * pi_over_two_fixed) >> 31U) + (scaled << 1U);
                ++leading;
            }
            const std::uint32_t exponent_bits = (126U - leading) << 23U;
            std::uint32_t mantissa_bits = (scaled + 1U) >> 7U;
            mantissa_bits = (mantissa_bits + 1U) >> 1U;
            return __int_as_float(static_cast<int>(mantissa_bits + exponent_bits + fraction_sign));
        }

        __device__ __forceinline__ float recovered_legacy_sin_from_reduced(float reduced, int quadrant)
        {
            const float squared = __fmul_rn(reduced, reduced);
            float result;
            if ((quadrant & 1) != 0)
            {
                float polynomial = __fmaf_rn(__int_as_float(0x37ccf5ce), squared, __int_as_float(0xbab6061a));
                polynomial = __fmaf_rn(polynomial, squared, __int_as_float(0x3d2aaaa5));
                polynomial = __fmaf_rn(polynomial, squared, __int_as_float(0xbf000000));
                result = __fmaf_rn(polynomial, squared, 1.0F);
            }
            else
            {
                float polynomial = __fmaf_rn(__int_as_float(0xb94ca1f9), squared, __int_as_float(0x3c08839e));
                polynomial = __fmaf_rn(polynomial, squared, __int_as_float(0xbe2aaaa3));
                polynomial = __fmaf_rn(polynomial, squared, 0.0F);
                result = __fmaf_rn(polynomial, reduced, reduced);
            }
            if ((quadrant & 2) != 0)
                result = __fmaf_rn(result, -1.0F, 0.0F);
            return result;
        }

        __device__ __forceinline__ RecoveredLegacySinCos recovered_legacy_sincosf(float value)
        {
            int quadrant = 0;
            const float reduced = recovered_legacy_reduce_trig(value, quadrant);
            return {recovered_legacy_sin_from_reduced(reduced, quadrant),
                    recovered_legacy_sin_from_reduced(reduced, quadrant + 1)};
        }

        __device__ __forceinline__ float recovered_legacy_tanf(float value)
        {
            int quadrant = 0;
            const float reduced = recovered_legacy_reduce_trig(value, quadrant);
            const float squared = __fmul_rn(reduced, reduced);
            float numerator = __fmaf_rn(__int_as_float(0x3b86d46d), squared, __int_as_float(0xbf52b7f4));
            const float reciprocal = __frcp_rn(__fadd_rn(squared, __int_as_float(0xc01e09d0)));
            numerator = __fmul_rn(numerator, reciprocal);
            numerator = __fmul_rn(squared, numerator);
            float result = __fmaf_rn(numerator, reduced, reduced);
            if ((quadrant & 1) != 0)
            {
                result = __fdiv_rn(-1.0F, result);
            }
            return result;
        }

        __device__ __forceinline__ float recovered_legacy_asinf(float value)
        {
            const float absolute = fabsf(value);
            const float half_complement = __fmul_rn(__fsub_rn(1.0F, absolute), 0.5F);
            const float root = __fsqrt_rn(half_complement);
            const bool upper = absolute > __int_as_float(0x3f11eb85);
            const float argument = upper ? root : absolute;
            const float squared = __fmul_rn(argument, argument);
            float polynomial = __fmaf_rn(__int_as_float(0x3d53f941), squared, __int_as_float(0x3c94d2e9));
            polynomial = __fmaf_rn(polynomial, squared, __int_as_float(0x3d3f841f));
            polynomial = __fmaf_rn(polynomial, squared, __int_as_float(0x3d994929));
            polynomial = __fmaf_rn(polynomial, squared, __int_as_float(0x3e2aab94));
            polynomial = __fmul_rn(squared, polynomial);
            float result = __fmaf_rn(polynomial, argument, argument);
            if (upper)
            {
                result = __fmaf_rn(-2.0F, result, __int_as_float(0x3fc90fdb));
            }
            const std::uint32_t result_bits = static_cast<std::uint32_t>(__float_as_int(result));
            const std::uint32_t input_bits = static_cast<std::uint32_t>(__float_as_int(value));
            return __int_as_float(static_cast<int>(result_bits | (input_bits & 0x80000000U)));
        }

        __device__ __forceinline__ float recovered_legacy_atan2f(float y, float x)
        {
            const float absolute_x = fabsf(x);
            const float absolute_y = fabsf(y);
            if (isnan(x) || isnan(y))
                return __fadd_rn(x, y);
            if (isinf(absolute_x) && isinf(absolute_y))
            {
                float angle = __int_as_float(0x3f490fdb);
                if (x < 0.0F)
                    angle = __fsub_rn(__int_as_float(0x40490fdb), angle);
                return __int_as_float(__float_as_int(angle) | (__float_as_int(y) & 0x80000000));
            }
            if (absolute_x == 0.0F && absolute_y == 0.0F)
            {
                const float angle =
                    x < 0.0F || (__float_as_int(x) & 0x80000000) != 0 ? __int_as_float(0x40490fdb) : 0.0F;
                return __int_as_float(__float_as_int(angle) | (__float_as_int(y) & 0x80000000));
            }
            const float maximum = fmaxf(absolute_y, absolute_x);
            const float minimum = fminf(absolute_y, absolute_x);
            const float ratio = __fdiv_rn(minimum, maximum);
            const float squared = __fmul_rn(ratio, ratio);
            float numerator = __fmaf_rn(squared, __int_as_float(0xbf52c7ea), __int_as_float(0xc0b59883));
            numerator = __fmaf_rn(numerator, squared, __int_as_float(0xc0d21907));
            numerator = __fmul_rn(ratio, __fmul_rn(squared, numerator));
            float denominator = __fadd_rn(squared, __int_as_float(0x41355dc0));
            denominator = __fmaf_rn(denominator, squared, __int_as_float(0x41e6bd60));
            denominator = __fmaf_rn(denominator, squared, __int_as_float(0x419d92c8));
            float angle = __fmaf_rn(numerator, __frcp_rn(denominator), ratio);
            if (absolute_y > absolute_x)
                angle = __fsub_rn(__int_as_float(0x3fc90fdb), angle);
            if (x < 0.0F)
                angle = __fsub_rn(__int_as_float(0x40490fdb), angle);
            return __int_as_float(__float_as_int(angle) | (__float_as_int(y) & 0x80000000));
        }

        __device__ __forceinline__ float recovered_legacy_atanf(float value)
        {
            return recovered_legacy_atan2f(value, 1.0F);
        }

        __device__ __forceinline__ void recovered_calibration_normalized_projection(
            const DepthVotingCalibrationCu& calibration, float projection_x, float projection_y, float& x, float& y)
        {
            const float* values = reinterpret_cast<const float*>(&calibration);
            const auto width = recovered_opaque_load<std::int32_t>(calibration, 84U);
            const auto height = recovered_opaque_load<std::int32_t>(calibration, 88U);
            y = __fmaf_rn(__int2float_rn(height), -0.5F, projection_y);
            y = __fsub_rn(y, values[25U]);
            y = __fdiv_rn(y, values[23U]);
            x = __fmaf_rn(__int2float_rn(width), -0.5F, projection_x);
            x = __fsub_rn(x, values[24U]);
            x = __fmaf_rn(-y, values[27U], x);
            x = __fdiv_rn(x, __fadd_rn(values[23U], values[26U]));
        }

        __device__ __forceinline__ void recovered_calibration_pixel_projection(
            const DepthVotingCalibrationCu& calibration, float x, float y, RecoveredFloat3& projection)
        {
            const float* values = reinterpret_cast<const float*>(&calibration);
            const auto type = recovered_opaque_load<std::uint32_t>(calibration, 80U);
            if (type == 0U || type == 2U || type == 3U || type == 10U)
            {
                const float width_center =
                    __fmaf_rn(__int2float_rn(recovered_opaque_load<std::int32_t>(calibration, 84U)), 0.5F, values[24U]);
                const float height_center =
                    __fmaf_rn(__int2float_rn(recovered_opaque_load<std::int32_t>(calibration, 88U)), 0.5F, values[25U]);
                float projected_x = __fmul_rn(values[26U], x);
                projected_x = __fmaf_rn(values[23U], x, projected_x);
                projected_x = __fmaf_rn(values[27U], y, projected_x);
                projection.x = __fadd_rn(width_center, projected_x);
                projection.y = __fmaf_rn(values[23U], y, height_center);
                return;
            }
            float projected_x = __fmaf_rn(values[23U], x, values[24U]);
            projected_x = __fmaf_rn(values[26U], x, projected_x);
            projected_x = __fmaf_rn(values[27U], y, projected_x);
            projection.x =
                __fmaf_rn(__int2float_rn(recovered_opaque_load<std::int32_t>(calibration, 84U)), 0.5F, projected_x);
            const float projected_y = __fmaf_rn(values[23U], y, values[25U]);
            projection.y =
                __fmaf_rn(__int2float_rn(recovered_opaque_load<std::int32_t>(calibration, 88U)), 0.5F, projected_y);
        }

        __device__ __forceinline__ void
        recovered_calibration_brown_project(const DepthVotingCalibrationCu& calibration, float& x, float& y)
        {
            if (recovered_opaque_load<std::uint32_t>(calibration, 112U) == 0U)
                return;
            const float* values = reinterpret_cast<const float*>(&calibration);
            const float x2 = __fmul_rn(x, x);
            const float y2 = __fmul_rn(y, y);
            float r2 = __fadd_rn(x2, y2);
            float norm2 = 1.0F;
            if (r2 > values[8U])
            {
                norm2 = __fdiv_rn(values[8U], r2);
                r2 = values[8U];
            }
            const float r4 = __fmul_rn(r2, r2);
            float radial = __fmaf_rn(values[0U], r2, __fmul_rn(values[1U], r4));
            radial = __fmaf_rn(__fmul_rn(values[2U], r2), r4, radial);
            radial = __fmaf_rn(r4, __fmul_rn(values[3U], r4), radial);
            const float pdist = __fmaf_rn(values[7U], r4, __fmaf_rn(values[6U], r2, 1.0F));
            const float dx_inner = __fmaf_rn(__fmul_rn(__fadd_rn(values[5U], values[5U]), x),
                                             y,
                                             __fmul_rn(values[4U], __fmaf_rn(x, __fmul_rn(3.0F, x), y2)));
            const float dy_inner = __fmaf_rn(__fmul_rn(__fadd_rn(values[4U], values[4U]), x),
                                             y,
                                             __fmul_rn(values[5U], __fmaf_rn(y, __fmul_rn(3.0F, y), x2)));
            x = __fadd_rn(x, __fmaf_rn(norm2, __fmul_rn(dx_inner, pdist), __fmul_rn(x, radial)));
            y = __fadd_rn(y, __fmaf_rn(norm2, __fmul_rn(dy_inner, pdist), __fmul_rn(y, radial)));
        }

        __device__ __forceinline__ bool recovered_calibration_unproject3(const DepthVotingCalibrationCu& calibration,
                                                                         float projection_x,
                                                                         float projection_y,
                                                                         float depth,
                                                                         RecoveredFloat3& output,
                                                                         std::uint32_t normalization_order)
        {
            const auto type = recovered_opaque_load<std::uint32_t>(calibration, 80U);
            if (type == 1U)
            {
                output = recovered_depth_radius_unproject_type1(calibration, projection_x, projection_y, depth);
                return true;
            }
            float x = 0.0F;
            float y = 0.0F;
            recovered_calibration_normalized_projection(calibration, projection_x, projection_y, x, y);
            if (type == 0U)
            {
                output = {__fmul_rn(x, depth), __fmul_rn(y, depth), depth};
                return true;
            }
            if (type == 2U || type == 3U || type == 10U)
            {
                const RecoveredFloat3 corrected =
                    recovered_depth_radius_unproject_type1(calibration, projection_x, projection_y, 1.0F);
                x = corrected.x;
                y = corrected.y;
                const float radius = __fsqrt_rn(__fmaf_rn(x, x, __fmul_rn(y, y)));
                float phi = radius;
                if (type == 10U)
                {
                    if (radius > 2.0F)
                        return false;
                    phi = __fmul_rn(2.0F, recovered_legacy_asinf(__fmul_rn(0.5F, radius)));
                }
                if (type == 2U)
                {
                    if (radius > 0.0F)
                    {
                        const float scale = __fdiv_rn(recovered_legacy_tanf(phi), radius);
                        x = __fmul_rn(x, scale);
                        y = __fmul_rn(y, scale);
                    }
                    output = {__fmul_rn(x, depth), __fmul_rn(y, depth), depth};
                    return true;
                }
                const RecoveredLegacySinCos trig = recovered_legacy_sincosf(phi);
                if (radius > 0.0F)
                {
                    const float scale = __fdiv_rn(trig.sine, radius);
                    x = __fmul_rn(x, scale);
                    y = __fmul_rn(y, scale);
                }
                output = {__fmul_rn(x, depth), __fmul_rn(y, depth), __fmul_rn(trig.cosine, depth)};
                return true;
            }
            if (type == 4U || type == 5U)
            {
                const RecoveredLegacySinCos trig_x = recovered_legacy_sincosf(x);
                const RecoveredLegacySinCos trig_y = recovered_legacy_sincosf(y);
                if (type == 4U)
                    output = {__fmul_rn(__fmul_rn(trig_x.sine, trig_y.cosine), depth),
                              __fmul_rn(trig_y.sine, depth),
                              __fmul_rn(__fmul_rn(trig_x.cosine, trig_y.cosine), depth)};
                else
                    output = {__fmul_rn(trig_x.sine, depth),
                              __fmul_rn(__fmul_rn(trig_y.sine, trig_x.cosine), depth),
                              __fmul_rn(__fmul_rn(trig_x.cosine, trig_y.cosine), depth)};
                return true;
            }
            if (type == 6U || type == 7U || type == 239U)
            {
                RecoveredFloat3 ray{};
                if (type == 6U)
                {
                    const RecoveredLegacySinCos trig = recovered_legacy_sincosf(x);
                    ray = {trig.sine, y, trig.cosine};
                }
                else if (type == 7U)
                {
                    const RecoveredLegacySinCos trig = recovered_legacy_sincosf(y);
                    ray = {x, trig.sine, trig.cosine};
                }
                else
                {
                    const float* values = reinterpret_cast<const float*>(&calibration);
                    const RecoveredLegacySinCos trig = recovered_legacy_sincosf(y);
                    float delta = __fmaf_rn(values[11U], trig.cosine, values[9U]);
                    delta = __fmaf_rn(values[12U], trig.sine, delta);
                    const float warp_scale = recovered_warp_exp_from_c0(values[10U]);
                    const float warped_tangent = recovered_legacy_tanf(x);
                    ray = {__fmaf_rn(warp_scale, warped_tangent, -delta), trig.sine, trig.cosine};
                }
                float squared_length = 0.0F;
                const bool y_first =
                    type == 7U || (type == 6U && (normalization_order == 1U || normalization_order == 3U));
                if (y_first)
                {
                    squared_length = __fmaf_rn(ray.z, ray.z, __fmaf_rn(ray.x, ray.x, __fmul_rn(ray.y, ray.y)));
                }
                else
                {
                    squared_length = __fmaf_rn(ray.z, ray.z, __fmaf_rn(ray.y, ray.y, __fmul_rn(ray.x, ray.x)));
                }
                const float length = __fsqrt_rn(squared_length);
                output = {__fmul_rn(__fdiv_rn(ray.x, length), depth),
                          __fmul_rn(__fdiv_rn(ray.y, length), depth),
                          __fmul_rn(__fdiv_rn(ray.z, length), depth)};
                return true;
            }
            return false;
        }

        __device__ __forceinline__ bool recovered_calibration_project3(const DepthVotingCalibrationCu& calibration,
                                                                       const RecoveredFloat3& point,
                                                                       RecoveredFloat3& projection)
        {
            const auto type = recovered_opaque_load<std::uint32_t>(calibration, 80U);
            if (type == 1U)
                return recovered_voting_project_type1(calibration, point, projection);
            float x = point.x;
            float y = point.y;
            const float z = point.z;
            if (type == 0U)
            {
                if (z <= 0.0F)
                    return false;
                const float inverse_z = __frcp_rn(z);
                x = __fmul_rn(x, inverse_z);
                y = __fmul_rn(y, inverse_z);
            }
            else if (type == 2U || type == 3U || type == 10U)
            {
                if (type == 2U && z <= 0.0F)
                    return false;
                const float radius = __fsqrt_rn(__fmaf_rn(x, x, __fmul_rn(y, y)));
                if (radius > 0.0F)
                {
                    const float phi = recovered_legacy_atan2f(radius, z);
                    const float scale =
                        type == 10U
                            ? __fdiv_rn(__fmul_rn(2.0F, recovered_legacy_sincosf(__fmul_rn(0.5F, phi)).sine), radius)
                            : __fdiv_rn(phi, radius);
                    x = __fmul_rn(x, scale);
                    y = __fmul_rn(y, scale);
                }
                recovered_calibration_brown_project(calibration, x, y);
            }
            else if (type == 4U)
            {
                x = recovered_legacy_atan2f(point.x, point.z);
                y = recovered_legacy_atan2f(point.y,
                                            __fsqrt_rn(__fmaf_rn(point.x, point.x, __fmul_rn(point.z, point.z))));
            }
            else if (type == 5U)
            {
                x = recovered_legacy_atan2f(point.x,
                                            __fsqrt_rn(__fmaf_rn(point.y, point.y, __fmul_rn(point.z, point.z))));
                y = recovered_legacy_atan2f(point.y, point.z);
            }
            else if (type == 6U)
            {
                x = recovered_legacy_atan2f(point.x, point.z);
                y = __fdiv_rn(point.y, __fsqrt_rn(__fmaf_rn(point.x, point.x, __fmul_rn(point.z, point.z))));
            }
            else if (type == 7U)
            {
                x = __fdiv_rn(point.x, __fsqrt_rn(__fmaf_rn(point.y, point.y, __fmul_rn(point.z, point.z))));
                y = recovered_legacy_atan2f(point.y, point.z);
            }
            else if (type == 239U)
            {
                const float* values = reinterpret_cast<const float*>(&calibration);
                const float theta = recovered_legacy_atan2f(point.y, point.z);
                const RecoveredLegacySinCos trig = recovered_legacy_sincosf(theta);
                float delta = __fmaf_rn(values[11U], trig.cosine, values[9U]);
                delta = __fmaf_rn(values[12U], trig.sine, delta);
                const float tan_phi =
                    __fdiv_rn(point.x, __fsqrt_rn(__fmaf_rn(point.y, point.y, __fmul_rn(point.z, point.z))));
                x = recovered_legacy_atanf(
                    __fmul_rn(__fadd_rn(tan_phi, delta), recovered_warp_exp_from_c0(-values[10U])));
                y = theta;
            }
            else
            {
                return false;
            }
            recovered_calibration_pixel_projection(calibration, x, y, projection);
            if (type == 0U || type == 1U || type == 2U)
            {
                projection.z = z;
            }
            else
            {
                projection.z = __fsqrt_rn(__fmaf_rn(z, z, __fmaf_rn(point.x, point.x, __fmul_rn(point.y, point.y))));
            }
            return true;
        }

        template <std::uint32_t CameraType>
        __device__ __forceinline__ bool recovered_unproject_local_perspective(const PatchMatchCamera& camera,
                                                                              float sample_x,
                                                                              float sample_y,
                                                                              float depth,
                                                                              std::uint32_t downscale,
                                                                              RecoveredFloat3& output)
        {
            float y = __fmul_rn(__uint2float_rn(downscale), sample_y);
            y = __fsub_rn(y, camera.cy);
            y = __fsub_rn(y, __fmul_rn(__uint2float_rn(camera.height_original), 0.5F));
            y = __fdiv_rn(y, camera.f);
            float x = __fmul_rn(__uint2float_rn(downscale), sample_x);
            x = __fsub_rn(x, camera.cx);
            x = __fsub_rn(x, __fmul_rn(__uint2float_rn(camera.width_original), 0.5F));
            x = __fsub_rn(x, __fmul_rn(y, camera.b2));
            x = __fdiv_rn(x, __fadd_rn(camera.f, camera.b1));
            if constexpr (CameraType == 8U)
            {
                const float* camera_values =
                    reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(&camera));
                const float* inv_line_num = camera_values + 28U;
                const float* inv_line_den = camera_values + 36U;
                const float* inv_samp_num = camera_values + 40U;
                const float* inv_samp_den = camera_values + 48U;
                const auto polynomial8 = [&](const float* c)
                {
                    float value = __fmaf_rn(c[1], x, c[0]);
                    value = __fmaf_rn(c[2], y, value);
                    value = __fmaf_rn(c[3], depth, value);
                    float term = __fmul_rn(x, c[4]);
                    value = __fmaf_rn(y, term, value);
                    term = __fmul_rn(x, c[5]);
                    value = __fmaf_rn(depth, term, value);
                    term = __fmul_rn(y, c[6]);
                    value = __fmaf_rn(depth, term, value);
                    term = __fmul_rn(x, c[7]);
                    term = __fmul_rn(y, term);
                    return __fmaf_rn(depth, term, value);
                };
                const auto polynomial4 = [&](const float* c)
                {
                    float value = __fmaf_rn(c[1], x, c[0]);
                    value = __fmaf_rn(c[2], y, value);
                    const float term = __fmul_rn(x, c[3]);
                    return __fmaf_rn(y, term, value);
                };
                const float line = __fdiv_rn(polynomial8(inv_line_num), polynomial4(inv_line_den));
                const float sample = __fdiv_rn(polynomial8(inv_samp_num), polynomial4(inv_samp_den));
                output = {sample, line, depth};
                return true;
            }
            if constexpr (CameraType == 9U)
            {
                output = {x, y, depth};
                return true;
            }
            if constexpr (CameraType == 2U || CameraType == 3U || CameraType == 10U)
            {
                // Target PTX rounds x*x first, then accumulates y*y with one FMA.
                const float radius = __fsqrt_rn(__fmaf_rn(y, y, __fmul_rn(x, x)));
                float phi = radius;
                if constexpr (CameraType == 10U)
                {
                    if (radius > 2.0F)
                        return false;
                    phi = __fmul_rn(2.0F, recovered_legacy_asinf(__fmul_rn(0.5F, radius)));
                }
                if constexpr (CameraType == 2U)
                {
                    if (radius > 0.0F)
                    {
                        const float scale = __fdiv_rn(recovered_legacy_tanf(phi), radius);
                        x = __fmul_rn(x, scale);
                        y = __fmul_rn(y, scale);
                    }
                    output = {__fmul_rn(x, depth), __fmul_rn(y, depth), depth};
                    return true;
                }
                const RecoveredLegacySinCos trig = recovered_legacy_sincosf(phi);
                if (radius > 0.0F)
                {
                    const float scale = __fdiv_rn(trig.sine, radius);
                    x = __fmul_rn(x, scale);
                    y = __fmul_rn(y, scale);
                }
                output = {__fmul_rn(x, depth), __fmul_rn(y, depth), __fmul_rn(trig.cosine, depth)};
                return true;
            }
            if constexpr (CameraType == 4U)
            {
                const RecoveredLegacySinCos trig_x = recovered_legacy_sincosf(x);
                const RecoveredLegacySinCos trig_y = recovered_legacy_sincosf(y);
                output = {__fmul_rn(__fmul_rn(trig_x.sine, trig_y.cosine), depth),
                          __fmul_rn(trig_y.sine, depth),
                          __fmul_rn(__fmul_rn(trig_x.cosine, trig_y.cosine), depth)};
                return true;
            }
            output = {__fmul_rn(x, depth), __fmul_rn(y, depth), depth};
            return true;
        }

        // Keep the perspective path separate from the camera-model dispatcher above.
        // The recovered sm_30 kernel is sensitive to the exact scalar instruction
        // graph: routing type 0 through a bool/out-parameter helper changed the finer
        // pyramid levels even though the coarsest capture still matched.  This helper
        // deliberately preserves the original return-by-value expression tree.
        __device__ __forceinline__ RecoveredFloat3 recovered_unproject_local_type0(
            const PatchMatchCamera& camera, float sample_x, float sample_y, float depth, std::uint32_t downscale)
        {
            float y = __fmul_rn(__uint2float_rn(downscale), sample_y);
            y = __fsub_rn(y, camera.cy);
            y = __fsub_rn(y, __fmul_rn(__uint2float_rn(camera.height_original), 0.5F));
            y = __fdiv_rn(y, camera.f);
            float x = __fmul_rn(__uint2float_rn(downscale), sample_x);
            x = __fsub_rn(x, camera.cx);
            x = __fsub_rn(x, __fmul_rn(__uint2float_rn(camera.width_original), 0.5F));
            x = __fsub_rn(x, __fmul_rn(y, camera.b2));
            x = __fdiv_rn(x, __fadd_rn(camera.f, camera.b1));
            return {__fmul_rn(x, depth), __fmul_rn(y, depth), depth};
        }

        __device__ __forceinline__ void recovered_try_add_propagation_neighbor(std::int32_t x,
                                                                               std::int32_t y,
                                                                               const float* cost,
                                                                               const float* depth,
                                                                               std::uint32_t width,
                                                                               std::uint32_t height,
                                                                               float minimum_depth,
                                                                               float maximum_depth,
                                                                               std::uint32_t* best_x,
                                                                               std::uint32_t* best_y,
                                                                               float* best_cost,
                                                                               std::uint32_t& count)
        {
            if (x < 0 || y < 0 || x >= static_cast<std::int32_t>(width) || y >= static_cast<std::int32_t>(height))
                return;
            const std::uint32_t index = static_cast<std::uint32_t>(y) * width + static_cast<std::uint32_t>(x);
            const float candidate_cost = cost[index];
            if (candidate_cost == -1.0F)
                return;
            if (minimum_depth != 0.0F && maximum_depth != 0.0F)
            {
                const float candidate_depth = depth[index];
                if (candidate_depth <= minimum_depth || candidate_depth >= maximum_depth)
                    return;
            }
            std::uint32_t insertion = 0U;
            while (insertion < count && best_cost[insertion] < candidate_cost)
                ++insertion;
            if (insertion < count)
            {
                std::uint32_t last = count;
                if (count < 8U)
                    ++count;
                else
                    --last;
                while (last > insertion)
                {
                    best_cost[last] = best_cost[last - 1U];
                    best_x[last] = best_x[last - 1U];
                    best_y[last] = best_y[last - 1U];
                    --last;
                }
            }
            else if (count < 8U)
            {
                ++count;
            }
            if (insertion < 8U)
            {
                best_cost[insertion] = candidate_cost;
                best_x[insertion] = static_cast<std::uint32_t>(x);
                best_y[insertion] = static_cast<std::uint32_t>(y);
            }
        }

        __device__ __forceinline__ std::uint32_t recovered_choose_propagation_neighbors(std::uint32_t x,
                                                                                        std::uint32_t y,
                                                                                        std::uint32_t step,
                                                                                        const float* cost,
                                                                                        const float* depth,
                                                                                        std::uint32_t width,
                                                                                        std::uint32_t height,
                                                                                        float minimum_depth,
                                                                                        float maximum_depth,
                                                                                        std::uint32_t* best_x,
                                                                                        std::uint32_t* best_y)
        {
            float best_cost[8];
            std::uint32_t count = 0U;
            const auto add = [&](std::int32_t dx, std::int32_t dy)
            {
                recovered_try_add_propagation_neighbor(
                    static_cast<std::int32_t>(x) + dx * static_cast<std::int32_t>(step),
                    static_cast<std::int32_t>(y) + dy * static_cast<std::int32_t>(step),
                    cost,
                    depth,
                    width,
                    height,
                    minimum_depth,
                    maximum_depth,
                    best_x,
                    best_y,
                    best_cost,
                    count);
            };
            for (std::int32_t k = 1; k < 6; k += 4)
            {
                add(0, -k);
                add(k, 0);
                add(0, k);
                add(-k, 0);
            }
            constexpr std::int32_t far = 17;
            add(0, -far);
            add(far, 0);
            add(0, far);
            add(-far, 0);
            add(far, -far);
            add(far, far);
            add(-far, far);
            add(-far, -far);
            constexpr std::int32_t k = 2;
            add(-k, -(k + 1));
            add(k, -(k + 1));
            add(k + 1, -k);
            add(k + 1, k);
            add(-k, k + 1);
            add(k, k + 1);
            add(-(k + 1), -k);
            add(-(k + 1), k);
            return count;
        }

        __global__ __launch_bounds__(128, 1) void patchmatch_propagation_u8_perspective_kernel(
            float* depth,
            std::uint8_t* normal,
            float* cost,
            const float* coarse_depth,
            const float* coarse_radius,
            PatchMatchCamera camera,
            const float* rotation_to_local,
            std::uint32_t depth_downscale,
            const std::uint8_t* reference_image,
            std::uint32_t image_one_step_more_detailed,
            float deviation_threshold_multiplier,
            float* candidate_depth,
            float* candidate_normal,
            std::uint32_t checkerboard_step,
            std::uint32_t only_each_fourth_pixel,
            std::uint32_t pixel_offset)
        {
            constexpr std::uint32_t capacity = 128U * 1024U;
            const std::uint32_t temporary_index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t logical = pixel_offset + temporary_index;
            const std::uint32_t width = (camera.width_original + depth_downscale - 1U) / depth_downscale;
            const std::uint32_t height = (camera.height_original + depth_downscale - 1U) / depth_downscale;
            std::uint32_t x;
            std::uint32_t y;
            if (only_each_fourth_pixel == 0U)
            {
                const std::uint32_t half_width = (width + 1U) / 2U;
                y = logical / half_width;
                x = ((y & 1U) + checkerboard_step) % 2U + 2U * (logical % half_width);
            }
            else
            {
                const std::uint32_t quarter_width = (width + 3U) / 4U;
                y = 1U + 2U * logical / quarter_width;
                x = 1U + ((((y / 2U) & 1U) + checkerboard_step) % 2U) * 2U + 4U * (logical % quarter_width);
            }
            if (x >= width || y >= height)
                return;
            const std::uint32_t pixel = y * width + x;

            const std::uint32_t image_downscale =
                image_one_step_more_detailed != 0U ? depth_downscale / 2U : depth_downscale;
            const std::uint32_t image_width = (camera.width_original + image_downscale - 1U) / image_downscale;
            const std::uint32_t image_height = (camera.height_original + image_downscale - 1U) / image_downscale;
            const std::int32_t image_x =
                image_one_step_more_detailed != 0U ? static_cast<std::int32_t>(2U * x) : static_cast<std::int32_t>(x);
            const std::int32_t image_y =
                image_one_step_more_detailed != 0U ? static_cast<std::int32_t>(2U * y) : static_cast<std::int32_t>(y);
            // loadRefPatch(..., window_radius=3): the detailed path expands around
            // 2*(x,y) to [-2,+4), while the same-level path uses [-2,+3).
            const std::int32_t upper_extent = image_one_step_more_detailed != 0U ? 4 : 3;
            const std::int32_t from_x = max(0, image_x - 2);
            const std::int32_t to_x = min(static_cast<std::int32_t>(image_width), image_x + upper_extent);
            const std::int32_t from_y = max(0, image_y - 2);
            const std::int32_t to_y = min(static_cast<std::int32_t>(image_height), image_y + upper_extent);
            float sum = 0.0F;
            float sum_squared = 0.0F;
            float sample_count = 0.0F;
            for (std::int32_t sample_x = from_x; sample_x < to_x; ++sample_x)
            {
                const bool parity = (((from_y & 1) == (sample_x & 1)) != ((image_x & 1) == (image_y & 1)));
                for (std::int32_t sample_y = from_y + static_cast<int>(parity); sample_y < to_y; sample_y += 2)
                {
                    const std::uint8_t sample = reference_image[static_cast<std::uint32_t>(sample_y) * image_width +
                                                                static_cast<std::uint32_t>(sample_x)];
                    const float value = __fdiv_rn(__uint2float_rn(sample), 255.0F);
                    sum = __fadd_rn(sum, value);
                    sum_squared = __fmaf_rn(value, value, sum_squared);
                    sample_count = __fadd_rn(sample_count, 1.0F);
                }
            }
            const float mean = __fdiv_rn(sum, sample_count);
            const float mean_square = __fdiv_rn(sum_squared, sample_count);
            const float variance = __fadd_rn(mean_square, -__fmul_rn(mean, mean));
            const float deviation = __fsqrt_rn(variance < 0.0F ? 0.0F : variance);
            float minimum_depth = 0.0F;
            float maximum_depth = 0.0F;
            if (deviation < __fmul_rn(0.0025F, deviation_threshold_multiplier) && coarse_depth[pixel] != 0.0F)
            {
                const float extent = __fmul_rn(3.0F, coarse_radius[pixel]);
                minimum_depth = __fadd_rn(coarse_depth[pixel], -extent);
                maximum_depth = __fadd_rn(coarse_depth[pixel], extent);
            }
            std::uint32_t best_x[8];
            std::uint32_t best_y[8];
            const std::uint32_t neighbor_count =
                recovered_choose_propagation_neighbors(x,
                                                       y,
                                                       only_each_fourth_pixel != 0U ? 2U : 1U,
                                                       cost,
                                                       depth,
                                                       width,
                                                       height,
                                                       minimum_depth,
                                                       maximum_depth,
                                                       best_x,
                                                       best_y);
            const RecoveredFloat3 ray_point = recovered_unproject_local_type0(camera,
                                                                              __fadd_rn(__uint2float_rn(x), 0.5F),
                                                                              __fadd_rn(__uint2float_rn(y), 0.5F),
                                                                              1.0F,
                                                                              depth_downscale);
            const RecoveredFloat3 ray_origin{0.0F, 0.0F, 0.0F};
            const float ray_length = __fsqrt_rn(__fmaf_rn(
                ray_point.z, ray_point.z, __fmaf_rn(ray_point.x, ray_point.x, __fmul_rn(ray_point.y, ray_point.y))));
            const RecoveredFloat3 ray_direction{__fdiv_rn(ray_point.x, ray_length),
                                                __fdiv_rn(ray_point.y, ray_length),
                                                __fdiv_rn(ray_point.z, ray_length)};
            for (std::uint32_t hypothesis = 0U; hypothesis < neighbor_count; ++hypothesis)
            {
                const std::uint32_t neighbor_pixel = best_y[hypothesis] * width + best_x[hypothesis];
                const RecoveredFloat3 plane_point =
                    recovered_unproject_local_type0(camera,
                                                    __fadd_rn(__uint2float_rn(best_x[hypothesis]), 0.5F),
                                                    __fadd_rn(__uint2float_rn(best_y[hypothesis]), 0.5F),
                                                    depth[neighbor_pixel],
                                                    depth_downscale);
                const RecoveredFloat3 global_normal = unpack_recovered_normal(normal + 3U * neighbor_pixel);
                const auto rotation_row = [&](std::uint32_t row)
                {
                    const std::uint32_t base = 4U * row;
                    return __fmaf_rn(rotation_to_local[base + 2U],
                                     global_normal.z,
                                     __fmaf_rn(rotation_to_local[base + 0U],
                                               global_normal.x,
                                               __fmul_rn(rotation_to_local[base + 1U], global_normal.y)));
                };
                const RecoveredFloat3 local_normal{rotation_row(0U), rotation_row(1U), rotation_row(2U)};
                const float pace =
                    __fmaf_rn(ray_direction.z,
                              local_normal.z,
                              __fmaf_rn(ray_direction.x, local_normal.x, __fmul_rn(ray_direction.y, local_normal.y)));
                RecoveredFloat3 intersection;
                if (fabsf(pace) <= 0.001F)
                {
                    intersection = {FLT_MAX, FLT_MAX, FLT_MAX};
                }
                else
                {
                    const RecoveredFloat3 plane_offset = recovered_subtract(plane_point, ray_origin);
                    const float numerator =
                        __fmaf_rn(plane_offset.z,
                                  local_normal.z,
                                  __fmaf_rn(plane_offset.y, local_normal.y, __fmul_rn(plane_offset.x, local_normal.x)));
                    const float scale = __fdiv_rn(numerator, pace);
                    intersection = {__fmaf_rn(ray_direction.x, scale, ray_origin.x),
                                    __fmaf_rn(ray_direction.y, scale, ray_origin.y),
                                    __fmaf_rn(ray_direction.z, scale, ray_origin.z)};
                }
                float extrapolated_depth = intersection.z;
                RecoveredFloat3 output_normal = global_normal;
                if (intersection.x == FLT_MAX)
                {
                    extrapolated_depth = 0.0F;
                    output_normal = {0.0F, 0.0F, 0.0F};
                }
                const std::uint32_t candidate = hypothesis * capacity + temporary_index;
                candidate_depth[candidate] = extrapolated_depth;
                candidate_normal[3U * candidate + 0U] = output_normal.x;
                candidate_normal[3U * candidate + 1U] = output_normal.y;
                candidate_normal[3U * candidate + 2U] = output_normal.z;
            }
            for (std::uint32_t hypothesis = neighbor_count; hypothesis < 8U; ++hypothesis)
            {
                const std::uint32_t candidate = hypothesis * capacity + temporary_index;
                candidate_depth[candidate] = 0.0F;
                candidate_normal[3U * candidate + 0U] = 0.0F;
                candidate_normal[3U * candidate + 1U] = 0.0F;
                candidate_normal[3U * candidate + 2U] = 0.0F;
            }
        }

        template <class ReferenceSample>
        __device__ __forceinline__ float recovered_propagation_sample(ReferenceSample sample)
        {
            if constexpr (sizeof(ReferenceSample) == 1U)
                return __fdiv_rn(__uint2float_rn(sample), 255.0F);
            if constexpr (sizeof(ReferenceSample) == 2U)
                return __fdiv_rn(__uint2float_rn(sample), 65535.0F);
            return sample;
        }

        // Isolated specializations for non-production sample/camera combinations.
        // U8/type-0 deliberately remains in the dedicated kernel above so compiling
        // another camera model cannot perturb its accepted instruction graph.
        template <class ReferenceSample, std::uint32_t CameraType>
        __global__
            __launch_bounds__(128,
                              1) void patchmatch_propagation_extended_kernel(float* depth,
                                                                             std::uint8_t* normal,
                                                                             float* cost,
                                                                             const float* coarse_depth,
                                                                             const float* coarse_radius,
                                                                             PatchMatchCamera camera,
                                                                             const float* rotation_to_local,
                                                                             std::uint32_t depth_downscale,
                                                                             const ReferenceSample* reference_image,
                                                                             std::uint32_t image_one_step_more_detailed,
                                                                             float deviation_threshold_multiplier,
                                                                             float* candidate_depth,
                                                                             float* candidate_normal,
                                                                             std::uint32_t checkerboard_step,
                                                                             std::uint32_t only_each_fourth_pixel,
                                                                             std::uint32_t pixel_offset)
        {
            constexpr std::uint32_t capacity = 128U * 1024U;
            const std::uint32_t temporary_index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t logical = pixel_offset + temporary_index;
            const std::uint32_t width = (camera.width_original + depth_downscale - 1U) / depth_downscale;
            const std::uint32_t height = (camera.height_original + depth_downscale - 1U) / depth_downscale;
            std::uint32_t x;
            std::uint32_t y;
            if (only_each_fourth_pixel == 0U)
            {
                const std::uint32_t half_width = (width + 1U) / 2U;
                y = logical / half_width;
                x = ((y & 1U) + checkerboard_step) % 2U + 2U * (logical % half_width);
            }
            else
            {
                const std::uint32_t quarter_width = (width + 3U) / 4U;
                y = 1U + 2U * logical / quarter_width;
                x = 1U + ((((y / 2U) & 1U) + checkerboard_step) % 2U) * 2U + 4U * (logical % quarter_width);
            }
            if (x >= width || y >= height)
                return;
            const std::uint32_t pixel = y * width + x;

            const std::uint32_t image_downscale =
                image_one_step_more_detailed != 0U ? depth_downscale / 2U : depth_downscale;
            const std::uint32_t image_width = (camera.width_original + image_downscale - 1U) / image_downscale;
            const std::uint32_t image_height = (camera.height_original + image_downscale - 1U) / image_downscale;
            const std::int32_t image_x =
                image_one_step_more_detailed != 0U ? static_cast<std::int32_t>(2U * x) : static_cast<std::int32_t>(x);
            const std::int32_t image_y =
                image_one_step_more_detailed != 0U ? static_cast<std::int32_t>(2U * y) : static_cast<std::int32_t>(y);
            const std::int32_t upper_extent = image_one_step_more_detailed != 0U ? 4 : 3;
            const std::int32_t from_x = max(0, image_x - 2);
            const std::int32_t to_x = min(static_cast<std::int32_t>(image_width), image_x + upper_extent);
            const std::int32_t from_y = max(0, image_y - 2);
            const std::int32_t to_y = min(static_cast<std::int32_t>(image_height), image_y + upper_extent);
            float sum = 0.0F;
            float sum_squared = 0.0F;
            float sample_count = 0.0F;
            for (std::int32_t sample_x = from_x; sample_x < to_x; ++sample_x)
            {
                const bool parity = (((from_y & 1) == (sample_x & 1)) != ((image_x & 1) == (image_y & 1)));
                for (std::int32_t sample_y = from_y + static_cast<int>(parity); sample_y < to_y; sample_y += 2)
                {
                    const float value = recovered_propagation_sample(
                        reference_image[static_cast<std::uint32_t>(sample_y) * image_width +
                                        static_cast<std::uint32_t>(sample_x)]);
                    sum = __fadd_rn(sum, value);
                    sum_squared = __fmaf_rn(value, value, sum_squared);
                    sample_count = __fadd_rn(sample_count, 1.0F);
                }
            }
            const float mean = __fdiv_rn(sum, sample_count);
            const float mean_square = __fdiv_rn(sum_squared, sample_count);
            const float variance = __fadd_rn(mean_square, -__fmul_rn(mean, mean));
            const float deviation = __fsqrt_rn(variance < 0.0F ? 0.0F : variance);
            float minimum_depth = 0.0F;
            float maximum_depth = 0.0F;
            if (deviation < __fmul_rn(0.0025F, deviation_threshold_multiplier) && coarse_depth[pixel] != 0.0F)
            {
                const float extent = __fmul_rn(3.0F, coarse_radius[pixel]);
                minimum_depth = __fadd_rn(coarse_depth[pixel], -extent);
                maximum_depth = __fadd_rn(coarse_depth[pixel], extent);
            }
            std::uint32_t best_x[8];
            std::uint32_t best_y[8];
            const std::uint32_t neighbor_count =
                recovered_choose_propagation_neighbors(x,
                                                       y,
                                                       only_each_fourth_pixel != 0U ? 2U : 1U,
                                                       cost,
                                                       depth,
                                                       width,
                                                       height,
                                                       minimum_depth,
                                                       maximum_depth,
                                                       best_x,
                                                       best_y);

            RecoveredFloat3 ray_point;
            if (!recovered_unproject_local_perspective<CameraType>(camera,
                                                                   __fadd_rn(__uint2float_rn(x), 0.5F),
                                                                   __fadd_rn(__uint2float_rn(y), 0.5F),
                                                                   1.0F,
                                                                   depth_downscale,
                                                                   ray_point))
            {
                for (std::uint32_t hypothesis = 0U; hypothesis < 8U; ++hypothesis)
                {
                    const std::uint32_t candidate = hypothesis * capacity + temporary_index;
                    candidate_depth[candidate] = 0.0F;
                    candidate_normal[3U * candidate + 0U] = 0.0F;
                    candidate_normal[3U * candidate + 1U] = 0.0F;
                    candidate_normal[3U * candidate + 2U] = 0.0F;
                }
                return;
            }
            RecoveredFloat3 ray_origin;
            RecoveredFloat3 ray_direction;
            if constexpr (CameraType == 8U || CameraType == 9U)
            {
                ray_origin = {ray_point.x, ray_point.y, 0.0F};
                ray_direction = {0.0F, 0.0F, 1.0F};
            }
            else
            {
                ray_origin = {0.0F, 0.0F, 0.0F};
                const float ray_length =
                    __fsqrt_rn(__fmaf_rn(ray_point.z,
                                         ray_point.z,
                                         __fmaf_rn(ray_point.x, ray_point.x, __fmul_rn(ray_point.y, ray_point.y))));
                ray_direction = {__fdiv_rn(ray_point.x, ray_length),
                                 __fdiv_rn(ray_point.y, ray_length),
                                 __fdiv_rn(ray_point.z, ray_length)};
            }
            for (std::uint32_t hypothesis = 0U; hypothesis < neighbor_count; ++hypothesis)
            {
                const std::uint32_t neighbor_pixel = best_y[hypothesis] * width + best_x[hypothesis];
                RecoveredFloat3 plane_point;
                if (!recovered_unproject_local_perspective<CameraType>(
                        camera,
                        __fadd_rn(__uint2float_rn(best_x[hypothesis]), 0.5F),
                        __fadd_rn(__uint2float_rn(best_y[hypothesis]), 0.5F),
                        depth[neighbor_pixel],
                        depth_downscale,
                        plane_point))
                {
                    const std::uint32_t candidate = hypothesis * capacity + temporary_index;
                    candidate_depth[candidate] = 0.0F;
                    candidate_normal[3U * candidate + 0U] = 0.0F;
                    candidate_normal[3U * candidate + 1U] = 0.0F;
                    candidate_normal[3U * candidate + 2U] = 0.0F;
                    continue;
                }
                const RecoveredFloat3 global_normal = unpack_recovered_normal(normal + 3U * neighbor_pixel);
                const auto rotation_row = [&](std::uint32_t row)
                {
                    const std::uint32_t base = 4U * row;
                    return __fmaf_rn(rotation_to_local[base + 2U],
                                     global_normal.z,
                                     __fmaf_rn(rotation_to_local[base + 0U],
                                               global_normal.x,
                                               __fmul_rn(rotation_to_local[base + 1U], global_normal.y)));
                };
                const RecoveredFloat3 local_normal{rotation_row(0U), rotation_row(1U), rotation_row(2U)};
                const float pace =
                    __fmaf_rn(ray_direction.z,
                              local_normal.z,
                              __fmaf_rn(ray_direction.x, local_normal.x, __fmul_rn(ray_direction.y, local_normal.y)));
                RecoveredFloat3 intersection;
                if (fabsf(pace) <= 0.001F)
                {
                    intersection = {FLT_MAX, FLT_MAX, FLT_MAX};
                }
                else
                {
                    const RecoveredFloat3 plane_offset = recovered_subtract(plane_point, ray_origin);
                    const float numerator =
                        __fmaf_rn(plane_offset.z,
                                  local_normal.z,
                                  __fmaf_rn(plane_offset.y, local_normal.y, __fmul_rn(plane_offset.x, local_normal.x)));
                    const float scale = __fdiv_rn(numerator, pace);
                    intersection = {__fmaf_rn(ray_direction.x, scale, ray_origin.x),
                                    __fmaf_rn(ray_direction.y, scale, ray_origin.y),
                                    __fmaf_rn(ray_direction.z, scale, ray_origin.z)};
                }
                float extrapolated_depth;
                if constexpr (CameraType == 4U)
                {
                    extrapolated_depth = __fsqrt_rn(__fmaf_rn(
                        intersection.z,
                        intersection.z,
                        __fmaf_rn(intersection.x, intersection.x, __fmul_rn(intersection.y, intersection.y))));
                }
                else
                {
                    extrapolated_depth = intersection.z;
                }
                RecoveredFloat3 output_normal = global_normal;
                if (intersection.x == FLT_MAX)
                {
                    extrapolated_depth = 0.0F;
                    output_normal = {0.0F, 0.0F, 0.0F};
                }
                const std::uint32_t candidate = hypothesis * capacity + temporary_index;
                candidate_depth[candidate] = extrapolated_depth;
                candidate_normal[3U * candidate + 0U] = output_normal.x;
                candidate_normal[3U * candidate + 1U] = output_normal.y;
                candidate_normal[3U * candidate + 2U] = output_normal.z;
            }
            for (std::uint32_t hypothesis = neighbor_count; hypothesis < 8U; ++hypothesis)
            {
                const std::uint32_t candidate = hypothesis * capacity + temporary_index;
                candidate_depth[candidate] = 0.0F;
                candidate_normal[3U * candidate + 0U] = 0.0F;
                candidate_normal[3U * candidate + 1U] = 0.0F;
                candidate_normal[3U * candidate + 2U] = 0.0F;
            }
        }

        template <class ReferenceSample, std::uint32_t CameraType>
        __global__ __launch_bounds__(128, 1) void patchmatch_final_refinement_extended_kernel(
            float* depth,
            std::uint8_t* normal,
            float* cost,
            PatchMatchCamera camera,
            std::uint32_t depth_downscale,
            const ReferenceSample* reference_image,
            std::uint32_t image_one_step_more_detailed,
            float deviation_threshold_multiplier,
            float* candidate_depth,
            float* candidate_normal,
            std::uint32_t pixel_offset)
        {
            constexpr std::uint32_t capacity = 128U * 1024U;
            const std::uint32_t temporary_index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t global_index = pixel_offset + temporary_index;
            const std::uint32_t width = (camera.width_original + depth_downscale - 1U) / depth_downscale;
            const std::uint32_t height = (camera.height_original + depth_downscale - 1U) / depth_downscale;
            const std::uint32_t y = global_index / width;
            const std::uint32_t x = global_index % width;
            if (x >= width || y >= height)
                return;

            const std::uint32_t image_downscale =
                image_one_step_more_detailed != 0U ? depth_downscale / 2U : depth_downscale;
            const std::uint32_t image_width = (camera.width_original + image_downscale - 1U) / image_downscale;
            const std::uint32_t image_height = (camera.height_original + image_downscale - 1U) / image_downscale;
            const std::int32_t image_x =
                image_one_step_more_detailed != 0U ? static_cast<std::int32_t>(2U * x) : static_cast<std::int32_t>(x);
            const std::int32_t image_y =
                image_one_step_more_detailed != 0U ? static_cast<std::int32_t>(2U * y) : static_cast<std::int32_t>(y);
            const std::int32_t extra = image_one_step_more_detailed != 0U ? 1 : 0;
            const std::int32_t from_x = max(0, image_x - 1);
            const std::int32_t to_x = min(static_cast<std::int32_t>(image_width), image_x + 2 + extra);
            const std::int32_t from_y = max(0, image_y - 1);
            const std::int32_t to_y = min(static_cast<std::int32_t>(image_height), image_y + 2 + extra);
            float sum = 0.0F;
            float sum_squared = 0.0F;
            float count = 0.0F;
            for (std::int32_t sample_x = from_x; sample_x < to_x; ++sample_x)
            {
                const bool parity = (((from_y & 1) == (sample_x & 1)) != ((image_x & 1) == (image_y & 1)));
                for (std::int32_t sample_y = from_y + static_cast<int>(parity); sample_y < to_y; sample_y += 2)
                {
                    const float value = recovered_propagation_sample(
                        reference_image[static_cast<std::uint32_t>(sample_y) * image_width +
                                        static_cast<std::uint32_t>(sample_x)]);
                    sum = __fadd_rn(sum, value);
                    sum_squared = __fmaf_rn(value, value, sum_squared);
                    count = __fadd_rn(count, 1.0F);
                }
            }
            const float mean = __fdiv_rn(sum, count);
            const float mean_squared = __fdiv_rn(sum_squared, count);
            const float variance = __fadd_rn(mean_squared, -__fmul_rn(mean, mean));
            const float deviation = __fsqrt_rn(variance < 0.0F ? 0.0F : variance);
            if (deviation < __fmul_rn(0.01F, deviation_threshold_multiplier))
            {
                recovered_final_zero_candidates(candidate_depth, candidate_normal, temporary_index);
                return;
            }

            cost[global_index] = -1.0F;
            const float center_depth = depth[global_index];
            const RecoveredFloat3 center_normal = unpack_recovered_normal(normal + 3U * global_index);
            const float pixel_x = __fadd_rn(__uint2float_rn(x), 0.5F);
            const float pixel_y = __fadd_rn(__uint2float_rn(y), 0.5F);
            RecoveredFloat3 center;
            RecoveredFloat3 step_x;
            RecoveredFloat3 step_y;
            bool unprojected = true;
            if constexpr (CameraType == 0U)
            {
                // Preserve the accepted type-0 scalar expression tree.
                center = recovered_final_unproject_perspective(camera, pixel_x, pixel_y, center_depth, depth_downscale);
                step_x = recovered_final_unproject_perspective(
                    camera, __fadd_rn(__uint2float_rn(x), 1.0F), pixel_y, center_depth, depth_downscale);
                step_y = recovered_final_unproject_perspective(
                    camera, pixel_x, __fadd_rn(__uint2float_rn(y), 1.0F), center_depth, depth_downscale);
            }
            else
            {
                unprojected =
                    recovered_unproject_local_perspective<CameraType>(
                        camera, pixel_x, pixel_y, center_depth, depth_downscale, center) &&
                    recovered_unproject_local_perspective<CameraType>(
                        camera, __fadd_rn(__uint2float_rn(x), 1.0F), pixel_y, center_depth, depth_downscale, step_x) &&
                    recovered_unproject_local_perspective<CameraType>(
                        camera, pixel_x, __fadd_rn(__uint2float_rn(y), 1.0F), center_depth, depth_downscale, step_y);
                if (unprojected)
                {
                    center = recovered_final_transform_point(camera, center);
                    step_x = recovered_final_transform_point(camera, step_x);
                    step_y = recovered_final_transform_point(camera, step_y);
                }
            }
            if (!unprojected)
            {
                recovered_final_zero_candidates(candidate_depth, candidate_normal, temporary_index);
                return;
            }
            const RecoveredFloat3 delta_x = recovered_subtract(step_x, center);
            const RecoveredFloat3 delta_y = recovered_subtract(step_y, center);
            const float distance_x = recovered_squared_distance(step_x, center);
            const float distance_y =
                __fmaf_rn(delta_y.z, delta_y.z, __fmaf_rn(delta_y.y, delta_y.y, __fmul_rn(delta_y.x, delta_y.x)));
            RecoveredFloat3 direction = distance_x > distance_y ? delta_x : delta_y;
            const float direction_squared = __fmaf_rn(
                direction.z, direction.z, __fmaf_rn(direction.x, direction.x, __fmul_rn(direction.y, direction.y)));
            const float orthogonal_radius = __fsqrt_rn(direction_squared);
            direction.x = __fdiv_rn(direction.x, orthogonal_radius);
            direction.y = __fdiv_rn(direction.y, orthogonal_radius);
            direction.z = __fdiv_rn(direction.z, orthogonal_radius);
            float cosine = __fmaf_rn(direction.z,
                                     center_normal.z,
                                     __fmaf_rn(direction.x, center_normal.x, __fmul_rn(direction.y, center_normal.y)));
            cosine = fabsf(cosine);
            // The legacy PTX spells this as a multiply followed by a subtraction, but
            // its sm89 JIT contracts the pair into the observable FFMA below.
            const float sine = __fsqrt_rn(__fmaf_rn(-cosine, cosine, 1.0F));
            // PTX `setp.lt + selp` preserves a NaN sine (the comparison is false).
            // CUDA fmaxf instead returns the numeric operand for a single NaN and
            // changes zero-depth candidates from the target NaN sentinel to zero.
            const float clamped_sine = sine < 0.1F ? 0.1F : sine;
            const float sample_radius = __fdiv_rn(orthogonal_radius, clamped_sine);
            const float double_radius = __fadd_rn(sample_radius, sample_radius);
            const float depth_min = __fadd_rn(center_depth, -double_radius);
            const float depth_max = __fadd_rn(center_depth, double_radius);
            const float depth_step = __fdiv_rn(__fadd_rn(depth_max, -depth_min), 7.0F);
#pragma unroll
            for (std::uint32_t hypothesis = 0U; hypothesis < 8U; ++hypothesis)
            {
                const std::uint32_t candidate_index = hypothesis * capacity + temporary_index;
                candidate_depth[candidate_index] = hypothesis == 1U
                                                       ? __fadd_rn(depth_min, depth_step)
                                                       : __fmaf_rn(depth_step, __uint2float_rn(hypothesis), depth_min);
                const std::uint32_t normal_index = 3U * candidate_index;
                candidate_normal[normal_index + 0U] = center_normal.x;
                candidate_normal[normal_index + 1U] = center_normal.y;
                candidate_normal[normal_index + 2U] = center_normal.z;
            }
        }

        struct RecoveredFastRandom
        {
            std::uint32_t x;
            std::uint32_t y;
            std::uint32_t z;
        };

        __device__ __forceinline__ void
        recovered_random_init(RecoveredFastRandom& random, std::uint32_t global_seed, std::uint32_t local_seed)
        {
            random.x = global_seed * 1000000007U + local_seed * 239017U;
            random.y = 362436069U;
            random.z = 521288629U;
        }

        __device__ __forceinline__ std::uint32_t
        recovered_random_next(RecoveredFastRandom& random, std::uint32_t minimum, std::uint32_t maximum)
        {
            random.x ^= random.x << 16U;
            random.x ^= random.x >> 5U;
            random.x ^= random.x << 1U;
            const std::uint32_t value = random.x;
            random.x = random.y;
            random.y = random.z;
            random.z = value ^ random.x ^ random.y;
            return minimum + random.z % (maximum - minimum + 1U);
        }

        __device__ __forceinline__ float
        recovered_random_next_float(RecoveredFastRandom& random, float minimum, float maximum)
        {
            constexpr std::uint32_t prime = 1000000007U;
            float value = __fdiv_rn(__uint2float_rn(recovered_random_next(random, 0U, prime)), __uint2float_rn(prime));
            return __fmaf_rn(value, __fadd_rn(maximum, -minimum), minimum);
        }

        __device__ __forceinline__ float recovered_random_next_unit_float(RecoveredFastRandom& random)
        {
            constexpr std::uint32_t prime = 1000000007U;
            const float value =
                __fdiv_rn(__uint2float_rn(recovered_random_next(random, 0U, prime)), __uint2float_rn(prime));
            // The target PTX preserves the generic random-range add-by-zero here;
            // spelling it explicitly avoids an identity FFMA in the native cubin.
            return __fadd_rn(value, 0.0F);
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_refinement_normalize(const RecoveredFloat3& value)
        {
            const float squared = __fmaf_rn(value.z, value.z, __fmaf_rn(value.x, value.x, __fmul_rn(value.y, value.y)));
            const float length = __fsqrt_rn(squared);
            return {__fdiv_rn(value.x, length), __fdiv_rn(value.y, length), __fdiv_rn(value.z, length)};
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_random_sphere(RecoveredFastRandom& random)
        {
            float q1;
            float q2;
            float squared;
            do
            {
                q1 = recovered_random_next_float(random, -1.0F, 1.0F);
                q2 = recovered_random_next_float(random, -1.0F, 1.0F);
                squared = __fmaf_rn(q1, q1, __fmul_rn(q2, q2));
            } while (squared >= 1.0F);
            const float root = __fsqrt_rn(__fadd_rn(1.0F, -squared));
            return {__fmaf_rn(-2.0F, squared, 1.0F),
                    __fmul_rn(__fmul_rn(2.0F, q1), root),
                    __fmul_rn(__fmul_rn(2.0F, q2), root)};
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_refinement_cross(const RecoveredFloat3& left,
                                                                              const RecoveredFloat3& right)
        {
            return {__fmaf_rn(left.y, right.z, -__fmul_rn(left.z, right.y)),
                    __fmaf_rn(left.z, right.x, -__fmul_rn(left.x, right.z)),
                    __fmaf_rn(left.x, right.y, -__fmul_rn(left.y, right.x))};
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_refinement_unproject_type0(
            const PatchMatchCamera& camera, float sample_x, float sample_y, float depth, std::uint32_t downscale)
        {
            return recovered_final_transform_point(
                camera, recovered_unproject_local_type0(camera, sample_x, sample_y, depth, downscale));
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_random_normal_type0(const PatchMatchCamera& camera,
                                                                                 RecoveredFastRandom& random,
                                                                                 std::uint32_t x,
                                                                                 std::uint32_t y,
                                                                                 std::uint32_t downscale)
        {
            RecoveredFloat3 local = recovered_random_sphere(random);
            if (local.z < 0.0F)
                local.z = __fmul_rn(local.z, -1.0F);
            const RecoveredFloat3 center = recovered_refinement_unproject_type0(
                camera, __fadd_rn(__uint2float_rn(x), 0.5F), __fadd_rn(__uint2float_rn(y), 0.5F), 1.0F, downscale);
            const RecoveredFloat3 right = recovered_refinement_unproject_type0(
                camera, __fadd_rn(__uint2float_rn(x), 1.0F), __fadd_rn(__uint2float_rn(y), 0.5F), 1.0F, downscale);
            const RecoveredFloat3 down = recovered_refinement_unproject_type0(
                camera, __fadd_rn(__uint2float_rn(x), 0.5F), __fadd_rn(__uint2float_rn(y), 1.0F), 1.0F, downscale);
            const RecoveredFloat3 axis_x = recovered_refinement_normalize(recovered_subtract(right, center));
            const RecoveredFloat3 axis_y = recovered_refinement_normalize(recovered_subtract(down, center));
            const RecoveredFloat3 axis_z = recovered_refinement_cross(axis_y, axis_x);
            const auto component = [&](float x_component, float y_component, float z_component)
            {
                float value = __fmul_rn(local.y, y_component);
                value = __fmaf_rn(local.x, x_component, value);
                return __fmaf_rn(local.z, z_component, value);
            };
            return {component(axis_x.x, axis_y.x, axis_z.x),
                    component(axis_x.y, axis_y.y, axis_z.y),
                    component(axis_x.z, axis_y.z, axis_z.z)};
        }

        template <class ReferenceSample>
        __global__
            __launch_bounds__(128,
                              1) void patchmatch_refinement_type0_kernel(const float* depth,
                                                                         const std::uint8_t* normal,
                                                                         const float* cost,
                                                                         const float* coarse_depth,
                                                                         const float* coarse_radius,
                                                                         PatchMatchCamera camera,
                                                                         std::uint32_t depth_downscale,
                                                                         float input_depth_min,
                                                                         float input_depth_max,
                                                                         const ReferenceSample* reference_image,
                                                                         std::uint32_t image_one_step_more_detailed,
                                                                         float deviation_threshold_multiplier,
                                                                         float* candidate_depth,
                                                                         float* candidate_normal,
                                                                         std::uint32_t iteration,
                                                                         std::uint32_t only_each_fourth_pixel,
                                                                         std::uint32_t pixel_offset)
        {
            constexpr std::uint32_t capacity = 128U * 1024U;
            const std::uint32_t temporary_index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t logical = pixel_offset + temporary_index;
            const std::uint32_t width = (camera.width_original + depth_downscale - 1U) / depth_downscale;
            const std::uint32_t height = (camera.height_original + depth_downscale - 1U) / depth_downscale;
            std::uint32_t x;
            std::uint32_t y;
            if (only_each_fourth_pixel == 0U)
            {
                y = logical / width;
                x = logical % width;
            }
            else
            {
                const std::uint32_t half_width = (width + 1U) / 2U;
                y = 1U + 2U * (logical / half_width);
                x = 1U + 2U * (logical % half_width);
            }
            if (x >= width || y >= height)
                return;
            const std::uint32_t pixel = y * width + x;
            const std::uint32_t image_downscale =
                image_one_step_more_detailed != 0U ? depth_downscale / 2U : depth_downscale;
            const std::uint32_t image_width = (camera.width_original + image_downscale - 1U) / image_downscale;
            const std::uint32_t image_height = (camera.height_original + image_downscale - 1U) / image_downscale;
            const std::int32_t image_x =
                image_one_step_more_detailed != 0U ? static_cast<std::int32_t>(2U * x) : static_cast<std::int32_t>(x);
            const std::int32_t image_y =
                image_one_step_more_detailed != 0U ? static_cast<std::int32_t>(2U * y) : static_cast<std::int32_t>(y);
            const std::int32_t upper = image_one_step_more_detailed != 0U ? 4 : 3;
            const std::int32_t from_x = max(0, image_x - 2);
            const std::int32_t to_x = min(static_cast<std::int32_t>(image_width), image_x + upper);
            const std::int32_t from_y = max(0, image_y - 2);
            const std::int32_t to_y = min(static_cast<std::int32_t>(image_height), image_y + upper);
            float sum = 0.0F;
            float sum_squared = 0.0F;
            float count = 0.0F;
            for (std::int32_t sx = from_x; sx < to_x; ++sx)
            {
                const bool parity = (((from_y & 1) == (sx & 1)) != ((image_x & 1) == (image_y & 1)));
                for (std::int32_t sy = from_y + static_cast<int>(parity); sy < to_y; sy += 2)
                {
                    const float value = recovered_propagation_sample(
                        reference_image[static_cast<std::uint32_t>(sy) * image_width + static_cast<std::uint32_t>(sx)]);
                    sum = __fadd_rn(sum, value);
                    sum_squared = __fmaf_rn(value, value, sum_squared);
                    count = __fadd_rn(count, 1.0F);
                }
            }
            const float mean = __fdiv_rn(sum, count);
            const float variance = __fadd_rn(__fdiv_rn(sum_squared, count), -__fmul_rn(mean, mean));
            const float deviation = __fsqrt_rn(variance < 0.0F ? 0.0F : variance);
            float depth_min = input_depth_min;
            float depth_max = input_depth_max;
            if (deviation < __fmul_rn(0.0025F, deviation_threshold_multiplier) && coarse_depth[pixel] != 0.0F)
            {
                // Legacy PTX spells these as mul+sub/add, but the sm89 JIT contracts
                // both observable expressions into FFMA.  Preserve that contraction
                // explicitly in the source implementation.
                const float lower = __fmaf_rn(-3.0F, coarse_radius[pixel], coarse_depth[pixel]);
                const float upper = __fmaf_rn(3.0F, coarse_radius[pixel], coarse_depth[pixel]);
                // Ordered PTX comparisons deliberately preserve a NaN coarse bound.
                depth_min = lower < depth_min ? depth_min : lower;
                depth_max = upper > depth_max ? depth_max : upper;
            }

            RecoveredFastRandom random;
            recovered_random_init(random, iteration, pixel);
            float estimated_radius = 0.0F;
            RecoveredFloat3 estimated_normal{0.0F, 0.0F, 0.0F};
            std::uint32_t estimated_count = 0U;
            const float center_depth = depth[pixel];
            constexpr std::int32_t dx[4] = {-1, 0, 1, 0};
            constexpr std::int32_t dy[4] = {0, -1, 0, 1};
            const std::uint32_t direction = (iteration + x + y) % 4U;
            const std::int32_t x0 = static_cast<std::int32_t>(x) + dx[direction];
            const std::int32_t y0 = static_cast<std::int32_t>(y) + dy[direction];
            const std::int32_t x1 = static_cast<std::int32_t>(x) + dx[(direction + 1U) % 4U];
            const std::int32_t y1 = static_cast<std::int32_t>(y) + dy[(direction + 1U) % 4U];
            if (x0 >= 0 && y0 >= 0 && x1 >= 0 && y1 >= 0 && x0 < static_cast<std::int32_t>(width) &&
                x1 < static_cast<std::int32_t>(width) && y0 < static_cast<std::int32_t>(height) &&
                y1 < static_cast<std::int32_t>(height))
            {
                const RecoveredFloat3 center = recovered_refinement_unproject_type0(camera,
                                                                                    __fadd_rn(__uint2float_rn(x), 0.5F),
                                                                                    __fadd_rn(__uint2float_rn(y), 0.5F),
                                                                                    center_depth,
                                                                                    depth_downscale);
                const RecoveredFloat3 point0 = recovered_refinement_unproject_type0(
                    camera,
                    __fadd_rn(__int2float_rn(x0), 0.5F),
                    __fadd_rn(__int2float_rn(y0), 0.5F),
                    depth[static_cast<std::uint32_t>(y0) * width + static_cast<std::uint32_t>(x0)],
                    depth_downscale);
                const RecoveredFloat3 point1 = recovered_refinement_unproject_type0(
                    camera,
                    __fadd_rn(__int2float_rn(x1), 0.5F),
                    __fadd_rn(__int2float_rn(y1), 0.5F),
                    depth[static_cast<std::uint32_t>(y1) * width + static_cast<std::uint32_t>(x1)],
                    depth_downscale);
                const RecoveredFloat3 delta0 = recovered_subtract(point0, center);
                const RecoveredFloat3 delta1 = recovered_subtract(point1, center);
                const float distance0 =
                    __fmaf_rn(delta0.z, delta0.z, __fmaf_rn(delta0.x, delta0.x, __fmul_rn(delta0.y, delta0.y)));
                const float distance1 =
                    __fmaf_rn(delta1.z, delta1.z, __fmaf_rn(delta1.x, delta1.x, __fmul_rn(delta1.y, delta1.y)));
                estimated_radius = __fmul_rn(__fsqrt_rn(fminf(distance1, distance0)), 0.5F);
                estimated_normal = recovered_refinement_normalize(recovered_refinement_cross(delta1, delta0));
                estimated_count = 1U;
            }
            const auto random_depth = [&]() {
                return __fmaf_rn(recovered_random_next_unit_float(random), __fadd_rn(depth_max, -depth_min), depth_min);
            };
            float depth_random = random_depth();
            RecoveredFloat3 normal_random = recovered_random_normal_type0(camera, random, x, y, depth_downscale);
            const RecoveredFloat3 normal_center = unpack_recovered_normal(normal + 3U * pixel);
            float depth_perturbed = __fmaf_rn(
                __fmul_rn(4.0F, estimated_radius), recovered_random_next_float(random, -1.0F, 1.0F), center_depth);
            RecoveredFloat3 sphere = recovered_random_sphere(random);
            RecoveredFloat3 normal_perturbed =
                recovered_refinement_normalize({__fmaf_rn(0.2F, sphere.x, normal_center.x),
                                                __fmaf_rn(0.2F, sphere.y, normal_center.y),
                                                __fmaf_rn(0.2F, sphere.z, normal_center.z)});
            if (deviation < __fmul_rn(0.0025F, deviation_threshold_multiplier) &&
                (depth_perturbed <= depth_min || depth_perturbed >= depth_max))
                depth_perturbed = random_depth();

            std::uint32_t output_count = 0U;
            for (std::uint32_t candidate = 0U; candidate < 16U; ++candidate)
            {
                float candidate_value;
                RecoveredFloat3 candidate_n;
                if (candidate < 12U)
                {
                    const std::uint32_t depth_choice = candidate / 4U;
                    const std::uint32_t normal_choice = candidate % 4U;
                    candidate_value = depth_choice == 0U   ? depth_random
                                      : depth_choice == 1U ? center_depth
                                                           : depth_perturbed;
                    if (normal_choice == 0U)
                        candidate_n = normal_random;
                    else if (normal_choice == 1U)
                        candidate_n = normal_center;
                    else if (normal_choice == 2U)
                        candidate_n = normal_perturbed;
                    else if (estimated_count != 0U)
                        candidate_n = estimated_normal;
                    else
                        continue;
                }
                else
                {
                    const std::uint32_t step = only_each_fourth_pixel != 0U ? 2U : 1U;
                    if (candidate == 12U)
                    {
                        if (y >= step && y + step < height)
                        {
                            depth_perturbed = __fmul_rn(
                                __fadd_rn(depth[(y - step) * width + x], depth[(y + step) * width + x]), 0.5F);
                            if (deviation < __fmul_rn(0.0025F, deviation_threshold_multiplier) &&
                                (depth_perturbed <= depth_min || depth_perturbed >= depth_max))
                                depth_perturbed = random_depth();
                            const RecoveredFloat3 a = unpack_recovered_normal(normal + 3U * ((y - step) * width + x));
                            const RecoveredFloat3 b = unpack_recovered_normal(normal + 3U * ((y + step) * width + x));
                            normal_perturbed = {__fmul_rn(__fadd_rn(a.x, b.x), 0.5F),
                                                __fmul_rn(__fadd_rn(a.y, b.y), 0.5F),
                                                __fmul_rn(__fadd_rn(a.z, b.z), 0.5F)};
                        }
                        else
                        {
                            depth_perturbed = random_depth();
                            normal_perturbed = recovered_random_normal_type0(camera, random, x, y, depth_downscale);
                        }
                        if (x >= step && x + step < width)
                        {
                            depth_random =
                                __fmul_rn(__fadd_rn(depth[y * width + x - step], depth[y * width + x + step]), 0.5F);
                            if (deviation < __fmul_rn(0.0025F, deviation_threshold_multiplier) &&
                                (depth_perturbed <= depth_min || depth_perturbed >= depth_max))
                                depth_random = random_depth();
                            const RecoveredFloat3 a = unpack_recovered_normal(normal + 3U * (y * width + x - step));
                            const RecoveredFloat3 b = unpack_recovered_normal(normal + 3U * (y * width + x + step));
                            normal_random = {__fmul_rn(__fadd_rn(a.x, b.x), 0.5F),
                                             __fmul_rn(__fadd_rn(a.y, b.y), 0.5F),
                                             __fmul_rn(__fadd_rn(a.z, b.z), 0.5F)};
                        }
                        else
                        {
                            depth_random = random_depth();
                            normal_random = recovered_random_normal_type0(camera, random, x, y, depth_downscale);
                        }
                    }
                    const std::uint32_t depth_choice = (candidate - 12U) / 2U;
                    const std::uint32_t normal_choice = (candidate - 12U) % 2U;
                    candidate_value = depth_choice == 0U ? depth_perturbed : depth_random;
                    candidate_n = normal_choice == 0U ? normal_perturbed : normal_random;
                }
                if (candidate == 1U || candidate == 3U || candidate == 4U || candidate == 5U || candidate == 10U ||
                    candidate == 11U || candidate == 13U || candidate == 14U)
                    continue;
                const std::uint32_t destination = output_count * capacity + temporary_index;
                candidate_depth[destination] = candidate_value;
                candidate_normal[3U * destination + 0U] = candidate_n.x;
                candidate_normal[3U * destination + 1U] = candidate_n.y;
                candidate_normal[3U * destination + 2U] = candidate_n.z;
                ++output_count;
            }
            while (output_count < 8U)
            {
                const std::uint32_t destination = output_count * capacity + temporary_index;
                candidate_depth[destination] = 0.0F;
                candidate_normal[3U * destination + 0U] = 0.0F;
                candidate_normal[3U * destination + 1U] = 0.0F;
                candidate_normal[3U * destination + 2U] = 0.0F;
                ++output_count;
            }
        }

        __device__ __forceinline__ bool recovered_cost_project_type0(const PatchMatchCamera& camera,
                                                                     const RecoveredFloat3& point,
                                                                     float& projection_x,
                                                                     float& projection_y)
        {
            if (point.z <= 0.0F)
                return false;
            const float inverse_z = __fdiv_rn(1.0F, point.z);
            const float x = __fmul_rn(point.x, inverse_z);
            const float y = __fmul_rn(point.y, inverse_z);
            float px = __fmul_rn(camera.f, x);
            px = __fmaf_rn(camera.b1, x, px);
            px = __fmaf_rn(camera.b2, y, px);
            float py = __fmul_rn(camera.f, y);
            px = __fadd_rn(px, camera.cx);
            px = __fmaf_rn(__uint2float_rn(camera.width_original), 0.5F, px);
            py = __fadd_rn(py, camera.cy);
            py = __fmaf_rn(__uint2float_rn(camera.height_original), 0.5F, py);
            projection_x = px;
            projection_y = py;
            return true;
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_cost_intersect_plane(const RecoveredFloat3& plane_point,
                                                                                  const RecoveredFloat3& normal,
                                                                                  const RecoveredFloat3& ray_direction)
        {
            const auto cost_dot = [](const RecoveredFloat3& a, const RecoveredFloat3& b)
            { return __fmaf_rn(a.z, b.z, __fmaf_rn(a.x, b.x, __fmul_rn(a.y, b.y))); };
            const float pace = cost_dot(ray_direction, normal);
            if (fabsf(pace) <= 0.001F)
                return {FLT_MAX, FLT_MAX, FLT_MAX};
            const float distance = __fdiv_rn(cost_dot(plane_point, normal), pace);
            return {__fmul_rn(ray_direction.x, distance),
                    __fmul_rn(ray_direction.y, distance),
                    __fmul_rn(ray_direction.z, distance)};
        }

        __device__ __forceinline__ void recovered_cost_pixel(std::uint32_t logical,
                                                             std::uint32_t width,
                                                             bool checkerboard,
                                                             std::uint32_t checkerboard_step,
                                                             bool fourth,
                                                             std::uint32_t& x,
                                                             std::uint32_t& y)
        {
            if (checkerboard)
            {
                if (!fourth)
                {
                    const std::uint32_t half = (width + 1U) / 2U;
                    y = logical / half;
                    x = ((y & 1U) + checkerboard_step) % 2U + 2U * (logical % half);
                }
                else
                {
                    const std::uint32_t part = (width + 3U) / 4U;
                    y = 1U + 2U * logical / part;
                    x = 1U + ((((y / 2U) & 1U) + checkerboard_step) % 2U) * 2U + 4U * (logical % part);
                }
            }
            else if (!fourth)
            {
                y = logical / width;
                x = logical % width;
            }
            else
            {
                const std::uint32_t half = (width + 1U) / 2U;
                y = 1U + 2U * (logical / half);
                x = 1U + 2U * (logical % half);
            }
        }

        template <typename Sample> __device__ __forceinline__ float recovered_cost_sample_to_float(Sample value)
        {
            if constexpr (sizeof(Sample) == 1U)
                return __fdiv_rn(__uint2float_rn(value), 255.0F);
            else if constexpr (sizeof(Sample) == 2U)
                return __fdiv_rn(__uint2float_rn(value), 65535.0F);
            else
                return value;
        }

        template <typename Sample>
        __device__ __forceinline__ Sample recovered_cost_texture_sample(cudaTextureObject_t texture, float x, float y)
        {
            const float sampled = tex2D<float>(texture, x, y);
            if constexpr (sizeof(Sample) == 1U)
            {
                const std::uint8_t value =
                    static_cast<std::uint8_t>(static_cast<std::uint32_t>(__fmaf_rn(sampled, 255.0F, 0.5F)));
                return value == 0U ? 1U : value;
            }
            else if constexpr (sizeof(Sample) == 2U)
            {
                const std::uint16_t value =
                    static_cast<std::uint16_t>(static_cast<std::uint32_t>(__fmaf_rn(sampled, 65535.0F, 0.5F)));
                return value == 0U ? 1U : value;
            }
            else
            {
                return sampled;
            }
        }

        template <typename Sample, std::uint32_t Radius, std::uint32_t Hypotheses>
        __device__ float recovered_cost_ncc(std::uint32_t x_depth,
                                            std::uint32_t y_depth,
                                            bool detailed,
                                            std::uint32_t depth_downscale,
                                            float candidate_depth,
                                            const RecoveredFloat3& candidate_normal,
                                            float coarse_depth,
                                            float coarse_radius,
                                            float deviation_multiplier,
                                            const Sample* reference_image,
                                            cudaTextureObject_t texture,
                                            const std::uint8_t* neighbor_mask,
                                            const PatchMatchCamera& reference_camera,
                                            const PatchMatchCamera& neighbor_camera)
        {
            std::uint32_t image_downscale = depth_downscale;
            if (detailed)
                image_downscale /= 2U;
            const std::uint32_t image_width =
                (reference_camera.width_original + image_downscale - 1U) / image_downscale;
            const std::uint32_t image_height =
                (reference_camera.height_original + image_downscale - 1U) / image_downscale;
            std::int32_t x = static_cast<std::int32_t>(x_depth);
            std::int32_t y = static_cast<std::int32_t>(y_depth);
            std::int32_t from_x;
            std::int32_t to_x;
            std::int32_t from_y;
            std::int32_t to_y;
            if (!detailed)
            {
                from_x = max(0, x - static_cast<std::int32_t>(Radius) + 1);
                to_x = min(static_cast<std::int32_t>(image_width), x + static_cast<std::int32_t>(Radius));
                from_y = max(0, y - static_cast<std::int32_t>(Radius) + 1);
                to_y = min(static_cast<std::int32_t>(image_height), y + static_cast<std::int32_t>(Radius));
            }
            else
            {
                x *= 2;
                y *= 2;
                from_x = max(0, x - static_cast<std::int32_t>(Radius) + 1);
                to_x = min(static_cast<std::int32_t>(image_width), x + static_cast<std::int32_t>(Radius) + 1);
                from_y = max(0, y - static_cast<std::int32_t>(Radius) + 1);
                to_y = min(static_cast<std::int32_t>(image_height), y + static_cast<std::int32_t>(Radius) + 1);
            }
            float mean0 = 0.0F;
            float mean_square0 = 0.0F;
            float count0 = 0.0F;
#pragma unroll 1
            for (std::int32_t sx = from_x; sx < to_x; ++sx)
            {
                const bool parity = (((from_y & 1) == (sx & 1)) != ((x & 1) == (y & 1)));
#pragma unroll 1
                for (std::int32_t sy = from_y + static_cast<int>(parity); sy < to_y; sy += 2)
                {
                    const float value = recovered_cost_sample_to_float(
                        reference_image[static_cast<std::uint32_t>(sy) * image_width + static_cast<std::uint32_t>(sx)]);
                    mean0 = __fadd_rn(mean0, value);
                    mean_square0 = __fmaf_rn(value, value, mean_square0);
                    count0 = __fadd_rn(count0, 1.0F);
                }
            }
            mean0 = __fdiv_rn(mean0, count0);
            mean_square0 = __fdiv_rn(mean_square0, count0);
            const float variance0 = fmaxf(0.0F, __fmaf_rn(-mean0, mean0, mean_square0));
            const float dev0 = __fsqrt_rn(variance0);

            RecoveredFloat3 center_local = recovered_unproject_local_type0(reference_camera,
                                                                           __fadd_rn(__uint2float_rn(x_depth), 0.5F),
                                                                           __fadd_rn(__uint2float_rn(y_depth), 0.5F),
                                                                           candidate_depth,
                                                                           depth_downscale);
            const RecoveredFloat3 center_neighbor = recovered_final_transform_point(neighbor_camera, center_local);
            float center_px;
            float center_py;
            if (!recovered_cost_project_type0(neighbor_camera, center_neighbor, center_px, center_py) ||
                center_px < 0.0F || center_px >= neighbor_camera.width_original || center_py < 0.0F ||
                center_py >= neighbor_camera.height_original)
                return -1.0F;
            const std::uint32_t level0 = neighbor_camera.pyramid_level0_downscale;
            const std::uint32_t mask_width = (neighbor_camera.width_original + level0 - 1U) / level0;
            const std::uint32_t mask_height = (neighbor_camera.height_original + level0 - 1U) / level0;
            const std::uint32_t mask_x = static_cast<std::uint32_t>(__fdiv_rn(center_px, __uint2float_rn(level0)));
            const std::uint32_t mask_y = static_cast<std::uint32_t>(__fdiv_rn(center_py, __uint2float_rn(level0)));
            if (mask_x >= mask_width || mask_y >= mask_height)
                return -1.0F;
            const std::uint32_t mask_bit = mask_y * mask_width + mask_x;
            if ((neighbor_mask[mask_bit / 8U] & (1U << (mask_bit & 7U))) != 0U)
                return -1.0F;

            const float shift = static_cast<float>(Radius - 1U);
            auto border_projection = [&](float sample_x, float sample_y, float& px, float& py)
            {
                RecoveredFloat3 ray = recovered_unproject_local_type0(
                    reference_camera, sample_x, sample_y, candidate_depth, depth_downscale);
                ray = recovered_refinement_normalize(ray);
                const RecoveredFloat3 intersection =
                    recovered_cost_intersect_plane(center_local, candidate_normal, ray);
                if (intersection.x == FLT_MAX)
                    return false;
                const RecoveredFloat3 neighbor_point = recovered_final_transform_point(neighbor_camera, intersection);
                return recovered_cost_project_type0(neighbor_camera, neighbor_point, px, py);
            };
            float border_x_px;
            float border_x_py;
            float border_y_px;
            float border_y_py;
            if (!border_projection(__fadd_rn(__fadd_rn(__uint2float_rn(x_depth), shift), 0.5F),
                                   __fadd_rn(__uint2float_rn(y_depth), 0.5F),
                                   border_x_px,
                                   border_x_py) ||
                !border_projection(__fadd_rn(__uint2float_rn(x_depth), 0.5F),
                                   __fadd_rn(__fadd_rn(__uint2float_rn(y_depth), shift), 0.5F),
                                   border_y_px,
                                   border_y_py))
                return -1.0F;
            const float border = __fmul_rn(__fmul_rn(__uint2float_rn(Radius), __uint2float_rn(image_downscale)), 4.0F);
            const auto outside = [&](float px, float py)
            {
                return px < -border || px >= neighbor_camera.width_original + border || py < -border ||
                       py >= neighbor_camera.height_original + border;
            };
            if (outside(border_x_px, border_x_py) || outside(border_y_px, border_y_py))
                return -1.0F;
            float dx_x = __fdiv_rn(__fsub_rn(border_x_px, center_px), shift);
            float dx_y = __fdiv_rn(__fsub_rn(border_x_py, center_py), shift);
            float dy_x = __fdiv_rn(__fsub_rn(border_y_px, center_px), shift);
            float dy_y = __fdiv_rn(__fsub_rn(border_y_py, center_py), shift);
            if (detailed)
            {
                dx_x = __fmul_rn(dx_x, 0.5F);
                dx_y = __fmul_rn(dx_y, 0.5F);
                dy_x = __fmul_rn(dy_x, 0.5F);
                dy_y = __fmul_rn(dy_y, 0.5F);
            }
            const float from_dx =
                __fsub_rn(__fadd_rn(__int2float_rn(x), detailed ? 0.5F : 0.0F), __int2float_rn(from_x));
            const float from_dy =
                __fsub_rn(__fadd_rn(__int2float_rn(y), detailed ? 0.5F : 0.0F), __int2float_rn(from_y));
            // The old PTX's unqualified mul/sub pairs contract on sm89 to FFMA
            // (uchar<3,8> SASS 0x15110/0x15120, then 0x15170/0x15180).
            // Separate RN operations change rare cost results: same-run camera114
            // iteration1 exposes two differences, both removed by these four FMAs.
            float projection_from_x = __fmaf_rn(-dx_x, from_dx, center_px);
            projection_from_x = __fmaf_rn(-dy_x, from_dy, projection_from_x);
            float projection_from_y = __fmaf_rn(-dx_y, from_dx, center_py);
            projection_from_y = __fmaf_rn(-dy_y, from_dy, projection_from_y);
            const float pixel_size_x = __fsqrt_rn(__fmaf_rn(dx_x, dx_x, __fmul_rn(dx_y, dx_y)));
            const float pixel_size_y = __fsqrt_rn(__fmaf_rn(dy_x, dy_x, __fmul_rn(dy_y, dy_y)));
            float pixel_size = fmaxf(pixel_size_x, pixel_size_y);
            pixel_size = __fdiv_rn(pixel_size, __uint2float_rn(level0));
            float level_downscale = __uint2float_rn(level0);
            std::int32_t level_width = static_cast<std::int32_t>(mask_width);
            std::int32_t level_height = static_cast<std::int32_t>(mask_height);
            std::int32_t level = 0;
            float mip_x = 0.0F;
            float mip_y = 0.0F;
            while (pixel_size >= 1.5F &&
                   ((((level_width + 1) / 2 >= 64) && ((level_height + 1) / 2 >= 64)) || level <= 1))
            {
                pixel_size = __fmul_rn(pixel_size, 0.5F);
                level_downscale = __fadd_rn(level_downscale, level_downscale);
                if (level == 0)
                    mip_y = __int2float_rn(level_height);
                else
                    mip_x = __fadd_rn(mip_x, __int2float_rn(level_width));
                level_width = (level_width + 1) / 2;
                level_height = (level_height + 1) / 2;
                ++level;
            }

            Sample neighbor_colors[18];
            std::uint32_t color_count = 0U;
            std::uint32_t failed = 0U;
            std::uint32_t ok = 0U;
#pragma unroll 1
            for (std::int32_t sx = from_x; sx < to_x; ++sx)
            {
                const bool parity = (((from_y & 1) == (sx & 1)) != ((x & 1) == (y & 1)));
#pragma unroll 1
                for (std::int32_t sy = from_y + static_cast<int>(parity); sy < to_y; sy += 2)
                {
                    const float local_x = __int2float_rn(sx - from_x);
                    const float local_y = __int2float_rn(sy - from_y);
                    const float column_px = __fmaf_rn(dx_x, local_x, projection_from_x);
                    const float column_py = __fmaf_rn(dx_y, local_x, projection_from_y);
                    const float px = __fmaf_rn(dy_x, local_y, column_px);
                    const float py = __fmaf_rn(dy_y, local_y, column_py);
                    bool invalid = px <= 0.0F || px >= neighbor_camera.width_original || py <= 0.0F ||
                                   py >= neighbor_camera.height_original;
                    if (!invalid)
                    {
                        const std::uint32_t mx = static_cast<std::uint32_t>(__fdiv_rn(px, __uint2float_rn(level0)));
                        const std::uint32_t my = static_cast<std::uint32_t>(__fdiv_rn(py, __uint2float_rn(level0)));
                        invalid = mx >= mask_width || my >= mask_height;
                        if (!invalid)
                        {
                            const std::uint32_t bit = my * mask_width + mx;
                            invalid = (neighbor_mask[bit / 8U] & (1U << (bit & 7U))) != 0U;
                        }
                    }
                    Sample color = static_cast<Sample>(0);
                    if (invalid)
                    {
                        ++failed;
                    }
                    else
                    {
                        ++ok;
                        const float tx = __fadd_rn(mip_x, __fdiv_rn(px, level_downscale));
                        const float ty = __fadd_rn(mip_y, __fdiv_rn(py, level_downscale));
                        color = recovered_cost_texture_sample<Sample>(texture, tx, ty);
                    }
                    neighbor_colors[color_count++] = color;
                }
            }
            const std::uint32_t max_pixels =
                detailed ? (2U * Radius) * (2U * Radius) / 2U : ((2U * Radius - 1U) * (2U * Radius - 1U) + 1U) / 2U;
            if (failed > color_count * 60U / 100U || ok < max_pixels * 45U / 100U)
                return -1.0F;
            float mean1 = 0.0F;
            float mean_square1 = 0.0F;
            float count1 = 0.0F;
#pragma unroll 1
            for (std::uint32_t index = 0U; index < color_count; ++index)
            {
                if (neighbor_colors[index] == 0U)
                    continue;
                const float value = recovered_cost_sample_to_float(neighbor_colors[index]);
                mean1 = __fadd_rn(mean1, value);
                mean_square1 = __fmaf_rn(value, value, mean_square1);
                count1 = __fadd_rn(count1, 1.0F);
            }
            mean1 = __fdiv_rn(mean1, count1);
            mean_square1 = __fdiv_rn(mean_square1, count1);
            const float variance1 = fmaxf(0.0F, __fsub_rn(mean_square1, __fmul_rn(mean1, mean1)));
            const float dev1 = __fsqrt_rn(variance1);
            float zncc = 0.0F;
            std::uint32_t color_index = 0U;
#pragma unroll 1
            for (std::int32_t sx = from_x; sx < to_x; ++sx)
            {
                const bool parity = (((from_y & 1) == (sx & 1)) != ((x & 1) == (y & 1)));
#pragma unroll 1
                for (std::int32_t sy = from_y + static_cast<int>(parity); sy < to_y; sy += 2)
                {
                    const Sample color = neighbor_colors[color_index++];
                    if (color == 0U)
                        continue;
                    const float a = recovered_cost_sample_to_float(
                        reference_image[static_cast<std::uint32_t>(sy) * image_width + static_cast<std::uint32_t>(sx)]);
                    const float b = recovered_cost_sample_to_float(color);
                    zncc = __fmaf_rn(__fsub_rn(a, mean0), __fsub_rn(b, mean1), zncc);
                }
            }
            float threshold = __fmul_rn(0.0025F, deviation_multiplier);
            const float delta = fabsf(__fsub_rn(candidate_depth, coarse_depth));
            const float alpha = __fmul_rn(3.0F, coarse_radius);
            if (coarse_depth != 0.0F && delta < alpha)
                threshold = __fmul_rn(0.00025F, deviation_multiplier);
            if (!(dev0 > threshold && dev1 > threshold))
                return -1.0F;
            zncc = __fdiv_rn(zncc, __fmul_rn(__fmul_rn(dev0, dev1), count1));
            float result = __fmul_rn(__fsub_rn(1.0F, zncc), 0.5F);
            if (coarse_depth != 0.0F && delta < alpha)
            {
                const float ratio = __fdiv_rn(delta, alpha);
                const float bonus = __fmul_rn(0.25F, __fsub_rn(1.0F, ratio));
                result = fminf(__fmul_rn(result, __fsub_rn(1.0F, bonus)), fmaxf(0.000001F, result));
            }
            return fmaxf(0.0F, fminf(result, 1.0F));
        }

        template <typename Sample, std::uint32_t Radius, std::uint32_t Hypotheses>
        __global__ __launch_bounds__(128, 1) void patchmatch_cost_kernel(const float* depth,
                                                                         const std::uint8_t* normal,
                                                                         const float* cost,
                                                                         const float* coarse_depth,
                                                                         const float* coarse_radius,
                                                                         PatchMatchCamera reference_camera,
                                                                         std::uint32_t depth_downscale,
                                                                         const Sample* reference_image,
                                                                         std::uint32_t detailed,
                                                                         float deviation_multiplier,
                                                                         PatchMatchCamera neighbor_camera,
                                                                         std::uint32_t neighbor_level,
                                                                         std::uint32_t result_index,
                                                                         cudaTextureObject_t texture,
                                                                         const std::uint64_t* mask_offsets,
                                                                         const std::uint8_t* masks,
                                                                         const float* candidate_depth,
                                                                         const float* candidate_normal,
                                                                         float* neighbor_cost,
                                                                         std::uint32_t checkerboard,
                                                                         std::uint32_t checkerboard_step,
                                                                         std::uint32_t fourth,
                                                                         std::uint32_t pixel_offset)
        {
            (void)depth;
            (void)normal;
            (void)cost;
            const std::uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t hypothesis = thread % Hypotheses;
            const std::uint32_t temporary_index = thread / Hypotheses;
            const std::uint32_t logical = pixel_offset + temporary_index;
            const std::uint32_t width = (reference_camera.width_original + depth_downscale - 1U) / depth_downscale;
            const std::uint32_t height = (reference_camera.height_original + depth_downscale - 1U) / depth_downscale;
            std::uint32_t x;
            std::uint32_t y;
            recovered_cost_pixel(logical, width, checkerboard != 0U, checkerboard_step, fourth != 0U, x, y);
            if (x >= width || y >= height)
                return;
            constexpr std::uint32_t capacity = 128U * 1024U;
            const std::uint32_t candidate = hypothesis * capacity + temporary_index;
            const float candidate_value = candidate_depth[candidate];
            float result = -1.0F;
            if (candidate_value != 0.0F)
            {
                const RecoveredFloat3 candidate_n{candidate_normal[3U * candidate],
                                                  candidate_normal[3U * candidate + 1U],
                                                  candidate_normal[3U * candidate + 2U]};
                const std::uint32_t pixel = y * width + x;
                result = recovered_cost_ncc<Sample, Radius, Hypotheses>(x,
                                                                        y,
                                                                        detailed != 0U,
                                                                        depth_downscale,
                                                                        candidate_value,
                                                                        candidate_n,
                                                                        coarse_depth[pixel],
                                                                        coarse_radius[pixel],
                                                                        deviation_multiplier,
                                                                        reference_image,
                                                                        texture,
                                                                        masks + mask_offsets[neighbor_level],
                                                                        reference_camera,
                                                                        neighbor_camera);
            }
            neighbor_cost[result_index * capacity * Hypotheses + temporary_index * Hypotheses + hypothesis] = result;
        }

        __global__ __launch_bounds__(128, 1) void patchmatch_coarse_to_precise_kernel(const float* depth,
                                                                                      const std::uint8_t* normal,
                                                                                      const float* cost,
                                                                                      std::uint32_t width_original,
                                                                                      std::uint32_t height_original,
                                                                                      std::uint32_t depth_downscale,
                                                                                      float* candidate_depth,
                                                                                      float* candidate_normal,
                                                                                      std::uint32_t pixel_offset)
        {
            constexpr std::uint32_t capacity = 128U * 1024U;
            const std::uint32_t temporary_index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t global_index = pixel_offset + temporary_index;
            const std::uint32_t width = (width_original + depth_downscale - 1U) / depth_downscale;
            const std::uint32_t height = (height_original + depth_downscale - 1U) / depth_downscale;
            const std::uint32_t x = global_index % width;
            const std::uint32_t y = global_index / width;
            if (x >= width || y >= height || ((x & 1U) != 0U && (y & 1U) != 0U))
                return;

            float depth_sum = 0.0F;
            RecoveredFloat3 normal_sum{0.0F, 0.0F, 0.0F};
            std::int32_t sample_count = 0;
            std::uint32_t next_hypothesis = 0U;
            for (std::uint32_t sample = 0U; sample < 6U; ++sample)
            {
                float sample_depth = 0.0F;
                RecoveredFloat3 sample_normal{0.0F, 0.0F, 0.0F};
                if (sample == 5U)
                {
                    if (sample_count == 0)
                        continue;
                    sample_depth = __fdiv_rn(depth_sum, __int2float_rn(sample_count));
                    const float squared =
                        __fmaf_rn(normal_sum.z,
                                  normal_sum.z,
                                  __fmaf_rn(normal_sum.x, normal_sum.x, __fmul_rn(normal_sum.y, normal_sum.y)));
                    const float length = sqrtf(squared);
                    sample_normal.x = __fdiv_rn(normal_sum.x, length);
                    sample_normal.y = __fdiv_rn(normal_sum.y, length);
                    sample_normal.z = __fdiv_rn(normal_sum.z, length);
                }
                else if (sample == 4U)
                {
                    const std::uint32_t pixel = y * width + x;
                    sample_depth = depth[pixel];
                    sample_normal = unpack_recovered_normal(normal + 3U * pixel);
                }
                else
                {
                    std::uint32_t sample_x = x - 1U + (x & 1U);
                    std::uint32_t sample_y = y - 1U + (y & 1U);
                    sample_x += sample & 1U;
                    sample_y += sample / 2U;
                    if (sample_x >= width || sample_y >= height)
                        continue;
                    const std::uint32_t pixel = sample_y * width + sample_x;
                    sample_depth = depth[pixel];
                    sample_normal = unpack_recovered_normal(normal + 3U * pixel);
                    if (sample_depth != 0.0F && cost[pixel] < 0.15F)
                    {
                        depth_sum = __fadd_rn(depth_sum, sample_depth);
                        normal_sum.x = __fadd_rn(normal_sum.x, sample_normal.x);
                        normal_sum.y = __fadd_rn(normal_sum.y, sample_normal.y);
                        normal_sum.z = __fadd_rn(normal_sum.z, sample_normal.z);
                        ++sample_count;
                    }
                }
                if (sample == 1U || sample == 2U || sample == 3U || sample == 4U)
                    continue;
                const std::uint32_t destination = next_hypothesis * capacity + temporary_index;
                candidate_depth[destination] = sample_depth;
                candidate_normal[3U * destination + 0U] = sample_normal.x;
                candidate_normal[3U * destination + 1U] = sample_normal.y;
                candidate_normal[3U * destination + 2U] = sample_normal.z;
                ++next_hypothesis;
            }
            while (next_hypothesis < 2U)
            {
                const std::uint32_t destination = next_hypothesis * capacity + temporary_index;
                candidate_depth[destination] = 0.0F;
                candidate_normal[3U * destination + 0U] = 0.0F;
                candidate_normal[3U * destination + 1U] = 0.0F;
                candidate_normal[3U * destination + 2U] = 0.0F;
                ++next_hypothesis;
            }
        }

        // Source reconstruction of
        // cuda::pm_filtering_clear_depth_map_wrt_filtered_mask. Complete 128-thread
        // blocks preserve the target instruction path; the explicit batch bound also
        // permits a final partial block for source-CUDA production images.
        __global__ __launch_bounds__(128,
                                     1) void patchmatch_filter_clear_depth_kernel(float* depth,
                                                                                  const std::uint8_t* filtered_mask,
                                                                                  std::uint32_t* counter_not_empty,
                                                                                  std::uint32_t pixel_offset,
                                                                                  std::size_t work_items)
        {
            __shared__ std::uint32_t local_not_empty;
            if (threadIdx.x == 0U)
                local_not_empty = 0U;
            __syncthreads();

            const std::size_t local_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (local_index < work_items)
            {
                const std::size_t index = local_index + pixel_offset;
                if (filtered_mask[index] != 0U)
                {
                    depth[index] = 0.0F;
                }
                else if (depth[index] != 0.0F)
                {
                    atomicAdd(&local_not_empty, 1U);
                }
            }

            __syncthreads();
            if (threadIdx.x == 0U)
                atomicAdd(counter_not_empty, local_not_empty);
        }

        // Source reconstruction of cuda::pm_filtering_check_cost.  The camera and
        // downscale remain in the public ABI but are dead parameters in the recovered
        // PTX. Complete blocks remain target-equivalent; the source-only active bound
        // prevents the final partial block from reaching allocation tails.
        __global__ __launch_bounds__(128, 1) void patchmatch_filter_check_cost_kernel(float* depth,
                                                                                      const float* cost,
                                                                                      std::uint32_t* counter_no_cost,
                                                                                      std::uint32_t* counter_big_cost,
                                                                                      std::uint32_t pixel_offset,
                                                                                      std::size_t work_items)
        {
            __shared__ std::uint32_t local_no_cost;
            __shared__ std::uint32_t local_big_cost;
            if (threadIdx.x == 0U)
            {
                local_no_cost = 0U;
                local_big_cost = 0U;
            }
            __syncthreads();

            const std::size_t local_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (local_index < work_items)
            {
                const std::size_t index = local_index + pixel_offset;
                const float value = cost[index];
                if (value == -1.0F)
                {
                    depth[index] = 0.0F;
                    atomicAdd(&local_no_cost, 1U);
                }
                else if (value > 0.15F)
                {
                    depth[index] = 0.0F;
                    atomicAdd(&local_big_cost, 1U);
                }
            }

            __syncthreads();
            if (threadIdx.x == 0U)
            {
                atomicAdd(counter_no_cost, local_no_cost);
                atomicAdd(counter_big_cost, local_big_cost);
            }
        }

        __global__ __launch_bounds__(128, 1) void patchmatch_filter_check_neighbours_kernel(
            const float* depth,
            std::uint8_t* filtered_mask,
            PatchMatchCamera camera,
            std::uint32_t downscale,
            float depth_min,
            float depth_max,
            std::uint32_t* counter_no_neighbours,
            std::uint32_t* counter_no_close_neighbours,
            std::uint32_t pixel_offset,
            std::size_t work_items)
        {
            __shared__ std::uint32_t local_no_neighbours;
            __shared__ std::uint32_t local_no_close_neighbours;
            if (threadIdx.x == 0U)
            {
                local_no_neighbours = 0U;
                local_no_close_neighbours = 0U;
            }
            __syncthreads();

            const std::size_t local_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const bool active = local_index < work_items;
            const std::size_t index = local_index + pixel_offset;
            const std::uint32_t width = (camera.width_original + downscale - 1U) / downscale;
            const std::uint32_t height = (camera.height_original + downscale - 1U) / downscale;
            const std::uint32_t y = static_cast<std::uint32_t>(index / width);
            const std::uint32_t x = static_cast<std::uint32_t>(index % width);
            if (active)
            {
                filtered_mask[index] = 0U;
            }
            const float center = active ? depth[index] : 0.0F;
            if (active && center != 0.0F)
            {
                const float absolute_delta = __fmul_rn(__fadd_rn(depth_max, -depth_min), 0.05F);
                const float absolute_low = __fadd_rn(center, -absolute_delta);
                const float absolute_high = __fadd_rn(center, absolute_delta);
                const float relative_low = __fmul_rn(center, 0.9F);
                const float relative_high = __fmul_rn(center, 1.1F);
                const bool absolute_mode = (camera.type & ~1U) == 8U;
                std::uint32_t neighbours = 0U;
                std::uint32_t close_neighbours = 0U;
                for (std::int32_t dx = -1; dx <= 1; ++dx)
                {
                    for (std::int32_t dy = -1; dy <= 1; ++dy)
                    {
                        if (dx == 0 && dy == 0)
                            continue;
                        const std::int32_t neighbor_x = static_cast<std::int32_t>(x) + dx;
                        const std::int32_t neighbor_y = static_cast<std::int32_t>(y) + dy;
                        if (neighbor_x < 0 || neighbor_y < 0 || neighbor_x >= static_cast<std::int32_t>(width) ||
                            neighbor_y >= static_cast<std::int32_t>(height))
                            continue;
                        const float value = depth[static_cast<std::uint32_t>(neighbor_y) * width +
                                                  static_cast<std::uint32_t>(neighbor_x)];
                        if (value == 0.0F)
                            continue;
                        ++neighbours;
                        const float low = absolute_mode ? absolute_low : relative_low;
                        const float high = absolute_mode ? absolute_high : relative_high;
                        if (value >= low && value <= high)
                            ++close_neighbours;
                    }
                }
                if (neighbours <= 3U)
                {
                    filtered_mask[index] = 1U;
                    atomicAdd(&local_no_neighbours, 1U);
                }
                else if (close_neighbours <= 2U)
                {
                    filtered_mask[index] = 1U;
                    atomicAdd(&local_no_close_neighbours, 1U);
                }
            }

            __syncthreads();
            if (threadIdx.x == 0U)
            {
                atomicAdd(counter_no_neighbours, local_no_neighbours);
                atomicAdd(counter_no_close_neighbours, local_no_close_neighbours);
            }
        }

        __device__ __forceinline__ bool recovered_float3_is_zero(const RecoveredFloat3& value)
        {
            return value.x == 0.0F && value.y == 0.0F && value.z == 0.0F;
        }

        __device__ __forceinline__ float recovered_squared_distance(const RecoveredFloat3& left,
                                                                    const RecoveredFloat3& right)
        {
            const float dx = __fadd_rn(left.x, -right.x);
            const float dy = __fadd_rn(left.y, -right.y);
            const float dz = __fadd_rn(left.z, -right.z);
            return __fmaf_rn(dz, dz, __fmaf_rn(dx, dx, __fmul_rn(dy, dy)));
        }

        // Production camera-to-world transforms are affine.  Keep the generic
        // projective form for captured or synthetic camera ABIs whose last row
        // is not exactly [0, 0, 0, 1].
        template <bool AffineTransform>
        __device__ __forceinline__ bool recovered_unproject_perspective(const PatchMatchCamera& camera,
                                                                        std::uint32_t pixel_x,
                                                                        std::uint32_t pixel_y,
                                                                        float depth,
                                                                        std::uint32_t downscale,
                                                                        RecoveredFloat3& output)
        {
            const float sample_x = __fadd_rn(__uint2float_rn(pixel_x), 0.5F);
            const float sample_y = __fadd_rn(__uint2float_rn(pixel_y), 0.5F);
            float local_y = __fmul_rn(__uint2float_rn(downscale), sample_y);
            local_y = __fadd_rn(local_y, -camera.cy);
            local_y = __fadd_rn(local_y, -__fmul_rn(__uint2float_rn(camera.height_original), 0.5F));
            local_y = __fdiv_rn(local_y, camera.f);
            float local_x = __fmul_rn(__uint2float_rn(downscale), sample_x);
            local_x = __fadd_rn(local_x, -camera.cx);
            local_x = __fadd_rn(local_x, -__fmul_rn(__uint2float_rn(camera.width_original), 0.5F));
            local_x = __fadd_rn(local_x, -__fmul_rn(local_y, camera.b2));
            local_x = __fdiv_rn(local_x, __fadd_rn(camera.f, camera.b1));

            const float x = __fmul_rn(local_x, depth);
            const float y = __fmul_rn(local_y, depth);
            const float z = depth;
            const float* transform =
                reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(&camera) + 208U);
            const auto row = [&](std::uint32_t index)
            {
                const std::uint32_t base = 4U * index;
                float value = __fmul_rn(y, transform[base + 1U]);
                value = __fmaf_rn(x, transform[base], value);
                value = __fmaf_rn(z, transform[base + 2U], value);
                return __fadd_rn(transform[base + 3U], value);
            };
            const float transformed_x = row(0U);
            const float transformed_y = row(1U);
            const float transformed_z = row(2U);
            const float transformed_w = row(3U);
            if constexpr (AffineTransform)
            {
                output.x = transformed_x;
                output.y = transformed_y;
                output.z = transformed_z;
            }
            else
            {
                output.x = __fdiv_rn(transformed_x, transformed_w);
                output.y = __fdiv_rn(transformed_y, transformed_w);
                output.z = __fdiv_rn(transformed_z, transformed_w);
            }
            return true;
        }

        template <std::uint32_t CameraType, bool AffineTransform = false>
        __device__ __forceinline__ bool recovered_filter_unproject(const PatchMatchCamera& camera,
                                                                   std::uint32_t pixel_x,
                                                                   std::uint32_t pixel_y,
                                                                   float depth,
                                                                   std::uint32_t downscale,
                                                                   RecoveredFloat3& output)
        {
            if constexpr (CameraType == 0U)
            {
                return recovered_unproject_perspective<AffineTransform>(
                    camera, pixel_x, pixel_y, depth, downscale, output);
            }
            else
            {
                RecoveredFloat3 local;
                const bool valid =
                    recovered_unproject_local_perspective<CameraType>(camera,
                                                                      __fadd_rn(__uint2float_rn(pixel_x), 0.5F),
                                                                      __fadd_rn(__uint2float_rn(pixel_y), 0.5F),
                                                                      depth,
                                                                      downscale,
                                                                      local);
                if (valid)
                    output = recovered_final_transform_point(camera, local);
                return valid;
            }
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_subtract(const RecoveredFloat3& left,
                                                                      const RecoveredFloat3& right)
        {
            return {
                __fadd_rn(left.x, -right.x),
                __fadd_rn(left.y, -right.y),
                __fadd_rn(left.z, -right.z),
            };
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_normalize(const RecoveredFloat3& value)
        {
            const float squared = __fmaf_rn(value.z, value.z, __fmaf_rn(value.x, value.x, __fmul_rn(value.y, value.y)));
            const float length = sqrtf(squared);
            return {
                __fdiv_rn(value.x, length),
                __fdiv_rn(value.y, length),
                __fdiv_rn(value.z, length),
            };
        }

        __device__ __forceinline__ float recovered_dot(const RecoveredFloat3& left, const RecoveredFloat3& right)
        {
            return __fmaf_rn(left.z, right.z, __fmaf_rn(left.x, right.x, __fmul_rn(left.y, right.y)));
        }

        __device__ __forceinline__ RecoveredFloat3 recovered_cross(const RecoveredFloat3& left,
                                                                   const RecoveredFloat3& right)
        {
            return {
                __fmaf_rn(left.y, right.z, -__fmul_rn(left.z, right.y)),
                __fmaf_rn(left.z, right.x, -__fmul_rn(left.x, right.z)),
                __fmaf_rn(left.x, right.y, -__fmul_rn(left.y, right.x)),
            };
        }

        __device__ __forceinline__ std::uint8_t recovered_pack_normal_component(float value)
        {
            value = fmaxf(-1.0F, fminf(value, 1.0F));
            value = __fadd_rn(value, 1.0F);
            value = __fdiv_rn(__fmul_rn(value, 255.0F), 2.0F);
            value = __fadd_rn(value, 0.5F);
            return static_cast<std::uint8_t>(value);
        }

        // Perspective-camera specialization of cuda::pm_filtering_normals.  The
        // block-local reductions, operation order, thresholds and last-valid-neighbor
        // fallback mirror the recovered OpenCL body and sm_30 PTX.
        template <std::uint32_t CameraType>
        __global__ __launch_bounds__(128, 1) void patchmatch_filter_normals_kernel(
            const float* depth,
            const std::uint8_t* normal,
            std::uint8_t* estimated_normal,
            std::uint8_t estimate_normal_map,
            std::uint8_t* filtered_mask,
            PatchMatchCamera camera,
            std::uint32_t downscale,
            std::uint32_t* counter_inconsistent_normal,
            std::uint32_t* counter_bad_view_angle_estimated_normal,
            std::uint32_t* counter_bad_view_angle_found_normal,
            float* counter_cos_sum,
            std::uint32_t* counter_ncos_sum,
            std::uint32_t pixel_offset,
            std::size_t work_items)
        {
            __shared__ std::uint32_t local_inconsistent;
            __shared__ std::uint32_t local_bad_estimated;
            __shared__ std::uint32_t local_bad_found;
            __shared__ float local_cos_sum;
            __shared__ std::uint32_t local_ncos_sum;
            if (threadIdx.x == 0U)
            {
                local_inconsistent = 0U;
                local_bad_estimated = 0U;
                local_bad_found = 0U;
                local_cos_sum = 0.0F;
                local_ncos_sum = 0U;
            }
            __syncthreads();

            const std::size_t local_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const bool active = local_index < work_items;
            const std::size_t global_index = local_index + pixel_offset;
            const std::uint32_t width = (camera.width_original + downscale - 1U) / downscale;
            const std::uint32_t height = (camera.height_original + downscale - 1U) / downscale;
            const std::uint32_t y = static_cast<std::uint32_t>(global_index / width);
            const std::uint32_t x = static_cast<std::uint32_t>(global_index % width);
            if (active)
            {
                filtered_mask[global_index] = 0U;
                const float center_depth = depth[global_index];
                RecoveredFloat3 center{};
                RecoveredFloat3 deeper{};
                bool valid = false;
                if (center_depth != 0.0F)
                {
                    valid = recovered_filter_unproject<CameraType>(camera, x, y, center_depth, downscale, center);
                    if (valid)
                    {
                        valid = recovered_filter_unproject<CameraType>(
                            camera, x, y, __fmul_rn(center_depth, 1.1F), downscale, deeper);
                    }
                }
                if (valid)
                {
                    const RecoveredFloat3 found_normal = unpack_recovered_normal(normal + 3U * global_index);
                    RecoveredFloat3 view_ray = recovered_normalize(recovered_subtract(deeper, center));
                    if (center_depth < 0.0F)
                    {
                        view_ray.x = -view_ray.x;
                        view_ray.y = -view_ray.y;
                        view_ray.z = -view_ray.z;
                    }
                    const RecoveredFloat3 negative_view{-view_ray.x, -view_ray.y, -view_ray.z};
                    const bool found_view_angle_ok = recovered_dot(found_normal, negative_view) > -0.17F;

                    constexpr std::int32_t dx[4] = {-1, 0, 1, 0};
                    constexpr std::int32_t dy[4] = {0, -1, 0, 1};
                    RecoveredFloat3 estimated{};
                    RecoveredFloat3 some_estimated{};
                    bool estimated_is_great = false;
                    bool normal_consistent = false;
                    bool estimated_view_angle_ok = false;
                    for (std::int32_t k = 0; k < 4; ++k)
                    {
                        const std::int32_t x0 = static_cast<std::int32_t>(x) + dx[k];
                        const std::int32_t y0 = static_cast<std::int32_t>(y) + dy[k];
                        const std::int32_t next = (k + 1) & 3;
                        const std::int32_t x1 = static_cast<std::int32_t>(x) + dx[next];
                        const std::int32_t y1 = static_cast<std::int32_t>(y) + dy[next];
                        if (x0 < 0 || x1 < 0 || y0 < 0 || y1 < 0 || x0 >= static_cast<std::int32_t>(width) ||
                            x1 >= static_cast<std::int32_t>(width) || y0 >= static_cast<std::int32_t>(height) ||
                            y1 >= static_cast<std::int32_t>(height))
                            continue;
                        const std::uint32_t index0 =
                            static_cast<std::uint32_t>(y0) * width + static_cast<std::uint32_t>(x0);
                        const std::uint32_t index1 =
                            static_cast<std::uint32_t>(y1) * width + static_cast<std::uint32_t>(x1);
                        const float depth0 = depth[index0];
                        const float depth1 = depth[index1];
                        if (depth0 == 0.0F || depth1 == 0.0F)
                            continue;
                        RecoveredFloat3 point0{};
                        RecoveredFloat3 point1{};
                        if (!recovered_filter_unproject<CameraType>(camera,
                                                                    static_cast<std::uint32_t>(x0),
                                                                    static_cast<std::uint32_t>(y0),
                                                                    depth0,
                                                                    downscale,
                                                                    point0) ||
                            !recovered_filter_unproject<CameraType>(camera,
                                                                    static_cast<std::uint32_t>(x1),
                                                                    static_cast<std::uint32_t>(y1),
                                                                    depth1,
                                                                    downscale,
                                                                    point1))
                            continue;
                        const RecoveredFloat3 expected = recovered_normalize(
                            recovered_cross(recovered_subtract(point1, center), recovered_subtract(point0, center)));
                        some_estimated = expected;
                        const float cosine = recovered_dot(expected, found_normal);
                        if (cosine > 0.50F)
                        {
                            atomicAdd(&local_cos_sum, cosine);
                            atomicAdd(&local_ncos_sum, 1U);
                            normal_consistent = true;
                            if (!estimated_is_great)
                                estimated = expected;
                        }
                        const float view_cosine = recovered_dot(expected, negative_view);
                        if (view_cosine > -0.17F)
                        {
                            estimated_view_angle_ok = true;
                            if (!estimated_is_great)
                                estimated = expected;
                        }
                        if (cosine > 0.50F && view_cosine > -0.17F)
                        {
                            estimated_is_great = true;
                            estimated = expected;
                        }
                    }
                    if (recovered_float3_is_zero(estimated))
                        estimated = some_estimated;
                    if (estimate_normal_map != 0U)
                    {
                        estimated_normal[3U * global_index + 0U] = recovered_pack_normal_component(estimated.x);
                        estimated_normal[3U * global_index + 1U] = recovered_pack_normal_component(estimated.y);
                        estimated_normal[3U * global_index + 2U] = recovered_pack_normal_component(estimated.z);
                    }
                    if (!normal_consistent)
                    {
                        filtered_mask[global_index] = 1U;
                        atomicAdd(&local_inconsistent, 1U);
                    }
                    else if (!estimated_view_angle_ok)
                    {
                        filtered_mask[global_index] = 1U;
                        atomicAdd(&local_bad_estimated, 1U);
                    }
                    else if (!found_view_angle_ok || recovered_float3_is_zero(estimated))
                    {
                        filtered_mask[global_index] = 1U;
                        atomicAdd(&local_bad_found, 1U);
                    }
                }
            }

            __syncthreads();
            if (threadIdx.x == 0U)
            {
                atomicAdd(counter_inconsistent_normal, local_inconsistent);
                atomicAdd(counter_bad_view_angle_estimated_normal, local_bad_estimated);
                atomicAdd(counter_bad_view_angle_found_normal, local_bad_found);
                atomicAdd(counter_cos_sum, local_cos_sum);
                atomicAdd(counter_ncos_sum, local_ncos_sum);
            }
        }

        __device__ __forceinline__ void recovered_insert_sorted_five(float value, float* values, std::uint32_t& count)
        {
            std::uint32_t index = 0U;
            while (index < count && values[index] < value)
                ++index;
            if (index < count)
            {
                std::uint32_t last = count;
                if (count < 5U)
                    ++count;
                else
                    --last;
                for (std::uint32_t move = last; move > index; --move)
                    values[move] = values[move - 1U];
            }
            else if (count < 5U)
            {
                ++count;
            }
            if (index < 5U)
                values[index] = value;
        }

        template <std::uint32_t CameraType, bool AffineTransform = false>
        __device__ __forceinline__ float recovered_estimate_sample_radius(const PatchMatchCamera& camera,
                                                                          const float* depth,
                                                                          std::int32_t center_x,
                                                                          std::int32_t center_y,
                                                                          std::uint32_t width,
                                                                          std::uint32_t height,
                                                                          std::uint32_t downscale)
        {
            RecoveredFloat3 points[9];
            for (RecoveredFloat3& point : points)
                point = {0.0F, 0.0F, 0.0F};
            for (std::int32_t dy = 0; dy < 3; ++dy)
            {
                for (std::int32_t dx = 0; dx < 3; ++dx)
                {
                    const std::int32_t x = center_x + dx - 1;
                    const std::int32_t y = center_y + dy - 1;
                    if (x < 0 || y < 0 || x >= static_cast<std::int32_t>(width) ||
                        y >= static_cast<std::int32_t>(height))
                        continue;
                    const float sample_depth =
                        depth[static_cast<std::uint32_t>(y) * width + static_cast<std::uint32_t>(x)];
                    if (sample_depth == 0.0F)
                        continue;
                    recovered_filter_unproject<CameraType, AffineTransform>(camera,
                                                                            static_cast<std::uint32_t>(x),
                                                                            static_cast<std::uint32_t>(y),
                                                                            sample_depth,
                                                                            downscale,
                                                                            points[3 * dy + dx]);
                }
            }

            float sorted_distances[5];
            std::uint32_t sorted_count = 0U;
            std::int32_t distance_count = 0;
            for (std::int32_t dy = 0; dy < 3; ++dy)
            {
                for (std::int32_t dx = 0; dx < 3; ++dx)
                {
                    const RecoveredFloat3& point = points[3 * dy + dx];
                    if (recovered_float3_is_zero(point))
                        continue;
                    if (dx + 1 < 3)
                    {
                        const RecoveredFloat3& right = points[3 * dy + dx + 1];
                        if (!recovered_float3_is_zero(right))
                        {
                            ++distance_count;
                            recovered_insert_sorted_five(
                                recovered_squared_distance(right, point), sorted_distances, sorted_count);
                        }
                    }
                    if (dy + 1 < 3)
                    {
                        const RecoveredFloat3& below = points[3 * (dy + 1) + dx];
                        if (!recovered_float3_is_zero(below))
                        {
                            ++distance_count;
                            recovered_insert_sorted_five(
                                recovered_squared_distance(below, point), sorted_distances, sorted_count);
                        }
                    }
                    if (dx + 1 < 3 && dy + 1 < 3)
                    {
                        const RecoveredFloat3& diagonal = points[3 * (dy + 1) + dx + 1];
                        if (!recovered_float3_is_zero(diagonal))
                        {
                            ++distance_count;
                            recovered_insert_sorted_five(
                                recovered_squared_distance(diagonal, point), sorted_distances, sorted_count);
                        }
                    }
                }
            }
            if (recovered_float3_is_zero(points[4]) || distance_count <= 1)
                return 0.0F;
            return sqrtf(sorted_distances[distance_count * 3 / 10]);
        }

        template <std::uint32_t CameraType, bool AffineTransform = false>
        __global__ __launch_bounds__(128, 1) void patchmatch_filter_speckles_edges_kernel(const float* depth,
                                                                                          std::uint8_t* filtered_mask,
                                                                                          PatchMatchCamera camera,
                                                                                          std::uint32_t downscale,
                                                                                          std::uint32_t pixel_offset,
                                                                                          std::size_t work_items)
        {
            const std::size_t local_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (local_index >= work_items)
                return;
            const std::size_t index = local_index + pixel_offset;
            const std::uint32_t width = (camera.width_original + downscale - 1U) / downscale;
            const std::uint32_t height = (camera.height_original + downscale - 1U) / downscale;
            const std::uint32_t y = static_cast<std::uint32_t>(index / width);
            const std::uint32_t x = static_cast<std::uint32_t>(index % width);
            std::uint8_t neighbor_mask = 0U;
            const float center_depth = depth[index];
            RecoveredFloat3 center{0.0F, 0.0F, 0.0F};
            bool center_valid = false;
            if (center_depth != 0.0F)
                center_valid = recovered_filter_unproject<CameraType, AffineTransform>(
                    camera, x, y, center_depth, downscale, center);
            if (center_valid)
            {
                const float sample_radius =
                    recovered_estimate_sample_radius<CameraType, AffineTransform>(camera,
                                                                                  depth,
                                                                                  static_cast<std::int32_t>(x),
                                                                                  static_cast<std::int32_t>(y),
                                                                                  width,
                                                                                  height,
                                                                                  downscale);
                const float radius = __fadd_rn(sample_radius, sample_radius);
                const float radius_squared = __fmul_rn(radius, radius);
                const float diagonal_radius_squared = __fadd_rn(radius_squared, radius_squared);
                std::int32_t edge_index = -1;
                for (std::int32_t dy = 0; dy < 3; ++dy)
                {
                    for (std::int32_t dx = 0; dx < 3; ++dx)
                    {
                        const std::int32_t neighbor_x = static_cast<std::int32_t>(x) + dx - 1;
                        const std::int32_t neighbor_y = static_cast<std::int32_t>(y) + dy - 1;
                        if (dx == 1 && dy == 1)
                            continue;
                        ++edge_index;
                        if (neighbor_x < 0 || neighbor_y < 0 || neighbor_x >= static_cast<std::int32_t>(width) ||
                            neighbor_y >= static_cast<std::int32_t>(height))
                            continue;
                        const std::uint32_t neighbor_index =
                            static_cast<std::uint32_t>(neighbor_y) * width + static_cast<std::uint32_t>(neighbor_x);
                        const float neighbor_depth = depth[neighbor_index];
                        if (neighbor_depth == 0.0F)
                            continue;
                        const float neighbor_sample_radius =
                            recovered_estimate_sample_radius<CameraType, AffineTransform>(
                                camera, depth, neighbor_x, neighbor_y, width, height, downscale);
                        const float neighbor_threshold = __fadd_rn(neighbor_sample_radius, neighbor_sample_radius);
                        if (neighbor_threshold < radius)
                            continue;
                        RecoveredFloat3 neighbor;
                        if (!recovered_filter_unproject<CameraType, AffineTransform>(
                                camera,
                                static_cast<std::uint32_t>(neighbor_x),
                                static_cast<std::uint32_t>(neighbor_y),
                                neighbor_depth,
                                downscale,
                                neighbor))
                            continue;
                        const float distance_squared = recovered_squared_distance(neighbor, center);
                        const float threshold = (dx == 1 || dy == 1) ? radius_squared : diagonal_radius_squared;
                        if (distance_squared < threshold)
                            neighbor_mask = static_cast<std::uint8_t>(neighbor_mask | (1U << edge_index));
                    }
                }
            }
            filtered_mask[index] = neighbor_mask;
        }

        __device__ __forceinline__ bool recovered_ooc_bit(std::uint8_t value, std::uint32_t bit)
        {
            return (value & static_cast<std::uint8_t>(1U << bit)) != 0U;
        }

        __device__ __forceinline__ std::uint32_t recovered_ooc_child_offset(std::uint32_t axis, std::uint32_t child)
        {
            if (axis == 0U)
                return child;
            if (axis == 1U)
            {
                constexpr std::uint32_t values[4] = {0U, 4U, 1U, 5U};
                return values[child];
            }
            constexpr std::uint32_t values[4] = {0U, 4U, 2U, 6U};
            return values[child];
        }

        __device__ __forceinline__ float recovered_ooc_half(const std::uint16_t* values, std::uint32_t index)
        {
            return __half2float(reinterpret_cast<const __half*>(values)[index]);
        }

        __device__ __forceinline__ void
        recovered_ooc_store_half(std::uint16_t* values, std::uint32_t index, float value)
        {
            reinterpret_cast<__half*>(values)[index] = __float2half_rn(value);
        }

        __device__ __forceinline__ float recovered_ooc_histogram_center(std::uint32_t bin)
        {
            constexpr std::uint32_t bits[10] = {0xbf800000U,
                                                0xbf471c72U,
                                                0xbf0e38e4U,
                                                0xbeaaaaaaU,
                                                0xbde38e38U,
                                                0x3de38e40U,
                                                0x3eaaaaacU,
                                                0x3f0e38e4U,
                                                0x3f471c72U,
                                                0x3f800000U};
            return __uint_as_float(bits[bin]);
        }

        __device__ __forceinline__ float recovered_ooc_histogram_left(std::uint32_t bin)
        {
            constexpr std::uint32_t bits[10] = {0xbf8e38e4U,
                                                0xbf638e39U,
                                                0xbf2aaaabU,
                                                0xbee38e38U,
                                                0xbe638e38U,
                                                0x33600000U,
                                                0x3e638e3cU,
                                                0x3ee38e3aU,
                                                0x3f2aaaabU,
                                                0x3f638e39U};
            return __uint_as_float(bits[bin]);
        }

        __device__ __forceinline__ float recovered_ooc_histogram_right(std::uint32_t bin)
        {
            constexpr std::uint32_t bits[10] = {0xbf638e39U,
                                                0xbf2aaaabU,
                                                0xbee38e3aU,
                                                0xbe638e38U,
                                                0x32000000U,
                                                0x3e638e3cU,
                                                0x3ee38e3aU,
                                                0x3f2aaaabU,
                                                0x3f638e39U,
                                                0x3f8e38e4U};
            return __uint_as_float(bits[bin]);
        }

        // Production functional-mode-1 reconstruction of cuda::ooc_update_u.
        __global__ __launch_bounds__(128, 1) void recovered_ooc_update_u_kernel(const std::uint8_t* weights,
                                                                                const std::uint8_t* histogram,
                                                                                const std::uint32_t* neighbors,
                                                                                const std::uint8_t* connectivity,
                                                                                const std::uint8_t* refinement,
                                                                                const std::uint8_t* flags,
                                                                                float* u,
                                                                                float* u_old,
                                                                                const std::uint16_t* p,
                                                                                std::uint16_t* v,
                                                                                std::uint16_t* v_old,
                                                                                const std::uint16_t* q,
                                                                                std::uint32_t count,
                                                                                float alpha,
                                                                                float beta,
                                                                                float data_weight,
                                                                                std::uint32_t offset)
        {
            constexpr float inverse_sqrt_12 = 0.28867528F;
            const std::uint32_t voxel = offset + blockIdx.x * blockDim.x + threadIdx.x;
            if (voxel >= count || (flags[voxel] & 4U) != 0U)
                return;
            const float current_u = u[voxel];
            u_old[voxel] = current_u;
            float divergence_p = 0.0F;
            float local_p[3];
            for (std::uint32_t axis = 0U; axis < 3U; ++axis)
            {
                const std::uint32_t negative = 2U * axis;
                local_p[axis] = recovered_ooc_half(p, 3U * voxel + axis);
                if (recovered_ooc_bit(connectivity[voxel], negative + 1U))
                    divergence_p = __fsub_rn(divergence_p, local_p[axis]);
                if (!recovered_ooc_bit(connectivity[voxel], negative))
                    continue;
                const std::uint32_t adjacent = neighbors[6U * voxel + negative];
                if (recovered_ooc_bit(refinement[voxel], negative))
                {
                    for (std::uint32_t child = 0U; child < 4U; ++child)
                    {
                        const float value =
                            recovered_ooc_half(p, 3U * (adjacent + recovered_ooc_child_offset(axis, child)) + axis);
                        divergence_p = __fmaf_rn(value, 0.25F, divergence_p);
                    }
                }
                else
                {
                    divergence_p = __fadd_rn(divergence_p, recovered_ooc_half(p, 3U * adjacent + axis));
                }
            }
            const float u_bar = __fmaf_rn(-__fmul_rn(inverse_sqrt_12, alpha), divergence_p, current_u);
            float total = 0.0F;
            float hard_value = 256.0F;
            for (std::uint32_t bin = 0U; bin < 10U; ++bin)
            {
                const float center = recovered_ooc_histogram_center(bin);
                const std::uint8_t vote = histogram[10U * voxel + bin];
                if (vote == 255U)
                    hard_value = center;
                total = __fadd_rn(total, __uint2float_rn(vote));
            }
            const float normalized_weight = __fdiv_rn(__uint2float_rn(weights[voxel]), __uint_as_float(0x437fe666U));
            float local_weight = __fmul_rn(__fmaf_rn(normalized_weight, 0.75F, 0.75F), data_weight);
            float histogram_scale = 1.0F;
            if (total != 0.0F)
            {
                histogram_scale = __fdiv_rn(5.0F, total);
                total = __fmul_rn(total, histogram_scale);
            }
            local_weight = __fmul_rn(local_weight, inverse_sqrt_12);
            // The target's unqualified PTX mul/sub pair contracts to one FFMA on the
            // production sm_89 JIT path; retaining that contraction is observable.
            float histogram_offset = __fmaf_rn(local_weight, -total, -u_bar);
            float result = __uint_as_float(0x45157338U);
            bool prox_resolved = false;
            const float doubled_weight = __fadd_rn(local_weight, local_weight);
            const float update_scale = __fmul_rn(histogram_scale, doubled_weight);
            for (std::uint32_t bin = 0U; bin < 10U; ++bin)
            {
                const float center = recovered_ooc_histogram_center(bin);
                const float left = __fadd_rn(histogram_offset, recovered_ooc_histogram_left(bin));
                if (left > 0.0F)
                {
                    result = -histogram_offset;
                    prox_resolved = true;
                    break;
                }
                histogram_offset =
                    __fmaf_rn(update_scale, __uint2float_rn(histogram[10U * voxel + bin]), histogram_offset);
                const float right = __fadd_rn(histogram_offset, recovered_ooc_histogram_right(bin));
                if (right > 0.0F)
                {
                    const float span = __fsub_rn(right, left);
                    const float ratio = __fdiv_rn(__fmul_rn(right, -2.0F), span);
                    const float interpolation = __fadd_rn(ratio, 1.0F);
                    result = __fmaf_rn(interpolation, __uint_as_float(0x3de38e39U), center);
                    prox_resolved = true;
                    break;
                }
            }
            if (!prox_resolved)
            {
                const float left = __fadd_rn(histogram_offset, __uint_as_float(0x45157171U));
                if (left > 0.0F)
                {
                    result = -histogram_offset;
                }
                else
                {
                    const float right = __fadd_rn(histogram_offset, __uint_as_float(0x451574ffU));
                    if (right > 0.0F)
                    {
                        const float span = __fsub_rn(right, left);
                        const float ratio = __fdiv_rn(__fmul_rn(right, -2.0F), span);
                        const float interpolation = __fadd_rn(ratio, 1.0F);
                        result = __fmaf_rn(interpolation, __uint_as_float(0x3de38e39U), __uint_as_float(0x45157338U));
                    }
                }
            }
            if (hard_value != 256.0F)
            {
                result = hard_value;
                u_old[voxel] = hard_value;
            }
            u[voxel] = result;

            for (std::uint32_t component = 0U; component < 3U; ++component)
            {
                const std::uint32_t vector_index = 3U * voxel + component;
                v_old[vector_index] = v[vector_index];
                float divergence_q = 0.0F;
                for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                {
                    const std::uint32_t negative = 2U * axis;
                    const float value = recovered_ooc_half(q, (3U * voxel + component) * 3U + axis);
                    if (recovered_ooc_bit(connectivity[voxel], negative + 1U))
                        divergence_q = __fsub_rn(divergence_q, value);
                    if (!recovered_ooc_bit(connectivity[voxel], negative))
                        continue;
                    const std::uint32_t adjacent = neighbors[6U * voxel + negative];
                    if (recovered_ooc_bit(refinement[voxel], negative))
                    {
                        for (std::uint32_t child = 0U; child < 4U; ++child)
                        {
                            const std::uint32_t index = adjacent + recovered_ooc_child_offset(axis, child);
                            divergence_q = __fmaf_rn(
                                recovered_ooc_half(q, (3U * index + component) * 3U + axis), 0.25F, divergence_q);
                        }
                    }
                    else
                    {
                        divergence_q =
                            __fadd_rn(divergence_q, recovered_ooc_half(q, (3U * adjacent + component) * 3U + axis));
                    }
                }
                const float delta = __fsub_rn(__fmul_rn(local_p[component], alpha), __fmul_rn(divergence_q, beta));
                float candidate = __fmaf_rn(delta, inverse_sqrt_12, recovered_ooc_half(v, vector_index));
                candidate = candidate < -2.0F ? -2.0F : candidate;
                candidate = candidate > 2.0F ? 2.0F : candidate;
                recovered_ooc_store_half(v, vector_index, candidate);
            }
        }

        // Production functional-mode-1 reconstruction of cuda::ooc_update_p.
        __global__ __launch_bounds__(128, 1) void recovered_ooc_update_p_kernel(const std::uint32_t* neighbors,
                                                                                const std::uint8_t* connectivity,
                                                                                const std::uint8_t* refinement,
                                                                                const std::uint8_t* flags,
                                                                                const float* u,
                                                                                const float* u_old,
                                                                                std::uint16_t* p,
                                                                                const std::uint16_t* v,
                                                                                const std::uint16_t* v_old,
                                                                                std::uint16_t* q,
                                                                                std::uint32_t count,
                                                                                float alpha,
                                                                                float beta,
                                                                                std::uint32_t offset)
        {
            constexpr float inverse_sqrt_12 = 0.28867528F;
            const std::uint32_t voxel = offset + blockIdx.x * blockDim.x + threadIdx.x;
            if (voxel >= count || (flags[voxel] & 4U) != 0U)
                return;
            const float center_u_bar = __fadd_rn(__fmaf_rn(u[voxel], -2.0F, 0.0F), u_old[voxel]);
            float center_v_bar[3];
            float p_candidate[3];
            float q_candidate[3][3];
            for (std::uint32_t component = 0U; component < 3U; ++component)
            {
                const float current = recovered_ooc_half(v, 3U * voxel + component);
                center_v_bar[component] =
                    __fsub_rn(__fmaf_rn(current, 2.0F, 0.0F), recovered_ooc_half(v_old, 3U * voxel + component));
            }
            for (std::uint32_t component = 0U; component < 3U; ++component)
            {
                const std::uint32_t positive = 2U * component + 1U;
                float gradient_u = 0.0F;
                if (recovered_ooc_bit(connectivity[voxel], positive))
                {
                    const std::uint32_t adjacent = neighbors[6U * voxel + positive];
                    float neighbor_u_bar = 0.0F;
                    if (recovered_ooc_bit(refinement[voxel], positive))
                    {
                        for (std::uint32_t child = 0U; child < 4U; ++child)
                        {
                            const std::uint32_t index = adjacent + recovered_ooc_child_offset(component, child);
                            neighbor_u_bar = __fmaf_rn(u[index], 2.0F, neighbor_u_bar);
                            neighbor_u_bar = __fsub_rn(neighbor_u_bar, u_old[index]);
                        }
                        neighbor_u_bar = __fmul_rn(neighbor_u_bar, 0.25F);
                    }
                    else
                    {
                        neighbor_u_bar = __fsub_rn(__fmaf_rn(u[adjacent], 2.0F, 0.0F), u_old[adjacent]);
                    }
                    gradient_u = __fadd_rn(neighbor_u_bar, center_u_bar);
                }
                p_candidate[component] = __fmaf_rn(__fmul_rn(alpha, inverse_sqrt_12),
                                                   __fsub_rn(gradient_u, center_v_bar[component]),
                                                   recovered_ooc_half(p, 3U * voxel + component));
                for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                {
                    const std::uint32_t positive_axis = 2U * axis + 1U;
                    float gradient_v = 0.0F;
                    if (recovered_ooc_bit(connectivity[voxel], positive_axis))
                    {
                        const std::uint32_t adjacent = neighbors[6U * voxel + positive_axis];
                        float neighbor_v_bar = 0.0F;
                        if (recovered_ooc_bit(refinement[voxel], positive_axis))
                        {
                            for (std::uint32_t child = 0U; child < 4U; ++child)
                            {
                                const std::uint32_t index = adjacent + recovered_ooc_child_offset(axis, child);
                                const float current = recovered_ooc_half(v, 3U * index + component);
                                neighbor_v_bar = __fmaf_rn(current, 2.0F, neighbor_v_bar);
                                neighbor_v_bar =
                                    __fsub_rn(neighbor_v_bar, recovered_ooc_half(v_old, 3U * index + component));
                            }
                            neighbor_v_bar = __fmul_rn(neighbor_v_bar, 0.25F);
                        }
                        else
                        {
                            const float current = recovered_ooc_half(v, 3U * adjacent + component);
                            neighbor_v_bar = __fsub_rn(__fmaf_rn(current, 2.0F, 0.0F),
                                                       recovered_ooc_half(v_old, 3U * adjacent + component));
                        }
                        gradient_v = __fsub_rn(neighbor_v_bar, center_v_bar[component]);
                    }
                    q_candidate[component][axis] =
                        __fmaf_rn(__fmul_rn(beta, inverse_sqrt_12),
                                  gradient_v,
                                  recovered_ooc_half(q, (3U * voxel + component) * 3U + axis));
                }
            }
            const float p_norm_sq =
                __fmaf_rn(p_candidate[2],
                          p_candidate[2],
                          __fmaf_rn(p_candidate[0], p_candidate[0], __fmul_rn(p_candidate[1], p_candidate[1])));
            float p_divisor = 1.0F;
            if (p_norm_sq > __fmul_rn(alpha, alpha))
                p_divisor = __fdiv_rn(sqrtf(p_norm_sq), alpha);
            for (std::uint32_t component = 0U; component < 3U; ++component)
            {
                recovered_ooc_store_half(p, 3U * voxel + component, __fdiv_rn(p_candidate[component], p_divisor));
            }
            float q_norm_sq = 0.0F;
            for (std::uint32_t row = 0U; row < 3U; ++row)
            {
                q_norm_sq = __fmaf_rn(q_candidate[row][0], q_candidate[row][0], q_norm_sq);
                q_norm_sq = __fmaf_rn(q_candidate[row][1], q_candidate[row][1], q_norm_sq);
                q_norm_sq = __fmaf_rn(q_candidate[row][2], q_candidate[row][2], q_norm_sq);
            }
            float q_divisor = 1.0F;
            if (q_norm_sq > __fmul_rn(beta, beta))
                q_divisor = __fdiv_rn(sqrtf(q_norm_sq), beta);
            for (std::uint32_t row = 0U; row < 3U; ++row)
            {
                for (std::uint32_t column = 0U; column < 3U; ++column)
                {
                    recovered_ooc_store_half(
                        q, (3U * voxel + row) * 3U + column, __fdiv_rn(q_candidate[row][column], q_divisor));
                }
            }
        }

    } // namespace

    cudaError_t launch_recovered_patchmatch_undistort_u8_source(const std::uint8_t* source,
                                                                const std::uint8_t* source_mask,
                                                                std::uint8_t* result,
                                                                std::uint8_t* result_mask,
                                                                const DepthVotingCalibrationCu& source_calibration,
                                                                const DepthVotingCalibrationCu& target_calibration,
                                                                std::uint32_t width,
                                                                std::uint32_t height,
                                                                bool with_mask,
                                                                std::uint32_t pixel_offset,
                                                                std::size_t work_items,
                                                                cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_undistort_kernel<<<blocks, threads, 0U, stream>>>(source,
                                                                     source_mask,
                                                                     result,
                                                                     result_mask,
                                                                     source_calibration,
                                                                     target_calibration,
                                                                     width,
                                                                     height,
                                                                     1U,
                                                                     with_mask ? 1U : 0U,
                                                                     pixel_offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_undistort_source(const void* source,
                                                             const std::uint8_t* source_mask,
                                                             void* result,
                                                             std::uint8_t* result_mask,
                                                             std::uint32_t sample,
                                                             std::uint32_t channels,
                                                             const DepthVotingCalibrationCu& source_calibration,
                                                             const DepthVotingCalibrationCu& target_calibration,
                                                             std::uint32_t width,
                                                             std::uint32_t height,
                                                             bool with_mask,
                                                             std::uint32_t pixel_offset,
                                                             std::size_t work_items,
                                                             cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
#define METMODEL_LAUNCH_UNDISTORT(Source, Result)                                                                      \
    patchmatch_undistort_kernel<Source, Result><<<blocks, threads, 0U, stream>>>(static_cast<const Source*>(source),   \
                                                                                 source_mask,                          \
                                                                                 static_cast<Result*>(result),         \
                                                                                 result_mask,                          \
                                                                                 source_calibration,                   \
                                                                                 target_calibration,                   \
                                                                                 width,                                \
                                                                                 height,                               \
                                                                                 channels,                             \
                                                                                 with_mask ? 1U : 0U,                  \
                                                                                 pixel_offset)
        switch (sample)
        {
        case 0U:
            METMODEL_LAUNCH_UNDISTORT(std::uint8_t, std::uint8_t);
            break;
        case 1U:
            METMODEL_LAUNCH_UNDISTORT(std::uint16_t, std::uint16_t);
            break;
        case 2U:
            METMODEL_LAUNCH_UNDISTORT(std::uint32_t, float);
            break;
        case 3U:
            METMODEL_LAUNCH_UNDISTORT(float, float);
            break;
        default:
            return cudaErrorInvalidValue;
        }
#undef METMODEL_LAUNCH_UNDISTORT
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_ooc_histogram_mode0_source(OocHistogramVoxel* voxels,
                                                            const float* pyramid,
                                                            std::uint32_t pyramid_levels,
                                                            std::uint32_t mode_b,
                                                            const OocHistogramCalibrationCu& calibration,
                                                            const OocHistogramCameraExteriorTransformCu& exterior,
                                                            float threshold_a,
                                                            float threshold_b,
                                                            std::uint32_t begin,
                                                            std::uint32_t end,
                                                            cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = (end - begin + threads - 1U) / threads;
        ooc_histogram_mode0_kernel<<<blocks, threads, 0U, stream>>>(
            voxels, pyramid, pyramid_levels, mode_b, calibration, exterior, threshold_a, threshold_b, begin, end);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_depth_radius_estimate_source(const float* depth,
                                                              float* radius,
                                                              std::uint32_t level_offset,
                                                              const DepthVotingCalibrationCu& calibration,
                                                              const DepthVotingMatrix4x4f& transform,
                                                              std::uint32_t kernel_offset,
                                                              std::size_t work_items,
                                                              cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        depth_radius_estimate_type1_kernel<<<blocks, threads, 0U, stream>>>(
            depth, radius, level_offset, calibration, transform, kernel_offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_depth_neighbor_votes_source(std::int32_t* votes,
                                                             const float* reference_depth,
                                                             const float* reference_radius,
                                                             const std::uint8_t* neighbor_inlier_mask,
                                                             const float* neighbor_depth_levels,
                                                             const float* neighbor_radius_levels,
                                                             std::uint32_t reference_level,
                                                             const DepthVotingCalibrationCu& reference_calibration,
                                                             const DepthVotingMatrix4x4f& reference_to_neighbor,
                                                             std::uint32_t neighbor_levels,
                                                             const DepthVotingCalibrationCu& neighbor_calibration,
                                                             std::uint32_t* counter_inlier_supports,
                                                             std::uint32_t* counter_inlier_intersects,
                                                             std::uint32_t* counter_inlier_does_not_reach,
                                                             std::uint32_t* counter_inlier_no_depth,
                                                             std::uint32_t* counter_outlier_supports,
                                                             std::uint32_t* counter_outlier_intersects,
                                                             std::uint32_t* counter_outlier_does_not_reach,
                                                             std::uint32_t kernel_offset,
                                                             std::size_t work_items,
                                                             cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        depth_neighbor_votes_type1_kernel<<<blocks, threads, 0U, stream>>>(votes,
                                                                           reference_depth,
                                                                           reference_radius,
                                                                           neighbor_inlier_mask,
                                                                           neighbor_depth_levels,
                                                                           neighbor_radius_levels,
                                                                           reference_level,
                                                                           reference_calibration,
                                                                           reference_to_neighbor,
                                                                           neighbor_levels,
                                                                           neighbor_calibration,
                                                                           counter_inlier_supports,
                                                                           counter_inlier_intersects,
                                                                           counter_inlier_does_not_reach,
                                                                           counter_inlier_no_depth,
                                                                           counter_outlier_supports,
                                                                           counter_outlier_intersects,
                                                                           counter_outlier_does_not_reach,
                                                                           kernel_offset);
        return cudaGetLastError();
    }

    cudaError_t
    launch_recovered_depth_neighbor_occlusion_votes_source(std::int32_t* votes,
                                                           const float* reference_depth,
                                                           const float* reference_radius,
                                                           const std::uint8_t* neighbor_inlier_mask,
                                                           std::uint32_t neighbor_level,
                                                           const float* neighbor_depth_levels,
                                                           const float* neighbor_radius_levels,
                                                           std::uint32_t reference_level,
                                                           const DepthVotingCalibrationCu& reference_calibration,
                                                           const DepthVotingMatrix4x4f& neighbor_to_reference,
                                                           const DepthVotingCalibrationCu& neighbor_calibration,
                                                           std::uint32_t* counter_inlier_occludes,
                                                           std::uint32_t* counter_outlier_occludes,
                                                           std::uint32_t kernel_offset,
                                                           std::size_t work_items,
                                                           cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        depth_neighbor_occlusion_votes_type1_kernel<<<blocks, threads, 0U, stream>>>(votes,
                                                                                     reference_depth,
                                                                                     reference_radius,
                                                                                     neighbor_inlier_mask,
                                                                                     neighbor_level,
                                                                                     neighbor_depth_levels,
                                                                                     neighbor_radius_levels,
                                                                                     reference_level,
                                                                                     reference_calibration,
                                                                                     neighbor_to_reference,
                                                                                     neighbor_calibration,
                                                                                     counter_inlier_occludes,
                                                                                     counter_outlier_occludes,
                                                                                     kernel_offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_depth_voting_finalize_source(float* depth,
                                                              const std::int32_t* votes,
                                                              std::uint32_t width,
                                                              std::uint32_t height,
                                                              std::uint32_t* counter_empty,
                                                              std::uint32_t* counter_bad,
                                                              std::uint32_t* counter_normal,
                                                              std::uint32_t* counter_good,
                                                              std::uint32_t pixel_offset,
                                                              std::size_t work_items,
                                                              cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        depth_voting_finalize_kernel<<<blocks, threads, 0U, stream>>>(
            depth, votes, width, height, counter_empty, counter_bad, counter_normal, counter_good, pixel_offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_copy_inlier_masks_source(std::uint8_t* neighbor_inlier_masks,
                                                                     std::uint32_t width,
                                                                     std::uint32_t height,
                                                                     std::uint32_t hypotheses_per_pixel,
                                                                     std::uint32_t neighbor_count,
                                                                     const std::uint8_t* temporary_inlier_masks,
                                                                     const std::uint8_t* winner,
                                                                     std::uint32_t is_checkboard,
                                                                     std::uint32_t checkboard_step,
                                                                     std::uint32_t only_each_fourth_pixel,
                                                                     std::uint32_t pixel_offset,
                                                                     std::size_t work_items,
                                                                     cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_copy_inlier_masks_kernel<<<blocks, threads, 0U, stream>>>(neighbor_inlier_masks,
                                                                             width,
                                                                             height,
                                                                             hypotheses_per_pixel,
                                                                             neighbor_count,
                                                                             temporary_inlier_masks,
                                                                             winner,
                                                                             is_checkboard,
                                                                             checkboard_step,
                                                                             only_each_fourth_pixel,
                                                                             pixel_offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_coarse_to_precise_source(const float* depth,
                                                                     const std::uint8_t* normal,
                                                                     const float* cost,
                                                                     std::uint32_t width_original,
                                                                     std::uint32_t height_original,
                                                                     std::uint32_t depth_downscale,
                                                                     float* candidate_depth,
                                                                     float* candidate_normal,
                                                                     std::uint32_t pixel_offset,
                                                                     std::size_t work_items,
                                                                     cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_coarse_to_precise_kernel<<<blocks, threads, 0U, stream>>>(depth,
                                                                             normal,
                                                                             cost,
                                                                             width_original,
                                                                             height_original,
                                                                             depth_downscale,
                                                                             candidate_depth,
                                                                             candidate_normal,
                                                                             pixel_offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_filter_clear_depth_source(float* depth,
                                                                      const std::uint8_t* filtered_mask,
                                                                      std::uint32_t* counter_not_empty,
                                                                      std::uint32_t pixel_offset,
                                                                      std::size_t work_items,
                                                                      cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_filter_clear_depth_kernel<<<blocks, threads, 0U, stream>>>(
            depth, filtered_mask, counter_not_empty, pixel_offset, work_items);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_rotate_normals_source(float* normals,
                                                                  const PatchMatchCamera& camera_transform,
                                                                  std::size_t work_items,
                                                                  cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_rotate_normals_kernel<<<blocks, threads, 0U, stream>>>(normals, camera_transform);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_average_costs_source(const float* per_neighbor_cost,
                                                                 float* average_cost,
                                                                 std::uint8_t* auxiliary,
                                                                 std::uint32_t width_original,
                                                                 std::uint32_t height_original,
                                                                 std::uint32_t depth_downscale,
                                                                 std::uint32_t hypotheses_per_pixel,
                                                                 std::uint32_t neighbor_count,
                                                                 std::uint32_t is_checkboard,
                                                                 std::uint32_t checkboard_step,
                                                                 std::uint32_t only_each_fourth_pixel,
                                                                 std::uint32_t pixel_offset,
                                                                 std::size_t work_items,
                                                                 cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const std::size_t thread_count = work_items * static_cast<std::size_t>(hypotheses_per_pixel);
        const unsigned int blocks = static_cast<unsigned int>((thread_count + threads - 1U) / threads);
        patchmatch_average_costs_kernel<<<blocks, threads, 0U, stream>>>(per_neighbor_cost,
                                                                         average_cost,
                                                                         auxiliary,
                                                                         width_original,
                                                                         height_original,
                                                                         depth_downscale,
                                                                         hypotheses_per_pixel,
                                                                         neighbor_count,
                                                                         is_checkboard,
                                                                         checkboard_step,
                                                                         only_each_fourth_pixel,
                                                                         pixel_offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_wta_source(float* depth,
                                                       std::uint8_t* normal,
                                                       float* cost,
                                                       std::uint32_t width_original,
                                                       std::uint32_t height_original,
                                                       std::uint32_t depth_downscale,
                                                       std::uint32_t hypotheses_per_pixel,
                                                       const float* candidate_depth,
                                                       const float* candidate_normal,
                                                       const float* average_cost,
                                                       std::uint8_t* winner,
                                                       std::uint32_t is_checkboard,
                                                       std::uint32_t checkboard_step,
                                                       std::uint32_t only_each_fourth_pixel,
                                                       std::uint32_t pixel_offset,
                                                       std::size_t work_items,
                                                       cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_wta_kernel<<<blocks, threads, 0U, stream>>>(depth,
                                                               normal,
                                                               cost,
                                                               width_original,
                                                               height_original,
                                                               depth_downscale,
                                                               hypotheses_per_pixel,
                                                               candidate_depth,
                                                               candidate_normal,
                                                               average_cost,
                                                               winner,
                                                               is_checkboard,
                                                               checkboard_step,
                                                               only_each_fourth_pixel,
                                                               pixel_offset);
        return cudaGetLastError();
    }

    template <class ReferenceSample, std::uint32_t CameraType>
    cudaError_t launch_recovered_patchmatch_final_refinement_typed_source(float* depth,
                                                                          std::uint8_t* normal,
                                                                          float* cost,
                                                                          const PatchMatchCamera& camera,
                                                                          std::uint32_t depth_downscale,
                                                                          const ReferenceSample* reference_image,
                                                                          std::uint32_t image_one_step_more_detailed,
                                                                          float deviation_threshold_multiplier,
                                                                          float* candidate_depth,
                                                                          float* candidate_normal,
                                                                          std::uint32_t pixel_offset,
                                                                          std::size_t work_items,
                                                                          cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_final_refinement_extended_kernel<ReferenceSample, CameraType>
            <<<blocks, threads, 0U, stream>>>(depth,
                                              normal,
                                              cost,
                                              camera,
                                              depth_downscale,
                                              reference_image,
                                              image_one_step_more_detailed,
                                              deviation_threshold_multiplier,
                                              candidate_depth,
                                              candidate_normal,
                                              pixel_offset);
        return cudaGetLastError();
    }

    template <class ReferenceSample>
    cudaError_t dispatch_recovered_patchmatch_final_refinement_camera_source(float* depth,
                                                                             std::uint8_t* normal,
                                                                             float* cost,
                                                                             const PatchMatchCamera& camera,
                                                                             std::uint32_t depth_downscale,
                                                                             const ReferenceSample* reference_image,
                                                                             std::uint32_t image_one_step_more_detailed,
                                                                             float deviation_threshold_multiplier,
                                                                             float* candidate_depth,
                                                                             float* candidate_normal,
                                                                             std::uint32_t pixel_offset,
                                                                             std::size_t work_items,
                                                                             cudaStream_t stream)
    {
#define METMODEL_FINAL_CAMERA_CASE(value)                                                                              \
    case value:                                                                                                        \
        return launch_recovered_patchmatch_final_refinement_typed_source<ReferenceSample, value>(                      \
            depth,                                                                                                     \
            normal,                                                                                                    \
            cost,                                                                                                      \
            camera,                                                                                                    \
            depth_downscale,                                                                                           \
            reference_image,                                                                                           \
            image_one_step_more_detailed,                                                                              \
            deviation_threshold_multiplier,                                                                            \
            candidate_depth,                                                                                           \
            candidate_normal,                                                                                          \
            pixel_offset,                                                                                              \
            work_items,                                                                                                \
            stream)
        switch (camera.type)
        {
            METMODEL_FINAL_CAMERA_CASE(0U);
            METMODEL_FINAL_CAMERA_CASE(2U);
            METMODEL_FINAL_CAMERA_CASE(3U);
            METMODEL_FINAL_CAMERA_CASE(4U);
            METMODEL_FINAL_CAMERA_CASE(8U);
            METMODEL_FINAL_CAMERA_CASE(9U);
            METMODEL_FINAL_CAMERA_CASE(10U);
        default:
            return cudaErrorInvalidValue;
        }
#undef METMODEL_FINAL_CAMERA_CASE
    }

    cudaError_t launch_recovered_patchmatch_final_refinement_source(float* depth,
                                                                    std::uint8_t* normal,
                                                                    float* cost,
                                                                    const PatchMatchCamera& camera,
                                                                    std::uint32_t depth_downscale,
                                                                    const void* reference_image,
                                                                    std::uint32_t reference_image_sample_bytes,
                                                                    std::uint32_t image_one_step_more_detailed,
                                                                    float deviation_threshold_multiplier,
                                                                    float* candidate_depth,
                                                                    float* candidate_normal,
                                                                    std::uint32_t pixel_offset,
                                                                    std::size_t work_items,
                                                                    cudaStream_t stream)
    {
        if (depth_downscale == 0U || (image_one_step_more_detailed != 0U && depth_downscale < 2U))
            return cudaErrorInvalidValue;
        if (reference_image_sample_bytes == 1U)
            return dispatch_recovered_patchmatch_final_refinement_camera_source(
                depth,
                normal,
                cost,
                camera,
                depth_downscale,
                static_cast<const std::uint8_t*>(reference_image),
                image_one_step_more_detailed,
                deviation_threshold_multiplier,
                candidate_depth,
                candidate_normal,
                pixel_offset,
                work_items,
                stream);
        if (reference_image_sample_bytes == 2U)
            return dispatch_recovered_patchmatch_final_refinement_camera_source(
                depth,
                normal,
                cost,
                camera,
                depth_downscale,
                static_cast<const std::uint16_t*>(reference_image),
                image_one_step_more_detailed,
                deviation_threshold_multiplier,
                candidate_depth,
                candidate_normal,
                pixel_offset,
                work_items,
                stream);
        if (reference_image_sample_bytes == 4U)
            return dispatch_recovered_patchmatch_final_refinement_camera_source(
                depth,
                normal,
                cost,
                camera,
                depth_downscale,
                static_cast<const float*>(reference_image),
                image_one_step_more_detailed,
                deviation_threshold_multiplier,
                candidate_depth,
                candidate_normal,
                pixel_offset,
                work_items,
                stream);
        return cudaErrorInvalidValue;
    }

    template <class ReferenceSample>
    cudaError_t launch_recovered_patchmatch_refinement_typed_source(const float* depth,
                                                                    const std::uint8_t* normal,
                                                                    const float* cost,
                                                                    const float* coarse_depth,
                                                                    const float* coarse_radius,
                                                                    const PatchMatchCamera& camera,
                                                                    std::uint32_t depth_downscale,
                                                                    float depth_min,
                                                                    float depth_max,
                                                                    const ReferenceSample* reference_image,
                                                                    std::uint32_t image_one_step_more_detailed,
                                                                    float deviation_threshold_multiplier,
                                                                    float* candidate_depth,
                                                                    float* candidate_normal,
                                                                    std::uint32_t iteration,
                                                                    std::uint32_t only_each_fourth_pixel,
                                                                    std::uint32_t pixel_offset,
                                                                    std::size_t work_items,
                                                                    cudaStream_t stream)
    {
        if (camera.type != 0U || depth_downscale == 0U || (image_one_step_more_detailed != 0U && depth_downscale < 2U))
            return cudaErrorInvalidValue;
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_refinement_type0_kernel<ReferenceSample>
            <<<blocks, threads, 0U, stream>>>(depth,
                                              normal,
                                              cost,
                                              coarse_depth,
                                              coarse_radius,
                                              camera,
                                              depth_downscale,
                                              depth_min,
                                              depth_max,
                                              reference_image,
                                              image_one_step_more_detailed,
                                              deviation_threshold_multiplier,
                                              candidate_depth,
                                              candidate_normal,
                                              iteration,
                                              only_each_fourth_pixel,
                                              pixel_offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_refinement_source(const float* depth,
                                                              const std::uint8_t* normal,
                                                              const float* cost,
                                                              const float* coarse_depth,
                                                              const float* coarse_radius,
                                                              const PatchMatchCamera& camera,
                                                              std::uint32_t depth_downscale,
                                                              float depth_min,
                                                              float depth_max,
                                                              const void* reference_image,
                                                              std::uint32_t reference_image_sample_bytes,
                                                              std::uint32_t image_one_step_more_detailed,
                                                              float deviation_threshold_multiplier,
                                                              float* candidate_depth,
                                                              float* candidate_normal,
                                                              std::uint32_t iteration,
                                                              std::uint32_t only_each_fourth_pixel,
                                                              std::uint32_t pixel_offset,
                                                              std::size_t work_items,
                                                              cudaStream_t stream)
    {
        if (reference_image_sample_bytes == 1U)
            return launch_recovered_patchmatch_refinement_typed_source(
                depth,
                normal,
                cost,
                coarse_depth,
                coarse_radius,
                camera,
                depth_downscale,
                depth_min,
                depth_max,
                static_cast<const std::uint8_t*>(reference_image),
                image_one_step_more_detailed,
                deviation_threshold_multiplier,
                candidate_depth,
                candidate_normal,
                iteration,
                only_each_fourth_pixel,
                pixel_offset,
                work_items,
                stream);
        if (reference_image_sample_bytes == 2U)
            return launch_recovered_patchmatch_refinement_typed_source(
                depth,
                normal,
                cost,
                coarse_depth,
                coarse_radius,
                camera,
                depth_downscale,
                depth_min,
                depth_max,
                static_cast<const std::uint16_t*>(reference_image),
                image_one_step_more_detailed,
                deviation_threshold_multiplier,
                candidate_depth,
                candidate_normal,
                iteration,
                only_each_fourth_pixel,
                pixel_offset,
                work_items,
                stream);
        if (reference_image_sample_bytes == 4U)
            return launch_recovered_patchmatch_refinement_typed_source(depth,
                                                                       normal,
                                                                       cost,
                                                                       coarse_depth,
                                                                       coarse_radius,
                                                                       camera,
                                                                       depth_downscale,
                                                                       depth_min,
                                                                       depth_max,
                                                                       static_cast<const float*>(reference_image),
                                                                       image_one_step_more_detailed,
                                                                       deviation_threshold_multiplier,
                                                                       candidate_depth,
                                                                       candidate_normal,
                                                                       iteration,
                                                                       only_each_fourth_pixel,
                                                                       pixel_offset,
                                                                       work_items,
                                                                       stream);
        return cudaErrorInvalidValue;
    }

    cudaError_t
    launch_recovered_patchmatch_propagation_u8_perspective_source(float* depth,
                                                                  std::uint8_t* normal,
                                                                  float* cost,
                                                                  const float* coarse_depth,
                                                                  const float* coarse_radius,
                                                                  const PatchMatchCamera& camera,
                                                                  const float* rotation_to_local,
                                                                  std::uint32_t depth_downscale,
                                                                  const std::uint8_t* reference_image,
                                                                  std::uint32_t image_one_step_more_detailed,
                                                                  float deviation_threshold_multiplier,
                                                                  float* candidate_depth,
                                                                  float* candidate_normal,
                                                                  std::uint32_t checkerboard_step,
                                                                  std::uint32_t only_each_fourth_pixel,
                                                                  std::uint32_t pixel_offset,
                                                                  std::size_t work_items,
                                                                  cudaStream_t stream)
    {
        if ((camera.type != 0U && camera.type != 2U && camera.type != 3U && camera.type != 4U && camera.type != 8U &&
             camera.type != 9U && camera.type != 10U) ||
            depth_downscale == 0U || (image_one_step_more_detailed != 0U && depth_downscale < 2U))
            return cudaErrorInvalidValue;
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        if (camera.type == 0U)
        {
            patchmatch_propagation_u8_perspective_kernel<<<blocks, threads, 0U, stream>>>(
                depth,
                normal,
                cost,
                coarse_depth,
                coarse_radius,
                camera,
                rotation_to_local,
                depth_downscale,
                reference_image,
                image_one_step_more_detailed,
                deviation_threshold_multiplier,
                candidate_depth,
                candidate_normal,
                checkerboard_step,
                only_each_fourth_pixel,
                pixel_offset);
        }
        else if (camera.type == 2U)
        {
            patchmatch_propagation_extended_kernel<std::uint8_t, 2U>
                <<<blocks, threads, 0U, stream>>>(depth,
                                                  normal,
                                                  cost,
                                                  coarse_depth,
                                                  coarse_radius,
                                                  camera,
                                                  rotation_to_local,
                                                  depth_downscale,
                                                  reference_image,
                                                  image_one_step_more_detailed,
                                                  deviation_threshold_multiplier,
                                                  candidate_depth,
                                                  candidate_normal,
                                                  checkerboard_step,
                                                  only_each_fourth_pixel,
                                                  pixel_offset);
        }
        else if (camera.type == 3U)
        {
            patchmatch_propagation_extended_kernel<std::uint8_t, 3U>
                <<<blocks, threads, 0U, stream>>>(depth,
                                                  normal,
                                                  cost,
                                                  coarse_depth,
                                                  coarse_radius,
                                                  camera,
                                                  rotation_to_local,
                                                  depth_downscale,
                                                  reference_image,
                                                  image_one_step_more_detailed,
                                                  deviation_threshold_multiplier,
                                                  candidate_depth,
                                                  candidate_normal,
                                                  checkerboard_step,
                                                  only_each_fourth_pixel,
                                                  pixel_offset);
        }
        else if (camera.type == 4U)
        {
            patchmatch_propagation_extended_kernel<std::uint8_t, 4U>
                <<<blocks, threads, 0U, stream>>>(depth,
                                                  normal,
                                                  cost,
                                                  coarse_depth,
                                                  coarse_radius,
                                                  camera,
                                                  rotation_to_local,
                                                  depth_downscale,
                                                  reference_image,
                                                  image_one_step_more_detailed,
                                                  deviation_threshold_multiplier,
                                                  candidate_depth,
                                                  candidate_normal,
                                                  checkerboard_step,
                                                  only_each_fourth_pixel,
                                                  pixel_offset);
        }
        else if (camera.type == 8U)
        {
            patchmatch_propagation_extended_kernel<std::uint8_t, 8U>
                <<<blocks, threads, 0U, stream>>>(depth,
                                                  normal,
                                                  cost,
                                                  coarse_depth,
                                                  coarse_radius,
                                                  camera,
                                                  rotation_to_local,
                                                  depth_downscale,
                                                  reference_image,
                                                  image_one_step_more_detailed,
                                                  deviation_threshold_multiplier,
                                                  candidate_depth,
                                                  candidate_normal,
                                                  checkerboard_step,
                                                  only_each_fourth_pixel,
                                                  pixel_offset);
        }
        else if (camera.type == 9U)
        {
            patchmatch_propagation_extended_kernel<std::uint8_t, 9U>
                <<<blocks, threads, 0U, stream>>>(depth,
                                                  normal,
                                                  cost,
                                                  coarse_depth,
                                                  coarse_radius,
                                                  camera,
                                                  rotation_to_local,
                                                  depth_downscale,
                                                  reference_image,
                                                  image_one_step_more_detailed,
                                                  deviation_threshold_multiplier,
                                                  candidate_depth,
                                                  candidate_normal,
                                                  checkerboard_step,
                                                  only_each_fourth_pixel,
                                                  pixel_offset);
        }
        else
        {
            patchmatch_propagation_extended_kernel<std::uint8_t, 10U>
                <<<blocks, threads, 0U, stream>>>(depth,
                                                  normal,
                                                  cost,
                                                  coarse_depth,
                                                  coarse_radius,
                                                  camera,
                                                  rotation_to_local,
                                                  depth_downscale,
                                                  reference_image,
                                                  image_one_step_more_detailed,
                                                  deviation_threshold_multiplier,
                                                  candidate_depth,
                                                  candidate_normal,
                                                  checkerboard_step,
                                                  only_each_fourth_pixel,
                                                  pixel_offset);
        }
        return cudaGetLastError();
    }

    cudaError_t
    launch_recovered_patchmatch_propagation_u16_perspective_source(float* depth,
                                                                   std::uint8_t* normal,
                                                                   float* cost,
                                                                   const float* coarse_depth,
                                                                   const float* coarse_radius,
                                                                   const PatchMatchCamera& camera,
                                                                   const float* rotation_to_local,
                                                                   std::uint32_t depth_downscale,
                                                                   const std::uint16_t* reference_image,
                                                                   std::uint32_t image_one_step_more_detailed,
                                                                   float deviation_threshold_multiplier,
                                                                   float* candidate_depth,
                                                                   float* candidate_normal,
                                                                   std::uint32_t checkerboard_step,
                                                                   std::uint32_t only_each_fourth_pixel,
                                                                   std::uint32_t pixel_offset,
                                                                   std::size_t work_items,
                                                                   cudaStream_t stream)
    {
        if ((camera.type != 0U && camera.type != 2U && camera.type != 3U && camera.type != 4U && camera.type != 8U &&
             camera.type != 9U && camera.type != 10U) ||
            depth_downscale == 0U || (image_one_step_more_detailed != 0U && depth_downscale < 2U))
            return cudaErrorInvalidValue;
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
#define METMODEL_LAUNCH_PROPAGATION_U16(camera_type)                                                                   \
    patchmatch_propagation_extended_kernel<std::uint16_t, camera_type>                                                 \
        <<<blocks, threads, 0U, stream>>>(depth,                                                                       \
                                          normal,                                                                      \
                                          cost,                                                                        \
                                          coarse_depth,                                                                \
                                          coarse_radius,                                                               \
                                          camera,                                                                      \
                                          rotation_to_local,                                                           \
                                          depth_downscale,                                                             \
                                          reference_image,                                                             \
                                          image_one_step_more_detailed,                                                \
                                          deviation_threshold_multiplier,                                              \
                                          candidate_depth,                                                             \
                                          candidate_normal,                                                            \
                                          checkerboard_step,                                                           \
                                          only_each_fourth_pixel,                                                      \
                                          pixel_offset)
        switch (camera.type)
        {
        case 0U:
            METMODEL_LAUNCH_PROPAGATION_U16(0U);
            break;
        case 2U:
            METMODEL_LAUNCH_PROPAGATION_U16(2U);
            break;
        case 3U:
            METMODEL_LAUNCH_PROPAGATION_U16(3U);
            break;
        case 4U:
            METMODEL_LAUNCH_PROPAGATION_U16(4U);
            break;
        case 8U:
            METMODEL_LAUNCH_PROPAGATION_U16(8U);
            break;
        case 9U:
            METMODEL_LAUNCH_PROPAGATION_U16(9U);
            break;
        default:
            METMODEL_LAUNCH_PROPAGATION_U16(10U);
            break;
        }
#undef METMODEL_LAUNCH_PROPAGATION_U16
        return cudaGetLastError();
    }

    cudaError_t
    launch_recovered_patchmatch_propagation_f32_perspective_source(float* depth,
                                                                   std::uint8_t* normal,
                                                                   float* cost,
                                                                   const float* coarse_depth,
                                                                   const float* coarse_radius,
                                                                   const PatchMatchCamera& camera,
                                                                   const float* rotation_to_local,
                                                                   std::uint32_t depth_downscale,
                                                                   const float* reference_image,
                                                                   std::uint32_t image_one_step_more_detailed,
                                                                   float deviation_threshold_multiplier,
                                                                   float* candidate_depth,
                                                                   float* candidate_normal,
                                                                   std::uint32_t checkerboard_step,
                                                                   std::uint32_t only_each_fourth_pixel,
                                                                   std::uint32_t pixel_offset,
                                                                   std::size_t work_items,
                                                                   cudaStream_t stream)
    {
        if ((camera.type != 0U && camera.type != 2U && camera.type != 3U && camera.type != 4U && camera.type != 8U &&
             camera.type != 9U && camera.type != 10U) ||
            depth_downscale == 0U || (image_one_step_more_detailed != 0U && depth_downscale < 2U))
            return cudaErrorInvalidValue;
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
#define METMODEL_LAUNCH_PROPAGATION_F32(camera_type)                                                                   \
    patchmatch_propagation_extended_kernel<float, camera_type>                                                         \
        <<<blocks, threads, 0U, stream>>>(depth,                                                                       \
                                          normal,                                                                      \
                                          cost,                                                                        \
                                          coarse_depth,                                                                \
                                          coarse_radius,                                                               \
                                          camera,                                                                      \
                                          rotation_to_local,                                                           \
                                          depth_downscale,                                                             \
                                          reference_image,                                                             \
                                          image_one_step_more_detailed,                                                \
                                          deviation_threshold_multiplier,                                              \
                                          candidate_depth,                                                             \
                                          candidate_normal,                                                            \
                                          checkerboard_step,                                                           \
                                          only_each_fourth_pixel,                                                      \
                                          pixel_offset)
        switch (camera.type)
        {
        case 0U:
            METMODEL_LAUNCH_PROPAGATION_F32(0U);
            break;
        case 2U:
            METMODEL_LAUNCH_PROPAGATION_F32(2U);
            break;
        case 3U:
            METMODEL_LAUNCH_PROPAGATION_F32(3U);
            break;
        case 4U:
            METMODEL_LAUNCH_PROPAGATION_F32(4U);
            break;
        case 8U:
            METMODEL_LAUNCH_PROPAGATION_F32(8U);
            break;
        case 9U:
            METMODEL_LAUNCH_PROPAGATION_F32(9U);
            break;
        default:
            METMODEL_LAUNCH_PROPAGATION_F32(10U);
            break;
        }
#undef METMODEL_LAUNCH_PROPAGATION_F32
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_cost_u8_source(const float* depth,
                                                           const std::uint8_t* normal,
                                                           const float* cost,
                                                           const float* coarse_depth,
                                                           const float* coarse_radius,
                                                           const PatchMatchCamera& reference_camera,
                                                           std::uint32_t depth_downscale,
                                                           const std::uint8_t* reference_image,
                                                           std::uint32_t image_one_step_more_detailed,
                                                           float deviation_threshold_multiplier,
                                                           const PatchMatchCamera& neighbor_camera,
                                                           std::uint32_t neighbor_level,
                                                           std::uint32_t result_index,
                                                           cudaTextureObject_t neighbor_texture,
                                                           const std::uint64_t* neighbor_mask_offsets,
                                                           const std::uint8_t* neighbor_masks,
                                                           const float* candidate_depth,
                                                           const float* candidate_normal,
                                                           float* neighbor_cost,
                                                           std::uint32_t reference_patch_radius,
                                                           std::uint32_t hypotheses_per_pixel,
                                                           std::uint32_t is_checkerboard,
                                                           std::uint32_t checkerboard_step,
                                                           std::uint32_t only_each_fourth_pixel,
                                                           std::uint32_t pixel_offset,
                                                           std::size_t work_items,
                                                           cudaStream_t stream)
    {
        return launch_recovered_patchmatch_cost_source(depth,
                                                       normal,
                                                       cost,
                                                       coarse_depth,
                                                       coarse_radius,
                                                       reference_camera,
                                                       depth_downscale,
                                                       reference_image,
                                                       1U,
                                                       image_one_step_more_detailed,
                                                       deviation_threshold_multiplier,
                                                       neighbor_camera,
                                                       neighbor_level,
                                                       result_index,
                                                       neighbor_texture,
                                                       neighbor_mask_offsets,
                                                       neighbor_masks,
                                                       candidate_depth,
                                                       candidate_normal,
                                                       neighbor_cost,
                                                       reference_patch_radius,
                                                       hypotheses_per_pixel,
                                                       is_checkerboard,
                                                       checkerboard_step,
                                                       only_each_fourth_pixel,
                                                       pixel_offset,
                                                       work_items,
                                                       stream);
    }

    cudaError_t launch_recovered_patchmatch_cost_source(const float* depth,
                                                        const std::uint8_t* normal,
                                                        const float* cost,
                                                        const float* coarse_depth,
                                                        const float* coarse_radius,
                                                        const PatchMatchCamera& reference_camera,
                                                        std::uint32_t depth_downscale,
                                                        const void* reference_image,
                                                        std::uint32_t image_sample_bytes,
                                                        std::uint32_t image_one_step_more_detailed,
                                                        float deviation_threshold_multiplier,
                                                        const PatchMatchCamera& neighbor_camera,
                                                        std::uint32_t neighbor_level,
                                                        std::uint32_t result_index,
                                                        cudaTextureObject_t neighbor_texture,
                                                        const std::uint64_t* neighbor_mask_offsets,
                                                        const std::uint8_t* neighbor_masks,
                                                        const float* candidate_depth,
                                                        const float* candidate_normal,
                                                        float* neighbor_cost,
                                                        std::uint32_t reference_patch_radius,
                                                        std::uint32_t hypotheses_per_pixel,
                                                        std::uint32_t is_checkerboard,
                                                        std::uint32_t checkerboard_step,
                                                        std::uint32_t only_each_fourth_pixel,
                                                        std::uint32_t pixel_offset,
                                                        std::size_t work_items,
                                                        cudaStream_t stream)
    {
        if (reference_camera.type != 0U || neighbor_camera.type != 0U ||
            (image_sample_bytes != 1U && image_sample_bytes != 2U && image_sample_bytes != 4U) ||
            depth_downscale == 0U || (image_one_step_more_detailed != 0U && depth_downscale < 2U) ||
            (reference_patch_radius != 2U && reference_patch_radius != 3U) ||
            (hypotheses_per_pixel != 2U && hypotheses_per_pixel != 8U) ||
            (reference_patch_radius == 2U && hypotheses_per_pixel != 8U))
            return cudaErrorInvalidValue;
        constexpr unsigned int threads = 128U;
        const std::size_t thread_count = work_items * static_cast<std::size_t>(hypotheses_per_pixel);
        const unsigned int blocks = static_cast<unsigned int>((thread_count + threads - 1U) / threads);
#define METMODEL_LAUNCH_COST(sample_type, radius, hypotheses)                                                          \
    patchmatch_cost_kernel<sample_type, radius, hypotheses>                                                            \
        <<<blocks, threads, 0U, stream>>>(depth,                                                                       \
                                          normal,                                                                      \
                                          cost,                                                                        \
                                          coarse_depth,                                                                \
                                          coarse_radius,                                                               \
                                          reference_camera,                                                            \
                                          depth_downscale,                                                             \
                                          static_cast<const sample_type*>(reference_image),                            \
                                          image_one_step_more_detailed,                                                \
                                          deviation_threshold_multiplier,                                              \
                                          neighbor_camera,                                                             \
                                          neighbor_level,                                                              \
                                          result_index,                                                                \
                                          neighbor_texture,                                                            \
                                          neighbor_mask_offsets,                                                       \
                                          neighbor_masks,                                                              \
                                          candidate_depth,                                                             \
                                          candidate_normal,                                                            \
                                          neighbor_cost,                                                               \
                                          is_checkerboard,                                                             \
                                          checkerboard_step,                                                           \
                                          only_each_fourth_pixel,                                                      \
                                          pixel_offset)
#define METMODEL_DISPATCH_COST(sample_type)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (reference_patch_radius == 3U && hypotheses_per_pixel == 8U)                                                \
        {                                                                                                              \
            METMODEL_LAUNCH_COST(sample_type, 3U, 8U);                                                                 \
        }                                                                                                              \
        else if (reference_patch_radius == 2U)                                                                         \
        {                                                                                                              \
            METMODEL_LAUNCH_COST(sample_type, 2U, 8U);                                                                 \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            METMODEL_LAUNCH_COST(sample_type, 3U, 2U);                                                                 \
        }                                                                                                              \
    } while (false)
        if (image_sample_bytes == 1U)
            METMODEL_DISPATCH_COST(std::uint8_t);
        else if (image_sample_bytes == 2U)
            METMODEL_DISPATCH_COST(std::uint16_t);
        else
            METMODEL_DISPATCH_COST(float);
#undef METMODEL_DISPATCH_COST
#undef METMODEL_LAUNCH_COST
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_bilateral_u8_source(const float* depth,
                                                                const std::uint8_t* normal,
                                                                const std::uint8_t* image,
                                                                float* filtered_depth,
                                                                std::uint8_t* filtered_normal,
                                                                std::uint32_t width,
                                                                std::uint32_t height,
                                                                float sigma_d,
                                                                float sigma_r,
                                                                std::uint32_t pixel_offset,
                                                                std::size_t work_items,
                                                                cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_bilateral_kernel<std::uint8_t><<<blocks, threads, 0U, stream>>>(
            depth, normal, image, filtered_depth, filtered_normal, width, height, sigma_d, sigma_r, pixel_offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_bilateral_u16_source(const float* depth,
                                                                 const std::uint8_t* normal,
                                                                 const std::uint16_t* image,
                                                                 float* filtered_depth,
                                                                 std::uint8_t* filtered_normal,
                                                                 std::uint32_t width,
                                                                 std::uint32_t height,
                                                                 float sigma_d,
                                                                 float sigma_r,
                                                                 std::uint32_t pixel_offset,
                                                                 std::size_t work_items,
                                                                 cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_bilateral_kernel<std::uint16_t><<<blocks, threads, 0U, stream>>>(
            depth, normal, image, filtered_depth, filtered_normal, width, height, sigma_d, sigma_r, pixel_offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_bilateral_f32_source(const float* depth,
                                                                 const std::uint8_t* normal,
                                                                 const float* image,
                                                                 float* filtered_depth,
                                                                 std::uint8_t* filtered_normal,
                                                                 std::uint32_t width,
                                                                 std::uint32_t height,
                                                                 float sigma_d,
                                                                 float sigma_r,
                                                                 std::uint32_t pixel_offset,
                                                                 std::size_t work_items,
                                                                 cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_bilateral_kernel<float><<<blocks, threads, 0U, stream>>>(
            depth, normal, image, filtered_depth, filtered_normal, width, height, sigma_d, sigma_r, pixel_offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_filter_check_cost_source(float* depth,
                                                                     const float* cost,
                                                                     std::uint32_t* counter_no_cost,
                                                                     std::uint32_t* counter_big_cost,
                                                                     std::uint32_t pixel_offset,
                                                                     std::size_t work_items,
                                                                     cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_filter_check_cost_kernel<<<blocks, threads, 0U, stream>>>(
            depth, cost, counter_no_cost, counter_big_cost, pixel_offset, work_items);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_filter_check_neighbours_source(const float* depth,
                                                                           std::uint8_t* filtered_mask,
                                                                           const PatchMatchCamera& camera,
                                                                           std::uint32_t depth_downscale,
                                                                           float depth_min,
                                                                           float depth_max,
                                                                           std::uint32_t* counter_no_neighbours,
                                                                           std::uint32_t* counter_no_close_neighbours,
                                                                           std::uint32_t pixel_offset,
                                                                           std::size_t work_items,
                                                                           cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_filter_check_neighbours_kernel<<<blocks, threads, 0U, stream>>>(depth,
                                                                                   filtered_mask,
                                                                                   camera,
                                                                                   depth_downscale,
                                                                                   depth_min,
                                                                                   depth_max,
                                                                                   counter_no_neighbours,
                                                                                   counter_no_close_neighbours,
                                                                                   pixel_offset,
                                                                                   work_items);
        return cudaGetLastError();
    }

    template <std::uint32_t CameraType>
    cudaError_t
    launch_recovered_patchmatch_filter_normals_typed_source(const float* depth,
                                                            const std::uint8_t* normal,
                                                            std::uint8_t* estimated_normal,
                                                            bool estimate_normal_map,
                                                            std::uint8_t* filtered_mask,
                                                            const PatchMatchCamera& camera,
                                                            std::uint32_t depth_downscale,
                                                            std::uint32_t* counter_inconsistent_normal,
                                                            std::uint32_t* counter_bad_view_angle_estimated_normal,
                                                            std::uint32_t* counter_bad_view_angle_found_normal,
                                                            float* counter_cos_sum,
                                                            std::uint32_t* counter_ncos_sum,
                                                            std::uint32_t pixel_offset,
                                                            std::size_t work_items,
                                                            cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_filter_normals_kernel<CameraType>
            <<<blocks, threads, 0U, stream>>>(depth,
                                              normal,
                                              estimated_normal,
                                              estimate_normal_map ? 1U : 0U,
                                              filtered_mask,
                                              camera,
                                              depth_downscale,
                                              counter_inconsistent_normal,
                                              counter_bad_view_angle_estimated_normal,
                                              counter_bad_view_angle_found_normal,
                                              counter_cos_sum,
                                              counter_ncos_sum,
                                              pixel_offset,
                                              work_items);
        return cudaGetLastError();
    }

    cudaError_t
    launch_recovered_patchmatch_filter_normals_source(const float* depth,
                                                      const std::uint8_t* normal,
                                                      std::uint8_t* estimated_normal,
                                                      bool estimate_normal_map,
                                                      std::uint8_t* filtered_mask,
                                                      const PatchMatchCamera& camera,
                                                      std::uint32_t depth_downscale,
                                                      std::uint32_t* counter_inconsistent_normal,
                                                      std::uint32_t* counter_bad_view_angle_estimated_normal,
                                                      std::uint32_t* counter_bad_view_angle_found_normal,
                                                      float* counter_cos_sum,
                                                      std::uint32_t* counter_ncos_sum,
                                                      std::uint32_t pixel_offset,
                                                      std::size_t work_items,
                                                      cudaStream_t stream)
    {
#define METMODEL_FILTER_NORMALS_CASE(value)                                                                            \
    case value:                                                                                                        \
        return launch_recovered_patchmatch_filter_normals_typed_source<value>(depth,                                   \
                                                                              normal,                                  \
                                                                              estimated_normal,                        \
                                                                              estimate_normal_map,                     \
                                                                              filtered_mask,                           \
                                                                              camera,                                  \
                                                                              depth_downscale,                         \
                                                                              counter_inconsistent_normal,             \
                                                                              counter_bad_view_angle_estimated_normal, \
                                                                              counter_bad_view_angle_found_normal,     \
                                                                              counter_cos_sum,                         \
                                                                              counter_ncos_sum,                        \
                                                                              pixel_offset,                            \
                                                                              work_items,                              \
                                                                              stream)
        switch (camera.type)
        {
            METMODEL_FILTER_NORMALS_CASE(0U);
            METMODEL_FILTER_NORMALS_CASE(2U);
            METMODEL_FILTER_NORMALS_CASE(3U);
            METMODEL_FILTER_NORMALS_CASE(4U);
            METMODEL_FILTER_NORMALS_CASE(8U);
            METMODEL_FILTER_NORMALS_CASE(9U);
            METMODEL_FILTER_NORMALS_CASE(10U);
        default:
            return cudaErrorInvalidValue;
        }
#undef METMODEL_FILTER_NORMALS_CASE
    }

    template <std::uint32_t CameraType, bool AffineTransform = false>
    cudaError_t launch_recovered_patchmatch_filter_speckles_edges_typed_source(const float* depth,
                                                                               std::uint8_t* filtered_mask,
                                                                               const PatchMatchCamera& camera,
                                                                               std::uint32_t depth_downscale,
                                                                               std::uint32_t pixel_offset,
                                                                               std::size_t work_items,
                                                                               cudaStream_t stream)
    {
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        patchmatch_filter_speckles_edges_kernel<CameraType, AffineTransform>
            <<<blocks, threads, 0U, stream>>>(depth, filtered_mask, camera, depth_downscale, pixel_offset, work_items);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_patchmatch_filter_speckles_edges_source(const float* depth,
                                                                         std::uint8_t* filtered_mask,
                                                                         const PatchMatchCamera& camera,
                                                                         std::uint32_t depth_downscale,
                                                                         std::uint32_t pixel_offset,
                                                                         std::size_t work_items,
                                                                         cudaStream_t stream)
    {
        const bool exact_affine_transform = std::bit_cast<std::uint32_t>(camera.transform[12]) == 0x00000000U &&
                                            std::bit_cast<std::uint32_t>(camera.transform[13]) == 0x00000000U &&
                                            std::bit_cast<std::uint32_t>(camera.transform[14]) == 0x00000000U &&
                                            std::bit_cast<std::uint32_t>(camera.transform[15]) == 0x3f800000U;
        // Dispatch once on the host so the hot unprojection body does not
        // repeat three IEEE divisions by an identically-one coordinate.
        if (camera.type == 0U && exact_affine_transform)
        {
            return launch_recovered_patchmatch_filter_speckles_edges_typed_source<0U, true>(
                depth, filtered_mask, camera, depth_downscale, pixel_offset, work_items, stream);
        }
#define METMODEL_FILTER_SPECKLES_CASE(value)                                                                           \
    case value:                                                                                                        \
        return launch_recovered_patchmatch_filter_speckles_edges_typed_source<value>(                                  \
            depth, filtered_mask, camera, depth_downscale, pixel_offset, work_items, stream)
        switch (camera.type)
        {
            METMODEL_FILTER_SPECKLES_CASE(0U);
            METMODEL_FILTER_SPECKLES_CASE(2U);
            METMODEL_FILTER_SPECKLES_CASE(3U);
            METMODEL_FILTER_SPECKLES_CASE(4U);
            METMODEL_FILTER_SPECKLES_CASE(8U);
            METMODEL_FILTER_SPECKLES_CASE(9U);
            METMODEL_FILTER_SPECKLES_CASE(10U);
        default:
            return cudaErrorInvalidValue;
        }
#undef METMODEL_FILTER_SPECKLES_CASE
    }

    cudaError_t launch_recovered_ooc_update_u_source(const std::uint8_t* weights,
                                                     const std::uint8_t* histogram,
                                                     const std::uint32_t* neighbors,
                                                     const std::uint8_t* connectivity,
                                                     const std::uint8_t* refinement,
                                                     const std::uint8_t* flags,
                                                     float* u,
                                                     float* u_old,
                                                     const std::uint16_t* p,
                                                     std::uint16_t* v,
                                                     std::uint16_t* v_old,
                                                     const std::uint16_t* q,
                                                     std::uint32_t count,
                                                     float spatial_scale,
                                                     float threshold,
                                                     float alpha,
                                                     float beta,
                                                     float data_weight,
                                                     std::uint32_t functional_mode,
                                                     std::uint32_t offset,
                                                     std::size_t work_items,
                                                     cudaStream_t stream)
    {
        (void)spatial_scale;
        (void)threshold;
        if (functional_mode != 1U)
            return cudaErrorInvalidValue;
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        recovered_ooc_update_u_kernel<<<blocks, threads, 0U, stream>>>(weights,
                                                                       histogram,
                                                                       neighbors,
                                                                       connectivity,
                                                                       refinement,
                                                                       flags,
                                                                       u,
                                                                       u_old,
                                                                       p,
                                                                       v,
                                                                       v_old,
                                                                       q,
                                                                       count,
                                                                       alpha,
                                                                       beta,
                                                                       data_weight,
                                                                       offset);
        return cudaGetLastError();
    }

    cudaError_t launch_recovered_ooc_update_p_source(const std::uint32_t* neighbors,
                                                     const std::uint8_t* connectivity,
                                                     const std::uint8_t* refinement,
                                                     const std::uint8_t* flags,
                                                     const float* u,
                                                     const float* u_old,
                                                     std::uint16_t* p,
                                                     const std::uint16_t* v,
                                                     const std::uint16_t* v_old,
                                                     std::uint16_t* q,
                                                     std::uint32_t count,
                                                     float spatial_scale,
                                                     float alpha,
                                                     float beta,
                                                     float data_weight,
                                                     std::uint32_t functional_mode,
                                                     std::uint32_t offset,
                                                     std::size_t work_items,
                                                     cudaStream_t stream)
    {
        (void)spatial_scale;
        (void)data_weight;
        if (functional_mode != 1U)
            return cudaErrorInvalidValue;
        constexpr unsigned int threads = 128U;
        const unsigned int blocks = static_cast<unsigned int>((work_items + threads - 1U) / threads);
        recovered_ooc_update_p_kernel<<<blocks, threads, 0U, stream>>>(
            neighbors, connectivity, refinement, flags, u, u_old, p, v, v_old, q, count, alpha, beta, offset);
        return cudaGetLastError();
    }

} // namespace metmodel
