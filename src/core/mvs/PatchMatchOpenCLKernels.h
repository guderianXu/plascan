#pragma once

namespace xjw
{
namespace mvs
{
namespace detail
{

inline constexpr const char *kPatchMatchOpenClBuildOptions = "-cl-mad-enable";

inline constexpr const char *kPatchMatchOpenClSourcePrefix = R"CLC(
#define MAX_SOURCES 16
#define MAX_ROBUST_SUPPORT (MAX_SOURCES / 2 + 1)
#define WORK_GROUP_SIZE 16
#define MAX_PATCH_RADIUS 7
#define REFERENCE_TILE_SIZE (WORK_GROUP_SIZE + 2 * MAX_PATCH_RADIUS)
#define CHECKERBOARD_TILE_WIDTH (2 * WORK_GROUP_SIZE + 2 * MAX_PATCH_RADIUS)

static inline float sample_bilinear(__global const float *image,
                             int width,
                             int height,
                             float x,
                             float y)
{
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    if (x0 < 0 || y0 < 0 || x0 + 1 >= width || y0 + 1 >= height)
    {
        return 0.0f;
    }
    float tx = x - (float)x0;
    float ty = y - (float)y0;
    int row0 = y0 * width;
    int row1 = (y0 + 1) * width;
    float top = image[row0 + x0] * (1.0f - tx) + image[row0 + x0 + 1] * tx;
    float bottom = image[row1 + x0] * (1.0f - tx) + image[row1 + x0 + 1] * tx;
    return top * (1.0f - ty) + bottom * ty;
}

static inline int source_mask_valid(__global const uchar *masks,
                             int source_index,
                             int pixel_count,
                             int width,
                             int height,
                             float x,
                             float y,
                             int has_source_masks)
{
    if (!has_source_masks)
    {
        return 1;
    }
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    if (x0 < 0 || y0 < 0 || x0 + 1 >= width || y0 + 1 >= height)
    {
        return 0;
    }
    int base = source_index * pixel_count;
    return masks[base + y0 * width + x0] != 0
        && masks[base + y0 * width + x0 + 1] != 0
        && masks[base + (y0 + 1) * width + x0] != 0
        && masks[base + (y0 + 1) * width + x0 + 1] != 0;
}

static inline float3 reference_ray(int x,
                            int y,
                            float inv_fx,
                            float inv_fy,
                            float cx,
                            float cy)
{
    return (float3)(((float)x - cx) * inv_fx,
                    ((float)y - cy) * inv_fy,
                    1.0f);
}

static inline float4 face_normal_toward_camera(float4 normal, float3 ray)
{
    float length_squared = dot(normal.xyz, normal.xyz);
    if (!(length_squared > 1.0e-10f) || !isfinite(length_squared))
    {
        return (float4)(0.0f, 0.0f, -1.0f, 0.0f);
    }
    normal.xyz *= 1.0f / sqrt(length_squared);
    if (dot(normal.xyz, ray) > 0.0f)
    {
        normal.xyz = -normal.xyz;
    }
    normal.w = 0.0f;
    return normal;
}

static inline uint patchmatch_hash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

static inline float patchmatch_random(uint *state)
{
    *state = patchmatch_hash(*state + 0x9e3779b9u);
    return (float)(*state & 0x00ffffffu) * (1.0f / 16777216.0f);
}

static inline float4 random_facing_normal(uint *state, float3 ray)
{
    float z = patchmatch_random(state) * 2.0f - 1.0f;
    float angle = patchmatch_random(state) * 6.28318530718f;
    float radius = sqrt(fmax(0.0f, 1.0f - z * z));
    float4 normal = (float4)(radius * cos(angle),
                             radius * sin(angle),
                             z,
                             0.0f);
    return face_normal_toward_camera(normal, ray);
}

static inline float4 perturb_facing_normal(float4 normal,
                                    float amount,
                                    uint *state,
                                    float3 ray)
{
    float4 perturbation = (float4)(
        patchmatch_random(state) * 2.0f - 1.0f,
        patchmatch_random(state) * 2.0f - 1.0f,
        patchmatch_random(state) * 2.0f - 1.0f,
        0.0f);
    return face_normal_toward_camera(normal + amount * perturbation, ray);
}

static inline int same_plane_hypothesis(float left_depth,
                                 float4 left_normal,
                                 float right_depth,
                                 float4 right_normal)
{
    return left_depth == right_depth
        && left_normal.x == right_normal.x
        && left_normal.y == right_normal.y
        && left_normal.z == right_normal.z;
}

static inline float propagated_plane_depth(int from_x,
                                    int from_y,
                                    float from_depth,
                                    float4 normal,
                                    int to_x,
                                    int to_y,
                                    float inv_fx,
                                    float inv_fy,
                                    float cx,
                                    float cy)
{
    float3 from_ray = reference_ray(from_x, from_y, inv_fx, inv_fy, cx, cy);
    float3 to_ray = reference_ray(to_x, to_y, inv_fx, inv_fy, cx, cy);
    float denominator = dot(normal.xyz, to_ray);
    if (fabs(denominator) < 1.0e-6f)
    {
        return 0.0f;
    }
    float plane_distance = from_depth * dot(normal.xyz, from_ray);
    float depth = plane_distance / denominator;
    return depth > 0.0f && isfinite(depth) ? depth : 0.0f;
}

static inline void compose_plane_homography(float4 normal,
                                     float plane_distance,
                                     __global const float *camera,
                                     float inv_fx,
                                     float inv_fy,
                                     float cx,
                                     float cy,
                                     __private float *homography)
{
    float inverse_cx = -cx * inv_fx;
    float inverse_cy = -cy * inv_fy;
    float inverse_distance = 1.0f / plane_distance;
    float3 inverse_distance_normal = normal.xyz * inverse_distance;

    float source_x_column = camera[0]
        * (camera[4] + inverse_distance_normal.x * camera[13])
        + camera[1] * (camera[10] + inverse_distance_normal.x * camera[15]);
    float source_x_row = camera[0]
        * (camera[5] + inverse_distance_normal.y * camera[13])
        + camera[1] * (camera[11] + inverse_distance_normal.y * camera[15]);
    float source_x_offset = camera[0]
        * (camera[6] + inverse_distance_normal.z * camera[13])
        + camera[1] * (camera[12] + inverse_distance_normal.z * camera[15]);
    float source_y_column = camera[2]
        * (camera[7] + inverse_distance_normal.x * camera[14])
        + camera[3] * (camera[10] + inverse_distance_normal.x * camera[15]);
    float source_y_row = camera[2]
        * (camera[8] + inverse_distance_normal.y * camera[14])
        + camera[3] * (camera[11] + inverse_distance_normal.y * camera[15]);
    float source_y_offset = camera[2]
        * (camera[9] + inverse_distance_normal.z * camera[14])
        + camera[3] * (camera[12] + inverse_distance_normal.z * camera[15]);
    float source_z_column = camera[10]
        + inverse_distance_normal.x * camera[15];
    float source_z_row = camera[11]
        + inverse_distance_normal.y * camera[15];
    float source_z_offset = camera[12]
        + inverse_distance_normal.z * camera[15];

    homography[0] = inv_fx * source_x_column;
    homography[1] = inv_fy * source_x_row;
    homography[2] = source_x_offset
        + inverse_cx * source_x_column + inverse_cy * source_x_row;
    homography[3] = inv_fx * source_y_column;
    homography[4] = inv_fy * source_y_row;
    homography[5] = source_y_offset
        + inverse_cx * source_y_column + inverse_cy * source_y_row;
    homography[6] = inv_fx * source_z_column;
    homography[7] = inv_fy * source_z_row;
    homography[8] = source_z_offset
        + inverse_cx * source_z_column + inverse_cy * source_z_row;
}
)CLC";

inline constexpr const char *kPatchMatchOpenClSourcePhotometric = R"CLC(
static inline float source_ncc(int center_x,
                        int center_y,
                        float depth,
                        float4 normal,
                        int source_index,
                         __local const float *reference_tile,
                         __local const uchar *reference_mask_tile,
                         __global const float *sources,
                        __global const float *source_cameras,
                         __global const uchar *source_masks,
                        int width,
                        int height,
                        int patch_half,
                        int patch_step,
                        float minimum_mask_ratio,
                        int has_reference_mask,
                        int has_source_masks,
                        float inv_fx,
                        float inv_fy,
                        float cx,
                         float cy,
                         int tile_origin_x,
                         int tile_origin_y,
                         int reference_tile_stride)
{
    int pixel_count = width * height;
    __global const float *source = sources + source_index * pixel_count;
    __global const float *camera = source_cameras + source_index * 16;
    int radius = clamp(patch_half, 1, 7);
    // The caller selects the sampling stride explicitly. Coarse search and
    // plane propagation use dense support because sparse NCC changes candidate
    // ordering; only local refinement may use stride two.
    int step = clamp(patch_step, 1, 3);
    float sum_reference = 0.0f;
    float sum_source = 0.0f;
    float sum_reference_squared = 0.0f;
    float sum_source_squared = 0.0f;
    float sum_product = 0.0f;
    float sum_reference_gradient = 0.0f;
    float sum_source_gradient = 0.0f;
    float sum_reference_gradient_squared = 0.0f;
    float sum_source_gradient_squared = 0.0f;
    float sum_gradient_product = 0.0f;
    int candidate_count = 0;
    int valid_count = 0;
    int gradient_candidate_count = 0;
    int gradient_valid_count = 0;
    int census_candidate_count = 0;
    int census_valid_count = 0;
    int census_agreement_count = 0;
    float3 center_ray = reference_ray(
        center_x, center_y, inv_fx, inv_fy, cx, cy);
    float plane_distance = depth * dot(normal.xyz, center_ray);
    if (plane_distance == 0.0f || !isfinite(plane_distance))
    {
        return 0.0f;
    }
    float homography[9];
    compose_plane_homography(normal,
                             plane_distance,
                             camera,
                             inv_fx,
                             inv_fy,
                             cx,
                             cy,
                             homography);
    float center_projected_x = homography[0] * (float)center_x
        + homography[1] * (float)center_y + homography[2];
    float center_projected_y = homography[3] * (float)center_x
        + homography[4] * (float)center_y + homography[5];
    float center_projected_z = homography[6] * (float)center_x
        + homography[7] * (float)center_y + homography[8];
    float center_source_x = fabs(center_projected_z) > 1.0e-6f
        ? center_projected_x / center_projected_z
        : -1.0f;
    float center_source_y = fabs(center_projected_z) > 1.0e-6f
        ? center_projected_y / center_projected_z
        : -1.0f;
    int center_tile_x = center_x - tile_origin_x + MAX_PATCH_RADIUS;
    int center_tile_y = center_y - tile_origin_y + MAX_PATCH_RADIUS;
    float reference_center = reference_tile[
        center_tile_y * reference_tile_stride + center_tile_x];
    int center_valid = center_source_x >= 0.0f && center_source_y >= 0.0f
        && center_source_x < (float)(width - 1)
        && center_source_y < (float)(height - 1)
        && source_mask_valid(source_masks,
                             source_index,
                             pixel_count,
                             width,
                             height,
                             center_source_x,
                             center_source_y,
                             has_source_masks);
    float source_center = center_valid
        ? sample_bilinear(source,
                          width,
                          height,
                          center_source_x,
                          center_source_y)
        : -1.0f;
    center_valid = center_valid && source_center >= 0.0f;

    for (int dy = -radius; dy <= radius; dy += step)
    {
        int reference_y = center_y + dy;
        if (reference_y < 0 || reference_y >= height)
        {
            continue;
        }
        int first_reference_x = center_x - radius;
        float projected_x = homography[0] * (float)first_reference_x
            + homography[1] * (float)reference_y + homography[2];
        float projected_y = homography[3] * (float)first_reference_x
            + homography[4] * (float)reference_y + homography[5];
        float projected_z = homography[6] * (float)first_reference_x
            + homography[7] * (float)reference_y + homography[8];
        float plane_denominator = normal.x
                * (((float)first_reference_x - cx) * inv_fx)
            + normal.y * (((float)reference_y - cy) * inv_fy)
            + normal.z;
        float projected_x_step = homography[0] * (float)step;
        float projected_y_step = homography[3] * (float)step;
        float projected_z_step = homography[6] * (float)step;
        float plane_denominator_step = normal.x * inv_fx * (float)step;
        for (int dx = -radius;
             dx <= radius;
             dx += step,
             projected_x += projected_x_step,
             projected_y += projected_y_step,
             projected_z += projected_z_step,
             plane_denominator += plane_denominator_step)
        {
            int reference_x = center_x + dx;
            if (reference_x < 0 || reference_x >= width)
            {
                continue;
            }
            int tile_x = reference_x - tile_origin_x + MAX_PATCH_RADIUS;
            int tile_y = reference_y - tile_origin_y + MAX_PATCH_RADIUS;
            int tile_index = tile_y * reference_tile_stride + tile_x;
            if (has_reference_mask && reference_mask_tile[tile_index] == 0)
            {
                continue;
            }

            // Match the CUDA/CPU mask semantics: reference-mask exclusions
            // are not candidate observations. Only a failed projection or
            // source-mask lookup for a trusted reference sample contributes
            // to the minimum valid-patch ratio denominator.
            ++candidate_count;
            gradient_candidate_count += 2;
            ++census_candidate_count;

            if (fabs(plane_denominator) < 1.0e-6f)
            {
                continue;
            }
            float patch_depth = plane_distance / plane_denominator;
            if (!(patch_depth > 1.0e-6f) || !isfinite(patch_depth))
            {
                continue;
            }
            if (!(projected_z > 1.0e-6f))
            {
                continue;
            }
            float source_x = projected_x / projected_z;
            float source_y = projected_y / projected_z;
            if (!source_mask_valid(source_masks,
                                   source_index,
                                   pixel_count,
                                   width,
                                   height,
                                   source_x,
                                   source_y,
                                   has_source_masks))
            {
                continue;
            }
            int source_x0 = (int)floor(source_x);
            int source_y0 = (int)floor(source_y);
            if (source_x0 < 0 || source_y0 < 0
                || source_x0 + 1 >= width || source_y0 + 1 >= height)
            {
                continue;
            }

            float reference_value = reference_tile[tile_index];
            float source_value = sample_bilinear(source, width, height, source_x, source_y);
            sum_reference += reference_value;
            sum_source += source_value;
            sum_reference_squared += reference_value * reference_value;
            sum_source_squared += source_value * source_value;
            sum_product += reference_value * source_value;
            ++valid_count;
)CLC";

inline constexpr const char *kPatchMatchOpenClSourceGradientCensus = R"CLC(

            int gradient_valid = reference_x > 0 && reference_y > 0
                && reference_x + 1 < width && reference_y + 1 < height;
            if (gradient_valid && has_reference_mask)
            {
                gradient_valid =
                    reference_mask_tile[tile_index - 1] != 0
                    && reference_mask_tile[tile_index + 1] != 0
                    && reference_mask_tile[
                        tile_index - reference_tile_stride] != 0
                    && reference_mask_tile[
                        tile_index + reference_tile_stride] != 0;
            }
            gradient_valid = gradient_valid
                && source_x - 1.0f >= 0.0f
                && source_y - 1.0f >= 0.0f
                && source_x + 1.0f < (float)(width - 1)
                && source_y + 1.0f < (float)(height - 1)
                && source_mask_valid(source_masks,
                                     source_index,
                                     pixel_count,
                                     width,
                                     height,
                                     source_x - 1.0f,
                                     source_y,
                                     has_source_masks)
                && source_mask_valid(source_masks,
                                     source_index,
                                     pixel_count,
                                     width,
                                     height,
                                     source_x + 1.0f,
                                     source_y,
                                     has_source_masks)
                && source_mask_valid(source_masks,
                                     source_index,
                                     pixel_count,
                                     width,
                                     height,
                                     source_x,
                                     source_y - 1.0f,
                                     has_source_masks)
                && source_mask_valid(source_masks,
                                     source_index,
                                     pixel_count,
                                     width,
                                     height,
                                     source_x,
                                     source_y + 1.0f,
                                     has_source_masks);
            if (gradient_valid)
            {
                float reference_gradient_x = 0.5f
                    * (reference_tile[tile_index + 1]
                       - reference_tile[tile_index - 1]);
                float reference_gradient_y = 0.5f
                    * (reference_tile[tile_index + reference_tile_stride]
                       - reference_tile[tile_index - reference_tile_stride]);
                float source_gradient_x = 0.5f
                    * (sample_bilinear(source,
                                       width,
                                       height,
                                       source_x + 1.0f,
                                       source_y)
                       - sample_bilinear(source,
                                         width,
                                         height,
                                         source_x - 1.0f,
                                         source_y));
                float source_gradient_y = 0.5f
                    * (sample_bilinear(source,
                                       width,
                                       height,
                                       source_x,
                                       source_y + 1.0f)
                       - sample_bilinear(source,
                                         width,
                                         height,
                                         source_x,
                                         source_y - 1.0f));
                sum_reference_gradient += reference_gradient_x
                    + reference_gradient_y;
                sum_source_gradient += source_gradient_x + source_gradient_y;
                sum_reference_gradient_squared +=
                    reference_gradient_x * reference_gradient_x
                    + reference_gradient_y * reference_gradient_y;
                sum_source_gradient_squared +=
                    source_gradient_x * source_gradient_x
                    + source_gradient_y * source_gradient_y;
                sum_gradient_product +=
                    reference_gradient_x * source_gradient_x
                    + reference_gradient_y * source_gradient_y;
                gradient_valid_count += 2;
            }

            if (center_valid)
            {
                float reference_delta = reference_value - reference_center;
                float source_delta = source_value - source_center;
                int reference_rank = reference_delta > 0.01f
                    ? 1
                    : (reference_delta < -0.01f ? -1 : 0);
                int source_rank = source_delta > 0.01f
                    ? 1
                    : (source_delta < -0.01f ? -1 : 0);
                ++census_valid_count;
                census_agreement_count += reference_rank == source_rank;
            }
        }
    }

    int required_count = max(4, (int)ceil((float)candidate_count * minimum_mask_ratio));
    if (valid_count < required_count)
    {
        return 0.0f;
    }
    float inverse_count = 1.0f / (float)valid_count;
    float mean_reference = sum_reference * inverse_count;
    float mean_source = sum_source * inverse_count;
    float variance_reference = fmax(0.0f,
        sum_reference_squared * inverse_count - mean_reference * mean_reference);
    float variance_source = fmax(0.0f,
        sum_source_squared * inverse_count - mean_source * mean_source);
    float variance_product = variance_reference * variance_source;
    if (variance_product < 1.0e-10f)
    {
        return 0.0f;
    }
    float covariance = sum_product * inverse_count - mean_reference * mean_source;
    float intensity_score = clamp(
        (covariance / sqrt(variance_product) + 1.0f) * 0.5f,
        0.0f,
        1.0f);
    float weighted_score = 0.50f * intensity_score;
    float weight_sum = 0.50f;

    int gradient_required = max(
        4,
        (int)ceil((float)gradient_candidate_count * minimum_mask_ratio));
    if (gradient_valid_count >= gradient_required)
    {
        float inverse_gradient_count = 1.0f / (float)gradient_valid_count;
        float mean_reference_gradient =
            sum_reference_gradient * inverse_gradient_count;
        float mean_source_gradient =
            sum_source_gradient * inverse_gradient_count;
        float variance_reference_gradient = fmax(
            0.0f,
            sum_reference_gradient_squared * inverse_gradient_count
                - mean_reference_gradient * mean_reference_gradient);
        float variance_source_gradient = fmax(
            0.0f,
            sum_source_gradient_squared * inverse_gradient_count
                - mean_source_gradient * mean_source_gradient);
        float gradient_variance_product =
            variance_reference_gradient * variance_source_gradient;
        if (gradient_variance_product >= 1.0e-10f)
        {
            float gradient_covariance =
                sum_gradient_product * inverse_gradient_count
                - mean_reference_gradient * mean_source_gradient;
            float gradient_score = clamp(
                (gradient_covariance / sqrt(gradient_variance_product) + 1.0f)
                    * 0.5f,
                0.0f,
                1.0f);
            weighted_score += 0.30f * gradient_score;
            weight_sum += 0.30f;
        }
    }

    int census_required = max(
        4,
        (int)ceil((float)census_candidate_count * minimum_mask_ratio));
    if (census_valid_count >= census_required)
    {
        weighted_score += 0.20f
            * (float)census_agreement_count / (float)census_valid_count;
        weight_sum += 0.20f;
    }
    return weighted_score / weight_sum;
}
)CLC";

inline constexpr const char *kPatchMatchOpenClSourceMain = R"CLC(
static inline float robust_depth_score(int x,
                                int y,
                                float depth,
                                float4 normal,
                                __local const float *reference_tile,
                                __local const uchar *reference_mask_tile,
                                __global const float *sources,
                                __global const float *source_cameras,
                                __global const uchar *source_masks,
                                int width,
                                int height,
                                int source_count,
                                int patch_half,
                                int patch_step,
                                float minimum_mask_ratio,
                                int has_reference_mask,
                                int has_source_masks,
                                float inv_fx,
                                float inv_fy,
                                float cx,
                                float cy,
                                int tile_origin_x,
                                int tile_origin_y,
                                int reference_tile_stride)
{
    int required_support = source_count <= 2 ? source_count : source_count / 2 + 1;
    float strongest[MAX_ROBUST_SUPPORT];
    for (int index = 0; index < required_support; ++index)
    {
        strongest[index] = 0.0f;
    }
    int support_count = 0;
    int stored_count = 0;
    for (int source_index = 0; source_index < source_count; ++source_index)
    {
        float score = source_ncc(x, y, depth, normal, source_index,
                                 reference_tile, reference_mask_tile,
                                 sources, source_cameras, source_masks,
                                 width, height, patch_half, patch_step,
                                 minimum_mask_ratio,
                                 has_reference_mask, has_source_masks,
                                 inv_fx, inv_fy, cx, cy,
                                 tile_origin_x, tile_origin_y,
                                 reference_tile_stride);
        if (!(score > 0.05f))
        {
            continue;
        }

        ++support_count;
        if (stored_count == required_support
            && score <= strongest[required_support - 1])
        {
            continue;
        }
        int insert_at = stored_count < required_support
            ? stored_count
            : required_support - 1;
        while (insert_at > 0 && strongest[insert_at - 1] < score)
        {
            if (insert_at < required_support)
            {
                strongest[insert_at] = strongest[insert_at - 1];
            }
            --insert_at;
        }
        strongest[insert_at] = score;
        if (stored_count < required_support)
        {
            ++stored_count;
        }
    }
    if (support_count < required_support || stored_count < required_support)
    {
        return 0.0f;
    }
    float sum = 0.0f;
    for (int index = 0; index < required_support; ++index)
    {
        sum += strongest[index];
    }
    return sum / (float)required_support;
}

__kernel void initialize_planes(
    __global const float *reference,
    __global const float *sources,
    __global const float *source_cameras,
    __global const uchar *reference_mask,
    __global const uchar *source_masks,
    __global const float *hint_depth,
    __global const float *hint_radius,
    __global float *depth_output,
    __global float4 *normal_output,
    __global float *score_output,
    int width,
    int height,
    int source_count,
    int patch_half,
    int depth_sample_count,
    float minimum_mask_ratio,
    float z_near,
    float z_far,
    float confidence_threshold,
    float uniqueness_relative_step,
    float uniqueness_minimum_margin,
    float uniqueness_minimum_scale,
    int has_reference_mask,
    int has_source_masks,
    int has_hint,
    int has_hint_radius,
    float inv_fx,
    float inv_fy,
    float cx,
    float cy)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    int local_linear_index = (int)get_local_id(1) * WORK_GROUP_SIZE
        + (int)get_local_id(0);
    int tile_origin_x = (int)get_group_id(0) * WORK_GROUP_SIZE;
    int tile_origin_y = (int)get_group_id(1) * WORK_GROUP_SIZE;
    __local float reference_tile[REFERENCE_TILE_SIZE * REFERENCE_TILE_SIZE];
    __local uchar reference_mask_tile[REFERENCE_TILE_SIZE * REFERENCE_TILE_SIZE];
    for (int tile_index = local_linear_index;
         tile_index < REFERENCE_TILE_SIZE * REFERENCE_TILE_SIZE;
         tile_index += WORK_GROUP_SIZE * WORK_GROUP_SIZE)
    {
        int tile_x = tile_index % REFERENCE_TILE_SIZE;
        int tile_y = tile_index / REFERENCE_TILE_SIZE;
        int image_x = tile_origin_x + tile_x - MAX_PATCH_RADIUS;
        int image_y = tile_origin_y + tile_y - MAX_PATCH_RADIUS;
        int inside = image_x >= 0 && image_y >= 0 && image_x < width && image_y < height;
        int image_index = inside ? image_y * width + image_x : 0;
        reference_tile[tile_index] = inside ? reference[image_index] : 0.0f;
        reference_mask_tile[tile_index] = inside
            ? (has_reference_mask ? reference_mask[image_index] : (uchar)255)
            : (uchar)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x >= width || y >= height)
    {
        return;
    }
    int index = y * width + x;
    if (has_reference_mask && reference_mask[index] == 0)
    {
        depth_output[index] = 0.0f;
        normal_output[index] = (float4)(0.0f, 0.0f, -1.0f, 0.0f);
        score_output[index] = 0.0f;
        return;
    }

    float4 fronto_parallel_normal = face_normal_toward_camera(
        (float4)(0.0f, 0.0f, -1.0f, 0.0f),
        reference_ray(x, y, inv_fx, inv_fy, cx, cy));

    float local_near = z_near;
    float local_far = z_far;
    if (has_hint && hint_depth[index] > 0.0f && isfinite(hint_depth[index]))
    {
        float radius = has_hint_radius ? hint_radius[index] : 0.0f;
        if (!(radius > 0.0f))
        {
            radius = fmax(0.05f * hint_depth[index], 0.01f * (z_far - z_near));
        }
        local_near = fmax(z_near, hint_depth[index] - radius);
        local_far = fmin(z_far, hint_depth[index] + radius);
        if (!(local_far > local_near))
        {
            local_near = z_near;
            local_far = z_far;
        }
    }

    int coarse_samples = clamp(depth_sample_count, 16, 96);
    float inverse_far = 1.0f / local_far;
    float inverse_near = 1.0f / local_near;
    float inverse_step = (inverse_near - inverse_far) / (float)(coarse_samples - 1);
    float best_depth = 0.0f;
    float best_score = 0.0f;
    for (int sample_index = 0; sample_index < coarse_samples; ++sample_index)
    {
        float inverse_depth = inverse_far + inverse_step * (float)sample_index;
        float depth = 1.0f / inverse_depth;
        float score = robust_depth_score(x, y, depth, fronto_parallel_normal,
                                         reference_tile, reference_mask_tile,
                                         sources, source_cameras, source_masks,
                                         width, height, source_count, patch_half, 1,
                                         minimum_mask_ratio, has_reference_mask,
                                         has_source_masks, inv_fx, inv_fy, cx, cy,
                                         tile_origin_x, tile_origin_y,
                                         REFERENCE_TILE_SIZE);
        if (score > best_score)
        {
            best_score = score;
            best_depth = depth;
        }
    }

    if (best_depth > 0.0f)
    {
        float best_inverse = 1.0f / best_depth;
        float refine_step = inverse_step / 6.0f;
        for (int refine_index = -6; refine_index <= 6; ++refine_index)
        {
            float inverse_depth = clamp(best_inverse + refine_step * (float)refine_index,
                                        inverse_far,
                                        inverse_near);
            float depth = 1.0f / inverse_depth;
            float score = robust_depth_score(x, y, depth, fronto_parallel_normal,
                                             reference_tile, reference_mask_tile,
                                             sources, source_cameras, source_masks,
                                             width, height, source_count, patch_half, 1,
                                             minimum_mask_ratio, has_reference_mask,
                                             has_source_masks, inv_fx, inv_fy, cx, cy,
                                             tile_origin_x, tile_origin_y,
                                             REFERENCE_TILE_SIZE);
            if (score > best_score)
            {
                best_score = score;
                best_depth = depth;
            }
        }
    }

    depth_output[index] = best_depth;
    normal_output[index] = fronto_parallel_normal;
    score_output[index] = best_score;
}
)CLC";

inline constexpr const char *kPatchMatchOpenClSourcePropagation = R"CLC(

__kernel void propagate_planes(
    __global const float *reference,
    __global const float *sources,
    __global const float *source_cameras,
    __global const uchar *reference_mask,
    __global const uchar *source_masks,
    __global const float *hint_depth,
    __global const float *hint_radius,
    __global float *depth_output,
    __global float4 *normal_output,
    __global float *score_output,
    int width,
    int height,
    int source_count,
    int patch_half,
    int depth_sample_count,
    float minimum_mask_ratio,
    float z_near,
    float z_far,
    float confidence_threshold,
    float uniqueness_relative_step,
    float uniqueness_minimum_margin,
    float uniqueness_minimum_scale,
    int has_reference_mask,
    int has_source_masks,
    int has_hint,
    int has_hint_radius,
    float inv_fx,
    float inv_fy,
    float cx,
    float cy,
    int checkerboard,
    int iteration,
    float perturbation)
{
    int compact_x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    int x = compact_x * 2 + ((y + checkerboard) & 1);
    int local_linear_index = (int)get_local_id(1) * WORK_GROUP_SIZE
        + (int)get_local_id(0);
    int tile_origin_x = (int)get_group_id(0) * 2 * WORK_GROUP_SIZE;
    int tile_origin_y = (int)get_group_id(1) * WORK_GROUP_SIZE;
    __local float reference_tile[CHECKERBOARD_TILE_WIDTH * REFERENCE_TILE_SIZE];
    __local uchar reference_mask_tile[CHECKERBOARD_TILE_WIDTH * REFERENCE_TILE_SIZE];
    for (int tile_index = local_linear_index;
         tile_index < CHECKERBOARD_TILE_WIDTH * REFERENCE_TILE_SIZE;
         tile_index += WORK_GROUP_SIZE * WORK_GROUP_SIZE)
    {
        int tile_x = tile_index % CHECKERBOARD_TILE_WIDTH;
        int tile_y = tile_index / CHECKERBOARD_TILE_WIDTH;
        int image_x = tile_origin_x + tile_x - MAX_PATCH_RADIUS;
        int image_y = tile_origin_y + tile_y - MAX_PATCH_RADIUS;
        int inside = image_x >= 0 && image_y >= 0 && image_x < width && image_y < height;
        int image_index = inside ? image_y * width + image_x : 0;
        reference_tile[tile_index] = inside ? reference[image_index] : 0.0f;
        reference_mask_tile[tile_index] = inside
            ? (has_reference_mask ? reference_mask[image_index] : (uchar)255)
            : (uchar)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x >= width || y >= height)
    {
        return;
    }
    int index = y * width + x;
    if (has_reference_mask && reference_mask[index] == 0)
    {
        return;
    }

    float local_near = z_near;
    float local_far = z_far;
    if (has_hint && hint_depth[index] > 0.0f && isfinite(hint_depth[index]))
    {
        float radius = has_hint_radius ? hint_radius[index] : 0.0f;
        if (!(radius > 0.0f))
        {
            radius = fmax(0.05f * hint_depth[index], 0.01f * (z_far - z_near));
        }
        local_near = fmax(z_near, hint_depth[index] - radius);
        local_far = fmin(z_far, hint_depth[index] + radius);
        if (!(local_far > local_near))
        {
            local_near = z_near;
            local_far = z_far;
        }
    }

    float best_depth = depth_output[index];
    float4 best_normal = face_normal_toward_camera(
        normal_output[index], reference_ray(x, y, inv_fx, inv_fy, cx, cy));
    float best_score = score_output[index];
    const int offset_x[4] = {-1, 1, 0, 0};
    const int offset_y[4] = {0, 0, -1, 1};
    for (int neighbor = 0; neighbor < 4; ++neighbor)
    {
        int neighbor_x = x + offset_x[neighbor];
        int neighbor_y = y + offset_y[neighbor];
        if (neighbor_x < 0 || neighbor_y < 0
            || neighbor_x >= width || neighbor_y >= height)
        {
            continue;
        }
        int neighbor_index = neighbor_y * width + neighbor_x;
        float neighbor_depth = depth_output[neighbor_index];
        if (!(neighbor_depth > 0.0f))
        {
            continue;
        }
        float4 neighbor_normal = face_normal_toward_camera(
            normal_output[neighbor_index],
            reference_ray(x, y, inv_fx, inv_fy, cx, cy));
        float candidate_depth = propagated_plane_depth(
            neighbor_x, neighbor_y, neighbor_depth, neighbor_normal,
            x, y, inv_fx, inv_fy, cx, cy);
        float candidate_score = 0.0f;
        int has_candidate_score = 0;
        if (candidate_depth >= local_near && candidate_depth <= local_far)
        {
            candidate_score = same_plane_hypothesis(
                candidate_depth, neighbor_normal, best_depth, best_normal)
                ? best_score
                : robust_depth_score(
                    x, y, candidate_depth, neighbor_normal,
                    reference_tile, reference_mask_tile,
                    sources, source_cameras, source_masks,
                    width, height, source_count, patch_half, 1,
                    minimum_mask_ratio, has_reference_mask, has_source_masks,
                    inv_fx, inv_fy, cx, cy, tile_origin_x, tile_origin_y,
                    CHECKERBOARD_TILE_WIDTH);
            has_candidate_score = 1;
            if (candidate_score > best_score)
            {
                best_depth = candidate_depth;
                best_normal = neighbor_normal;
                best_score = candidate_score;
            }
        }

        // A propagated plane changes depth and normal together. Also testing
        // the neighbor normal at the current depth lets the surface orientation
        // converge without forcing an otherwise worse depth displacement.
        float normal_only_score = same_plane_hypothesis(
            best_depth, neighbor_normal, best_depth, best_normal)
            ? best_score
            : (has_candidate_score && candidate_depth == best_depth
                ? candidate_score
                : robust_depth_score(
                    x, y, best_depth, neighbor_normal,
                    reference_tile, reference_mask_tile,
                    sources, source_cameras, source_masks,
                    width, height, source_count, patch_half, 1,
                    minimum_mask_ratio, has_reference_mask, has_source_masks,
                    inv_fx, inv_fy, cx, cy, tile_origin_x, tile_origin_y,
                    CHECKERBOARD_TILE_WIDTH));
        if (normal_only_score > best_score)
        {
            best_normal = neighbor_normal;
            best_score = normal_only_score;
        }
    }

    uint random_state = patchmatch_hash(
        (uint)index ^ ((uint)(iteration + 1) * 0x85ebca6bu)
        ^ ((uint)(checkerboard + 1) * 0xc2b2ae35u));
    float3 current_ray = reference_ray(x, y, inv_fx, inv_fy, cx, cy);
    float4 random_normal = iteration == 0
        ? random_facing_normal(&random_state, current_ray)
        : perturb_facing_normal(
            best_normal, fmax(0.02f, perturbation), &random_state, current_ray);
    float depth_span = local_far - local_near;
    float random_depth = clamp(
        best_depth + (patchmatch_random(&random_state) * 2.0f - 1.0f)
            * depth_span * perturbation,
        local_near,
        local_far);
    float candidate_depths[3] = {best_depth, random_depth, random_depth};
    float4 candidate_normals[3] = {random_normal, best_normal, random_normal};
    for (int candidate = 0; candidate < 3; ++candidate)
    {
        if (!(candidate_depths[candidate] > 0.0f))
        {
            continue;
        }
        if (same_plane_hypothesis(candidate_depths[candidate],
                                  candidate_normals[candidate],
                                  best_depth,
                                  best_normal))
        {
            continue;
        }
        float candidate_score = robust_depth_score(
            x, y, candidate_depths[candidate], candidate_normals[candidate],
            reference_tile, reference_mask_tile,
            sources, source_cameras, source_masks,
            width, height, source_count, patch_half, 1,
            minimum_mask_ratio, has_reference_mask, has_source_masks,
            inv_fx, inv_fy, cx, cy, tile_origin_x, tile_origin_y,
            CHECKERBOARD_TILE_WIDTH);
        if (candidate_score > best_score)
        {
            best_depth = candidate_depths[candidate];
            best_normal = candidate_normals[candidate];
            best_score = candidate_score;
        }
    }

    depth_output[index] = best_depth;
    normal_output[index] = best_normal;
    score_output[index] = best_score;
}
)CLC";

inline constexpr const char *kPatchMatchOpenClSourceFinalize = R"CLC(

__kernel void finalize_planes(
    __global const float *reference,
    __global const float *sources,
    __global const float *source_cameras,
    __global const uchar *reference_mask,
    __global const uchar *source_masks,
    __global const float *hint_depth,
    __global const float *hint_radius,
    __global float *depth_output,
    __global float4 *normal_output,
    __global float *score_output,
    int width,
    int height,
    int source_count,
    int patch_half,
    int depth_sample_count,
    float minimum_mask_ratio,
    float z_near,
    float z_far,
    float confidence_threshold,
    float uniqueness_relative_step,
    float uniqueness_minimum_margin,
    float uniqueness_minimum_scale,
    int has_reference_mask,
    int has_source_masks,
    int has_hint,
    int has_hint_radius,
    float inv_fx,
    float inv_fy,
    float cx,
    float cy)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    int local_linear_index = (int)get_local_id(1) * WORK_GROUP_SIZE
        + (int)get_local_id(0);
    int tile_origin_x = (int)get_group_id(0) * WORK_GROUP_SIZE;
    int tile_origin_y = (int)get_group_id(1) * WORK_GROUP_SIZE;
    __local float reference_tile[REFERENCE_TILE_SIZE * REFERENCE_TILE_SIZE];
    __local uchar reference_mask_tile[REFERENCE_TILE_SIZE * REFERENCE_TILE_SIZE];
    for (int tile_index = local_linear_index;
         tile_index < REFERENCE_TILE_SIZE * REFERENCE_TILE_SIZE;
         tile_index += WORK_GROUP_SIZE * WORK_GROUP_SIZE)
    {
        int tile_x = tile_index % REFERENCE_TILE_SIZE;
        int tile_y = tile_index / REFERENCE_TILE_SIZE;
        int image_x = tile_origin_x + tile_x - MAX_PATCH_RADIUS;
        int image_y = tile_origin_y + tile_y - MAX_PATCH_RADIUS;
        int inside = image_x >= 0 && image_y >= 0 && image_x < width && image_y < height;
        int image_index = inside ? image_y * width + image_x : 0;
        reference_tile[tile_index] = inside ? reference[image_index] : 0.0f;
        reference_mask_tile[tile_index] = inside
            ? (has_reference_mask ? reference_mask[image_index] : (uchar)255)
            : (uchar)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x >= width || y >= height)
    {
        return;
    }
    int index = y * width + x;
    float best_depth = depth_output[index];
    float best_score = score_output[index];
    float4 best_normal = normal_output[index];
    if (!(best_depth > 0.0f) || !(best_score > 0.0f))
    {
        depth_output[index] = 0.0f;
        score_output[index] = 0.0f;
        return;
    }

    float confidence = best_score;
    if (uniqueness_minimum_margin > 0.0f)
    {
        float lower_depth = fmax(
            z_near, best_depth * (1.0f - uniqueness_relative_step));
        float upper_depth = fmin(
            z_far, best_depth * (1.0f + uniqueness_relative_step));
        float minimum_distinct_depth = fmax(
            1.0e-6f, best_depth * uniqueness_relative_step * 0.25f);
        float lower_score = 0.0f;
        if (best_depth - lower_depth >= minimum_distinct_depth)
        {
            lower_score = robust_depth_score(
                x, y, lower_depth, best_normal,
                reference_tile, reference_mask_tile,
                sources, source_cameras, source_masks,
                width, height, source_count, patch_half, 1,
                minimum_mask_ratio, has_reference_mask, has_source_masks,
                inv_fx, inv_fy, cx, cy, tile_origin_x, tile_origin_y,
                REFERENCE_TILE_SIZE);
        }
        float upper_score = 0.0f;
        if (upper_depth - best_depth >= minimum_distinct_depth)
        {
            upper_score = robust_depth_score(
                x, y, upper_depth, best_normal,
                reference_tile, reference_mask_tile,
                sources, source_cameras, source_masks,
                width, height, source_count, patch_half, 1,
                minimum_mask_ratio, has_reference_mask, has_source_masks,
                inv_fx, inv_fy, cx, cy, tile_origin_x, tile_origin_y,
                REFERENCE_TILE_SIZE);
        }
        float competing_score = fmax(lower_score, upper_score);
        float margin_scale = clamp(
            (best_score - competing_score) / uniqueness_minimum_margin,
            0.0f,
            1.0f);
        confidence *= uniqueness_minimum_scale
            + (1.0f - uniqueness_minimum_scale) * margin_scale;
    }
    depth_output[index] = confidence >= confidence_threshold ? best_depth : 0.0f;
    score_output[index] = confidence;
}
)CLC";

} // namespace detail
} // namespace mvs
} // namespace xjw
