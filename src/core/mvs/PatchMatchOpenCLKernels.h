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
#define WORK_GROUP_SIZE 16
#define MAX_PATCH_RADIUS 7
#define REFERENCE_TILE_SIZE (WORK_GROUP_SIZE + 2 * MAX_PATCH_RADIUS)

inline float sample_bilinear(__global const float *image,
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

inline int source_mask_valid(__global const uchar *masks,
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

inline float source_ncc(int center_x,
                        int center_y,
                        float depth,
                        int source_index,
                         __local const float *reference_tile,
                         __local const uchar *reference_mask_tile,
                         __global const float *sources,
                        __global const float *source_cameras,
                         __global const uchar *source_masks,
                        int width,
                        int height,
                        int patch_half,
                        float minimum_mask_ratio,
                        int has_reference_mask,
                        int has_source_masks,
                        float inv_fx,
                        float inv_fy,
                        float cx,
                         float cy,
                         int tile_origin_x,
                         int tile_origin_y)
{
    int pixel_count = width * height;
    __global const float *source = sources + source_index * pixel_count;
    __global const float *camera = source_cameras + source_index * 16;
    int radius = clamp(patch_half, 1, 7);
    int step = max(1, radius / 2);
    float sum_reference = 0.0f;
    float sum_source = 0.0f;
    float sum_reference_squared = 0.0f;
    float sum_source_squared = 0.0f;
    float sum_product = 0.0f;
    int candidate_count = 0;
    int valid_count = 0;

    for (int dy = -radius; dy <= radius; dy += step)
    {
        for (int dx = -radius; dx <= radius; dx += step)
        {
            ++candidate_count;
            int reference_x = center_x + dx;
            int reference_y = center_y + dy;
            if (reference_x < 0 || reference_y < 0
                || reference_x >= width || reference_y >= height)
            {
                continue;
            }
            int tile_x = reference_x - tile_origin_x + MAX_PATCH_RADIUS;
            int tile_y = reference_y - tile_origin_y + MAX_PATCH_RADIUS;
            int tile_index = tile_y * REFERENCE_TILE_SIZE + tile_x;
            if (has_reference_mask && reference_mask_tile[tile_index] == 0)
            {
                continue;
            }

            float normalized_x = ((float)reference_x - cx) * inv_fx;
            float normalized_y = ((float)reference_y - cy) * inv_fy;
            float transformed_x = camera[4] * normalized_x
                + camera[5] * normalized_y + camera[6] + camera[13] / depth;
            float transformed_y = camera[7] * normalized_x
                + camera[8] * normalized_y + camera[9] + camera[14] / depth;
            float transformed_z = camera[10] * normalized_x
                + camera[11] * normalized_y + camera[12] + camera[15] / depth;
            if (!(transformed_z > 1.0e-6f))
            {
                continue;
            }
            float source_x = camera[0] * transformed_x / transformed_z + camera[1];
            float source_y = camera[2] * transformed_y / transformed_z + camera[3];
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
    return clamp((covariance / sqrt(variance_product) + 1.0f) * 0.5f,
                 0.0f,
                 1.0f);
}
)CLC";

inline constexpr const char *kPatchMatchOpenClSourceMain = R"CLC(
inline float robust_depth_score(int x,
                                int y,
                                float depth,
                                __local const float *reference_tile,
                                __local const uchar *reference_mask_tile,
                                __global const float *sources,
                                __global const float *source_cameras,
                                __global const uchar *source_masks,
                                int width,
                                int height,
                                int source_count,
                                int patch_half,
                                float minimum_mask_ratio,
                                int has_reference_mask,
                                int has_source_masks,
                                float inv_fx,
                                float inv_fy,
                                float cx,
                                float cy,
                                int tile_origin_x,
                                int tile_origin_y)
{
    float scores[MAX_SOURCES];
    int support_count = 0;
    for (int source_index = 0; source_index < source_count; ++source_index)
    {
        float score = source_ncc(x, y, depth, source_index,
                                 reference_tile, reference_mask_tile,
                                 sources, source_cameras, source_masks,
                                 width, height, patch_half, minimum_mask_ratio,
                                 has_reference_mask, has_source_masks,
                                 inv_fx, inv_fy, cx, cy,
                                 tile_origin_x, tile_origin_y);
        if (score > 0.05f)
        {
            scores[support_count++] = score;
        }
    }
    int required_support = source_count <= 2 ? source_count : source_count / 2 + 1;
    if (support_count < required_support)
    {
        return 0.0f;
    }
    for (int left = 0; left < support_count; ++left)
    {
        for (int right = left + 1; right < support_count; ++right)
        {
            if (scores[right] > scores[left])
            {
                float value = scores[left];
                scores[left] = scores[right];
                scores[right] = value;
            }
        }
    }
    float sum = 0.0f;
    for (int index = 0; index < required_support; ++index)
    {
        sum += scores[index];
    }
    return sum / (float)required_support;
}

__kernel void estimate_depth(
    __global const float *reference,
    __global const float *sources,
    __global const float *source_cameras,
    __global const uchar *reference_mask,
    __global const uchar *source_masks,
    __global const float *hint_depth,
    __global const float *hint_radius,
    __global float *depth_output,
    __global float *confidence_output,
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
        confidence_output[index] = 0.0f;
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
        float score = robust_depth_score(x, y, depth,
                                         reference_tile, reference_mask_tile,
                                         sources, source_cameras, source_masks,
                                         width, height, source_count, patch_half,
                                         minimum_mask_ratio, has_reference_mask,
                                         has_source_masks, inv_fx, inv_fy, cx, cy,
                                         tile_origin_x, tile_origin_y);
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
            float score = robust_depth_score(x, y, depth,
                                             reference_tile, reference_mask_tile,
                                             sources, source_cameras, source_masks,
                                             width, height, source_count, patch_half,
                                             minimum_mask_ratio, has_reference_mask,
                                             has_source_masks, inv_fx, inv_fy, cx, cy,
                                             tile_origin_x, tile_origin_y);
            if (score > best_score)
            {
                best_score = score;
                best_depth = depth;
            }
        }
    }

    float confidence = best_score;
    if (best_depth > 0.0f && uniqueness_minimum_margin > 0.0f)
    {
        float lower_depth = clamp(best_depth * (1.0f - uniqueness_relative_step),
                                  local_near,
                                  local_far);
        float upper_depth = clamp(best_depth * (1.0f + uniqueness_relative_step),
                                  local_near,
                                  local_far);
        float competing_score = fmax(
            robust_depth_score(x, y, lower_depth,
                               reference_tile, reference_mask_tile,
                               sources, source_cameras, source_masks,
                               width, height, source_count, patch_half,
                               minimum_mask_ratio, has_reference_mask,
                               has_source_masks, inv_fx, inv_fy, cx, cy,
                               tile_origin_x, tile_origin_y),
            robust_depth_score(x, y, upper_depth,
                               reference_tile, reference_mask_tile,
                               sources, source_cameras, source_masks,
                               width, height, source_count, patch_half,
                               minimum_mask_ratio, has_reference_mask,
                               has_source_masks, inv_fx, inv_fy, cx, cy,
                               tile_origin_x, tile_origin_y));
        float margin_scale = clamp((best_score - competing_score) / uniqueness_minimum_margin,
                                   0.0f,
                                   1.0f);
        confidence *= uniqueness_minimum_scale
            + (1.0f - uniqueness_minimum_scale) * margin_scale;
    }

    depth_output[index] = confidence >= confidence_threshold ? best_depth : 0.0f;
    confidence_output[index] = confidence;
}
)CLC";

} // namespace detail
} // namespace mvs
} // namespace xjw
