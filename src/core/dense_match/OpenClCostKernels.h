#pragma once

namespace xjw::dense_match
{

    inline constexpr const char* kDenseMatchOpenClSource = R"CLC(
#define INVALID_COST 1.0e20f

inline size_t image_offset(int x, int y, int image_width)
{
    return (size_t)y * (size_t)image_width + (size_t)x;
}

inline int right_image_x(int left_x, int disparity, int image_width)
{
    const long shifted_x = (long)left_x - (long)disparity;
    return shifted_x >= 0L && shifted_x < (long)image_width ? (int)shifted_x : -1;
}

inline float ad_cost(__global const uchar *left,
                     __global const uchar *right,
                     int x,
                     int y,
                     int disparity,
                     int kernel_width,
                     int kernel_height,
                     int image_width,
                     int image_height)
{
    float sum = 0.0f;
    int count = 0;
    const int half_width = kernel_width / 2;
    const int half_height = kernel_height / 2;
    for (int dy = -half_height; dy <= half_height; ++dy)
    {
        const int sample_y = y + dy;
        if (sample_y < 0 || sample_y >= image_height)
        {
            continue;
        }
        for (int dx = -half_width; dx <= half_width; ++dx)
        {
            const int left_x = x + dx;
            const int right_x = right_image_x(left_x, disparity, image_width);
            if (left_x < 0 || left_x >= image_width
                || right_x < 0 || right_x >= image_width)
            {
                continue;
            }
            sum += fabs((float)left[image_offset(left_x, sample_y, image_width)]
                        - (float)right[image_offset(right_x, sample_y, image_width)]);
            ++count;
        }
    }
    return count > 0 ? sum / (float)count : 0.0f;
}

inline float sd_cost(__global const uchar *left,
                     __global const uchar *right,
                     int x,
                     int y,
                     int disparity,
                     int kernel_width,
                     int kernel_height,
                     int image_width,
                     int image_height)
{
    float sum = 0.0f;
    int count = 0;
    const int half_width = kernel_width / 2;
    const int half_height = kernel_height / 2;
    for (int dy = -half_height; dy <= half_height; ++dy)
    {
        const int sample_y = y + dy;
        if (sample_y < 0 || sample_y >= image_height)
        {
            continue;
        }
        for (int dx = -half_width; dx <= half_width; ++dx)
        {
            const int left_x = x + dx;
            const int right_x = right_image_x(left_x, disparity, image_width);
            if (left_x < 0 || left_x >= image_width
                || right_x < 0 || right_x >= image_width)
            {
                continue;
            }
            const float difference =
                (float)left[image_offset(left_x, sample_y, image_width)]
                - (float)right[image_offset(right_x, sample_y, image_width)];
            sum += difference * difference;
            ++count;
        }
    }
    return count > 0 ? sum / (float)count : 0.0f;
}

inline float ncc_cost(__global const uchar *left,
                      __global const uchar *right,
                      int x,
                      int y,
                      int disparity,
                      int kernel_width,
                      int kernel_height,
                      int image_width,
                      int image_height)
{
    const int half_width = kernel_width / 2;
    const int half_height = kernel_height / 2;
    float mean_left = 0.0f;
    float mean_right = 0.0f;
    int count = 0;
    for (int dy = -half_height; dy <= half_height; ++dy)
    {
        const int sample_y = y + dy;
        if (sample_y < 0 || sample_y >= image_height)
        {
            continue;
        }
        for (int dx = -half_width; dx <= half_width; ++dx)
        {
            const int left_x = x + dx;
            const int right_x = right_image_x(left_x, disparity, image_width);
            if (left_x < 0 || left_x >= image_width
                || right_x < 0 || right_x >= image_width)
            {
                continue;
            }
            mean_left += (float)left[image_offset(left_x, sample_y, image_width)];
            mean_right += (float)right[image_offset(right_x, sample_y, image_width)];
            ++count;
        }
    }
    if (count == 0)
    {
        return 0.0f;
    }
    mean_left /= (float)count;
    mean_right /= (float)count;

    float covariance = 0.0f;
    float variance_left = 0.0f;
    float variance_right = 0.0f;
    for (int dy = -half_height; dy <= half_height; ++dy)
    {
        const int sample_y = y + dy;
        if (sample_y < 0 || sample_y >= image_height)
        {
            continue;
        }
        for (int dx = -half_width; dx <= half_width; ++dx)
        {
            const int left_x = x + dx;
            const int right_x = right_image_x(left_x, disparity, image_width);
            if (left_x < 0 || left_x >= image_width
                || right_x < 0 || right_x >= image_width)
            {
                continue;
            }
            const float left_delta =
                (float)left[image_offset(left_x, sample_y, image_width)] - mean_left;
            const float right_delta =
                (float)right[image_offset(right_x, sample_y, image_width)] - mean_right;
            covariance += left_delta * right_delta;
            variance_left += left_delta * left_delta;
            variance_right += right_delta * right_delta;
        }
    }
    const float std_left = sqrt(variance_left);
    const float std_right = sqrt(variance_right);
    if (std_left < 1.0e-8f && std_right < 1.0e-8f)
    {
        return fabs(mean_left - mean_right) < 1.0e-8f ? 0.0f : 2.0f;
    }
    if (std_left < 1.0e-8f || std_right < 1.0e-8f)
    {
        return 1.0f;
    }
    return 1.0f - clamp(covariance / (std_left * std_right), -1.0f, 1.0f);
}

inline float census_cost(__global const uchar *left,
                         __global const uchar *right,
                         int x,
                         int y,
                         int disparity,
                         int kernel_width,
                         int kernel_height,
                         int image_width,
                         int image_height)
{
    const int half_width = kernel_width / 2;
    const int half_height = kernel_height / 2;
    const uchar center_left = left[image_offset(x, y, image_width)];
    const int center_right_x = right_image_x(x, disparity, image_width);
    if (center_right_x < 0)
    {
        return INVALID_COST;
    }
    const uchar center_right = right[image_offset(center_right_x, y, image_width)];
    int hamming = 0;
    int count = 0;
    for (int dy = -half_height; dy <= half_height; ++dy)
    {
        const int sample_y = y + dy;
        if (sample_y < 0 || sample_y >= image_height)
        {
            continue;
        }
        for (int dx = -half_width; dx <= half_width; ++dx)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }
            const int left_x = x + dx;
            const int right_x = right_image_x(left_x, disparity, image_width);
            if (left_x < 0 || left_x >= image_width
                || right_x < 0 || right_x >= image_width)
            {
                continue;
            }
            const int left_bit = left[image_offset(left_x, sample_y, image_width)] > center_left;
            const int right_bit = right[image_offset(right_x, sample_y, image_width)] > center_right;
            hamming += left_bit != right_bit;
            ++count;
        }
    }
    return count > 0 ? (float)hamming / (float)count : INVALID_COST;
}

inline float ternary_census_cost(__global const uchar *left,
                                 __global const uchar *right,
                                 int x,
                                 int y,
                                 int disparity,
                                 int kernel_width,
                                 int kernel_height,
                                 int image_width,
                                 int image_height)
{
    const int half_width = kernel_width / 2;
    const int half_height = kernel_height / 2;
    const int center_left = (int)left[image_offset(x, y, image_width)];
    const int center_right_x = right_image_x(x, disparity, image_width);
    if (center_right_x < 0)
    {
        return INVALID_COST;
    }
    const int center_right = (int)right[image_offset(center_right_x, y, image_width)];
    int hamming = 0;
    int count = 0;
    for (int dy = -half_height; dy <= half_height; ++dy)
    {
        const int sample_y = y + dy;
        if (sample_y < 0 || sample_y >= image_height)
        {
            continue;
        }
        for (int dx = -half_width; dx <= half_width; ++dx)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }
            const int left_x = x + dx;
            const int right_x = right_image_x(left_x, disparity, image_width);
            if (left_x < 0 || left_x >= image_width
                || right_x < 0 || right_x >= image_width)
            {
                continue;
            }
            const int left_difference =
                (int)left[image_offset(left_x, sample_y, image_width)] - center_left;
            const int right_difference =
                (int)right[image_offset(right_x, sample_y, image_width)] - center_right;
            const int left_state = left_difference > 5 ? 1 : (left_difference < -5 ? 0 : 2);
            const int right_state = right_difference > 5 ? 1 : (right_difference < -5 ? 0 : 2);
            if (left_state != 2 && right_state != 2)
            {
                hamming += left_state != right_state;
                ++count;
            }
        }
    }
    return count > 0 ? (float)hamming / (float)count : INVALID_COST;
}

__kernel void compute_cost_volume(__global const uchar *left,
                                  __global const uchar *right,
                                  __global float *cost_volume,
                                  int image_width,
                                  int image_height,
                                  int min_disparity,
                                  int num_disparities,
                                  int kernel_width,
                                  int kernel_height,
                                  int cost_function)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    const int disparity_index = (int)get_global_id(2);
    if (x >= image_width || y >= image_height || disparity_index >= num_disparities)
    {
        return;
    }

    const int disparity = min_disparity + disparity_index;
    const int right_x = right_image_x(x, disparity, image_width);
    float cost = INVALID_COST;
    if (right_x >= 0 && right_x < image_width)
    {
        switch (cost_function)
        {
        case 0:
            cost = ad_cost(left, right, x, y, disparity, kernel_width, kernel_height,
                           image_width, image_height);
            break;
        case 1:
            cost = sd_cost(left, right, x, y, disparity, kernel_width, kernel_height,
                           image_width, image_height);
            break;
        case 2:
            cost = ncc_cost(left, right, x, y, disparity, kernel_width, kernel_height,
                            image_width, image_height);
            break;
        case 3:
            cost = census_cost(left, right, x, y, disparity, kernel_width, kernel_height,
                               image_width, image_height);
            break;
        case 4:
            cost = ternary_census_cost(left, right, x, y, disparity, kernel_width,
                                       kernel_height, image_width, image_height);
            break;
        }
    }
    const size_t plane_stride = (size_t)image_width * (size_t)image_height;
    cost_volume[(size_t)disparity_index * plane_stride
                + image_offset(x, y, image_width)] = cost;
}

__kernel void select_cost_volume(__global const float *cost_volume,
                                 __global float *disparity,
                                 __global float *confidence,
                                 __global uchar *valid_mask,
                                 int image_width,
                                 int image_height,
                                 int min_disparity,
                                 int num_disparities,
                                 int subpixel_mode)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= image_width || y >= image_height)
    {
        return;
    }
    const size_t plane_stride = (size_t)image_width * (size_t)image_height;
    const size_t pixel_offset = image_offset(x, y, image_width);
    float best_cost = INVALID_COST;
    float second_best_cost = INVALID_COST;
    int best_index = -1;
    int candidate_count = 0;
    for (int disparity_index = 0; disparity_index < num_disparities; ++disparity_index)
    {
        const float cost = cost_volume[(size_t)disparity_index * plane_stride + pixel_offset];
        if (!isfinite(cost) || cost >= INVALID_COST)
        {
            continue;
        }
        ++candidate_count;
        if (cost < best_cost)
        {
            second_best_cost = best_cost;
            best_cost = cost;
            best_index = disparity_index;
        }
        else if (cost < second_best_cost)
        {
            second_best_cost = cost;
        }
    }

    disparity[pixel_offset] = 0.0f;
    confidence[pixel_offset] = 0.0f;
    valid_mask[pixel_offset] = (uchar)0;
    if (best_index < 0)
    {
        return;
    }

    float selected_confidence = 1.0f;
    if (candidate_count > 1)
    {
        const float scale = fmax(1.0f, fmax(fabs(best_cost), fabs(second_best_cost)));
        const float margin = second_best_cost - best_cost;
        if (margin <= 1.0e-6f * scale)
        {
            return;
        }
        selected_confidence = clamp(
            margin / fmax(fabs(second_best_cost), 1.0e-6f), 0.0f, 1.0f);
    }

    float selected_disparity = (float)(min_disparity + best_index);
    if (subpixel_mode == 1 && best_index > 0 && best_index + 1 < num_disparities)
    {
        const float previous_cost = cost_volume[
            (size_t)(best_index - 1) * plane_stride + pixel_offset];
        const float next_cost = cost_volume[
            (size_t)(best_index + 1) * plane_stride + pixel_offset];
        if (isfinite(previous_cost) && previous_cost < INVALID_COST
            && isfinite(next_cost) && next_cost < INVALID_COST)
        {
            const float denominator =
                2.0f * (previous_cost + next_cost - 2.0f * best_cost);
            if (denominator > 1.0e-10f)
            {
                selected_disparity += clamp(
                    (previous_cost - next_cost) / denominator, -1.0f, 1.0f);
            }
        }
    }

    disparity[pixel_offset] = selected_disparity;
    confidence[pixel_offset] = selected_confidence;
    valid_mask[pixel_offset] = (uchar)1;
}
)CLC";

} // namespace xjw::dense_match
