#pragma once

namespace xjw::image_matching
{

    inline constexpr const char* kSiftOpenClSource = R"OPENCL(
typedef struct
{
    float x;
    float y;
    float size;
    float angle;
    float response;
    uint octave;
    uint layer;
    uint padding;
} SiftCandidate;

typedef struct
{
    int index;
    float similarity;
    float ambiguity;
    float padding;
} NearestMatch;

__kernel void convert_u8(__global const uchar* input,
                         __global float* output,
                         const uint count)
{
    const uint index = get_global_id(0);
    if (index < count)
    {
        output[index] = (float)input[index] / 255.0f;
    }
}

__kernel void gaussian_horizontal(__global const float* input,
                                  __global float* output,
                                  const uint width,
                                  const uint height,
                                  const float sigma,
                                  const int radius)
{
    const uint x0 = get_global_id(0);
    const uint y0 = get_global_id(1);
    if (x0 >= width || y0 >= height)
    {
        return;
    }
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset)
    {
        const int x = clamp((int)x0 + offset, 0, (int)width - 1);
        const float weight = exp(-(float)(offset * offset) / (2.0f * sigma * sigma));
        sum += input[y0 * width + (uint)x] * weight;
        weight_sum += weight;
    }
    output[y0 * width + x0] = sum / fmax(weight_sum, 1.0e-12f);
}

__kernel void gaussian_vertical(__global const float* input,
                                __global float* output,
                                const uint width,
                                const uint height,
                                const float sigma,
                                const int radius)
{
    const uint x0 = get_global_id(0);
    const uint y0 = get_global_id(1);
    if (x0 >= width || y0 >= height)
    {
        return;
    }
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset)
    {
        const int y = clamp((int)y0 + offset, 0, (int)height - 1);
        const float weight = exp(-(float)(offset * offset) / (2.0f * sigma * sigma));
        sum += input[(uint)y * width + x0] * weight;
        weight_sum += weight;
    }
    output[y0 * width + x0] = sum / fmax(weight_sum, 1.0e-12f);
}

__kernel void downsample_half(__global const float* input,
                              __global float* output,
                              const uint input_width,
                              const uint output_width,
                              const uint output_height)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    if (x < output_width && y < output_height)
    {
        output[y * output_width + x] = input[(y * 2u) * input_width + x * 2u];
    }
}

__kernel void difference(__global const float* low,
                         __global const float* high,
                         __global float* output,
                         const uint count)
{
    const uint index = get_global_id(0);
    if (index < count)
    {
        output[index] = high[index] - low[index];
    }
}

__kernel void detect_extrema(__global const float* previous,
                             __global const float* current,
                             __global const float* next,
                             __global SiftCandidate* candidates,
                             volatile __global uint* candidate_count,
                             const uint width,
                             const uint height,
                             const uint octave,
                             const uint layer,
                             const float threshold,
                             const uint capacity)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    if (x < 2u || y < 2u || x + 2u >= width || y + 2u >= height)
    {
        return;
    }
    const uint index = y * width + x;
    const float value = current[index];
    if (fabs(value) < threshold)
    {
        return;
    }
    int maximum = 1;
    int minimum = 1;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const uint neighbor = (uint)((int)y + dy) * width + (uint)((int)x + dx);
            if (dx != 0 || dy != 0)
            {
                maximum = maximum && value > current[neighbor];
                minimum = minimum && value < current[neighbor];
            }
            maximum = maximum && value > previous[neighbor] && value > next[neighbor];
            minimum = minimum && value < previous[neighbor] && value < next[neighbor];
        }
    }
    if (!maximum && !minimum)
    {
        return;
    }
    const float dxx = current[index + 1u] + current[index - 1u] - 2.0f * value;
    const float dyy = current[index + width] + current[index - width] - 2.0f * value;
    const float dxy = 0.25f * (current[index + width + 1u] - current[index + width - 1u] -
                               current[index - width + 1u] + current[index - width - 1u]);
    const float determinant = dxx * dyy - dxy * dxy;
    const float trace = dxx + dyy;
    if (determinant <= 0.0f || trace * trace >= 12.1f * determinant)
    {
        return;
    }
    const uint output_index = atomic_inc(candidate_count);
    if (output_index >= capacity)
    {
        return;
    }
    const float scale = exp2((float)layer / 3.0f);
    SiftCandidate candidate;
    candidate.x = (float)x;
    candidate.y = (float)y;
    candidate.size = 3.2f * scale;
    candidate.angle = 0.0f;
    candidate.response = fabs(value);
    candidate.octave = octave;
    candidate.layer = layer;
    candidate.padding = 0u;
    candidates[output_index] = candidate;
}

__kernel void assign_orientation(__global const float* image,
                                 __global SiftCandidate* candidates,
                                 volatile __global uint* candidate_count,
                                 const uint width,
                                 const uint height,
                                 const uint capacity)
{
    const uint index = get_global_id(0);
    const uint count = min(*candidate_count, capacity);
    if (index >= count)
    {
        return;
    }
    SiftCandidate candidate = candidates[index];
    float histogram[36];
    for (uint bin = 0u; bin < 36u; ++bin)
    {
        histogram[bin] = 0.0f;
    }
    const float sigma = fmax(1.0f, candidate.size / 3.2f);
    const int radius = min(16, max(3, (int)(4.5f * sigma)));
    const float window = 2.25f * sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int y = (int)candidate.y + dy;
        if (y <= 0 || y + 1 >= (int)height)
        {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int x = (int)candidate.x + dx;
            if (x <= 0 || x + 1 >= (int)width)
            {
                continue;
            }
            const float gx = image[(uint)y * width + (uint)(x + 1)] - image[(uint)y * width + (uint)(x - 1)];
            const float gy = image[(uint)(y + 1) * width + (uint)x] - image[(uint)(y - 1) * width + (uint)x];
            float angle = atan2(gy, gx);
            if (angle < 0.0f)
            {
                angle += 2.0f * M_PI_F;
            }
            const uint bin = min(35u, (uint)(angle * (36.0f / (2.0f * M_PI_F))));
            histogram[bin] += hypot(gx, gy) * exp(-(float)(dx * dx + dy * dy) / window);
        }
    }
    uint best_bin = 0u;
    for (uint bin = 1u; bin < 36u; ++bin)
    {
        if (histogram[bin] > histogram[best_bin])
        {
            best_bin = bin;
        }
    }
    candidate.angle = (float)best_bin * 10.0f;
    candidates[index] = candidate;
}

__kernel void make_descriptor(__global const float* image,
                              __global const SiftCandidate* candidates,
                              volatile __global uint* candidate_count,
                              __global float* descriptors,
                              const uint width,
                              const uint height,
                              const uint capacity)
{
    const uint index = get_global_id(0);
    const uint count = min(*candidate_count, capacity);
    if (index >= count)
    {
        return;
    }
    const SiftCandidate candidate = candidates[index];
    float histogram[128];
    for (uint bin = 0u; bin < 128u; ++bin)
    {
        histogram[bin] = 0.0f;
    }
    const float theta = candidate.angle * (M_PI_F / 180.0f);
    const float cosine = cos(theta);
    const float sine = sin(theta);
    const float scale = fmax(1.0f, candidate.size / 3.2f);
    const int radius = min(32, max(8, (int)(8.0f * scale)));
    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int y = (int)candidate.y + dy;
        if (y <= 0 || y + 1 >= (int)height)
        {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int x = (int)candidate.x + dx;
            if (x <= 0 || x + 1 >= (int)width)
            {
                continue;
            }
            const float bin_x = (cosine * (float)dx + sine * (float)dy) / (4.0f * scale) + 1.5f;
            const float bin_y = (-sine * (float)dx + cosine * (float)dy) / (4.0f * scale) + 1.5f;
            const int first_x = (int)floor(bin_x);
            const int first_y = (int)floor(bin_y);
            if (first_x < -1 || first_x >= 4 || first_y < -1 || first_y >= 4)
            {
                continue;
            }
            const float gx = image[(uint)y * width + (uint)(x + 1)] - image[(uint)y * width + (uint)(x - 1)];
            const float gy = image[(uint)(y + 1) * width + (uint)x] - image[(uint)(y - 1) * width + (uint)x];
            float angle = atan2(gy, gx) - theta;
            if (angle < 0.0f)
            {
                angle += 2.0f * M_PI_F;
            }
            const float orientation = angle * (8.0f / (2.0f * M_PI_F));
            const int first_orientation = (int)floor(orientation);
            const float normalized_x = (float)dx / (8.0f * scale);
            const float normalized_y = (float)dy / (8.0f * scale);
            const float weight = exp(-(normalized_x * normalized_x + normalized_y * normalized_y) * 2.0f);
            const float magnitude = hypot(gx, gy) * weight;
            for (int y_offset = 0; y_offset <= 1; ++y_offset)
            {
                const int cell_y = first_y + y_offset;
                if (cell_y < 0 || cell_y >= 4)
                {
                    continue;
                }
                const float y_weight = y_offset == 0 ? 1.0f - (bin_y - (float)first_y) : bin_y - (float)first_y;
                for (int x_offset = 0; x_offset <= 1; ++x_offset)
                {
                    const int cell_x = first_x + x_offset;
                    if (cell_x < 0 || cell_x >= 4)
                    {
                        continue;
                    }
                    const float x_weight = x_offset == 0 ? 1.0f - (bin_x - (float)first_x) : bin_x - (float)first_x;
                    for (int orientation_offset = 0; orientation_offset <= 1; ++orientation_offset)
                    {
                        const int orientation_bin = (first_orientation + orientation_offset + 8) % 8;
                        const float fraction = orientation - floor(orientation);
                        const float orientation_weight = orientation_offset == 0 ? 1.0f - fraction : fraction;
                        const uint bin = (uint)((cell_y * 4 + cell_x) * 8 + orientation_bin);
                        histogram[bin] += magnitude * x_weight * y_weight * orientation_weight;
                    }
                }
            }
        }
    }
    float norm = 0.0f;
    for (uint bin = 0u; bin < 128u; ++bin)
    {
        norm += histogram[bin] * histogram[bin];
    }
    norm = rsqrt(fmax(norm, 1.0e-12f));
    float clipped_norm = 0.0f;
    for (uint bin = 0u; bin < 128u; ++bin)
    {
        histogram[bin] = fmin(0.2f, histogram[bin] * norm);
        clipped_norm += histogram[bin] * histogram[bin];
    }
    clipped_norm = rsqrt(fmax(clipped_norm, 1.0e-12f));
    for (uint bin = 0u; bin < 128u; ++bin)
    {
        descriptors[index * 128u + bin] = histogram[bin] * clipped_norm;
    }
}

__kernel void nearest_match(__global const float* query,
                            __global const float* train,
                            __global NearestMatch* output,
                            const uint query_count,
                            const uint train_count)
{
    const uint query_index = get_global_id(0);
    if (query_index >= query_count)
    {
        return;
    }
    float best = -INFINITY;
    float second = -INFINITY;
    int best_index = -1;
    for (uint train_index = 0u; train_index < train_count; ++train_index)
    {
        float similarity = 0.0f;
        for (uint dimension = 0u; dimension < 128u; ++dimension)
        {
            similarity += query[query_index * 128u + dimension] * train[train_index * 128u + dimension];
        }
        if (similarity > best)
        {
            second = best;
            best = similarity;
            best_index = (int)train_index;
        }
        else if (similarity > second)
        {
            second = similarity;
        }
    }
    const float best_distance = sqrt(fmax(0.0f, 2.0f - 2.0f * best));
    const float second_distance = sqrt(fmax(1.0e-12f, 2.0f - 2.0f * second));
    NearestMatch match;
    match.index = best_index;
    match.similarity = clamp(best, 0.0f, 1.0f);
    match.ambiguity = clamp(best_distance / second_distance, 0.0f, 1.0f);
    match.padding = 0.0f;
    output[query_index] = match;
}
)OPENCL";

} // namespace xjw::image_matching
