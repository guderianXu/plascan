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

)OPENCL"
                                                     R"OPENCL(
inline void insert_nearest_candidate(const float similarity,
                                     const int index,
                                     __private float* best,
                                     __private int* best_index,
                                     __private float* second,
                                     __private int* second_index)
{
    if (index < 0)
    {
        return;
    }
    if (*best_index < 0 || similarity > *best || (similarity == *best && index < *best_index))
    {
        if (index != *best_index)
        {
            *second = *best;
            *second_index = *best_index;
        }
        *best = similarity;
        *best_index = index;
    }
    else if (index != *best_index &&
             (*second_index < 0 || similarity > *second ||
              (similarity == *second && index < *second_index)))
    {
        *second = similarity;
        *second_index = index;
    }
}

typedef struct
{
    int best_index;
    int second_index;
    float best;
    float second;
} PartialNearestMatch;

__kernel void nearest_match_tiles(__global const float* query,
                                  __global const float* train,
                                  __global PartialNearestMatch* partial_output,
                                  const uint query_count,
                                  const uint train_count,
                                  const uint query_offset,
                                  const uint query_batch_count,
                                  const uint train_tile_count)
{
    const uint query_lane = get_local_id(0);
    const uint train_lane = get_local_id(1);
    const uint local_index = query_lane * 16u + train_lane;
    const uint query_local_index = get_group_id(0) * 16u + query_lane;
    const uint query_index = query_offset + query_local_index;
    const uint train_tile_index = get_group_id(1);
    const uint train_tile_begin = train_tile_index * 1024u;
    const uint train_tile_end = min(train_tile_begin + 1024u, train_count);
    __local float query_tile[16u * 128u];
    __local float train_tile[16u * 128u];
    __local float group_best[16u * 16u];
    __local float group_second[16u * 16u];
    __local int group_best_index[16u * 16u];
    __local int group_second_index[16u * 16u];

    for (uint index = local_index; index < 16u * 128u; index += 256u)
    {
        const uint row = index / 128u;
        const uint dimension = index % 128u;
        const uint source_query = query_offset + get_group_id(0) * 16u + row;
        query_tile[index] = source_query < query_count ? query[source_query * 128u + dimension] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float best = -INFINITY;
    float second = -INFINITY;
    int best_index = -1;
    int second_index = -1;
    for (uint train_block = train_tile_begin; train_block < train_tile_end; train_block += 16u)
    {
        for (uint index = local_index; index < 16u * 128u; index += 256u)
        {
            const uint row = index / 128u;
            const uint dimension = index % 128u;
            const uint source_train = train_block + row;
            train_tile[index] = source_train < train_tile_end ? train[source_train * 128u + dimension] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const uint train_index = train_block + train_lane;
        if (query_local_index < query_batch_count && query_index < query_count && train_index < train_tile_end)
        {
            float similarity = 0.0f;
            for (uint vector_index = 0u; vector_index < 32u; ++vector_index)
            {
                const float4 query_values = vload4(vector_index, query_tile + query_lane * 128u);
                const float4 train_values = vload4(vector_index, train_tile + train_lane * 128u);
                similarity += dot(query_values, train_values);
            }
            insert_nearest_candidate(
                similarity, (int)train_index, &best, &best_index, &second, &second_index);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    group_best[local_index] = best;
    group_second[local_index] = second;
    group_best_index[local_index] = best_index;
    group_second_index[local_index] = second_index;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 8u; stride > 0u; stride /= 2u)
    {
        if (train_lane < stride)
        {
            const uint other_index = local_index + stride;
            best = group_best[local_index];
            second = group_second[local_index];
            best_index = group_best_index[local_index];
            second_index = group_second_index[local_index];
            insert_nearest_candidate(group_best[other_index],
                                     group_best_index[other_index],
                                     &best,
                                     &best_index,
                                     &second,
                                     &second_index);
            insert_nearest_candidate(group_second[other_index],
                                     group_second_index[other_index],
                                     &best,
                                     &best_index,
                                     &second,
                                     &second_index);
            group_best[local_index] = best;
            group_second[local_index] = second;
            group_best_index[local_index] = best_index;
            group_second_index[local_index] = second_index;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (train_lane == 0u && query_local_index < query_batch_count && query_index < query_count)
    {
        PartialNearestMatch partial;
        partial.best_index = group_best_index[local_index];
        partial.second_index = group_second_index[local_index];
        partial.best = group_best[local_index];
        partial.second = group_second[local_index];
        partial_output[query_local_index * train_tile_count + train_tile_index] = partial;
    }
}

__kernel void nearest_match_reduce(__global const PartialNearestMatch* partial_input,
                                   __global NearestMatch* output,
                                   const uint query_count,
                                   const uint train_tile_count,
                                   const uint query_offset)
{
    const uint lane = get_local_id(0);
    const uint query_local_index = get_group_id(0);
    const uint query_index = query_offset + query_local_index;
    __local float group_best[64];
    __local float group_second[64];
    __local int group_best_index[64];
    __local int group_second_index[64];

    float best = -INFINITY;
    float second = -INFINITY;
    int best_index = -1;
    int second_index = -1;
    for (uint tile_index = lane; tile_index < train_tile_count; tile_index += 64u)
    {
        const PartialNearestMatch partial =
            partial_input[query_local_index * train_tile_count + tile_index];
        insert_nearest_candidate(
            partial.best, partial.best_index, &best, &best_index, &second, &second_index);
        insert_nearest_candidate(
            partial.second, partial.second_index, &best, &best_index, &second, &second_index);
    }

    group_best[lane] = best;
    group_second[lane] = second;
    group_best_index[lane] = best_index;
    group_second_index[lane] = second_index;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 32u; stride > 0u; stride /= 2u)
    {
        if (lane < stride)
        {
            best = group_best[lane];
            second = group_second[lane];
            best_index = group_best_index[lane];
            second_index = group_second_index[lane];
            insert_nearest_candidate(group_best[lane + stride],
                                     group_best_index[lane + stride],
                                     &best,
                                     &best_index,
                                     &second,
                                     &second_index);
            insert_nearest_candidate(group_second[lane + stride],
                                     group_second_index[lane + stride],
                                     &best,
                                     &best_index,
                                     &second,
                                     &second_index);
            group_best[lane] = best;
            group_second[lane] = second;
            group_best_index[lane] = best_index;
            group_second_index[lane] = second_index;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0u && query_index < query_count)
    {
        const float best_distance = sqrt(fmax(0.0f, 2.0f - 2.0f * group_best[0]));
        const float second_distance = sqrt(fmax(1.0e-12f, 2.0f - 2.0f * group_second[0]));
        NearestMatch match;
        match.index = group_best_index[0];
        match.similarity = clamp(group_best[0], 0.0f, 1.0f);
        match.ambiguity = clamp(best_distance / second_distance, 0.0f, 1.0f);
        match.padding = 0.0f;
        output[query_index] = match;
    }
}
)OPENCL";

} // namespace xjw::image_matching
