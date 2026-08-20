#pragma once

namespace xjw::image_matching
{

    inline constexpr const char* kSiftMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct SiftCandidate
{
    float x;
    float y;
    float size;
    float angle;
    float response;
    uint octave;
    uint layer;
    uint padding;
};

struct NearestMatch
{
    int index;
    float similarity;
    float ambiguity;
    float padding;
};

kernel void convert_u8(device const uchar* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint index [[thread_position_in_grid]])
{
    if (index < count)
    {
        output[index] = float(input[index]) / 255.0f;
    }
}

kernel void gaussian_horizontal(device const float* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& height [[buffer(3)]],
                                constant float& sigma [[buffer(4)]],
                                constant int& radius [[buffer(5)]],
                                uint2 position [[thread_position_in_grid]])
{
    if (position.x >= width || position.y >= height)
    {
        return;
    }
    float sum = 0.0f;
    float weightSum = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset)
    {
        const int x = clamp(int(position.x) + offset, 0, int(width) - 1);
        const float weight = exp(-float(offset * offset) / (2.0f * sigma * sigma));
        sum += input[position.y * width + uint(x)] * weight;
        weightSum += weight;
    }
    output[position.y * width + position.x] = sum / max(weightSum, 1.0e-12f);
}

kernel void gaussian_vertical(device const float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              constant uint& width [[buffer(2)]],
                              constant uint& height [[buffer(3)]],
                              constant float& sigma [[buffer(4)]],
                              constant int& radius [[buffer(5)]],
                              uint2 position [[thread_position_in_grid]])
{
    if (position.x >= width || position.y >= height)
    {
        return;
    }
    float sum = 0.0f;
    float weightSum = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset)
    {
        const int y = clamp(int(position.y) + offset, 0, int(height) - 1);
        const float weight = exp(-float(offset * offset) / (2.0f * sigma * sigma));
        sum += input[uint(y) * width + position.x] * weight;
        weightSum += weight;
    }
    output[position.y * width + position.x] = sum / max(weightSum, 1.0e-12f);
}

kernel void downsample_half(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& inputWidth [[buffer(2)]],
                            constant uint& outputWidth [[buffer(3)]],
                            constant uint& outputHeight [[buffer(4)]],
                            uint2 position [[thread_position_in_grid]])
{
    if (position.x < outputWidth && position.y < outputHeight)
    {
        output[position.y * outputWidth + position.x] =
            input[(position.y * 2u) * inputWidth + position.x * 2u];
    }
}

kernel void difference(device const float* low [[buffer(0)]],
                       device const float* high [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       uint index [[thread_position_in_grid]])
{
    if (index < count)
    {
        output[index] = high[index] - low[index];
    }
}

kernel void detect_extrema(device const float* previous [[buffer(0)]],
                           device const float* current [[buffer(1)]],
                           device const float* next [[buffer(2)]],
                           device SiftCandidate* candidates [[buffer(3)]],
                           device atomic_uint* candidateCount [[buffer(4)]],
                           constant uint& width [[buffer(5)]],
                           constant uint& height [[buffer(6)]],
                           constant uint& octave [[buffer(7)]],
                           constant uint& layer [[buffer(8)]],
                           constant float& threshold [[buffer(9)]],
                           constant uint& capacity [[buffer(10)]],
                           uint2 position [[thread_position_in_grid]])
{
    if (position.x < 2u || position.y < 2u ||
        position.x + 2u >= width || position.y + 2u >= height)
    {
        return;
    }
    const uint index = position.y * width + position.x;
    const float value = current[index];
    if (abs(value) < threshold)
    {
        return;
    }

    bool maximum = true;
    bool minimum = true;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const uint neighbor = uint(int(position.y) + dy) * width + uint(int(position.x) + dx);
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

    const uint outputIndex = atomic_fetch_add_explicit(candidateCount, 1u, memory_order_relaxed);
    if (outputIndex >= capacity)
    {
        return;
    }
    const float scale = exp2(float(layer) / 3.0f);
    SiftCandidate candidate;
    candidate.x = float(position.x);
    candidate.y = float(position.y);
    candidate.size = 3.2f * scale;
    candidate.angle = 0.0f;
    candidate.response = abs(value);
    candidate.octave = octave;
    candidate.layer = layer;
    candidate.padding = 0u;
    candidates[outputIndex] = candidate;
}

kernel void assign_orientation(device const float* image [[buffer(0)]],
                               device SiftCandidate* candidates [[buffer(1)]],
                               device atomic_uint* candidateCount [[buffer(2)]],
                               constant uint& width [[buffer(3)]],
                               constant uint& height [[buffer(4)]],
                               constant uint& capacity [[buffer(5)]],
                               uint index [[thread_position_in_grid]])
{
    const uint count = min(atomic_load_explicit(candidateCount, memory_order_relaxed), capacity);
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
    const float sigma = max(1.0f, candidate.size / 3.2f);
    const int radius = min(16, max(3, int(4.5f * sigma)));
    const float window = 2.25f * sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int y = int(candidate.y) + dy;
        if (y <= 0 || y + 1 >= int(height))
        {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int x = int(candidate.x) + dx;
            if (x <= 0 || x + 1 >= int(width))
            {
                continue;
            }
            const float gx = image[uint(y) * width + uint(x + 1)] - image[uint(y) * width + uint(x - 1)];
            const float gy = image[uint(y + 1) * width + uint(x)] - image[uint(y - 1) * width + uint(x)];
            float angle = atan2(gy, gx);
            if (angle < 0.0f)
            {
                angle += 2.0f * M_PI_F;
            }
            const uint bin = min(35u, uint(angle * (36.0f / (2.0f * M_PI_F))));
            histogram[bin] += length(float2(gx, gy)) * exp(-float(dx * dx + dy * dy) / window);
        }
    }
    uint bestBin = 0u;
    for (uint bin = 1u; bin < 36u; ++bin)
    {
        if (histogram[bin] > histogram[bestBin])
        {
            bestBin = bin;
        }
    }
    candidate.angle = float(bestBin) * 10.0f;
    candidates[index] = candidate;
}

kernel void make_descriptor(device const float* image [[buffer(0)]],
                            device const SiftCandidate* candidates [[buffer(1)]],
                            device atomic_uint* candidateCount [[buffer(2)]],
                            device float* descriptors [[buffer(3)]],
                            constant uint& width [[buffer(4)]],
                            constant uint& height [[buffer(5)]],
                            constant uint& capacity [[buffer(6)]],
                            uint index [[thread_position_in_grid]])
{
    const uint count = min(atomic_load_explicit(candidateCount, memory_order_relaxed), capacity);
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
    const float scale = max(1.0f, candidate.size / 3.2f);
    const int radius = min(32, max(8, int(8.0f * scale)));
    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int y = int(candidate.y) + dy;
        if (y <= 0 || y + 1 >= int(height))
        {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int x = int(candidate.x) + dx;
            if (x <= 0 || x + 1 >= int(width))
            {
                continue;
            }
            const float binX = (cosine * float(dx) + sine * float(dy)) / (4.0f * scale) + 1.5f;
            const float binY = (-sine * float(dx) + cosine * float(dy)) / (4.0f * scale) + 1.5f;
            const int firstX = int(floor(binX));
            const int firstY = int(floor(binY));
            if (firstX < -1 || firstX >= 4 || firstY < -1 || firstY >= 4)
            {
                continue;
            }
            const float gx = image[uint(y) * width + uint(x + 1)] - image[uint(y) * width + uint(x - 1)];
            const float gy = image[uint(y + 1) * width + uint(x)] - image[uint(y - 1) * width + uint(x)];
            float angle = atan2(gy, gx) - theta;
            if (angle < 0.0f)
            {
                angle += 2.0f * M_PI_F;
            }
            const float orientation = angle * (8.0f / (2.0f * M_PI_F));
            const int firstOrientation = int(floor(orientation));
            const float normalizedX = float(dx) / (8.0f * scale);
            const float normalizedY = float(dy) / (8.0f * scale);
            const float weight = exp(-(normalizedX * normalizedX + normalizedY * normalizedY) * 2.0f);
            const float magnitude = length(float2(gx, gy)) * weight;
            for (int yOffset = 0; yOffset <= 1; ++yOffset)
            {
                const int cellY = firstY + yOffset;
                if (cellY < 0 || cellY >= 4)
                {
                    continue;
                }
                const float yWeight = yOffset == 0 ? 1.0f - (binY - float(firstY)) : binY - float(firstY);
                for (int xOffset = 0; xOffset <= 1; ++xOffset)
                {
                    const int cellX = firstX + xOffset;
                    if (cellX < 0 || cellX >= 4)
                    {
                        continue;
                    }
                    const float xWeight = xOffset == 0 ? 1.0f - (binX - float(firstX)) : binX - float(firstX);
                    for (int orientationOffset = 0; orientationOffset <= 1; ++orientationOffset)
                    {
                        const int orientationBin = (firstOrientation + orientationOffset + 8) % 8;
                        const float orientationWeight = orientationOffset == 0
                            ? 1.0f - (orientation - floor(orientation))
                            : orientation - floor(orientation);
                        const uint bin = uint((cellY * 4 + cellX) * 8 + orientationBin);
                        histogram[bin] += magnitude * xWeight * yWeight * orientationWeight;
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
    norm = rsqrt(max(norm, 1.0e-12f));
    float clippedNorm = 0.0f;
    for (uint bin = 0u; bin < 128u; ++bin)
    {
        histogram[bin] = min(0.2f, histogram[bin] * norm);
        clippedNorm += histogram[bin] * histogram[bin];
    }
    clippedNorm = rsqrt(max(clippedNorm, 1.0e-12f));
    for (uint bin = 0u; bin < 128u; ++bin)
    {
        descriptors[index * 128u + bin] = histogram[bin] * clippedNorm;
    }
}

kernel void nearest_match(device const float* query [[buffer(0)]],
                          device const float* train [[buffer(1)]],
                          device NearestMatch* output [[buffer(2)]],
                          constant uint& queryCount [[buffer(3)]],
                          constant uint& trainCount [[buffer(4)]],
                          uint queryIndex [[thread_position_in_grid]])
{
    if (queryIndex >= queryCount)
    {
        return;
    }
    float best = -INFINITY;
    float second = -INFINITY;
    int bestIndex = -1;
    for (uint trainIndex = 0u; trainIndex < trainCount; ++trainIndex)
    {
        float similarity = 0.0f;
        for (uint dimension = 0u; dimension < 128u; ++dimension)
        {
            similarity += query[queryIndex * 128u + dimension] * train[trainIndex * 128u + dimension];
        }
        if (similarity > best)
        {
            second = best;
            best = similarity;
            bestIndex = int(trainIndex);
        }
        else if (similarity > second)
        {
            second = similarity;
        }
    }
    const float bestDistance = sqrt(max(0.0f, 2.0f - 2.0f * best));
    const float secondDistance = sqrt(max(1.0e-12f, 2.0f - 2.0f * second));
    NearestMatch match;
    match.index = bestIndex;
    match.similarity = clamp(best, 0.0f, 1.0f);
    match.ambiguity = clamp(bestDistance / secondDistance, 0.0f, 1.0f);
    match.padding = 0.0f;
    output[queryIndex] = match;
}
)METAL";

} // namespace xjw::image_matching
