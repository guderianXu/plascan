#include "processing/PointCloudProcessor.h"

#include "Logger.h"
#include "spatial/KDTree3D.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace xjw::pointcloud::detail
{

namespace
{

struct VoxelKey
{
    long long ix = 0;
    long long iy = 0;
    long long iz = 0;

    bool operator==(const VoxelKey &other) const
    {
        return ix == other.ix && iy == other.iy && iz == other.iz;
    }
};

struct VoxelKeyHasher
{
    std::size_t operator()(const VoxelKey &key) const
    {
        std::size_t seed = 0;
        seed ^= std::hash<long long>{}(key.ix) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<long long>{}(key.iy) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<long long>{}(key.iz) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct VoxelAccumulator
{
    Point3f positionSum{};
    Point3f normalSum{};
    float colorSum[4]{0.0f, 0.0f, 0.0f, 0.0f};
    float confidenceSum = 0.0f;
    float reprojectionErrorSum = 0.0f;
    int trackLengthMax = 0;
    bool anyControlPoint = false;
    bool anyValid = false;
    int count = 0;
};

std::vector<float> packCoords(const PointCloud &input)
{
    std::vector<float> coords(input.size() * 3, 0.0f);
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        const Point3f &point = input.positions()[index];
        coords[index * 3 + 0] = point.x;
        coords[index * 3 + 1] = point.y;
        coords[index * 3 + 2] = point.z;
    }
    return coords;
}

int resolveThreadCount(int requested)
{
    if (requested > 0)
    {
        return requested;
    }

    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    return static_cast<int>(hardwareThreads == 0 ? 4 : hardwareThreads);
}

template <typename Function>
void parallelFor(std::size_t size, int threads, Function &&function)
{
    const int workerCount = std::max(1, std::min<int>(threads, static_cast<int>(size == 0 ? 1 : size)));
    const std::size_t chunkSize = (size + static_cast<std::size_t>(workerCount) - 1) / static_cast<std::size_t>(workerCount);

    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<std::size_t>(workerCount));
    for (int workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        const std::size_t begin = static_cast<std::size_t>(workerIndex) * chunkSize;
        const std::size_t end = std::min(size, begin + chunkSize);
        if (begin >= end)
        {
            break;
        }

        futures.push_back(std::async(std::launch::async, [begin, end, &function]() {
            function(begin, end);
        }));
    }

    for (std::future<void> &future : futures)
    {
        future.get();
    }
}

PointCloud createPointCloudFromIndices(const PointCloud &input, const std::vector<std::size_t> &indices)
{
    PointCloud output;
    output.setMetadata(input.metadata());
    output.reserve(indices.size());

    for (std::size_t index : indices)
    {
        const Point3f &position = input.positions()[index];
        const Point3f *normal = input.normalAt(index);
        const ColorRGBA *color = input.colorAt(index);
        const PhotogrammetryPointAttributes *photogrammetry = input.photogrammetryAttributesAt(index);

        if (normal && color && photogrammetry)
        {
            output.addPoint(position, *normal, *color, *photogrammetry);
        }
        else if (normal && color)
        {
            output.addPoint(position, *normal, *color);
        }
        else if (normal)
        {
            output.addPoint(position, *normal);
        }
        else if (color)
        {
            output.addPoint(position, *color);
        }
        else
        {
            output.addPoint(position);
        }
    }

    return output;
}

} // namespace

PointCloud statisticalFilterMultithread(const PointCloud &input, int k, double stdMul, int threads)
{
    if (input.size() <= 8 || k <= 0)
    {
        return input;
    }

    const int workerCount = resolveThreadCount(threads);
    const int kNeighbors = std::max(1, std::min(k, static_cast<int>(input.size()) - 1));

    const std::vector<float> coords = packCoords(input);
    xjw::common::spatial::KDTree3D tree;
    tree.build(coords.data(), static_cast<int>(input.size()));

    std::vector<double> localMean(input.size(), 0.0);
    parallelFor(input.size(), workerCount, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index)
        {
            localMean[index] = static_cast<double>(tree.knnMeanDist(static_cast<int>(index), kNeighbors));
        }
    });

    const double mean = std::accumulate(localMean.begin(), localMean.end(), 0.0) / static_cast<double>(localMean.size());
    double variance = 0.0;
    for (double value : localMean)
    {
        const double diff = value - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(localMean.size());
    const double threshold = mean + stdMul * std::sqrt(std::max(1e-12, variance));

    std::vector<std::size_t> keptIndices;
    keptIndices.reserve(input.size());
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        if (localMean[index] <= threshold)
        {
            keptIndices.push_back(index);
        }
    }

    LOG_INFO("PointCloudProcessor::statisticalFilterMultithread: before=%zu after=%zu threads=%d",
             input.size(),
             keptIndices.size(),
             workerCount);
    return createPointCloudFromIndices(input, keptIndices);
}

PointCloud radiusFilterMultithread(const PointCloud &input, double radius, int minNeighbors, int threads)
{
    if (input.size() <= 8 || radius <= 0.0 || minNeighbors <= 0)
    {
        return input;
    }

    const int workerCount = resolveThreadCount(threads);
    const float radiusValue = static_cast<float>(radius);
    const int minNeighborCount = std::max(1, minNeighbors);

    const std::vector<float> coords = packCoords(input);
    xjw::common::spatial::KDTree3D tree;
    tree.build(coords.data(), static_cast<int>(input.size()));

    std::vector<bool> keep(input.size(), false);

    parallelFor(input.size(), workerCount, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index)
        {
            const int neighbors = tree.radiusCount(static_cast<int>(index), radiusValue, minNeighborCount);
            keep[index] = neighbors >= minNeighborCount;
        }
    });

    std::vector<std::size_t> keptIndices;
    keptIndices.reserve(input.size());
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        if (keep[index])
        {
            keptIndices.push_back(index);
        }
    }

    LOG_INFO("PointCloudProcessor::radiusFilterMultithread: before=%zu after=%zu threads=%d",
             input.size(),
             keptIndices.size(),
             workerCount);
    return createPointCloudFromIndices(input, keptIndices);
}

PointCloud uniformDownsample(const PointCloud &input, int step)
{
    const int actualStep = std::max(1, step);
    if (actualStep <= 1)
    {
        return input;
    }

    std::vector<std::size_t> keptIndices;
    keptIndices.reserve(input.size() / static_cast<std::size_t>(actualStep) + 1);
    for (std::size_t index = 0; index < input.size(); index += static_cast<std::size_t>(actualStep))
    {
        keptIndices.push_back(index);
    }

    LOG_INFO("PointCloudProcessor::uniformDownsample: before=%zu after=%zu step=%d",
             input.size(),
             keptIndices.size(),
             actualStep);
    return createPointCloudFromIndices(input, keptIndices);
}

PointCloud voxelDownsampleMultithread(const PointCloud &input, double voxelSize, int threads)
{
    if (input.empty() || voxelSize <= 0.0)
    {
        return input;
    }

    const int workerCount = resolveThreadCount(threads);
    using VoxelMap = std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHasher>;
    std::vector<VoxelMap> localMaps(static_cast<std::size_t>(workerCount));

    parallelFor(input.size(), workerCount, [&](std::size_t begin, std::size_t end) {
        const std::size_t workerIndex = begin / std::max<std::size_t>(1, (input.size() + static_cast<std::size_t>(workerCount) - 1) / static_cast<std::size_t>(workerCount));
        VoxelMap &voxelMap = localMaps[std::min<std::size_t>(workerIndex, localMaps.size() - 1)];
        for (std::size_t index = begin; index < end; ++index)
        {
            const Point3f &position = input.positions()[index];
            const VoxelKey key{
                static_cast<long long>(std::floor(static_cast<double>(position.x) / voxelSize)),
                static_cast<long long>(std::floor(static_cast<double>(position.y) / voxelSize)),
                static_cast<long long>(std::floor(static_cast<double>(position.z) / voxelSize))};

            VoxelAccumulator &accumulator = voxelMap[key];
            accumulator.positionSum.x += position.x;
            accumulator.positionSum.y += position.y;
            accumulator.positionSum.z += position.z;

            if (const Point3f *normal = input.normalAt(index))
            {
                accumulator.normalSum.x += normal->x;
                accumulator.normalSum.y += normal->y;
                accumulator.normalSum.z += normal->z;
            }

            if (const ColorRGBA *color = input.colorAt(index))
            {
                accumulator.colorSum[0] += static_cast<float>(color->r);
                accumulator.colorSum[1] += static_cast<float>(color->g);
                accumulator.colorSum[2] += static_cast<float>(color->b);
                accumulator.colorSum[3] += static_cast<float>(color->a);
            }

            if (const PhotogrammetryPointAttributes *photogrammetry = input.photogrammetryAttributesAt(index))
            {
                accumulator.confidenceSum += photogrammetry->confidence;
                accumulator.reprojectionErrorSum += photogrammetry->reprojectionError;
                accumulator.trackLengthMax = std::max(accumulator.trackLengthMax, photogrammetry->trackLength);
                accumulator.anyControlPoint = accumulator.anyControlPoint || photogrammetry->isControlPoint;
                accumulator.anyValid = accumulator.anyValid || photogrammetry->isValid;
            }

            ++accumulator.count;
        }
    });

    VoxelMap merged;
    for (VoxelMap &localMap : localMaps)
    {
        for (auto &entry : localMap)
        {
            VoxelAccumulator &target = merged[entry.first];
            const VoxelAccumulator &source = entry.second;
            target.positionSum.x += source.positionSum.x;
            target.positionSum.y += source.positionSum.y;
            target.positionSum.z += source.positionSum.z;
            target.normalSum.x += source.normalSum.x;
            target.normalSum.y += source.normalSum.y;
            target.normalSum.z += source.normalSum.z;
            for (int channel = 0; channel < 4; ++channel)
            {
                target.colorSum[channel] += source.colorSum[channel];
            }
            target.confidenceSum += source.confidenceSum;
            target.reprojectionErrorSum += source.reprojectionErrorSum;
            target.trackLengthMax = std::max(target.trackLengthMax, source.trackLengthMax);
            target.anyControlPoint = target.anyControlPoint || source.anyControlPoint;
            target.anyValid = target.anyValid || source.anyValid;
            target.count += source.count;
        }
    }

    PointCloud output;
    output.setMetadata(input.metadata());
    output.reserve(merged.size());

    for (const auto &entry : merged)
    {
        const VoxelAccumulator &accumulator = entry.second;
        if (accumulator.count <= 0)
        {
            continue;
        }

        const float inverseCount = 1.0f / static_cast<float>(accumulator.count);
        const Point3f position{
            accumulator.positionSum.x * inverseCount,
            accumulator.positionSum.y * inverseCount,
            accumulator.positionSum.z * inverseCount};

        if (input.hasNormals() || input.hasColors() || input.hasPhotogrammetryAttributes())
        {
            const Point3f normal{
                accumulator.normalSum.x * inverseCount,
                accumulator.normalSum.y * inverseCount,
                accumulator.normalSum.z * inverseCount};
            const ColorRGBA color{
                static_cast<uint8_t>(std::clamp(accumulator.colorSum[0] * inverseCount, 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(accumulator.colorSum[1] * inverseCount, 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(accumulator.colorSum[2] * inverseCount, 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(accumulator.colorSum[3] * inverseCount, 0.0f, 255.0f))};
            const PhotogrammetryPointAttributes photogrammetry{
                -1,
                accumulator.trackLengthMax,
                accumulator.reprojectionErrorSum * inverseCount,
                accumulator.confidenceSum * inverseCount,
                accumulator.anyControlPoint,
                accumulator.anyValid};

            if (input.hasNormals() && input.hasColors() && input.hasPhotogrammetryAttributes())
            {
                output.addPoint(position, normal, color, photogrammetry);
            }
            else if (input.hasNormals() && input.hasColors())
            {
                output.addPoint(position, normal, color);
                if (input.hasPhotogrammetryAttributes())
                {
                    output.setPhotogrammetryAttributes(output.size() - 1, photogrammetry);
                }
            }
            else if (input.hasNormals())
            {
                output.addPoint(position, normal);
                if (input.hasPhotogrammetryAttributes())
                {
                    output.setPhotogrammetryAttributes(output.size() - 1, photogrammetry);
                }
            }
            else if (input.hasColors())
            {
                output.addPoint(position, color);
                if (input.hasPhotogrammetryAttributes())
                {
                    output.setPhotogrammetryAttributes(output.size() - 1, photogrammetry);
                }
            }
            else
            {
                output.addPoint(position);
                if (input.hasPhotogrammetryAttributes())
                {
                    output.setPhotogrammetryAttributes(output.size() - 1, photogrammetry);
                }
            }
        }
        else
        {
            output.addPoint(position);
        }
    }

    LOG_INFO("PointCloudProcessor::voxelDownsampleMultithread: before=%zu after=%zu threads=%d",
             input.size(),
             output.size(),
             workerCount);
    return output;
}

} // namespace xjw::pointcloud::detail