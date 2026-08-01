#include "DepthTsdfNarrowBandActivation.h"

#include "Camera.h"
#include "DepthTsdfSurfaceBuilder.h"

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace xjw::mesh
{
namespace
{

bool checkedProduct(int x, int y, int z, std::size_t *product)
{
    if (!product || x <= 0 || y <= 0 || z <= 0)
    {
        return false;
    }
    const std::size_t size_x = static_cast<std::size_t>(x);
    const std::size_t size_y = static_cast<std::size_t>(y);
    const std::size_t size_z = static_cast<std::size_t>(z);
    if (size_x > std::numeric_limits<std::size_t>::max() / size_y)
    {
        return false;
    }
    const std::size_t xy = size_x * size_y;
    if (xy > std::numeric_limits<std::size_t>::max() / size_z)
    {
        return false;
    }
    *product = xy * size_z;
    return true;
}

bool validLayout(const DepthTsdfLayout &layout)
{
    if (!layout.ok)
    {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        if (layout.cells[axis] < 0 ||
            !(layout.voxelSize[axis] > 0.0f) ||
            !std::isfinite(layout.voxelSize[axis]) ||
            !std::isfinite(layout.boundsMin[axis]) ||
            !std::isfinite(layout.boundsMax[axis]) ||
            layout.boundsMax[axis] < layout.boundsMin[axis])
        {
            return false;
        }
    }
    return true;
}

bool validOptionalMask(const cv::Mat *mask, const cv::Mat &depth)
{
    return !mask ||
           (mask->type() == CV_8UC1 && mask->size() == depth.size());
}

std::size_t blockIndex(int x, int y, int z, int count_x, int count_y)
{
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(count_x) *
               (static_cast<std::size_t>(y) +
                static_cast<std::size_t>(count_y) *
                    static_cast<std::size_t>(z));
}

bool worldToSampleBlock(
    const DepthTsdfLayout &layout,
    const double world[3],
    int block_size,
    int block_count_x,
    int block_count_y,
    int block_count_z,
    std::size_t *index)
{
    int block[3]{};
    const int block_counts[3] = {
        block_count_x, block_count_y, block_count_z};
    for (int axis = 0; axis < 3; ++axis)
    {
        double coordinate =
            (world[axis] - layout.boundsMin[axis]) /
            layout.voxelSize[axis];
        constexpr double kBoundaryTolerance = 1.0e-6;
        if (!std::isfinite(coordinate) ||
            coordinate < -kBoundaryTolerance ||
            coordinate > layout.cells[axis] + kBoundaryTolerance)
        {
            return false;
        }
        coordinate = std::clamp(
            coordinate, 0.0, static_cast<double>(layout.cells[axis]));
        const int sample = std::min(
            layout.cells[axis],
            static_cast<int>(std::floor(coordinate + kBoundaryTolerance)));
        block[axis] = sample / block_size;
        if (block[axis] < 0 || block[axis] >= block_counts[axis])
        {
            return false;
        }
    }
    *index = blockIndex(
        block[0], block[1], block[2], block_count_x, block_count_y);
    return true;
}

bool isPixelUsable(
    const cv::Mat &depth,
    const cv::Mat *depth_valid,
    const cv::Mat *support,
    int row,
    int column,
    float *value)
{
    if ((depth_valid &&
         depth_valid->at<std::uint8_t>(row, column) == 0) ||
        (support && support->at<std::uint8_t>(row, column) == 0))
    {
        return false;
    }
    const float candidate = depth.at<float>(row, column);
    if (!(candidate > 0.0f) || !std::isfinite(candidate))
    {
        return false;
    }
    *value = candidate;
    return true;
}

} // namespace

bool DepthTsdfNarrowBandActivation::build(
    const DepthTsdfLayout &layout,
    const std::vector<DepthTsdfNarrowBandFrameView> &frames,
    const DepthTsdfNarrowBandActivationOptions &options)
{
    _valid = false;
    _cancelled = false;
    _activeBlocks.clear();
    _statistics = {};

    if (!validLayout(layout) ||
        options.blockSizeSamples <= 0 ||
        options.depthStride <= 0 ||
        !(options.truncationDistance > 0.0f) ||
        !std::isfinite(options.truncationDistance) ||
        !(options.rayStepVoxels > 0.0f) ||
        !std::isfinite(options.rayStepVoxels) ||
        options.haloBlocks < 0)
    {
        return false;
    }

    const int sample_count_x = layout.cells[0] + 1;
    const int sample_count_y = layout.cells[1] + 1;
    const int sample_count_z = layout.cells[2] + 1;
    const int block_count_x =
        (sample_count_x + options.blockSizeSamples - 1) /
        options.blockSizeSamples;
    const int block_count_y =
        (sample_count_y + options.blockSizeSamples - 1) /
        options.blockSizeSamples;
    const int block_count_z =
        (sample_count_z + options.blockSizeSamples - 1) /
        options.blockSizeSamples;
    std::size_t total_blocks = 0;
    if (!checkedProduct(
            block_count_x, block_count_y, block_count_z, &total_blocks))
    {
        return false;
    }

    std::vector<std::uint8_t> core_blocks(total_blocks, 0);
    DepthTsdfNarrowBandActivationStatistics statistics;
    statistics.totalBlocks = static_cast<std::uint64_t>(total_blocks);
    const float minimum_voxel_size = std::min(
        {layout.voxelSize[0], layout.voxelSize[1], layout.voxelSize[2]});
    const double maximum_ray_step =
        minimum_voxel_size * options.rayStepVoxels;
    const int half_band_intervals = std::max(
        1,
        static_cast<int>(std::ceil(
            options.truncationDistance / maximum_ray_step)));

    const auto cancel = [&]()
    {
        if (options.isCancelled && options.isCancelled())
        {
            _cancelled = true;
            return true;
        }
        return false;
    };

    for (const DepthTsdfNarrowBandFrameView &frame : frames)
    {
        if (cancel())
        {
            return false;
        }
        if (!frame.camera || !frame.camera->isValid() ||
            !frame.depth || frame.depth->empty() ||
            frame.depth->type() != CV_32FC1 ||
            !validOptionalMask(frame.depthValidMask, *frame.depth) ||
            !validOptionalMask(frame.supportMask, *frame.depth))
        {
            continue;
        }

        for (int row = 0; row < frame.depth->rows;
             row += options.depthStride)
        {
            if (cancel())
            {
                return false;
            }
            for (int column = 0; column < frame.depth->cols;
                 column += options.depthStride)
            {
                if (cancel())
                {
                    return false;
                }
                float source_depth = 0.0f;
                if (!isPixelUsable(
                        *frame.depth,
                        frame.depthValidMask,
                        frame.supportMask,
                        row,
                        column,
                        &source_depth))
                {
                    continue;
                }
                ++statistics.validSourceSamples;
                const double pixel[2] = {
                    static_cast<double>(column),
                    static_cast<double>(row)};
                for (int interval = -half_band_intervals;
                     interval <= half_band_intervals;
                     ++interval)
                {
                    const double ratio =
                        static_cast<double>(interval) /
                        half_band_intervals;
                    const double ray_depth =
                        source_depth +
                        ratio * options.truncationDistance;
                    if (!(ray_depth > 0.0))
                    {
                        continue;
                    }
                    double world[3]{};
                    if (!frame.camera->unprojectPixel(
                            pixel, ray_depth, world))
                    {
                        continue;
                    }
                    std::size_t index = 0;
                    if (worldToSampleBlock(
                            layout,
                            world,
                            options.blockSizeSamples,
                            block_count_x,
                            block_count_y,
                            block_count_z,
                            &index))
                    {
                        core_blocks[index] = 1;
                        ++statistics.markedRaySamples;
                    }
                }
            }
        }
    }

    std::vector<std::uint8_t> active_blocks = core_blocks;
    if (options.haloBlocks > 0)
    {
        for (int z = 0; z < block_count_z; ++z)
        {
            for (int y = 0; y < block_count_y; ++y)
            {
                for (int x = 0; x < block_count_x; ++x)
                {
                    if (core_blocks[blockIndex(
                            x, y, z, block_count_x, block_count_y)] == 0)
                    {
                        continue;
                    }
                    for (int dz = -options.haloBlocks;
                         dz <= options.haloBlocks;
                         ++dz)
                    {
                        for (int dy = -options.haloBlocks;
                             dy <= options.haloBlocks;
                             ++dy)
                        {
                            for (int dx = -options.haloBlocks;
                                 dx <= options.haloBlocks;
                                 ++dx)
                            {
                                const int halo_x = x + dx;
                                const int halo_y = y + dy;
                                const int halo_z = z + dz;
                                if (halo_x >= 0 &&
                                    halo_x < block_count_x &&
                                    halo_y >= 0 &&
                                    halo_y < block_count_y &&
                                    halo_z >= 0 &&
                                    halo_z < block_count_z)
                                {
                                    active_blocks[blockIndex(
                                        halo_x,
                                        halo_y,
                                        halo_z,
                                        block_count_x,
                                        block_count_y)] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    statistics.activeBlocks = static_cast<std::uint64_t>(
        std::count(active_blocks.cbegin(), active_blocks.cend(), 1));
    _blockSizeSamples = options.blockSizeSamples;
    _sampleCountX = sample_count_x;
    _sampleCountY = sample_count_y;
    _sampleCountZ = sample_count_z;
    _blockCountX = block_count_x;
    _blockCountY = block_count_y;
    _blockCountZ = block_count_z;
    _activeBlocks = std::move(active_blocks);
    _statistics = statistics;
    _valid = true;
    return true;
}

bool DepthTsdfNarrowBandActivation::isSampleActive(
    int x, int y, int z) const
{
    if (!_valid ||
        x < 0 || x >= _sampleCountX ||
        y < 0 || y >= _sampleCountY ||
        z < 0 || z >= _sampleCountZ)
    {
        return false;
    }
    const int block_x = x / _blockSizeSamples;
    const int block_y = y / _blockSizeSamples;
    const int block_z = z / _blockSizeSamples;
    return _activeBlocks[blockIndex(
               block_x,
               block_y,
               block_z,
               _blockCountX,
               _blockCountY)] != 0;
}

} // namespace xjw::mesh
