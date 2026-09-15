#include "DepthTsdfInternals.h"
namespace xjw::mesh::tsdf_detail
{
    using namespace tsdf_detail;

    void integrateWeighted(float* value, float* weight, float observation, float observationWeight)
    {
        if (!value || !weight || observationWeight <= 0.0f)
        {
            return;
        }
        const float updatedWeight = *weight + observationWeight;
        *value = (*value * *weight + observation * observationWeight) / updatedWeight;
        *weight = updatedWeight;
    }

    std::size_t sampleIndex(const DepthTsdfLayout& layout, int x, int y, int z)
    {
        const std::size_t rowSize = static_cast<std::size_t>(layout.cells[0] + 1);
        const std::size_t layerSize = rowSize * static_cast<std::size_t>(layout.cells[1] + 1);
        return static_cast<std::size_t>(z) * layerSize + static_cast<std::size_t>(y) * rowSize +
               static_cast<std::size_t>(x);
    }

    std::vector<std::uint8_t>
    dilateSampleMask(const DepthTsdfLayout& layout, const std::vector<std::uint8_t>& source, int passes)
    {
        std::vector<std::uint8_t> reach = source;
        if (reach.empty() || passes <= 0)
        {
            return reach;
        }
        const int x_samples = layout.cells[0] + 1;
        const int y_samples = layout.cells[1] + 1;
        const int z_samples = layout.cells[2] + 1;
        std::vector<std::uint8_t> next(reach.size(), 0);
        for (int pass = 0; pass < passes; ++pass)
        {
            next = reach;
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int z = 0; z < z_samples; ++z)
            {
                for (int y = 0; y < y_samples; ++y)
                {
                    for (int x = 0; x < x_samples; ++x)
                    {
                        const std::size_t index = sampleIndex(layout, x, y, z);
                        if (reach[index] != 0)
                        {
                            continue;
                        }
                        const bool adjacent = (x > 0 && reach[sampleIndex(layout, x - 1, y, z)] != 0) ||
                                              (x + 1 < x_samples && reach[sampleIndex(layout, x + 1, y, z)] != 0) ||
                                              (y > 0 && reach[sampleIndex(layout, x, y - 1, z)] != 0) ||
                                              (y + 1 < y_samples && reach[sampleIndex(layout, x, y + 1, z)] != 0) ||
                                              (z > 0 && reach[sampleIndex(layout, x, y, z - 1)] != 0) ||
                                              (z + 1 < z_samples && reach[sampleIndex(layout, x, y, z + 1)] != 0);
                        if (adjacent)
                        {
                            next[index] = 1;
                        }
                    }
                }
            }
            reach.swap(next);
        }
        return reach;
    }

    int bitCount(std::uint16_t value)
    {
        return std::popcount(value);
    }

    int bitCount(DepthGeometrySourceMask value)
    {
        return static_cast<int>(value.count());
    }

    DepthUncertaintyBandEstimate estimateDepthUncertaintyBand(const QVector<DepthTsdfFrame>& frames,
                                                              const DepthTsdfOptions& options,
                                                              float maximum_voxel_size)
    {
        DepthUncertaintyBandEstimate estimate;
        if (!options.enableUncertaintyAdaptiveTruncation || !std::isfinite(maximum_voxel_size) ||
            maximum_voxel_size <= std::numeric_limits<float>::epsilon())
        {
            return estimate;
        }

        const int maximum_samples_per_frame = std::max(256, options.uncertaintyAdaptiveMaximumSamplesPerFrame);
        std::vector<float> uncertainty_voxels;
        uncertainty_voxels.reserve(static_cast<std::size_t>(frames.size()) *
                                   static_cast<std::size_t>(maximum_samples_per_frame));
        for (const DepthTsdfFrame& frame : frames)
        {
            if (frame.depth.type() != CV_32FC1 || frame.inverseDepthRelativeSpread.type() != CV_32FC1 ||
                frame.depth.size() != frame.inverseDepthRelativeSpread.size())
            {
                continue;
            }
            const bool use_confidence =
                frame.confidence.type() == CV_32FC1 && frame.confidence.size() == frame.depth.size();
            const bool use_depth_valid =
                frame.depthValidMask.type() == CV_8UC1 && frame.depthValidMask.size() == frame.depth.size();
            const bool use_support =
                frame.supportMask.type() == CV_8UC1 && frame.supportMask.size() == frame.depth.size();
            const int pixel_count = frame.depth.rows * frame.depth.cols;
            const int sampling_step =
                std::max(1,
                         static_cast<int>(std::ceil(std::sqrt(static_cast<double>(pixel_count) /
                                                              static_cast<double>(maximum_samples_per_frame)))));
            const int sampling_offset = sampling_step / 2;
            for (int row = sampling_offset; row < frame.depth.rows; row += sampling_step)
            {
                const float* depth_row = frame.depth.ptr<float>(row);
                const float* spread_row = frame.inverseDepthRelativeSpread.ptr<float>(row);
                const float* confidence_row = use_confidence ? frame.confidence.ptr<float>(row) : nullptr;
                const std::uint8_t* depth_valid_row =
                    use_depth_valid ? frame.depthValidMask.ptr<std::uint8_t>(row) : nullptr;
                const std::uint8_t* support_row = use_support ? frame.supportMask.ptr<std::uint8_t>(row) : nullptr;
                for (int column = sampling_offset; column < frame.depth.cols; column += sampling_step)
                {
                    if ((depth_valid_row && depth_valid_row[column] == 0) ||
                        (support_row && support_row[column] == 0) ||
                        (confidence_row && confidence_row[column] < options.minimumConfidence))
                    {
                        continue;
                    }
                    const float depth = depth_row[column];
                    const float relative_spread = spread_row[column];
                    if (!std::isfinite(depth) || depth <= 0.0f || !std::isfinite(relative_spread) ||
                        relative_spread <= 0.0f)
                    {
                        continue;
                    }
                    if (options.maximumObservationInverseDepthSpread > 0.0f &&
                        relative_spread > options.maximumObservationInverseDepthSpread)
                    {
                        continue;
                    }
                    const float uncertainty = depth * relative_spread / maximum_voxel_size;
                    if (std::isfinite(uncertainty) && uncertainty > 0.0f)
                    {
                        uncertainty_voxels.push_back(uncertainty);
                    }
                }
            }
        }
        estimate.sampleCount = uncertainty_voxels.size();
        if (estimate.sampleCount <
            static_cast<std::uint64_t>(std::max(64, options.uncertaintyAdaptiveMinimumSampleCount)))
        {
            return estimate;
        }
        const std::size_t p90_index =
            static_cast<std::size_t>(std::floor(0.90 * static_cast<double>(uncertainty_voxels.size() - 1)));
        std::nth_element(uncertainty_voxels.begin(),
                         uncertainty_voxels.begin() + static_cast<std::ptrdiff_t>(p90_index),
                         uncertainty_voxels.end());
        estimate.p90Voxels = uncertainty_voxels[p90_index];
        return estimate;
    }

    bool volumeNormalAt(
        const DepthTsdfLayout& layout, const std::vector<float>& tsdf, int x, int y, int z, cv::Vec3f* normal)
    {
        if (!normal || x <= 0 || x >= layout.cells[0] || y <= 0 || y >= layout.cells[1] || z <= 0 ||
            z >= layout.cells[2])
        {
            return false;
        }
        cv::Vec3f gradient(tsdf[sampleIndex(layout, x + 1, y, z)] - tsdf[sampleIndex(layout, x - 1, y, z)],
                           tsdf[sampleIndex(layout, x, y + 1, z)] - tsdf[sampleIndex(layout, x, y - 1, z)],
                           tsdf[sampleIndex(layout, x, y, z + 1)] - tsdf[sampleIndex(layout, x, y, z - 1)]);
        const float length = std::sqrt(gradient.dot(gradient));
        if (!std::isfinite(length) || length <= 1.0e-6f)
        {
            return false;
        }
        *normal = gradient / length;
        return true;
    }
} // namespace xjw::mesh::tsdf_detail
