#include "DepthMeshCompleteness.h"

#include "TriangleDistanceIndex.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace xjw::mesh
{
namespace
{

double quantile(const std::vector<double> &sorted, double fraction)
{
    if (sorted.empty())
    {
        return 0.0;
    }
    const double position = std::clamp(fraction, 0.0, 1.0) *
        static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double blend = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - blend) + sorted[upper] * blend;
}

} // namespace

DepthMeshCompletenessStatistics DepthMeshCompleteness::evaluate(
    const TriMesh &mesh,
    const QVector<DepthTsdfFrame> &frames,
    const DepthMeshCompletenessOptions &options)
{
    DepthMeshCompletenessStatistics result;
    result.tolerance = std::max(0.0, options.tolerance);
    if (mesh.empty() || frames.empty() || result.tolerance <= 0.0)
    {
        return result;
    }

    const TriangleDistanceIndex triangle_index(mesh);
    if (triangle_index.empty())
    {
        return result;
    }
    const double tolerance_squared = result.tolerance * result.tolerance;

    std::vector<double> frame_recalls;
    frame_recalls.reserve(static_cast<std::size_t>(frames.size()));
    for (const DepthTsdfFrame &frame : frames)
    {
        if (options.excludeAuxiliaryFrames && frame.auxiliarySurfaceOnly)
        {
            continue;
        }
        if (std::find(
                options.excludedRefIndices.cbegin(),
                options.excludedRefIndices.cend(),
                frame.refIndex) != options.excludedRefIndices.cend())
        {
            continue;
        }
        if (!frame.camera.isValid() || frame.depth.empty() ||
            frame.depth.type() != CV_32FC1 ||
            frame.depthValidMask.type() != CV_8UC1 ||
            frame.supportMask.type() != CV_8UC1)
        {
            continue;
        }

        DepthMeshFrameCompleteness frame_result;
        frame_result.refIndex = frame.refIndex;
        frame_result.auxiliarySurfaceOnly = frame.auxiliarySurfaceOnly;
        const double requested_stride = std::sqrt(
            static_cast<double>(frame.depth.total()) /
            std::max(1, options.maximumDepthSamplesPerFrame));
        const int stride = std::max(
            1, static_cast<int>(std::ceil(requested_stride)));
        for (int row = stride / 2; row < frame.depth.rows; row += stride)
        {
            for (int column = stride / 2;
                 column < frame.depth.cols;
                 column += stride)
            {
                if (frame.depthValidMask.at<std::uint8_t>(row, column) == 0 ||
                    frame.supportMask.at<std::uint8_t>(row, column) == 0)
                {
                    continue;
                }
                const float depth = frame.depth.at<float>(row, column);
                if (!std::isfinite(depth) || depth <= 0.0f)
                {
                    continue;
                }
                const double pixel[2] = {
                    static_cast<double>(column),
                    static_cast<double>(row)
                };
                double world[3]{};
                if (!frame.camera.unprojectPixel(pixel, depth, world))
                {
                    continue;
                }
                ++frame_result.sampledDepthPointCount;
                if (triangle_index.nearestDistanceSquared(
                        {world[0], world[1], world[2]}) <=
                    tolerance_squared)
                {
                    ++frame_result.explainedDepthPointCount;
                }
            }
        }
        if (frame_result.sampledDepthPointCount == 0)
        {
            continue;
        }
        frame_result.recall =
            static_cast<double>(frame_result.explainedDepthPointCount) /
            static_cast<double>(frame_result.sampledDepthPointCount);
        result.sampledDepthPointCount += frame_result.sampledDepthPointCount;
        result.explainedDepthPointCount += frame_result.explainedDepthPointCount;
        frame_recalls.push_back(frame_result.recall);
        result.frames.push_back(frame_result);
    }

    if (frame_recalls.size() < 3 || result.sampledDepthPointCount == 0)
    {
        return result;
    }
    std::sort(frame_recalls.begin(), frame_recalls.end());
    result.available = true;
    result.aggregateRecall =
        static_cast<double>(result.explainedDepthPointCount) /
        static_cast<double>(result.sampledDepthPointCount);
    result.minimumFrameRecall = frame_recalls.front();
    result.p10FrameRecall = quantile(frame_recalls, 0.10);
    result.medianFrameRecall = quantile(frame_recalls, 0.50);
    result.gatePassed =
        result.p10FrameRecall >= options.minimumP10FrameRecall &&
        result.medianFrameRecall >= options.minimumMedianFrameRecall;
    return result;
}

} // namespace xjw::mesh
