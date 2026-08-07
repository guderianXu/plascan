#include "HierarchicalBaBlockSolver.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace xjw::hierarchical_ba_detail
{

BlockOutcome solveBlock(std::size_t block_index,
                        const CovisibilityBlock &block,
                        const SfmReconstruction &reconstruction,
                        const BAOptions &base_options,
                        int threads_per_block,
                        bool use_ceres)
{
    BlockOutcome outcome;
    outcome.blockIndex = block_index;
    outcome.cameraIds = block.activeImageIds();
    if (outcome.cameraIds.size() < 2)
    {
        return outcome;
    }

    const std::unordered_set<ImageId> core_ids(
        block.coreImageIds.begin(), block.coreImageIds.end());
    const std::unordered_set<ImageId> overlap_ids(
        block.overlapImageIds.begin(), block.overlapImageIds.end());
    std::unordered_map<ImageId, int> camera_index;
    std::vector<Camera> cameras;
    cameras.reserve(outcome.cameraIds.size());
    for (ImageId image_id : outcome.cameraIds)
    {
        if (!reconstruction.hasCamera(image_id))
        {
            return outcome;
        }
        camera_index.emplace(image_id, static_cast<int>(cameras.size()));
        cameras.push_back(reconstruction.camera(image_id));
    }

    std::vector<BATrack> tracks;
    const std::vector<Point3DId> all_point_ids = reconstruction.allPoint3DIds();
    tracks.reserve(all_point_ids.size() / 2);
    for (Point3DId point_id : all_point_ids)
    {
        if (!reconstruction.hasPoint3D(point_id))
        {
            continue;
        }
        const ScenePoint3D &point = reconstruction.point3D(point_id);
        BATrack track;
        track.initialPoint = point.xyz;
        bool touches_core = false;
        for (const TrackElement &element : point.track.elements)
        {
            const auto index = camera_index.find(element.imageId);
            if (index == camera_index.end() || !reconstruction.hasImage(element.imageId))
            {
                continue;
            }
            const ImageData &image = reconstruction.image(element.imageId);
            if (element.featureIdx >= image.keypoints.size())
            {
                continue;
            }
            const FeatureKeypoint &keypoint = image.keypoints[element.featureIdx];
            track.observations.push_back(
                {index->second, keypoint.x, keypoint.y, point.track.confidence});
            touches_core = touches_core || core_ids.count(element.imageId) > 0;
        }
        if (touches_core && track.observations.size() >= 2)
        {
            tracks.push_back(std::move(track));
            outcome.pointIds.push_back(point_id);
        }
    }
    if (tracks.empty())
    {
        return outcome;
    }

    BAOptions options = base_options;
    options.backend = use_ceres ? BABackend::CeresCpu : options.backend;
    options.numThreads = std::max(1, threads_per_block);
    options.maxIterations = std::max(1, base_options.maxIterations);
    options.logIterationProgress = false;
    options.progressCallback = nullptr;
    options.enablePointFilter = false;
    options.refineSharedFocalLength = false;
    options.refineSharedFocalAspectRatio = false;
    options.refineSharedPrincipalPoint = false;
    options.refineSharedRadialDistortion = false;
    options.cameraCalibrationGroupIds.clear();
    options.cameraPosePriors.clear();
    options.enableControlPointConstraints = false;
    options.enableScaleBarConstraints = false;
    options.scaleBarConstraints.clear();
    options.cameraPlaneConstraint = {};
    options.fixedCameraIndices.clear();

    for (ImageId image_id : outcome.cameraIds)
    {
        if (overlap_ids.count(image_id) > 0)
        {
            options.fixedCameraIndices.push_back(camera_index.at(image_id));
        }
    }
    // 每个块至少固定两台相机，直接保留共同坐标系的旋转、平移和尺度。
    for (ImageId image_id : block.coreImageIds)
    {
        if (options.fixedCameraIndices.size() >= 2)
        {
            break;
        }
        options.fixedCameraIndices.push_back(camera_index.at(image_id));
    }
    std::sort(options.fixedCameraIndices.begin(), options.fixedCameraIndices.end());
    options.fixedCameraIndices.erase(
        std::unique(options.fixedCameraIndices.begin(), options.fixedCameraIndices.end()),
        options.fixedCameraIndices.end());
    options.gaugePolicy = BAGaugePolicy::RequireExplicitGauge;

    outcome.result = BundleAdjust::optimizePoints(cameras, tracks, options);
    const double tolerance = std::max(
        1.0e-9, std::abs(outcome.result.meanRmsBefore) * 0.01);
    outcome.accepted = outcome.result.solutionUsable &&
        std::isfinite(outcome.result.meanRmsBefore) &&
        std::isfinite(outcome.result.meanRmsAfter) &&
        outcome.result.meanRmsAfter <= outcome.result.meanRmsBefore + tolerance &&
        outcome.result.refinedCameras.size() == cameras.size() &&
        outcome.result.points.size() == tracks.size();
    return outcome;
}

} // namespace xjw::hierarchical_ba_detail
