#include "HierarchicalBaBlockSolver.h"

#include "SfmBundleCameraCodec.h"

#include <plabundle/solver.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace xjw::hierarchical_ba_detail
{

    BlockOutcome solveBlock(std::size_t block_index,
                            const CovisibilityBlock& block,
                            const SfmReconstruction& reconstruction,
                            const std::vector<Point3DId>& candidate_point_ids,
                            const plabundle::SolveOptions& base_options,
                            int threads_per_block)
    {
        BlockOutcome outcome;
        outcome.blockIndex = block_index;
        outcome.cameraIds = block.activeImageIds();
        if (outcome.cameraIds.size() < 2)
        {
            return outcome;
        }

        const std::unordered_set<ImageId> core_ids(block.coreImageIds.begin(), block.coreImageIds.end());
        const std::unordered_set<ImageId> overlap_ids(block.overlapImageIds.begin(), block.overlapImageIds.end());
        std::unordered_map<ImageId, int> camera_index;
        std::vector<placamera::FramePinholeNumericState> cameras;
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

        std::vector<plabundle::Track> tracks;
        std::vector<int> fixed_track_indices;
        tracks.reserve(candidate_point_ids.size());
        for (Point3DId point_id : candidate_point_ids)
        {
            if (!reconstruction.hasPoint3D(point_id))
            {
                continue;
            }
            const ScenePoint3D& point = reconstruction.point3D(point_id);
            plabundle::Track track;
            track.initialPoint = point.xyz;
            bool touches_core = false;
            bool crosses_block = false;
            for (const TrackElement& element : point.track.elements)
            {
                crosses_block = crosses_block ||
                                (reconstruction.hasCamera(element.imageId) && core_ids.count(element.imageId) == 0);
                const auto index = camera_index.find(element.imageId);
                if (!reconstruction.hasImage(element.imageId))
                {
                    continue;
                }
                if (index == camera_index.end())
                {
                    continue;
                }
                const ImageData& image = reconstruction.image(element.imageId);
                if (element.featureIdx >= image.keypoints.size())
                {
                    continue;
                }
                const FeatureKeypoint& keypoint = image.keypoints[element.featureIdx];
                track.observations.push_back(
                    {index->second, keypoint.x, keypoint.y, point.track.confidence, keypoint.scale});
                touches_core = touches_core || core_ids.count(element.imageId) > 0;
            }
            if (touches_core && track.observations.size() >= 2)
            {
                const int track_index = static_cast<int>(tracks.size());
                tracks.push_back(std::move(track));
                outcome.pointIds.push_back(point_id);
                if (crosses_block)
                {
                    fixed_track_indices.push_back(track_index);
                }
            }
        }
        if (tracks.empty())
        {
            return outcome;
        }

        plabundle::SolveOptions options = base_options;
        // 独立块并行时统一使用 CPU，避免多个求解器同时争抢同一 GPU 上下文。
        options.backend.requested = plabundle::Backend::PlaMatrixCpu;
        options.solver.numThreads = std::max(1, threads_per_block);
        options.solver.maxIterations = std::max(1, base_options.solver.maxIterations);
        options.solver.logIterationProgress = false;
        options.solver.progressCallback = nullptr;
        options.solver.enablePointFilter = false;
        options.calibration.refineSharedFocalLength = false;
        options.calibration.refineSharedFocalAspectRatio = false;
        options.calibration.refineSharedPrincipalPoint = false;
        options.calibration.refineSharedRadialDistortion = false;
        plabundle::Problem problem;
        problem.tracks = std::move(tracks);
        problem.fixedTrackIndices = std::move(fixed_track_indices);
        outcome.fixedTrackCount = static_cast<int>(problem.fixedTrackIndices.size());
        if (!sfm_bundle_camera::encodeAll(cameras, &problem.cameras))
        {
            return outcome;
        }

        for (ImageId image_id : outcome.cameraIds)
        {
            if (overlap_ids.count(image_id) > 0)
            {
                problem.fixedCameraIndices.push_back(camera_index.at(image_id));
            }
        }
        // 每个块至少固定两台相机，直接保留共同坐标系的旋转、平移和尺度。
        for (ImageId image_id : block.coreImageIds)
        {
            if (problem.fixedCameraIndices.size() >= 2)
            {
                break;
            }
            problem.fixedCameraIndices.push_back(camera_index.at(image_id));
        }
        std::sort(problem.fixedCameraIndices.begin(), problem.fixedCameraIndices.end());
        problem.fixedCameraIndices.erase(
            std::unique(problem.fixedCameraIndices.begin(), problem.fixedCameraIndices.end()),
            problem.fixedCameraIndices.end());
        problem.gauge.policy = plabundle::GaugePolicy::RequireExplicitGauge;

        outcome.result = plabundle::Solver().solve(problem, options);
        outcome.refinedCameras = cameras;
        const bool cameras_applied =
            !outcome.result.usable() ||
            sfm_bundle_camera::decodeAll(outcome.result.refinedCameras, &outcome.refinedCameras);
        const double tolerance = std::max(1.0e-9, std::abs(outcome.result.quality.meanRmsBefore) * 0.01);
        outcome.accepted =
            outcome.result.usable() && cameras_applied && std::isfinite(outcome.result.quality.meanRmsBefore) &&
            std::isfinite(outcome.result.quality.meanRmsAfter) &&
            outcome.result.quality.meanRmsAfter <= outcome.result.quality.meanRmsBefore + tolerance &&
            outcome.refinedCameras.size() == cameras.size() && outcome.result.points.size() == problem.tracks.size();
        return outcome;
    }

} // namespace xjw::hierarchical_ba_detail
