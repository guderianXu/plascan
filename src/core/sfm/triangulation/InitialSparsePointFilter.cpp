#include "InitialSparsePointFilter.h"

#include <plamatrix/dense/matrix.h>
#include <placamera/camera_baseline.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{

    namespace
    {

        double
        minimumTriangulationAngleDeg(const std::vector<std::shared_ptr<const placamera::FramePinholeModel>>& cameras,
                                     const plabundle::Track& track,
                                     const std::array<double, 3>& point)
        {
            double minimum_angle = std::numeric_limits<double>::infinity();
            for (std::size_t left = 0; left < track.observations.size(); ++left)
            {
                const int left_index = track.observations[left].cameraIndex;
                if (left_index < 0 || left_index >= static_cast<int>(cameras.size()) || !cameras[left_index])
                {
                    continue;
                }
                for (std::size_t right = left + 1; right < track.observations.size(); ++right)
                {
                    const int right_index = track.observations[right].cameraIndex;
                    if (right_index < 0 || right_index >= static_cast<int>(cameras.size()) || !cameras[right_index])
                    {
                        continue;
                    }
                    const auto baseline =
                        placamera::CameraBaseline::evaluate(*cameras[left_index], *cameras[right_index], point);
                    if (baseline.isPointInFrontOfBothCameras() && baseline.triangulationAngleDeg())
                    {
                        minimum_angle = std::min(minimum_angle, *baseline.triangulationAngleDeg());
                    }
                }
            }
            return std::isfinite(minimum_angle) ? minimum_angle : 0.0;
        }

    } // namespace

    InitialSparseTriangulationResult
    InitialSparsePointFilter::filter(const std::vector<std::shared_ptr<const placamera::FramePinholeModel>>& cameras,
                                     const std::vector<plabundle::Track>& tracks,
                                     const InitialSparseTriangulationOptions& options)
    {
        InitialSparseTriangulationResult result;
        if (cameras.size() < 2)
        {
            result.errorMessage = "At least two cameras are required";
            return result;
        }
        if (tracks.empty())
        {
            result.errorMessage = "No triangulation tracks available";
            return result;
        }

        const int camCount = static_cast<int>(cameras.size());
        // minTrackLength 不能超过实际相机数量，否则两张图的情况下 minTrackLen=4 会过滤掉所有点
        const int minObservations = std::max(2, std::min(options.minObservations, camCount));
        const int minTrackLength = std::max(2, std::min(options.minTrackLength, camCount));
        result.candidateTrackCount = static_cast<int>(tracks.size());
        result.points.reserve(tracks.size());

        for (int trackIdx = 0; trackIdx < static_cast<int>(tracks.size()); ++trackIdx)
        {
            const plabundle::Track& track = tracks[static_cast<std::size_t>(trackIdx)];
            const int observationCount = static_cast<int>(track.observations.size());
            if (observationCount < minObservations || observationCount < minTrackLength)
            {
                ++result.rejectedByObservationCount;
                continue;
            }
            if (options.ignoreTwoViewTracks && observationCount <= 2)
            {
                ++result.rejectedByObservationCount;
                continue;
            }

            if (!plamatrix::Vector3d(track.initialPoint[0], track.initialPoint[1], track.initialPoint[2]).allFinite())
            {
                ++result.rejectedByReprojCount;
                continue;
            }

            const std::array<double, 3>& triangulatedPoint = track.initialPoint;
            const double minTriAngleDeg = minimumTriangulationAngleDeg(cameras, track, triangulatedPoint);
            if (minTriAngleDeg < options.minTriAngleDeg)
            {
                ++result.rejectedByTriAngleCount;
                continue;
            }

            double squaredErrorSum = 0.0;
            bool reprojectionValid = true;
            for (const plabundle::Observation& observation : track.observations)
            {
                if (observation.cameraIndex < 0 || observation.cameraIndex >= static_cast<int>(cameras.size()) ||
                    !cameras[static_cast<std::size_t>(observation.cameraIndex)])
                {
                    reprojectionValid = false;
                    break;
                }

                const auto& camera = *cameras[static_cast<std::size_t>(observation.cameraIndex)];
                const auto projection =
                    camera.groundToImage(placamera::GroundCoordinate{camera.groundFrame(), triangulatedPoint});
                if (!projection)
                {
                    reprojectionValid = false;
                    break;
                }
                const double errorPx = std::hypot(projection.value().image.sample - observation.u,
                                                  projection.value().image.line - observation.v);
                if (!std::isfinite(errorPx) || errorPx > options.maxReprojErrorPx)
                {
                    reprojectionValid = false;
                    break;
                }
                squaredErrorSum += errorPx * errorPx;
            }

            if (!reprojectionValid)
            {
                ++result.rejectedByReprojCount;
                continue;
            }

            InitialSparsePoint point;
            point.xyz = triangulatedPoint;
            point.sourceTrackIndex = trackIdx;
            point.trackLength = observationCount;
            point.minTriAngleDeg = minTriAngleDeg;
            point.rmsReprojPx = std::sqrt(squaredErrorSum / observationCount);
            result.points.push_back(point);
        }

        result.exportedPointCount = static_cast<int>(result.points.size());
        result.success = !result.points.empty();
        if (!result.success)
        {
            result.errorMessage = "No valid sparse points survived filtering";
        }
        return result;
    }

} // namespace xjw
