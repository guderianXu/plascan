#include "InitialSparsePointFilter.h"
#include "geometry/ProjectionGeometry.h"
#include "geometry/TriangulationQuality.h"

#include <plamatrix/ops/vector.h>

#include <cmath>

namespace xjw
{

InitialSparseTriangulationResult InitialSparsePointFilter::filter(
    const std::vector<Camera> &cameras,
    const std::vector<BATrack> &tracks,
    const InitialSparseTriangulationOptions &options)
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
        const BATrack &track = tracks[static_cast<std::size_t>(trackIdx)];
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

        if (!plamatrix::isFinite(plamatrix::Vec3<double>(track.initialPoint)))
        {
            ++result.rejectedByReprojCount;
            continue;
        }

        const std::array<double, 3> &triangulatedPoint = track.initialPoint;
        const double minTriAngleDeg = minimumTriangulationAngleDeg(cameras,
                                                                   track,
                                                                   triangulatedPoint);
        if (minTriAngleDeg < options.minTriAngleDeg)
        {
            ++result.rejectedByTriAngleCount;
            continue;
        }

        double squaredErrorSum = 0.0;
        bool reprojectionValid = true;
        for (const BAObservation &observation : track.observations)
        {
            if (observation.cameraIndex < 0
                || observation.cameraIndex >= static_cast<int>(cameras.size()))
            {
                reprojectionValid = false;
                break;
            }

            const double errorPx = reprojectionErrorPx(
                cameras[static_cast<size_t>(observation.cameraIndex)],
                triangulatedPoint,
                {observation.u, observation.v});
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
