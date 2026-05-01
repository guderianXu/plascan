#include "InitialSparsePointCloudTriangulator.h"

#include "math/Vec3Ops.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{

namespace
{


bool projectPoint(const Camera &camera,
                  const std::array<double, 3> &xyz,
                  double projectedPoint[2])
{
    const double worldPoint[3] = {xyz[0], xyz[1], xyz[2]};
    if (camera.projectWorldPoint(worldPoint, projectedPoint))
    {
        return true;
    }
    return camera.projectWorldPointSigned(worldPoint, projectedPoint);
}

double reprojectionErrorPx(const Camera &camera,
                           const std::array<double, 3> &xyz,
                           const BAObservation &observation)
{
    double projectedPoint[2] = {0.0, 0.0};
    if (!projectPoint(camera, xyz, projectedPoint))
    {
        return std::numeric_limits<double>::infinity();
    }

    const double dx = projectedPoint[0] - observation.u;
    const double dy = projectedPoint[1] - observation.v;
    return std::sqrt(dx * dx + dy * dy);
}

double minimumTriangulationAngleDeg(const std::vector<Camera> &cameras,
                                    const BATrack &track,
                                    const std::array<double, 3> &xyz)
{
    if (track.observations.size() < 2)
    {
        return 0.0;
    }

    double minAngleDeg = 180.0;
    for (size_t leftIndex = 0; leftIndex < track.observations.size(); ++leftIndex)
    {
        const BAObservation &leftObservation = track.observations[leftIndex];
        if (leftObservation.cameraIndex < 0
            || leftObservation.cameraIndex >= static_cast<int>(cameras.size()))
        {
            continue;
        }

        const std::array<double, 3> leftCenter =
            cameras[static_cast<size_t>(leftObservation.cameraIndex)].cameraCenter();
        const std::array<double, 3> leftVector{{leftCenter[0] - xyz[0],
                                                leftCenter[1] - xyz[1],
                                                leftCenter[2] - xyz[2]}};
        const double leftNorm = std::sqrt(leftVector[0] * leftVector[0]
                                        + leftVector[1] * leftVector[1]
                                        + leftVector[2] * leftVector[2]);
        if (leftNorm < 1e-12)
        {
            continue;
        }

        for (size_t rightIndex = leftIndex + 1; rightIndex < track.observations.size(); ++rightIndex)
        {
            const BAObservation &rightObservation = track.observations[rightIndex];
            if (rightObservation.cameraIndex < 0
                || rightObservation.cameraIndex >= static_cast<int>(cameras.size()))
            {
                continue;
            }

            const std::array<double, 3> rightCenter =
                cameras[static_cast<size_t>(rightObservation.cameraIndex)].cameraCenter();
            const std::array<double, 3> rightVector{{rightCenter[0] - xyz[0],
                                                     rightCenter[1] - xyz[1],
                                                     rightCenter[2] - xyz[2]}};
            const double rightNorm = std::sqrt(rightVector[0] * rightVector[0]
                                             + rightVector[1] * rightVector[1]
                                             + rightVector[2] * rightVector[2]);
            if (rightNorm < 1e-12)
            {
                continue;
            }

            const double dot = leftVector[0] * rightVector[0]
                             + leftVector[1] * rightVector[1]
                             + leftVector[2] * rightVector[2];
            const double cosine = std::clamp(dot / (leftNorm * rightNorm), -1.0, 1.0);
            minAngleDeg = std::min(minAngleDeg, std::acos(cosine) * 180.0 / M_PI);
        }
    }

    return std::isfinite(minAngleDeg) ? minAngleDeg : 0.0;
}
} // namespace

InitialSparseTriangulationResult InitialSparsePointCloudFilter::triangulate(
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

        if (!vec3::isFinite(track.initialPoint))
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
                observation);
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