#include "TriangulationQuality.h"

#include "Intersection.h"
#include "geometry/ProjectionGeometry.h"

#include <algorithm>
#include <cmath>

namespace xjw
{

double minimumTriangulationAngleDeg(const std::vector<Camera> &cameras,
                                    const BATrack &track,
                                    const std::array<double, 3> &worldPoint)
{
    if (track.observations.size() < 2)
    {
        return 0.0;
    }

    double minimumAngleDeg = 180.0;
    for (std::size_t leftIndex = 0; leftIndex < track.observations.size(); ++leftIndex)
    {
        const BAObservation &leftObservation = track.observations[leftIndex];
        if (leftObservation.cameraIndex < 0
            || leftObservation.cameraIndex >= static_cast<int>(cameras.size()))
        {
            continue;
        }

        const std::array<double, 3> leftCenter =
            cameras[static_cast<std::size_t>(leftObservation.cameraIndex)].cameraCenter();
        const std::array<double, 3> leftVector{{leftCenter[0] - worldPoint[0],
                                                leftCenter[1] - worldPoint[1],
                                                leftCenter[2] - worldPoint[2]}};
        const double leftNorm = std::sqrt(leftVector[0] * leftVector[0]
                                        + leftVector[1] * leftVector[1]
                                        + leftVector[2] * leftVector[2]);
        if (leftNorm < 1e-12)
        {
            continue;
        }

        for (std::size_t rightIndex = leftIndex + 1;
             rightIndex < track.observations.size();
             ++rightIndex)
        {
            const BAObservation &rightObservation = track.observations[rightIndex];
            if (rightObservation.cameraIndex < 0
                || rightObservation.cameraIndex >= static_cast<int>(cameras.size()))
            {
                continue;
            }

            const std::array<double, 3> rightCenter =
                cameras[static_cast<std::size_t>(rightObservation.cameraIndex)].cameraCenter();
            const std::array<double, 3> rightVector{{rightCenter[0] - worldPoint[0],
                                                     rightCenter[1] - worldPoint[1],
                                                     rightCenter[2] - worldPoint[2]}};
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
            constexpr double radiansToDegrees = 180.0 / 3.14159265358979323846;
            minimumAngleDeg = std::min(minimumAngleDeg, std::acos(cosine) * radiansToDegrees);
        }
    }

    return std::isfinite(minimumAngleDeg) ? minimumAngleDeg : 0.0;
}

double pairRmsReprojectionErrorPx(const Camera &cameraA,
                                  const std::array<double, 2> &pixelA,
                                  const Camera &cameraB,
                                  const std::array<double, 2> &pixelB,
                                  const std::array<double, 3> &worldPoint)
{
    const double errorA = reprojectionErrorPx(cameraA, worldPoint, pixelA);
    const double errorB = reprojectionErrorPx(cameraB, worldPoint, pixelB);
    if (!std::isfinite(errorA) || !std::isfinite(errorB))
    {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(0.5 * (errorA * errorA + errorB * errorB));
}

PairIntersectionCandidate triangulatePairWithDirectionFallback(
    const Camera &cameraA,
    const std::array<double, 2> &pixelA,
    const Camera &cameraB,
    const std::array<double, 2> &pixelB)
{
    PairIntersectionCandidate bestCandidate;
    for (int flipMask = 0; flipMask < 4; ++flipMask)
    {
        Camera testCameraA = cameraA;
        Camera testCameraB = cameraB;
        if ((flipMask & 0x1) != 0)
        {
            testCameraA.setDepthAxisFlipped(!testCameraA.depthAxisFlipped());
        }
        if ((flipMask & 0x2) != 0)
        {
            testCameraB.setDepthAxisFlipped(!testCameraB.depthAxisFlipped());
        }

        const auto pairResult = Intersection::intersectPair(
            testCameraA, pixelA[0], pixelA[1], testCameraB, pixelB[0], pixelB[1]);
        if (!std::isfinite(pairResult.point[0])
            || !std::isfinite(pairResult.point[1])
            || !std::isfinite(pairResult.point[2]))
        {
            continue;
        }

        const double rmsError = pairRmsReprojectionErrorPx(
            testCameraA, pixelA, testCameraB, pixelB, pairResult.point);
        if (std::isfinite(rmsError)
            && (!bestCandidate.valid || rmsError < bestCandidate.rmsReprojectionPx))
        {
            bestCandidate.point = pairResult.point;
            bestCandidate.rmsReprojectionPx = rmsError;
            bestCandidate.valid = true;
        }
    }
    return bestCandidate;
}

} // namespace xjw
