#include "TriangulationQuality.h"

#include "CameraBaseline.h"
#include "Intersection.h"
#include "geometry/ProjectionGeometry.h"

#include <algorithm>
#include <cmath>

namespace xjw
{

double minimumTriangulationAngleDeg(const std::vector<FramePinholeCamera> &cameras,
                                    const BATrack &track,
                                    const std::array<double, 3> &worldPoint)
{
    if (track.observations.size() < 2)
    {
        return 0.0;
    }

    // 对轨迹中所有有效相机对取最小角，作为保守稳定性指标。最大角只能证明
    // 某一对基线良好，无法发现其余观测几乎共线。
    double minimumAngleDeg = 180.0;
    for (std::size_t leftIndex = 0; leftIndex < track.observations.size(); ++leftIndex)
    {
        const BAObservation &leftObservation = track.observations[leftIndex];
        if (leftObservation.cameraIndex < 0
            || leftObservation.cameraIndex >= static_cast<int>(cameras.size()))
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

            const CameraBaseline baseline = CameraBaseline::evaluate(
                cameras[static_cast<std::size_t>(leftObservation.cameraIndex)],
                cameras[static_cast<std::size_t>(rightObservation.cameraIndex)],
                worldPoint);
            if (!baseline.isValid()
                || !baseline.hasPointGeometry()
                || !baseline.isPointInFrontOfBothCameras()
                || !baseline.triangulationAngleDeg().has_value())
            {
                continue;
            }
            minimumAngleDeg = std::min(minimumAngleDeg, *baseline.triangulationAngleDeg());
        }
    }

    return std::isfinite(minimumAngleDeg) ? minimumAngleDeg : 0.0;
}

double pairRmsReprojectionErrorPx(const FramePinholeCamera &cameraA,
                                  const std::array<double, 2> &pixelA,
                                  const FramePinholeCamera &cameraB,
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
    const FramePinholeCamera &cameraA,
    const std::array<double, 2> &pixelA,
    const FramePinholeCamera &cameraB,
    const std::array<double, 2> &pixelB)
{
    PairIntersectionCandidate bestCandidate;
    // 历史 .tsai/工程相机可能错误标记局部前向轴。四种组合只用于寻找可优化初值，
    // 输入相机保持不变，后续严格正深度检查仍会暴露元数据问题。
    for (int flipMask = 0; flipMask < 4; ++flipMask)
    {
        FramePinholeCamera testCameraA = cameraA;
        FramePinholeCamera testCameraB = cameraB;
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
