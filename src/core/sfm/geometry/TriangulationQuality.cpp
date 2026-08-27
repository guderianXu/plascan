#include "TriangulationQuality.h"

#include "CameraBaseline.h"
#include "Intersection.h"
#include "geometry/ProjectionGeometry.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>

namespace xjw
{

namespace
{

constexpr double kMaximumReconstructionUncertainty = 1.0e6;

bool pointProjectionJacobian(const FramePinholeCamera &camera,
                             const std::array<double, 3> &worldPoint,
                             cv::Matx<double, 2, 3> *jacobian)
{
    if (!jacobian)
    {
        return false;
    }

    const std::array<double, 3> center = camera.cameraCenter();
    const double range = std::hypot(worldPoint[0] - center[0],
                                    worldPoint[1] - center[1],
                                    worldPoint[2] - center[2]);
    if (!std::isfinite(range) || range <= 1e-9)
    {
        return false;
    }

    const double step = std::max(1e-7, range * 1e-6);
    for (int axis = 0; axis < 3; ++axis)
    {
        std::array<double, 3> plus = worldPoint;
        std::array<double, 3> minus = worldPoint;
        plus[static_cast<std::size_t>(axis)] += step;
        minus[static_cast<std::size_t>(axis)] -= step;
        double projectedPlus[2]{};
        double projectedMinus[2]{};
        const bool plusProjected = camera.projectWorldPoint(plus.data(), projectedPlus)
            || camera.projectWorldPointSigned(plus.data(), projectedPlus);
        const bool minusProjected = camera.projectWorldPoint(minus.data(), projectedMinus)
            || camera.projectWorldPointSigned(minus.data(), projectedMinus);
        if (!plusProjected || !minusProjected)
        {
            return false;
        }
        for (int pixelAxis = 0; pixelAxis < 2; ++pixelAxis)
        {
            const double derivative =
                (projectedPlus[pixelAxis] - projectedMinus[pixelAxis]) / (2.0 * step);
            if (!std::isfinite(derivative))
            {
                return false;
            }
            (*jacobian)(pixelAxis, axis) = derivative;
        }
    }
    return true;
}

} // namespace

double reconstructionUncertainty(
    const std::vector<TiePointQualityObservation> &observations,
    const std::array<double, 3> &worldPoint)
{
    cv::Matx33d information = cv::Matx33d::zeros();
    int validObservationCount = 0;
    for (const TiePointQualityObservation &observation : observations)
    {
        if (!observation.camera)
        {
            continue;
        }
        cv::Matx<double, 2, 3> jacobian;
        if (!pointProjectionJacobian(*observation.camera, worldPoint, &jacobian))
        {
            continue;
        }
        const double scale = std::isfinite(observation.measurementScale)
                && observation.measurementScale > 0.0
            ? observation.measurementScale
            : 1.0;
        information += (jacobian.t() * jacobian) / (scale * scale);
        ++validObservationCount;
    }
    if (validObservationCount < 2)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    cv::Mat eigenvalues;
    if (!cv::eigen(cv::Mat(information), eigenvalues) || eigenvalues.total() != 3)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double largest = eigenvalues.at<double>(0);
    const double smallest = eigenvalues.at<double>(2);
    if (!std::isfinite(largest) || !std::isfinite(smallest) || largest <= 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double stableSmallest = std::max(smallest, largest / (kMaximumReconstructionUncertainty
                                                                 * kMaximumReconstructionUncertainty));
    return std::sqrt(largest / stableSmallest);
}

double projectionAccuracy(
    const std::vector<TiePointQualityObservation> &observations)
{
    if (observations.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (const TiePointQualityObservation &observation : observations)
    {
        if (!std::isfinite(observation.measurementScale)
            || observation.measurementScale <= 0.0)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        sum += observation.measurementScale;
    }
    return sum / static_cast<double>(observations.size());
}

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
