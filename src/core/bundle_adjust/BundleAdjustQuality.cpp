#include "BundleAdjustQuality.h"

#include "BundleAdjustValidation.h"
#include "concurrency/SafeWorkerGroup.h"

#include <plamatrix/ops/statistics.h>
#include <plamatrix/ops/vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <thread>
#include <utility>

namespace xjw::detail
{
namespace
{

struct ConstraintStats
{
    int count = 0;
    double rms = 0.0;
    double median = 0.0;
};

std::array<double, 3> pointForConstraintStats(
    const std::vector<BATrack> &tracks,
    const std::vector<BARefinedPoint> *points,
    std::size_t index)
{
    if (points && index < points->size())
    {
        return (*points)[index].point;
    }
    return tracks[index].initialPoint;
}

bool trackIsEligibleForConstraintStats(
    const std::vector<BARefinedPoint> *eligiblePoints,
    std::size_t index)
{
    return !eligiblePoints ||
           (index < eligiblePoints->size() && (*eligiblePoints)[index].valid);
}

ConstraintStats computeLaserStats(
    const std::vector<BATrack> &tracks,
    const std::vector<BARefinedPoint> *points,
    const std::vector<BARefinedPoint> *eligiblePoints)
{
    std::vector<double> distances;
    double sumSquared = 0.0;
    for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
    {
        if (!trackIsEligibleForConstraintStats(eligiblePoints, trackIndex))
        {
            continue;
        }
        const auto point =
            pointForConstraintStats(tracks, points, trackIndex);
        for (const BALaserPlaneConstraint &constraint :
             tracks[trackIndex].laserPlaneConstraints)
        {
            const double distance =
                (point[0] - constraint.point[0]) * constraint.normal[0] +
                (point[1] - constraint.point[1]) * constraint.normal[1] +
                (point[2] - constraint.point[2]) * constraint.normal[2];
            if (!std::isfinite(distance))
            {
                continue;
            }
            const double absoluteDistance = std::abs(distance);
            sumSquared += distance * distance;
            distances.push_back(absoluteDistance);
        }
    }

    ConstraintStats stats;
    stats.count = static_cast<int>(distances.size());
    stats.rms = stats.count > 0
                    ? std::sqrt(sumSquared / static_cast<double>(stats.count))
                    : 0.0;
    if (!distances.empty())
    {
        stats.median =
            plamatrix::finiteMedian(std::move(distances)).value_or(0.0);
    }
    return stats;
}

double computedLaserRange(const FramePinholeCamera &camera,
                          const std::array<double, 3> &leverArmCameraMeters,
                          const std::array<double, 3> &point)
{
    const std::array<double, 3> center = camera.cameraCenter();
    const std::array<double, 9> rotation = camera.cameraToWorldRotation();
    std::array<double, 3> emitter = center;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            emitter[static_cast<size_t>(row)] +=
                rotation[static_cast<size_t>(row * 3 + column)] *
                leverArmCameraMeters[static_cast<size_t>(column)];
        }
    }

    const double dx = point[0] - emitter[0];
    const double dy = point[1] - emitter[1];
    const double dz = point[2] - emitter[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void updateLaserRangeStats(const std::vector<FramePinholeCamera> &inputCameras,
                           const std::vector<FramePinholeCamera> &refinedCameras,
                           const BAOptions &options,
                           BAResult *result)
{
    if (!options.enableLaserRangeConstraints)
    {
        return;
    }

    const auto &constraints = options.laserRangeConstraints;
    if (result->laserRangeShots.size() != constraints.size())
    {
        result->laserRangeShots.resize(constraints.size());
    }

    double sumSquaredBefore = 0.0;
    double sumSquaredAfter = 0.0;
    bool finiteBefore = true;
    bool finiteAfter = true;
    for (size_t shotIndex = 0; shotIndex < constraints.size(); ++shotIndex)
    {
        const BALaserRangeConstraint &constraint = constraints[shotIndex];
        BARefinedLaserRangeShot &shot = result->laserRangeShots[shotIndex];
        shot.shotId = constraint.shotId;
        shot.ephemerisTimeSeconds = constraint.ephemerisTimeSeconds;
        shot.sourceIndex = constraint.sourceIndex;
        shot.pointMode = constraint.pointMode;

        const double beforeRange = computedLaserRange(
            inputCameras[static_cast<size_t>(constraint.cameraIndex)],
            constraint.leverArmCameraMeters,
            constraint.initialPoint);
        const double beforeResidual =
            beforeRange - constraint.observedRangeMeters;
        shot.computedRangeBeforeMeters = beforeRange;
        shot.residualBeforeMeters = beforeResidual;
        finiteBefore = finiteBefore && std::isfinite(beforeResidual);
        if (std::isfinite(beforeResidual))
        {
            sumSquaredBefore += beforeResidual * beforeResidual;
        }

        const bool validPoint =
            shot.valid &&
            plamatrix::isFinite(plamatrix::Vec3<double>(shot.point));
        const double afterRange = validPoint
                                      ? computedLaserRange(
                                            refinedCameras[static_cast<size_t>(
                                                constraint.cameraIndex)],
                                            constraint.leverArmCameraMeters,
                                            shot.point)
                                      : std::numeric_limits<double>::infinity();
        const double afterResidual =
            afterRange - constraint.observedRangeMeters;
        shot.computedRangeAfterMeters = afterRange;
        shot.residualAfterMeters = afterResidual;
        shot.normalizedResidualAfter =
            afterResidual / constraint.sigmaRangeMeters;
        shot.valid = validPoint &&
                     std::isfinite(afterResidual) &&
                     std::isfinite(shot.normalizedResidualAfter);
        finiteAfter = finiteAfter && shot.valid;
        if (shot.valid)
        {
            sumSquaredAfter += afterResidual * afterResidual;
        }
    }

    result->laserRangeConstraintCount =
        static_cast<int>(constraints.size());
    if (constraints.empty())
    {
        result->laserRangeRmsBeforeMeters = 0.0;
        result->laserRangeRmsAfterMeters = 0.0;
        return;
    }
    result->laserRangeRmsBeforeMeters =
        finiteBefore
            ? std::sqrt(sumSquaredBefore / static_cast<double>(constraints.size()))
            : std::numeric_limits<double>::infinity();
    result->laserRangeRmsAfterMeters =
        finiteAfter
            ? std::sqrt(sumSquaredAfter / static_cast<double>(constraints.size()))
            : std::numeric_limits<double>::infinity();
}

ConstraintStats computeControlPointStats(
    const std::vector<BATrack> &tracks,
    const std::vector<BARefinedPoint> *points,
    const std::vector<BARefinedPoint> *eligiblePoints)
{
    double sumSquared = 0.0;
    int count = 0;
    for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
    {
        if (!trackIsEligibleForConstraintStats(eligiblePoints, trackIndex))
        {
            continue;
        }
        const auto point =
            pointForConstraintStats(tracks, points, trackIndex);
        for (const BAControlPointConstraint &constraint :
             tracks[trackIndex].controlPointConstraints)
        {
            const double dx = point[0] - constraint.point[0];
            const double dy = point[1] - constraint.point[1];
            const double dz = point[2] - constraint.point[2];
            const double distanceSquared = dx * dx + dy * dy + dz * dz;
            if (!std::isfinite(distanceSquared))
            {
                continue;
            }
            sumSquared += distanceSquared;
            ++count;
        }
    }

    ConstraintStats stats;
    stats.count = count;
    stats.rms = count > 0
                    ? std::sqrt(sumSquared / static_cast<double>(count))
                    : 0.0;
    return stats;
}

bool scaleBarIsUsable(const BAScaleBarConstraint &constraint,
                      std::size_t trackCount)
{
    return constraint.trackIndexA >= 0 &&
           constraint.trackIndexB >= 0 &&
           constraint.trackIndexA != constraint.trackIndexB &&
           static_cast<std::size_t>(constraint.trackIndexA) < trackCount &&
           static_cast<std::size_t>(constraint.trackIndexB) < trackCount &&
           std::isfinite(constraint.measuredDistanceMeters) &&
           constraint.measuredDistanceMeters > 0.0;
}

ConstraintStats computeScaleBarStats(
    const std::vector<BATrack> &tracks,
    const std::vector<BARefinedPoint> *points,
    const std::vector<BARefinedPoint> *eligiblePoints,
    const std::vector<BAScaleBarConstraint> &constraints)
{
    double sumSquared = 0.0;
    int count = 0;
    for (const BAScaleBarConstraint &constraint : constraints)
    {
        if (!scaleBarIsUsable(constraint, tracks.size()))
        {
            continue;
        }
        if (!trackIsEligibleForConstraintStats(
                eligiblePoints,
                static_cast<std::size_t>(constraint.trackIndexA)) ||
            !trackIsEligibleForConstraintStats(
                eligiblePoints,
                static_cast<std::size_t>(constraint.trackIndexB)))
        {
            continue;
        }

        const auto pointA = pointForConstraintStats(
            tracks, points, static_cast<std::size_t>(constraint.trackIndexA));
        const auto pointB = pointForConstraintStats(
            tracks, points, static_cast<std::size_t>(constraint.trackIndexB));
        const double dx = pointA[0] - pointB[0];
        const double dy = pointA[1] - pointB[1];
        const double dz = pointA[2] - pointB[2];
        const double residual =
            std::sqrt(dx * dx + dy * dy + dz * dz) -
            constraint.measuredDistanceMeters;
        if (!std::isfinite(residual))
        {
            continue;
        }
        sumSquared += residual * residual;
        ++count;
    }

    ConstraintStats stats;
    stats.count = count;
    stats.rms = count > 0
                    ? std::sqrt(sumSquared / static_cast<double>(count))
                    : 0.0;
    return stats;
}

void updateConstraintStats(const std::vector<FramePinholeCamera> &inputCameras,
                           const std::vector<FramePinholeCamera> &refinedCameras,
                           const std::vector<BATrack> &tracks,
                           const BAOptions &options,
                           BAResult *result)
{
    updateLaserRangeStats(inputCameras, refinedCameras, options, result);
    if (options.enableLaserPlaneConstraints)
    {
        const ConstraintStats before =
            computeLaserStats(tracks, nullptr, &result->points);
        const ConstraintStats after =
            computeLaserStats(tracks, &result->points, &result->points);
        result->laserConstraintCount = before.count;
        result->laserRmsBeforeMeters = before.rms;
        result->laserRmsAfterMeters = after.rms;
        result->laserMedianBeforeMeters = before.median;
        result->laserMedianAfterMeters = after.median;
    }
    if (options.enableControlPointConstraints)
    {
        const ConstraintStats before =
            computeControlPointStats(tracks, nullptr, &result->points);
        const ConstraintStats after =
            computeControlPointStats(tracks, &result->points, &result->points);
        result->controlPointConstraintCount = before.count;
        result->controlPointRmsBeforeMeters = before.rms;
        result->controlPointRmsAfterMeters = after.rms;
    }
    if (options.enableScaleBarConstraints)
    {
        const ConstraintStats before = computeScaleBarStats(
            tracks, nullptr, &result->points, options.scaleBarConstraints);
        const ConstraintStats after = computeScaleBarStats(
            tracks, &result->points, &result->points, options.scaleBarConstraints);
        result->scaleBarConstraintCount = before.count;
        result->scaleBarRmsBeforeMeters = before.rms;
        result->scaleBarRmsAfterMeters = after.rms;
    }
}

double strictTrackRms(const std::vector<FramePinholeCamera> &cameras,
                      const BATrack &track,
                      const std::array<double, 3> &point)
{
    if (!plamatrix::isFinite(plamatrix::Vec3<double>(point)))
    {
        return std::numeric_limits<double>::infinity();
    }

    // “严格”表示任何有效观测投影失败都会拒绝整条 track，而不是忽略该观测后
    // 继续平均。否则后端可通过把困难观测推到相机后方来获得虚假的低 RMS。
    double sumSquared = 0.0;
    int residualCount = 0;
    int firstCameraIndex = -1;
    bool hasSecondDistinctCamera = false;
    for (const BAObservation &observation : track.observations)
    {
        const double weight = sanitizedObservationWeight(observation);
        if (!(weight > 0.0))
        {
            continue;
        }
        if (observation.cameraIndex < 0 ||
            observation.cameraIndex >= static_cast<int>(cameras.size()) ||
            !std::isfinite(observation.u) ||
            !std::isfinite(observation.v))
        {
            return std::numeric_limits<double>::infinity();
        }

        const double world[3] = {point[0], point[1], point[2]};
        double pixel[2] = {0.0, 0.0};
        if (!cameras[static_cast<size_t>(observation.cameraIndex)]
                 .projectWorldPoint(world, pixel) ||
            !std::isfinite(pixel[0]) ||
            !std::isfinite(pixel[1]))
        {
            // 任一参与残差的观测位于相机后方时，整条 track 均不可用。
            return std::numeric_limits<double>::infinity();
        }

        const double du = pixel[0] - observation.u;
        const double dv = pixel[1] - observation.v;
        sumSquared += weight * (du * du + dv * dv);
        residualCount += 2;
        if (firstCameraIndex < 0)
        {
            firstCameraIndex = observation.cameraIndex;
        }
        else if (observation.cameraIndex != firstCameraIndex)
        {
            hasSecondDistinctCamera = true;
        }
    }

    if (residualCount < 4 || !hasSecondDistinctCamera)
    {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(sumSquared / static_cast<double>(residualCount));
}

void appendBackendMessage(BAResult *result, const std::string &message)
{
    if (!result || message.empty())
    {
        return;
    }
    if (!result->backendMessage.empty())
    {
        result->backendMessage += "；";
    }
    result->backendMessage += message;
}

} // namespace

double adaptivePointFilterThreshold(const std::vector<double> &pointRms,
                                    double absoluteFloor,
                                    double medianFactor)
{
    std::vector<double> finiteValues;
    finiteValues.reserve(pointRms.size());
    for (const double value : pointRms)
    {
        if (std::isfinite(value) && value >= 0.0)
        {
            finiteValues.push_back(value);
        }
    }

    double threshold =
        std::isfinite(absoluteFloor) && absoluteFloor >= 0.0
            ? absoluteFloor
            : 0.0;
    if (finiteValues.empty() ||
        !std::isfinite(medianFactor) ||
        !(medianFactor > 0.0))
    {
        return threshold;
    }

    const double median =
        plamatrix::finiteMedian(std::move(finiteValues)).value_or(0.0);
    return std::max(threshold, medianFactor * median);
}

void finalizeBundleAdjustResult(const std::vector<FramePinholeCamera> &inputCameras,
                                const std::vector<BATrack> &tracks,
                                const BAOptions &options,
                                BAResult *result)
{
    if (!result)
    {
        return;
    }

    result->totalTracks = static_cast<int>(tracks.size());
    if (result->points.size() != tracks.size())
    {
        result->points.resize(tracks.size());
    }
    const std::vector<FramePinholeCamera> &refinedCameras =
        result->refinedCameras.size() == inputCameras.size()
            ? result->refinedCameras
            : inputCameras;

    // 先对所有后端结果使用同一投影实现重算前后 RMS。后端内部 cost 可能包含
    // Huber、控制点或位姿先验，因此不能直接拿求解器 cost 作为像素质量指标。
    std::vector<double> candidateRmsByTrack(
        tracks.size(), std::numeric_limits<double>::infinity());
    const std::size_t qualityWorkerCount = static_cast<std::size_t>(
        options.numThreads > 0
            ? options.numThreads
            : std::max(1u, std::thread::hardware_concurrency()));
    common::concurrency::parallelForIndices(
        tracks.size(), qualityWorkerCount, [&](std::size_t index)
        {
            BARefinedPoint &point = result->points[index];
            point.rmsBefore = strictTrackRms(
                inputCameras, tracks[index], tracks[index].initialPoint);
            if (!point.valid)
            {
                point.rmsAfter = std::numeric_limits<double>::infinity();
                return;
            }

            point.rmsAfter = strictTrackRms(
                refinedCameras, tracks[index], point.point);
            point.valid =
                plamatrix::isFinite(plamatrix::Vec3<double>(point.point)) &&
                std::isfinite(point.rmsAfter);
            if (point.valid)
            {
                candidateRmsByTrack[index] = point.rmsAfter;
            }
        });

    std::vector<double> candidateRms;
    candidateRms.reserve(tracks.size());
    for (double rms : candidateRmsByTrack)
    {
        if (std::isfinite(rms))
        {
            candidateRms.push_back(rms);
        }
    }

    // 阈值在全部候选点上一次性估计，随后统一过滤，避免遍历顺序影响中位数。
    const double filterThreshold =
        options.enablePointFilter
            ? adaptivePointFilterThreshold(
                  candidateRms,
                  options.filterMaxReprojError,
                  options.filterSigmaFactor)
            : std::numeric_limits<double>::infinity();

    result->optimizedTracks = 0;
    double sumBefore = 0.0;
    double sumAfter = 0.0;
    for (BARefinedPoint &point : result->points)
    {
        if (point.valid && point.rmsAfter > filterThreshold)
        {
            point.valid = false;
        }
        if (point.valid)
        {
            ++result->optimizedTracks;
            sumBefore += point.rmsBefore;
            sumAfter += point.rmsAfter;
        }
    }

    result->meanRmsBefore =
        result->optimizedTracks > 0
            ? sumBefore / static_cast<double>(result->optimizedTracks)
            : std::numeric_limits<double>::infinity();
    result->meanRmsAfter =
        result->optimizedTracks > 0
            ? sumAfter / static_cast<double>(result->optimizedTracks)
            : std::numeric_limits<double>::infinity();
    result->validTrackRatio =
        result->totalTracks > 0
            ? static_cast<double>(result->optimizedTracks) /
                  static_cast<double>(result->totalTracks)
            : 0.0;

    updateConstraintStats(inputCameras, refinedCameras, tracks, options, result);

    // 数值求解器的“成功”只说明迭代正常结束。若摄影测量门控未保留任何点，
    // 必须把结果降级，防止 SfM 继续使用空或负深度模型。
    if (result->optimizedTracks == 0 &&
        result->laserRangeConstraintCount == 0 &&
        (result->solveStatus == BASolveStatus::Success ||
         result->solveStatus == BASolveStatus::NoConvergence))
    {
        result->solveStatus = BASolveStatus::NumericalFailure;
        result->solutionUsable = false;
        appendBackendMessage(result, "BA 最终质量检查未保留任何有效 track");
    }
}

bool constraintRmsPassesQualityGate(int constraintCount,
                                    double rmsBefore,
                                    double rmsAfter,
                                    double maxGrowth,
                                    const char *constraintName,
                                    std::string *message)
{
    if (constraintCount <= 0)
    {
        return true;
    }

    const std::string name =
        constraintName && constraintName[0] != '\0'
            ? constraintName
            : "物方约束";
    if (!std::isfinite(rmsBefore) || !std::isfinite(rmsAfter))
    {
        if (message)
        {
            *message = "质量门控拒绝: " + name + " RMS 非有限";
        }
        return false;
    }

    const double growth = std::max(1.0, maxGrowth);
    if (rmsBefore > 1.0e-12)
    {
        if (rmsAfter > rmsBefore * growth)
        {
            if (message)
            {
                *message = "质量门控拒绝: " + name + " RMS 增长超过阈值";
            }
            return false;
        }
    }
    else if (rmsAfter > 1.0e-9)
    {
        if (message)
        {
            *message = "质量门控拒绝: " + name + " 零残差被优化为非零残差";
        }
        return false;
    }
    return true;
}

} // namespace xjw::detail
