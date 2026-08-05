#include "BundleAdjustQuality.h"

#include "BundleAdjustValidation.h"

#include <plamatrix/ops/statistics.h>
#include <plamatrix/ops/vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
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
    if (points && index < points->size() && (*points)[index].valid)
    {
        return (*points)[index].point;
    }
    return tracks[index].initialPoint;
}

ConstraintStats computeLaserStats(
    const std::vector<BATrack> &tracks,
    const std::vector<BARefinedPoint> *points)
{
    std::vector<double> distances;
    double sumSquared = 0.0;
    for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
    {
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

ConstraintStats computeControlPointStats(
    const std::vector<BATrack> &tracks,
    const std::vector<BARefinedPoint> *points)
{
    double sumSquared = 0.0;
    int count = 0;
    for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
    {
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

void updateConstraintStats(const std::vector<BATrack> &tracks,
                           const BAOptions &options,
                           BAResult *result)
{
    if (options.enableLaserPlaneConstraints)
    {
        const ConstraintStats before = computeLaserStats(tracks, nullptr);
        const ConstraintStats after = computeLaserStats(tracks, &result->points);
        result->laserConstraintCount = before.count;
        result->laserRmsBeforeMeters = before.rms;
        result->laserRmsAfterMeters = after.rms;
        result->laserMedianBeforeMeters = before.median;
        result->laserMedianAfterMeters = after.median;
    }
    if (options.enableControlPointConstraints)
    {
        const ConstraintStats before =
            computeControlPointStats(tracks, nullptr);
        const ConstraintStats after =
            computeControlPointStats(tracks, &result->points);
        result->controlPointConstraintCount = before.count;
        result->controlPointRmsBeforeMeters = before.rms;
        result->controlPointRmsAfterMeters = after.rms;
    }
    if (options.enableScaleBarConstraints)
    {
        const ConstraintStats before = computeScaleBarStats(
            tracks, nullptr, options.scaleBarConstraints);
        const ConstraintStats after = computeScaleBarStats(
            tracks, &result->points, options.scaleBarConstraints);
        result->scaleBarConstraintCount = before.count;
        result->scaleBarRmsBeforeMeters = before.rms;
        result->scaleBarRmsAfterMeters = after.rms;
    }
}

double strictTrackRms(const std::vector<Camera> &cameras,
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
    std::set<int> uniqueCameras;
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
        uniqueCameras.insert(observation.cameraIndex);
    }

    if (residualCount < 4 || uniqueCameras.size() < 2)
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

void finalizeBundleAdjustResult(const std::vector<Camera> &inputCameras,
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
    const std::vector<Camera> &refinedCameras =
        result->refinedCameras.size() == inputCameras.size()
            ? result->refinedCameras
            : inputCameras;

    // 先对所有后端结果使用同一投影实现重算前后 RMS。后端内部 cost 可能包含
    // Huber、控制点或位姿先验，因此不能直接拿求解器 cost 作为像素质量指标。
    std::vector<double> candidateRms;
    candidateRms.reserve(tracks.size());
    double sumBefore = 0.0;
    int countBefore = 0;
    for (size_t index = 0; index < tracks.size(); ++index)
    {
        BARefinedPoint &point = result->points[index];
        point.rmsBefore = strictTrackRms(
            inputCameras, tracks[index], tracks[index].initialPoint);
        if (std::isfinite(point.rmsBefore))
        {
            sumBefore += point.rmsBefore;
            ++countBefore;
        }

        if (!point.valid)
        {
            point.rmsAfter = std::numeric_limits<double>::infinity();
            continue;
        }

        point.rmsAfter = strictTrackRms(
            refinedCameras, tracks[index], point.point);
        point.valid =
            plamatrix::isFinite(plamatrix::Vec3<double>(point.point)) &&
            std::isfinite(point.rmsAfter);
        if (point.valid)
        {
            candidateRms.push_back(point.rmsAfter);
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
            sumAfter += point.rmsAfter;
        }
    }

    result->meanRmsBefore =
        countBefore > 0
            ? sumBefore / static_cast<double>(countBefore)
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

    updateConstraintStats(tracks, options, result);

    // 数值求解器的“成功”只说明迭代正常结束。若摄影测量门控未保留任何点，
    // 必须把结果降级，防止 SfM 继续使用空或负深度模型。
    if (result->optimizedTracks == 0 &&
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
