#include "BundleAdjustQuality.h"

#include <plamatrix/ops/statistics.h>
#include <plamatrix/ops/vector.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace xjw::detail
{
namespace
{

double observationWeight(const BAObservation &observation)
{
    if (!std::isfinite(observation.weight))
    {
        return 0.0;
    }
    return std::max(0.0, observation.weight);
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
        const double weight = observationWeight(observation);
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
