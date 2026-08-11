#include "BundleAdjustValidation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace xjw::detail
{
namespace
{

// 位姿先验消除全局刚体漂移；尺度是否同时受约束由后续的基线分析单独判断。
// 约束数值是否合法和最终是否满足仍由后端及统一质量门控负责。
bool hasEnabledPosePrior(const BAOptions &options)
{
    return std::any_of(
        options.cameraPosePriors.begin(),
        options.cameraPosePriors.end(),
        [](const BACameraPosePrior &prior)
        {
            return prior.enabled;
        });
}

bool isFinitePoint(const std::array<double, 3> &point)
{
    return std::all_of(point.begin(), point.end(), [](double value)
    {
        return std::isfinite(value);
    });
}

bool isUsableLaserPointSqrtInformation(
    const std::array<double, 9> &sqrtInformation)
{
    double maxAbsoluteValue = 0.0;
    for (const double value : sqrtInformation)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
        maxAbsoluteValue = std::max(maxAbsoluteValue, std::abs(value));
    }
    if (!(maxAbsoluteValue > 0.0))
    {
        return false;
    }

    const double determinant =
        sqrtInformation[0] *
            (sqrtInformation[4] * sqrtInformation[8] -
             sqrtInformation[5] * sqrtInformation[7]) -
        sqrtInformation[1] *
            (sqrtInformation[3] * sqrtInformation[8] -
             sqrtInformation[5] * sqrtInformation[6]) +
        sqrtInformation[2] *
            (sqrtInformation[3] * sqrtInformation[7] -
             sqrtInformation[4] * sqrtInformation[6]);
    const double relativeRankThreshold =
        1.0e-12 * maxAbsoluteValue * maxAbsoluteValue * maxAbsoluteValue;
    return std::isfinite(determinant) &&
           std::abs(determinant) > relativeRankThreshold;
}

bool hasMetricPosePriorBaseline(const BAOptions &options)
{
    for (size_t left = 0; left < options.cameraPosePriors.size(); ++left)
    {
        const BACameraPosePrior &first = options.cameraPosePriors[left];
        if (!first.enabled || !isFinitePoint(first.cameraCenter))
        {
            continue;
        }
        for (size_t right = left + 1; right < options.cameraPosePriors.size(); ++right)
        {
            const BACameraPosePrior &second = options.cameraPosePriors[right];
            if (!second.enabled || !isFinitePoint(second.cameraCenter))
            {
                continue;
            }
            double squared_distance = 0.0;
            for (int axis = 0; axis < 3; ++axis)
            {
                const double delta = first.cameraCenter[axis] - second.cameraCenter[axis];
                squared_distance += delta * delta;
            }
            if (squared_distance > 1.0e-16)
            {
                return true;
            }
        }
    }
    return false;
}

bool hasUsableTrackObservations(const BATrack &track)
{
    std::set<int> camera_indices;
    for (const BAObservation &observation : track.observations)
    {
        if (observation.cameraIndex >= 0 &&
            std::isfinite(observation.u) &&
            std::isfinite(observation.v) &&
            std::isfinite(observation.weight) &&
            observation.weight > 0.0)
        {
            camera_indices.insert(observation.cameraIndex);
        }
    }
    return camera_indices.size() >= 2;
}

double squaredDistance(const std::array<double, 3> &left,
                       const std::array<double, 3> &right)
{
    double squared_distance = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double delta = left[axis] - right[axis];
        squared_distance += delta * delta;
    }
    return squared_distance;
}

bool hasPosePriorBaselineToFixedCamera(const std::vector<Camera> &cameras,
                                       const BAOptions &options,
                                       const std::set<int> &fixedCameras)
{
    const size_t prior_count = std::min(cameras.size(), options.cameraPosePriors.size());
    for (size_t prior_index = 0; prior_index < prior_count; ++prior_index)
    {
        const BACameraPosePrior &prior = options.cameraPosePriors[prior_index];
        if (!prior.enabled || !isFinitePoint(prior.cameraCenter))
        {
            continue;
        }
        for (const int fixed_index : fixedCameras)
        {
            if (fixed_index == static_cast<int>(prior_index))
            {
                continue;
            }
            const std::array<double, 3> center =
                cameras[static_cast<size_t>(fixed_index)].cameraCenter();
            if (isFinitePoint(center) &&
                squaredDistance(prior.cameraCenter, center) > 1.0e-16)
            {
                return true;
            }
        }
    }
    return false;
}

bool hasControlPointBaselineToFixedCamera(const std::vector<Camera> &cameras,
                                          const std::vector<BATrack> &tracks,
                                          const BAOptions &options,
                                          const std::set<int> &fixedCameras)
{
    if (!options.enableControlPointConstraints)
    {
        return false;
    }
    for (const BATrack &track : tracks)
    {
        if (!hasUsableTrackObservations(track))
        {
            continue;
        }
        for (const BAControlPointConstraint &constraint : track.controlPointConstraints)
        {
            if (!isFinitePoint(constraint.point) ||
                !std::isfinite(constraint.sigmaMeters) ||
                constraint.sigmaMeters <= 0.0 ||
                !std::isfinite(constraint.weight) ||
                constraint.weight <= 0.0)
            {
                continue;
            }
            for (const int fixed_index : fixedCameras)
            {
                const std::array<double, 3> center =
                    cameras[static_cast<size_t>(fixed_index)].cameraCenter();
                if (isFinitePoint(center) &&
                    squaredDistance(constraint.point, center) > 1.0e-16)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

bool hasCompleteControlPointGauge(const std::vector<BATrack> &tracks,
                                  const BAOptions &options)
{
    if (!options.enableControlPointConstraints)
    {
        return false;
    }

    std::vector<std::array<double, 3>> points;
    for (const BATrack &track : tracks)
    {
        if (!hasUsableTrackObservations(track))
        {
            continue;
        }
        for (const BAControlPointConstraint &constraint : track.controlPointConstraints)
        {
            if (isFinitePoint(constraint.point) &&
                std::isfinite(constraint.sigmaMeters) &&
                constraint.sigmaMeters > 0.0 &&
                std::isfinite(constraint.weight) &&
                constraint.weight > 0.0)
            {
                points.push_back(constraint.point);
            }
        }
    }

    for (size_t first = 0; first < points.size(); ++first)
    {
        for (size_t second = first + 1; second < points.size(); ++second)
        {
            const std::array<double, 3> ab{{
                points[second][0] - points[first][0],
                points[second][1] - points[first][1],
                points[second][2] - points[first][2],
            }};
            const double ab_squared = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
            if (ab_squared <= 1.0e-16)
            {
                continue;
            }
            for (size_t third = second + 1; third < points.size(); ++third)
            {
                const std::array<double, 3> ac{{
                    points[third][0] - points[first][0],
                    points[third][1] - points[first][1],
                    points[third][2] - points[first][2],
                }};
                const double ac_squared = ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2];
                const std::array<double, 3> cross{{
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                }};
                const double cross_squared =
                    cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
                if (ac_squared > 1.0e-16 &&
                    cross_squared > 1.0e-12 * ab_squared * ac_squared)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

bool hasUsableScaleBar(const std::vector<BATrack> &tracks,
                       const BAOptions &options)
{
    if (!options.enableScaleBarConstraints)
    {
        return false;
    }
    return std::any_of(
        options.scaleBarConstraints.begin(),
        options.scaleBarConstraints.end(),
        [&tracks](const BAScaleBarConstraint &constraint)
        {
            return constraint.trackIndexA >= 0 &&
                   constraint.trackIndexB >= 0 &&
                   constraint.trackIndexA != constraint.trackIndexB &&
                   constraint.trackIndexA < static_cast<int>(tracks.size()) &&
                   constraint.trackIndexB < static_cast<int>(tracks.size()) &&
                   std::isfinite(constraint.measuredDistanceMeters) &&
                   constraint.measuredDistanceMeters > 0.0;
        });
}

int farthestCameraFrom(const std::vector<Camera> &cameras,
                       int anchorIndex,
                       const std::set<int> &excluded)
{
    if (anchorIndex < 0 || anchorIndex >= static_cast<int>(cameras.size()))
    {
        return -1;
    }

    // 自动尺度锚优先选择与首锚距离最远的相机，以减少近零基线对 gauge 的数值放大。
    const auto anchor = cameras[static_cast<size_t>(anchorIndex)].cameraCenter();
    int bestIndex = -1;
    double bestDistanceSquared = 1.0e-20;
    for (int index = 0; index < static_cast<int>(cameras.size()); ++index)
    {
        if (excluded.count(index) > 0)
        {
            continue;
        }

        const auto center = cameras[static_cast<size_t>(index)].cameraCenter();
        const double dx = center[0] - anchor[0];
        const double dy = center[1] - anchor[1];
        const double dz = center[2] - anchor[2];
        const double distanceSquared = dx * dx + dy * dy + dz * dz;
        if (std::isfinite(distanceSquared) && distanceSquared > bestDistanceSquared)
        {
            bestDistanceSquared = distanceSquared;
            bestIndex = index;
        }
    }
    return bestIndex;
}

BundleAdjustValidationResult invalid(BASolveStatus status, std::string message)
{
    BundleAdjustValidationResult result;
    result.ok = false;
    result.status = status;
    result.message = std::move(message);
    return result;
}

} // namespace

double sanitizedObservationWeight(const BAObservation &observation)
{
    return std::isfinite(observation.weight) && observation.weight > 0.0
               ? observation.weight
               : 0.0;
}

bool observationIsUsable(const BAObservation &observation,
                         std::size_t cameraCount)
{
    return observation.cameraIndex >= 0 &&
           static_cast<std::size_t>(observation.cameraIndex) < cameraCount &&
           observationDataIsUsable(observation);
}

bool observationDataIsUsable(const BAObservation &observation)
{
    return std::isfinite(observation.u) &&
           std::isfinite(observation.v) &&
           sanitizedObservationWeight(observation) > 0.0;
}

BAProblemStats summarizeUsableProblem(const std::vector<Camera> &cameras,
                                      const std::vector<BATrack> &tracks)
{
    BAProblemStats stats;
    stats.cameraCount = static_cast<int>(cameras.size());

    std::vector<int> cameraGeneration(cameras.size(), -1);
    int generation = 0;
    for (const BATrack &track : tracks)
    {
        if (!std::isfinite(track.initialPoint[0]) ||
            !std::isfinite(track.initialPoint[1]) ||
            !std::isfinite(track.initialPoint[2]))
        {
            continue;
        }

        ++generation;
        int observationCount = 0;
        int uniqueCameraCount = 0;
        for (const BAObservation &observation : track.observations)
        {
            if (!observationIsUsable(observation, cameras.size()))
            {
                continue;
            }

            ++observationCount;
            const std::size_t cameraIndex =
                static_cast<std::size_t>(observation.cameraIndex);
            if (cameraGeneration[cameraIndex] != generation)
            {
                cameraGeneration[cameraIndex] = generation;
                ++uniqueCameraCount;
            }
        }

        if (observationCount >= 2 && uniqueCameraCount >= 2)
        {
            ++stats.trackCount;
            stats.observationCount += observationCount;
        }
    }
    return stats;
}

BundleAdjustValidationResult validateAndNormalizeBundleAdjustOptions(
    const std::vector<Camera> &cameras,
    const std::vector<BATrack> &tracks,
    const BAOptions &requestedOptions,
    BAOptions *normalizedOptions)
{
    if (!normalizedOptions)
    {
        return invalid(BASolveStatus::InvalidInput,
                       "BA 输入验证失败: normalizedOptions 为空");
    }
    *normalizedOptions = requestedOptions;

    // 第一阶段只验证后端通用的数值域和数组契约。这里提前拒绝可以避免不同后端
    // 对 NaN、负迭代次数或标定分组越界产生不一致行为。
    if (requestedOptions.maxIterations <= 0 ||
        requestedOptions.maxPointIterations <= 0 ||
        requestedOptions.maxCameraIterations <= 0)
    {
        return invalid(BASolveStatus::InvalidInput,
                       "BA 输入验证失败: 迭代次数必须大于 0");
    }
    if (!std::isfinite(requestedOptions.huberDelta) ||
        !std::isfinite(requestedOptions.finiteDiffEps) ||
        !std::isfinite(requestedOptions.damping) ||
        !std::isfinite(requestedOptions.stepTolerance) ||
        requestedOptions.finiteDiffEps <= 0.0 ||
        requestedOptions.damping < 0.0 ||
        requestedOptions.stepTolerance < 0.0)
    {
        return invalid(BASolveStatus::InvalidInput,
                       "BA 输入验证失败: 鲁棒核、有限差分或收敛参数非法");
    }
    const auto parameterEnabled = [&](BAIntrinsicParameter parameter)
    {
        return sharedIntrinsicParameterEnabled(requestedOptions, parameter);
    };
    const bool sharedFocalEnabled = parameterEnabled(
        BAIntrinsicParameter::FocalLength);
    if (sharedFocalEnabled &&
        (!(requestedOptions.minSharedFocalScale > 0.0) ||
         requestedOptions.maxSharedFocalScale < requestedOptions.minSharedFocalScale))
    {
        return invalid(BASolveStatus::InvalidInput,
                       "BA 输入验证失败: 共享焦距范围非法");
    }
    const bool extendedSharedIntrinsicEnabled =
        parameterEnabled(BAIntrinsicParameter::FocalAspectRatio) ||
        parameterEnabled(BAIntrinsicParameter::PrincipalPointX) ||
        parameterEnabled(BAIntrinsicParameter::PrincipalPointY) ||
        parameterEnabled(BAIntrinsicParameter::RadialK1) ||
        parameterEnabled(BAIntrinsicParameter::RadialK2) ||
        parameterEnabled(BAIntrinsicParameter::RadialK3) ||
        parameterEnabled(BAIntrinsicParameter::TangentialP1) ||
        parameterEnabled(BAIntrinsicParameter::TangentialP2);
    if (extendedSharedIntrinsicEnabled && !sharedFocalEnabled)
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: 宽高比或主点优化以及径向畸变优化必须同时启用共享焦距优化");
    }
    if (!std::isfinite(requestedOptions.minSharedFocalAspectScale) ||
        !std::isfinite(requestedOptions.maxSharedFocalAspectScale) ||
        !(requestedOptions.minSharedFocalAspectScale > 0.0) ||
        requestedOptions.maxSharedFocalAspectScale <
            requestedOptions.minSharedFocalAspectScale)
    {
        return invalid(BASolveStatus::InvalidInput,
                       "BA 输入验证失败: 共享焦距宽高比范围非法");
    }
    if (!std::isfinite(requestedOptions.sharedFocalPriorSigma) ||
        !std::isfinite(requestedOptions.maxSharedPrincipalPointOffsetFraction) ||
        !std::isfinite(requestedOptions.sharedPrincipalPointPriorSigmaFraction) ||
        !std::isfinite(requestedOptions.sharedFocalAspectPriorSigma) ||
        requestedOptions.sharedFocalPriorSigma <= 0.0 ||
        requestedOptions.maxSharedPrincipalPointOffsetFraction <= 0.0 ||
        requestedOptions.sharedPrincipalPointPriorSigmaFraction <= 0.0 ||
        requestedOptions.sharedFocalAspectPriorSigma <= 0.0)
    {
        return invalid(BASolveStatus::InvalidInput,
                       "BA 输入验证失败: 扩展共享内参边界或先验非法");
    }
    if (!std::isfinite(requestedOptions.maxSharedRadialK1Abs) ||
        !std::isfinite(requestedOptions.maxSharedRadialK2Abs) ||
        !std::isfinite(requestedOptions.maxSharedRadialK3Abs) ||
        !std::isfinite(requestedOptions.maxSharedTangentialP1Abs) ||
        !std::isfinite(requestedOptions.maxSharedTangentialP2Abs) ||
        !std::isfinite(requestedOptions.sharedRadialK1PriorSigma) ||
        !std::isfinite(requestedOptions.sharedRadialK2PriorSigma) ||
        !std::isfinite(requestedOptions.sharedRadialK3PriorSigma) ||
        !std::isfinite(requestedOptions.sharedTangentialP1PriorSigma) ||
        !std::isfinite(requestedOptions.sharedTangentialP2PriorSigma) ||
        !std::isfinite(requestedOptions.sharedLowOrderDistortionScale) ||
        requestedOptions.maxSharedRadialK1Abs <= 0.0 ||
        requestedOptions.maxSharedRadialK2Abs <= 0.0 ||
        requestedOptions.maxSharedRadialK3Abs <= 0.0 ||
        requestedOptions.maxSharedTangentialP1Abs <= 0.0 ||
        requestedOptions.maxSharedTangentialP2Abs <= 0.0 ||
        requestedOptions.sharedRadialK1PriorSigma <= 0.0 ||
        requestedOptions.sharedRadialK2PriorSigma <= 0.0 ||
        requestedOptions.sharedRadialK3PriorSigma <= 0.0 ||
        requestedOptions.sharedTangentialP1PriorSigma <= 0.0 ||
        requestedOptions.sharedTangentialP2PriorSigma <= 0.0 ||
        requestedOptions.sharedLowOrderDistortionScale < 1.0)
    {
        return invalid(BASolveStatus::InvalidInput,
                       "BA 输入验证失败: 共享径向畸变边界或先验非法");
    }
    if (!requestedOptions.cameraCalibrationGroupIds.empty() &&
        requestedOptions.cameraCalibrationGroupIds.size() != cameras.size())
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: 相机标定分组数量必须与相机数量一致");
    }
    if (!requestedOptions.sharedIntrinsicReferenceCameras.empty() &&
        requestedOptions.sharedIntrinsicReferenceCameras.size() !=
            cameras.size())
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: 共享内参参考相机数量必须与相机数量一致");
    }
    if (std::any_of(
            requestedOptions.sharedIntrinsicReferenceCameras.begin(),
            requestedOptions.sharedIntrinsicReferenceCameras.end(),
            [](const Camera &camera)
            {
                const Camera::Intrinsics intrinsics = camera.intrinsics();
                const Camera::Distortion distortion = camera.distortion();
                return !camera.isValid() ||
                       !std::isfinite(intrinsics.focalX) ||
                       !std::isfinite(intrinsics.focalY) ||
                       intrinsics.focalX <= 0.0 || intrinsics.focalY <= 0.0 ||
                       !std::isfinite(intrinsics.principalX) ||
                       !std::isfinite(intrinsics.principalY) ||
                       !std::isfinite(distortion.radialK1) ||
                       !std::isfinite(distortion.radialK2) ||
                       !std::isfinite(distortion.radialK3) ||
                       !std::isfinite(distortion.tangentialP1) ||
                       !std::isfinite(distortion.tangentialP2);
            }))
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: 共享内参参考相机包含非法标定参数");
    }
    if (std::any_of(
            requestedOptions.cameraCalibrationGroupIds.begin(),
            requestedOptions.cameraCalibrationGroupIds.end(),
            [](const int groupId)
            {
                return groupId < 0;
            }))
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: 相机标定分组 ID 不能为负数");
    }
    if (!std::isfinite(requestedOptions.sharedFocalWarmupFraction) ||
        requestedOptions.sharedFocalWarmupFraction < 0.0 ||
        requestedOptions.sharedFocalWarmupFraction >= 1.0)
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: 自标定预热比例必须位于 [0, 1)");
    }
    if (requestedOptions.maxDenseSchurCameras <= 0 ||
        requestedOptions.maxSparseSchurCameras <
            requestedOptions.maxDenseSchurCameras)
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: Ceres Dense/Sparse Schur 相机阈值非法");
    }
    if (!std::isfinite(requestedOptions.maxCeresInitialTrackRms) ||
        requestedOptions.maxCeresInitialTrackRms < 0.0)
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: Ceres 初始 track RMS 粗差阈值必须有限且非负");
    }
    if (requestedOptions.enableLaserPlaneConstraints &&
        (!std::isfinite(requestedOptions.laserPlaneWeight) ||
         requestedOptions.laserPlaneWeight <= 0.0 ||
         !std::isfinite(requestedOptions.laserHuberDeltaMeters) ||
         requestedOptions.laserHuberDeltaMeters < 0.0))
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: LiDAR 统计权重必须为正，Huber 阈值必须有限且非负");
    }
    if (!requestedOptions.enableLaserRangeConstraints &&
        !requestedOptions.laserRangeConstraints.empty())
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: 已提供激光测距 shot，但未启用激光测距约束");
    }
    if (requestedOptions.enableLaserRangeConstraints)
    {
        if (requestedOptions.laserRangeConstraints.empty())
        {
            return invalid(
                BASolveStatus::InvalidInput,
                "BA 输入验证失败: 已启用激光测距约束但 shot 列表为空");
        }
        if (!std::isfinite(requestedOptions.laserRangeWeight) ||
            requestedOptions.laserRangeWeight <= 0.0 ||
            !std::isfinite(requestedOptions.laserRangeHuberDelta) ||
            requestedOptions.laserRangeHuberDelta < 0.0)
        {
            return invalid(
                BASolveStatus::InvalidInput,
                "BA 输入验证失败: 激光测距全局权重必须为正，Huber 阈值必须有限且非负");
        }

        for (size_t shotIndex = 0;
             shotIndex < requestedOptions.laserRangeConstraints.size();
             ++shotIndex)
        {
            const BALaserRangeConstraint &constraint =
                requestedOptions.laserRangeConstraints[shotIndex];
            const std::string prefix =
                "BA 输入验证失败: 激光测距 shot[" +
                std::to_string(shotIndex) + "] ";
            if (constraint.cameraIndex < 0 ||
                constraint.cameraIndex >= static_cast<int>(cameras.size()))
            {
                return invalid(BASolveStatus::InvalidInput,
                               prefix + "cameraIndex 越界");
            }
            if (!isFinitePoint(constraint.initialPoint) ||
                !isFinitePoint(constraint.leverArmCameraMeters) ||
                !std::isfinite(constraint.observedRangeMeters) ||
                constraint.observedRangeMeters <= 0.0 ||
                !std::isfinite(constraint.sigmaRangeMeters) ||
                constraint.sigmaRangeMeters <= 0.0 ||
                !std::isfinite(constraint.weight) ||
                constraint.weight <= 0.0 ||
                !std::isfinite(constraint.ephemerisTimeSeconds))
            {
                return invalid(
                    BASolveStatus::InvalidInput,
                    prefix + "必须包含有限落点/杆臂/时间以及正的 range、sigma 和 weight");
            }
            const Camera &camera = cameras[static_cast<size_t>(constraint.cameraIndex)];
            const std::array<double, 3> cameraCenter = camera.cameraCenter();
            const std::array<double, 9> cameraToWorld =
                camera.cameraToWorldRotation();
            std::array<double, 3> emitter = cameraCenter;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    emitter[static_cast<size_t>(row)] +=
                        cameraToWorld[static_cast<size_t>(row * 3 + column)] *
                        constraint.leverArmCameraMeters[static_cast<size_t>(column)];
                }
            }
            const double dx = constraint.initialPoint[0] - emitter[0];
            const double dy = constraint.initialPoint[1] - emitter[1];
            const double dz = constraint.initialPoint[2] - emitter[2];
            const double initialRangeSquared = dx * dx + dy * dy + dz * dz;
            if (!std::isfinite(initialRangeSquared) || initialRangeSquared <= 1.0e-18)
            {
                return invalid(
                    BASolveStatus::InvalidInput,
                    prefix + "初始落点不能与按杆臂换算后的激光发射位置重合");
            }
            const double effectiveWeight =
                requestedOptions.laserRangeWeight * constraint.weight;
            const double normalizedScale =
                std::sqrt(effectiveWeight) / constraint.sigmaRangeMeters;
            const double scaledHuberDelta =
                requestedOptions.laserRangeHuberDelta *
                std::sqrt(effectiveWeight);
            if (!std::isfinite(effectiveWeight) ||
                !std::isfinite(normalizedScale) ||
                !std::isfinite(scaledHuberDelta))
            {
                return invalid(
                    BASolveStatus::InvalidInput,
                    prefix + "全局/shot 权重与 sigma 组合导致非有限残差尺度");
            }

            std::set<int> measuredCameras;
            bool measuredInitialPointProjects = true;
            for (const BAObservation &observation :
                 constraint.measuredImageObservations)
            {
                if (!observationIsUsable(observation, cameras.size()))
                {
                    return invalid(
                        BASolveStatus::InvalidInput,
                        prefix + "包含非法真实 measured 像点");
                }
                measuredCameras.insert(observation.cameraIndex);
                const double world[3] = {
                    constraint.initialPoint[0],
                    constraint.initialPoint[1],
                    constraint.initialPoint[2],
                };
                double pixel[2] = {0.0, 0.0};
                measuredInitialPointProjects =
                    measuredInitialPointProjects &&
                    cameras[static_cast<size_t>(observation.cameraIndex)]
                        .projectWorldPoint(world, pixel);
            }
            if (!measuredInitialPointProjects)
            {
                return invalid(
                    BASolveStatus::InvalidInput,
                    prefix + "真实 measured 像点的辅助点初值必须在对应帧相机前方");
            }

            switch (constraint.pointMode)
            {
            case BALaserPointMode::Fixed:
                break;
            case BALaserPointMode::Constrained:
                if (!isFinitePoint(constraint.pointPrior) ||
                    !isUsableLaserPointSqrtInformation(
                        constraint.pointPriorSqrtInformation))
                {
                    return invalid(
                        BASolveStatus::InvalidInput,
                        prefix + "Constrained 落点必须包含有限先验和满秩 3x3 平方根信息矩阵");
                }
                break;
            case BALaserPointMode::Free:
            {
                bool hasNondegenerateBaseline = false;
                for (auto left = measuredCameras.begin();
                     left != measuredCameras.end() && !hasNondegenerateBaseline;
                     ++left)
                {
                    auto right = left;
                    ++right;
                    for (; right != measuredCameras.end(); ++right)
                    {
                        const std::array<double, 3> leftCenter =
                            cameras[static_cast<size_t>(*left)].cameraCenter();
                        const std::array<double, 3> rightCenter =
                            cameras[static_cast<size_t>(*right)].cameraCenter();
                        if (isFinitePoint(leftCenter) &&
                            isFinitePoint(rightCenter) &&
                            squaredDistance(leftCenter, rightCenter) > 1.0e-16)
                        {
                            hasNondegenerateBaseline = true;
                            break;
                        }
                    }
                }
                if (constraint.measuredImageObservations.size() < 2 ||
                    measuredCameras.size() < 2 ||
                    !hasNondegenerateBaseline)
                {
                    return invalid(
                        BASolveStatus::UnsupportedConfiguration,
                        prefix + "Free 落点至少需要两台具有非零基线相机的真实 measured 像点");
                }
                break;
            }
            default:
                return invalid(BASolveStatus::InvalidInput,
                               prefix + "pointMode 非法");
            }
        }
    }
    if (!std::isfinite(requestedOptions.maxCeresCudaMemoryFraction) ||
        requestedOptions.maxCeresCudaMemoryFraction <= 0.0 ||
        requestedOptions.maxCeresCudaMemoryFraction > 1.0)
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: Ceres CUDA 显存预算比例必须位于 (0, 1]");
    }
    if (!std::isfinite(requestedOptions.maxAcceptedConstraintRmsGrowth) ||
        requestedOptions.maxAcceptedConstraintRmsGrowth < 1.0)
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: 物方约束 RMS 增长倍率不能小于 1");
    }
    if (!std::isfinite(requestedOptions.nativeCudaMaxPointStepNorm) ||
        requestedOptions.nativeCudaMaxPointStepNorm <= 0.0)
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: native CUDA 点步长上限必须大于 0");
    }
    if (requestedOptions.cameraPlaneConstraint.enabled)
    {
        const auto &constraint = requestedOptions.cameraPlaneConstraint;
        const double normalNormSquared =
            constraint.normal[0] * constraint.normal[0] +
            constraint.normal[1] * constraint.normal[1] +
            constraint.normal[2] * constraint.normal[2];
        const bool finitePoint = std::all_of(
            constraint.point.begin(), constraint.point.end(),
            [](double value) { return std::isfinite(value); });
        const bool finiteNormal = std::all_of(
            constraint.normal.begin(), constraint.normal.end(),
            [](double value) { return std::isfinite(value); });
        const bool validReferenceDistances =
            constraint.referenceSignedDistances.empty() ||
            (constraint.referenceSignedDistances.size() == cameras.size() &&
             std::all_of(
                 constraint.referenceSignedDistances.begin(),
                 constraint.referenceSignedDistances.end(),
                 [](double value) { return std::isfinite(value); }));
        if (!finitePoint || !finiteNormal || !std::isfinite(normalNormSquared) ||
            std::abs(normalNormSquared - 1.0) > 1.0e-6 ||
            !validReferenceDistances ||
            !std::isfinite(constraint.sigmaMeters) || constraint.sigmaMeters <= 0.0 ||
            !std::isfinite(constraint.weight) || constraint.weight <= 0.0 ||
            !std::isfinite(requestedOptions.cameraPlaneHuberDelta) ||
            requestedOptions.cameraPlaneHuberDelta < 0.0)
        {
            return invalid(
                BASolveStatus::InvalidInput,
                "BA 输入验证失败: 相机参考层约束必须包含单位法向、"
                "与相机等长的有限参考距离和正的 sigma/weight");
        }
    }

    std::set<int> fixedCameras;
    for (const int index : requestedOptions.fixedCameraIndices)
    {
        if (index < 0 || index >= static_cast<int>(cameras.size()))
        {
            return invalid(BASolveStatus::InvalidInput,
                           "BA 输入验证失败: 固定相机索引越界");
        }
        if (!fixedCameras.insert(index).second)
        {
            return invalid(BASolveStatus::InvalidInput,
                           "BA 输入验证失败: 固定相机索引重复");
        }
    }

    std::set<int> fixedTracks;
    for (const int index : requestedOptions.fixedTrackIndices)
    {
        if (index < 0 || index >= static_cast<int>(tracks.size()))
        {
            return invalid(BASolveStatus::InvalidInput,
                           "BA 输入验证失败: 固定轨迹索引越界");
        }
        if (!fixedTracks.insert(index).second)
        {
            return invalid(BASolveStatus::InvalidInput,
                           "BA 输入验证失败: 固定轨迹索引重复");
        }
    }
    std::sort(normalizedOptions->fixedTrackIndices.begin(),
              normalizedOptions->fixedTrackIndices.end());

    if (!requestedOptions.refineCameraPose ||
        requestedOptions.gaugePolicy == BAGaugePolicy::CallerManaged ||
        cameras.empty())
    {
        return {};
    }

    // 第二阶段判断调用方已经提供了哪些 gauge 信息。一个位姿先验只消除刚体
    // 自由度，至少两处非重合位姿参考才包含尺度；控制点需至少三个不共线物方点
    // 才能完整约束 Sim(3)。比例尺只消除尺度自由度。LiDAR 点到面或激光测距
    // 约束不能仅凭“存在”就视为完整 7-DOF gauge：平行/近共面法向可能缺少
    // 平面内自由度，零杆臂测距也不能直接约束相机姿态。
    // 因此 AutoAnchor 对 LiDAR BA 保持与纯影像 BA 相同的两相机锚定。
    const bool completeControlPointGauge =
        hasCompleteControlPointGauge(tracks, requestedOptions);
    const bool absolutePose =
        hasEnabledPosePrior(requestedOptions) || completeControlPointGauge;
    bool absoluteScale =
        hasMetricPosePriorBaseline(requestedOptions) ||
        completeControlPointGauge ||
        hasUsableScaleBar(tracks, requestedOptions);
    auto refreshScaleFromFixedCameras = [&]()
    {
        absoluteScale =
            absoluteScale ||
            hasPosePriorBaselineToFixedCamera(cameras, requestedOptions, fixedCameras) ||
            hasControlPointBaselineToFixedCamera(
                cameras, tracks, requestedOptions, fixedCameras);
    };
    refreshScaleFromFixedCameras();

    auto hasCompleteGauge = [&]()
    {
        if (absolutePose && absoluteScale)
        {
            return true;
        }
        if (fixedCameras.size() >= 2)
        {
            return true;
        }
        return fixedCameras.size() == 1 && absoluteScale;
    };

    if (requestedOptions.gaugePolicy == BAGaugePolicy::RequireExplicitGauge)
    {
        if (!hasCompleteGauge())
        {
            return invalid(
                BASolveStatus::UnsupportedConfiguration,
                "BA gauge 约束不完整: 联合 BA 需固定两台非退化相机，"
                "或提供固定相机与绝对尺度/位姿约束");
        }
        return {};
    }

    if (absolutePose && absoluteScale)
    {
        return {};
    }

    // AutoAnchor 只补最少约束：一台相机固定全局刚体自由度；若没有绝对尺度，
    // 再固定一条非退化基线。补入的是 normalizedOptions，不污染 GUI 原始参数。
    if (fixedCameras.empty())
    {
        normalizedOptions->fixedCameraIndices.push_back(0);
        fixedCameras.insert(0);
        refreshScaleFromFixedCameras();
    }

    if (!absoluteScale && fixedCameras.size() < 2)
    {
        const int anchorIndex = *fixedCameras.begin();
        const int secondIndex = farthestCameraFrom(cameras, anchorIndex, fixedCameras);
        if (secondIndex < 0)
        {
            return invalid(
                BASolveStatus::UnsupportedConfiguration,
                "BA gauge 自动锚定失败: 未找到具有非零基线的第二台相机");
        }
        normalizedOptions->fixedCameraIndices.push_back(secondIndex);
    }

    return {};
}

} // namespace xjw::detail
