#include "BundleAdjustValidation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace xjw::detail
{
namespace
{

// 绝对位姿约束同时消除全局平移、旋转和尺度自由度。这里只判断约束是否存在，
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

bool hasControlPointConstraint(const std::vector<BATrack> &tracks,
                               const BAOptions &options)
{
    if (!options.enableControlPointConstraints)
    {
        return false;
    }
    return std::any_of(
        tracks.begin(),
        tracks.end(),
        [](const BATrack &track)
        {
            return !track.controlPointConstraints.empty();
        });
}

bool hasLaserConstraint(const std::vector<BATrack> &tracks,
                        const BAOptions &options)
{
    if (!options.enableLaserPlaneConstraints)
    {
        return false;
    }
    return std::any_of(
        tracks.begin(),
        tracks.end(),
        [](const BATrack &track)
        {
            return !track.laserPlaneConstraints.empty();
        });
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
    if (requestedOptions.refineSharedFocalLength &&
        (!(requestedOptions.minSharedFocalScale > 0.0) ||
         requestedOptions.maxSharedFocalScale < requestedOptions.minSharedFocalScale))
    {
        return invalid(BASolveStatus::InvalidInput,
                       "BA 输入验证失败: 共享焦距范围非法");
    }
    if ((requestedOptions.refineSharedFocalAspectRatio ||
         requestedOptions.refineSharedPrincipalPoint) &&
        !requestedOptions.refineSharedFocalLength)
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: 宽高比或主点优化必须同时启用共享焦距优化");
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
    if (!std::isfinite(requestedOptions.maxSharedPrincipalPointOffsetFraction) ||
        !std::isfinite(requestedOptions.sharedPrincipalPointPriorSigmaFraction) ||
        !std::isfinite(requestedOptions.sharedFocalAspectPriorSigma) ||
        requestedOptions.maxSharedPrincipalPointOffsetFraction <= 0.0 ||
        requestedOptions.sharedPrincipalPointPriorSigmaFraction <= 0.0 ||
        requestedOptions.sharedFocalAspectPriorSigma <= 0.0)
    {
        return invalid(BASolveStatus::InvalidInput,
                       "BA 输入验证失败: 扩展共享内参边界或先验非法");
    }
    if (!requestedOptions.cameraCalibrationGroupIds.empty() &&
        requestedOptions.cameraCalibrationGroupIds.size() != cameras.size())
    {
        return invalid(
            BASolveStatus::InvalidInput,
            "BA 输入验证失败: 相机标定分组数量必须与相机数量一致");
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

    if (!requestedOptions.refineCameraPose ||
        requestedOptions.gaugePolicy == BAGaugePolicy::CallerManaged ||
        cameras.empty())
    {
        return {};
    }

    // 第二阶段判断调用方已经提供了哪些 gauge 信息。控制点/激光平面/位姿先验
    // 提供绝对参考；比例尺仅消除尺度自由度，不能单独固定世界坐标原点和朝向。
    const bool absolutePose =
        hasEnabledPosePrior(requestedOptions) ||
        hasControlPointConstraint(tracks, requestedOptions) ||
        hasLaserConstraint(tracks, requestedOptions);
    const bool absoluteScale =
        absolutePose || hasUsableScaleBar(tracks, requestedOptions);

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
