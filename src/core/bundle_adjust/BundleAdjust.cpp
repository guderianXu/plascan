// ============================================================
// 文件：BundleAdjust.cpp
// 功能：统一束平差后端选择、质量门控与参考 CPU 回退。
// CPU 求解统一委托给 PlaMatrix 联合 Schur BA；LegacyCpu 仅保留为
// 工程文件和 CLI 的兼容枚举，不再包含独立的旧交替优化实现。
// ============================================================

#include "BundleAdjustSolver.h"

#include "BundleAdjustPlaMatrix.h"
#include "BundleAdjustQuality.h"
#include "BundleAdjustValidation.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace xjw
{
    namespace
    {
        void updateDerivedResultStats(BAResult& result)
        {
            result.validTrackRatio = result.totalTracks > 0 ? static_cast<double>(result.optimizedTracks) /
                                                                  static_cast<double>(result.totalTracks)
                                                            : 0.0;
        }

        bool resultFailsQualityGate(const BAResult& result, const BAOptions& options, std::string* message)
        {
            if (!options.enableBackendQualityGate)
            {
                return false;
            }

            if (!result.solutionUsable)
            {
                if (message)
                {
                    *message = "质量门控拒绝: BA 求解状态不可写回";
                }
                return true;
            }

            if (result.totalTracks > 0 && !std::isfinite(result.meanRmsAfter))
            {
                if (message)
                {
                    *message = "质量门控拒绝: BA 后 RMS 非有限";
                }
                return true;
            }

            if (result.totalTracks > 0 && result.validTrackRatio < std::max(0.0, options.minAcceptedValidTrackRatio))
            {
                if (message)
                {
                    *message = "质量门控拒绝: 有效 track 比例过低";
                }
                return true;
            }

            const double maxGrowth = std::max(0.0, options.maxAcceptedRmsGrowth);
            if (result.totalTracks > 0 && maxGrowth > 0.0 && std::isfinite(result.meanRmsBefore))
            {
                if (result.meanRmsBefore > 1e-12)
                {
                    const double maxAccepted = result.meanRmsBefore * maxGrowth;
                    if (result.meanRmsAfter > maxAccepted)
                    {
                        if (message)
                        {
                            *message = "质量门控拒绝: BA 后 RMS 增长超过阈值";
                        }
                        return true;
                    }
                }
                else if (result.meanRmsAfter > 1e-9)
                {
                    if (message)
                    {
                        *message = "质量门控拒绝: 零残差输入被优化为非零残差";
                    }
                    return true;
                }
            }

            const double maxConstraintGrowth = std::max(1.0, options.maxAcceptedConstraintRmsGrowth);
            if (!detail::constraintRmsPassesQualityGate(result.laserConstraintCount,
                                                        result.laserRmsBeforeMeters,
                                                        result.laserRmsAfterMeters,
                                                        maxConstraintGrowth,
                                                        "LiDAR 点到面约束",
                                                        message) ||
                !detail::constraintRmsPassesQualityGate(result.laserRangeConstraintCount,
                                                        result.laserRangeRmsBeforeMeters,
                                                        result.laserRangeRmsAfterMeters,
                                                        maxConstraintGrowth,
                                                        "行星激光测距约束",
                                                        message) ||
                !detail::constraintRmsPassesQualityGate(result.controlPointConstraintCount,
                                                        result.controlPointRmsBeforeMeters,
                                                        result.controlPointRmsAfterMeters,
                                                        maxConstraintGrowth,
                                                        "控制点约束",
                                                        message) ||
                !detail::constraintRmsPassesQualityGate(result.scaleBarConstraintCount,
                                                        result.scaleBarRmsBeforeMeters,
                                                        result.scaleBarRmsAfterMeters,
                                                        maxConstraintGrowth,
                                                        "比例尺约束",
                                                        message))
            {
                return true;
            }

            return false;
        }

        bool legacyIsBetterThanCandidate(const BAResult& candidate,
                                         const BAResult& legacy,
                                         const BAOptions& options,
                                         std::string* message)
        {
            if (!options.enableBackendQualityGate || !options.compareAutoBackendWithLegacy || legacy.totalTracks <= 0 ||
                legacy.optimizedTracks <= 0 || !std::isfinite(legacy.meanRmsAfter))
            {
                return false;
            }

            const double maxGrowth = std::max(1.0, options.maxAcceptedRmsGrowth);
            if (std::isfinite(candidate.meanRmsAfter) && candidate.meanRmsAfter > legacy.meanRmsAfter * maxGrowth)
            {
                if (message)
                {
                    *message = "质量门控拒绝: 候选后端 RMS 明显差于参考 CPU BA";
                }
                return true;
            }

            if (candidate.validTrackRatio + 1e-12 < legacy.validTrackRatio &&
                candidate.validTrackRatio < std::max(0.0, options.minAcceptedValidTrackRatio))
            {
                if (message)
                {
                    *message = "质量门控拒绝: 候选后端有效 track 比例低于参考 CPU BA";
                }
                return true;
            }

            return false;
        }

    } // namespace

    const char* BundleAdjust::backendName(BABackend backend)
    {
        switch (backend)
        {
        case BABackend::Auto:
            return "auto";
        case BABackend::LegacyCpu:
            return "legacy_cpu";
        case BABackend::PlaMatrixCpu:
            return "plamatrix_cpu";
        case BABackend::PlaMatrixCuda:
            return "plamatrix_cuda";
        case BABackend::PlaMatrixOpenCl:
            return "plamatrix_opencl";
        }
        return "unknown";
    }

    const char* BundleAdjust::solveStatusName(BASolveStatus status)
    {
        switch (status)
        {
        case BASolveStatus::NotRun:
            return "not_run";
        case BASolveStatus::Success:
            return "success";
        case BASolveStatus::NoConvergence:
            return "no_convergence";
        case BASolveStatus::Cancelled:
            return "cancelled";
        case BASolveStatus::InvalidInput:
            return "invalid_input";
        case BASolveStatus::UnsupportedConfiguration:
            return "unsupported_configuration";
        case BASolveStatus::BackendUnavailable:
            return "backend_unavailable";
        case BASolveStatus::NumericalFailure:
            return "numerical_failure";
        }
        return "unknown";
    }

    BABackendCapabilities BundleAdjust::backendCapabilities(BABackend backend)
    {
        switch (backend)
        {
        case BABackend::Auto:
            return {true, true, true, true, true, true, true, true};
        case BABackend::LegacyCpu:
            return {true, true, true, true, true, true, true, true};
        case BABackend::PlaMatrixCpu:
        case BABackend::PlaMatrixCuda:
        case BABackend::PlaMatrixOpenCl:
            return {true, true, true, true, true, true, true, true};
        }
        return {};
    }

    bool BundleAdjust::isBackendAvailable(BABackend backend)
    {
        switch (backend)
        {
        case BABackend::Auto:
            return true;
        case BABackend::LegacyCpu:
            return true;
        case BABackend::PlaMatrixCpu:
            return true;
        case BABackend::PlaMatrixCuda:
        case BABackend::PlaMatrixOpenCl:
        {
            std::string message;
            return detail::isPlaMatrixBackendAvailable(backend, 0, &message);
        }
        }
        return false;
    }

    BAProblemStats BundleAdjust::summarizeProblem(const std::vector<FramePinholeCamera>& cameras,
                                                  const std::vector<BATrack>& tracks)
    {
        return detail::summarizeUsableProblem(cameras, tracks);
    }

    bool BundleAdjust::autoBackendMeetsScaleThreshold(BABackend backend,
                                                      const BAProblemStats& stats,
                                                      const BAOptions& options)
    {
        const int camera_count = std::max(0, stats.cameraCount);
        const int observation_count = std::max(0, stats.observationCount);
        const int dense_camera_threshold = std::max(1, options.minPlaMatrixDenseCameras);

        if (backend == BABackend::PlaMatrixCuda)
        {
            const bool regular_scale = camera_count >= std::max(1, options.minPlaMatrixCudaCameras) &&
                                       observation_count >= std::max(1, options.minPlaMatrixCudaObservations);
            const bool dense_scale = camera_count >= dense_camera_threshold &&
                                     observation_count >= std::max(1, options.minPlaMatrixCudaDenseObservations);
            return regular_scale || dense_scale;
        }
        if (backend == BABackend::PlaMatrixOpenCl)
        {
            const bool regular_scale = camera_count >= std::max(1, options.minPlaMatrixOpenClCameras) &&
                                       observation_count >= std::max(1, options.minPlaMatrixOpenClObservations);
            const bool dense_scale = camera_count >= dense_camera_threshold &&
                                     observation_count >= std::max(1, options.minPlaMatrixOpenClDenseObservations);
            return regular_scale || dense_scale;
        }
        return false;
    }

    BABackendDecision BundleAdjust::decideBackendForProblem(const BAProblemStats& stats, const BAOptions& options)
    {
        if (options.backend != BABackend::Auto)
        {
            return {options.backend, "explicit_backend"};
        }
        const bool refineSharedFocal = sharedIntrinsicParameterEnabled(options, BAIntrinsicParameter::FocalLength);
        const bool refineExtendedIntrinsics =
            sharedIntrinsicParameterEnabled(options, BAIntrinsicParameter::FocalAspectRatio) ||
            sharedIntrinsicParameterEnabled(options, BAIntrinsicParameter::PrincipalPointX) ||
            sharedIntrinsicParameterEnabled(options, BAIntrinsicParameter::PrincipalPointY) ||
            sharedIntrinsicParameterEnabled(options, BAIntrinsicParameter::RadialK1) ||
            sharedIntrinsicParameterEnabled(options, BAIntrinsicParameter::RadialK2) ||
            sharedIntrinsicParameterEnabled(options, BAIntrinsicParameter::RadialK3) ||
            sharedIntrinsicParameterEnabled(options, BAIntrinsicParameter::TangentialP1) ||
            sharedIntrinsicParameterEnabled(options, BAIntrinsicParameter::TangentialP2);
        const bool hasPosePrior = std::any_of(options.cameraPosePriors.begin(),
                                              options.cameraPosePriors.end(),
                                              [](const BACameraPosePrior& prior) { return prior.enabled; });
        const bool hasSoftConstraints = options.enableLaserPlaneConstraints || options.enableLaserRangeConstraints ||
                                        options.enableControlPointConstraints || options.enableScaleBarConstraints ||
                                        options.cameraPlaneConstraint.enabled || hasPosePrior;
        const bool needsJointSolver =
            options.refineCameraPose || refineSharedFocal || refineExtendedIntrinsics || hasSoftConstraints;
        if (!needsJointSolver)
        {
            return {BABackend::PlaMatrixCpu, "point_only_uses_reference_ba"};
        }

        if (autoBackendMeetsScaleThreshold(BABackend::PlaMatrixCuda, stats, options) &&
            isBackendAvailable(BABackend::PlaMatrixCuda))
        {
            const bool regular_scale = stats.cameraCount >= std::max(1, options.minPlaMatrixCudaCameras) &&
                                       stats.observationCount >= std::max(1, options.minPlaMatrixCudaObservations);
            return {BABackend::PlaMatrixCuda,
                    regular_scale ? "large_joint_problem_uses_plamatrix_cuda"
                                  : "dense_joint_problem_uses_plamatrix_cuda"};
        }
        if (autoBackendMeetsScaleThreshold(BABackend::PlaMatrixOpenCl, stats, options) &&
            isBackendAvailable(BABackend::PlaMatrixOpenCl))
        {
            const bool regular_scale = stats.cameraCount >= std::max(1, options.minPlaMatrixOpenClCameras) &&
                                       stats.observationCount >= std::max(1, options.minPlaMatrixOpenClObservations);
            return {BABackend::PlaMatrixOpenCl,
                    regular_scale ? "large_joint_problem_uses_plamatrix_opencl"
                                  : "dense_joint_problem_uses_plamatrix_opencl"};
        }
        return {BABackend::PlaMatrixCpu,
                hasSoftConstraints ? "constraint_problem_uses_plamatrix_cpu" : "joint_problem_uses_plamatrix_cpu"};
    }

    BABackend BundleAdjust::selectBackendForProblem(const BAProblemStats& stats, const BAOptions& options)
    {
        return decideBackendForProblem(stats, options).backend;
    }

    BAResult BundleAdjust::optimizePoints(const std::vector<FramePinholeCamera>& cameras,
                                          const std::vector<BATrack>& tracks,
                                          const BAOptions& requestedOptions)
    {
        BAOptions normalizedOptions;
        const detail::BundleAdjustValidationResult validation =
            detail::validateAndNormalizeBundleAdjustOptions(cameras, tracks, requestedOptions, &normalizedOptions);
        if (!validation.ok)
        {
            BAResult result;
            result.requestedBackend = requestedOptions.backend;
            result.usedBackend = requestedOptions.backend;
            result.solveStatus = validation.status;
            result.solutionUsable = false;
            result.backendMessage = validation.message;
            result.totalTracks = static_cast<int>(tracks.size());
            result.observationCount = summarizeProblem(cameras, tracks).observationCount;
            result.refinedCameras = cameras;
            result.points.resize(tracks.size());
            updateDerivedResultStats(result);
            return result;
        }
        const BAOptions& options = normalizedOptions;
        const std::string cpuFallbackUnsupportedReason;

        auto runLegacy = [&](const std::string& fallbackMessage)
        {
            BAOptions cpuOptions = options;
            cpuOptions.backend = BABackend::PlaMatrixCpu;
            cpuOptions.allowBackendFallback = false;
            BAResult result = detail::optimizePointsWithPlaMatrix(cameras, tracks, cpuOptions);
            result.requestedBackend = options.backend;
            result.usedBackend = BABackend::PlaMatrixCpu;
            result.usedGpu = false;
            result.backendFallback = true;
            result.backendMessage = fallbackMessage + "；旧 CPU 名称已映射到参考 CPU 联合 BA；" + result.backendMessage;
            result.backendSelectionReason = fallbackMessage;
            updateDerivedResultStats(result);
            return result;
        };

        auto runPlaMatrixCpu = [&](const std::string& fallbackMessage)
        {
            BAOptions cpuOptions = options;
            cpuOptions.backend = BABackend::PlaMatrixCpu;
            cpuOptions.allowBackendFallback = false;
            BAResult result = detail::optimizePointsWithPlaMatrix(cameras, tracks, cpuOptions);
            result.requestedBackend = options.backend;
            result.backendFallback = options.backend != BABackend::PlaMatrixCpu;
            result.backendMessage = fallbackMessage + "；" + result.backendMessage;
            result.backendSelectionReason = fallbackMessage;
            updateDerivedResultStats(result);
            return result;
        };

        if (options.backend == BABackend::Auto)
        {
            const BAProblemStats stats = summarizeProblem(cameras, tracks);
            BAOptions selectedOptions = options;
            const BABackendDecision decision = decideBackendForProblem(stats, options);
            selectedOptions.backend = decision.backend;
            if (options.enableLaserRangeConstraints)
            {
                selectedOptions.allowBackendFallback = false;
            }
            const std::string selectedName = backendName(selectedOptions.backend);

            if (selectedOptions.backend == BABackend::LegacyCpu)
            {
                BAResult result = runPlaMatrixCpu("自动 CPU 已统一为参考联合 BA: " + decision.reason);
                result.requestedBackend = BABackend::Auto;
                return result;
            }

            BAResult candidate = optimizePoints(cameras, tracks, selectedOptions);
            candidate.requestedBackend = BABackend::Auto;
            updateDerivedResultStats(candidate);

            std::string qualityMessage;
            bool rejectCandidate = resultFailsQualityGate(candidate, options, &qualityMessage);
            BAResult legacy;
            bool comparedWithLegacy = false;
            if (!rejectCandidate && options.enableBackendQualityGate && options.compareAutoBackendWithLegacy &&
                cpuFallbackUnsupportedReason.empty())
            {
                BAOptions legacyOptions = options;
                legacyOptions.backend = BABackend::PlaMatrixCpu;
                legacy = detail::optimizePointsWithPlaMatrix(cameras, tracks, legacyOptions);
                legacy.requestedBackend = BABackend::Auto;
                legacy.usedBackend = BABackend::PlaMatrixCpu;
                legacy.usedGpu = false;
                updateDerivedResultStats(legacy);
                comparedWithLegacy = true;
                rejectCandidate = legacyIsBetterThanCandidate(candidate, legacy, options, &qualityMessage);
            }

            if (rejectCandidate)
            {
                if (selectedOptions.backend == BABackend::PlaMatrixCpu)
                {
                    candidate.qualityGateRejected = true;
                    candidate.qualityGateMessage = qualityMessage;
                    candidate.solutionUsable = false;
                    if (candidate.solveStatus == BASolveStatus::Success ||
                        candidate.solveStatus == BASolveStatus::NoConvergence)
                    {
                        candidate.solveStatus = BASolveStatus::NumericalFailure;
                    }
                    candidate.backendFallback = false;
                    candidate.backendSelectionReason = "自动选择的参考 CPU BA 被质量门控拒绝，当前没有其它 CPU 回退";
                    candidate.backendMessage = candidate.backendSelectionReason + "；" + qualityMessage;
                    return candidate;
                }
                if (!cpuFallbackUnsupportedReason.empty())
                {
                    candidate.qualityGateRejected = true;
                    candidate.qualityGateMessage = qualityMessage;
                    candidate.solutionUsable = false;
                    if (candidate.solveStatus == BASolveStatus::Success ||
                        candidate.solveStatus == BASolveStatus::NoConvergence)
                    {
                        candidate.solveStatus = BASolveStatus::NumericalFailure;
                    }
                    candidate.backendFallback = false;
                    candidate.backendSelectionReason = "自动候选 " + selectedName + " 被质量门控拒绝；" +
                                                       cpuFallbackUnsupportedReason + "，没有兼容的 CPU 回退";
                    candidate.backendMessage = candidate.backendSelectionReason + "；" + qualityMessage;
                    return candidate;
                }
                if (legacy.totalTracks <= 0)
                {
                    BAOptions legacyOptions = options;
                    legacyOptions.backend = BABackend::PlaMatrixCpu;
                    legacy = detail::optimizePointsWithPlaMatrix(cameras, tracks, legacyOptions);
                    legacy.requestedBackend = BABackend::Auto;
                    legacy.usedBackend = BABackend::PlaMatrixCpu;
                    legacy.usedGpu = false;
                    updateDerivedResultStats(legacy);
                }
                legacy.setupSeconds += candidate.setupSeconds;
                legacy.solveSeconds += candidate.solveSeconds;
                legacy.postprocessSeconds += candidate.postprocessSeconds;
                legacy.totalSeconds += candidate.totalSeconds;
                legacy.backendFallback = true;
                legacy.qualityGateRejected = true;
                legacy.qualityGateMessage = qualityMessage;
                legacy.backendSelectionReason = "自动候选 " + selectedName + " 被质量门控拒绝，回退参考 CPU BA";
                legacy.backendMessage = legacy.backendSelectionReason + "；" + qualityMessage;
                return legacy;
            }

            candidate.backendSelectionReason = "自动选择 " + selectedName + ": 通过 BA 质量门控";
            if (comparedWithLegacy)
            {
                candidate.setupSeconds += legacy.setupSeconds;
                candidate.solveSeconds += legacy.solveSeconds;
                candidate.postprocessSeconds += legacy.postprocessSeconds;
                candidate.totalSeconds += legacy.totalSeconds;
            }
            if (!candidate.backendMessage.empty())
            {
                candidate.backendMessage = candidate.backendSelectionReason + "；" + candidate.backendMessage;
            }
            else
            {
                candidate.backendMessage = candidate.backendSelectionReason;
            }
            return candidate;
        }

        if (options.backend == BABackend::LegacyCpu)
        {
            return runLegacy("legacy_cpu 兼容请求");
        }

        const bool isPlaMatrixBackend = options.backend == BABackend::PlaMatrixCpu ||
                                        options.backend == BABackend::PlaMatrixCuda ||
                                        options.backend == BABackend::PlaMatrixOpenCl;
        if (isPlaMatrixBackend)
        {
            const BABackend requestedBackend = options.backend;
            std::string unavailableMessage;
            if (!detail::isPlaMatrixBackendAvailable(requestedBackend, options.plaMatrixDevice, &unavailableMessage))
            {
                if (options.allowBackendFallback && requestedBackend != BABackend::PlaMatrixCpu)
                {
                    return runPlaMatrixCpu(unavailableMessage + "，已回退到 plamatrix_cpu");
                }

                BAResult result;
                result.requestedBackend = requestedBackend;
                result.usedBackend = requestedBackend;
                result.solveStatus = BASolveStatus::BackendUnavailable;
                result.solutionUsable = false;
                result.backendMessage = unavailableMessage;
                result.totalTracks = static_cast<int>(tracks.size());
                result.observationCount = summarizeProblem(cameras, tracks).observationCount;
                result.refinedCameras = cameras;
                result.points.resize(tracks.size());
                updateDerivedResultStats(result);
                return result;
            }

            BAResult result = detail::optimizePointsWithPlaMatrix(cameras, tracks, options);
            result.requestedBackend = requestedBackend;
            updateDerivedResultStats(result);
            if (!result.solutionUsable && result.solveStatus != BASolveStatus::Cancelled &&
                options.allowBackendFallback && requestedBackend != BABackend::PlaMatrixCpu)
            {
                BAResult fallback = runPlaMatrixCpu(result.backendMessage + "，已回退到 plamatrix_cpu");
                fallback.setupSeconds += result.setupSeconds;
                fallback.solveSeconds += result.solveSeconds;
                fallback.postprocessSeconds += result.postprocessSeconds;
                fallback.totalSeconds += result.totalSeconds;
                return fallback;
            }
            return result;
        }

        return runPlaMatrixCpu("未知 BA 后端，已回退到参考 CPU 联合 BA");
    }

} // namespace xjw
