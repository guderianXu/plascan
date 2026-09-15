#include "tsdf/DepthTsdfStages.h"
namespace xjw::mesh
{
    using namespace tsdf_detail;

    bool DepthTsdfSurfaceBuilder::shouldReleaseVisibilityConstrainedHole(int loopVertexCount,
                                                                         int silhouetteProtectedVertexCount,
                                                                         int supportingViewCount,
                                                                         int conflictViewCount,
                                                                         int minimumSupportingViews,
                                                                         int maximumConflictViews,
                                                                         float strongSilhouetteRatio)
    {
        if (loopVertexCount < 3)
        {
            return false;
        }

        const int silhouette_vertex_count = std::clamp(silhouetteProtectedVertexCount, 0, loopVertexCount);
        const float silhouette_ratio =
            static_cast<float>(silhouette_vertex_count) / static_cast<float>(loopVertexCount);
        const int extra_strong_support =
            silhouette_vertex_count > 0 && silhouette_ratio >= std::clamp(strongSilhouetteRatio, 0.0f, 1.0f) ? 1 : 0;
        const int required_supporting_views = std::max(1, minimumSupportingViews) + extra_strong_support;
        return supportingViewCount >= required_supporting_views &&
               conflictViewCount <= std::max(0, maximumConflictViews);
    }

    bool DepthTsdfSurfaceBuilder::shouldAcceptFinalHoleFillPatchSample(int supportingViewCount,
                                                                       int conflictViewCount,
                                                                       int minimumSupportingViews,
                                                                       int maximumConflictViews)
    {
        return supportingViewCount >= std::max(2, minimumSupportingViews) &&
               conflictViewCount <= std::max(0, maximumConflictViews);
    }

    bool DepthTsdfSurfaceBuilder::shouldApplyVisibilityOccupancyCut(
        bool cutOk, bool emptyCut, std::uint64_t sampleCount, std::uint64_t fullSampleCount, float minimumFullFraction)
    {
        if (!cutOk || emptyCut || sampleCount == 0)
        {
            return false;
        }
        const double full_fraction = static_cast<double>(fullSampleCount) / static_cast<double>(sampleCount);
        return full_fraction >= static_cast<double>(std::clamp(minimumFullFraction, 0.0f, 0.25f));
    }

    DepthTsdfVisualHullTopologyGuardEvaluation
    DepthTsdfSurfaceBuilder::evaluateVisualHullCompletionTopologyGuard(const TriMesh& baseline,
                                                                       const TriMesh& candidate,
                                                                       int maximumTopologicalComplexityIncrease,
                                                                       double maximumSurfaceAreaRatio,
                                                                       double maximumBoundsDiagonalRatio)
    {
        DepthTsdfVisualHullTopologyGuardEvaluation evaluation;
        evaluation.baselineFaceCount = baseline.faceCount();
        evaluation.candidateFaceCount = candidate.faceCount();
        if (baseline.empty() || candidate.empty())
        {
            evaluation.rejectionFlags |= VisualHullTopologyGuardEmptyMesh;
            return evaluation;
        }

        const MeshTopologyQualityStatistics baseline_quality = evaluateMeshTopologyQuality(baseline);
        const MeshTopologyQualityStatistics candidate_quality = evaluateMeshTopologyQuality(candidate);
        evaluation.baselineBoundaryEdgeCount = baseline_quality.boundaryEdgeCount;
        evaluation.candidateBoundaryEdgeCount = candidate_quality.boundaryEdgeCount;
        evaluation.baselineNonManifoldEdgeCount = baseline_quality.nonManifoldEdgeCount;
        evaluation.candidateNonManifoldEdgeCount = candidate_quality.nonManifoldEdgeCount;
        evaluation.baselineComponentCount = baseline_quality.componentCount;
        evaluation.candidateComponentCount = candidate_quality.componentCount;
        evaluation.baselineEulerCharacteristic = baseline_quality.eulerCharacteristic;
        evaluation.candidateEulerCharacteristic = candidate_quality.eulerCharacteristic;
        evaluation.baselineTopologicalComplexity = baseline_quality.topologicalComplexity;
        evaluation.candidateTopologicalComplexity = candidate_quality.topologicalComplexity;

        if (candidate_quality.boundaryEdgeCount > baseline_quality.boundaryEdgeCount)
        {
            evaluation.rejectionFlags |= VisualHullTopologyGuardBoundaryEdgeGrowth;
        }
        if (candidate_quality.nonManifoldEdgeCount > baseline_quality.nonManifoldEdgeCount)
        {
            evaluation.rejectionFlags |= VisualHullTopologyGuardNonManifoldEdgeGrowth;
        }
        if (candidate_quality.nonManifoldVertexCount > baseline_quality.nonManifoldVertexCount)
        {
            evaluation.rejectionFlags |= VisualHullTopologyGuardNonManifoldVertexGrowth;
        }
        if (candidate_quality.componentCount > baseline_quality.componentCount)
        {
            evaluation.rejectionFlags |= VisualHullTopologyGuardComponentGrowth;
        }
        if (candidate_quality.topologicalComplexity >
            baseline_quality.topologicalComplexity + std::max(0, maximumTopologicalComplexityIncrease))
        {
            evaluation.rejectionFlags |= VisualHullTopologyGuardTopologicalComplexityGrowth;
        }

        const double baseline_area = meshSurfaceArea(baseline);
        const double candidate_area = meshSurfaceArea(candidate);
        evaluation.surfaceAreaRatio =
            baseline_area > 1.0e-18 ? candidate_area / baseline_area : std::numeric_limits<double>::infinity();
        if (!std::isfinite(evaluation.surfaceAreaRatio) ||
            evaluation.surfaceAreaRatio > std::max(1.0, maximumSurfaceAreaRatio))
        {
            evaluation.rejectionFlags |= VisualHullTopologyGuardSurfaceAreaGrowth;
        }

        const double baseline_diagonal = meshBoundsDiagonal(baseline);
        const double candidate_diagonal = meshBoundsDiagonal(candidate);
        evaluation.boundsDiagonalRatio = baseline_diagonal > 1.0e-18 ? candidate_diagonal / baseline_diagonal
                                                                     : std::numeric_limits<double>::infinity();
        if (!std::isfinite(evaluation.boundsDiagonalRatio) ||
            evaluation.boundsDiagonalRatio > std::max(1.0, maximumBoundsDiagonalRatio))
        {
            evaluation.rejectionFlags |= VisualHullTopologyGuardBoundsDiagonalGrowth;
        }

        evaluation.accepted = evaluation.rejectionFlags == VisualHullTopologyGuardAccepted;
        return evaluation;
    }

    bool DepthTsdfSurfaceBuilder::finalizeMeasuredSupportTopologyTransaction(
        const MeshTopologyQualityStatistics& baselineQuality,
        const MeshTopologyQualityStatistics& candidateQuality,
        const std::vector<std::uint8_t>& baselineSupport,
        std::uint64_t attemptedRecoveredSampleCount,
        std::vector<std::uint8_t>* candidateSupport,
        DepthTsdfStatistics* statistics)
    {
        if (!statistics)
        {
            if (candidateSupport)
            {
                *candidateSupport = baselineSupport;
            }
            return false;
        }

        DepthTsdfRecoveryTransactionEvaluation evaluation =
            evaluateDepthTsdfRecoveryTransaction(baselineQuality, candidateQuality);
        const bool committed = commitDepthTsdfRecoveryTransaction(evaluation, baselineSupport, candidateSupport);
        if (evaluation.accepted && !committed)
        {
            evaluation.accepted = false;
            evaluation.rejectionFlags |= DepthTsdfRecoveryTransactionSupportSizeMismatch;
            evaluation.reason = QStringLiteral("候选 support 尺寸与事务基线不一致");
        }

        statistics->measuredSupportTopologyTransactionEvaluated = true;
        statistics->measuredSupportTopologyTransactionAccepted = committed;
        statistics->measuredSupportTopologyTransactionRejectionFlags = evaluation.rejectionFlags;
        statistics->measuredSupportTopologyTransactionRejectionReason = evaluation.reason;
        statistics->measuredSupportRecoveredSampleCount = attemptedRecoveredSampleCount;
        statistics->measuredSupportAppliedRecoveredSampleCount = committed ? attemptedRecoveredSampleCount : 0;
        statistics->measuredSupportTopologyTransactionBaselineValidFaceCount = baselineQuality.validFaceCount;
        statistics->measuredSupportTopologyTransactionCandidateValidFaceCount = candidateQuality.validFaceCount;
        statistics->measuredSupportTopologyTransactionBaselineBoundaryEdgeCount = baselineQuality.boundaryEdgeCount;
        statistics->measuredSupportTopologyTransactionCandidateBoundaryEdgeCount = candidateQuality.boundaryEdgeCount;
        statistics->measuredSupportTopologyTransactionBaselineComponentCount = baselineQuality.componentCount;
        statistics->measuredSupportTopologyTransactionCandidateComponentCount = candidateQuality.componentCount;
        statistics->measuredSupportTopologyTransactionBaselineNonManifoldEdgeCount =
            baselineQuality.nonManifoldEdgeCount;
        statistics->measuredSupportTopologyTransactionCandidateNonManifoldEdgeCount =
            candidateQuality.nonManifoldEdgeCount;
        statistics->measuredSupportTopologyTransactionBaselineNonManifoldVertexCount =
            baselineQuality.nonManifoldVertexCount;
        statistics->measuredSupportTopologyTransactionCandidateNonManifoldVertexCount =
            candidateQuality.nonManifoldVertexCount;
        statistics->measuredSupportTopologyTransactionBaselineTopologicalComplexity =
            baselineQuality.topologicalComplexity;
        statistics->measuredSupportTopologyTransactionCandidateTopologicalComplexity =
            candidateQuality.topologicalComplexity;
        statistics->measuredSupportTopologyTransactionBaselineLargestComponentFaceRatio =
            baselineQuality.largestComponentFaceRatio;
        statistics->measuredSupportTopologyTransactionCandidateLargestComponentFaceRatio =
            candidateQuality.largestComponentFaceRatio;
        statistics->measuredSupportTopologyTransactionBaselineExtremeAspectFaceRatio =
            baselineQuality.extremeAspectFaceRatio;
        statistics->measuredSupportTopologyTransactionCandidateExtremeAspectFaceRatio =
            candidateQuality.extremeAspectFaceRatio;
        return committed;
    }

    bool DepthTsdfSurfaceBuilder::shouldTrimWeakBoundaryFace(int boundaryEdgeCount, int weakVertexCount)
    {
        // Remove ordinary boundary ears, and also peel a one-edge boundary face
        // only when every vertex comes from weak camera support.  The latter is
        // the common topology of the narrow triangular ribbons visible along
        // Temple's column silhouettes; requiring all three weak vertices avoids
        // eroding a well-supported surface merely because it reaches an opening.
        return (boundaryEdgeCount >= 2 && weakVertexCount >= 1) || (boundaryEdgeCount == 1 && weakVertexCount == 3);
    }

    bool DepthTsdfSurfaceBuilder::shouldAcceptQuadricSimplification(int inputFaceCount,
                                                                    int outputFaceCount,
                                                                    int boundaryEdgeCountBefore,
                                                                    int boundaryEdgeCountAfter,
                                                                    float maximumBoundaryEdgeGrowthRatio)
    {
        if (inputFaceCount <= 0 || outputFaceCount >= inputFaceCount || boundaryEdgeCountBefore < 0 ||
            boundaryEdgeCountAfter < 0)
        {
            return false;
        }
        if (boundaryEdgeCountBefore == 0)
        {
            return boundaryEdgeCountAfter == 0;
        }
        const int allowed_growth = static_cast<int>(
            std::ceil(boundaryEdgeCountBefore * std::clamp(maximumBoundaryEdgeGrowthRatio, 0.0f, 1.0f)));
        return boundaryEdgeCountAfter <= boundaryEdgeCountBefore + allowed_growth;
    }

    bool DepthTsdfSurfaceBuilder::shouldAttemptVoxelFallbackSimplification(int currentFaceCount,
                                                                           int targetFaceCount,
                                                                           bool quadricReachedTarget)
    {
        return targetFaceCount > 0 && !quadricReachedTarget && currentFaceCount > targetFaceCount;
    }

    bool DepthTsdfSurfaceBuilder::isCatastrophicComponentFilterLoss(int inputFaceCount,
                                                                    int outputFaceCount,
                                                                    int minimumComponentFaces)
    {
        const int minimum_viable_component_faces = std::clamp(minimumComponentFaces, 2, 64);
        return inputFaceCount >= 256 && outputFaceCount < minimum_viable_component_faces &&
               static_cast<std::int64_t>(outputFaceCount) * 1000 < inputFaceCount;
    }
} // namespace xjw::mesh

namespace xjw::mesh
{
    DepthTsdfResult DepthTsdfSurfaceBuilder::build(const QVector<DepthTsdfFrame>& frames,
                                                   const DepthTsdfOptions& requested)
    {
        DepthTsdfOptions options = requested;
        const auto control = options.execution;
        if (control.cancellation || control.progress)
        {
            options.isCancelled = task_runtime::combineWorkflowCancellation(options.isCancelled, control);
            const auto legacy = options.progress;
            options.progress = [control, legacy](const QString& stage, int percent)
            {
                control.reportProgress(stage.toUtf8().toStdString(), percent / 100.0);
                if (legacy)
                    legacy(stage, percent);
            };
            options.execution = {};
        }
        DepthTsdfResult result;
        if (options.isCancelled && options.isCancelled())
        {
            result.cancelled = true;
            result.errorMessage = QStringLiteral("TSDF reconstruction cancelled");
            return result;
        }
        try
        {
            result = tsdf_detail::buildTsdfVolume(frames, options);
        }
        catch (const std::exception& error)
        {
            result.errorMessage = QStringLiteral("TSDF reconstruction failed: %1").arg(QString::fromUtf8(error.what()));
        }
        if (!result.ok && options.isCancelled && options.isCancelled())
            result.cancelled = true;
        return result;
    }
} // namespace xjw::mesh
