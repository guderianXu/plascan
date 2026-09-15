#pragma once

#include <QString>
#include <QStringList>

// Explicit source sets for structural contracts after stage extraction.
// Behavioral tests exercise independently linked targets, not these text bundles.
namespace xjw::tests
{

    template <class Read> QString readMvsPipelineImplementation(Read read)
    {
        const QStringList paths = {
            QStringLiteral("src/core/mvs/pipeline/DepthEvidenceRules.cpp"),
            QStringLiteral("src/core/mvs/pipeline/DepthConsistencyVoting.cpp"),
            QStringLiteral("src/core/mvs/pipeline/DepthDiagnostics.cpp"),
            QStringLiteral("src/core/mvs/pipeline/SourcePlanningUtils.cpp"),
            QStringLiteral("src/core/mvs/pipeline/DepthResourcePolicy.cpp"),
            QStringLiteral("src/core/mvs/pipeline/DepthArtifactIO.cpp"),
            QStringLiteral("src/core/mvs/pipeline/FrameBackend.cpp"),
            QStringLiteral("src/core/mvs/pipeline/FrameBackendBridge.cpp"),
            QStringLiteral("src/core/mvs/pipeline/DepthArtifactSaveQueue.h"),
            QStringLiteral("src/core/mvs/pipeline/AdaptivePatchMatchBackend.h"),
            QStringLiteral("src/core/mvs/pipeline/DepthPublicationContract.cpp"),
            QStringLiteral("src/core/mvs/pipeline/ImageMasks.cpp"),
            QStringLiteral("src/core/mvs/pipeline/MvsPipelineService.cpp"),
            QStringLiteral("src/core/mvs/pipeline/DepthWorkspace.cpp"),
            QStringLiteral("src/core/mvs/pipeline/PreparedRasterPublisher.cpp"),
            QStringLiteral("src/core/mvs/pipeline/SparseVisibility.cpp"),
            QStringLiteral("src/core/mvs/pipeline/SourcePlanning.cpp"),
            QStringLiteral("src/core/mvs/pipeline/ImagePreparation.cpp"),
            QStringLiteral("src/core/mvs/pipeline/SparseDepthRange.cpp"),
            QStringLiteral("src/core/mvs/pipeline/SparseDepthHints.cpp"),
            QStringLiteral("src/core/mvs/pipeline/SparseSupportPrior.cpp"),
            QStringLiteral("src/core/mvs/pipeline/FrameEstimation.cpp"),
            QStringLiteral("src/core/mvs/pipeline/FusionFrameAssembler.cpp"),
            QStringLiteral("src/core/mvs/pipeline/DepthConsistency.cpp"),
            QStringLiteral("src/core/mvs/pipeline/ExperimentalDepthStages.cpp"),
            QStringLiteral("src/core/mvs/pipeline/DepthRecovery.cpp"),
            QStringLiteral("src/core/mvs/pipeline/StreamingConsistency.cpp"),
            QStringLiteral("src/core/mvs/pipeline/DepthArtifactPublisher.cpp"),
            QStringLiteral("src/core/mvs/pipeline/MvsPipelineExecution.cpp"),
        };
        QString source;
        for (const auto& path : paths)
        {
            source += QLatin1Char('\n') + read(path);
        }
        return source;
    }

    template <class Read> QString readModelWorkflowImplementation(Read read)
    {
        const QStringList paths = {
            QStringLiteral("src/core/mesh/workflow/ModelGenerationContract.cpp"),
            QStringLiteral("src/core/mesh/workflow/ModelArtifactPublisher.cpp"),
            QStringLiteral("src/core/mesh/workflow/ModelWorkflowControl.cpp"),
            QStringLiteral("src/core/mesh/workflow/ModelTextureStage.cpp"),
            QStringLiteral("src/core/mesh/workflow/ModelTsdfPolicy.cpp"),
            QStringLiteral("src/core/mesh/workflow/ModelSurfaceQuality.cpp"),
            QStringLiteral("src/core/mesh/workflow/DepthModelInputPolicy.cpp"),
            QStringLiteral("src/core/mesh/workflow/OrbitalDepthModelPolicy.cpp"),
            QStringLiteral("src/core/mesh/workflow/ModelSurfaceDenoising.cpp"),
            QStringLiteral("src/core/mesh/workflow/DepthModelQuality.cpp"),
            QStringLiteral("src/core/mesh/workflow/ModelSettings.cpp"),
            QStringLiteral("src/core/mesh/workflow/PointCloudModelWorkflow.cpp"),
            QStringLiteral("src/core/mesh/workflow/DepthModelWorkflow.cpp"),
            QStringLiteral("src/core/mesh/workflow/RecoveredDepthModelStage.cpp"),
            QStringLiteral("src/core/mesh/ModelWorkflowService.cpp"),
            QStringLiteral("src/core/mesh/workflow/TextureWorkflow.cpp"),
        };
        QString source;
        for (const auto& path : paths)
        {
            source += QLatin1Char('\n') + read(path);
        }
        return source;
    }

    template <class Read> QString readTsdfImplementation(Read read)
    {
        const QStringList paths = {
            QStringLiteral("src/core/mesh/tsdf/TsdfAllocation.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfTopology.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfHoleFillEvidence.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfWeakBoundaryTips.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfIsoSurface.cpp"),
            QStringLiteral("src/core/mesh/tsdf/DepthFrameReader.cpp"),
            QStringLiteral("src/core/mesh/tsdf/DepthByteMapReader.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfSampleEvidence.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfIntegration.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfSurface.cpp"),
            QStringLiteral("src/core/mesh/DepthTsdfSurfaceBuilder.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfLayout.cpp"),
            QStringLiteral("src/core/mesh/tsdf/DepthFrameLoading.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfInputPlanning.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfObservation.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfRecovery.cpp"),
            QStringLiteral("src/core/mesh/tsdf/DepthTsdfPipeline.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfStatistics.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfSupportStage.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfIsoSurfaceStage.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfMeshCleanup.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfMeshSimplification.cpp"),
            QStringLiteral("src/core/mesh/tsdf/TsdfMeshFinalization.cpp"),
        };
        QString source;
        for (const auto& path : paths)
        {
            source += QLatin1Char('\n') + read(path);
        }
        return source;
    }

} // namespace xjw::tests
