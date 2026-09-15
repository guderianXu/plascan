#include "DepthTsdfStages.h"
namespace xjw::mesh::tsdf_detail
{
    bool cleanTsdfMesh(const QVector<DepthTsdfFrame>& frames,
                       const DepthTsdfOptions& options,
                       DepthTsdfResult& result,
                       TsdfPostprocessContext& context)
    {
        auto& state = context.surface;

        auto& effective_frame_quality_weights = state.effective_frame_quality_weights;
        auto& effective_depth_valid_masks = state.effective_depth_valid_masks;
        auto& erosion_pixels = state.erosion_pixels;
        auto& geometry_source_encoding = state.geometry_source_encoding;
        auto& tsdf = state.tsdf;
        auto& weight = state.weight;
        auto& strongAdaptiveSurfaceObservation = state.strongAdaptiveSurfaceObservation;
        auto& support = state.support;
        auto& geometrySourceMask = state.geometrySourceMask;
        auto& minimumInverseDepthSpread = state.minimumInverseDepthSpread;
        auto& surfaceObservationWeight = state.surfaceObservationWeight;
        auto& maximum_voxel_size = state.maximum_voxel_size;
        auto& truncation = state.truncation;
        auto& cancelled = state.cancelled;
        auto& workerCount = state.workerCount;
        auto& supported = state.supported;
        auto& adaptiveTgvExtractionSupport = state.adaptiveTgvExtractionSupport;
        auto& visual_hull_completion_tsdf = state.visual_hull_completion_tsdf;
        auto& visual_hull_completion_support = state.visual_hull_completion_support;
        auto& native_carrier_bounds_min = state.native_carrier_bounds_min;
        auto& native_carrier_bounds_max = state.native_carrier_bounds_max;
        auto& native_carrier_dimensions = state.native_carrier_dimensions;
        auto& native_carrier_cells = state.native_carrier_cells;
        auto& native_carrier_occupied = state.native_carrier_occupied;
        auto& native_carrier_field = state.native_carrier_field;

        using PostprocessClock = std::chrono::steady_clock;
        const auto elapsedMilliseconds = [](const PostprocessClock::time_point& start)
        { return std::chrono::duration_cast<std::chrono::milliseconds>(PostprocessClock::now() - start).count(); };

        auto& capture_stage = context.capture;
        auto& reportProgress = context.progress;
        auto& postprocessCancelled = context.cancelled;
        const auto& postprocess_start = context.started;

        const PostprocessClock::time_point cleanup_start = PostprocessClock::now();
        context.minimumDegenerateFaceArea = options.visibilityOccupancyCellBoundaryExtraction ||
                                                    options.enableConsistentIsoSurfaceExtraction ||
                                                    options.enableMc33IsoSurfaceExtraction
                                                ? 0.0f
                                                : 5.0e-9f;
        const int faces_before_degenerate_cleanup = result.mesh.faceCount();
        reportProgress(QStringLiteral("正在移除退化三角面..."), 82);
        detail::removeDegenerateFaces(&result.mesh, context.minimumDegenerateFaceArea);
        result.statistics.initialDegenerateRemovedFaceCount =
            std::max(0, faces_before_degenerate_cleanup - result.mesh.faceCount());
        if (!options.visibilityOccupancyCellBoundaryExtraction && !options.enableConsistentIsoSurfaceExtraction &&
            !options.enableMc33IsoSurfaceExtraction)
        {
            const float maximum_voxel_size =
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            detail::weldCoincidentVertices(&result.mesh, 1.0e-6f, maximum_voxel_size * 2.0e-3f);
            const int faces_before_post_weld_cleanup = result.mesh.faceCount();
            detail::removeDegenerateFaces(&result.mesh, context.minimumDegenerateFaceArea);
            result.statistics.initialDegenerateRemovedFaceCount +=
                std::max(0, faces_before_post_weld_cleanup - result.mesh.faceCount());
        }
        const int faces_before_component_filter = result.mesh.faceCount();
        reportProgress(QStringLiteral("正在过滤孤立网格组件（%1 面）...").arg(faces_before_component_filter), 83);
        detail::removeSmallConnectedComponents(
            &result.mesh, std::max(2, options.minimumComponentFaces), options.minimumComponentFaceRatio);
        result.statistics.componentFilterRemovedFaceCount =
            std::max(0, faces_before_component_filter - result.mesh.faceCount());
        result.statistics.componentFilteredFaceCount = result.mesh.faceCount();
        const bool catastrophic_component_loss = DepthTsdfSurfaceBuilder::isCatastrophicComponentFilterLoss(
            faces_before_component_filter, result.mesh.faceCount(), options.minimumComponentFaces);
        if (catastrophic_component_loss)
        {
            result.errorMessage = QStringLiteral("TSDF 网格连通性异常：组件过滤前 %1 面，过滤后仅 %2 面。"
                                                 "这通常表示等值面顶点未正确共享或焊接，已停止保存退化模型。")
                                      .arg(faces_before_component_filter)
                                      .arg(result.mesh.faceCount());
            return false;
        }
        reportProgress(QStringLiteral("正在统计清理后边界（%1 面）...").arg(result.mesh.faceCount()), 83);
        result.statistics.componentFilteredBoundaryEdgeCount = boundaryEdgeCount(result.mesh);
        const WeakBoundaryTipResult weak_tips = trimWeakBoundaryTips(&result.mesh,
                                                                     result.layout,
                                                                     support,
                                                                     strongAdaptiveSurfaceObservation,
                                                                     std::max(2, options.minimumDistinctCameraSupport),
                                                                     options.weakBoundaryTipTrimPasses,
                                                                     options.trimWeakBoundaryTips);
        result.statistics.weakBoundaryTipVertexCount = weak_tips.weakVertexCount;
        result.statistics.candidateWeakBoundaryTipFaceCount = weak_tips.candidateFaceCount;
        result.statistics.trimmedWeakBoundaryTipFaceCount = weak_tips.trimmedFaceCount;
        if (weak_tips.trimmedFaceCount > 0)
        {
            detail::removeSmallConnectedComponents(
                &result.mesh, std::max(2, options.minimumComponentFaces), options.minimumComponentFaceRatio);
        }
        result.statistics.weakBoundaryTrimmedBoundaryEdgeCount = boundaryEdgeCount(result.mesh);
        reportProgress(QStringLiteral("正在清理 TSDF 网格拓扑..."), 84);
        if (postprocessCancelled())
        {
            return false;
        }
        if (options.fillSmallBoundaryHoles || options.enableQuadricSimplification)
        {
            result.statistics.removedDuplicateFaceCount = detail::removeDuplicateFaces(&result.mesh);
            result.statistics.removedNonManifoldFaceCount = detail::removeNonManifoldFaces(&result.mesh);
            detail::removeDegenerateFaces(&result.mesh, context.minimumDegenerateFaceArea);
        }
        result.statistics.topologyCleanedBoundaryEdgeCount = boundaryEdgeCount(result.mesh);
        if (options.collectZeroCrossingDiagnostics)
        {
            MeshBoundaryAttributionOptions attribution_options;
            attribution_options.minimumSourceCount = options.minimumSurfacePatchSourceCount;
            attribution_options.maximumInverseDepthSpread = options.maximumSurfacePatchInverseDepthSpread;
            attribution_options.minimumSurfaceWeightRatio = options.minimumSurfacePatchWeightRatio;
            attribution_options.maximumAbsoluteTsdf = options.maximumContourBandAbsoluteTsdf;
            const MeshBoundaryAttributionStatistics attribution = attributeMeshBoundaryEdges(result.mesh,
                                                                                             result.layout,
                                                                                             tsdf,
                                                                                             weight,
                                                                                             surfaceObservationWeight,
                                                                                             geometrySourceMask,
                                                                                             minimumInverseDepthSpread,
                                                                                             supported,
                                                                                             attribution_options);
            result.statistics.topologyCleanedAttributionEdgeCount = attribution.boundaryEdgeCount;
            result.statistics.topologyCleanedAttributionNoObservationEdgeCount = attribution.noObservationEdgeCount;
            result.statistics.topologyCleanedAttributionInsufficientSourceEdgeCount =
                attribution.insufficientSourceEdgeCount;
            result.statistics.topologyCleanedAttributionDepthSpreadRejectedEdgeCount =
                attribution.depthSpreadRejectedEdgeCount;
            result.statistics.topologyCleanedAttributionSurfaceWeightRejectedEdgeCount =
                attribution.surfaceWeightRejectedEdgeCount;
            result.statistics.topologyCleanedAttributionAbsoluteTsdfRejectedEdgeCount =
                attribution.absoluteTsdfRejectedEdgeCount;
            result.statistics.topologyCleanedAttributionSupportGateRejectedEdgeCount =
                attribution.supportGateRejectedEdgeCount;
            result.statistics.topologyCleanedAttributionExtractionOrPostprocessEdgeCount =
                attribution.extractionOrPostprocessEdgeCount;
            result.statistics.topologyCleanedAttributionUnclassifiedEdgeCount = attribution.unclassifiedEdgeCount;
        }
        result.statistics.boundaryEdgeCountBefore = boundaryEdgeCount(result.mesh);
        if (options.fillSmallBoundaryHoles)
        {
            if (options.splitPinchedBoundaryVertices)
            {
                result.statistics.splitPinchedBoundaryVertexCount = detail::splitPinchedBoundaryVertices(&result.mesh);
            }
            const int faces_before = result.mesh.faceCount();
            const float maximum_voxel_size =
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            result.statistics.filledBoundaryHoleCount =
                detail::fillSmallBoundaryHoles(&result.mesh,
                                               std::max(3, options.maximumHoleBoundaryEdges),
                                               std::max(0.0f, options.maximumHoleDiameterVoxels) * maximum_voxel_size);
            result.statistics.addedHoleFillFaceCount = std::max(0, result.mesh.faceCount() - faces_before);
            detail::removeDegenerateFaces(&result.mesh, context.minimumDegenerateFaceArea);
        }
        if (options.boundarySmoothingIterations > 0)
        {
            const float maximum_voxel_size =
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            result.statistics.smoothedBoundaryVertexCount = detail::smoothOpenBoundaryVertices(
                &result.mesh,
                options.boundarySmoothingIterations,
                options.boundarySmoothingLambda,
                options.maximumBoundarySmoothingDisplacementVoxels * maximum_voxel_size);
        }
        const auto pre_denoising_orientation = repairFaceOrientationSafely(&result.mesh);
        result.statistics.preDenoisingFaceOrientationRepairAccepted = pre_denoising_orientation.second;
        result.statistics.preDenoisingFaceOrientationInconsistentSharedEdgeCountBefore =
            pre_denoising_orientation.first.inconsistentSharedEdgeCountBefore;
        result.statistics.preDenoisingFaceOrientationFlippedFaceCount =
            pre_denoising_orientation.first.flippedFaceCount;
        result.statistics.effectiveSurfaceDenoisingIterations = options.surfaceDenoisingIterations;
        result.statistics.effectiveSurfaceDenoisingLambda = options.surfaceDenoisingLambda;
        result.statistics.effectiveSurfaceDenoisingMu = options.surfaceDenoisingMu;
        result.statistics.effectiveMaximumSurfaceDenoisingDisplacementVoxels =
            options.maximumSurfaceDenoisingDisplacementVoxels;
        result.statistics.effectiveMaximumSurfaceDenoisingNormalAngleDegrees =
            options.maximumSurfaceDenoisingNormalAngleDegrees;
        result.statistics.effectiveSurfaceDenoisingBoundaryProtectionRings =
            options.surfaceDenoisingBoundaryProtectionRings;
        result.statistics.effectiveProtectedTaubinSurfaceDenoising = options.enableProtectedTaubinSurfaceDenoising;
        result.statistics.effectivePostSimplificationSurfaceDenoising =
            options.enablePostSimplificationSurfaceDenoising && options.surfaceDenoisingIterations > 0;
        // Protected Taubin denoising is applied only after simplification, where
        // its topology and normal-quality guard can reject an unsafe candidate.
        if (options.surfaceDenoisingIterations > 0 && !options.enableProtectedTaubinSurfaceDenoising)
        {
            const float maximum_voxel_size =
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            result.statistics.smoothedSurfaceVertexCount = detail::smoothSurfaceVerticesNormalAware(
                &result.mesh,
                options.surfaceDenoisingIterations,
                options.surfaceDenoisingLambda,
                options.maximumSurfaceDenoisingDisplacementVoxels * maximum_voxel_size,
                options.maximumSurfaceDenoisingNormalAngleDegrees,
                options.surfaceDenoisingBoundaryProtectionRings);
        }
        detail::removeDegenerateFaces(&result.mesh, context.minimumDegenerateFaceArea);
        if (options.fillSmallBoundaryHoles)
        {
            result.statistics.splitPinchedBoundaryVertexCount += detail::splitPinchedBoundaryVertices(&result.mesh);
            const int faces_before_refill = result.mesh.faceCount();
            const float maximum_voxel_size =
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            result.statistics.filledBoundaryHoleCount +=
                detail::fillSmallBoundaryHoles(&result.mesh,
                                               std::max(3, options.maximumHoleBoundaryEdges),
                                               std::max(0.0f, options.maximumHoleDiameterVoxels) * maximum_voxel_size);
            result.statistics.addedHoleFillFaceCount += std::max(0, result.mesh.faceCount() - faces_before_refill);
            detail::removeDegenerateFaces(&result.mesh, context.minimumDegenerateFaceArea);
        }
        reportProgress(QStringLiteral("正在修补和整理网格边界..."), 86);
        if (postprocessCancelled())
        {
            return false;
        }
        result.statistics.compactedUnusedVertexCount = detail::compactReferencedVertices(&result.mesh);
        result.statistics.preSimplificationFaceCount = result.mesh.faceCount();
        capture_stage(QStringLiteral("02_pre_simplification"), result.mesh);
        result.statistics.meshCleanupElapsedMs = elapsedMilliseconds(cleanup_start);
        result.statistics.effectiveVoxelFallbackSimplification = options.enableVoxelFallbackSimplification;
        result.statistics.effectiveVoxelFallbackQemPolish = options.enableVoxelFallbackQemPolish;
        result.statistics.effectiveVoxelFallbackMinimumProtectedBoundaryVertices =
            options.voxelFallbackMinimumProtectedBoundaryVertices;
        result.statistics.effectiveVoxelFallbackMaximumCollapsibleBoundaryDiameterVoxels =
            options.voxelFallbackMaximumCollapsibleBoundaryDiameterVoxels;
        result.statistics.effectiveVoxelFallbackMaximumNormalClusterAngleDegrees =
            options.voxelFallbackMaximumNormalClusterAngleDegrees;
        result.statistics.effectiveVoxelFallbackInitialClusterFactor = options.voxelFallbackInitialClusterFactor;
        result.statistics.effectiveVoxelFallbackMultiViewSilhouetteProtection =
            options.enableVoxelFallbackMultiViewSilhouetteProtection;
        result.statistics.effectiveVoxelFallbackMinimumSilhouetteViews = options.voxelFallbackMinimumSilhouetteViews;
        result.statistics.effectiveVoxelFallbackSilhouetteBandPixels = options.voxelFallbackSilhouetteBandPixels;
        result.statistics.effectiveVoxelFallbackSilhouetteDepthToleranceVoxels =
            options.voxelFallbackSilhouetteDepthToleranceVoxels;
        result.statistics.requestedSimplifyTargetFaces = options.simplifyTargetFaces;

        return true;
    }
} // namespace xjw::mesh::tsdf_detail

namespace xjw::mesh::tsdf_detail
{
    std::pair<MeshFaceOrientationStatistics, bool> repairFaceOrientationSafely(TriMesh* mesh)
    {
        MeshFaceOrientationStatistics statistics;
        if (!mesh || mesh->faces.empty())
        {
            return std::make_pair(statistics, false);
        }
        TriMesh candidate = *mesh;
        statistics = repairMeshFaceOrientation(&candidate);
        const bool accepted = statistics.succeeded && statistics.removedContradictoryFaceCount == 0 &&
                              statistics.nonManifoldEdgeCount == 0 && candidate.faceCount() == mesh->faceCount();
        if (accepted)
        {
            *mesh = std::move(candidate);
        }
        return std::make_pair(statistics, accepted);
    }
} // namespace xjw::mesh::tsdf_detail
