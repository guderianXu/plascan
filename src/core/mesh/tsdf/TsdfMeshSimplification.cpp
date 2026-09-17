#include "DepthTsdfStages.h"
namespace xjw::mesh::tsdf_detail
{
    bool simplifyTsdfMesh(const QVector<DepthTsdfFrame>& frames,
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

        const PostprocessClock::time_point simplification_start = PostprocessClock::now();
        if (options.enableTopologySafeSimplification && options.simplifyTargetFaces > 0 &&
            result.mesh.faceCount() > options.simplifyTargetFaces)
        {
            reportProgress(QStringLiteral("正在进行原生拓扑安全简化..."), 87);
            result.statistics.topologySafeSimplificationAttempted = true;
            result.statistics.topologySafeSimplificationInputVertexCount = result.mesh.vertexCount();
            result.statistics.topologySafeSimplificationInputFaceCount = result.mesh.faceCount();

            TriMesh candidate = result.mesh;
            const MeshBoundaryTopology topology_before = boundaryTopology(result.mesh);
            const MeshTopologyQualityStatistics quality_before = evaluateMeshTopologyQuality(result.mesh);
            result.statistics.topologySafeBoundaryEdgeCountBefore = topology_before.boundaryEdgeCount;
            result.statistics.topologySafeNonManifoldEdgeCountBefore = topology_before.nonManifoldEdgeCount;

            NativeMeshSimplifyOptions simplify_options;
            simplify_options.targetFaceCount = options.simplifyTargetFaces;
            simplify_options.maximumPasses = options.simplificationMaximumPasses;
            simplify_options.workerCount = options.workerCount;
            simplify_options.minimumFaceArea = context.minimumDegenerateFaceArea;
            simplify_options.featureAngleDegrees = options.simplificationFeatureAngleDegrees;
            simplify_options.maximumNormalDeviationDegrees = options.topologySafeMaximumNormalDeviationDegrees;
            simplify_options.maximumNormalFlippingDegrees = options.topologySafeMaximumNormalFlippingDegrees;
            simplify_options.smoothingIterations = options.topologySafeSmoothingIterations;
            simplify_options.smoothingMaximumDisplacement =
                options.topologySafeSmoothingMaximumDisplacementVoxels *
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            simplify_options.smoothingFeatureAngleDegrees = options.topologySafeSmoothingFeatureAngleDegrees;
            simplify_options.isCancelled = [control = options.execution]() { return control.isCancelled(); };
            simplify_options.progress = [&reportProgress](int pass, int face_count)
            {
                reportProgress(
                    QStringLiteral("正在进行原生拓扑安全简化（第 %1 轮，%2 面）...").arg(pass).arg(face_count),
                    std::min(89, 87 + pass / 8));
            };

            const NativeMeshSimplifyStatistics simplify_statistics =
                simplifyMeshTopologySafe(&candidate, simplify_options);
            const MeshBoundaryTopology topology_after = boundaryTopology(candidate);
            const MeshTopologyQualityStatistics quality_after = evaluateMeshTopologyQuality(candidate);
            result.statistics.topologySafeSimplificationOutputVertexCount = candidate.vertexCount();
            result.statistics.topologySafeSimplificationOutputFaceCount = candidate.faceCount();
            result.statistics.topologySafeSimplificationCollapsedEdgeCount = simplify_statistics.collapsedEdgeCount;
            result.statistics.topologySafeSimplificationRejectedBoundaryEdgeCount =
                simplify_statistics.rejectedBoundaryEdgeCount;
            result.statistics.topologySafeSimplificationRejectedFeatureEdgeCount =
                simplify_statistics.rejectedFeatureEdgeCount;
            result.statistics.topologySafeSimplificationRejectedTopologyEdgeCount =
                simplify_statistics.rejectedTopologyEdgeCount;
            result.statistics.topologySafeSimplificationRejectedFlipEdgeCount =
                simplify_statistics.rejectedFlipEdgeCount;
            result.statistics.topologySafeSimplificationRejectedTriangleQualityEdgeCount =
                simplify_statistics.rejectedTriangleQualityEdgeCount;
            result.statistics.topologySafeSimplificationPassCount = simplify_statistics.passCount;
            result.statistics.topologySafeInconsistentSharedEdgeCountBefore =
                simplify_statistics.inconsistentSharedEdgeCountBefore;
            result.statistics.topologySafeReorientedInputFaceCount = simplify_statistics.reorientedInputFaceCount;
            result.statistics.topologySafeRemovedContradictoryFaceCount =
                simplify_statistics.removedContradictoryFaceCount;
            result.statistics.topologySafeOrientationConflictCount = simplify_statistics.orientationConflictCount;
            result.statistics.topologySafeSmoothingApplied = simplify_statistics.smoothingApplied;
            result.statistics.topologySafeSimplificationReachedTarget = simplify_statistics.reachedTarget;
            result.statistics.topologySafeSimplificationCancelled = simplify_statistics.cancelled;
            result.statistics.topologySafeSimplificationError = QString::fromStdString(simplify_statistics.error);
            result.statistics.topologySafeBoundaryEdgeCountAfter = topology_after.boundaryEdgeCount;
            result.statistics.topologySafeNonManifoldEdgeCountAfter = topology_after.nonManifoldEdgeCount;

            const auto topology_growth_is_safe = [](int before, int after)
            { return after <= before + std::max(8, static_cast<int>(std::ceil(before * 0.10f))); };
            const double maximum_accepted_normal_median =
                std::max(quality_before.adjacentNormalAngleMedianDegrees + 8.0,
                         static_cast<double>(options.topologySafeMaximumNormalDeviationDegrees));
            const double maximum_accepted_high_aspect_ratio = std::max(quality_before.highAspectFaceRatio * 1.5, 0.10);
            const bool accept_simplification =
                simplify_statistics.succeeded && candidate.faceCount() < result.mesh.faceCount() &&
                meshVerticesInsidePaddedLayout(candidate, result.layout) &&
                topology_growth_is_safe(topology_before.boundaryEdgeCount, topology_after.boundaryEdgeCount) &&
                topology_growth_is_safe(topology_before.danglingBoundaryVertexCount,
                                        topology_after.danglingBoundaryVertexCount) &&
                topology_after.nonManifoldEdgeCount <= topology_before.nonManifoldEdgeCount &&
                quality_after.adjacentNormalAngleMedianDegrees <= maximum_accepted_normal_median &&
                quality_after.highAspectFaceRatio <= maximum_accepted_high_aspect_ratio;
            result.statistics.topologySafeSimplificationAccepted = accept_simplification;
            result.statistics.effectiveTopologySafeSimplification = accept_simplification;
            if (accept_simplification)
            {
                result.mesh = std::move(candidate);
            }
            if (postprocessCancelled())
            {
                return false;
            }
        }
        if ((!result.statistics.topologySafeSimplificationAccepted ||
             !result.statistics.topologySafeSimplificationReachedTarget) &&
            options.enableQuadricSimplification && options.simplifyTargetFaces > 0 &&
            result.mesh.faceCount() > options.simplifyTargetFaces)
        {
            reportProgress(QStringLiteral("正在进行保拓扑 QEM 简化..."), 87);
            TriMesh unsimplified_mesh = result.mesh;
            const int input_face_count = result.mesh.faceCount();
            const MeshBoundaryTopology topology_before = boundaryTopology(result.mesh);
            const int simplification_boundary_edge_count_before = topology_before.boundaryEdgeCount;
            QuadricSimplifyOptions simplify_options;
            simplify_options.targetFaceCount = options.simplifyTargetFaces;
            simplify_options.maximumPasses = options.simplificationMaximumPasses;
            simplify_options.workerCount = options.workerCount;
            simplify_options.minimumFaceArea = context.minimumDegenerateFaceArea;
            simplify_options.featureAngleDegrees = options.simplificationFeatureAngleDegrees;
            simplify_options.maximumNormalDeviationDegrees = options.simplificationMaximumNormalDeviationDegrees;
            simplify_options.minimumSharpEdgeEndpointDegree = options.simplificationMinimumSharpEdgeEndpointDegree;
            simplify_options.simplifySimpleOpenBoundaries = options.simplifySimpleOpenBoundaries;
            simplify_options.isCancelled = [control = options.execution]() { return control.isCancelled(); };
            simplify_options.progress = [&reportProgress](int pass, int face_count)
            {
                reportProgress(
                    QStringLiteral("正在进行保拓扑 QEM 简化（第 %1 轮，%2 面）...").arg(pass).arg(face_count),
                    std::min(89, 87 + pass / 8));
            };
            const QuadricSimplifyStatistics simplify_statistics = simplifyMeshQuadric(&result.mesh, simplify_options);
            const MeshBoundaryTopology topology_after = boundaryTopology(result.mesh);
            const int simplified_boundary_edge_count = topology_after.boundaryEdgeCount;
            const auto topology_growth_is_safe = [](int before, int after)
            { return after <= before + std::max(8, static_cast<int>(std::ceil(before * 0.10f))); };
            const bool accept_simplification =
                DepthTsdfSurfaceBuilder::shouldAcceptQuadricSimplification(
                    input_face_count,
                    result.mesh.faceCount(),
                    simplification_boundary_edge_count_before,
                    simplified_boundary_edge_count,
                    options.maximumSimplificationBoundaryEdgeGrowthRatio) &&
                meshVerticesInsidePaddedLayout(result.mesh, result.layout) &&
                topology_growth_is_safe(topology_before.danglingBoundaryVertexCount,
                                        topology_after.danglingBoundaryVertexCount) &&
                topology_growth_is_safe(topology_before.nonManifoldEdgeCount, topology_after.nonManifoldEdgeCount);
            result.statistics.effectiveQuadricSimplification = accept_simplification;
            result.statistics.quadricSimplificationAccepted = accept_simplification;
            result.statistics.quadricSimplificationBoundarySafetyRejected =
                !accept_simplification && result.mesh.faceCount() < input_face_count;
            result.statistics.quadricBoundaryEdgeCountBefore = simplification_boundary_edge_count_before;
            result.statistics.quadricBoundaryEdgeCountAfter = simplified_boundary_edge_count;
            result.statistics.quadricDanglingBoundaryVertexCountBefore = topology_before.danglingBoundaryVertexCount;
            result.statistics.quadricDanglingBoundaryVertexCountAfter = topology_after.danglingBoundaryVertexCount;
            result.statistics.quadricNonManifoldEdgeCountBefore = topology_before.nonManifoldEdgeCount;
            result.statistics.quadricNonManifoldEdgeCountAfter = topology_after.nonManifoldEdgeCount;
            result.statistics.quadricCollapsedEdgeCount = simplify_statistics.collapsedEdgeCount;
            result.statistics.quadricRejectedBoundaryEdgeCount = simplify_statistics.rejectedBoundaryEdgeCount;
            result.statistics.quadricRejectedFeatureEdgeCount = simplify_statistics.rejectedFeatureEdgeCount;
            result.statistics.quadricRejectedTopologyEdgeCount = simplify_statistics.rejectedTopologyEdgeCount;
            result.statistics.quadricRejectedFlipEdgeCount = simplify_statistics.rejectedFlipEdgeCount;
            result.statistics.quadricSimplifyPassCount = simplify_statistics.passCount;
            result.statistics.quadricSimplifyReachedTarget = simplify_statistics.reachedTarget;
            result.statistics.quadricSimplifyStoppedByStagnation = simplify_statistics.stoppedByStagnation;
            if (!accept_simplification)
            {
                result.mesh = std::move(unsimplified_mesh);
                result.statistics.quadricSimplifyReachedTarget = false;
            }
            if (postprocessCancelled())
            {
                return false;
            }
        }
        if (options.enableQuadricSimplification && options.enableVoxelFallbackSimplification &&
            DepthTsdfSurfaceBuilder::shouldAttemptVoxelFallbackSimplification(
                result.mesh.faceCount(), options.simplifyTargetFaces, result.statistics.quadricSimplifyReachedTarget))
        {
            reportProgress(QStringLiteral("正在进行边界感知后备简化..."), 90);
            result.statistics.voxelFallbackAttempted = true;
            result.statistics.voxelFallbackPreservedOpenBoundaries = true;
            result.statistics.voxelFallbackInputFaceCount = result.mesh.faceCount();
            const MeshBoundaryTopology topology_before = boundaryTopology(result.mesh);
            const MeshTopologyQualityStatistics topology_quality_before = evaluateMeshTopologyQuality(result.mesh);
            result.statistics.voxelFallbackBoundaryEdgeCountBefore = topology_before.boundaryEdgeCount;
            const TriangleQualitySummary quality_before = triangleQualitySummary(result.mesh);
            result.statistics.voxelFallbackSliverFaceCountBefore = quality_before.sliverFaceCount;
            result.statistics.voxelFallbackSliverRatioBefore = quality_before.sliverRatio;

            TriMesh candidate = result.mesh;
            ReconstructionConfig fallback_config;
            fallback_config.resolution = options.resolution;
            const int fallback_target_faces = std::max(options.simplifyTargetFaces, 120000);
            fallback_config.simplifyTargetFaces = fallback_target_faces;
            fallback_config.voxelSimplifyFactor = options.voxelFallbackInitialClusterFactor;
            fallback_config.minComponentFaces = options.minimumComponentFaces;
            const float maximum_voxel_size =
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            std::vector<std::uint8_t> protected_silhouette_vertices;
            if (options.enableVoxelFallbackMultiViewSilhouetteProtection)
            {
                protected_silhouette_vertices =
                    multiViewSilhouetteBoundaryVertices(candidate,
                                                        frames,
                                                        effective_depth_valid_masks,
                                                        options.voxelFallbackMinimumSilhouetteViews,
                                                        options.voxelFallbackSilhouetteBandPixels,
                                                        options.voxelFallbackSilhouetteDepthToleranceVoxels,
                                                        maximum_voxel_size,
                                                        options.minimumConfidence);
                result.statistics.voxelFallbackProtectedSilhouetteVertexCount = static_cast<int>(std::count(
                    protected_silhouette_vertices.cbegin(), protected_silhouette_vertices.cend(), std::uint8_t{1}));
            }
            const std::vector<std::uint8_t>* boundary_protection =
                result.statistics.voxelFallbackProtectedSilhouetteVertexCount > 0 ? &protected_silhouette_vertices
                                                                                  : nullptr;
            detail::simplifyVoxelMeshAdaptive(&candidate,
                                              fallback_config,
                                              maximum_voxel_size * 0.5f,
                                              true,
                                              options.voxelFallbackMinimumProtectedBoundaryVertices,
                                              options.voxelFallbackMaximumCollapsibleBoundaryDiameterVoxels *
                                                  maximum_voxel_size,
                                              options.voxelFallbackMaximumNormalClusterAngleDegrees,
                                              boundary_protection);
            detail::removeDuplicateFaces(&candidate);
            detail::removeNonManifoldFaces(&candidate);
            detail::removeDegenerateFaces(&candidate, context.minimumDegenerateFaceArea);
            detail::removeSmallConnectedComponents(
                &candidate, std::max(2, options.minimumComponentFaces), options.minimumComponentFaceRatio);
            if (options.fillSmallBoundaryHoles)
            {
                result.statistics.splitPinchedBoundaryVertexCount += detail::splitPinchedBoundaryVertices(&candidate);
                const int faces_before_fallback_fill = candidate.faceCount();
                result.statistics.filledBoundaryHoleCount += detail::fillSmallBoundaryHoles(
                    &candidate,
                    std::max(3, options.maximumHoleBoundaryEdges),
                    std::max(0.0f, options.maximumHoleDiameterVoxels) * maximum_voxel_size);
                result.statistics.addedHoleFillFaceCount +=
                    std::max(0, candidate.faceCount() - faces_before_fallback_fill);
                detail::removeDegenerateFaces(&candidate, context.minimumDegenerateFaceArea);
            }
            detail::compactReferencedVertices(&candidate);

            const int polish_target_faces = fallback_target_faces;
            if (options.enableVoxelFallbackQemPolish && candidate.faceCount() > polish_target_faces)
            {
                result.statistics.voxelFallbackQemPolishAttempted = true;
                result.statistics.voxelFallbackQemPolishInputFaceCount = candidate.faceCount();
                TriMesh unpolished_candidate = candidate;
                const MeshBoundaryTopology polish_topology_before = boundaryTopology(candidate);
                QuadricSimplifyOptions polish_options;
                polish_options.targetFaceCount = polish_target_faces;
                polish_options.maximumPasses = options.simplificationMaximumPasses;
                polish_options.workerCount = options.workerCount;
                polish_options.minimumFaceArea = context.minimumDegenerateFaceArea;
                polish_options.featureAngleDegrees = std::max(options.simplificationFeatureAngleDegrees, 80.0f);
                polish_options.maximumNormalDeviationDegrees =
                    std::max(options.simplificationMaximumNormalDeviationDegrees, 80.0f);
                polish_options.minimumSharpEdgeEndpointDegree =
                    std::max(options.simplificationMinimumSharpEdgeEndpointDegree, 4);
                polish_options.simplifySimpleOpenBoundaries = true;
                polish_options.isCancelled = [control = options.execution]() { return control.isCancelled(); };
                polish_options.progress = [&reportProgress](int pass, int face_count)
                {
                    reportProgress(QStringLiteral("正在抛光后备网格（第 %1 轮，%2 面）...").arg(pass).arg(face_count),
                                   std::min(92, 90 + pass / 8));
                };
                const QuadricSimplifyStatistics polish_statistics = simplifyMeshQuadric(&candidate, polish_options);
                const MeshBoundaryTopology polish_topology_after = boundaryTopology(candidate);
                result.statistics.voxelFallbackQemPolishOutputFaceCount = candidate.faceCount();
                result.statistics.voxelFallbackQemPolishCollapsedEdgeCount = polish_statistics.collapsedEdgeCount;
                const bool accept_polish =
                    DepthTsdfSurfaceBuilder::shouldAcceptQuadricSimplification(
                        result.statistics.voxelFallbackQemPolishInputFaceCount,
                        candidate.faceCount(),
                        polish_topology_before.boundaryEdgeCount,
                        polish_topology_after.boundaryEdgeCount,
                        options.maximumSimplificationBoundaryEdgeGrowthRatio) &&
                    meshVerticesInsidePaddedLayout(candidate, result.layout) &&
                    polish_topology_after.danglingBoundaryVertexCount <=
                        polish_topology_before.danglingBoundaryVertexCount +
                            std::max(8,
                                     static_cast<int>(
                                         std::ceil(polish_topology_before.danglingBoundaryVertexCount * 0.10f))) &&
                    polish_topology_after.nonManifoldEdgeCount <= polish_topology_before.nonManifoldEdgeCount;
                result.statistics.voxelFallbackQemPolishAccepted = accept_polish;
                if (!accept_polish)
                {
                    candidate = std::move(unpolished_candidate);
                }
            }

            const MeshBoundaryTopology topology_after = boundaryTopology(candidate);
            const MeshTopologyQualityStatistics topology_quality_after = evaluateMeshTopologyQuality(candidate);
            const TriangleQualitySummary quality_after = triangleQualitySummary(candidate);
            result.statistics.voxelFallbackOutputFaceCount = candidate.faceCount();
            result.statistics.voxelFallbackBoundaryEdgeCountAfter = topology_after.boundaryEdgeCount;
            result.statistics.voxelFallbackNonManifoldEdgeCountAfter = topology_after.nonManifoldEdgeCount;
            result.statistics.voxelFallbackSliverFaceCountAfter = quality_after.sliverFaceCount;
            result.statistics.voxelFallbackSliverRatioAfter = quality_after.sliverRatio;
            const auto topology_growth_is_safe = [](int before, int after)
            { return after <= before + std::max(8, static_cast<int>(std::ceil(before * 0.10f))); };
            const bool triangle_quality_is_safe =
                quality_after.sliverRatio <= std::max(0.0f, options.voxelFallbackMaximumSliverRatio) ||
                quality_after.sliverRatio <= quality_before.sliverRatio * 0.75;
            result.statistics.voxelFallbackTriangleQualityRejected = !triangle_quality_is_safe;
            const bool accept_fallback =
                !candidate.empty() && candidate.faceCount() < result.mesh.faceCount() &&
                meshVerticesInsidePaddedLayout(candidate, result.layout) && triangle_quality_is_safe &&
                topology_growth_is_safe(topology_before.boundaryEdgeCount, topology_after.boundaryEdgeCount) &&
                topology_growth_is_safe(topology_before.danglingBoundaryVertexCount,
                                        topology_after.danglingBoundaryVertexCount) &&
                topology_growth_is_safe(topology_before.nonManifoldEdgeCount, topology_after.nonManifoldEdgeCount) &&
                topology_quality_after.nonManifoldVertexCount <= topology_quality_before.nonManifoldVertexCount &&
                topology_quality_after.topologicalComplexity <= topology_quality_before.topologicalComplexity;
            result.statistics.voxelFallbackAccepted = accept_fallback;
            if (accept_fallback)
            {
                result.mesh = std::move(candidate);
            }
            if (postprocessCancelled())
            {
                return false;
            }
        }
        // MC33 support filtering can leave multiple boundary fans touching at one
        // vertex even when hole filling is intentionally disabled.  Split those
        // fans before the final connectivity/quality audit; this is a topology
        // cleanup operation and does not synthesize any faces.
        if (options.splitPinchedBoundaryVertices)
        {
            result.statistics.splitPinchedBoundaryVertexCount += detail::splitPinchedBoundaryVertices(&result.mesh);
            detail::removeDegenerateFaces(&result.mesh, context.minimumDegenerateFaceArea);
            detail::compactReferencedVertices(&result.mesh);
        }
        const int faces_before_post_simplification_component_filter = result.mesh.faceCount();
        detail::removeSmallConnectedComponents(
            &result.mesh, std::max(2, options.minimumComponentFaces), options.minimumComponentFaceRatio);
        result.statistics.postSimplificationComponentFilterRemovedFaceCount =
            std::max(0, faces_before_post_simplification_component_filter - result.mesh.faceCount());
        if (result.statistics.postSimplificationComponentFilterRemovedFaceCount > 0)
        {
            detail::compactReferencedVertices(&result.mesh);
        }
        capture_stage(QStringLiteral("03_post_simplification"), result.mesh);
        result.statistics.effectiveSilhouetteAwareFinalHoleFill = options.enableSilhouetteAwareFinalHoleFill;
        result.statistics.effectiveVisibilityConstrainedFinalHoleFill =
            options.enableVisibilityConstrainedFinalHoleFill;
        if (options.enableSilhouetteAwareFinalHoleFill && options.fillSmallBoundaryHoles)
        {
            reportProgress(QStringLiteral("正在按多视图轮廓修补最终网格孔洞..."), 92);
            result.statistics.finalHoleFillAttempted = true;
            if (options.splitPinchedBoundaryVertices)
            {
                result.statistics.splitPinchedBoundaryVertexCount += detail::splitPinchedBoundaryVertices(&result.mesh);
            }
            const float maximum_voxel_size =
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            std::vector<std::uint8_t> protected_silhouette_vertices =
                multiViewSilhouetteBoundaryVertices(result.mesh,
                                                    frames,
                                                    effective_depth_valid_masks,
                                                    options.voxelFallbackMinimumSilhouetteViews,
                                                    options.voxelFallbackSilhouetteBandPixels,
                                                    options.voxelFallbackSilhouetteDepthToleranceVoxels,
                                                    maximum_voxel_size,
                                                    options.minimumConfidence);
            if (options.enableVisibilityConstrainedFinalHoleFill)
            {
                const VisibilityHoleProtectionResult visibility =
                    visibilityConstrainedHoleProtection(result.mesh,
                                                        frames,
                                                        effective_depth_valid_masks,
                                                        protected_silhouette_vertices,
                                                        std::max(3, options.finalHoleFillMaximumBoundaryEdges),
                                                        options.visibilityHoleFillMinimumSupportingViews,
                                                        options.visibilityHoleFillMaximumConflictViews,
                                                        options.visibilityHoleFillDepthToleranceVoxels,
                                                        options.visibilityHoleFillStrongSilhouetteRatio,
                                                        maximum_voxel_size,
                                                        options.minimumConfidence);
                protected_silhouette_vertices = visibility.protectedVertices;
                result.statistics.visibilityHoleFillConsideredLoopCount = visibility.consideredLoopCount;
                result.statistics.visibilityHoleFillReleasedLoopCount = visibility.releasedLoopCount;
                result.statistics.visibilityHoleFillRejectedSupportLoopCount = visibility.rejectedSupportLoopCount;
                result.statistics.visibilityHoleFillRejectedConflictLoopCount = visibility.rejectedConflictLoopCount;
            }
            result.statistics.finalHoleFillProtectedSilhouetteVertexCount = static_cast<int>(std::count(
                protected_silhouette_vertices.cbegin(), protected_silhouette_vertices.cend(), std::uint8_t{1}));

            const MeshBoundaryTopology topology_before = boundaryTopology(result.mesh);
            const TriangleQualitySummary quality_before = triangleQualitySummary(result.mesh);
            result.statistics.finalHoleFillBoundaryEdgeCountBefore = topology_before.boundaryEdgeCount;
            result.statistics.finalHoleFillNonManifoldEdgeCountBefore = topology_before.nonManifoldEdgeCount;
            result.statistics.finalHoleFillSliverRatioBefore = quality_before.sliverRatio;

            if (result.statistics.finalHoleFillProtectedSilhouetteVertexCount > 0 ||
                options.enableVisibilityConstrainedFinalHoleFill)
            {
                TriMesh candidate = result.mesh;
                const std::size_t vertex_count_before = candidate.vertices.size();
                const int face_count_before = candidate.faceCount();
                int protected_hole_count = 0;
                const int filled_hole_count = detail::fillSmallBoundaryHoles(
                    &candidate,
                    std::max(3, options.finalHoleFillMaximumBoundaryEdges),
                    std::max(0.0f, options.finalHoleFillMaximumDiameterVoxels) * maximum_voxel_size,
                    &protected_silhouette_vertices,
                    &protected_hole_count,
                    true);
                HoleFillPatchEvidenceResult patch_evidence;
                if (options.enableVisibilityConstrainedFinalHoleFill && filled_hole_count > 0)
                {
                    patch_evidence =
                        validateAddedHoleFillPatchEvidence(candidate,
                                                           vertex_count_before,
                                                           static_cast<std::size_t>(face_count_before),
                                                           frames,
                                                           effective_depth_valid_masks,
                                                           options.visibilityHoleFillMinimumSupportingViews,
                                                           options.visibilityHoleFillMaximumConflictViews,
                                                           options.visibilityHoleFillDepthToleranceVoxels,
                                                           maximum_voxel_size,
                                                           options.minimumConfidence);
                    result.statistics.finalHoleFillPatchEvidenceValidationAttempted = patch_evidence.attempted;
                    result.statistics.finalHoleFillPatchEvidenceValidationAccepted = patch_evidence.accepted;
                    result.statistics.finalHoleFillPatchEvidenceVertexSampleCount = patch_evidence.vertexSampleCount;
                    result.statistics.finalHoleFillPatchEvidenceFaceCenterSampleCount =
                        patch_evidence.faceCenterSampleCount;
                    result.statistics.finalHoleFillPatchEvidenceAcceptedSampleCount =
                        patch_evidence.acceptedSampleCount;
                    result.statistics.finalHoleFillPatchEvidenceRejectedSupportSampleCount =
                        patch_evidence.rejectedSupportSampleCount;
                    result.statistics.finalHoleFillPatchEvidenceRejectedConflictSampleCount =
                        patch_evidence.rejectedConflictSampleCount;
                    result.statistics.finalHoleFillPatchEvidenceMinimumSupportingViewCount =
                        patch_evidence.minimumSupportingViewCount;
                    result.statistics.finalHoleFillPatchEvidenceMaximumConflictViewCount =
                        patch_evidence.maximumConflictViewCount;
                }
                detail::removeDegenerateFaces(&candidate, context.minimumDegenerateFaceArea);
                detail::compactReferencedVertices(&candidate);

                const MeshBoundaryTopology topology_after = boundaryTopology(candidate);
                const TriangleQualitySummary quality_after = triangleQualitySummary(candidate);
                const int added_face_count = std::max(0, candidate.faceCount() - face_count_before);
                const int maximum_added_faces =
                    std::max(8,
                             static_cast<int>(std::ceil(face_count_before *
                                                        std::max(0.0f, options.finalHoleFillMaximumFaceGrowthRatio))));
                const bool triangle_quality_is_safe =
                    quality_after.sliverRatio <= std::max(0.0f, options.finalHoleFillMaximumSliverRatio) ||
                    quality_after.sliverRatio <= quality_before.sliverRatio;
                const bool patch_evidence_is_safe = !options.enableVisibilityConstrainedFinalHoleFill ||
                                                    (patch_evidence.attempted && patch_evidence.accepted);
                const bool accept_final_fill =
                    filled_hole_count > 0 && patch_evidence_is_safe &&
                    meshVerticesInsidePaddedLayout(candidate, result.layout) &&
                    topology_after.boundaryEdgeCount < topology_before.boundaryEdgeCount &&
                    topology_after.danglingBoundaryVertexCount <= topology_before.danglingBoundaryVertexCount &&
                    topology_after.nonManifoldEdgeCount <= topology_before.nonManifoldEdgeCount &&
                    added_face_count <= maximum_added_faces && triangle_quality_is_safe;

                if (accept_final_fill && !result.statistics.topologySafeSimplificationAccepted &&
                    options.enableQuadricSimplification && options.simplifyTargetFaces > 0 &&
                    candidate.faceCount() > options.simplifyTargetFaces)
                {
                    result.statistics.finalHoleFillPostSimplificationAttempted = true;
                    result.statistics.finalHoleFillPostSimplificationInputFaceCount = candidate.faceCount();
                    result.statistics.finalHoleFillPostSimplificationBoundaryEdgeCountBefore =
                        topology_after.boundaryEdgeCount;

                    TriMesh filled_candidate = candidate;
                    QuadricSimplifyOptions post_fill_options;
                    post_fill_options.targetFaceCount = options.simplifyTargetFaces;
                    post_fill_options.maximumPasses = options.simplificationMaximumPasses;
                    post_fill_options.workerCount = options.workerCount;
                    post_fill_options.minimumFaceArea = context.minimumDegenerateFaceArea;
                    post_fill_options.featureAngleDegrees = options.simplificationFeatureAngleDegrees;
                    post_fill_options.maximumNormalDeviationDegrees =
                        options.simplificationMaximumNormalDeviationDegrees;
                    post_fill_options.minimumSharpEdgeEndpointDegree =
                        options.simplificationMinimumSharpEdgeEndpointDegree;
                    post_fill_options.simplifySimpleOpenBoundaries = true;
                    post_fill_options.isCancelled = [control = options.execution]() { return control.isCancelled(); };
                    post_fill_options.progress = [&reportProgress](int pass, int face_count)
                    {
                        reportProgress(
                            QStringLiteral("正在整理补洞网格（第 %1 轮，%2 面）...").arg(pass).arg(face_count),
                            std::min(96, 94 + pass / 8));
                    };
                    const QuadricSimplifyStatistics post_fill_simplification =
                        simplifyMeshQuadric(&candidate, post_fill_options);
                    const MeshBoundaryTopology post_fill_topology = boundaryTopology(candidate);
                    const TriangleQualitySummary post_fill_quality = triangleQualitySummary(candidate);
                    const int permitted_boundary_edges =
                        topology_after.boundaryEdgeCount +
                        std::max(8, static_cast<int>(std::ceil(topology_after.boundaryEdgeCount * 0.10f)));
                    const bool accept_post_fill_simplification =
                        !candidate.empty() && candidate.faceCount() < filled_candidate.faceCount() &&
                        meshVerticesInsidePaddedLayout(candidate, result.layout) &&
                        post_fill_topology.boundaryEdgeCount <= permitted_boundary_edges &&
                        post_fill_topology.danglingBoundaryVertexCount <= topology_after.danglingBoundaryVertexCount &&
                        post_fill_topology.nonManifoldEdgeCount <= topology_after.nonManifoldEdgeCount &&
                        post_fill_quality.sliverRatio <=
                            std::max(static_cast<double>(options.finalHoleFillMaximumSliverRatio),
                                     quality_after.sliverRatio + 1.0e-12);
                    result.statistics.finalHoleFillPostSimplificationAccepted = accept_post_fill_simplification;
                    result.statistics.finalHoleFillPostSimplificationCollapsedEdgeCount =
                        post_fill_simplification.collapsedEdgeCount;
                    result.statistics.finalHoleFillPostSimplificationBoundaryEdgeCountAfter =
                        post_fill_topology.boundaryEdgeCount;
                    if (!accept_post_fill_simplification)
                    {
                        candidate = std::move(filled_candidate);
                        result.statistics.finalHoleFillPostSimplificationBoundaryEdgeCountAfter =
                            topology_after.boundaryEdgeCount;
                    }
                    result.statistics.finalHoleFillPostSimplificationOutputFaceCount = candidate.faceCount();
                }

                result.statistics.finalHoleFillProtectedHoleCount = protected_hole_count;
                result.statistics.finalHoleFillFilledHoleCount = filled_hole_count;
                result.statistics.finalHoleFillAddedFaceCount = added_face_count;
                result.statistics.finalHoleFillBoundaryEdgeCountAfter = topology_after.boundaryEdgeCount;
                result.statistics.finalHoleFillNonManifoldEdgeCountAfter = topology_after.nonManifoldEdgeCount;
                result.statistics.finalHoleFillSliverRatioAfter = quality_after.sliverRatio;
                result.statistics.finalHoleFillTriangleQualityRejected = !triangle_quality_is_safe;
                result.statistics.finalHoleFillAccepted = accept_final_fill;
                if (accept_final_fill)
                {
                    result.mesh = std::move(candidate);
                    result.statistics.filledBoundaryHoleCount += filled_hole_count;
                    result.statistics.addedHoleFillFaceCount += added_face_count;
                }
            }
            else
            {
                result.statistics.finalHoleFillBoundaryEdgeCountAfter = topology_before.boundaryEdgeCount;
                result.statistics.finalHoleFillNonManifoldEdgeCountAfter = topology_before.nonManifoldEdgeCount;
                result.statistics.finalHoleFillSliverRatioAfter = quality_before.sliverRatio;
            }
            if (postprocessCancelled())
            {
                return false;
            }
        }
        result.statistics.effectiveTinyBoundaryLoopCollapse = options.enableTinyBoundaryLoopCollapse;
        if (options.enableTinyBoundaryLoopCollapse)
        {
            reportProgress(QStringLiteral("正在收缩退化微孔边界..."), 92);
            result.statistics.tinyBoundaryLoopCollapseAttempted = true;
            MeshTopologyQualityThresholds quality_thresholds;
            quality_thresholds.maximumBoundaryEdgeRatio =
                std::max(0.0f, options.topologyQualityMaximumBoundaryEdgeRatio);
            quality_thresholds.maximumHighAspectFaceRatio =
                std::max(0.0f, options.topologyQualityMaximumHighAspectFaceRatio);
            quality_thresholds.maximumExtremeAspectFaceRatio =
                std::max(0.0f, options.topologyQualityMaximumExtremeAspectFaceRatio);
            MeshTopologyQualityStatistics current_quality =
                evaluateMeshTopologyQuality(result.mesh, quality_thresholds);
            result.statistics.tinyBoundaryLoopBoundaryEdgeCountBefore = current_quality.boundaryEdgeCount;
            result.statistics.tinyBoundaryLoopNonManifoldEdgeCountBefore = current_quality.nonManifoldEdgeCount;
            result.statistics.tinyBoundaryLoopHighAspectRatioBefore = current_quality.highAspectFaceRatio;

            const float maximum_voxel_size =
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            TriMesh accepted_mesh = result.mesh;
            constexpr double quality_epsilon = 1.0e-12;
            for (int pass = 0; pass < options.tinyBoundaryLoopCollapseMaximumPasses; ++pass)
            {
                TriMesh candidate = accepted_mesh;
                const int collapsed_edge_count = detail::collapseTinyBoundaryLoops(
                    &candidate,
                    std::max(3, options.tinyBoundaryLoopCollapseMaximumEdges),
                    std::max(0.0f, options.tinyBoundaryLoopCollapseMaximumDiameterVoxels) * maximum_voxel_size,
                    std::max(0.0f, options.tinyBoundaryLoopCollapseMaximumEdgeVoxels) * maximum_voxel_size);
                if (collapsed_edge_count <= 0)
                {
                    break;
                }

                const MeshTopologyQualityStatistics candidate_quality =
                    evaluateMeshTopologyQuality(candidate, quality_thresholds);
                const bool topology_is_safe =
                    candidate_quality.boundaryEdgeCount < current_quality.boundaryEdgeCount &&
                    candidate_quality.nonManifoldEdgeCount <= current_quality.nonManifoldEdgeCount &&
                    candidate_quality.componentCount <= current_quality.componentCount &&
                    candidate_quality.largestComponentFaceRatio + quality_epsilon >=
                        current_quality.largestComponentFaceRatio;
                const bool triangle_quality_is_safe =
                    candidate_quality.highAspectFaceRatio <= current_quality.highAspectFaceRatio + quality_epsilon &&
                    candidate_quality.extremeAspectFaceRatio <=
                        current_quality.extremeAspectFaceRatio + quality_epsilon;
                if (!topology_is_safe || !triangle_quality_is_safe)
                {
                    break;
                }

                accepted_mesh = std::move(candidate);
                current_quality = candidate_quality;
                ++result.statistics.tinyBoundaryLoopCollapsePassCount;
                result.statistics.tinyBoundaryLoopCollapsedEdgeCount += collapsed_edge_count;
            }
            result.statistics.tinyBoundaryLoopCollapseAccepted =
                result.statistics.tinyBoundaryLoopCollapsePassCount > 0;
            if (result.statistics.tinyBoundaryLoopCollapseAccepted)
            {
                result.mesh = std::move(accepted_mesh);
            }
            result.statistics.tinyBoundaryLoopBoundaryEdgeCountAfter = current_quality.boundaryEdgeCount;
            result.statistics.tinyBoundaryLoopNonManifoldEdgeCountAfter = current_quality.nonManifoldEdgeCount;
            result.statistics.tinyBoundaryLoopHighAspectRatioAfter = current_quality.highAspectFaceRatio;
            if (postprocessCancelled())
            {
                return false;
            }
        }
        result.statistics.effectiveTriangleQualityOptimization = options.enableTriangleQualityOptimization;
        if (options.enableTriangleQualityOptimization)
        {
            reportProgress(QStringLiteral("正在优化三角网格质量..."), 92);
            result.statistics.triangleQualityOptimizationAttempted = true;
            result.statistics.triangleQualityOptimizationInputFaceCount = result.mesh.faceCount();
            MeshTopologyQualityThresholds quality_thresholds;
            quality_thresholds.maximumBoundaryEdgeRatio =
                std::max(0.0f, options.topologyQualityMaximumBoundaryEdgeRatio);
            quality_thresholds.maximumHighAspectFaceRatio =
                std::max(0.0f, options.topologyQualityMaximumHighAspectFaceRatio);
            quality_thresholds.maximumExtremeAspectFaceRatio =
                std::max(0.0f, options.topologyQualityMaximumExtremeAspectFaceRatio);
            const MeshTopologyQualityStatistics quality_before =
                evaluateMeshTopologyQuality(result.mesh, quality_thresholds);
            result.statistics.triangleQualityHighAspectFaceRatioBefore = quality_before.highAspectFaceRatio;
            result.statistics.triangleQualityExtremeAspectFaceRatioBefore = quality_before.extremeAspectFaceRatio;

            TriMesh candidate = result.mesh;
            MeshTriangleOptimizationOptions optimization_options;
            optimization_options.maximumPasses = options.triangleQualityOptimizationMaximumPasses;
            optimization_options.minimumWorstAspectImprovementRatio =
                options.triangleQualityMinimumAspectImprovementRatio;
            optimization_options.maximumFeatureAngleDegrees = options.triangleQualityMaximumFeatureAngleDegrees;
            optimization_options.maximumNormalDeviationDegrees = options.triangleQualityMaximumNormalDeviationDegrees;
            optimization_options.enableTangentialRelaxation = options.enableTriangleQualityTangentialRelaxation;
            optimization_options.tangentialRelaxationPasses = options.triangleQualityTangentialRelaxationPasses;
            optimization_options.tangentialRelaxationLambda = options.triangleQualityTangentialRelaxationLambda;
            optimization_options.tangentialMaximumDisplacementEdgeRatio =
                options.triangleQualityTangentialMaximumDisplacementEdgeRatio;
            optimization_options.enableIsotropicRemeshing = options.enableTriangleQualityIsotropicRemeshing;
            optimization_options.isotropicRemeshingPasses = options.triangleQualityIsotropicRemeshingPasses;
            optimization_options.isotropicShortEdgeRatio = options.triangleQualityIsotropicShortEdgeRatio;
            optimization_options.isotropicLongEdgeRatio = options.triangleQualityIsotropicLongEdgeRatio;
            optimization_options.isotropicMaximumFaceGrowthRatio =
                options.triangleQualityIsotropicMaximumFaceGrowthRatio;
            optimization_options.isCancelled = [control = options.execution]() { return control.isCancelled(); };
            const MeshTriangleOptimizationStatistics optimization =
                optimizeTriangleQuality(&candidate, optimization_options);
            const MeshTopologyQualityStatistics quality_after =
                evaluateMeshTopologyQuality(candidate, quality_thresholds);
            result.statistics.triangleQualityOptimizationPassCount = optimization.passCount;
            result.statistics.triangleQualityOptimizationFlippedEdgeCount = optimization.flippedEdgeCount;
            result.statistics.triangleQualityTangentialRelaxationPassCount = optimization.tangentialRelaxationPassCount;
            result.statistics.triangleQualityTangentialRelaxedVertexCount = optimization.tangentialRelaxedVertexCount;
            result.statistics.triangleQualityIsotropicRemeshingPassCount = optimization.isotropicRemeshingPassCount;
            result.statistics.triangleQualityIsotropicCollapsedEdgeCount = optimization.isotropicCollapsedEdgeCount;
            result.statistics.triangleQualityIsotropicSplitEdgeCount = optimization.isotropicSplitEdgeCount;
            result.statistics.triangleQualityOptimizationOutputFaceCount = candidate.faceCount();
            result.statistics.triangleQualityHighAspectFaceRatioAfter = quality_after.highAspectFaceRatio;
            result.statistics.triangleQualityExtremeAspectFaceRatioAfter = quality_after.extremeAspectFaceRatio;

            constexpr double ratio_epsilon = 1.0e-12;
            const bool topology_preserved =
                quality_after.boundaryEdgeCount == quality_before.boundaryEdgeCount &&
                quality_after.nonManifoldEdgeCount == quality_before.nonManifoldEdgeCount &&
                quality_after.componentCount == quality_before.componentCount &&
                quality_after.largestComponentFaceRatio + ratio_epsilon >= quality_before.largestComponentFaceRatio;
            const bool triangle_quality_improved =
                quality_after.highAspectFaceRatio <= quality_before.highAspectFaceRatio + ratio_epsilon &&
                quality_after.extremeAspectFaceRatio <= quality_before.extremeAspectFaceRatio + ratio_epsilon &&
                quality_after.skinnyFaceRatio <= quality_before.skinnyFaceRatio + ratio_epsilon &&
                (quality_after.highAspectFaceRatio + ratio_epsilon < quality_before.highAspectFaceRatio ||
                 quality_after.extremeAspectFaceRatio + ratio_epsilon < quality_before.extremeAspectFaceRatio ||
                 quality_after.skinnyFaceRatio + ratio_epsilon < quality_before.skinnyFaceRatio);
            const bool accept_optimization =
                !optimization.cancelled &&
                (optimization.flippedEdgeCount > 0 || optimization.tangentialRelaxedVertexCount > 0 ||
                 optimization.isotropicCollapsedEdgeCount > 0 || optimization.isotropicSplitEdgeCount > 0) &&
                topology_preserved && triangle_quality_improved;
            result.statistics.triangleQualityOptimizationAccepted = accept_optimization;
            if (accept_optimization)
            {
                result.mesh = std::move(candidate);
            }
            if (postprocessCancelled())
            {
                return false;
            }
        }
        capture_stage(QStringLiteral("04_triangle_quality"), result.mesh);
        if (options.enableSilhouetteAwareFinalHoleFill && options.fillSmallBoundaryHoles)
        {
            reportProgress(QStringLiteral("正在修补简化后残余微孔..."), 92);
            result.statistics.residualMicroHoleFillAttempted = true;
            const float maximum_voxel_size =
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            const int maximum_boundary_edges = std::min(16, std::max(3, options.finalHoleFillMaximumBoundaryEdges));
            std::vector<std::uint8_t> protected_silhouette_vertices =
                multiViewSilhouetteBoundaryVertices(result.mesh,
                                                    frames,
                                                    effective_depth_valid_masks,
                                                    options.voxelFallbackMinimumSilhouetteViews,
                                                    options.voxelFallbackSilhouetteBandPixels,
                                                    options.voxelFallbackSilhouetteDepthToleranceVoxels,
                                                    maximum_voxel_size,
                                                    options.minimumConfidence);
            if (options.enableVisibilityConstrainedFinalHoleFill)
            {
                protected_silhouette_vertices =
                    visibilityConstrainedHoleProtection(result.mesh,
                                                        frames,
                                                        effective_depth_valid_masks,
                                                        protected_silhouette_vertices,
                                                        maximum_boundary_edges,
                                                        options.visibilityHoleFillMinimumSupportingViews,
                                                        options.visibilityHoleFillMaximumConflictViews,
                                                        options.visibilityHoleFillDepthToleranceVoxels,
                                                        options.visibilityHoleFillStrongSilhouetteRatio,
                                                        maximum_voxel_size,
                                                        options.minimumConfidence)
                        .protectedVertices;
            }

            const MeshBoundaryTopology topology_before = boundaryTopology(result.mesh);
            const TriangleQualitySummary quality_before = triangleQualitySummary(result.mesh);
            result.statistics.residualMicroHoleFillBoundaryEdgeCountBefore = topology_before.boundaryEdgeCount;
            result.statistics.residualMicroHoleFillNonManifoldEdgeCountBefore = topology_before.nonManifoldEdgeCount;
            result.statistics.residualMicroHoleFillSliverRatioBefore = quality_before.sliverRatio;

            TriMesh candidate = result.mesh;
            const int face_count_before = candidate.faceCount();
            int protected_hole_count = 0;
            const int filled_hole_count = detail::fillSmallBoundaryHoles(
                &candidate,
                maximum_boundary_edges,
                std::min(4.0f, std::max(0.0f, options.finalHoleFillMaximumDiameterVoxels)) * maximum_voxel_size,
                &protected_silhouette_vertices,
                &protected_hole_count,
                true);
            detail::removeDegenerateFaces(&candidate, context.minimumDegenerateFaceArea);
            detail::compactReferencedVertices(&candidate);

            const MeshBoundaryTopology topology_after = boundaryTopology(candidate);
            const TriangleQualitySummary quality_after = triangleQualitySummary(candidate);
            const int added_face_count = std::max(0, candidate.faceCount() - face_count_before);
            const int maximum_added_faces = std::max(8, static_cast<int>(std::ceil(face_count_before * 0.05f)));
            const bool triangle_quality_is_safe =
                quality_after.sliverRatio <= std::max(static_cast<double>(options.finalHoleFillMaximumSliverRatio),
                                                      quality_before.sliverRatio + 1.0e-12);
            const bool accept_residual_fill =
                filled_hole_count > 0 && meshVerticesInsidePaddedLayout(candidate, result.layout) &&
                topology_after.boundaryEdgeCount < topology_before.boundaryEdgeCount &&
                topology_after.danglingBoundaryVertexCount <= topology_before.danglingBoundaryVertexCount &&
                topology_after.nonManifoldEdgeCount <= topology_before.nonManifoldEdgeCount &&
                added_face_count <= maximum_added_faces && triangle_quality_is_safe;
            result.statistics.residualMicroHoleFillProtectedHoleCount = protected_hole_count;
            result.statistics.residualMicroHoleFillFilledHoleCount = filled_hole_count;
            result.statistics.residualMicroHoleFillAddedFaceCount = added_face_count;
            result.statistics.residualMicroHoleFillBoundaryEdgeCountAfter = topology_after.boundaryEdgeCount;
            result.statistics.residualMicroHoleFillNonManifoldEdgeCountAfter = topology_after.nonManifoldEdgeCount;
            result.statistics.residualMicroHoleFillSliverRatioAfter = quality_after.sliverRatio;
            result.statistics.residualMicroHoleFillAccepted = accept_residual_fill;
            if (accept_residual_fill)
            {
                result.mesh = std::move(candidate);
                result.statistics.filledBoundaryHoleCount += filled_hole_count;
                result.statistics.addedHoleFillFaceCount += added_face_count;
            }
            if (postprocessCancelled())
            {
                return false;
            }
        }
        reportProgress(QStringLiteral("正在统一最终网格面方向..."), 93);
        result.statistics.finalFaceOrientationRepairAttempted = true;
        const auto final_orientation = repairFaceOrientationSafely(&result.mesh);
        result.statistics.finalFaceOrientationInconsistentSharedEdgeCountBefore =
            final_orientation.first.inconsistentSharedEdgeCountBefore;
        result.statistics.finalFaceOrientationInconsistentSharedEdgeCountAfter =
            final_orientation.first.inconsistentSharedEdgeCountAfter;
        result.statistics.finalFaceOrientationFlippedFaceCount = final_orientation.first.flippedFaceCount;
        result.statistics.finalFaceOrientationRemovedFaceCount = final_orientation.first.removedContradictoryFaceCount;
        result.statistics.finalFaceOrientationNonManifoldEdgeCount = final_orientation.first.nonManifoldEdgeCount;
        result.statistics.finalFaceOrientationRepairAccepted = final_orientation.second;
        if (postprocessCancelled())
        {
            return false;
        }
        if (result.statistics.effectivePostSimplificationSurfaceDenoising)
        {
            reportProgress(QStringLiteral("正在进行轮廓保护的最终表面降噪..."), 93);
            result.statistics.postSimplificationSurfaceDenoisingAttempted = true;
            MeshTopologyQualityThresholds quality_thresholds;
            quality_thresholds.maximumBoundaryEdgeRatio =
                std::max(0.0f, options.topologyQualityMaximumBoundaryEdgeRatio);
            quality_thresholds.maximumHighAspectFaceRatio =
                std::max(0.0f, options.topologyQualityMaximumHighAspectFaceRatio);
            quality_thresholds.maximumExtremeAspectFaceRatio =
                std::max(0.0f, options.topologyQualityMaximumExtremeAspectFaceRatio);
            const MeshTopologyQualityStatistics quality_before =
                evaluateMeshTopologyQuality(result.mesh, quality_thresholds);
            result.statistics.postSimplificationNormalAngleMedianBefore =
                quality_before.adjacentNormalAngleMedianDegrees;
            result.statistics.postSimplificationNormalAngleP90Before = quality_before.adjacentNormalAngleP90Degrees;
            result.statistics.postSimplificationNormalAngleOver30RatioBefore =
                quality_before.adjacentNormalAngleOver30Ratio;
            result.statistics.postSimplificationHighAspectFaceRatioBefore = quality_before.highAspectFaceRatio;
            result.statistics.postSimplificationExtremeAspectFaceRatioBefore = quality_before.extremeAspectFaceRatio;

            TriMesh candidate = result.mesh;
            const float maximum_voxel_size =
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            const int moved_vertex_count =
                options.enableProtectedTaubinSurfaceDenoising
                    ? detail::smoothSurfaceVerticesTaubinProtected(&candidate,
                                                                   options.surfaceDenoisingIterations,
                                                                   options.surfaceDenoisingLambda,
                                                                   options.surfaceDenoisingMu,
                                                                   options.maximumSurfaceDenoisingDisplacementVoxels *
                                                                       maximum_voxel_size,
                                                                   options.maximumSurfaceDenoisingNormalAngleDegrees,
                                                                   options.surfaceDenoisingBoundaryProtectionRings)
                    : detail::smoothSurfaceVerticesNormalAware(&candidate,
                                                               options.surfaceDenoisingIterations,
                                                               options.surfaceDenoisingLambda,
                                                               options.maximumSurfaceDenoisingDisplacementVoxels *
                                                                   maximum_voxel_size,
                                                               options.maximumSurfaceDenoisingNormalAngleDegrees,
                                                               options.surfaceDenoisingBoundaryProtectionRings);
            MeshTriangleOptimizationOptions relaxation_options;
            relaxation_options.maximumPasses = 0;
            relaxation_options.maximumFeatureAngleDegrees = options.triangleQualityMaximumFeatureAngleDegrees;
            relaxation_options.maximumNormalDeviationDegrees = options.triangleQualityMaximumNormalDeviationDegrees;
            relaxation_options.enableTangentialRelaxation = true;
            relaxation_options.tangentialRelaxationPasses = 1;
            relaxation_options.tangentialRelaxationLambda =
                std::min(0.30f, options.triangleQualityTangentialRelaxationLambda);
            relaxation_options.tangentialMaximumDisplacementEdgeRatio =
                std::min(0.10f, options.triangleQualityTangentialMaximumDisplacementEdgeRatio);
            relaxation_options.enableIsotropicRemeshing = false;
            relaxation_options.isCancelled = [control = options.execution]() { return control.isCancelled(); };
            const MeshTriangleOptimizationStatistics relaxation =
                optimizeTriangleQuality(&candidate, relaxation_options);
            result.statistics.postSimplificationTangentialRelaxedVertexCount = relaxation.tangentialRelaxedVertexCount;
            detail::recomputeNormals(&candidate);
            const MeshTopologyQualityStatistics quality_after =
                evaluateMeshTopologyQuality(candidate, quality_thresholds);
            result.statistics.postSimplificationSmoothedSurfaceVertexCount = moved_vertex_count;
            result.statistics.postSimplificationNormalAngleMedianAfter = quality_after.adjacentNormalAngleMedianDegrees;
            result.statistics.postSimplificationNormalAngleP90After = quality_after.adjacentNormalAngleP90Degrees;
            result.statistics.postSimplificationNormalAngleOver30RatioAfter =
                quality_after.adjacentNormalAngleOver30Ratio;
            result.statistics.postSimplificationHighAspectFaceRatioAfter = quality_after.highAspectFaceRatio;
            result.statistics.postSimplificationExtremeAspectFaceRatioAfter = quality_after.extremeAspectFaceRatio;

            constexpr double ratio_epsilon = 1.0e-12;
            const bool topology_preserved =
                quality_after.validFaceCount == quality_before.validFaceCount &&
                quality_after.boundaryEdgeCount == quality_before.boundaryEdgeCount &&
                quality_after.nonManifoldEdgeCount == quality_before.nonManifoldEdgeCount &&
                quality_after.componentCount == quality_before.componentCount &&
                quality_after.largestComponentFaceRatio + ratio_epsilon >= quality_before.largestComponentFaceRatio;
            const bool triangle_quality_is_safe =
                quality_after.highAspectFaceRatio <=
                    std::max(quality_before.highAspectFaceRatio * 1.02, quality_before.highAspectFaceRatio + 0.002) &&
                quality_after.extremeAspectFaceRatio <= std::max(quality_before.extremeAspectFaceRatio * 1.02,
                                                                 quality_before.extremeAspectFaceRatio + 0.001);
            const bool normal_quality_is_safe =
                quality_after.adjacentNormalAngleMedianDegrees <=
                    quality_before.adjacentNormalAngleMedianDegrees + 0.25 &&
                quality_after.adjacentNormalAngleP90Degrees <= quality_before.adjacentNormalAngleP90Degrees + 1.0 &&
                quality_after.adjacentNormalAngleOver30Ratio <= quality_before.adjacentNormalAngleOver30Ratio + 0.0025;
            const bool normal_quality_improved =
                quality_after.adjacentNormalAngleMedianDegrees + 0.10 <
                    quality_before.adjacentNormalAngleMedianDegrees ||
                quality_after.adjacentNormalAngleP90Degrees + 0.50 < quality_before.adjacentNormalAngleP90Degrees ||
                quality_after.adjacentNormalAngleOver30Ratio + 0.001 < quality_before.adjacentNormalAngleOver30Ratio;
            const bool accept_denoising =
                moved_vertex_count > 0 && meshVerticesInsidePaddedLayout(candidate, result.layout) &&
                topology_preserved && triangle_quality_is_safe && normal_quality_is_safe && normal_quality_improved;
            result.statistics.postSimplificationSurfaceDenoisingAccepted = accept_denoising;
            if (accept_denoising)
            {
                result.mesh = std::move(candidate);
            }
            if (postprocessCancelled())
            {
                return false;
            }
        }
        result.statistics.meshSimplificationElapsedMs = elapsedMilliseconds(simplification_start);
        result.statistics.boundaryEdgeCountAfter = boundaryEdgeCount(result.mesh);
        result.statistics.postSimplificationFaceCount = result.mesh.faceCount();
        reportProgress(QStringLiteral("网格简化完成，正在重算法线..."), 93);
        detail::recomputeNormals(&result.mesh);
        capture_stage(QStringLiteral("05_builder_final"), result.mesh);
        if (result.mesh.empty() || result.mesh.faceCount() < 2)
        {
            result.errorMessage = QStringLiteral("TSDF cleanup produced an unusable mesh (%1 vertices, %2 faces)")
                                      .arg(result.mesh.vertexCount())
                                      .arg(result.mesh.faceCount());
            return false;
        }
        if (postprocessCancelled())
        {
            return false;
        }

        return true;
    }
} // namespace xjw::mesh::tsdf_detail
