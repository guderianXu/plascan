#include "DepthTsdfStages.h"
namespace xjw::mesh::tsdf_detail
{
    bool extractTsdfIsoSurface(const QVector<DepthTsdfFrame>& frames,
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

        const PostprocessClock::time_point marching_cubes_start = PostprocessClock::now();
        const bool use_native_carrier =
            options.visibilityOccupancyNativeCarrierExtraction && !native_carrier_field.empty() &&
            (options.enableMc33IsoSurfaceExtraction || options.enableConsistentIsoSurfaceExtraction);
        const std::array<float, 3>& extraction_bounds_min =
            use_native_carrier ? native_carrier_bounds_min : result.layout.boundsMin;
        const std::array<float, 3>& extraction_bounds_max =
            use_native_carrier ? native_carrier_bounds_max : result.layout.boundsMax;
        const std::array<int, 3>& extraction_cells = use_native_carrier ? native_carrier_cells : result.layout.cells;
        const std::vector<float>& extraction_tsdf =
            use_native_carrier ? native_carrier_field
                               : (!visual_hull_completion_tsdf.empty() ? visual_hull_completion_tsdf : tsdf);
        const std::vector<std::uint8_t> no_native_support;
        const std::vector<std::uint8_t>& extraction_support =
            use_native_carrier
                ? no_native_support
                : (!visual_hull_completion_support.empty()
                       ? visual_hull_completion_support
                       : (result.statistics.effectiveVisibilityOccupancyCompletion
                              ? supported
                              : (!adaptiveTgvExtractionSupport.empty() ? adaptiveTgvExtractionSupport : supported)));
        try
        {
            if (result.statistics.effectiveVisibilityOccupancyCellBoundaryExtraction)
            {
                VisibilityOccupancyBoundaryOptions extraction_options;
                extraction_options.isCancelled = [control = options.execution]() { return control.isCancelled(); };
                VisibilityOccupancyBoundaryResult extraction =
                    VisibilityOccupancyBoundaryExtractor::extract(native_carrier_bounds_min,
                                                                  native_carrier_bounds_max,
                                                                  native_carrier_dimensions,
                                                                  native_carrier_occupied,
                                                                  extraction_options);
                if (!extraction.ok)
                {
                    result.errorMessage = QStringLiteral("可见性占据体边界提取失败: %1")
                                              .arg(QString::fromStdString(extraction.errorMessage));
                    return false;
                }
                result.mesh = std::move(extraction.mesh);
                result.statistics.visibilityOccupancyBoundaryOccupiedCellCount =
                    extraction.statistics.occupiedCellCount;
                result.statistics.visibilityOccupancyBoundaryExposedQuadCount = extraction.statistics.exposedQuadCount;
                result.statistics.visibilityOccupancyBoundaryNonManifoldEdgeCount =
                    extraction.statistics.nonManifoldEdgeCount;
                result.statistics.visibilityOccupancyBoundaryNonManifoldVertexCount =
                    extraction.statistics.nonManifoldVertexCount;
                const int occupancy_body_euler = VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
                    native_carrier_dimensions, native_carrier_occupied);
                result.statistics.visibilityOccupancyBoundaryBodyEulerCharacteristic = occupancy_body_euler;
                result.statistics.visibilityOccupancyBoundarySurfaceEulerCharacteristic =
                    extraction.statistics.eulerCharacteristic;
                const bool topology_consistent = extraction.statistics.closedTwoManifold &&
                                                 extraction.statistics.eulerCharacteristic == 2 * occupancy_body_euler;
                result.statistics.visibilityOccupancyBoundaryTopologyConsistent = topology_consistent;
                if (!topology_consistent)
                {
                    result.errorMessage = QStringLiteral("可见性占据体不是良构闭合流形: 边界边=%1, "
                                                         "非流形边=%2, 非流形顶点=%3, 体欧拉特征=%4, "
                                                         "曲面欧拉特征=%5（期望 %6）")
                                              .arg(extraction.statistics.boundaryEdgeCount)
                                              .arg(extraction.statistics.nonManifoldEdgeCount)
                                              .arg(extraction.statistics.nonManifoldVertexCount)
                                              .arg(occupancy_body_euler)
                                              .arg(extraction.statistics.eulerCharacteristic)
                                              .arg(2 * occupancy_body_euler);
                    return false;
                }
            }
            else if (options.enableMc33IsoSurfaceExtraction)
            {
                Mc33IsoSurfaceOptions extraction_options;
                extraction_options.isoLevel = 0.0f;
                extraction_options.requireSupportedSignChange = options.mc33RequireSupportedSignChange;
                extraction_options.isCancelled = [control = options.execution]() { return control.isCancelled(); };
                Mc33IsoSurfaceResult extraction = Mc33IsoSurfaceExtractor::extract(extraction_bounds_min,
                                                                                   extraction_bounds_max,
                                                                                   extraction_cells,
                                                                                   extraction_tsdf,
                                                                                   extraction_support,
                                                                                   extraction_options);
                if (!extraction.ok)
                {
                    result.errorMessage = QStringLiteral("TSDF MC33 iso-surface extraction failed: %1")
                                              .arg(QString::fromStdString(extraction.errorMessage));
                    return false;
                }
                result.mesh = std::move(extraction.mesh);
                result.statistics.mc33SupportMaskedSampleCount = extraction.statistics.supportMaskedSampleCount;
                result.statistics.mc33RejectedUnsupportedCellFaceCount =
                    extraction.statistics.rejectedUnsupportedCellFaceCount;
            }
            else if (options.enableConsistentIsoSurfaceExtraction)
            {
                ConsistentIsoSurfaceOptions extraction_options;
                extraction_options.isoLevel = 0.0f;
                extraction_options.isCancelled = [control = options.execution]() { return control.isCancelled(); };
                ConsistentIsoSurfaceResult extraction = ConsistentIsoSurfaceExtractor::extract(extraction_bounds_min,
                                                                                               extraction_bounds_max,
                                                                                               extraction_cells,
                                                                                               extraction_tsdf,
                                                                                               extraction_support,
                                                                                               extraction_options);
                if (!extraction.ok)
                {
                    result.errorMessage = QStringLiteral("TSDF consistent iso-surface extraction failed: %1")
                                              .arg(QString::fromStdString(extraction.errorMessage));
                    return false;
                }
                result.mesh = std::move(extraction.mesh);
                result.statistics.isoSurfaceAmbiguousFaceCount = extraction.statistics.uniqueAmbiguousFaceCount;
                result.statistics.isoSurfaceTopologyAdjustedCellCount = extraction.statistics.topologyAdjustedCellCount;
                result.statistics.isoSurfaceDeciderTieCount = extraction.statistics.deciderTieCount;
                result.statistics.isoSurfaceMultipleLoopCellCount = extraction.statistics.multipleLoopCellCount;
                result.statistics.isoSurfaceEdgeVertexCacheHitCount = extraction.statistics.edgeVertexCacheHitCount;
                result.statistics.isoSurfaceEdgeVertexCacheMissCount = extraction.statistics.edgeVertexCacheMissCount;
                result.statistics.isoSurfaceInteriorLoopVertexCount = extraction.statistics.interiorLoopVertexCount;
                result.statistics.isoSurfaceRejectedDegenerateFaceCount =
                    extraction.statistics.rejectedDegenerateFaceCount;
                result.statistics.isoSurfaceUnresolvedCellCount = extraction.statistics.unresolvedCellCount;
            }
            else
            {
                plapoint::mesh::MarchingCubes<float> marchingCubes;
                marchingCubes.setBounds(
                    {result.layout.boundsMin[0], result.layout.boundsMin[1], result.layout.boundsMin[2]},
                    {result.layout.boundsMax[0], result.layout.boundsMax[1], result.layout.boundsMax[2]});
                marchingCubes.setResolution(result.layout.cells[0], result.layout.cells[1], result.layout.cells[2]);
                marchingCubes.setIsoLevel(0.0f);
                auto [vertices, faces] = marchingCubes.extract(
                    [&](float x, float y, float z)
                    {
                        const int ix = std::clamp(static_cast<int>(std::lround((x - result.layout.boundsMin[0]) /
                                                                               result.layout.voxelSize[0])),
                                                  0,
                                                  result.layout.cells[0]);
                        const int iy = std::clamp(static_cast<int>(std::lround((y - result.layout.boundsMin[1]) /
                                                                               result.layout.voxelSize[1])),
                                                  0,
                                                  result.layout.cells[1]);
                        const int iz = std::clamp(static_cast<int>(std::lround((z - result.layout.boundsMin[2]) /
                                                                               result.layout.voxelSize[2])),
                                                  0,
                                                  result.layout.cells[2]);
                        const std::size_t index = sampleIndex(result.layout, ix, iy, iz);
                        return extraction_support[index] != 0 ? extraction_tsdf[index] : 1.0f;
                    });
                result.mesh.vertices.resize(static_cast<std::size_t>(vertices.rows()));
                for (plamatrix::Index row = 0; row < vertices.rows(); ++row)
                {
                    MeshVertex& vertex = result.mesh.vertices[static_cast<std::size_t>(row)];
                    vertex.x = vertices(row, 0);
                    vertex.y = vertices(row, 1);
                    vertex.z = vertices(row, 2);
                }
                result.mesh.faces.resize(static_cast<std::size_t>(faces.rows()));
                for (plamatrix::Index row = 0; row < faces.rows(); ++row)
                {
                    Triangle& face = result.mesh.faces[static_cast<std::size_t>(row)];
                    face.v[0] = static_cast<int>(std::lround(faces(row, 0)));
                    face.v[1] = static_cast<int>(std::lround(faces(row, 1)));
                    face.v[2] = static_cast<int>(std::lround(faces(row, 2)));
                }
            }
            if (result.statistics.effectiveVisualHullCompletionTopologyGuard &&
                result.statistics.visualHullCompletionRecoveredSampleCount > 0)
            {
                const std::vector<std::uint8_t>& baseline_support =
                    !adaptiveTgvExtractionSupport.empty() ? adaptiveTgvExtractionSupport : supported;
                ComparableIsoSurfaceExtraction baseline_extraction =
                    extractComparableIsoSurface(result.layout.boundsMin,
                                                result.layout.boundsMax,
                                                result.layout.cells,
                                                tsdf,
                                                baseline_support,
                                                options.enableMc33IsoSurfaceExtraction,
                                                options.mc33RequireSupportedSignChange,
                                                options.enableConsistentIsoSurfaceExtraction,
                                                [control = options.execution]() { return control.isCancelled(); });
                if (!baseline_extraction.ok || baseline_extraction.mesh.empty())
                {
                    result.errorMessage = QStringLiteral("TSDF visual-hull topology guard could not extract the "
                                                         "uncompleted baseline: %1")
                                              .arg(baseline_extraction.errorMessage);
                    return false;
                }

                const DepthTsdfVisualHullTopologyGuardEvaluation evaluation =
                    DepthTsdfSurfaceBuilder::evaluateVisualHullCompletionTopologyGuard(
                        baseline_extraction.mesh,
                        result.mesh,
                        options.visualHullCompletionMaximumTopologicalComplexityIncrease,
                        options.visualHullCompletionMaximumSurfaceAreaRatio,
                        options.visualHullCompletionMaximumBoundsDiagonalRatio);
                result.statistics.visualHullCompletionTopologyGuardEvaluated = true;
                result.statistics.visualHullCompletionTopologyGuardAccepted = evaluation.accepted;
                result.statistics.visualHullCompletionTopologyGuardRejectionFlags = evaluation.rejectionFlags;
                result.statistics.visualHullCompletionTopologyGuardBaselineFaceCount = evaluation.baselineFaceCount;
                result.statistics.visualHullCompletionTopologyGuardCandidateFaceCount = evaluation.candidateFaceCount;
                result.statistics.visualHullCompletionTopologyGuardBaselineBoundaryEdgeCount =
                    evaluation.baselineBoundaryEdgeCount;
                result.statistics.visualHullCompletionTopologyGuardCandidateBoundaryEdgeCount =
                    evaluation.candidateBoundaryEdgeCount;
                result.statistics.visualHullCompletionTopologyGuardBaselineNonManifoldEdgeCount =
                    evaluation.baselineNonManifoldEdgeCount;
                result.statistics.visualHullCompletionTopologyGuardCandidateNonManifoldEdgeCount =
                    evaluation.candidateNonManifoldEdgeCount;
                result.statistics.visualHullCompletionTopologyGuardBaselineComponentCount =
                    evaluation.baselineComponentCount;
                result.statistics.visualHullCompletionTopologyGuardCandidateComponentCount =
                    evaluation.candidateComponentCount;
                result.statistics.visualHullCompletionTopologyGuardBaselineEulerCharacteristic =
                    evaluation.baselineEulerCharacteristic;
                result.statistics.visualHullCompletionTopologyGuardCandidateEulerCharacteristic =
                    evaluation.candidateEulerCharacteristic;
                result.statistics.visualHullCompletionTopologyGuardBaselineTopologicalComplexity =
                    evaluation.baselineTopologicalComplexity;
                result.statistics.visualHullCompletionTopologyGuardCandidateTopologicalComplexity =
                    evaluation.candidateTopologicalComplexity;
                result.statistics.visualHullCompletionTopologyGuardSurfaceAreaRatio = evaluation.surfaceAreaRatio;
                result.statistics.visualHullCompletionTopologyGuardBoundsDiagonalRatio = evaluation.boundsDiagonalRatio;

                if (!evaluation.accepted)
                {
                    result.mesh = std::move(baseline_extraction.mesh);
                    result.statistics.mc33SupportMaskedSampleCount = baseline_extraction.mc33SupportMaskedSampleCount;
                    result.statistics.mc33RejectedUnsupportedCellFaceCount =
                        baseline_extraction.mc33RejectedUnsupportedCellFaceCount;
                    result.statistics.isoSurfaceAmbiguousFaceCount = baseline_extraction.isoSurfaceAmbiguousFaceCount;
                    result.statistics.isoSurfaceTopologyAdjustedCellCount =
                        baseline_extraction.isoSurfaceTopologyAdjustedCellCount;
                    result.statistics.isoSurfaceDeciderTieCount = baseline_extraction.isoSurfaceDeciderTieCount;
                    result.statistics.isoSurfaceMultipleLoopCellCount =
                        baseline_extraction.isoSurfaceMultipleLoopCellCount;
                    result.statistics.isoSurfaceEdgeVertexCacheHitCount =
                        baseline_extraction.isoSurfaceEdgeVertexCacheHitCount;
                    result.statistics.isoSurfaceEdgeVertexCacheMissCount =
                        baseline_extraction.isoSurfaceEdgeVertexCacheMissCount;
                    result.statistics.isoSurfaceInteriorLoopVertexCount =
                        baseline_extraction.isoSurfaceInteriorLoopVertexCount;
                    result.statistics.isoSurfaceRejectedDegenerateFaceCount =
                        baseline_extraction.isoSurfaceRejectedDegenerateFaceCount;
                    result.statistics.isoSurfaceUnresolvedCellCount = baseline_extraction.isoSurfaceUnresolvedCellCount;
                }
            }
            result.statistics.marchingCubesVertexCount = result.mesh.vertexCount();
            result.statistics.marchingCubesFaceCount = result.mesh.faceCount();
            result.statistics.marchingCubesBoundaryEdgeCount = boundaryEdgeCount(result.mesh);
            const MeshTopologyQualityStatistics extraction_topology = evaluateMeshTopologyQuality(result.mesh);
            result.statistics.isoSurfaceExtractionNonManifoldEdgeCount = extraction_topology.nonManifoldEdgeCount;
            result.statistics.isoSurfaceExtractionComponentCount = extraction_topology.componentCount;
            result.statistics.isoSurfaceExtractionEulerCharacteristic = extraction_topology.eulerCharacteristic;
            result.statistics.marchingCubesElapsedMs = elapsedMilliseconds(marching_cubes_start);
        }
        catch (const std::exception& exception)
        {
            result.errorMessage =
                QStringLiteral("TSDF Marching Cubes failed: %1").arg(QString::fromUtf8(exception.what()));
            return false;
        }
        if (result.mesh.empty())
        {
            result.errorMessage = QStringLiteral("TSDF extraction produced an empty mesh");
            return false;
        }
        capture_stage(QStringLiteral("01_extracted_surface"), result.mesh);
        reportProgress(QStringLiteral("TSDF 零等值面提取完成：%1 点、%2 面，正在清理网格...")
                           .arg(result.mesh.vertexCount())
                           .arg(result.mesh.faceCount()),
                       82);
        if (postprocessCancelled())
        {
            return false;
        }

        return true;
    }
} // namespace xjw::mesh::tsdf_detail
