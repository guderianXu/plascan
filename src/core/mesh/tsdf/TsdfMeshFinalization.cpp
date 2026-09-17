#include "DepthTsdfStages.h"
namespace xjw::mesh::tsdf_detail
{
    bool finalizeTsdfMesh(const QVector<DepthTsdfFrame>& frames,
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

        const PostprocessClock::time_point colorization_start = PostprocessClock::now();
        if (options.calculateVertexColors)
        {
            reportProgress(QStringLiteral("正在计算网格顶点颜色..."), 94);
            QVector<MeshColorView> color_views;
            color_views.reserve(frames.size());
            for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
            {
                const DepthTsdfFrame& frame = frames[frame_index];
                MeshColorView view;
                view.camera = frame.camera;
                view.colorBgr = frame.colorBgr;
                view.depth = frame.depth;
                view.confidence = frame.confidence;
                view.depthValidMask =
                    erosion_pixels > 0 ? effective_depth_valid_masks[frame_index] : frame.depthValidMask;
                view.supportMask = frame.supportMask;
                view.qualityWeight = effective_frame_quality_weights[frame_index];
                color_views.push_back(std::move(view));
            }
            MeshColorOptions color_options;
            color_options.maximumVoxelSize =
                std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
            color_options.minimumConfidence = options.minimumConfidence;
            color_options.compensateExposure = options.compensateColorExposure;
            color_options.coherentFacePrimaryViews = options.coherentFacePrimaryViewColors;
            color_options.workerCount = options.workerCount;
            result.statistics.effectiveColorExposureCompensation = options.compensateColorExposure;
            result.statistics.effectiveCoherentFacePrimaryViewColors = options.coherentFacePrimaryViewColors;
            const MeshColorStatistics color_statistics =
                MeshColorizer::colorize(&result.mesh, color_views, color_options);
            result.statistics.colorCandidateObservationCount = color_statistics.candidateObservationCount;
            result.statistics.colorRejectedProjectionCount = color_statistics.rejectedProjectionCount;
            result.statistics.colorRejectedMaskCount = color_statistics.rejectedMaskCount;
            result.statistics.colorRejectedDepthCount = color_statistics.rejectedDepthCount;
            result.statistics.colorRejectedVisibilityCount = color_statistics.rejectedVisibilityCount;
            result.statistics.colorRejectedViewAngleCount = color_statistics.rejectedViewAngleCount;
            result.statistics.colorRejectedOutlierCount = color_statistics.rejectedColorOutlierCount;
            result.statistics.reliablyColoredVertexCount = color_statistics.reliablyColoredVertexCount;
            result.statistics.bestViewFallbackColorVertexCount = color_statistics.bestViewFallbackVertexCount;
            result.statistics.propagatedColorVertexCount = color_statistics.propagatedVertexCount;
            result.statistics.fallbackColorVertexCount = color_statistics.fallbackVertexCount;
            result.statistics.cleanedColorSpeckleVertexCount = color_statistics.cleanedSpeckleVertexCount;
            result.statistics.coherentPrimaryViewFaceCount = color_statistics.coherentPrimaryViewFaceCount;
            result.statistics.coherentPrimaryViewVertexCount = color_statistics.coherentPrimaryViewVertexCount;
            result.statistics.meshColorizationWorkerCount = color_statistics.effectiveWorkerCount;
        }
        else
        {
            result.mesh.hasVertexColors = false;
        }
        result.statistics.meshColorizationElapsedMs = elapsedMilliseconds(colorization_start);
        if (postprocessCancelled())
        {
            return false;
        }
        reportProgress(QStringLiteral("正在汇总网格质量统计..."), 98);
        MeshTopologyQualityThresholds topology_quality_thresholds;
        topology_quality_thresholds.maximumBoundaryEdgeRatio =
            std::max(0.0f, options.topologyQualityMaximumBoundaryEdgeRatio);
        topology_quality_thresholds.maximumHighAspectFaceRatio =
            std::max(0.0f, options.topologyQualityMaximumHighAspectFaceRatio);
        topology_quality_thresholds.maximumExtremeAspectFaceRatio =
            std::max(0.0f, options.topologyQualityMaximumExtremeAspectFaceRatio);
        topology_quality_thresholds.maximumClosedGenus = std::max(0.0f, options.topologyQualityMaximumClosedGenus);
        topology_quality_thresholds.maximumTopologicalComplexity =
            std::max(0, options.topologyQualityMaximumTopologicalComplexity);
        const MeshTopologyQualityStatistics topology_quality =
            evaluateMeshTopologyQuality(result.mesh, topology_quality_thresholds);
        result.statistics.topologyQualityUniqueEdgeCount = topology_quality.uniqueEdgeCount;
        result.statistics.topologyQualityReferencedVertexCount = topology_quality.referencedVertexCount;
        result.statistics.topologyQualityEulerCharacteristic = topology_quality.eulerCharacteristic;
        result.statistics.topologyQualityTopologicalComplexity = topology_quality.topologicalComplexity;
        result.statistics.topologyQualityClosedGenusEstimate = topology_quality.closedGenusEstimate;
        result.statistics.topologyQualityClosedTopologyEvaluated = topology_quality.closedTopologyEvaluated;
        result.statistics.topologyQualityBoundaryEdgeCount = topology_quality.boundaryEdgeCount;
        result.statistics.topologyQualityNonManifoldEdgeCount = topology_quality.nonManifoldEdgeCount;
        result.statistics.topologyQualityComponentCount = topology_quality.componentCount;
        result.statistics.topologyQualityHighAspectFaceCount = topology_quality.highAspectFaceCount;
        result.statistics.topologyQualityExtremeAspectFaceCount = topology_quality.extremeAspectFaceCount;
        result.statistics.topologyQualityBoundaryEdgeRatio = topology_quality.boundaryEdgeRatio;
        result.statistics.topologyQualityLargestComponentFaceRatio = topology_quality.largestComponentFaceRatio;
        result.statistics.topologyQualityHighAspectFaceRatio = topology_quality.highAspectFaceRatio;
        result.statistics.topologyQualityExtremeAspectFaceRatio = topology_quality.extremeAspectFaceRatio;
        result.statistics.topologyQualityAdjacentFacePairCount = topology_quality.adjacentFacePairCount;
        result.statistics.topologyQualityAdjacentNormalAngleMedianDegrees =
            topology_quality.adjacentNormalAngleMedianDegrees;
        result.statistics.topologyQualityAdjacentNormalAngleP90Degrees = topology_quality.adjacentNormalAngleP90Degrees;
        result.statistics.topologyQualityAdjacentNormalAngleOver30Ratio =
            topology_quality.adjacentNormalAngleOver30Ratio;
        if (options.collectZeroCrossingDiagnostics)
        {
            MeshBoundaryAttributionOptions attribution_options;
            attribution_options.minimumSourceCount = options.minimumSurfacePatchSourceCount;
            attribution_options.maximumInverseDepthSpread = options.maximumSurfacePatchInverseDepthSpread;
            attribution_options.minimumSurfaceWeightRatio = options.minimumSurfacePatchWeightRatio;
            attribution_options.maximumAbsoluteTsdf = options.maximumContourBandAbsoluteTsdf;
            std::vector<MeshBoundaryAttributionReason> vertex_reasons;
            std::vector<MeshBoundaryEdgeAttribution> edge_attributions;
            const MeshBoundaryAttributionStatistics attribution = attributeMeshBoundaryEdges(result.mesh,
                                                                                             result.layout,
                                                                                             tsdf,
                                                                                             weight,
                                                                                             surfaceObservationWeight,
                                                                                             geometrySourceMask,
                                                                                             minimumInverseDepthSpread,
                                                                                             supported,
                                                                                             attribution_options,
                                                                                             &vertex_reasons,
                                                                                             &edge_attributions);
            result.boundaryAttributionDebugMesh = result.mesh;
            applyMeshBoundaryAttributionColors(&result.boundaryAttributionDebugMesh, vertex_reasons);
            result.statistics.boundaryAttributionEdgeCount = attribution.boundaryEdgeCount;
            result.statistics.boundaryAttributionNoObservationEdgeCount = attribution.noObservationEdgeCount;
            result.statistics.boundaryAttributionInsufficientSourceEdgeCount = attribution.insufficientSourceEdgeCount;
            result.statistics.boundaryAttributionDepthSpreadRejectedEdgeCount =
                attribution.depthSpreadRejectedEdgeCount;
            result.statistics.boundaryAttributionSurfaceWeightRejectedEdgeCount =
                attribution.surfaceWeightRejectedEdgeCount;
            result.statistics.boundaryAttributionAbsoluteTsdfRejectedEdgeCount =
                attribution.absoluteTsdfRejectedEdgeCount;
            result.statistics.boundaryAttributionSupportGateRejectedEdgeCount =
                attribution.supportGateRejectedEdgeCount;
            result.statistics.boundaryAttributionExtractionOrPostprocessEdgeCount =
                attribution.extractionOrPostprocessEdgeCount;
            result.statistics.boundaryAttributionUnclassifiedEdgeCount = attribution.unclassifiedEdgeCount;
            if (options.collectAcquisitionGapReport)
            {
                QStringList frame_labels;
                frame_labels.reserve(static_cast<qsizetype>(geometry_source_encoding.slotToSourceIndex().size()));
                for (const int source_index : geometry_source_encoding.slotToSourceIndex())
                {
                    QString label = QStringLiteral("source_%1").arg(source_index);
                    for (const DepthTsdfFrame& frame : frames)
                    {
                        if (frame.refIndex == source_index && !frame.refImage.isEmpty())
                        {
                            label = frame.refImage;
                            break;
                        }
                    }
                    frame_labels.append(label);
                }
                result.acquisitionGapReport =
                    buildMeshAcquisitionGapReport(edge_attributions,
                                                  result.layout,
                                                  frame_labels,
                                                  geometry_source_encoding.slotToSourceIndex(),
                                                  frames.size(),
                                                  geometry_source_encoding.statistics().unmappedSourceCount == 0);
            }
        }
        result.statistics.topologyQualityStrictGatePassed = topology_quality.strictGatePassed;
        const MeshConnectivityStats connectivity = VisualHullReconstructor::analyzeConnectivity(result.mesh);
        result.statistics.componentCount = connectivity.componentCount;
        result.statistics.largestComponentFaceRatio = connectivity.largestComponentFaceRatio;
        result.statistics.componentFaceCounts = connectivity.componentFaceCounts;
        result.statistics.components = connectivity.components;
        result.statistics.vertexCount = result.mesh.vertexCount();
        result.statistics.faceCount = result.mesh.faceCount();
        result.statistics.effectiveDepthCompletenessDiagnostics = options.enableDepthCompletenessDiagnostics;
        result.statistics.effectiveDepthCompletenessGateEnforcement = options.enforceDepthCompletenessGate;
        const bool completeness_topology_safe =
            connectivity.componentCount == 1 && topology_quality.nonManifoldEdgeCount == 0 &&
            topology_quality.nonManifoldVertexCount == 0 && topology_quality.topologicalComplexity <= 128;
        result.statistics.depthCompletenessSkippedUnsafeTopology =
            options.enableDepthCompletenessDiagnostics && !completeness_topology_safe;
        if (options.enableDepthCompletenessDiagnostics && completeness_topology_safe)
        {
            if (options.execution.progress)
            {
                options.execution.reportProgress((QStringLiteral("正在检查各视角模型完整性...")).toUtf8().toStdString(),
                                                 (99) / 100.0);
            }
            DepthMeshCompletenessOptions completeness_options;
            completeness_options.maximumDepthSamplesPerFrame = options.depthCompletenessMaximumSamplesPerFrame;
            completeness_options.tolerance =
                maximum_voxel_size * std::max(1.0f, options.depthCompletenessToleranceVoxels);
            completeness_options.minimumP10FrameRecall = options.minimumDepthCompletenessP10Recall;
            completeness_options.minimumMedianFrameRecall = options.minimumDepthCompletenessMedianRecall;
            // Validation-only frames are deliberately downweighted auxiliary
            // evidence. They remain useful for surface support, but cannot serve
            // as hard geometric truth for accepting or rejecting the mesh.
            completeness_options.excludeAuxiliaryFrames = true;
            completeness_options.excludedRefIndices.assign(
                result.statistics.robustFrameQualityRejectedRefIndices.cbegin(),
                result.statistics.robustFrameQualityRejectedRefIndices.cend());
            const DepthMeshCompletenessStatistics completeness =
                DepthMeshCompleteness::evaluate(result.mesh, frames, completeness_options);
            result.statistics.depthCompletenessAvailable = completeness.available;
            result.statistics.depthCompletenessTolerance = completeness.tolerance;
            result.statistics.depthCompletenessSampledPointCount = completeness.sampledDepthPointCount;
            result.statistics.depthCompletenessExplainedPointCount = completeness.explainedDepthPointCount;
            result.statistics.depthCompletenessAggregateRecall = completeness.aggregateRecall;
            result.statistics.depthCompletenessMinimumFrameRecall = completeness.minimumFrameRecall;
            result.statistics.depthCompletenessP10FrameRecall = completeness.p10FrameRecall;
            result.statistics.depthCompletenessMedianFrameRecall = completeness.medianFrameRecall;
            for (const DepthMeshFrameCompleteness& frame : completeness.frames)
            {
                result.statistics.depthCompletenessRefIndices.push_back(frame.refIndex);
                result.statistics.depthCompletenessFrameRecalls.push_back(frame.recall);
            }
            std::unordered_map<int, QString> orbital_role_by_ref_index;
            for (const QJsonValue& value : result.statistics.depthCompletenessFrameRoles)
            {
                const QJsonObject role_object = value.toObject();
                orbital_role_by_ref_index.emplace(role_object.value(QStringLiteral("ref_index")).toInt(-1),
                                                  role_object.value(QStringLiteral("role")).toString());
            }
            double gap_boundary_minimum_recall = 1.0;
            for (const DepthMeshFrameCompleteness& frame : completeness.frames)
            {
                const auto role_it = orbital_role_by_ref_index.find(frame.refIndex);
                if (role_it == orbital_role_by_ref_index.end() || role_it->second != QStringLiteral("gap_boundary"))
                {
                    continue;
                }
                result.statistics.depthCompletenessGapBoundaryAvailable = true;
                result.statistics.depthCompletenessGapBoundaryRefIndices.push_back(frame.refIndex);
                result.statistics.depthCompletenessGapBoundaryFrameRecalls.push_back(frame.recall);
                gap_boundary_minimum_recall = std::min(gap_boundary_minimum_recall, frame.recall);
            }
            if (result.statistics.depthCompletenessGapBoundaryAvailable)
            {
                result.statistics.depthCompletenessGapBoundaryMinimumRecall = gap_boundary_minimum_recall;
                result.statistics.depthCompletenessGapBoundaryGatePassed =
                    gap_boundary_minimum_recall + 1.0e-9 >= options.minimumDepthCompletenessP10Recall;
            }
            result.statistics.depthCompletenessGatePassed =
                completeness.gatePassed && result.statistics.depthCompletenessGapBoundaryGatePassed;
            if (options.enforceDepthCompletenessGate &&
                (!completeness.available || !result.statistics.depthCompletenessGatePassed))
            {
                std::vector<DepthMeshFrameCompleteness> worst_frames(completeness.frames.cbegin(),
                                                                     completeness.frames.cend());
                std::sort(worst_frames.begin(),
                          worst_frames.end(),
                          [](const DepthMeshFrameCompleteness& lhs, const DepthMeshFrameCompleteness& rhs)
                          { return lhs.recall < rhs.recall; });
                QStringList worst_labels;
                const int worst_count = std::min(3, static_cast<int>(worst_frames.size()));
                for (int index = 0; index < worst_count; ++index)
                {
                    const DepthMeshFrameCompleteness& frame = worst_frames[static_cast<std::size_t>(index)];
                    QString role;
                    const auto role_it = orbital_role_by_ref_index.find(frame.refIndex);
                    if (role_it != orbital_role_by_ref_index.end())
                    {
                        role = role_it->second;
                    }
                    worst_labels.push_back(
                        role.isEmpty()
                            ? QStringLiteral("%1=%2%").arg(frame.refIndex).arg(100.0 * frame.recall, 0, 'f', 1)
                            : QStringLiteral("%1=%2%[%3]")
                                  .arg(frame.refIndex)
                                  .arg(100.0 * frame.recall, 0, 'f', 1)
                                  .arg(role));
                }
                result.errorMessage = QStringLiteral("TSDF 深度观测完整性质量门未通过：中位召回率=%1，"
                                                     "P10 召回率=%2，最低召回率=%3；最差视角=%4。"
                                                     "模型可能存在整侧缺失或大面积空洞。")
                                          .arg(completeness.medianFrameRecall, 0, 'f', 4)
                                          .arg(completeness.p10FrameRecall, 0, 'f', 4)
                                          .arg(completeness.minimumFrameRecall, 0, 'f', 4)
                                          .arg(worst_labels.join(QStringLiteral(", ")));
                return false;
            }
        }
        else if (result.statistics.depthCompletenessSkippedUnsafeTopology && options.execution.progress)
        {
            options.execution.reportProgress(
                (QStringLiteral("模型拓扑未通过安全前置检查，跳过深度完整性索引构建")).toUtf8().toStdString(),
                (99) / 100.0);
        }
        if (options.visibilityOccupancyCellBoundaryExtraction && !native_carrier_field.empty())
        {
            result.visibilityOccupancyCarrierField.sampleDimensions = native_carrier_dimensions;
            result.visibilityOccupancyCarrierField.boundsMin = native_carrier_bounds_min;
            result.visibilityOccupancyCarrierField.boundsMax = native_carrier_bounds_max;
            result.visibilityOccupancyCarrierField.signedWorldDistance = std::move(native_carrier_field);
        }
        if (options.visibilityOccupancyCellBoundaryExtraction)
        {
            const std::vector<float>& implicit_tsdf =
                visual_hull_completion_tsdf.empty() ? tsdf : visual_hull_completion_tsdf;
            if (implicit_tsdf.size() == static_cast<std::size_t>(result.layout.sampleCount))
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    result.depthImplicitField.sampleDimensions[axis] = result.layout.cells[axis] + 1;
                }
                result.depthImplicitField.boundsMin = result.layout.boundsMin;
                result.depthImplicitField.boundsMax = result.layout.boundsMax;
                result.depthImplicitField.signedWorldDistance.resize(implicit_tsdf.size());
                std::transform(implicit_tsdf.cbegin(),
                               implicit_tsdf.cend(),
                               result.depthImplicitField.signedWorldDistance.begin(),
                               [truncation](float value) { return std::clamp(value, -1.0f, 1.0f) * truncation; });
            }
        }
        result.statistics.postIntegrationElapsedMs = elapsedMilliseconds(postprocess_start);

        result.ok = true;
        result.errorMessage.clear();
        if (options.execution.progress)
        {
            options.execution.reportProgress((QStringLiteral("TSDF 表面重建完成")).toUtf8().toStdString(),
                                             (100) / 100.0);
        }
        return true;
    }
} // namespace xjw::mesh::tsdf_detail
