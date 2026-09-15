#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow
{
    using namespace workflow_detail;
    static_assert(kDepthGeometrySourceSlotCount == 256 && sizeof(DepthGeometrySourceMask) == 4 * sizeof(std::uint64_t),
                  "Depth TSDF workflow requires the collision-free 256-bit source encoding");

    float depthFrameTextureQualityWeight(float frame_quality_weight, bool auxiliary_surface_only) noexcept
    {
        return frame_quality_weight * (auxiliary_surface_only ? xjw::mvs::kCoverageAuxiliaryWeightMultiplier : 1.0f);
    }

} // namespace xjw::mesh::workflow

#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow::workflow_detail
{
    using namespace workflow_detail;

    QJsonObject textureResultToJson(const xjw::mesh::TextureMappingResult& result,
                                    const xjw::mesh::TextureMappingConfig* config)
    {
        QJsonObject object;
        object[QStringLiteral("model_obj")] = xjw::common::io::fromUtf8Path(result.modelObjPath);
        object[QStringLiteral("model_mtl")] = xjw::common::io::fromUtf8Path(result.modelMtlPath);
        object[QStringLiteral("texture_png")] = xjw::common::io::fromUtf8Path(result.texturePngPath);
        object[QStringLiteral("texture_image")] = xjw::common::io::fromUtf8Path(result.texturePngPath);
        object[QStringLiteral("textured")] = !result.modelObjPath.empty() && !result.texturePngPath.empty();
        object[QStringLiteral("texture_size")] = result.textureSize;
        object[QStringLiteral("texture_algorithm")] = QString::fromStdString(result.textureAlgorithm);
        object[QStringLiteral("uv_method")] = QString::fromStdString(result.uvMethod);
        object[QStringLiteral("blend_method")] = QString::fromStdString(result.blendMethod);
        object[QStringLiteral("texture_source_view_count")] = result.sourceViewCount;
        object[QStringLiteral("texture_mapped_face_count")] = result.mappedFaceCount;
        object[QStringLiteral("texture_fallback_mapped_face_count")] = result.fallbackMappedFaceCount;
        object[QStringLiteral("texture_coherence_adjusted_face_count")] = result.coherenceAdjustedFaceCount;
        object[QStringLiteral("texture_unmapped_face_count")] = result.unmappedFaceCount;
        object[QStringLiteral("texture_strict_mapped_face_count")] = result.strictMappedFaceCount;
        object[QStringLiteral("texture_mesh_recovered_face_count")] = result.meshRecoveredFaceCount;
        object[QStringLiteral("texture_chart_count")] = result.chartCount;
        object[QStringLiteral("texture_used_view_count")] = result.usedViewCount;
        object[QStringLiteral("texture_candidate_evaluation_count")] =
            static_cast<qint64>(result.candidateEvaluationCount);
        object[QStringLiteral("texture_rejected_projection_count")] =
            static_cast<qint64>(result.rejectedProjectionCount);
        object[QStringLiteral("texture_rejected_mask_count")] = static_cast<qint64>(result.rejectedMaskCount);
        object[QStringLiteral("texture_rejected_depth_count")] = static_cast<qint64>(result.rejectedDepthCount);
        object[QStringLiteral("texture_rejected_visibility_count")] =
            static_cast<qint64>(result.rejectedVisibilityCount);
        object[QStringLiteral("texture_rejected_angle_count")] = static_cast<qint64>(result.rejectedAngleCount);
        object[QStringLiteral("texture_rejected_resolution_count")] =
            static_cast<qint64>(result.rejectedResolutionCount);
        object[QStringLiteral("texture_rejected_color_outlier_count")] =
            static_cast<qint64>(result.rejectedColorOutlierCount);
        object[QStringLiteral("texture_visibility_rasterized_pixel_count")] =
            static_cast<qint64>(result.visibilityRasterizedPixelCount);
        object[QStringLiteral("texture_no_texel_face_count")] = static_cast<qint64>(result.noTexelFaceCount);
        object[QStringLiteral("texture_center_recovered_face_count")] =
            static_cast<qint64>(result.centerRecoveredFaceCount);
        object[QStringLiteral("texture_unresolved_bake_face_count")] =
            static_cast<qint64>(result.unresolvedBakeFaceCount);
        object[QStringLiteral("texture_exposure_correction_status")] =
            QString::fromStdString(result.exposureCorrectionStatus);
        object[QStringLiteral("texture_exposure_correction_applied")] = result.exposureCorrectionApplied;
        object[QStringLiteral("texture_exposure_correction_graph_connected")] = result.exposureCorrectionGraphConnected;
        object[QStringLiteral("texture_exposure_correction_observation_count")] =
            static_cast<qint64>(result.exposureCorrectionObservationCount);
        object[QStringLiteral("texture_exposure_correction_candidate_pair_count")] =
            static_cast<qint64>(result.exposureCorrectionCandidatePairCount);
        object[QStringLiteral("texture_exposure_correction_accepted_pair_count")] =
            static_cast<qint64>(result.exposureCorrectionAcceptedPairCount);
        object[QStringLiteral("texture_exposure_correction_rejected_insufficient_pair_count")] =
            static_cast<qint64>(result.exposureCorrectionRejectedInsufficientPairCount);
        object[QStringLiteral("texture_exposure_correction_rejected_high_mad_pair_count")] =
            static_cast<qint64>(result.exposureCorrectionRejectedHighMadPairCount);
        object[QStringLiteral("texture_exposure_correction_connected_component_count")] =
            result.exposureCorrectionConnectedComponentCount;
        object[QStringLiteral("texture_exposure_correction_corrected_view_count")] =
            result.exposureCorrectionCorrectedViewCount;
        object[QStringLiteral("texture_exposure_correction_maximum_accepted_log_mad")] =
            result.exposureCorrectionMaximumAcceptedLogMad;
        object[QStringLiteral("texture_exposure_correction_minimum_gain")] = result.exposureCorrectionMinimumGain;
        object[QStringLiteral("texture_exposure_correction_maximum_gain")] = result.exposureCorrectionMaximumGain;
        object[QStringLiteral("texture_atlas_occupancy")] = result.atlasOccupancy;
        object[QStringLiteral("texture_median_texel_density")] = result.medianTexelDensity;
        object[QStringLiteral("texture_seam_color_difference")] = result.seamColorDifference;
        object[QStringLiteral("texture_seam_constraint_count")] = result.seamConstraintCount;
        object[QStringLiteral("texture_seam_adjusted_chart_count")] = result.seamAdjustedChartCount;
        object[QStringLiteral("texture_seam_adjusted_pixel_count")] = result.seamAdjustedPixelCount;
        object[QStringLiteral("texture_seam_maximum_applied_linear_correction")] =
            result.seamMaximumAppliedLinearCorrection;
        object[QStringLiteral("texture_peak_memory_estimate_mib")] = result.peakMemoryEstimateMiB;
        if (config)
        {
            object[QStringLiteral("effective_texture_image_downscale")] = std::clamp(config->imageDownscale, 1, 8);
            object[QStringLiteral("effective_texture_anti_aliasing")] = std::clamp(config->antiAliasing, 1, 4);
            object[QStringLiteral("effective_texture_atlas_upscale_limit")] =
                std::clamp(config->atlasUpscaleLimit, 1.0f, 4.0f);
            object[QStringLiteral("effective_texture_padding")] = std::clamp(config->padding, 2, 64);
            object[QStringLiteral("effective_texture_seam_leveling")] = config->enableSeamLeveling;
            object[QStringLiteral("effective_texture_seam_border_blend_radius_pixels")] =
                std::clamp(config->seamBorderBlendRadiusPixels, 1, 64);
            object[QStringLiteral("effective_texture_seam_maximum_linear_correction")] =
                std::clamp(config->seamMaximumLinearCorrection, 0.0f, 0.25f);
            object[QStringLiteral("effective_texture_seam_global_correction_strength")] =
                std::clamp(config->seamGlobalCorrectionStrength, 0.0f, 1.0f);
            object[QStringLiteral("effective_texture_final_mesh_visibility")] = config->enableFinalMeshVisibility;
            object[QStringLiteral("effective_texture_ghost_filter")] = config->enableGhostFilter;
            object[QStringLiteral("effective_texture_out_of_focus_filter")] = config->enableOutOfFocusFilter;
            object[QStringLiteral("effective_texture_color_correction")] = config->enableColorCorrection;
            object[QStringLiteral("effective_texture_sharpening_strength")] =
                std::clamp(config->sharpeningStrength, 0.0f, 2.0f);
            object[QStringLiteral("effective_texture_hole_fill")] =
                config->holeFillMode != xjw::mesh::TextureHoleFillMode::Disabled;
        }
        return object;
    }

    MeshColorView textureViewFromFrame(const DepthTsdfFrame& frame)
    {
        MeshColorView view;
        view.camera = frame.camera;
        view.depth = frame.depth;
        view.confidence = frame.confidence;
        view.depthValidMask = frame.depthValidMask;
        view.supportMask = frame.supportMask;
        view.qualityWeight = depthFrameTextureQualityWeight(frame.frameQualityWeight, frame.auxiliarySurfaceOnly);

        if (!frame.refImage.isEmpty() && QFileInfo::exists(frame.refImage))
        {
            view.colorBgr = xjw::common::io::readImage(xjw::common::io::toUtf8Path(frame.refImage), cv::IMREAD_COLOR);
        }
        if (view.colorBgr.empty())
        {
            view.colorBgr = frame.colorBgr;
        }
        if (!view.colorBgr.empty() && frame.depth.cols > 0 && frame.depth.rows > 0)
        {
            view.colorCamera =
                frame.camera.scaledIntrinsics(static_cast<double>(view.colorBgr.cols) / frame.depth.cols,
                                              static_cast<double>(view.colorBgr.rows) / frame.depth.rows);
        }
        return view;
    }

    MeshColorView vertexColorViewFromFrame(const DepthTsdfFrame& frame)
    {
        return textureViewFromFrame(frame);
    }

    void addFinalMeshColorStatistics(const MeshColorStatistics& statistics, QJsonObject* payload)
    {
        if (!payload)
        {
            return;
        }

        (*payload)[QStringLiteral("final_mesh_recolorized")] = true;
        (*payload)[QStringLiteral("color_candidate_observation_count")] =
            static_cast<qint64>(statistics.candidateObservationCount);
        (*payload)[QStringLiteral("color_rejected_projection_count")] =
            static_cast<qint64>(statistics.rejectedProjectionCount);
        (*payload)[QStringLiteral("color_rejected_mask_count")] = static_cast<qint64>(statistics.rejectedMaskCount);
        (*payload)[QStringLiteral("color_rejected_depth_count")] = static_cast<qint64>(statistics.rejectedDepthCount);
        (*payload)[QStringLiteral("color_rejected_visibility_count")] =
            static_cast<qint64>(statistics.rejectedVisibilityCount);
        (*payload)[QStringLiteral("color_rejected_view_angle_count")] =
            static_cast<qint64>(statistics.rejectedViewAngleCount);
        (*payload)[QStringLiteral("color_rejected_outlier_count")] =
            static_cast<qint64>(statistics.rejectedColorOutlierCount);
        (*payload)[QStringLiteral("visibility_only_color_attempted_observation_count")] =
            static_cast<qint64>(statistics.visibilityOnlyAttemptedObservationCount);
        (*payload)[QStringLiteral("visibility_only_color_candidate_observation_count")] =
            static_cast<qint64>(statistics.visibilityOnlyCandidateObservationCount);
        (*payload)[QStringLiteral("visibility_only_color_rejected_foreground_count")] =
            static_cast<qint64>(statistics.visibilityOnlyRejectedForegroundCount);
        (*payload)[QStringLiteral("visibility_only_color_rejected_missing_foreground_count")] =
            static_cast<qint64>(statistics.visibilityOnlyRejectedMissingForegroundCount);
        (*payload)[QStringLiteral("visibility_only_color_rejected_visibility_count")] =
            static_cast<qint64>(statistics.visibilityOnlyRejectedVisibilityCount);
        (*payload)[QStringLiteral("visibility_only_color_rejected_view_angle_count")] =
            static_cast<qint64>(statistics.visibilityOnlyRejectedViewAngleCount);
        (*payload)[QStringLiteral("reliably_colored_vertex_count")] = statistics.reliablyColoredVertexCount;
        (*payload)[QStringLiteral("best_view_fallback_color_vertex_count")] = statistics.bestViewFallbackVertexCount;
        (*payload)[QStringLiteral("visibility_only_color_vertex_count")] = statistics.visibilityOnlyFallbackVertexCount;
        (*payload)[QStringLiteral("color_foreground_view_count")] = statistics.colorForegroundViewCount;
        (*payload)[QStringLiteral("visibility_only_color_fallback_enabled")] = statistics.visibilityOnlyFallbackEnabled;
        (*payload)[QStringLiteral("propagated_color_vertex_count")] = statistics.propagatedVertexCount;
        (*payload)[QStringLiteral("fallback_color_vertex_count")] = statistics.fallbackVertexCount;
        (*payload)[QStringLiteral("cleaned_color_speckle_vertex_count")] = statistics.cleanedSpeckleVertexCount;
        (*payload)[QStringLiteral("coherent_primary_view_face_count")] = statistics.coherentPrimaryViewFaceCount;
        (*payload)[QStringLiteral("coherent_primary_view_vertex_count")] = statistics.coherentPrimaryViewVertexCount;
        (*payload)[QStringLiteral("mesh_colorization_worker_count")] = statistics.effectiveWorkerCount;
        (*payload)[QStringLiteral("mesh_colorization_elapsed_ms")] = static_cast<qint64>(statistics.elapsedMs);
    }

    WorkflowResult saveMeshAndOptionalTexture(const xjw::mesh::TriMesh& mesh,
                                              const std::string& mesh_algorithm,
                                              const QString& output_root,
                                              bool export_obj,
                                              const xjw::mesh::TextureMappingConfig& texture,
                                              const std::function<void(const QString&, int)>& progress,
                                              const std::function<bool()>& isCancelled,
                                              const QVector<MeshColorView>* camera_views,
                                              const RecoveredModelResult* recovered_model)
    {
        if (cancellationRequested(isCancelled))
        {
            return cancelledWorkflowResult();
        }
        WorkflowResult result;
        const QString products_dir = QDir(output_root).filePath(QStringLiteral("products"));
        QDir().mkpath(products_dir);
        if (cancellationRequested(isCancelled))
        {
            return cancelledWorkflowResult();
        }
        const QString mesh_ply_path = QDir(products_dir).filePath(QStringLiteral("model_from_mesh.ply"));
        std::string mesh_error;
        if (cancellationRequested(isCancelled))
        {
            return cancelledWorkflowResult();
        }
        const bool mesh_saved = recovered_model ? writeRecoveredModelPly(*recovered_model, mesh_ply_path, &mesh_error)
                                                : mesh.savePLY(xjw::common::io::toUtf8Path(mesh_ply_path), &mesh_error);
        if (!mesh_saved)
        {
            result.errorMessage = QStringLiteral("网格保存失败: %1").arg(QString::fromStdString(mesh_error));
            return result;
        }
        if (cancellationRequested(isCancelled))
        {
            return cancelledWorkflowResult();
        }

        result.payload[QStringLiteral("mesh_ply")] = mesh_ply_path;
        result.payload[QStringLiteral("model_ply")] = mesh_ply_path;
        result.payload[QStringLiteral("vertex_count")] = mesh.vertexCount();
        result.payload[QStringLiteral("face_count")] = mesh.faceCount();
        result.payload[QStringLiteral("has_vertex_colors")] = mesh.hasVertexColors;
        if (mesh.hasVertexColors)
        {
            result.payload[QStringLiteral("vertex_color_format")] = QStringLiteral("3波段, uint8");
        }
        result.payload[QStringLiteral("mesh_algorithm")] =
            QString::fromStdString(mesh_algorithm.empty() ? "unknown" : mesh_algorithm);

        if (export_obj)
        {
            if (cancellationRequested(isCancelled))
            {
                return cancelledWorkflowResult();
            }
            std::string texture_error;
            xjw::mesh::TextureMappingConfig texture_config = texture;
            texture_config.isCancelled = isCancelled;
            if (progress)
            {
                texture_config.progressFn = [progress](const std::string& stage, int percent)
                { progress(QString::fromStdString(stage), percent); };
            }
            xjw::mesh::TextureMappingResult texture_result;
            const bool texture_ok = camera_views && !camera_views->empty()
                                        ? xjw::mesh::TextureMapper::generateCameraTexturedModelFromMeshFile(
                                              xjw::common::io::toUtf8Path(mesh_ply_path),
                                              xjw::common::io::toUtf8Path(products_dir),
                                              texture_config,
                                              *camera_views,
                                              &texture_result,
                                              &texture_error)
                                        : xjw::mesh::TextureMapper::generateTexturedModelFromMeshFile(
                                              xjw::common::io::toUtf8Path(mesh_ply_path),
                                              xjw::common::io::toUtf8Path(products_dir),
                                              texture_config,
                                              &texture_result,
                                              &texture_error);
            if (texture_ok)
            {
                const QJsonObject texture_json = textureResultToJson(texture_result, &texture_config);
                for (auto it = texture_json.begin(); it != texture_json.end(); ++it)
                {
                    result.payload[it.key()] = it.value();
                }
            }
            else if (!texture_error.empty())
            {
                result.payload[QStringLiteral("texture_warning")] = QString::fromStdString(texture_error);
            }
            if (cancellationRequested(isCancelled))
            {
                return cancelledWorkflowResult();
            }
        }

        if (cancellationRequested(isCancelled))
        {
            return cancelledWorkflowResult();
        }
        assignFinalModelFields(&result.payload, export_obj);
        result.ok = true;
        return result;
    }
} // namespace xjw::mesh::workflow::workflow_detail
