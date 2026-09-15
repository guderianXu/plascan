#include "MvsPipelineInternals.h"

namespace xjw::mvs::pipeline_detail
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    QJsonObject configuredEffectiveInteger(int configured, int effective)
    {
        return QJsonObject{{QStringLiteral("configured_full_raster"), configured},
                           {QStringLiteral("effective_grid"), effective}};
    }

    QJsonObject configuredEffectiveFloat(float configured, float effective)
    {
        return QJsonObject{{QStringLiteral("configured_full_raster"), configured},
                           {QStringLiteral("effective_grid"), effective}};
    }

    QJsonObject makePixelDomainDiagnostics(const cv::Size& raster_size,
                                           const cv::Size& grid_size,
                                           const FusionConfig& fusion_config,
                                           bool requested_native_final_grid,
                                           bool effective_native_final_grid)
    {
        const DepthPixelDomainScale scale = depthPixelDomainScale(raster_size, grid_size);
        const int quantized_boundary_edge_radius = scaleDepthPixelRadius(kFullRasterBoundaryEdgeRadiusPixels, scale);
        const int boundary_edge_radius = fusion_config.enableBoundaryAwareRetention && scale.usesReducedGrid() &&
                                                 kFullRasterBoundaryEdgeRadiusPixels > 0
                                             ? std::max(1, quantized_boundary_edge_radius)
                                             : quantized_boundary_edge_radius;
        const int boundary_protection_radius =
            scale.usesReducedGrid()
                ? std::max(fusion_config.enableBoundaryAwareRetention &&
                                   fusion_config.boundaryProtectionRadiusPixels > 0
                               ? 1
                               : 0,
                           static_cast<int>(std::floor(
                               static_cast<float>(fusion_config.boundaryProtectionRadiusPixels) * scale.linearScale +
                               1.0e-6f)))
                : scaleDepthPixelRadius(fusion_config.boundaryProtectionRadiusPixels, scale);
        const int local_kernel = scaleDepthLocalOutlierKernel(fusion_config.localDepthOutlierKernelSize, scale);
        const int same_layer_radius = scaleDepthPixelRadius(kFullRasterLocalSameLayerRadiusPixels, scale);
        const int speckle_area = scaleDepthPixelArea(fusion_config.minSpeckleComponentArea, scale);
        const int sparse_radius = scaleDepthPixelRadius(kFullRasterSparseResidualRadiusPixels, scale);
        const int consistency_radius = scaleDepthPixelRadius(kFullRasterConsistencySearchRadiusPixels, scale);
        const float consistency_round_trip = scaleDepthPixelDistance(kFullRasterConsistencyRoundTripPixels, scale);
        const int minimum_hole_area = scaleDepthPixelArea(kFullRasterMinimumSmallHoleArea, scale);
        const float fusion_reprojection = scaleDepthNearestSampleDistance(fusion_config.pixelThresh, scale);
        const bool boundary_retention_active = fusion_config.enableBoundaryAwareRetention && boundary_edge_radius > 0;

        QJsonObject boundary_edge = configuredEffectiveInteger(kFullRasterBoundaryEdgeRadiusPixels,
                                                               boundary_retention_active ? boundary_edge_radius : 0);
        boundary_edge.insert(QStringLiteral("quantized_grid"), quantized_boundary_edge_radius);
        boundary_edge.insert(QStringLiteral("active"), boundary_retention_active);
        if (!boundary_retention_active)
        {
            boundary_edge.insert(QStringLiteral("disabled_reason"),
                                 fusion_config.enableBoundaryAwareRetention
                                     ? QStringLiteral("edge_radius_subpixel_on_depth_grid")
                                     : QStringLiteral("configured_disabled"));
        }
        else if (scale.usesReducedGrid() && quantized_boundary_edge_radius == 0)
        {
            boundary_edge.insert(QStringLiteral("quantization_strategy"),
                                 QStringLiteral("minimum_single_grid_boundary_shell"));
        }

        QJsonObject boundary_protection = configuredEffectiveInteger(
            fusion_config.boundaryProtectionRadiusPixels, boundary_retention_active ? boundary_protection_radius : 0);
        boundary_protection.insert(QStringLiteral("quantized_grid"), boundary_protection_radius);
        boundary_protection.insert(QStringLiteral("active"), boundary_retention_active);
        if (boundary_retention_active && scale.usesReducedGrid() && fusion_config.boundaryProtectionRadiusPixels > 0 &&
            boundary_protection_radius == 1)
        {
            boundary_protection.insert(QStringLiteral("quantization_strategy"),
                                       QStringLiteral("minimum_single_grid_protection_shell"));
        }
        if (!boundary_retention_active)
        {
            boundary_protection.insert(QStringLiteral("disabled_reason"), QStringLiteral("boundary_edge_inactive"));
        }

        QJsonObject parameters;
        parameters.insert(QStringLiteral("boundary_edge_radius_pixels"), boundary_edge);
        parameters.insert(QStringLiteral("boundary_protection_radius_pixels"), boundary_protection);
        parameters.insert(QStringLiteral("local_depth_outlier_kernel_size"),
                          configuredEffectiveInteger(fusion_config.localDepthOutlierKernelSize, local_kernel));
        parameters.insert(QStringLiteral("local_same_layer_radius_pixels"),
                          configuredEffectiveInteger(kFullRasterLocalSameLayerRadiusPixels, same_layer_radius));
        parameters.insert(QStringLiteral("minimum_speckle_component_area"),
                          configuredEffectiveInteger(fusion_config.minSpeckleComponentArea, speckle_area));
        parameters.insert(QStringLiteral("sparse_residual_radius_pixels"),
                          configuredEffectiveInteger(kFullRasterSparseResidualRadiusPixels, sparse_radius));
        QJsonObject consistency_search =
            configuredEffectiveInteger(kFullRasterConsistencySearchRadiusPixels, consistency_radius);
        consistency_search.insert(QStringLiteral("subpixel_footprint_sampling"),
                                  scale.usesReducedGrid() && consistency_radius == 0);
        parameters.insert(QStringLiteral("consistency_search_radius_pixels"), consistency_search);
        parameters.insert(QStringLiteral("consistency_round_trip_pixels"),
                          configuredEffectiveFloat(kFullRasterConsistencyRoundTripPixels, consistency_round_trip));
        parameters.insert(QStringLiteral("minimum_small_hole_area"),
                          configuredEffectiveInteger(kFullRasterMinimumSmallHoleArea, minimum_hole_area));
        parameters.insert(QStringLiteral("maximum_small_hole_fraction"), kSmallHoleAreaFraction);
        parameters.insert(
            QStringLiteral("fusion_reprojection_base_error_pixels"),
            [&]()
            {
                QJsonObject value = configuredEffectiveFloat(fusion_config.pixelThresh, fusion_reprojection);
                value.insert(QStringLiteral("scope"),
                             QStringLiteral("base_before_view_count_or_streaming_runtime_override"));
                value.insert(QStringLiteral("runtime_scaled_per_target_frame"), true);
                value.insert(QStringLiteral("nearest_sample_quantization_floor"), scale.usesReducedGrid());
                return value;
            }());
        parameters.insert(QStringLiteral("fusion_local_gradient_base_radius_pixels"),
                          [&]()
                          {
                              QJsonObject value =
                                  configuredEffectiveInteger(kFullRasterLocalSameLayerRadiusPixels, same_layer_radius);
                              value.insert(QStringLiteral("runtime_scaled_per_target_frame"), true);
                              return value;
                          }());
        parameters.insert(QStringLiteral("legacy_inpaint_radius_pixels"),
                          QJsonObject{{QStringLiteral("configured_full_raster"), fusion_config.inpaintRadius},
                                      {QStringLiteral("active_in_current_postprocess"), false}});
        parameters.insert(QStringLiteral("connectivity"),
                          QJsonObject{{QStringLiteral("neighbors"), 8},
                                      {QStringLiteral("domain"), QStringLiteral("depth_grid_topology")}});

        QJsonObject diagnostics;
        diagnostics.insert(QStringLiteral("configured_pixel_domain"), QStringLiteral("prepared_full_raster"));
        diagnostics.insert(QStringLiteral("effective_pixel_domain"), QStringLiteral("depth_grid"));
        diagnostics.insert(QStringLiteral("requested_native_final_depth_grid"), requested_native_final_grid);
        diagnostics.insert(QStringLiteral("effective_native_final_depth_grid"), effective_native_final_grid);
        diagnostics.insert(QStringLiteral("raster_width"), raster_size.width);
        diagnostics.insert(QStringLiteral("raster_height"), raster_size.height);
        diagnostics.insert(QStringLiteral("grid_width"), grid_size.width);
        diagnostics.insert(QStringLiteral("grid_height"), grid_size.height);
        diagnostics.insert(QStringLiteral("scale_x"), scale.scaleX);
        diagnostics.insert(QStringLiteral("scale_y"), scale.scaleY);
        diagnostics.insert(QStringLiteral("linear_scale"), scale.linearScale);
        diagnostics.insert(QStringLiteral("area_scale"), scale.areaScale);
        diagnostics.insert(QStringLiteral("grid_matches_raster"), grid_size == raster_size);
        diagnostics.insert(QStringLiteral("parameters"), parameters);
        return diagnostics;
    }

    QJsonArray doubleArrayToJson(const double* values, int count)
    {
        QJsonArray array;
        for (int index = 0; index < count; ++index)
        {
            array.append(values[index]);
        }
        return array;
    }

    QJsonObject cameraModelToJson(const FramePinholeCamera& camera)
    {
        const FramePinholeCamera::Intrinsics intrinsics = camera.intrinsics();
        const std::array<double, 9> rotation = camera.worldToCameraRotation();
        const std::array<double, 3> translation = camera.worldToCameraTranslation();
        const std::array<double, 3> center = camera.cameraCenter();
        return QJsonObject{{QStringLiteral("fx"), intrinsics.focalX},
                           {QStringLiteral("fy"), intrinsics.focalY},
                           {QStringLiteral("cx"), intrinsics.principalX},
                           {QStringLiteral("cy"), intrinsics.principalY},
                           {QStringLiteral("rotation_world_to_camera"), doubleArrayToJson(rotation.data(), 9)},
                           {QStringLiteral("translation_world_to_camera"), doubleArrayToJson(translation.data(), 3)},
                           {QStringLiteral("camera_center"), doubleArrayToJson(center.data(), 3)}};
    }

    QJsonObject depthPoseRefinementCandidateToJson(const DepthPoseRefinementCandidate& candidate,
                                                   const DepthPoseRefinementStageResult& stage)
    {
        QJsonArray pivot;
        QJsonArray translation;
        QJsonArray rotation;
        for (int index = 0; index < 3; ++index)
        {
            pivot.append(candidate.correction.pivotWorld[index]);
            translation.append(candidate.correction.translation[index]);
        }
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                rotation.append(candidate.correction.rotation(row, column));
            }
        }
        return QJsonObject{{QStringLiteral("enabled"), stage.enabled},
                           {QStringLiteral("candidate_only"), stage.candidateOnly},
                           {QStringLiteral("application_status"), QStringLiteral("not_applied_candidate_only")},
                           {QStringLiteral("scale_locked"), true},
                           {QStringLiteral("anchor_camera_index"), stage.anchorCameraIndex},
                           {QStringLiteral("camera_index"), candidate.cameraIndex},
                           {QStringLiteral("evidence_complete"), candidate.evidenceComplete},
                           {QStringLiteral("accepted"), candidate.accepted},
                           {QStringLiteral("reason"), QString::fromStdString(candidate.reason)},
                           {QStringLiteral("evidence_pixel_count"), candidate.evidencePixelCount},
                           {QStringLiteral("correspondence_count"), candidate.generatedCorrespondenceCount},
                           {QStringLiteral("occluded_candidate_count"), candidate.occludedCandidateCount},
                           {QStringLiteral("depth_conflict_candidate_count"), candidate.depthConflictCandidateCount},
                           {QStringLiteral("evidence_sample_coverage"), candidate.evidenceSampleCoverage},
                           {QStringLiteral("projection_retention_ratio"), candidate.projectionRetentionRatio},
                           {QStringLiteral("translation_norm"), candidate.correctionTranslation},
                           {QStringLiteral("rotation_degrees"), candidate.correctionRotationDegrees},
                           {QStringLiteral("residual_median_before"), candidate.correction.residualMedianBefore},
                           {QStringLiteral("residual_median_after"), candidate.correction.residualMedianAfter},
                           {QStringLiteral("residual_p90_before"), candidate.correction.residualP90Before},
                           {QStringLiteral("residual_p90_after"), candidate.correction.residualP90After},
                           {QStringLiteral("correction_pivot_world"), pivot},
                           {QStringLiteral("correction_translation_world"), translation},
                           {QStringLiteral("correction_rotation_world"), rotation},
                           {QStringLiteral("rotation_mapping"), QStringLiteral("R_wc'=R_wc*Q^T; stored R_cw'=Q*R_cw")}};
    }

    QJsonArray depthPyramidLevelsToJson(const std::vector<DepthLevelSummary>& summaries)
    {
        QJsonArray array;
        for (const DepthLevelSummary& summary : summaries)
        {
            QJsonObject object;
            object.insert(QStringLiteral("level"), summary.level);
            object.insert(QStringLiteral("downsample_factor"), summary.downsampleFactor);
            object.insert(QStringLiteral("valid_pixel_count"), summary.validPixelCount);
            object.insert(QStringLiteral("valid_coverage"), summary.validCoverage);
            object.insert(QStringLiteral("mean_confidence"), summary.meanConfidence);
            object.insert(QStringLiteral("mean_support_views"), summary.meanSupportViews);
            object.insert(QStringLiteral("depth_discontinuity_ratio"), summary.depthDiscontinuityRatio);
            object.insert(QStringLiteral("elapsed_ms"), summary.elapsedMs);
            object.insert(QStringLiteral("success"), summary.success);
            object.insert(QStringLiteral("error"), QString::fromStdString(summary.errorMessage));
            array.append(object);
        }
        return array;
    }

    QString sceneProfileId(MvsSceneProfile profile)
    {
        switch (profile)
        {
        case MvsSceneProfile::AerialTerrain:
            return QStringLiteral("aerial_terrain");
        case MvsSceneProfile::OrbitalObject:
            return QStringLiteral("orbital_object");
        case MvsSceneProfile::Custom:
            return QStringLiteral("custom");
        case MvsSceneProfile::Auto:
        default:
            return QStringLiteral("auto");
        }
    }

    QString depthFilterModeId(DepthFilterMode mode)
    {
        switch (mode)
        {
        case DepthFilterMode::Mild:
            return QStringLiteral("mild");
        case DepthFilterMode::Aggressive:
            return QStringLiteral("aggressive");
        case DepthFilterMode::Moderate:
        default:
            return QStringLiteral("moderate");
        }
    }

    QJsonObject depthPostProcessStatsToJson(const DepthPostProcessStats& stats)
    {
        QJsonObject object;
        object.insert(QStringLiteral("valid_before"), stats.validBeforePostprocess);
        object.insert(QStringLiteral("valid_after_confidence_filter"), stats.validAfterConfidenceFilter);
        object.insert(QStringLiteral("low_confidence_candidate_count"), stats.lowConfidenceCandidateCount);
        object.insert(QStringLiteral("geometry_supported_low_confidence_retained"),
                      stats.geometrySupportedLowConfidenceRetained);
        object.insert(QStringLiteral("independent_geometry_confidence_retained"),
                      stats.independentGeometryConfidenceRetained);
        object.insert(QStringLiteral("boundary_geometry_retained"), stats.boundaryGeometryRetained);
        object.insert(QStringLiteral("confidence_removed"), stats.confidenceRemoved);
        object.insert(QStringLiteral("local_depth_outlier_removed"), stats.localDepthOutlierRemoved);
        object.insert(QStringLiteral("small_component_removed"), stats.smallComponentRemoved);
        object.insert(QStringLiteral("speckle_removed"), stats.speckleRemoved);
        object.insert(QStringLiteral("edge_confidence_removed"), stats.edgeConfidenceRemoved);
        object.insert(QStringLiteral("geom_consistency_removed"), stats.geomConsistencyRemoved);
        object.insert(QStringLiteral("valid_after"), stats.validAfterPostprocess);
        object.insert(QStringLiteral("effective_confidence_threshold"), stats.effectiveConfidenceThreshold);
        return object;
    }

    SourceQualitySummary summarizeSourceQuality(const QJsonArray& sourcePlan, int fallbackSourceViewCount)
    {
        SourceQualitySummary summary;
        summary.sourceViewCount = std::max(0, fallbackSourceViewCount);

        double qualitySum = 0.0;
        double minQuality = std::numeric_limits<double>::max();
        int qualityCount = 0;
        for (const QJsonValue& value : sourcePlan)
        {
            if (!value.isObject())
            {
                continue;
            }

            const QJsonObject entry = value.toObject();
            const QString sourceTier = entry.value(QStringLiteral("source_tier")).toString();
            if (sourceTier == QStringLiteral("verified_pair"))
            {
                ++summary.verifiedSourceViewCount;
            }
            else if (sourceTier == QStringLiteral("track_geometry_backfill"))
            {
                ++summary.backfillSourceViewCount;
            }
            else if (sourceTier == QStringLiteral("strict_pair_audit_backfill"))
            {
                ++summary.backfillSourceViewCount;
            }
            else if (sourceTier == QStringLiteral("sequence_fallback"))
            {
                ++summary.sequenceFallbackSourceViewCount;
            }

            const QJsonValue qualityValue = entry.value(QStringLiteral("source_quality_score"));
            if (!qualityValue.isDouble())
            {
                continue;
            }

            const double quality = std::clamp(qualityValue.toDouble(), 0.0, 1.0);
            qualitySum += quality;
            minQuality = std::min(minQuality, quality);
            ++qualityCount;
        }

        if (qualityCount > 0)
        {
            summary.meanQuality = qualitySum / static_cast<double>(qualityCount);
            summary.minQuality = minQuality;
        }
        return summary;
    }

    DepthConfidenceSummary summarizeDepthConfidence(const cv::Mat& depthMap, const cv::Mat* confidenceMap)
    {
        DepthConfidenceSummary summary;
        if (depthMap.empty() || depthMap.type() != CV_32F)
        {
            return summary;
        }

        const cv::Mat validMask = depthMap > 0.0f;
        summary.validPixelCount = cv::countNonZero(validMask);
        if (summary.validPixelCount <= 0 || !confidenceMap || confidenceMap->empty() ||
            confidenceMap->size() != depthMap.size() || confidenceMap->type() != CV_32F)
        {
            return summary;
        }

        summary.meanConfidence = std::clamp(cv::mean(*confidenceMap, validMask)[0], 0.0, 1.0);
        return summary;
    }
} // namespace xjw::mvs::pipeline_detail
