#include "MvsPipelineInternals.h"

namespace xjw::mvs::pipeline_detail
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    DepthPixelDomainScale pixelDomainScaleForResult(const DepthFrameResult& result, const cv::Size& grid_size)
    {
        const cv::Size raster_size = result.preparedRasterSize.width > 0 && result.preparedRasterSize.height > 0
                                         ? result.preparedRasterSize
                                         : grid_size;
        return depthPixelDomainScale(raster_size, grid_size);
    }

    int effectiveMinimumSmallHoleArea(const DepthFrameResult& result, const cv::Size& grid_size)
    {
        return scaleDepthPixelArea(kFullRasterMinimumSmallHoleArea, pixelDomainScaleForResult(result, grid_size));
    }

    int effectiveSparseResidualRadius(const DepthFrameResult& result, const cv::Size& grid_size)
    {
        return scaleDepthPixelRadius(kFullRasterSparseResidualRadiusPixels,
                                     pixelDomainScaleForResult(result, grid_size));
    }

    PreRepairDepthLayerReliability
    analyzePreRepairDepthLayerReliability(const DepthFrameResult& frame,
                                          const cv::Mat& depth,
                                          const cv::Mat* reference_gray,
                                          const cv::Mat& consistent_votes,
                                          const cv::Mat& occluded_votes,
                                          const cv::Mat& contradicted_votes,
                                          const cv::Mat& geometry_source_mask,
                                          const cv::Mat& source_inverse_depth_sum,
                                          const cv::Mat& source_inverse_depth_squared_sum,
                                          const AdaptiveGeometryEvidenceAccumulatorMaps* adaptive_accumulator)
    {
        const GeometryEvidenceMaps discrete_evidence = makeGeometryEvidenceMaps(
            depth, consistent_votes, geometry_source_mask, source_inverse_depth_sum, source_inverse_depth_squared_sum);
        cv::Mat effective_view_count;
        cv::Mat conflict_ratio;
        QString evidence_source = QStringLiteral("discrete_vote_proxy");
        if (adaptive_accumulator)
        {
            const AdaptiveGeometryEvidenceMaps adaptive_evidence =
                makeAdaptiveGeometryEvidenceMaps(depth, *adaptive_accumulator);
            effective_view_count = adaptive_evidence.effectiveViewCount;
            conflict_ratio = adaptive_evidence.conflictRatio;
            evidence_source = QStringLiteral("adaptive_weighted_geometry");
        }
        else
        {
            consistent_votes.convertTo(effective_view_count, CV_32FC1);
            conflict_ratio = cv::Mat(depth.size(), CV_32FC1, cv::Scalar(0.0f));
            for (int y = 0; y < depth.rows; ++y)
            {
                const std::uint16_t* consistent_row = consistent_votes.ptr<std::uint16_t>(y);
                const std::uint16_t* occluded_row = occluded_votes.ptr<std::uint16_t>(y);
                const std::uint16_t* contradicted_row = contradicted_votes.ptr<std::uint16_t>(y);
                float* conflict_row = conflict_ratio.ptr<float>(y);
                for (int x = 0; x < depth.cols; ++x)
                {
                    const int observed = consistent_row[x] + occluded_row[x] + contradicted_row[x];
                    if (observed > 0)
                    {
                        conflict_row[x] = static_cast<float>(contradicted_row[x]) / static_cast<float>(observed);
                    }
                }
            }
        }

        DepthLayerReliabilityOptions options;
        const DepthPixelDomainScale pixel_scale = pixelDomainScaleForResult(frame, depth.size());
        options.textureRadiusPixels = std::max(1, scaleDepthPixelRadius(options.textureRadiusPixels, pixel_scale));
        options.boundaryRingRadiusPixels =
            std::max(1, scaleDepthPixelRadius(options.boundaryRingRadiusPixels, pixel_scale));
        options.minimumComponentArea = std::max(1, scaleDepthPixelArea(options.minimumComponentArea, pixel_scale));
        options.minimumBoundaryAnchorCount =
            std::max(3, scaleDepthPixelArea(options.minimumBoundaryAnchorCount, pixel_scale));

        PreRepairDepthLayerReliability observation;
        observation.result = analyzeDepthLayerReliability(depth,
                                                          reference_gray ? *reference_gray : cv::Mat(),
                                                          effective_view_count,
                                                          conflict_ratio,
                                                          discrete_evidence.inverseDepthRelativeSpread,
                                                          options);
        observation.diagnostics = depthLayerReliabilityDiagnosticsToJson(observation.result, options);
        observation.diagnostics.insert(
            QStringLiteral("classification_boundary"),
            QStringLiteral("after_cross_view_vote_accumulation_before_layer_selection_or_repair"));
        observation.diagnostics.insert(QStringLiteral("geometry_evidence_source"), evidence_source);
        observation.diagnostics.insert(QStringLiteral("configured_full_raster_texture_radius_pixels"), 3);
        observation.diagnostics.insert(QStringLiteral("configured_full_raster_boundary_ring_radius_pixels"), 6);
        return observation;
    }

    void calibrateFinalDepthConfidenceMap(cv::Mat& confidence_map, const cv::Mat* depth_provenance)
    {
        if (confidence_map.empty() || confidence_map.type() != CV_32FC1)
        {
            return;
        }

#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int row = 0; row < confidence_map.rows; ++row)
        {
            float* confidence_row = confidence_map.ptr<float>(row);
            const std::uint8_t* provenance_row = depth_provenance && depth_provenance->type() == CV_8UC1 &&
                                                         depth_provenance->size() == confidence_map.size()
                                                     ? depth_provenance->ptr<std::uint8_t>(row)
                                                     : nullptr;
            for (int column = 0; column < confidence_map.cols; ++column)
            {
                if (provenance_row &&
                    provenance_row[column] == static_cast<std::uint8_t>(DepthProvenance::LearnedGeometryGated))
                {
                    continue;
                }
                confidence_row[column] = calibrateRobustPhotometricConfidence(confidence_row[column]);
            }
        }
    }

    DepthPostProcessEvidence updateDepthEvidenceConfidence(DepthFrameResult& frame,
                                                           const cv::Mat& depth,
                                                           cv::Mat& confidence,
                                                           DepthPostProcessEvidence evidence,
                                                           bool enabled)
    {
        if (!enabled || depth.empty() || confidence.empty())
        {
            return evidence;
        }
        const DepthEvidenceConfidenceResult confidence_result =
            buildDepthEvidenceConfidence(depth,
                                         confidence,
                                         evidence.geometrySupportCount,
                                         evidence.inverseDepthRelativeSpread,
                                         evidence.adaptiveSupportWeight,
                                         evidence.adaptiveEffectiveViewCount,
                                         evidence.adaptiveConflictRatio,
                                         frame.geometryRerankMaps.data());
        frame.evidenceConfidenceDiagnostics = depthEvidenceConfidenceSummaryToJson(confidence_result.summary);
        frame.evidenceConfidenceSummary = confidence_result.summary;
        if (!confidence_result.summary.available)
        {
            return evidence;
        }
        frame.photometricConfidence = QSharedPointer<cv::Mat>::create(confidence_result.photometric);
        frame.geometricConfidence = QSharedPointer<cv::Mat>::create(confidence_result.geometric);
        confidence = confidence_result.combined;
        evidence.photometricConfidence = confidence_result.photometric;
        evidence.geometricConfidence = confidence_result.geometric;
        return evidence;
    }

    int maximumConsistencySourceViews(const DepthGenConfig& config)
    {
        return std::max({config.numSourceViews,
                         config.patchMatch.numSourceViews,
                         config.crossViewHoleRepairSourceCount,
                         config.postConsistencyResidualSourceCount});
    }

    CrossViewHoleRepairOptions orbitalCrossViewHoleRepairOptions(const DepthGenConfig& config)
    {
        CrossViewHoleRepairOptions options;
        options.minimumDistinctSourceCount = 2;
        options.maximumRelativeDepthSpread = 0.010f;
        options.maximumProjectionDistancePixels = 0.8f;
        options.maximumLocalRelativeDepthDifference = 0.035f;
        options.repairedConfidence = 0.70f;
        options.enableTwoSourceGrowth = config.enableTwoSourceCrossViewGrowth;
        options.maximumGrowthDistancePixels = config.twoSourceGrowthDistancePixels;
        options.maximumGrowthInverseDepthSpread = config.twoSourceGrowthInverseDepthSpread;
        options.maximumGrowthNormalAngleDegrees = config.twoSourceGrowthNormalAngleDegrees;
        options.maximumGrowthComponentArea = config.twoSourceGrowthMaximumComponentArea;
        options.includeValidNativeInterpolationAnchors = true;
        options.anchoredInterpolation.enabled = true;
        options.anchoredInterpolation.maximumComponentArea = 32000;
        options.anchoredInterpolation.maximumComponentAreaRatio = 0.25f;
        options.anchoredInterpolation.allowSilhouetteConnectedInterior = true;
        options.anchoredInterpolation.silhouetteProtectionRadiusPixels = 4;
        return options;
    }

    std::vector<int> orbitalHoleRepairSourceIndices(const std::vector<DepthFrameResult>& frames,
                                                    const std::vector<int>& consistencySources,
                                                    int refIdx,
                                                    int viewCount,
                                                    int requestedSourceCount)
    {
        std::vector<bool> source_eligibility(static_cast<std::size_t>(std::max(0, viewCount)), false);
        const int available_frame_count = std::min(viewCount, static_cast<int>(frames.size()));
        for (int frame_index = 0; frame_index < available_frame_count; ++frame_index)
        {
            source_eligibility[static_cast<std::size_t>(frame_index)] =
                frames[static_cast<std::size_t>(frame_index)].eligibleAsConsistencySource();
        }
        return planMvsRepairSourceViews(consistencySources, source_eligibility, refIdx, requestedSourceCount);
    }

    float sourceGeometryReliabilityWeight(const DepthFrameResult& reference_frame, int source_view_index)
    {
        const auto entry = std::find_if(reference_frame.sourceViewPlan.cbegin(),
                                        reference_frame.sourceViewPlan.cend(),
                                        [source_view_index](const MvsSourcePlanEntry& candidate)
                                        { return candidate.viewIndex == source_view_index; });
        if (entry == reference_frame.sourceViewPlan.cend() || !std::isfinite(entry->sourceQualityScore) ||
            entry->sourceQualityScore <= 0.0f)
        {
            return 1.0f;
        }
        return std::clamp(entry->sourceQualityScore, 0.05f, 1.0f);
    }

    int cameraBaselineSector(const FramePinholeCamera& reference_camera, const FramePinholeCamera& source_camera)
    {
        const std::array<double, 3> reference_center = reference_camera.cameraCenter();
        const std::array<double, 3> source_center = source_camera.cameraCenter();
        std::array<double, 3> delta{source_center[0] - reference_center[0],
                                    source_center[1] - reference_center[1],
                                    source_center[2] - reference_center[2]};
        int dominant_axis = 0;
        for (int axis = 1; axis < 3; ++axis)
        {
            if (std::fabs(delta[static_cast<std::size_t>(axis)]) >
                std::fabs(delta[static_cast<std::size_t>(dominant_axis)]))
            {
                dominant_axis = axis;
            }
        }
        return dominant_axis * 2 + (delta[static_cast<std::size_t>(dominant_axis)] >= 0.0 ? 1 : 0);
    }
} // namespace xjw::mvs::pipeline_detail
