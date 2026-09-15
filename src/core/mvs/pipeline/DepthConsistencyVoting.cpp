#include "MvsPipelineInternals.h"

namespace xjw::mvs::pipeline_detail
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    void accumulateDepthConsistency(const cv::Mat& referenceDepth,
                                    const FramePinholeCamera& referenceCamera,
                                    const std::vector<DepthConsistencySourceInput>& sources,
                                    float relativeThreshold,
                                    float maximumRoundTripErrorPixels,
                                    int rowWorkers,
                                    const std::atomic<bool>& cancelled,
                                    cv::Mat& consistentVotes,
                                    cv::Mat& occludedVotes,
                                    cv::Mat& contradictedVotes,
                                    cv::Mat& unverifiableVotes,
                                    cv::Mat& geometrySourceMask,
                                    cv::Mat& sourceInverseDepthSum,
                                    cv::Mat& sourceInverseDepthSquaredSum,
                                    AdaptiveGeometryEvidenceAccumulatorMaps* adaptiveEvidence)
    {
        if (sources.empty())
        {
            return;
        }
        const bool accumulate_adaptive_evidence =
            adaptiveEvidence && adaptiveEvidence->positiveSupport.type() == CV_32FC1 &&
            adaptiveEvidence->squaredPositiveSupport.type() == CV_32FC1 &&
            adaptiveEvidence->conflict.type() == CV_32FC1 && adaptiveEvidence->observable.type() == CV_32FC1 &&
            adaptiveEvidence->positiveSupport.size() == referenceDepth.size() &&
            adaptiveEvidence->squaredPositiveSupport.size() == referenceDepth.size() &&
            adaptiveEvidence->conflict.size() == referenceDepth.size() &&
            adaptiveEvidence->observable.size() == referenceDepth.size();
        parallelForRows(
            referenceDepth.rows,
            rowWorkers,
            [&](int row)
            {
                if (cancelled.load())
                {
                    return;
                }
                const float* reference_row = referenceDepth.ptr<float>(row);
                uint16_t* consistent_row = consistentVotes.ptr<uint16_t>(row);
                uint16_t* occluded_row = occludedVotes.ptr<uint16_t>(row);
                uint16_t* contradicted_row = contradictedVotes.ptr<uint16_t>(row);
                uint16_t* unverifiable_row = unverifiableVotes.ptr<uint16_t>(row);
                uint16_t* source_mask_row = geometrySourceMask.ptr<uint16_t>(row);
                float* inverse_sum_row = sourceInverseDepthSum.ptr<float>(row);
                float* inverse_squared_sum_row = sourceInverseDepthSquaredSum.ptr<float>(row);
                float* adaptive_positive_row =
                    accumulate_adaptive_evidence ? adaptiveEvidence->positiveSupport.ptr<float>(row) : nullptr;
                float* adaptive_squared_row =
                    accumulate_adaptive_evidence ? adaptiveEvidence->squaredPositiveSupport.ptr<float>(row) : nullptr;
                float* adaptive_conflict_row =
                    accumulate_adaptive_evidence ? adaptiveEvidence->conflict.ptr<float>(row) : nullptr;
                float* adaptive_observable_row =
                    accumulate_adaptive_evidence ? adaptiveEvidence->observable.ptr<float>(row) : nullptr;
                for (int column = 0; column < referenceDepth.cols; ++column)
                {
                    if (cancelled.load())
                    {
                        return;
                    }
                    const float reference_depth = reference_row[column];
                    if (reference_depth <= 0.0f)
                    {
                        continue;
                    }
                    const cv::Point2f reference_pixel(static_cast<float>(column), static_cast<float>(row));
                    const double pixel[2] = {static_cast<double>(reference_pixel.x),
                                             static_cast<double>(reference_pixel.y)};
                    double world[3] = {0.0, 0.0, 0.0};
                    if (!referenceCamera.unprojectPixel(pixel, reference_depth, world))
                    {
                        // Preserve the former per-source evaluation semantics on the
                        // exceptional unprojection failure path: every source would
                        // have contributed one unverifiable observation.
                        unverifiable_row[column] =
                            static_cast<std::uint16_t>(unverifiable_row[column] + sources.size());
                        continue;
                    }
                    const std::array<double, 3> reference_world = {world[0], world[1], world[2]};

                    // Preserve the source-plan order for each pixel.  This keeps the
                    // exact floating accumulation order while amortizing reference
                    // unprojection and row-worker dispatch across all source views.
                    for (const DepthConsistencySourceInput& source : sources)
                    {
                        const ProjectedDepthConsistencyResult result =
                            evaluateProjectedDepthConsistencyFromReferenceWorld(referenceCamera,
                                                                                reference_pixel,
                                                                                reference_depth,
                                                                                reference_world,
                                                                                source.camera,
                                                                                source.depth,
                                                                                relativeThreshold,
                                                                                source.searchRadiusPixels,
                                                                                maximumRoundTripErrorPixels,
                                                                                accumulate_adaptive_evidence,
                                                                                source.evaluateSubpixelFootprint);
                        if (accumulate_adaptive_evidence)
                        {
                            AdaptiveGeometryEvidenceObservation observation;
                            observation.worldResidual = result.worldSurfaceResidual;
                            observation.worldPixelFootprint = result.jointWorldPixelFootprint;
                            observation.roundTripResidualPixels = result.roundTripErrorPixels;
                            observation.reliabilityWeight = std::clamp(source.reliabilityWeight, 0.0f, 1.0f);
                            if (!source.confidence.empty() && source.confidence.type() == CV_32FC1 &&
                                source.confidence.size() == source.depth.size() && result.sourcePixel.x >= 0 &&
                                result.sourcePixel.x < source.confidence.cols && result.sourcePixel.y >= 0 &&
                                result.sourcePixel.y < source.confidence.rows)
                            {
                                const float source_confidence =
                                    source.confidence.at<float>(result.sourcePixel.y, result.sourcePixel.x);
                                observation.reliabilityWeight *=
                                    std::isfinite(source_confidence) ? std::clamp(source_confidence, 0.0f, 1.0f) : 0.0f;
                            }
                            observation.evidenceClass = adaptiveGeometryEvidenceClass(result);
                            AdaptiveGeometryEvidenceAccumulator accumulator;
                            accumulator.positiveSupport = adaptive_positive_row[column];
                            accumulator.squaredPositiveSupport = adaptive_squared_row[column];
                            accumulator.conflict = adaptive_conflict_row[column];
                            accumulator.observable = adaptive_observable_row[column];
                            accumulator.add(observation);
                            adaptive_positive_row[column] = accumulator.positiveSupport;
                            adaptive_squared_row[column] = accumulator.squaredPositiveSupport;
                            adaptive_conflict_row[column] = accumulator.conflict;
                            adaptive_observable_row[column] = accumulator.observable;
                        }
                        switch (result.evidence)
                        {
                        case DepthConsistencyEvidence::Consistent:
                            ++consistent_row[column];
                            if (source.sourceOrdinal >= 0 && source.sourceOrdinal < 16)
                            {
                                source_mask_row[column] = static_cast<std::uint16_t>(
                                    source_mask_row[column] | (static_cast<std::uint16_t>(1U) << source.sourceOrdinal));
                            }
                            if (result.consistentReferenceDepth > 0.0f &&
                                std::isfinite(result.consistentReferenceDepth))
                            {
                                const float inverse_depth = 1.0f / result.consistentReferenceDepth;
                                inverse_sum_row[column] += inverse_depth;
                                inverse_squared_sum_row[column] += inverse_depth * inverse_depth;
                            }
                            break;
                        case DepthConsistencyEvidence::Occluded:
                            ++occluded_row[column];
                            break;
                        case DepthConsistencyEvidence::Contradicted:
                            ++contradicted_row[column];
                            break;
                        case DepthConsistencyEvidence::Unverifiable:
                        default:
                            ++unverifiable_row[column];
                            break;
                        }
                    }
                }
            });
    }

    cv::Mat makeDepthConsistencyMask(const cv::Mat& referenceDepth,
                                     int sourceViewCount,
                                     int minimumSourceConfirmations,
                                     const cv::Mat& consistentVotes,
                                     const cv::Mat& occludedVotes,
                                     const cv::Mat& contradictedVotes,
                                     int rowWorkerCount,
                                     const std::atomic<bool>* cancelled)
    {
        cv::Mat mask(referenceDepth.size(), CV_8U, cv::Scalar(0));
        parallelForRows(referenceDepth.rows,
                        rowWorkerCount,
                        [&](int row)
                        {
                            if (cancelled && cancelled->load(std::memory_order_relaxed))
                            {
                                return;
                            }
                            const float* depth_row = referenceDepth.ptr<float>(row);
                            const uint16_t* consistent_row = consistentVotes.ptr<uint16_t>(row);
                            const uint16_t* occluded_row = occludedVotes.ptr<uint16_t>(row);
                            const uint16_t* contradicted_row = contradictedVotes.ptr<uint16_t>(row);
                            uint8_t* mask_row = mask.ptr<uint8_t>(row);
                            for (int column = 0; column < referenceDepth.cols; ++column)
                            {
                                if ((column & 63) == 0 && cancelled && cancelled->load(std::memory_order_relaxed))
                                {
                                    break;
                                }
                                if (depth_row[column] <= 0.0f)
                                {
                                    continue;
                                }
                                if (shouldRetainDepthFromConsistencyVotes(sourceViewCount,
                                                                          consistent_row[column],
                                                                          occluded_row[column],
                                                                          contradicted_row[column],
                                                                          minimumSourceConfirmations))
                                {
                                    mask_row[column] = 255;
                                }
                            }
                        });
        return mask;
    }

    DepthConsistencyVoteTotals summarizeDepthConsistencyVotes(const cv::Mat& consistentVotes,
                                                              const cv::Mat& occludedVotes,
                                                              const cv::Mat& contradictedVotes,
                                                              const cv::Mat& unverifiableVotes,
                                                              int rowWorkerCount)
    {
        std::array<std::atomic<std::uint64_t>, 4> totals{};
        parallelForRows(consistentVotes.rows,
                        rowWorkerCount,
                        [&](int row)
                        {
                            const std::uint16_t* consistent_row = consistentVotes.ptr<std::uint16_t>(row);
                            const std::uint16_t* occluded_row = occludedVotes.ptr<std::uint16_t>(row);
                            const std::uint16_t* contradicted_row = contradictedVotes.ptr<std::uint16_t>(row);
                            const std::uint16_t* unverifiable_row = unverifiableVotes.ptr<std::uint16_t>(row);
                            std::array<std::uint64_t, 4> row_totals{};
                            for (int column = 0; column < consistentVotes.cols; ++column)
                            {
                                row_totals[0] += consistent_row[column];
                                row_totals[1] += occluded_row[column];
                                row_totals[2] += contradicted_row[column];
                                row_totals[3] += unverifiable_row[column];
                            }
                            for (std::size_t index = 0; index < totals.size(); ++index)
                            {
                                totals[index].fetch_add(row_totals[index], std::memory_order_relaxed);
                            }
                        });
        return {totals[0].load(std::memory_order_relaxed),
                totals[1].load(std::memory_order_relaxed),
                totals[2].load(std::memory_order_relaxed),
                totals[3].load(std::memory_order_relaxed)};
    }

    DepthAnchoredHoleInterpolationStats repairPostprocessedInternalDepthHoles(DepthFrameResult& result,
                                                                              cv::Mat& depth,
                                                                              cv::Mat& confidence,
                                                                              MvsSceneProfile sceneProfile,
                                                                              cv::Mat* anchoredInterpolationMask)
    {
        if (sceneProfile != MvsSceneProfile::OrbitalObject || !result.supportRegionMask ||
            result.supportRegionMask->empty() || !result.crossViewRepairedMask || result.crossViewRepairedMask->empty())
        {
            return {};
        }
        cv::Mat support_mask = *result.supportRegionMask;
        if (support_mask.size() != depth.size())
        {
            cv::resize(support_mask, support_mask, depth.size(), 0.0, 0.0, cv::INTER_NEAREST);
        }
        cv::Mat anchor_mask = depth > 0.0f;
        if (result.crossViewRepairedMask->size() != depth.size())
        {
            cv::resize(*result.crossViewRepairedMask,
                       *result.crossViewRepairedMask,
                       depth.size(),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
        }
        DepthAnchoredHoleInterpolationOptions options;
        options.enabled = true;
        options.maximumComponentArea = 32000;
        options.maximumComponentAreaRatio = 0.25f;
        options.allowSilhouetteConnectedInterior = true;
        options.silhouetteProtectionRadiusPixels = 4;
        cv::Mat local_interpolation_mask;
        cv::Mat* interpolation_mask = anchoredInterpolationMask ? anchoredInterpolationMask : &local_interpolation_mask;
        *interpolation_mask = cv::Mat(depth.size(), CV_8UC1, cv::Scalar(0));
        const DepthAnchoredHoleInterpolationStats stats =
            interpolateAnchoredInternalDepthHoles(depth,
                                                  support_mask,
                                                  anchor_mask,
                                                  nullptr,
                                                  options,
                                                  confidence.empty() ? nullptr : &confidence,
                                                  interpolation_mask);
        cv::bitwise_or(*result.crossViewRepairedMask, *interpolation_mask, *result.crossViewRepairedMask);
        return stats;
    }

    void updateDepthFrameQualityAfterConsistency(DepthFrameResult& result,
                                                 const cv::Mat& depth,
                                                 const cv::Mat& confidence,
                                                 MvsSceneProfile scene_profile,
                                                 DepthFilterMode filter_mode,
                                                 bool consistency_stage_expected,
                                                 const AdaptiveGeometryEvidenceSummary& adaptive_summary,
                                                 const DiscreteGeometryCoreSummary& discrete_summary)
    {
        // Frame acceptance describes the final depth product that enters fusion.
        // Repaired pixels retain their per-pixel provenance and are guarded again by
        // TSDF geometry support, so removing them here would penalize the same
        // evidence twice and could downgrade an otherwise complete orbital sequence.
        const cv::Mat& quality_depth = depth;
        const cv::Mat& quality_confidence = confidence;
        const float search_boundary_ratio = result.qualityMetrics.depthAtSearchBoundaryRatio;
        result.qualityMetrics = analyzeDepthMapQuality(
            quality_depth, quality_confidence, static_cast<int>(result.sourceViewIndices.size()));
        result.qualityMetrics.depthAtSearchBoundaryRatio = search_boundary_ratio;

        DepthFrameQualityInput quality_input;
        quality_input.sceneProfile = scene_profile;
        quality_input.filterMode = filter_mode;
        quality_input.sourceViewCount = static_cast<int>(result.sourceViewIndices.size());
        quality_input.validCoverage = result.qualityMetrics.validCoverage;
        quality_input.largestComponentRatio = result.qualityMetrics.largestComponentRatio;
        quality_input.meanConfidence = result.qualityMetrics.meanConfidence;
        quality_input.dualChannelConfidenceAvailable = result.evidenceConfidenceSummary.available;
        quality_input.meanPhotometricConfidence = result.evidenceConfidenceSummary.meanPhotometricConfidence;
        quality_input.meanIndependentGeometryConfidence = result.evidenceConfidenceSummary.meanGeometricConfidence;
        quality_input.strongIndependentGeometryCoverage = result.evidenceConfidenceSummary.strongGeometryCoverage;
        quality_input.geometryEvidenceTreatmentEnabled =
            result.geometryRerankMaps && result.evidenceConfidenceSummary.available;
        quality_input.geometryCorrectedPixelCount = result.evidenceConfidenceSummary.correctedPixelCount;
        quality_input.correctedMeanIndependentGeometryConfidence =
            result.evidenceConfidenceSummary.correctedMeanGeometricConfidence;
        quality_input.depthAtSearchBoundaryRatio = search_boundary_ratio;
        quality_input.hasProjectSupportMask = result.maskSource == "project" && result.maskCoverage < 0.999f;
        if (result.supportRegionMask && !result.supportRegionMask->empty())
        {
            cv::Mat effective_mask = *result.supportRegionMask;
            if (effective_mask.size() != quality_depth.size())
            {
                cv::resize(effective_mask, effective_mask, quality_depth.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            result.depthCompleteness.finalMetrics =
                analyzeDepthCompleteness(quality_depth,
                                         effective_mask,
                                         kSmallHoleAreaFraction,
                                         effectiveMinimumSmallHoleArea(result, quality_depth.size()));
            quality_input.validWithinMaskRatio = result.depthCompleteness.finalMetrics.validInputs
                                                     ? result.depthCompleteness.finalMetrics.validWithinMaskRatio
                                                     : -1.0f;
        }
        quality_input.outputFilterRetentionRatio = result.depthCompleteness.outputFilterRetentionRatio;
        const DepthConsistencyPublicationSummary consistency_publication =
            summarizeDepthConsistencyPublication(result.depthCompleteness.preConsistencyValidCount,
                                                 result.depthCompleteness.postConsistencyValidCount,
                                                 result.depthCompleteness.publishedPostConsistencyValidCount,
                                                 result.depthCompleteness.consistencyPublicationFallbackApplied,
                                                 consistency_stage_expected);
        applyDepthConsistencyPublicationSummary(consistency_publication, &quality_input);
        quality_input.fusionPostprocessRetentionRatio = result.depthCompleteness.fusionPostprocessRetentionRatio;
        quality_input.adaptiveGeometryEvidenceAvailable =
            adaptive_summary.validInputs && adaptive_summary.observablePixelCount > 0 &&
            std::isfinite(adaptive_summary.effectiveViewCountMean) && adaptive_summary.effectiveViewCountMean >= 0.0f &&
            std::isfinite(adaptive_summary.conflictRatioMean) && adaptive_summary.conflictRatioMean >= 0.0f;
        quality_input.adaptiveEffectiveViewCountMean = adaptive_summary.effectiveViewCountMean;
        quality_input.adaptiveConflictRatioMean = adaptive_summary.conflictRatioMean;
        quality_input.discreteGeometryCoreAvailable =
            discrete_summary.validInputs && discrete_summary.validPixelCount > 0 &&
            std::isfinite(discrete_summary.coreRatio) && discrete_summary.coreRatio >= 0.0f;
        quality_input.discreteGeometryCoreRatio = discrete_summary.coreRatio;
        quality_input.completeVisibilityCandidatePoolEnabled = result.completeVisibilityCandidatePoolEnabled;
        quality_input.completePoolChangedLegacyPlan = result.completePoolChangedLegacyPlan;
        quality_input.initialAcceptanceAvailable = result.initialQualityAcceptanceAvailable;
        quality_input.initialAcceptance = result.initialQualityAcceptance;
        result.sparseDepthResidual =
            summarizeSparseDepthResidual(quality_depth,
                                         result.projectedSparseDepthSamples,
                                         effectiveSparseResidualRadius(result, quality_depth.size()));
        quality_input.sparseDepthResidual = result.sparseDepthResidual;
        result.qualityDecision = evaluateDepthFrame(quality_input);
        detail::applySourceAngleCapShortfallSafety(result);
    }
} // namespace xjw::mvs::pipeline_detail
