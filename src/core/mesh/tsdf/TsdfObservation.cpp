#include "DepthTsdfInternals.h"
namespace xjw::mesh
{
    using namespace tsdf_detail;

    DepthTsdfObservationSample
    DepthTsdfSurfaceBuilder::sampleObservation(const DepthTsdfFrame& frame,
                                               const cv::Mat& effectiveDepthValidMask,
                                               const cv::Point2d& pixel,
                                               float minimumConfidence,
                                               bool discontinuityAware,
                                               float maximumRelativeDepthSpread,
                                               float maximumObservationInverseDepthSpread,
                                               bool allowInvalidNearestPixelRecovery,
                                               float maximumInvalidNearestPixelRecoveryInverseDepthSpread,
                                               bool enableCrossViewConsensusDepth,
                                               float maximumCrossViewConsensusInverseDepthSpread,
                                               const cv::Mat& crossViewConsensusMask,
                                               const cv::Mat& referenceAnchoredConsensusDepth,
                                               bool excludeAnchoredInterpolation)
    {
        DepthTsdfObservationSample result;
        result.useAdaptiveGeometryEvidence = frame.useAdaptiveGeometryEvidence;
        const int nearest_column = static_cast<int>(std::lround(pixel.x));
        const int nearest_row = static_cast<int>(std::lround(pixel.y));
        if (nearest_row < 0 || nearest_row >= frame.depth.rows || nearest_column < 0 ||
            nearest_column >= frame.depth.cols)
        {
            return result;
        }

        auto optional_float_evidence = [&](const cv::Mat& matrix, int row, int column)
        {
            return matrix.type() == CV_32FC1 && matrix.size() == frame.depth.size() ? matrix.at<float>(row, column)
                                                                                    : 0.0f;
        };
        auto interpolation_is_excluded = [&](int row, int column)
        {
            return excludeAnchoredInterpolation && frame.depthProvenance.type() == CV_8UC1 &&
                   frame.depthProvenance.size() == frame.depth.size() &&
                   xjw::mvs::isInterpolatedDepthProvenance(frame.depthProvenance.at<std::uint8_t>(row, column));
        };

        auto consensus_depth = [&](int row,
                                   int column,
                                   float raw_depth,
                                   std::uint16_t geometry_support,
                                   float inverse_depth_spread,
                                   bool* used)
        {
            if (used)
            {
                *used = false;
            }
            if (!enableCrossViewConsensusDepth)
            {
                return raw_depth;
            }
            // The precomputed map already encodes geometry, confidence, contour,
            // repair and bias gates. Legacy per-sample gates apply only without it.
            if (referenceAnchoredConsensusDepth.type() == CV_32FC1 &&
                referenceAnchoredConsensusDepth.size() == frame.depth.size())
            {
                const float depth = referenceAnchoredConsensusDepth.at<float>(row, column);
                if (!std::isfinite(depth) || depth <= 0.0f)
                {
                    return raw_depth;
                }
                if (used)
                {
                    *used = std::fabs(depth - raw_depth) > 1.0e-8f;
                }
                return depth;
            }
            if (geometry_support < 2 ||
                (!crossViewConsensusMask.empty() && crossViewConsensusMask.at<std::uint8_t>(row, column) == 0) ||
                !std::isfinite(inverse_depth_spread) || inverse_depth_spread < 0.0f ||
                inverse_depth_spread > maximumCrossViewConsensusInverseDepthSpread ||
                frame.inverseDepthMean.type() != CV_32FC1 || frame.inverseDepthMean.size() != frame.depth.size())
            {
                return raw_depth;
            }
            const float inverse_depth_mean = frame.inverseDepthMean.at<float>(row, column);
            if (!std::isfinite(inverse_depth_mean) || inverse_depth_mean <= 1.0e-12f)
            {
                return raw_depth;
            }
            const float depth = 1.0f / inverse_depth_mean;
            if (!std::isfinite(depth) || depth <= 0.0f)
            {
                return raw_depth;
            }
            if (used)
            {
                *used = true;
            }
            return depth;
        };

        auto classify_pixel = [&](int row, int column, DepthTsdfObservationSample* sample)
        {
            if (frame.supportMask.at<std::uint8_t>(row, column) == 0)
            {
                sample->failure = DepthTsdfObservationFailure::SupportMask;
                return false;
            }
            if (effectiveDepthValidMask.at<std::uint8_t>(row, column) == 0)
            {
                sample->failure = DepthTsdfObservationFailure::DepthValid;
                return false;
            }
            if (interpolation_is_excluded(row, column))
            {
                sample->failure = DepthTsdfObservationFailure::InterpolationPolicy;
                return false;
            }
            const float depth = frame.depth.at<float>(row, column);
            if (!std::isfinite(depth) || depth <= 0.0f)
            {
                sample->failure = DepthTsdfObservationFailure::Depth;
                return false;
            }
            const float confidence = frame.confidence.at<float>(row, column);
            if (!std::isfinite(confidence) || confidence < minimumConfidence)
            {
                sample->failure = DepthTsdfObservationFailure::Confidence;
                return false;
            }
            sample->geometrySupportCount = frame.geometrySupportCount.at<std::uint16_t>(row, column);
            sample->geometrySourceMask =
                frame.geometrySourceMask.type() == CV_16UC1 && frame.geometrySourceMask.size() == frame.depth.size()
                    ? frame.geometrySourceMask.at<std::uint16_t>(row, column)
                    : 0;
            sample->adaptiveGeometrySupportWeight =
                optional_float_evidence(frame.adaptiveGeometrySupportWeight, row, column);
            sample->adaptiveGeometryEffectiveViewCount =
                optional_float_evidence(frame.adaptiveGeometryEffectiveViewCount, row, column);
            sample->adaptiveGeometryConflictRatio =
                optional_float_evidence(frame.adaptiveGeometryConflictRatio, row, column);
            sample->inverseDepthRelativeSpread = frame.inverseDepthRelativeSpread.type() == CV_32FC1 &&
                                                         frame.inverseDepthRelativeSpread.size() == frame.depth.size()
                                                     ? frame.inverseDepthRelativeSpread.at<float>(row, column)
                                                     : 0.0f;
            if (!frame.useAdaptiveGeometryEvidence && maximumObservationInverseDepthSpread > 0.0f &&
                std::isfinite(sample->inverseDepthRelativeSpread) &&
                sample->inverseDepthRelativeSpread > maximumObservationInverseDepthSpread)
            {
                sample->valid = false;
                sample->failure = DepthTsdfObservationFailure::GeometryConsistency;
                return false;
            }
            sample->valid = true;
            sample->depth = consensus_depth(row,
                                            column,
                                            depth,
                                            sample->geometrySupportCount,
                                            sample->inverseDepthRelativeSpread,
                                            &sample->usedCrossViewConsensusDepth);
            sample->usedCrossViewRepairedDepth = frame.crossViewRepairedMask.type() == CV_8UC1 &&
                                                 frame.crossViewRepairedMask.size() == frame.depth.size() &&
                                                 frame.crossViewRepairedMask.at<std::uint8_t>(row, column) != 0;
            sample->confidence = confidence;
            sample->contributingPixelCount = 1;
            sample->failure = DepthTsdfObservationFailure::None;
            return true;
        };

        if (!discontinuityAware)
        {
            classify_pixel(nearest_row, nearest_column, &result);
            return result;
        }

        struct Candidate
        {
            int row = 0;
            int column = 0;
            float depth = 0.0f;
            float confidence = 0.0f;
            float spatialWeight = 0.0f;
            std::uint16_t geometrySupportCount = 0;
            std::uint16_t geometrySourceMask = 0;
            float adaptiveGeometrySupportWeight = 0.0f;
            float adaptiveGeometryEffectiveViewCount = 0.0f;
            float adaptiveGeometryConflictRatio = 0.0f;
            float inverseDepthRelativeSpread = 0.0f;
            bool nearest = false;
            bool usedCrossViewConsensusDepth = false;
            bool usedCrossViewRepairedDepth = false;
        };
        std::array<Candidate, 4> candidates{};
        int candidate_count = 0;
        bool passed_support = false;
        bool passed_depth_valid = false;
        bool passed_provenance = false;
        bool passed_depth = false;
        bool passed_confidence = false;
        const int floor_column = static_cast<int>(std::floor(pixel.x));
        const int floor_row = static_cast<int>(std::floor(pixel.y));
        for (int delta_row = 0; delta_row <= 1; ++delta_row)
        {
            for (int delta_column = 0; delta_column <= 1; ++delta_column)
            {
                const int row = std::clamp(floor_row + delta_row, 0, frame.depth.rows - 1);
                const int column = std::clamp(floor_column + delta_column, 0, frame.depth.cols - 1);
                bool duplicate = false;
                for (int index = 0; index < candidate_count; ++index)
                {
                    duplicate = duplicate || (candidates[index].row == row && candidates[index].column == column);
                }
                if (duplicate)
                {
                    continue;
                }
                if (frame.supportMask.at<std::uint8_t>(row, column) == 0)
                {
                    continue;
                }
                passed_support = true;
                if (effectiveDepthValidMask.at<std::uint8_t>(row, column) == 0)
                {
                    continue;
                }
                passed_depth_valid = true;
                if (interpolation_is_excluded(row, column))
                {
                    continue;
                }
                passed_provenance = true;
                const float depth = frame.depth.at<float>(row, column);
                if (!std::isfinite(depth) || depth <= 0.0f)
                {
                    continue;
                }
                passed_depth = true;
                const float confidence = frame.confidence.at<float>(row, column);
                if (!std::isfinite(confidence) || confidence < minimumConfidence)
                {
                    continue;
                }
                passed_confidence = true;
                const float inverse_depth_relative_spread =
                    frame.inverseDepthRelativeSpread.type() == CV_32FC1 &&
                            frame.inverseDepthRelativeSpread.size() == frame.depth.size()
                        ? frame.inverseDepthRelativeSpread.at<float>(row, column)
                        : 0.0f;
                if (!frame.useAdaptiveGeometryEvidence && maximumObservationInverseDepthSpread > 0.0f &&
                    std::isfinite(inverse_depth_relative_spread) &&
                    inverse_depth_relative_spread > maximumObservationInverseDepthSpread)
                {
                    continue;
                }
                Candidate& candidate = candidates[candidate_count++];
                candidate.row = row;
                candidate.column = column;
                candidate.confidence = confidence;
                const float weight_x = std::max(0.05f, 1.0f - std::fabs(static_cast<float>(pixel.x) - column));
                const float weight_y = std::max(0.05f, 1.0f - std::fabs(static_cast<float>(pixel.y) - row));
                candidate.spatialWeight = weight_x * weight_y;
                candidate.geometrySupportCount = frame.geometrySupportCount.at<std::uint16_t>(row, column);
                candidate.geometrySourceMask =
                    frame.geometrySourceMask.type() == CV_16UC1 && frame.geometrySourceMask.size() == frame.depth.size()
                        ? frame.geometrySourceMask.at<std::uint16_t>(row, column)
                        : 0;
                candidate.adaptiveGeometrySupportWeight =
                    optional_float_evidence(frame.adaptiveGeometrySupportWeight, row, column);
                candidate.adaptiveGeometryEffectiveViewCount =
                    optional_float_evidence(frame.adaptiveGeometryEffectiveViewCount, row, column);
                candidate.adaptiveGeometryConflictRatio =
                    optional_float_evidence(frame.adaptiveGeometryConflictRatio, row, column);
                candidate.inverseDepthRelativeSpread = inverse_depth_relative_spread;
                candidate.depth = consensus_depth(row,
                                                  column,
                                                  depth,
                                                  candidate.geometrySupportCount,
                                                  candidate.inverseDepthRelativeSpread,
                                                  &candidate.usedCrossViewConsensusDepth);
                candidate.usedCrossViewRepairedDepth = frame.crossViewRepairedMask.type() == CV_8UC1 &&
                                                       frame.crossViewRepairedMask.size() == frame.depth.size() &&
                                                       frame.crossViewRepairedMask.at<std::uint8_t>(row, column) != 0;
                candidate.nearest = row == nearest_row && column == nearest_column;
            }
        }

        if (candidate_count == 0)
        {
            result.failure =
                !passed_support
                    ? DepthTsdfObservationFailure::SupportMask
                    : (!passed_depth_valid
                           ? DepthTsdfObservationFailure::DepthValid
                           : (!passed_provenance
                                  ? DepthTsdfObservationFailure::InterpolationPolicy
                                  : (!passed_depth
                                         ? DepthTsdfObservationFailure::Depth
                                         : (!passed_confidence ? DepthTsdfObservationFailure::Confidence
                                                               : DepthTsdfObservationFailure::GeometryConsistency))));
            return result;
        }

        const bool has_nearest_candidate = std::any_of(candidates.cbegin(),
                                                       candidates.cbegin() + candidate_count,
                                                       [](const Candidate& candidate) { return candidate.nearest; });
        if (!allowInvalidNearestPixelRecovery && !has_nearest_candidate)
        {
            result.failure = DepthTsdfObservationFailure::DepthValid;
            result.rejectedInvalidNearestPixelRecovery = true;
            return result;
        }

        int anchor_index = -1;
        float best_anchor_score = -1.0f;
        for (int index = 0; index < candidate_count; ++index)
        {
            if (candidates[index].nearest)
            {
                anchor_index = index;
                break;
            }
            const float score = candidates[index].spatialWeight * candidates[index].confidence;
            if (score > best_anchor_score)
            {
                best_anchor_score = score;
                anchor_index = index;
            }
        }

        const float anchor_depth = candidates[anchor_index].depth;
        const float relative_threshold = std::max(0.0f, maximumRelativeDepthSpread);
        float depth_weight_sum = 0.0f;
        float confidence_weight_sum = 0.0f;
        float spatial_weight_sum = 0.0f;
        bool first_evidence = true;
        for (int index = 0; index < candidate_count; ++index)
        {
            const Candidate& candidate = candidates[index];
            const float relative_error = std::fabs(candidate.depth - anchor_depth) / std::max(anchor_depth, 1.0e-6f);
            if (relative_error > relative_threshold)
            {
                ++result.discontinuityRejectedPixelCount;
                continue;
            }
            const float depth_weight = candidate.spatialWeight * candidate.confidence;
            result.depth += candidate.depth * depth_weight;
            depth_weight_sum += depth_weight;
            result.confidence += candidate.confidence * candidate.spatialWeight;
            spatial_weight_sum += candidate.spatialWeight;
            confidence_weight_sum += candidate.confidence;
            result.geometrySupportCount = std::max(result.geometrySupportCount, candidate.geometrySupportCount);
            result.geometrySourceMask =
                first_evidence ? candidate.geometrySourceMask
                               : static_cast<std::uint16_t>(result.geometrySourceMask & candidate.geometrySourceMask);
            result.adaptiveGeometrySupportWeight += candidate.adaptiveGeometrySupportWeight * candidate.spatialWeight;
            result.adaptiveGeometryEffectiveViewCount +=
                candidate.adaptiveGeometryEffectiveViewCount * candidate.spatialWeight;
            result.adaptiveGeometryConflictRatio += candidate.adaptiveGeometryConflictRatio * candidate.spatialWeight;
            result.inverseDepthRelativeSpread =
                std::max(result.inverseDepthRelativeSpread, candidate.inverseDepthRelativeSpread);
            first_evidence = false;
            result.usedCrossViewConsensusDepth =
                result.usedCrossViewConsensusDepth || candidate.usedCrossViewConsensusDepth;
            result.usedCrossViewRepairedDepth =
                result.usedCrossViewRepairedDepth || candidate.usedCrossViewRepairedDepth;
            ++result.contributingPixelCount;
        }
        if (depth_weight_sum <= 0.0f || spatial_weight_sum <= 0.0f || confidence_weight_sum <= 0.0f)
        {
            result.failure = DepthTsdfObservationFailure::Depth;
            return result;
        }

        result.depth /= depth_weight_sum;
        result.confidence /= spatial_weight_sum;
        result.adaptiveGeometrySupportWeight /= spatial_weight_sum;
        result.adaptiveGeometryEffectiveViewCount /= spatial_weight_sum;
        result.adaptiveGeometryConflictRatio /= spatial_weight_sum;
        result.valid = true;
        result.failure = DepthTsdfObservationFailure::None;
        result.recoveredFromInvalidNearestPixel = !has_nearest_candidate;
        if (!frame.useAdaptiveGeometryEvidence && result.recoveredFromInvalidNearestPixel &&
            maximumInvalidNearestPixelRecoveryInverseDepthSpread > 0.0f &&
            result.inverseDepthRelativeSpread > maximumInvalidNearestPixelRecoveryInverseDepthSpread)
        {
            result.valid = false;
            result.failure = DepthTsdfObservationFailure::GeometryConsistency;
            result.rejectedInvalidNearestPixelRecovery = true;
        }
        return result;
    }

    bool DepthTsdfSurfaceBuilder::isSampleSupported(float accumulatedWeight,
                                                    int distinctSupportCount,
                                                    float maximumObservationWeight,
                                                    const DepthTsdfOptions& options,
                                                    bool* singleView,
                                                    bool* multiView,
                                                    int maximumGeometrySupportCount,
                                                    bool* geometryVerifiedSingleView,
                                                    bool hasStrongAdaptiveSurfaceObservation)
    {
        const bool multi_view_supported = distinctSupportCount >= std::max(2, options.minimumDistinctCameraSupport) &&
                                          accumulatedWeight >= options.minimumVoxelWeight;
        const bool legacy_single_view_supported = options.minimumDistinctCameraSupport <= 1 &&
                                                  distinctSupportCount == 1 &&
                                                  maximumObservationWeight >= options.minimumSingleObservationWeight;
        const bool legacy_geometry_verified_single_view_supported =
            distinctSupportCount == 1 && maximumObservationWeight >= options.minimumGeometryVerifiedObservationWeight &&
            options.allowGeometryVerifiedSingleObservation &&
            maximumGeometrySupportCount >= options.minimumGeometrySupportCount;
        // A strong adaptive observation already represents continuous agreement
        // from multiple source views.  Its TSDF weight is intentionally reduced by
        // confidence, frame quality and spread factors, so applying the legacy
        // 0.85 gate again can make the guarded path unreachable.  It remains only
        // a candidate here and must still pass neighborhood growth below.
        const bool strong_adaptive_single_view_supported = distinctSupportCount == 1 &&
                                                           options.enableGeometrySingleViewNeighborhoodGuard &&
                                                           hasStrongAdaptiveSurfaceObservation;
        const bool geometry_verified_single_view_supported =
            legacy_geometry_verified_single_view_supported || strong_adaptive_single_view_supported;
        const bool single_view_supported = legacy_single_view_supported || geometry_verified_single_view_supported;
        if (singleView)
        {
            *singleView = single_view_supported;
        }
        if (multiView)
        {
            *multiView = multi_view_supported;
        }
        if (geometryVerifiedSingleView)
        {
            *geometryVerifiedSingleView = geometry_verified_single_view_supported;
        }
        return multi_view_supported || single_view_supported;
    }

    float DepthTsdfSurfaceBuilder::observationEvidenceWeightMultiplier(const DepthTsdfObservationSample& observation,
                                                                       const DepthTsdfOptions& options)
    {
        if (observation.useAdaptiveGeometryEvidence)
        {
            if (observation.usedCrossViewRepairedDepth)
            {
                return std::clamp(options.repairedObservationMultiplier, 0.05f, 1.0f);
            }
            const float support_weight = std::isfinite(observation.adaptiveGeometrySupportWeight)
                                             ? std::clamp(observation.adaptiveGeometrySupportWeight, 0.0f, 1.0f)
                                             : 0.0f;
            float multiplier =
                std::max(std::clamp(options.adaptiveGeometryMinimumObservationMultiplier, 0.05f, 1.0f), support_weight);
            if (!options.enableAdaptiveConflictRobustWeighting ||
                !std::isfinite(observation.adaptiveGeometryConflictRatio))
            {
                return multiplier;
            }

            const float conflict_ratio = std::clamp(observation.adaptiveGeometryConflictRatio, 0.0f, 1.0f);
            const float knee = std::clamp(options.adaptiveConflictWeightKnee, 0.0f, 0.99f);
            const float zero = std::clamp(options.adaptiveConflictWeightZero, knee + 1.0e-6f, 1.0f);
            const float minimum_conflict_multiplier =
                std::clamp(options.minimumAdaptiveConflictWeightMultiplier, 0.0f, 1.0f);
            if (conflict_ratio <= knee)
            {
                return multiplier;
            }
            if (conflict_ratio >= zero)
            {
                return multiplier * minimum_conflict_multiplier;
            }
            const float normalized = (conflict_ratio - knee) / (zero - knee);
            const float smooth = normalized * normalized * (3.0f - 2.0f * normalized);
            return multiplier * (1.0f - smooth * (1.0f - minimum_conflict_multiplier));
        }
        if (!options.enablePixelEvidenceWeighting)
        {
            return 1.0f;
        }
        if (observation.usedCrossViewRepairedDepth)
        {
            return std::clamp(options.repairedObservationMultiplier, 0.05f, 1.0f);
        }
        if (observation.geometrySupportCount == 0)
        {
            return std::clamp(options.unconfirmedNativeObservationMultiplier, 0.05f, 1.0f);
        }
        if (observation.geometrySupportCount == 1)
        {
            return std::clamp(options.weakNativeObservationMultiplier, 0.05f, 1.0f);
        }
        return 1.0f;
    }

    float DepthTsdfSurfaceBuilder::observationInverseDepthSpreadWeightMultiplier(
        const DepthTsdfObservationSample& observation, const DepthTsdfOptions& options)
    {
        if (!options.enableInverseDepthSpreadWeighting || !std::isfinite(observation.inverseDepthRelativeSpread) ||
            observation.inverseDepthRelativeSpread <= 0.0f)
        {
            return 1.0f;
        }

        const float knee = std::clamp(options.inverseDepthSpreadWeightKnee, 0.0f, 0.099f);
        const float zero = std::clamp(options.inverseDepthSpreadWeightZero, knee + 1.0e-6f, 0.10f);
        const float minimum_multiplier = std::clamp(options.minimumInverseDepthSpreadWeightMultiplier, 0.0f, 1.0f);
        if (observation.inverseDepthRelativeSpread <= knee)
        {
            return 1.0f;
        }
        if (observation.inverseDepthRelativeSpread >= zero)
        {
            return minimum_multiplier;
        }

        const float normalized = (observation.inverseDepthRelativeSpread - knee) / (zero - knee);
        const float smooth = normalized * normalized * (3.0f - 2.0f * normalized);
        return 1.0f - smooth * (1.0f - minimum_multiplier);
    }

    float DepthTsdfSurfaceBuilder::observationInverseDepthSpreadSupportWeightMultiplier(
        const DepthTsdfObservationSample& observation, const DepthTsdfOptions& options)
    {
        const float field_multiplier = observationInverseDepthSpreadWeightMultiplier(observation, options);
        if (!options.enableInverseDepthSpreadSupportWeightDecoupling || !options.enableInverseDepthSpreadWeighting)
        {
            return field_multiplier;
        }
        if (field_multiplier <= 0.0f)
        {
            return 0.0f;
        }
        return std::pow(field_multiplier, std::clamp(options.inverseDepthSpreadSupportWeightExponent, 0.05f, 1.0f));
    }

    float
    DepthTsdfSurfaceBuilder::observationEvidenceSupportWeightMultiplier(const DepthTsdfObservationSample& observation,
                                                                        const DepthTsdfOptions& options)
    {
        const float field_multiplier = observationEvidenceWeightMultiplier(observation, options);
        if (!options.enableEvidenceSupportWeightDecoupling)
        {
            return field_multiplier;
        }
        return std::pow(field_multiplier, std::clamp(options.evidenceSupportWeightExponent, 0.0f, 1.0f));
    }

    bool
    DepthTsdfSurfaceBuilder::observationHasStrongAdaptiveGeometryEvidence(const DepthTsdfObservationSample& observation,
                                                                          const DepthTsdfOptions& options)
    {
        if (!observation.useAdaptiveGeometryEvidence)
        {
            return false;
        }
        if (observation.usedCrossViewRepairedDepth)
        {
            return false;
        }
        if (!std::isfinite(observation.adaptiveGeometrySupportWeight) ||
            !std::isfinite(observation.adaptiveGeometryEffectiveViewCount) ||
            !std::isfinite(observation.adaptiveGeometryConflictRatio))
        {
            return false;
        }
        return observation.adaptiveGeometrySupportWeight >=
                   std::clamp(options.adaptiveGeometryFullIntegrationMinimumSupportWeight, 0.0f, 1.0f) &&
               observation.adaptiveGeometryEffectiveViewCount >=
                   std::max(1.0f, options.adaptiveGeometryFullIntegrationMinimumEffectiveViewCount) &&
               observation.adaptiveGeometryConflictRatio <=
                   std::clamp(options.adaptiveGeometryFullIntegrationMaximumConflictRatio, 0.0f, 1.0f);
    }

    bool DepthTsdfSurfaceBuilder::observationUsesSurfaceOnlyIntegration(const DepthTsdfObservationSample& observation,
                                                                        const DepthTsdfOptions& options)
    {
        if (observation.useAdaptiveGeometryEvidence)
        {
            return !observationHasStrongAdaptiveGeometryEvidence(observation, options);
        }
        return options.enableWeakEvidenceSurfaceOnlyIntegration &&
               (observation.usedCrossViewRepairedDepth || observation.geometrySupportCount <= 1);
    }
} // namespace xjw::mesh
