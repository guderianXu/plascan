#include "DepthTsdfStages.h"
namespace xjw::mesh
{
    using namespace tsdf_detail;

    DepthTsdfResult tsdf_detail::buildTsdfVolume(const QVector<DepthTsdfFrame>& frames, const DepthTsdfOptions& options)
    {
        DepthTsdfResult result;
        if (!options.enableAuxiliarySurfaceOnlyIntegration)
        {
            QVector<DepthTsdfFrame> geometry_frames;
            QVector<int> auxiliary_ref_indices;
            QVector<int> excluded_ref_indices;
            geometry_frames.reserve(frames.size());
            excluded_ref_indices.reserve(frames.size());
            for (const DepthTsdfFrame& frame : frames)
            {
                if (frame.auxiliarySurfaceOnly)
                {
                    auxiliary_ref_indices.push_back(frame.refIndex);
                }
                else
                {
                    geometry_frames.push_back(frame);
                }
            }

            DepthAuxiliaryBridgeSelectionResult bridge_selection;
            if (options.enableAuxiliaryBridgeOnlyIntegration)
            {
                const bool exact_bridge_contract =
                    std::all_of(frames.cbegin(),
                                frames.cend(),
                                [](const DepthTsdfFrame& frame)
                                {
                                    return xjw::mvs::isOrbitalDepthSceneProfile(frame.sceneProfile) &&
                                           frame.algorithmRevision >= xjw::mvs::kMvsGeometrySourceOrdinalRevision &&
                                           !frame.geometrySourceIndices.isEmpty();
                                });
                if (exact_bridge_contract)
                {
                    std::vector<DepthAuxiliaryBridgeNode> bridge_nodes;
                    bridge_nodes.reserve(static_cast<std::size_t>(frames.size()));
                    for (int index = 0; index < frames.size(); ++index)
                    {
                        bridge_nodes.push_back(bridgeNode(frames[index], index));
                    }
                    bridge_selection = DepthAuxiliaryBridgeSelector::select(bridge_nodes);
                }
                else
                {
                    bridge_selection.failClosed = true;
                }
            }

            QSet<int> selected_bridge_indices;
            if (bridge_selection.connected)
            {
                for (const int frame_index : bridge_selection.selectedAuxiliaryFrameIndices)
                {
                    selected_bridge_indices.insert(frame_index);
                    geometry_frames.push_back(frames[frame_index]);
                }
            }
            for (int index = 0; index < frames.size(); ++index)
            {
                if (frames[index].auxiliarySurfaceOnly && !selected_bridge_indices.contains(index))
                {
                    excluded_ref_indices.push_back(frames[index].refIndex);
                }
            }
            if (options.execution.progress && options.enableAuxiliaryBridgeOnlyIntegration)
            {
                QStringList bridge_refs;
                for (const int ref_index : bridge_selection.selectedAuxiliaryRefIndices)
                {
                    bridge_refs.push_back(QString::number(ref_index));
                }
                options.execution.reportProgress(
                    (bridge_selection.connected ? QStringLiteral("实测桥接选择：主来源图=%1 个分量，桥接帧=[%2]")
                                                      .arg(bridge_selection.primaryComponentCount)
                                                      .arg(bridge_refs.join(QStringLiteral(",")))
                                                : QStringLiteral("实测桥接关闭：主来源图=%1 个分量，合格验证帧无法连通")
                                                      .arg(bridge_selection.primaryComponentCount))
                        .toUtf8()
                        .toStdString(),
                    (3) / 100.0);
            }

            if (geometry_frames.size() < options.minimumInputFrames)
            {
                result.statistics.inputFrameCount = frames.size();
                result.statistics.geometryInputFrameCount = geometry_frames.size();
                result.statistics.effectiveAuxiliarySurfaceOnlyIntegration = false;
                result.statistics.auxiliarySurfaceOnlyFrameCount = auxiliary_ref_indices.size();
                result.statistics.auxiliarySurfaceOnlyRefIndices = auxiliary_ref_indices;
                result.statistics.effectiveAuxiliaryBridgeOnlyIntegration = false;
                result.statistics.auxiliaryBridgePrimaryComponentCount = bridge_selection.primaryComponentCount;
                result.statistics.auxiliaryBridgeGraphConnected = bridge_selection.connected;
                result.statistics.auxiliaryBridgeSelectionFailClosed = bridge_selection.failClosed;
                result.statistics.excludedAuxiliarySurfaceOnlyFrameCount = excluded_ref_indices.size();
                result.statistics.excludedAuxiliarySurfaceOnlyRefIndices = excluded_ref_indices;
                result.errorMessage = QStringLiteral("TSDF observation-only geometry requires at least %1 primary "
                                                     "frames; received primary=%2, auxiliary excluded=%3")
                                          .arg(options.minimumInputFrames)
                                          .arg(geometry_frames.size())
                                          .arg(excluded_ref_indices.size());
                return result;
            }

            DepthTsdfOptions geometry_options = options;
            geometry_options.enableAuxiliarySurfaceOnlyIntegration = true;
            result = DepthTsdfSurfaceBuilder::build(geometry_frames, geometry_options);
            result.statistics.inputFrameCount = frames.size();
            result.statistics.geometryInputFrameCount = geometry_frames.size();
            result.statistics.effectiveAuxiliarySurfaceOnlyIntegration = false;
            result.statistics.auxiliarySurfaceOnlyFrameCount = auxiliary_ref_indices.size();
            result.statistics.auxiliarySurfaceOnlyRefIndices = auxiliary_ref_indices;
            result.statistics.effectiveAuxiliaryBridgeOnlyIntegration =
                options.enableAuxiliaryBridgeOnlyIntegration && bridge_selection.connected &&
                !bridge_selection.selectedAuxiliaryFrameIndices.empty();
            result.statistics.auxiliaryBridgePrimaryComponentCount = bridge_selection.primaryComponentCount;
            result.statistics.auxiliaryBridgeGraphConnected = bridge_selection.connected;
            result.statistics.auxiliaryBridgeSelectionFailClosed = bridge_selection.failClosed;
            result.statistics.auxiliaryBridgeRefIndices =
                QVector<int>(bridge_selection.selectedAuxiliaryRefIndices.cbegin(),
                             bridge_selection.selectedAuxiliaryRefIndices.cend());
            result.statistics.excludedAuxiliarySurfaceOnlyFrameCount = excluded_ref_indices.size();
            result.statistics.excludedAuxiliarySurfaceOnlyRefIndices = excluded_ref_indices;
            return result;
        }

        result.statistics.effectiveAuxiliarySurfaceOnlyIntegration = true;
        result.statistics.geometryInputFrameCount = frames.size();
        const bool requested_robust_frame_quality_weighting = options.enableRobustFrameQualityWeighting;
        const float requested_robust_frame_quality_minimum_multiplier = options.robustFrameQualityMinimumMultiplier;
        const float requested_robust_frame_quality_mad_floor = options.robustFrameQualityMadFloor;
        const float requested_robust_frame_quality_penalty_onset = options.robustFrameQualityPenaltyOnset;
        const float requested_robust_frame_quality_penalty_strength = options.robustFrameQualityPenaltyStrength;
        const bool requested_robust_frame_quality_rejection = options.enableRobustFrameQualityRejection;
        const float requested_robust_frame_quality_rejection_sigma = options.robustFrameQualityRejectionSigma;
        const float requested_robust_frame_quality_maximum_rejected_ratio =
            options.robustFrameQualityMaximumRejectedRatio;
        const int requested_robust_frame_quality_minimum_retained_frames =
            options.robustFrameQualityMinimumRetainedFrames;
        result.statistics.inputFrameCount = frames.size();
        if (frames.size() < options.minimumInputFrames)
        {
            result.errorMessage = QStringLiteral("TSDF requires at least %1 input frames; received=%2")
                                      .arg(options.minimumInputFrames)
                                      .arg(frames.size());
            return result;
        }
        for (const DepthTsdfFrame& frame : frames)
        {
            if (!frame.camera.isValid() || frame.depth.type() != CV_32FC1 || frame.confidence.type() != CV_32FC1 ||
                frame.geometrySupportCount.type() != CV_16UC1 || frame.depthValidMask.type() != CV_8UC1 ||
                frame.supportMask.type() != CV_8UC1 || frame.confidence.size() != frame.depth.size() ||
                frame.geometrySupportCount.size() != frame.depth.size() ||
                (!frame.geometrySourceMask.empty() && (frame.geometrySourceMask.type() != CV_16UC1 ||
                                                       frame.geometrySourceMask.size() != frame.depth.size())) ||
                (!frame.inverseDepthRelativeSpread.empty() &&
                 (frame.inverseDepthRelativeSpread.type() != CV_32FC1 ||
                  frame.inverseDepthRelativeSpread.size() != frame.depth.size())) ||
                frame.depthValidMask.size() != frame.depth.size() || frame.supportMask.size() != frame.depth.size())
            {
                result.errorMessage =
                    QStringLiteral("TSDF input frame %1 has invalid camera or matrix layout").arg(frame.refIndex);
                return result;
            }
        }
        result.statistics.acceptedFrameCount = frames.size();
        QVector<float> raw_frame_quality_weights;
        raw_frame_quality_weights.reserve(frames.size());
        std::vector<DepthFusionView> fusion_views;
        fusion_views.reserve(static_cast<std::size_t>(frames.size()));
        QVector<int> auxiliary_surface_only_ref_indices;
        for (const DepthTsdfFrame& frame : frames)
        {
            raw_frame_quality_weights.push_back(std::clamp(frame.frameQualityWeight, 0.0f, 1.0f));
            DepthFusionView view;
            view.frameIndex = static_cast<int>(fusion_views.size());
            view.refIndex = frame.refIndex;
            view.cameraCenter = frame.camera.cameraCenter();
            fusion_views.push_back(view);
            if (frame.auxiliarySurfaceOnly)
            {
                auxiliary_surface_only_ref_indices.push_back(frame.refIndex);
            }
        }
        QVector<float> effective_frame_quality_weights = raw_frame_quality_weights;
        const bool robust_frame_quality_weighting_enabled = requested_robust_frame_quality_weighting;
        float robust_frame_quality_median = 0.0f;
        float robust_frame_quality_scale = 0.0f;
        float robust_frame_quality_minimum_effective_weight = 1.0f;
        int robust_frame_quality_downweighted_frame_count = 0;
        result.statistics.effectiveRobustFrameQualityWeighting = robust_frame_quality_weighting_enabled;
        if (robust_frame_quality_weighting_enabled)
        {
            effective_frame_quality_weights =
                DepthTsdfSurfaceBuilder::robustFrameQualityWeights(raw_frame_quality_weights,
                                                                   requested_robust_frame_quality_minimum_multiplier,
                                                                   requested_robust_frame_quality_mad_floor,
                                                                   requested_robust_frame_quality_penalty_onset,
                                                                   requested_robust_frame_quality_penalty_strength,
                                                                   &robust_frame_quality_median,
                                                                   &robust_frame_quality_scale);
        }
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            if (frames[frame_index].auxiliarySurfaceOnly)
            {
                effective_frame_quality_weights[frame_index] *=
                    std::clamp(options.validationOnlyFrameWeightMultiplier, 0.05f, 1.0f);
            }
        }
        QVector<int> rejected_frame_indices;
        QVector<int> rejected_frame_ref_indices;
        QVector<int> coverage_protected_ref_indices;
        const bool robust_frame_quality_rejection_enabled =
            requested_robust_frame_quality_rejection && robust_frame_quality_scale > 0.0f;
        if (robust_frame_quality_rejection_enabled)
        {
            QVector<int> low_tail_candidates;
            for (int frame_index = 0; frame_index < raw_frame_quality_weights.size(); ++frame_index)
            {
                const float low_tail_sigma =
                    (robust_frame_quality_median - raw_frame_quality_weights[frame_index]) / robust_frame_quality_scale;
                if (low_tail_sigma > std::max(0.0f, requested_robust_frame_quality_rejection_sigma))
                {
                    low_tail_candidates.push_back(frame_index);
                }
            }
            std::sort(low_tail_candidates.begin(),
                      low_tail_candidates.end(),
                      [&raw_frame_quality_weights](int left, int right)
                      { return raw_frame_quality_weights[left] < raw_frame_quality_weights[right]; });
            const int frame_count = static_cast<int>(frames.size());
            const int minimum_retained_frames =
                std::clamp(std::max(options.minimumInputFrames, requested_robust_frame_quality_minimum_retained_frames),
                           1,
                           frame_count);
            const int ratio_limit = static_cast<int>(std::floor(
                frame_count * std::clamp(requested_robust_frame_quality_maximum_rejected_ratio, 0.0f, 0.50f)));
            const int rejection_limit = std::max(0, std::min(ratio_limit, frame_count - minimum_retained_frames));
            for (int candidate_index = 0;
                 candidate_index < low_tail_candidates.size() && candidate_index < rejection_limit;
                 ++candidate_index)
            {
                const int frame_index = low_tail_candidates[candidate_index];
                if (options.enableOrbitalFrameCoverageProtection)
                {
                    const std::vector<float> trial_weights(effective_frame_quality_weights.cbegin(),
                                                           effective_frame_quality_weights.cend());
                    if (!DepthFusionFramePolicy::canRejectWithoutCoverageGap(fusion_views,
                                                                             trial_weights,
                                                                             frame_index,
                                                                             options.maximumOrbitalAngularGapRatio,
                                                                             minimum_retained_frames))
                    {
                        effective_frame_quality_weights[frame_index] =
                            std::max(effective_frame_quality_weights[frame_index],
                                     raw_frame_quality_weights[frame_index] *
                                         std::clamp(options.coverageProtectedFrameMinimumMultiplier, 0.05f, 1.0f));
                        coverage_protected_ref_indices.push_back(frames[frame_index].refIndex);
                        continue;
                    }
                }
                effective_frame_quality_weights[frame_index] = 0.0f;
                rejected_frame_indices.push_back(frame_index);
                rejected_frame_ref_indices.push_back(frames[frame_index].refIndex);
            }
        }
        std::vector<float> final_frame_quality_weights(effective_frame_quality_weights.cbegin(),
                                                       effective_frame_quality_weights.cend());
        OrbitalCoverageStatistics orbital_coverage =
            options.enableOrbitalFrameCoverageProtection
                ? DepthFusionFramePolicy::evaluateOrbitalCoverage(fusion_views, final_frame_quality_weights)
                : OrbitalCoverageStatistics{};
        QVector<int> gap_quality_floor_ref_indices;
        if (options.enableOrbitalGapBoundaryRecovery && orbital_coverage.significantGap)
        {
            for (const OrbitalFrameRoleAssignment& assignment : orbital_coverage.frameRoles)
            {
                if (assignment.frameIndex < 0 || assignment.frameIndex >= effective_frame_quality_weights.size() ||
                    effective_frame_quality_weights[assignment.frameIndex] <= 0.0f)
                {
                    continue;
                }
                float minimum_multiplier = 0.0f;
                if (assignment.role == OrbitalFrameRole::GapBoundary)
                {
                    minimum_multiplier = std::clamp(options.orbitalGapBoundaryMinimumQualityMultiplier, 0.05f, 1.0f);
                }
                else if (assignment.role == OrbitalFrameRole::GapOpposite)
                {
                    minimum_multiplier = std::clamp(options.orbitalGapOppositeMinimumQualityMultiplier, 0.05f, 1.0f);
                }
                if (minimum_multiplier <= 0.0f)
                {
                    continue;
                }
                const float minimum_weight = raw_frame_quality_weights[assignment.frameIndex] * minimum_multiplier;
                if (effective_frame_quality_weights[assignment.frameIndex] + 1.0e-6f < minimum_weight)
                {
                    effective_frame_quality_weights[assignment.frameIndex] = minimum_weight;
                    gap_quality_floor_ref_indices.push_back(assignment.refIndex);
                }
            }
            final_frame_quality_weights.assign(effective_frame_quality_weights.cbegin(),
                                               effective_frame_quality_weights.cend());
            orbital_coverage =
                DepthFusionFramePolicy::evaluateOrbitalCoverage(fusion_views, final_frame_quality_weights);
        }
        QJsonArray orbital_frame_roles;
        for (const OrbitalFrameRoleAssignment& assignment : orbital_coverage.frameRoles)
        {
            orbital_frame_roles.append(
                QJsonObject{{QStringLiteral("frame_index"), assignment.frameIndex},
                            {QStringLiteral("ref_index"), assignment.refIndex},
                            {QStringLiteral("azimuth_degrees"), assignment.azimuthDegrees},
                            {QStringLiteral("role"), QString::fromLatin1(orbitalFrameRoleId(assignment.role))}});
        }
        std::vector<float> depth_completeness_frame_weights = final_frame_quality_weights;
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            if (frames[frame_index].auxiliarySurfaceOnly)
            {
                depth_completeness_frame_weights[static_cast<std::size_t>(frame_index)] = 0.0f;
            }
        }
        const OrbitalCoverageStatistics depth_completeness_orbital_coverage =
            options.enableOrbitalFrameCoverageProtection
                ? DepthFusionFramePolicy::evaluateOrbitalCoverage(fusion_views, depth_completeness_frame_weights)
                : OrbitalCoverageStatistics{};
        QJsonArray depth_completeness_frame_roles;
        for (const OrbitalFrameRoleAssignment& assignment : depth_completeness_orbital_coverage.frameRoles)
        {
            depth_completeness_frame_roles.append(
                QJsonObject{{QStringLiteral("frame_index"), assignment.frameIndex},
                            {QStringLiteral("ref_index"), assignment.refIndex},
                            {QStringLiteral("azimuth_degrees"), assignment.azimuthDegrees},
                            {QStringLiteral("role"), QString::fromLatin1(orbitalFrameRoleId(assignment.role))}});
        }
        result.statistics.effectiveRobustFrameQualityRejection = robust_frame_quality_rejection_enabled;
        result.statistics.robustFrameQualityRejectedFrameCount = rejected_frame_indices.size();
        result.statistics.acceptedFrameCount =
            static_cast<int>(std::count_if(effective_frame_quality_weights.cbegin(),
                                           effective_frame_quality_weights.cend(),
                                           [](float weight) { return weight > 0.0f; }));
        result.statistics.auxiliarySurfaceOnlyFrameCount = auxiliary_surface_only_ref_indices.size();
        result.statistics.auxiliarySurfaceOnlyRefIndices = auxiliary_surface_only_ref_indices;
        result.statistics.effectiveOrbitalFrameCoverageProtection = options.enableOrbitalFrameCoverageProtection;
        result.statistics.orbitalCoverageProtectedFrameCount = coverage_protected_ref_indices.size();
        result.statistics.orbitalCoverageProtectedRefIndices = coverage_protected_ref_indices;
        result.statistics.orbitalMedianAngularSpacingDegrees = orbital_coverage.medianAngularSpacingDegrees;
        result.statistics.orbitalMaximumAngularGapDegrees = orbital_coverage.maximumAngularGapDegrees;
        result.statistics.orbitalMaximumAngularGapRatio = orbital_coverage.maximumAngularGapRatio;
        result.statistics.orbitalSignificantAngularGap = orbital_coverage.significantGap;
        result.statistics.orbitalGapStartRefIndex = orbital_coverage.gapStartRefIndex;
        result.statistics.orbitalGapEndRefIndex = orbital_coverage.gapEndRefIndex;
        result.statistics.orbitalGapOppositeRefIndex = orbital_coverage.gapOppositeRefIndex;
        result.statistics.orbitalFrameRoles = orbital_frame_roles;
        result.statistics.depthCompletenessFrameRoles = depth_completeness_frame_roles;
        result.statistics.effectiveOrbitalGapBoundaryRecovery =
            options.enableOrbitalGapBoundaryRecovery && orbital_coverage.significantGap;
        result.statistics.orbitalGapQualityFloorFrameCount = gap_quality_floor_ref_indices.size();
        result.statistics.orbitalGapQualityFloorRefIndices = gap_quality_floor_ref_indices;
        result.statistics.effectiveOrbitalGapBoundaryMinimumObservationWeight =
            options.orbitalGapBoundaryMinimumObservationWeight;
        for (int frame_index = 0; frame_index < effective_frame_quality_weights.size(); ++frame_index)
        {
            robust_frame_quality_minimum_effective_weight =
                std::min(robust_frame_quality_minimum_effective_weight, effective_frame_quality_weights[frame_index]);
            if (effective_frame_quality_weights[frame_index] + 1.0e-6f < raw_frame_quality_weights[frame_index])
            {
                ++robust_frame_quality_downweighted_frame_count;
            }
        }
        result.statistics.robustFrameQualityDownweightedFrameCount = robust_frame_quality_downweighted_frame_count;
        result.statistics.robustFrameQualityMedian = robust_frame_quality_median;
        result.statistics.robustFrameQualityScale = robust_frame_quality_scale;
        result.statistics.robustFrameQualityMinimumEffectiveWeight = robust_frame_quality_minimum_effective_weight;

        QVector<cv::Mat> effective_depth_valid_masks;
        effective_depth_valid_masks.reserve(frames.size());
        const int erosion_pixels = std::clamp(options.depthValidBoundaryErosionPixels, 0, 4);
        std::uint64_t boundary_recovered_depth_valid_pixel_count = 0;
        if (erosion_pixels > 0)
        {
            const cv::Mat kernel =
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erosion_pixels * 2 + 1, erosion_pixels * 2 + 1));
            for (const DepthTsdfFrame& frame : frames)
            {
                if (frame.useAdaptiveGeometryEvidence)
                {
                    effective_depth_valid_masks.push_back(frame.depthValidMask.clone());
                    continue;
                }
                cv::Mat eroded;
                cv::erode(frame.depthValidMask, eroded, kernel);
                if (options.enableGeometryVerifiedBoundaryRecovery && erosion_pixels > 1 &&
                    !frame.geometrySupportCount.empty() && !frame.inverseDepthRelativeSpread.empty())
                {
                    const int recovery_erosion_pixels = erosion_pixels - 1;
                    const cv::Mat recovery_kernel = cv::getStructuringElement(
                        cv::MORPH_ELLIPSE, cv::Size(recovery_erosion_pixels * 2 + 1, recovery_erosion_pixels * 2 + 1));
                    cv::Mat less_eroded;
                    cv::erode(frame.depthValidMask, less_eroded, recovery_kernel);
                    cv::Mat recovery_ring;
                    cv::subtract(less_eroded, eroded, recovery_ring);
                    cv::Mat geometry_supported;
                    cv::compare(frame.geometrySupportCount,
                                options.minimumBoundaryRecoveryGeometrySupport,
                                geometry_supported,
                                cv::CMP_GE);
                    cv::Mat spread_supported;
                    cv::compare(frame.inverseDepthRelativeSpread,
                                options.maximumBoundaryRecoveryInverseDepthSpread,
                                spread_supported,
                                cv::CMP_LE);
                    cv::bitwise_and(recovery_ring, geometry_supported, recovery_ring);
                    cv::bitwise_and(recovery_ring, spread_supported, recovery_ring);
                    cv::bitwise_and(recovery_ring, frame.supportMask, recovery_ring);
                    boundary_recovered_depth_valid_pixel_count +=
                        static_cast<std::uint64_t>(cv::countNonZero(recovery_ring));
                    cv::bitwise_or(eroded, recovery_ring, eroded);
                }
                effective_depth_valid_masks.push_back(std::move(eroded));
            }
        }

        QVector<cv::Mat> reference_anchored_consensus_depths;
        if (options.enableCrossViewConsensusDepth)
        {
            reference_anchored_consensus_depths.reserve(frames.size());
            xjw::mvs::ReferenceAnchoredDepthConsensusOptions consensus_options;
            consensus_options.maximumInverseDepthSpread = options.maximumCrossViewConsensusInverseDepthSpread;
            consensus_options.minimumConfidence =
                std::max(consensus_options.minimumConfidence, options.minimumConfidence);
            for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
            {
                const DepthTsdfFrame& frame = frames[frame_index];
                const cv::Mat& valid_mask =
                    erosion_pixels > 0 ? effective_depth_valid_masks[frame_index] : frame.depthValidMask;
                xjw::mvs::ReferenceAnchoredDepthConsensusResult consensus =
                    xjw::mvs::makeReferenceAnchoredDepthConsensus(frame.depth,
                                                                  frame.inverseDepthMean,
                                                                  frame.geometrySupportCount,
                                                                  frame.inverseDepthRelativeSpread,
                                                                  frame.confidence,
                                                                  frame.crossViewRepairedMask,
                                                                  frame.supportMask,
                                                                  valid_mask,
                                                                  consensus_options);
                reference_anchored_consensus_depths.push_back(consensus.depth.empty() ? frame.depth.clone()
                                                                                      : std::move(consensus.depth));
            }
        }

        QVector<cv::Mat> contour_band_masks;
        std::uint64_t cross_view_consensus_contour_band_pixel_count = 0;
        if ((options.enableCrossViewConsensusDepth && options.crossViewConsensusContourBandOnly) ||
            options.enableContourBandZeroCrossingSupport)
        {
            contour_band_masks.reserve(frames.size());
            const cv::Mat contour_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
            {
                const DepthTsdfFrame& frame = frames[frame_index];
                const cv::Mat& valid_mask =
                    erosion_pixels > 0 ? effective_depth_valid_masks[frame_index] : frame.depthValidMask;
                cv::Mat inner_mask;
                cv::erode(valid_mask, inner_mask, contour_kernel);
                cv::Mat contour_band;
                cv::subtract(valid_mask, inner_mask, contour_band);
                cv::Mat geometry_supported;
                cv::compare(frame.geometrySupportCount,
                            options.minimumBoundaryRecoveryGeometrySupport,
                            geometry_supported,
                            cv::CMP_GE);
                cv::Mat spread_supported(frame.depth.size(), CV_8UC1, cv::Scalar(0));
                if (!frame.inverseDepthRelativeSpread.empty())
                {
                    cv::compare(frame.inverseDepthRelativeSpread,
                                std::min(options.maximumCrossViewConsensusInverseDepthSpread,
                                         options.maximumBoundaryRecoveryInverseDepthSpread),
                                spread_supported,
                                cv::CMP_LE);
                }
                cv::bitwise_and(contour_band, frame.supportMask, contour_band);
                cv::bitwise_and(contour_band, geometry_supported, contour_band);
                cv::bitwise_and(contour_band, spread_supported, contour_band);
                if (frame.geometrySourceMask.empty())
                {
                    contour_band.setTo(cv::Scalar(0));
                }
                else
                {
                    for (int row = 0; row < contour_band.rows; ++row)
                    {
                        std::uint8_t* band_row = contour_band.ptr<std::uint8_t>(row);
                        const std::uint16_t* source_row = frame.geometrySourceMask.ptr<std::uint16_t>(row);
                        for (int column = 0; column < contour_band.cols; ++column)
                        {
                            if (band_row[column] != 0 && bitCount(source_row[column]) < 2)
                            {
                                band_row[column] = 0;
                            }
                        }
                    }
                }
                cross_view_consensus_contour_band_pixel_count +=
                    static_cast<std::uint64_t>(cv::countNonZero(contour_band));
                contour_band_masks.push_back(std::move(contour_band));
            }
        }

        QVector<DepthTsdfFrame> retained_frames;
        retained_frames.reserve(frames.size() - rejected_frame_indices.size());
        std::vector<std::vector<int>> retained_source_indices;
        retained_source_indices.reserve(static_cast<std::size_t>(frames.size() - rejected_frame_indices.size()));
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            if (effective_frame_quality_weights[frame_index] > 0.0f)
            {
                retained_frames.push_back(frames[frame_index]);
                const DepthTsdfFrame& frame = frames[frame_index];
                const int source_count = std::min(static_cast<int>(kDepthGeometryLocalSourceSlotCount),
                                                  static_cast<int>(frame.geometrySourceIndices.size()));
                retained_source_indices.emplace_back(frame.geometrySourceIndices.cbegin(),
                                                     frame.geometrySourceIndices.cbegin() + source_count);
            }
        }
        const DepthGeometrySourceEncoding geometry_source_encoding =
            DepthGeometrySourceEncoding::build(retained_source_indices);
        std::vector<DepthGeometryLocalSourceEncoding> local_source_encodings(static_cast<std::size_t>(frames.size()));
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            if (effective_frame_quality_weights[frame_index] <= 0.0f)
            {
                continue;
            }
            const DepthTsdfFrame& frame = frames[frame_index];
            const int source_count = std::min(static_cast<int>(kDepthGeometryLocalSourceSlotCount),
                                              static_cast<int>(frame.geometrySourceIndices.size()));
            local_source_encodings[static_cast<std::size_t>(frame_index)] =
                geometry_source_encoding.makeLocalEncoding(std::vector<int>(
                    frame.geometrySourceIndices.cbegin(), frame.geometrySourceIndices.cbegin() + source_count));
        }
        const DepthTsdfBoundsResult bounds = DepthTsdfSurfaceBuilder::estimateBounds(
            retained_frames, options.useEvidenceAwareBounds, options.preservePerFrameCoverageBounds);
        if (!bounds.ok)
        {
            result.errorMessage = bounds.errorMessage;
            return result;
        }
        result = DepthTsdfSurfaceBuilder::validateAllocation(bounds.minimum, bounds.maximum, options);
        result.statistics.boundsCandidateSampleCount = bounds.candidateSampleCount;
        result.statistics.boundsTrustedSampleCount = bounds.trustedSampleCount;
        result.statistics.boundsSelectedSampleCount = bounds.sampleCount;
        result.statistics.boundsUsedEvidenceAwareSamples = bounds.usedEvidenceAwareSamples;
        result.statistics.boundsFellBackToCandidateSamples = bounds.fellBackToCandidateSamples;
        result.statistics.boundsSelectionReason = bounds.selectionReason;
        result.statistics.inputFrameCount = frames.size();
        result.statistics.acceptedFrameCount = retained_frames.size();
        const DepthGeometrySourceEncodingStatistics& source_encoding_statistics = geometry_source_encoding.statistics();
        result.statistics.geometrySourceEncodingFrameCount = source_encoding_statistics.frameCount;
        result.statistics.geometrySourceValidReferenceCount = source_encoding_statistics.validReferenceCount;
        result.statistics.geometrySourceIgnoredNegativeReferenceCount =
            source_encoding_statistics.ignoredNegativeReferenceCount;
        result.statistics.geometrySourceMappedReferenceCount = source_encoding_statistics.mappedReferenceCount;
        result.statistics.geometrySourceUnmappedReferenceCount = source_encoding_statistics.unmappedReferenceCount;
        result.statistics.geometrySourceDistinctCount =
            static_cast<int>(source_encoding_statistics.distinctSourceCount);
        result.statistics.geometrySourceMappedCount = static_cast<int>(source_encoding_statistics.mappedSourceCount);
        result.statistics.geometrySourceUnmappedCount =
            static_cast<int>(source_encoding_statistics.unmappedSourceCount);
        result.statistics.geometrySourceMappedReferenceRatio =
            source_encoding_statistics.validReferenceCount > 0
                ? static_cast<double>(source_encoding_statistics.mappedReferenceCount) /
                      static_cast<double>(source_encoding_statistics.validReferenceCount)
                : 1.0;
        for (const DepthGeometrySourceReferenceCount& source : geometry_source_encoding.sourceReferenceCounts())
        {
            if (source.mappedSlot < 0)
            {
                continue;
            }
            result.statistics.geometrySourceEncodingSlots.append(
                QJsonObject{{QStringLiteral("slot"), source.mappedSlot},
                            {QStringLiteral("source_index"), source.sourceIndex},
                            {QStringLiteral("reference_count"), static_cast<double>(source.referenceCount)}});
        }
        result.statistics.effectiveRobustFrameQualityWeighting = robust_frame_quality_weighting_enabled;
        result.statistics.robustFrameQualityDownweightedFrameCount = robust_frame_quality_downweighted_frame_count;
        result.statistics.robustFrameQualityMedian = robust_frame_quality_median;
        result.statistics.robustFrameQualityScale = robust_frame_quality_scale;
        result.statistics.robustFrameQualityMinimumEffectiveWeight = robust_frame_quality_minimum_effective_weight;
        result.statistics.effectiveRobustFrameQualityRejection = robust_frame_quality_rejection_enabled;
        result.statistics.robustFrameQualityRejectedFrameCount = rejected_frame_indices.size();
        result.statistics.robustFrameQualityRejectedRefIndices = rejected_frame_ref_indices;
        result.statistics.auxiliarySurfaceOnlyFrameCount = auxiliary_surface_only_ref_indices.size();
        result.statistics.auxiliarySurfaceOnlyRefIndices = auxiliary_surface_only_ref_indices;
        result.statistics.effectiveOrbitalFrameCoverageProtection = options.enableOrbitalFrameCoverageProtection;
        result.statistics.orbitalCoverageProtectedFrameCount = coverage_protected_ref_indices.size();
        result.statistics.orbitalCoverageProtectedRefIndices = coverage_protected_ref_indices;
        result.statistics.orbitalMedianAngularSpacingDegrees = orbital_coverage.medianAngularSpacingDegrees;
        result.statistics.orbitalMaximumAngularGapDegrees = orbital_coverage.maximumAngularGapDegrees;
        result.statistics.orbitalMaximumAngularGapRatio = orbital_coverage.maximumAngularGapRatio;
        result.statistics.orbitalSignificantAngularGap = orbital_coverage.significantGap;
        result.statistics.orbitalGapStartRefIndex = orbital_coverage.gapStartRefIndex;
        result.statistics.orbitalGapEndRefIndex = orbital_coverage.gapEndRefIndex;
        result.statistics.orbitalGapOppositeRefIndex = orbital_coverage.gapOppositeRefIndex;
        result.statistics.orbitalFrameRoles = orbital_frame_roles;
        result.statistics.depthCompletenessFrameRoles = depth_completeness_frame_roles;
        result.statistics.effectiveOrbitalGapBoundaryRecovery =
            options.enableOrbitalGapBoundaryRecovery && orbital_coverage.significantGap;
        result.statistics.orbitalGapQualityFloorFrameCount = gap_quality_floor_ref_indices.size();
        result.statistics.orbitalGapQualityFloorRefIndices = gap_quality_floor_ref_indices;
        result.statistics.effectiveOrbitalGapBoundaryMinimumObservationWeight =
            options.orbitalGapBoundaryMinimumObservationWeight;
        if (!result.ok)
        {
            return result;
        }
        result.ok = false;
        std::vector<std::uint8_t> orbital_gap_boundary_frames(static_cast<std::size_t>(frames.size()), 0);
        if (result.statistics.effectiveOrbitalGapBoundaryRecovery)
        {
            for (const OrbitalFrameRoleAssignment& assignment : orbital_coverage.frameRoles)
            {
                if (assignment.role == OrbitalFrameRole::GapBoundary && assignment.frameIndex >= 0 &&
                    assignment.frameIndex < frames.size())
                {
                    orbital_gap_boundary_frames[static_cast<std::size_t>(assignment.frameIndex)] = 1;
                }
            }
        }
        if (options.execution.progress)
        {
            options.execution.reportProgress((QStringLiteral("正在融合置信度加权 TSDF...")).toUtf8().toStdString(),
                                             (5) / 100.0);
        }

        std::vector<float> tsdf;
        std::vector<float> weight;
        std::vector<float> evidenceSupportWeight;
        std::vector<float> maximumObservationWeight;
        std::vector<float> maximumEvidenceSupportObservationWeight;
        std::vector<std::uint16_t> maximumGeometrySupportCount;
        std::vector<std::uint8_t> strongAdaptiveSurfaceObservation;
        std::vector<std::uint16_t> support;
        std::vector<DepthGeometrySourceMask> geometrySourceMask;
        std::vector<std::uint16_t> minimumInverseDepthSpread;
        std::vector<float> surfaceTsdfWeightedSum;
        std::vector<float> surfaceObservationWeight;
        std::vector<std::uint8_t> crossViewRepairedSurfaceWeight;
        std::vector<float> orbitalGapBoundaryObservationWeight;
        std::vector<float> contourBandObservationWeight;
        std::vector<DepthVisibilityHistogram> visibilityHistograms;
        std::vector<std::uint8_t> primarySurfaceObservation;
        try
        {
            tsdf.assign(static_cast<std::size_t>(result.layout.sampleCount), 1.0f);
            weight.assign(static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
            evidenceSupportWeight.assign(static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
            maximumObservationWeight.assign(static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
            maximumEvidenceSupportObservationWeight.assign(static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
            maximumGeometrySupportCount.assign(static_cast<std::size_t>(result.layout.sampleCount), 0);
            strongAdaptiveSurfaceObservation.assign(static_cast<std::size_t>(result.layout.sampleCount), 0);
            support.assign(static_cast<std::size_t>(result.layout.sampleCount), 0);
            if (options.enableAuxiliaryBridgeOnlyIntegration)
            {
                primarySurfaceObservation.assign(static_cast<std::size_t>(result.layout.sampleCount), 0);
            }
            if (options.enableSurfacePatchSupport || options.enableContourBandZeroCrossingSupport ||
                options.enableMeasuredSupportConnectivity || options.collectZeroCrossingDiagnostics ||
                options.collectAcquisitionGapReport || options.enableCrossViewAnchoredSurfaceRecovery ||
                options.enableGlobalImplicitRegularization || options.enableAdaptiveTgvRegularization ||
                result.statistics.effectiveOrbitalGapBoundaryRecovery)
            {
                geometrySourceMask.assign(static_cast<std::size_t>(result.layout.sampleCount), 0);
                minimumInverseDepthSpread.assign(static_cast<std::size_t>(result.layout.sampleCount),
                                                 std::numeric_limits<std::uint16_t>::max());
                surfaceTsdfWeightedSum.assign(static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
                surfaceObservationWeight.assign(static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
                if (options.enableCrossViewAnchoredSurfaceRecovery)
                {
                    crossViewRepairedSurfaceWeight.assign(static_cast<std::size_t>(result.layout.sampleCount), 0);
                }
                if (result.statistics.effectiveOrbitalGapBoundaryRecovery)
                {
                    orbitalGapBoundaryObservationWeight.assign(static_cast<std::size_t>(result.layout.sampleCount),
                                                               0.0f);
                }
                if (options.enableContourBandZeroCrossingSupport)
                {
                    contourBandObservationWeight.assign(static_cast<std::size_t>(result.layout.sampleCount), 0.0f);
                }
            }
            if (options.enableAdaptiveTgvRegularization)
            {
                visibilityHistograms.resize(static_cast<std::size_t>(result.layout.sampleCount));
            }
        }
        catch (const std::bad_alloc&)
        {
            result.errorMessage = QStringLiteral("TSDF allocation failed: resolution=%1 required=%2 bytes")
                                      .arg(options.resolution)
                                      .arg(result.layout.requiredBytes);
            return result;
        }

        const float maximum_voxel_size =
            std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]});
        const float base_truncation_voxels = std::max(1.0f, options.truncationVoxels);
        const DepthUncertaintyBandEstimate uncertainty_band =
            estimateDepthUncertaintyBand(retained_frames, options, maximum_voxel_size);
        const bool uncertainty_adaptation_available =
            options.enableUncertaintyAdaptiveTruncation &&
            uncertainty_band.p90Voxels >
                base_truncation_voxels * std::max(1.0f, options.uncertaintyAdaptiveActivationRatio);
        const bool orbital_gap_adaptive_truncation = uncertainty_adaptation_available &&
                                                     options.enableOrbitalGapAdaptiveTruncation &&
                                                     orbital_coverage.significantGap;
        const float effective_uncertainty_adaptive_scale =
            orbital_gap_adaptive_truncation
                ? std::max(options.uncertaintyAdaptiveScale, options.orbitalGapAdaptiveTruncationScale)
                : options.uncertaintyAdaptiveScale;
        const float adaptive_maximum_truncation_voxels =
            std::max({base_truncation_voxels,
                      options.uncertaintyAdaptiveMaximumTruncationVoxels,
                      orbital_gap_adaptive_truncation ? options.orbitalGapAdaptiveMaximumTruncationVoxels
                                                      : base_truncation_voxels});
        const float uncertainty_adaptive_added_voxels =
            uncertainty_adaptation_available
                ? std::max(0.0f, effective_uncertainty_adaptive_scale) * uncertainty_band.p90Voxels
                : 0.0f;
        const float effective_truncation_voxels = std::clamp(base_truncation_voxels + uncertainty_adaptive_added_voxels,
                                                             base_truncation_voxels,
                                                             adaptive_maximum_truncation_voxels);
        const float base_surface_support_band_voxels =
            options.surfaceSupportBandVoxels > 0.0f
                ? std::clamp(options.surfaceSupportBandVoxels, 0.5f, base_truncation_voxels)
                : base_truncation_voxels;
        const float effective_surface_support_band_voxels =
            std::clamp(base_surface_support_band_voxels + uncertainty_adaptive_added_voxels,
                       base_surface_support_band_voxels,
                       effective_truncation_voxels);
        const float truncation = maximum_voxel_size * effective_truncation_voxels;
        const float surface_support_distance = maximum_voxel_size * effective_surface_support_band_voxels;
        const float weak_evidence_surface_band_voxels =
            options.weakEvidenceSurfaceBandVoxels > 0.0f
                ? std::clamp(options.weakEvidenceSurfaceBandVoxels, 0.5f, effective_truncation_voxels)
                : effective_surface_support_band_voxels;
        const float weak_evidence_surface_distance = maximum_voxel_size * weak_evidence_surface_band_voxels;
        const float maximum_free_space_distance =
            options.maximumFreeSpaceVoxels > 0.0f
                ? std::max({result.layout.voxelSize[0], result.layout.voxelSize[1], result.layout.voxelSize[2]}) *
                      std::max(effective_truncation_voxels, options.maximumFreeSpaceVoxels)
                : std::numeric_limits<float>::infinity();
        DepthTsdfNarrowBandActivation narrow_band_activation;
        if (options.enableNarrowBandActivation)
        {
            if (options.execution.progress)
            {
                options.execution.reportProgress(
                    (QStringLiteral("正在激活有效深度附近的 TSDF 窄带...")).toUtf8().toStdString(), (4) / 100.0);
            }
            std::vector<DepthTsdfNarrowBandFrameView> activation_frames;
            activation_frames.reserve(static_cast<std::size_t>(frames.size()));
            for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
            {
                if (effective_frame_quality_weights[frame_index] <= 0.0f)
                {
                    continue;
                }
                const DepthTsdfFrame& frame = frames[frame_index];
                if ((options.enableAuxiliaryBridgeOnlyIntegration ||
                     options.enableAuxiliaryPrimaryNeighborhoodConstraint) &&
                    frame.auxiliarySurfaceOnly)
                {
                    // Bridge evidence may only extend the primary observed field
                    // inside its small spatial halo; it must not activate an
                    // independent shell elsewhere in the volume.
                    continue;
                }
                DepthTsdfNarrowBandFrameView view;
                view.camera = &frame.camera;
                view.depth = &frame.depth;
                view.depthValidMask =
                    erosion_pixels > 0 ? &effective_depth_valid_masks[frame_index] : &frame.depthValidMask;
                view.supportMask = &frame.supportMask;
                activation_frames.push_back(view);
            }

            DepthTsdfNarrowBandActivationOptions activation_options;
            activation_options.blockSizeSamples = std::clamp(options.narrowBandActivationBlockSizeSamples, 2, 32);
            activation_options.depthStride = std::clamp(options.narrowBandActivationDepthStride, 1, 16);
            activation_options.truncationDistance = truncation;
            activation_options.rayStepVoxels = std::clamp(options.narrowBandActivationRayStepVoxels, 0.25f, 4.0f);
            activation_options.haloBlocks = std::clamp(options.narrowBandActivationHaloBlocks, 0, 3);
            activation_options.isCancelled = [control = options.execution]() { return control.isCancelled(); };
            if (!narrow_band_activation.build(result.layout, activation_frames, activation_options))
            {
                result.errorMessage = narrow_band_activation.wasCancelled()
                                          ? QStringLiteral("TSDF 窄带激活已取消")
                                          : QStringLiteral("TSDF 窄带激活失败：输入布局或参数无效");
                return result;
            }
            const DepthTsdfNarrowBandActivationStatistics& activation_statistics = narrow_band_activation.statistics();
            if (activation_statistics.activeBlocks == 0)
            {
                result.errorMessage = QStringLiteral("TSDF 窄带激活没有找到可用深度样本；"
                                                     "无效深度和蒙版外区域保持为 unknown");
                return result;
            }
            result.statistics.narrowBandActivationTotalBlockCount = activation_statistics.totalBlocks;
            result.statistics.narrowBandActivationActiveBlockCount = activation_statistics.activeBlocks;
            result.statistics.narrowBandActivationValidSourceSampleCount = activation_statistics.validSourceSamples;
            result.statistics.narrowBandActivationMarkedRaySampleCount = activation_statistics.markedRaySamples;
        }
        std::vector<std::uint8_t> primary_bridge_reach;
        if (!primarySurfaceObservation.empty())
        {
            if (options.execution.progress)
            {
                options.execution.reportProgress(
                    (QStringLiteral("正在约束实测桥接到主观测表面邻域...")).toUtf8().toStdString(), (5) / 100.0);
            }
            std::atomic_bool bridge_prepass_cancelled{false};
            const int prepass_z_samples = result.layout.cells[2] + 1;
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int z = 0; z < prepass_z_samples; ++z)
            {
                if (bridge_prepass_cancelled.load(std::memory_order_relaxed) || (options.execution.isCancelled()))
                {
                    bridge_prepass_cancelled.store(true, std::memory_order_relaxed);
                    continue;
                }
                const double world_z = result.layout.boundsMin[2] + result.layout.voxelSize[2] * static_cast<float>(z);
                for (int y = 0; y <= result.layout.cells[1]; ++y)
                {
                    const double world_y =
                        result.layout.boundsMin[1] + result.layout.voxelSize[1] * static_cast<float>(y);
                    for (int x = 0; x <= result.layout.cells[0]; ++x)
                    {
                        if (options.enableNarrowBandActivation && !narrow_band_activation.isSampleActive(x, y, z))
                        {
                            continue;
                        }
                        const double world[3]{result.layout.boundsMin[0] +
                                                  result.layout.voxelSize[0] * static_cast<float>(x),
                                              world_y,
                                              world_z};
                        const std::size_t index = sampleIndex(result.layout, x, y, z);
                        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
                        {
                            const DepthTsdfFrame& frame = frames[frame_index];
                            if (frame.auxiliarySurfaceOnly || effective_frame_quality_weights[frame_index] <= 0.0f)
                            {
                                continue;
                            }
                            double pixel[2]{};
                            double voxel_depth = 0.0;
                            if (!frame.camera.projectWorldPointWithDepth(world, pixel, voxel_depth))
                            {
                                continue;
                            }
                            const cv::Mat& depth_valid_mask = !frame.useAdaptiveGeometryEvidence && erosion_pixels > 0
                                                                  ? effective_depth_valid_masks[frame_index]
                                                                  : frame.depthValidMask;
                            const DepthTsdfObservationSample observation = DepthTsdfSurfaceBuilder::sampleObservation(
                                frame,
                                depth_valid_mask,
                                cv::Point2d(pixel[0], pixel[1]),
                                options.minimumConfidence,
                                options.enableDiscontinuityAwareSampling,
                                options.maximumInterpolationRelativeDepthSpread,
                                options.maximumObservationInverseDepthSpread,
                                options.allowInvalidNearestPixelRecovery,
                                options.maximumInvalidNearestPixelRecoveryInverseDepthSpread,
                                options.enableCrossViewConsensusDepth,
                                options.maximumCrossViewConsensusInverseDepthSpread,
                                options.crossViewConsensusContourBandOnly ? contour_band_masks[frame_index] : cv::Mat(),
                                options.enableCrossViewConsensusDepth ? reference_anchored_consensus_depths[frame_index]
                                                                      : cv::Mat(),
                                options.excludeAnchoredInterpolationObservations);
                            if (observation.valid && std::fabs(observation.depth - static_cast<float>(voxel_depth)) <=
                                                         surface_support_distance)
                            {
                                primarySurfaceObservation[index] = 1;
                                break;
                            }
                        }
                    }
                }
            }
            if (bridge_prepass_cancelled.load(std::memory_order_relaxed))
            {
                result.errorMessage = QStringLiteral("TSDF auxiliary bridge prepass cancelled");
                return result;
            }
            primary_bridge_reach = dilateSampleMask(result.layout,
                                                    primarySurfaceObservation,
                                                    std::clamp(options.enableAuxiliaryBridgeOnlyIntegration
                                                                   ? options.auxiliaryBridgeMaximumExtensionVoxels
                                                                   : options.auxiliaryPrimaryNeighborhoodVoxels,
                                                               1,
                                                               4));
        }
        std::atomic_bool cancelled{false};
        std::atomic<int> completed_z_slices{0};
        std::atomic<int> last_progress_percent{5};
        std::mutex progress_callback_mutex;
        const int zSamples = result.layout.cells[2] + 1;
#ifdef MESHING_OPENMP
        const int workerCount =
            std::max(1, std::min(zSamples, options.workerCount > 0 ? options.workerCount : omp_get_max_threads()));
#else
        const int workerCount = 1;
#endif
        result.statistics.effectiveWorkerCount = workerCount;
        if (!integrateDepthFrames(frames,
                                  options,
                                  result,
                                  {effective_frame_quality_weights,
                                   effective_depth_valid_masks,
                                   erosion_pixels,
                                   boundary_recovered_depth_valid_pixel_count,
                                   reference_anchored_consensus_depths,
                                   contour_band_masks,
                                   cross_view_consensus_contour_band_pixel_count,
                                   local_source_encodings,
                                   orbital_gap_boundary_frames,
                                   tsdf,
                                   weight,
                                   evidenceSupportWeight,
                                   maximumObservationWeight,
                                   maximumEvidenceSupportObservationWeight,
                                   maximumGeometrySupportCount,
                                   strongAdaptiveSurfaceObservation,
                                   support,
                                   geometrySourceMask,
                                   minimumInverseDepthSpread,
                                   surfaceTsdfWeightedSum,
                                   surfaceObservationWeight,
                                   crossViewRepairedSurfaceWeight,
                                   orbitalGapBoundaryObservationWeight,
                                   contourBandObservationWeight,
                                   visibilityHistograms,
                                   primarySurfaceObservation,
                                   base_truncation_voxels,
                                   uncertainty_band,
                                   uncertainty_adaptation_available,
                                   orbital_gap_adaptive_truncation,
                                   effective_uncertainty_adaptive_scale,
                                   adaptive_maximum_truncation_voxels,
                                   effective_truncation_voxels,
                                   effective_surface_support_band_voxels,
                                   truncation,
                                   surface_support_distance,
                                   weak_evidence_surface_band_voxels,
                                   weak_evidence_surface_distance,
                                   maximum_free_space_distance,
                                   narrow_band_activation,
                                   primary_bridge_reach,
                                   cancelled,
                                   completed_z_slices,
                                   last_progress_percent,
                                   progress_callback_mutex,
                                   zSamples,
                                   workerCount}))
        {
            return result;
        }

        std::vector<std::uint8_t> supported(static_cast<std::size_t>(result.layout.sampleCount), 0);
        std::vector<std::uint8_t> adaptiveTgvExtractionSupport;
        std::vector<float> visual_hull_completion_tsdf;
        std::vector<std::uint8_t> visual_hull_completion_support;
        std::array<float, 3> native_carrier_bounds_min{};
        std::array<float, 3> native_carrier_bounds_max{};
        std::array<int, 3> native_carrier_dimensions{};
        std::array<int, 3> native_carrier_cells{};
        std::vector<std::uint8_t> native_carrier_occupied;
        std::vector<float> native_carrier_field;
        if (!recoverVolumeSupport(frames,
                                  options,
                                  result,
                                  {supported,
                                   effective_frame_quality_weights,
                                   effective_depth_valid_masks,
                                   erosion_pixels,
                                   retained_frames,
                                   tsdf,
                                   weight,
                                   evidenceSupportWeight,
                                   maximumObservationWeight,
                                   maximumEvidenceSupportObservationWeight,
                                   maximumGeometrySupportCount,
                                   strongAdaptiveSurfaceObservation,
                                   support,
                                   geometrySourceMask,
                                   minimumInverseDepthSpread,
                                   surfaceTsdfWeightedSum,
                                   surfaceObservationWeight,
                                   crossViewRepairedSurfaceWeight,
                                   orbitalGapBoundaryObservationWeight,
                                   contourBandObservationWeight,
                                   visibilityHistograms,
                                   truncation,
                                   cancelled,
                                   workerCount,
                                   adaptiveTgvExtractionSupport,
                                   visual_hull_completion_tsdf,
                                   visual_hull_completion_support,
                                   native_carrier_bounds_min,
                                   native_carrier_bounds_max,
                                   native_carrier_dimensions,
                                   native_carrier_cells,
                                   native_carrier_occupied,
                                   native_carrier_field}))
        {
            return result;
        }

        extractAndPostprocessSurface(frames,
                                     options,
                                     result,
                                     {effective_frame_quality_weights,
                                      effective_depth_valid_masks,
                                      erosion_pixels,
                                      geometry_source_encoding,
                                      tsdf,
                                      weight,
                                      strongAdaptiveSurfaceObservation,
                                      support,
                                      geometrySourceMask,
                                      minimumInverseDepthSpread,
                                      surfaceObservationWeight,
                                      maximum_voxel_size,
                                      truncation,
                                      cancelled,
                                      workerCount,
                                      supported,
                                      adaptiveTgvExtractionSupport,
                                      visual_hull_completion_tsdf,
                                      visual_hull_completion_support,
                                      native_carrier_bounds_min,
                                      native_carrier_bounds_max,
                                      native_carrier_dimensions,
                                      native_carrier_cells,
                                      native_carrier_occupied,
                                      native_carrier_field});
        return result;
    }

} // namespace xjw::mesh
