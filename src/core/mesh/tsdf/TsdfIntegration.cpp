#include "DepthTsdfStages.h"
namespace xjw::mesh
{
    using namespace tsdf_detail;
    bool tsdf_detail::integrateDepthFrames(const QVector<DepthTsdfFrame>& frames,
                                           const DepthTsdfOptions& options,
                                           DepthTsdfResult& result,
                                           TsdfIntegrationState state)
    {
        auto& effective_frame_quality_weights = state.effective_frame_quality_weights;
        auto& effective_depth_valid_masks = state.effective_depth_valid_masks;
        auto& erosion_pixels = state.erosion_pixels;
        auto& boundary_recovered_depth_valid_pixel_count = state.boundary_recovered_depth_valid_pixel_count;
        auto& reference_anchored_consensus_depths = state.reference_anchored_consensus_depths;
        auto& contour_band_masks = state.contour_band_masks;
        auto& cross_view_consensus_contour_band_pixel_count = state.cross_view_consensus_contour_band_pixel_count;
        auto& local_source_encodings = state.local_source_encodings;
        auto& orbital_gap_boundary_frames = state.orbital_gap_boundary_frames;
        auto& tsdf = state.tsdf;
        auto& weight = state.weight;
        auto& evidenceSupportWeight = state.evidenceSupportWeight;
        auto& maximumObservationWeight = state.maximumObservationWeight;
        auto& maximumEvidenceSupportObservationWeight = state.maximumEvidenceSupportObservationWeight;
        auto& maximumGeometrySupportCount = state.maximumGeometrySupportCount;
        auto& strongAdaptiveSurfaceObservation = state.strongAdaptiveSurfaceObservation;
        auto& support = state.support;
        auto& geometrySourceMask = state.geometrySourceMask;
        auto& minimumInverseDepthSpread = state.minimumInverseDepthSpread;
        auto& surfaceTsdfWeightedSum = state.surfaceTsdfWeightedSum;
        auto& surfaceObservationWeight = state.surfaceObservationWeight;
        auto& crossViewRepairedSurfaceWeight = state.crossViewRepairedSurfaceWeight;
        auto& orbitalGapBoundaryObservationWeight = state.orbitalGapBoundaryObservationWeight;
        auto& contourBandObservationWeight = state.contourBandObservationWeight;
        auto& visibilityHistograms = state.visibilityHistograms;
        auto& primarySurfaceObservation = state.primarySurfaceObservation;
        auto& base_truncation_voxels = state.base_truncation_voxels;
        auto& uncertainty_band = state.uncertainty_band;
        auto& uncertainty_adaptation_available = state.uncertainty_adaptation_available;
        auto& orbital_gap_adaptive_truncation = state.orbital_gap_adaptive_truncation;
        auto& effective_uncertainty_adaptive_scale = state.effective_uncertainty_adaptive_scale;
        auto& adaptive_maximum_truncation_voxels = state.adaptive_maximum_truncation_voxels;
        auto& effective_truncation_voxels = state.effective_truncation_voxels;
        auto& effective_surface_support_band_voxels = state.effective_surface_support_band_voxels;
        auto& truncation = state.truncation;
        auto& surface_support_distance = state.surface_support_distance;
        auto& weak_evidence_surface_band_voxels = state.weak_evidence_surface_band_voxels;
        auto& weak_evidence_surface_distance = state.weak_evidence_surface_distance;
        auto& maximum_free_space_distance = state.maximum_free_space_distance;
        auto& narrow_band_activation = state.narrow_band_activation;
        auto& primary_bridge_reach = state.primary_bridge_reach;
        auto& cancelled = state.cancelled;
        auto& completed_z_slices = state.completed_z_slices;
        auto& last_progress_percent = state.last_progress_percent;
        auto& progress_callback_mutex = state.progress_callback_mutex;
        auto& zSamples = state.zSamples;
        auto& workerCount = state.workerCount;
        unsigned long long integratedVoxelUpdates = 0;
        unsigned long long narrowBandActivationSkippedSampleCount = 0;
        unsigned long long rejectedProjectionCount = 0;
        unsigned long long rejectedSupportMaskCount = 0;
        unsigned long long supportMaskFreeSpaceUpdateCount = 0;
        unsigned long long supportMaskFreeSpaceSurfaceVetoCount = 0;
        unsigned long long auxiliaryOutsideSurfaceBandRejectedCount = 0;
        unsigned long long auxiliaryOutsidePrimaryNeighborhoodRejectedCount = 0;
        unsigned long long auxiliaryBridgeRejectedGeometrySupportCount = 0;
        unsigned long long auxiliaryBridgeRejectedSourceCount = 0;
        unsigned long long auxiliaryBridgeRejectedSpreadCount = 0;
        unsigned long long auxiliaryBridgeRejectedExtensionSampleCount = 0;
        unsigned long long rejectedDepthValidCount = 0;
        unsigned long long rejectedInterpolationPolicyCount = 0;
        unsigned long long rejectedDepthCount = 0;
        unsigned long long rejectedConfidenceCount = 0;
        unsigned long long subpixelObservationCount = 0;
        unsigned long long recoveredNeighborObservationCount = 0;
        unsigned long long discontinuityRejectedCandidateCount = 0;
        unsigned long long rejectedGeometryConsistencyCount = 0;
        unsigned long long rejectedInvalidNearestPixelRecoveryCount = 0;
        unsigned long long crossViewConsensusDepthObservationCount = 0;
        unsigned long long unconfirmedNativeObservationCount = 0;
        unsigned long long weakNativeObservationCount = 0;
        unsigned long long repairedObservationCount = 0;
        unsigned long long strongNativeObservationCount = 0;
        unsigned long long inverseDepthSpreadDownweightedObservationCount = 0;
        unsigned long long inverseDepthSpreadVeryWeakObservationCount = 0;
        unsigned long long inverseDepthSpreadSupportLiftedObservationCount = 0;
        unsigned long long weakEvidenceOutsideSurfaceBandRejectedCount = 0;
        unsigned long long geometrySourceNonzeroLocalMaskObservationCount = 0;
        unsigned long long geometrySourceFullyMappedObservationCount = 0;
        unsigned long long geometrySourcePartiallyMappedObservationCount = 0;
        unsigned long long geometrySourceFullyUnmappedObservationCount = 0;
        const auto integration_start = std::chrono::steady_clock::now();
        const double integration_cpu_start = detail::processCpuTimeMilliseconds();
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static) num_threads(workerCount)                                                     \
    reduction(+ : integratedVoxelUpdates,                                                                              \
                  narrowBandActivationSkippedSampleCount,                                                              \
                  rejectedProjectionCount,                                                                             \
                  rejectedSupportMaskCount,                                                                            \
                  supportMaskFreeSpaceUpdateCount,                                                                     \
                  supportMaskFreeSpaceSurfaceVetoCount,                                                                \
                  auxiliaryOutsideSurfaceBandRejectedCount,                                                            \
                  auxiliaryOutsidePrimaryNeighborhoodRejectedCount,                                                    \
                  auxiliaryBridgeRejectedGeometrySupportCount,                                                         \
                  auxiliaryBridgeRejectedSourceCount,                                                                  \
                  auxiliaryBridgeRejectedSpreadCount,                                                                  \
                  auxiliaryBridgeRejectedExtensionSampleCount,                                                         \
                  rejectedDepthValidCount,                                                                             \
                  rejectedInterpolationPolicyCount,                                                                    \
                  rejectedDepthCount,                                                                                  \
                  rejectedConfidenceCount,                                                                             \
                  subpixelObservationCount,                                                                            \
                  recoveredNeighborObservationCount,                                                                   \
                  discontinuityRejectedCandidateCount,                                                                 \
                  rejectedGeometryConsistencyCount,                                                                    \
                  rejectedInvalidNearestPixelRecoveryCount,                                                            \
                  crossViewConsensusDepthObservationCount,                                                             \
                  unconfirmedNativeObservationCount,                                                                   \
                  weakNativeObservationCount,                                                                          \
                  repairedObservationCount,                                                                            \
                  strongNativeObservationCount,                                                                        \
                  inverseDepthSpreadDownweightedObservationCount,                                                      \
                  inverseDepthSpreadVeryWeakObservationCount,                                                          \
                  inverseDepthSpreadSupportLiftedObservationCount,                                                     \
                  weakEvidenceOutsideSurfaceBandRejectedCount)                                                         \
    reduction(+ : geometrySourceNonzeroLocalMaskObservationCount,                                                      \
                  geometrySourceFullyMappedObservationCount,                                                           \
                  geometrySourcePartiallyMappedObservationCount,                                                       \
                  geometrySourceFullyUnmappedObservationCount)
#endif
        for (int z = 0; z < zSamples; ++z)
        {
            if (cancelled.load(std::memory_order_relaxed) || (options.execution.isCancelled()))
            {
                cancelled.store(true, std::memory_order_relaxed);
                continue;
            }
            const double worldZ = result.layout.boundsMin[2] + result.layout.voxelSize[2] * static_cast<float>(z);
            for (int y = 0; y <= result.layout.cells[1]; ++y)
            {
                const double worldY = result.layout.boundsMin[1] + result.layout.voxelSize[1] * static_cast<float>(y);
                for (int x = 0; x <= result.layout.cells[0]; ++x)
                {
                    const double world[3] = {result.layout.boundsMin[0] +
                                                 result.layout.voxelSize[0] * static_cast<float>(x),
                                             worldY,
                                             worldZ};
                    const std::size_t index = sampleIndex(result.layout, x, y, z);
                    if (options.enableNarrowBandActivation && !narrow_band_activation.isSampleActive(x, y, z))
                    {
                        ++narrowBandActivationSkippedSampleCount;
                        continue;
                    }
                    int support_mask_free_space_votes = 0;
                    for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
                    {
                        if (effective_frame_quality_weights[frame_index] <= 0.0f)
                        {
                            continue;
                        }
                        const DepthTsdfFrame& frame = frames[frame_index];
                        if (frame.auxiliarySurfaceOnly && options.enableAuxiliaryPrimaryNeighborhoodConstraint &&
                            !options.enableAuxiliaryBridgeOnlyIntegration &&
                            (primary_bridge_reach.empty() || primary_bridge_reach[index] == 0))
                        {
                            ++auxiliaryOutsidePrimaryNeighborhoodRejectedCount;
                            continue;
                        }
                        if (frame.auxiliarySurfaceOnly && options.enableAuxiliaryBridgeOnlyIntegration &&
                            (primary_bridge_reach.empty() || primary_bridge_reach[index] == 0 ||
                             primarySurfaceObservation[index] != 0))
                        {
                            ++auxiliaryBridgeRejectedExtensionSampleCount;
                            continue;
                        }
                        double pixel[2]{};
                        double voxelDepth = 0.0;
                        if (!frame.camera.projectWorldPointWithDepth(world, pixel, voxelDepth))
                        {
                            ++rejectedProjectionCount;
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
                        if (!observation.valid)
                        {
                            switch (observation.failure)
                            {
                            case DepthTsdfObservationFailure::SupportMask:
                                if (options.enableSupportMaskFreeSpaceCarving && !frame.auxiliarySurfaceOnly &&
                                    !frame.useAdaptiveGeometryEvidence)
                                {
                                    ++support_mask_free_space_votes;
                                }
                                ++rejectedSupportMaskCount;
                                break;
                            case DepthTsdfObservationFailure::DepthValid:
                                ++rejectedDepthValidCount;
                                break;
                            case DepthTsdfObservationFailure::InterpolationPolicy:
                                ++rejectedInterpolationPolicyCount;
                                break;
                            case DepthTsdfObservationFailure::Depth:
                                ++rejectedDepthCount;
                                break;
                            case DepthTsdfObservationFailure::Confidence:
                                ++rejectedConfidenceCount;
                                break;
                            case DepthTsdfObservationFailure::GeometryConsistency:
                                ++rejectedGeometryConsistencyCount;
                                break;
                            case DepthTsdfObservationFailure::Projection:
                            case DepthTsdfObservationFailure::None:
                            default:
                                ++rejectedProjectionCount;
                                break;
                            }
                            rejectedInvalidNearestPixelRecoveryCount += observation.rejectedInvalidNearestPixelRecovery;
                            continue;
                        }
                        subpixelObservationCount += observation.contributingPixelCount > 1;
                        recoveredNeighborObservationCount += observation.recoveredFromInvalidNearestPixel;
                        discontinuityRejectedCandidateCount += observation.discontinuityRejectedPixelCount;
                        crossViewConsensusDepthObservationCount += observation.usedCrossViewConsensusDepth;
                        bool contour_band_evidence = false;
                        if (options.enableContourBandZeroCrossingSupport && !contour_band_masks.empty())
                        {
                            const int contour_column = static_cast<int>(std::lround(pixel[0]));
                            const int contour_row = static_cast<int>(std::lround(pixel[1]));
                            contour_band_evidence =
                                contour_row >= 0 && contour_row < contour_band_masks[frame_index].rows &&
                                contour_column >= 0 && contour_column < contour_band_masks[frame_index].cols &&
                                contour_band_masks[frame_index].at<std::uint8_t>(contour_row, contour_column) != 0;
                        }
                        const float observedDepth = observation.depth;
                        const float confidence = observation.confidence;
                        if (frame.auxiliarySurfaceOnly && options.enableAuxiliaryBridgeOnlyIntegration)
                        {
                            if (observation.geometrySupportCount <
                                std::max(1, options.auxiliaryBridgeMinimumGeometrySupport))
                            {
                                ++auxiliaryBridgeRejectedGeometrySupportCount;
                                continue;
                            }
                            if (bitCount(observation.geometrySourceMask) <
                                std::max(1, options.auxiliaryBridgeMinimumSourceCount))
                            {
                                ++auxiliaryBridgeRejectedSourceCount;
                                continue;
                            }
                            if (!std::isfinite(observation.inverseDepthRelativeSpread) ||
                                observation.inverseDepthRelativeSpread < 0.0f ||
                                observation.inverseDepthRelativeSpread >
                                    std::max(0.0f, options.auxiliaryBridgeMaximumInverseDepthSpread))
                            {
                                ++auxiliaryBridgeRejectedSpreadCount;
                                continue;
                            }
                        }
                        const float signedDistance = observedDepth - static_cast<float>(voxelDepth);
                        if (frame.auxiliarySurfaceOnly && std::fabs(signedDistance) > surface_support_distance)
                        {
                            ++auxiliaryOutsideSurfaceBandRejectedCount;
                            continue;
                        }
                        if (DepthTsdfSurfaceBuilder::observationUsesSurfaceOnlyIntegration(observation, options) &&
                            std::fabs(signedDistance) > weak_evidence_surface_distance)
                        {
                            ++weakEvidenceOutsideSurfaceBandRejectedCount;
                            continue;
                        }
                        const float evidence_weight_multiplier =
                            DepthTsdfSurfaceBuilder::observationEvidenceWeightMultiplier(observation, options);
                        const float evidence_support_weight_multiplier =
                            DepthTsdfSurfaceBuilder::observationEvidenceSupportWeightMultiplier(observation, options);
                        const float spread_weight_multiplier =
                            DepthTsdfSurfaceBuilder::observationInverseDepthSpreadWeightMultiplier(observation,
                                                                                                   options);
                        const float spread_support_weight_multiplier =
                            DepthTsdfSurfaceBuilder::observationInverseDepthSpreadSupportWeightMultiplier(observation,
                                                                                                          options);
                        const float observationWeight = confidence * effective_frame_quality_weights[frame_index] *
                                                        evidence_weight_multiplier * spread_weight_multiplier;
                        const float evidenceSupportObservationWeight =
                            confidence * effective_frame_quality_weights[frame_index] *
                            evidence_support_weight_multiplier * spread_support_weight_multiplier;
                        if (spread_weight_multiplier < 1.0f)
                        {
                            ++inverseDepthSpreadDownweightedObservationCount;
                        }
                        if (options.enableInverseDepthSpreadWeighting &&
                            spread_weight_multiplier <= options.minimumInverseDepthSpreadWeightMultiplier + 1.0e-6f)
                        {
                            ++inverseDepthSpreadVeryWeakObservationCount;
                        }
                        if (spread_support_weight_multiplier > spread_weight_multiplier + 1.0e-6f)
                        {
                            ++inverseDepthSpreadSupportLiftedObservationCount;
                        }
                        if (observation.usedCrossViewRepairedDepth)
                        {
                            ++repairedObservationCount;
                        }
                        else if (observation.geometrySupportCount == 0)
                        {
                            ++unconfirmedNativeObservationCount;
                        }
                        else if (observation.geometrySupportCount == 1)
                        {
                            ++weakNativeObservationCount;
                        }
                        else
                        {
                            ++strongNativeObservationCount;
                        }
                        if (!visibilityHistograms.empty() && options.adaptiveTgvUseGlobalVisibilityField)
                        {
                            visibilityHistograms[index].add(std::clamp(signedDistance / truncation, -1.0f, 1.0f),
                                                            observationWeight);
                        }
                        if (signedDistance < -truncation)
                        {
                            continue;
                        }
                        if (signedDistance > maximum_free_space_distance)
                        {
                            continue;
                        }
                        const float normalized = std::clamp(signedDistance / truncation, -1.0f, 1.0f);
                        if (!visibilityHistograms.empty() && !options.adaptiveTgvUseGlobalVisibilityField)
                        {
                            visibilityHistograms[index].add(normalized, observationWeight);
                        }
                        integrateWeighted(&tsdf[index], &weight[index], normalized, observationWeight);
                        maximumObservationWeight[index] = std::max(maximumObservationWeight[index], observationWeight);
                        evidenceSupportWeight[index] += evidenceSupportObservationWeight;
                        maximumEvidenceSupportObservationWeight[index] =
                            std::max(maximumEvidenceSupportObservationWeight[index], evidenceSupportObservationWeight);
                        ++integratedVoxelUpdates;
                        if (std::fabs(signedDistance) <= surface_support_distance)
                        {
                            maximumGeometrySupportCount[index] =
                                std::max(maximumGeometrySupportCount[index], observation.geometrySupportCount);
                            if (!frame.auxiliarySurfaceOnly &&
                                DepthTsdfSurfaceBuilder::observationHasStrongAdaptiveGeometryEvidence(observation,
                                                                                                      options))
                            {
                                strongAdaptiveSurfaceObservation[index] = 1;
                            }
                            if (options.enableSurfacePatchSupport || options.enableContourBandZeroCrossingSupport ||
                                options.enableMeasuredSupportConnectivity || options.collectZeroCrossingDiagnostics ||
                                options.collectAcquisitionGapReport || options.enableCrossViewAnchoredSurfaceRecovery ||
                                options.enableGlobalImplicitRegularization || options.enableAdaptiveTgvRegularization ||
                                result.statistics.effectiveOrbitalGapBoundaryRecovery)
                            {
                                const DepthGeometryLocalSourceEncoding& local_source_encoding =
                                    local_source_encodings[static_cast<std::size_t>(frame_index)];
                                geometrySourceMask[index] |=
                                    local_source_encoding.encode(observation.geometrySourceMask);
                                if (observation.geometrySourceMask != 0)
                                {
                                    ++geometrySourceNonzeroLocalMaskObservationCount;
                                    const std::uint16_t mapped_local_mask =
                                        local_source_encoding.mappedLocalMask(observation.geometrySourceMask);
                                    if (mapped_local_mask == observation.geometrySourceMask)
                                    {
                                        ++geometrySourceFullyMappedObservationCount;
                                    }
                                    else if (mapped_local_mask != 0)
                                    {
                                        ++geometrySourcePartiallyMappedObservationCount;
                                    }
                                    else
                                    {
                                        ++geometrySourceFullyUnmappedObservationCount;
                                    }
                                }
                                if (observation.geometrySourceMask != 0 &&
                                    std::isfinite(observation.inverseDepthRelativeSpread) &&
                                    observation.inverseDepthRelativeSpread >= 0.0f)
                                {
                                    const int quantized_spread = static_cast<int>(
                                        std::lround(observation.inverseDepthRelativeSpread * 100000.0f));
                                    minimumInverseDepthSpread[index] =
                                        std::min(minimumInverseDepthSpread[index],
                                                 static_cast<std::uint16_t>(std::clamp(quantized_spread, 0, 65535)));
                                }
                                if (bitCount(observation.geometrySourceMask) >=
                                        std::max(2, options.minimumSurfacePatchSourceCount) &&
                                    std::isfinite(observation.inverseDepthRelativeSpread) &&
                                    observation.inverseDepthRelativeSpread <=
                                        options.maximumSurfacePatchInverseDepthSpread)
                                {
                                    surfaceTsdfWeightedSum[index] += normalized * observationWeight;
                                    surfaceObservationWeight[index] += observationWeight;
                                }
                                if (!crossViewRepairedSurfaceWeight.empty() && observation.usedCrossViewRepairedDepth)
                                {
                                    const int quantized_weight = static_cast<int>(
                                        std::lround(std::clamp(observationWeight, 0.0f, 1.0f) * 255.0f));
                                    crossViewRepairedSurfaceWeight[index] = static_cast<std::uint8_t>(std::max(
                                        static_cast<int>(crossViewRepairedSurfaceWeight[index]), quantized_weight));
                                }
                                if (!orbitalGapBoundaryObservationWeight.empty() &&
                                    orbital_gap_boundary_frames[static_cast<std::size_t>(frame_index)] != 0)
                                {
                                    orbitalGapBoundaryObservationWeight[index] =
                                        std::max(orbitalGapBoundaryObservationWeight[index], observationWeight);
                                }
                                if (contour_band_evidence && !contourBandObservationWeight.empty())
                                {
                                    contourBandObservationWeight[index] += observationWeight;
                                }
                            }
                            support[index] = static_cast<std::uint16_t>(std::min<int>(
                                std::numeric_limits<std::uint16_t>::max(), static_cast<int>(support[index]) + 1));
                        }
                    }
                    const int minimum_free_space_views = std::clamp(options.minimumSupportMaskFreeSpaceViews, 1, 16);
                    if (options.enableSupportMaskFreeSpaceCarving &&
                        support_mask_free_space_votes >= minimum_free_space_views)
                    {
                        const bool has_surface_evidence =
                            support[index] > 0 ||
                            (!surfaceObservationWeight.empty() && surfaceObservationWeight[index] > 0.0f);
                        if (options.enableSurfaceEvidenceFreeSpaceVeto && has_surface_evidence)
                        {
                            supportMaskFreeSpaceSurfaceVetoCount += support_mask_free_space_votes;
                            continue;
                        }
                        const float carving_weight =
                            std::max(0.0f, options.supportMaskFreeSpaceWeight) * support_mask_free_space_votes;
                        integrateWeighted(&tsdf[index], &weight[index], 1.0f, carving_weight);
                        if (!visibilityHistograms.empty())
                        {
                            visibilityHistograms[index].add(1.0f, carving_weight);
                        }
                        ++integratedVoxelUpdates;
                        supportMaskFreeSpaceUpdateCount += support_mask_free_space_votes;
                    }
                }
            }
            const int completed = completed_z_slices.fetch_add(1, std::memory_order_relaxed) + 1;
            const int progress_percent = 5 + completed * 65 / std::max(1, zSamples);
            if (options.execution.progress)
            {
                const std::lock_guard<std::mutex> progress_lock(progress_callback_mutex);
                const int previous_progress = last_progress_percent.load(std::memory_order_relaxed);
                if (progress_percent >= previous_progress + 5)
                {
                    last_progress_percent.store(progress_percent, std::memory_order_relaxed);
                    options.execution.reportProgress(
                        (QStringLiteral("正在融合置信度加权 TSDF...")).toUtf8().toStdString(),
                        (progress_percent) / 100.0);
                }
            }
        }

        result.statistics.tsdfIntegrationElapsedMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - integration_start)
                .count();
        result.statistics.tsdfIntegrationCpuTimeMs = detail::processCpuTimeMilliseconds() - integration_cpu_start;
        result.statistics.tsdfIntegrationCpuDuty =
            result.statistics.tsdfIntegrationElapsedMs > 0
                ? result.statistics.tsdfIntegrationCpuTimeMs /
                      (static_cast<double>(result.statistics.tsdfIntegrationElapsedMs) * workerCount)
                : 0.0;

        if (cancelled.load(std::memory_order_relaxed))
        {
            result.errorMessage = QStringLiteral("TSDF integration cancelled");
            return false;
        }
        result.statistics.integratedVoxelUpdates = integratedVoxelUpdates;
        result.statistics.narrowBandActivationSkippedSampleCount = narrowBandActivationSkippedSampleCount;
        result.statistics.rejectedProjectionCount = rejectedProjectionCount;
        result.statistics.rejectedSupportMaskCount = rejectedSupportMaskCount;
        result.statistics.supportMaskFreeSpaceUpdateCount = supportMaskFreeSpaceUpdateCount;
        result.statistics.supportMaskFreeSpaceSurfaceVetoCount = supportMaskFreeSpaceSurfaceVetoCount;
        result.statistics.auxiliaryOutsideSurfaceBandRejectedCount = auxiliaryOutsideSurfaceBandRejectedCount;
        result.statistics.auxiliaryOutsidePrimaryNeighborhoodRejectedCount =
            auxiliaryOutsidePrimaryNeighborhoodRejectedCount;
        result.statistics.auxiliaryBridgeRejectedGeometrySupportCount = auxiliaryBridgeRejectedGeometrySupportCount;
        result.statistics.auxiliaryBridgeRejectedSourceCount = auxiliaryBridgeRejectedSourceCount;
        result.statistics.auxiliaryBridgeRejectedSpreadCount = auxiliaryBridgeRejectedSpreadCount;
        result.statistics.auxiliaryBridgeRejectedExtensionSampleCount = auxiliaryBridgeRejectedExtensionSampleCount;
        result.statistics.rejectedDepthValidCount = rejectedDepthValidCount;
        result.statistics.rejectedInterpolationPolicyCount = rejectedInterpolationPolicyCount;
        result.statistics.rejectedDepthCount = rejectedDepthCount;
        result.statistics.rejectedConfidenceCount = rejectedConfidenceCount;
        result.statistics.subpixelObservationCount = subpixelObservationCount;
        result.statistics.recoveredNeighborObservationCount = recoveredNeighborObservationCount;
        result.statistics.discontinuityRejectedCandidateCount = discontinuityRejectedCandidateCount;
        result.statistics.rejectedGeometryConsistencyCount = rejectedGeometryConsistencyCount;
        result.statistics.rejectedInvalidNearestPixelRecoveryCount = rejectedInvalidNearestPixelRecoveryCount;
        result.statistics.crossViewConsensusDepthObservationCount = crossViewConsensusDepthObservationCount;
        result.statistics.unconfirmedNativeObservationCount = unconfirmedNativeObservationCount;
        result.statistics.weakNativeObservationCount = weakNativeObservationCount;
        result.statistics.repairedObservationCount = repairedObservationCount;
        result.statistics.strongNativeObservationCount = strongNativeObservationCount;
        result.statistics.inverseDepthSpreadDownweightedObservationCount =
            inverseDepthSpreadDownweightedObservationCount;
        result.statistics.inverseDepthSpreadVeryWeakObservationCount = inverseDepthSpreadVeryWeakObservationCount;
        result.statistics.inverseDepthSpreadSupportLiftedObservationCount =
            inverseDepthSpreadSupportLiftedObservationCount;
        result.statistics.weakEvidenceOutsideSurfaceBandRejectedCount = weakEvidenceOutsideSurfaceBandRejectedCount;
        result.statistics.geometrySourceNonzeroLocalMaskObservationCount =
            geometrySourceNonzeroLocalMaskObservationCount;
        result.statistics.geometrySourceFullyMappedObservationCount = geometrySourceFullyMappedObservationCount;
        result.statistics.geometrySourcePartiallyMappedObservationCount = geometrySourcePartiallyMappedObservationCount;
        result.statistics.geometrySourceFullyUnmappedObservationCount = geometrySourceFullyUnmappedObservationCount;
        if (geometrySourceNonzeroLocalMaskObservationCount > 0)
        {
            const double denominator = static_cast<double>(geometrySourceNonzeroLocalMaskObservationCount);
            result.statistics.geometrySourceFullyMappedObservationRatio =
                static_cast<double>(geometrySourceFullyMappedObservationCount) / denominator;
            result.statistics.geometrySourcePartiallyMappedObservationRatio =
                static_cast<double>(geometrySourcePartiallyMappedObservationCount) / denominator;
            result.statistics.geometrySourceFullyUnmappedObservationRatio =
                static_cast<double>(geometrySourceFullyUnmappedObservationCount) / denominator;
        }
        result.statistics.effectivePixelEvidenceWeighting = options.enablePixelEvidenceWeighting;
        result.statistics.effectiveUnconfirmedNativeObservationMultiplier =
            options.unconfirmedNativeObservationMultiplier;
        result.statistics.effectiveWeakNativeObservationMultiplier = options.weakNativeObservationMultiplier;
        result.statistics.effectiveRepairedObservationMultiplier = options.repairedObservationMultiplier;
        result.statistics.effectiveExcludeAnchoredInterpolationObservations =
            options.excludeAnchoredInterpolationObservations;
        result.statistics.effectiveInverseDepthSpreadWeighting = options.enableInverseDepthSpreadWeighting;
        result.statistics.effectiveInverseDepthSpreadWeightKnee = options.inverseDepthSpreadWeightKnee;
        result.statistics.effectiveInverseDepthSpreadWeightZero = options.inverseDepthSpreadWeightZero;
        result.statistics.effectiveMinimumInverseDepthSpreadWeightMultiplier =
            options.minimumInverseDepthSpreadWeightMultiplier;
        result.statistics.effectiveInverseDepthSpreadSupportWeightDecoupling =
            options.enableInverseDepthSpreadSupportWeightDecoupling;
        result.statistics.effectiveInverseDepthSpreadSupportWeightExponent =
            std::clamp(options.inverseDepthSpreadSupportWeightExponent, 0.05f, 1.0f);
        result.statistics.effectiveEvidenceSupportWeightDecoupling = options.enableEvidenceSupportWeightDecoupling;
        result.statistics.effectiveEvidenceSupportWeightExponent = options.evidenceSupportWeightExponent;
        result.statistics.effectiveWeakEvidenceSurfaceOnlyIntegration =
            options.enableWeakEvidenceSurfaceOnlyIntegration;
        result.statistics.effectiveWeakEvidenceSurfaceBandVoxels = weak_evidence_surface_band_voxels;
        result.statistics.crossViewConsensusContourBandPixelCount = cross_view_consensus_contour_band_pixel_count;
        result.statistics.effectiveMinimumVoxelWeight = options.minimumVoxelWeight;
        result.statistics.effectiveMinimumSingleObservationWeight = options.minimumSingleObservationWeight;
        result.statistics.effectiveMinimumGeometryVerifiedObservationWeight =
            options.minimumGeometryVerifiedObservationWeight;
        result.statistics.effectiveMinimumGeometrySupportCount = options.minimumGeometrySupportCount;
        result.statistics.effectiveAllowGeometryVerifiedSingleObservation =
            options.allowGeometryVerifiedSingleObservation;
        result.statistics.effectiveGeometrySingleViewNeighborhoodGuard =
            options.enableGeometrySingleViewNeighborhoodGuard;
        result.statistics.effectiveMinimumGeometrySingleViewNeighborCount =
            options.minimumGeometrySingleViewNeighborCount;
        result.statistics.effectiveGeometrySingleViewGrowthPasses = options.geometrySingleViewGrowthPasses;
        result.statistics.effectiveMaximumGeometrySingleViewNeighborTsdfDelta =
            options.maximumGeometrySingleViewNeighborTsdfDelta;
        result.statistics.effectiveDiscontinuityAwareSampling = options.enableDiscontinuityAwareSampling;
        result.statistics.effectiveMaximumInterpolationRelativeDepthSpread =
            options.maximumInterpolationRelativeDepthSpread;
        result.statistics.effectiveMaximumObservationInverseDepthSpread = options.maximumObservationInverseDepthSpread;
        result.statistics.effectiveAllowInvalidNearestPixelRecovery = options.allowInvalidNearestPixelRecovery;
        result.statistics.effectiveMaximumInvalidNearestPixelRecoveryInverseDepthSpread =
            options.maximumInvalidNearestPixelRecoveryInverseDepthSpread;
        result.statistics.effectiveCrossViewConsensusDepth = options.enableCrossViewConsensusDepth;
        result.statistics.effectiveMaximumCrossViewConsensusInverseDepthSpread =
            options.maximumCrossViewConsensusInverseDepthSpread;
        result.statistics.effectiveCrossViewConsensusContourBandOnly = options.crossViewConsensusContourBandOnly;
        result.statistics.effectiveSurfacePatchSupport = options.enableSurfacePatchSupport;
        result.statistics.effectiveContourBandZeroCrossingSupport = options.enableContourBandZeroCrossingSupport;
        result.statistics.effectiveGeometryZeroCrossingRecovery = options.enableGeometryZeroCrossingRecovery;
        result.statistics.effectiveCrossViewAnchoredSurfaceRecovery = options.enableCrossViewAnchoredSurfaceRecovery;
        result.statistics.effectiveCrossViewAnchoredMinimumObservationWeight =
            options.crossViewAnchoredMinimumObservationWeight;
        result.statistics.effectiveCrossViewAnchoredMinimumSupportedCorners =
            options.crossViewAnchoredMinimumSupportedCorners;
        result.statistics.effectiveCrossViewAnchoredMinimumCellVotes = options.crossViewAnchoredMinimumCellVotes;
        result.statistics.effectiveCrossViewAnchoredGrowthPasses = options.crossViewAnchoredGrowthPasses;
        result.statistics.effectiveGeometryZeroCrossingCellSheets = options.enableGeometryZeroCrossingCellSheets;
        result.statistics.effectiveMaximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf =
            options.maximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf;
        result.statistics.effectiveGlobalImplicitRegularization = options.enableGlobalImplicitRegularization;
        result.statistics.effectiveAdaptiveTgvRegularization = options.enableAdaptiveTgvRegularization;
        result.statistics.effectiveAdaptiveTgvGlobalVisibilityField =
            options.enableAdaptiveTgvRegularization && options.adaptiveTgvUseGlobalVisibilityField;
        result.statistics.effectiveImplicitRegularizationLevels = options.implicitRegularizationLevels;
        result.statistics.effectiveImplicitRegularizationPassesPerLevel = options.implicitRegularizationPassesPerLevel;
        result.statistics.effectiveImplicitRegularizationSmoothness = options.implicitRegularizationSmoothness;
        result.statistics.effectiveImplicitRegularizationDataFidelity = options.implicitRegularizationDataFidelity;
        result.statistics.effectiveImplicitRegularizationMaximumUpdate = options.implicitRegularizationMaximumUpdate;
        result.statistics.effectiveImplicitRegularizationEdgeThreshold = options.implicitRegularizationEdgeThreshold;
        result.statistics.effectiveImplicitRegularizationRecoverAxialGaps =
            options.implicitRegularizationRecoverAxialGaps;
        result.statistics.effectiveImplicitRegularizationMinimumBridgeAxes =
            options.implicitRegularizationMinimumBridgeAxes;
        result.statistics.effectiveImplicitRegularizationMaximumBridgePredictionDelta =
            options.implicitRegularizationMaximumBridgePredictionDelta;
        result.statistics.effectiveMinimumSurfacePatchObservationWeight = options.minimumSurfacePatchObservationWeight;
        result.statistics.effectiveMinimumSurfacePatchSourceCount = options.minimumSurfacePatchSourceCount;
        result.statistics.effectiveMinimumSurfacePatchCoreNeighborCount = options.minimumSurfacePatchCoreNeighborCount;
        result.statistics.effectiveSurfacePatchGrowthPasses = options.surfacePatchGrowthPasses;
        result.statistics.effectiveMaximumSurfacePatchInverseDepthSpread =
            options.maximumSurfacePatchInverseDepthSpread;
        result.statistics.effectiveMaximumSurfacePatchNormalAngleDegrees =
            options.maximumSurfacePatchNormalAngleDegrees;
        result.statistics.effectiveMaximumSurfacePatchAbsoluteTsdf = options.maximumSurfacePatchAbsoluteTsdf;
        result.statistics.effectiveMinimumSurfacePatchWeightRatio = options.minimumSurfacePatchWeightRatio;
        result.statistics.effectiveMinimumDistinctCameraSupport = options.minimumDistinctCameraSupport;
        result.statistics.effectiveUncertaintyAdaptiveTruncation = uncertainty_adaptation_available;
        result.statistics.uncertaintyAdaptiveSampleCount = uncertainty_band.sampleCount;
        result.statistics.uncertaintyAdaptiveP90Voxels = uncertainty_band.p90Voxels;
        result.statistics.uncertaintyAdaptiveAddedVoxels = effective_truncation_voxels - base_truncation_voxels;
        result.statistics.effectiveUncertaintyAdaptiveScale = effective_uncertainty_adaptive_scale;
        result.statistics.effectiveUncertaintyAdaptiveActivationRatio = options.uncertaintyAdaptiveActivationRatio;
        result.statistics.effectiveUncertaintyAdaptiveMaximumTruncationVoxels = adaptive_maximum_truncation_voxels;
        result.statistics.effectiveOrbitalGapAdaptiveTruncation = orbital_gap_adaptive_truncation;
        result.statistics.effectiveOrbitalGapAdaptiveTruncationScale = options.orbitalGapAdaptiveTruncationScale;
        result.statistics.effectiveOrbitalGapAdaptiveMaximumTruncationVoxels =
            options.orbitalGapAdaptiveMaximumTruncationVoxels;
        result.statistics.effectiveTruncationVoxels = effective_truncation_voxels;
        result.statistics.effectiveSurfaceSupportBandVoxels = effective_surface_support_band_voxels;
        result.statistics.effectiveMaximumFreeSpaceVoxels =
            options.maximumFreeSpaceVoxels > 0.0f
                ? std::max(effective_truncation_voxels, options.maximumFreeSpaceVoxels)
                : 0.0f;
        result.statistics.effectiveMinimumSupportMaskFreeSpaceViews =
            std::clamp(options.minimumSupportMaskFreeSpaceViews, 1, 16);
        result.statistics.effectiveSurfaceEvidenceFreeSpaceVeto = options.enableSurfaceEvidenceFreeSpaceVeto;
        result.statistics.effectiveNarrowBandActivation = options.enableNarrowBandActivation;
        result.statistics.effectiveNarrowBandActivationBlockSizeSamples =
            options.enableNarrowBandActivation ? std::clamp(options.narrowBandActivationBlockSizeSamples, 2, 32) : 0;
        result.statistics.effectiveNarrowBandActivationDepthStride =
            options.enableNarrowBandActivation ? std::clamp(options.narrowBandActivationDepthStride, 1, 16) : 0;
        result.statistics.effectiveNarrowBandActivationRayStepVoxels =
            options.enableNarrowBandActivation ? std::clamp(options.narrowBandActivationRayStepVoxels, 0.25f, 4.0f)
                                               : 0.0f;
        result.statistics.effectiveNarrowBandActivationHaloBlocks =
            options.enableNarrowBandActivation ? std::clamp(options.narrowBandActivationHaloBlocks, 0, 3) : 0;
        result.statistics.effectiveDepthValidBoundaryErosionPixels = erosion_pixels;
        result.statistics.effectiveGeometryVerifiedBoundaryRecovery = options.enableGeometryVerifiedBoundaryRecovery;
        result.statistics.effectiveMinimumBoundaryRecoveryGeometrySupport =
            options.minimumBoundaryRecoveryGeometrySupport;
        result.statistics.effectiveMaximumBoundaryRecoveryInverseDepthSpread =
            options.maximumBoundaryRecoveryInverseDepthSpread;
        result.statistics.boundaryRecoveredDepthValidPixelCount = boundary_recovered_depth_valid_pixel_count;

        return true;
    }

} // namespace xjw::mesh
