#include "DepthTsdfStages.h"
namespace xjw::mesh
{
    using namespace tsdf_detail;
    bool tsdf_detail::recoverVolumeSupport(const QVector<DepthTsdfFrame>& frames,
                                           const DepthTsdfOptions& options,
                                           DepthTsdfResult& result,
                                           TsdfSupportState state)
    {
        auto& supported = state.supported;
        auto& effective_frame_quality_weights = state.effective_frame_quality_weights;
        auto& effective_depth_valid_masks = state.effective_depth_valid_masks;
        auto& erosion_pixels = state.erosion_pixels;
        auto& retained_frames = state.retained_frames;
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
        auto& truncation = state.truncation;
        auto& cancelled = state.cancelled;
        auto& workerCount = state.workerCount;
        auto& adaptiveTgvExtractionSupport = state.adaptiveTgvExtractionSupport;
        auto& visual_hull_completion_tsdf = state.visual_hull_completion_tsdf;
        auto& visual_hull_completion_support = state.visual_hull_completion_support;
        auto& native_carrier_bounds_min = state.native_carrier_bounds_min;
        auto& native_carrier_bounds_max = state.native_carrier_bounds_max;
        auto& native_carrier_dimensions = state.native_carrier_dimensions;
        auto& native_carrier_cells = state.native_carrier_cells;
        auto& native_carrier_occupied = state.native_carrier_occupied;
        auto& native_carrier_field = state.native_carrier_field;

        std::vector<std::size_t> guarded_geometry_single_view_candidates;
        std::vector<std::size_t> orbital_gap_boundary_recovery_candidates;
        if (options.enableGeometrySingleViewNeighborhoodGuard)
        {
            guarded_geometry_single_view_candidates.reserve(static_cast<std::size_t>(result.layout.sampleCount / 32));
        }
        if (result.statistics.effectiveOrbitalGapBoundaryRecovery)
        {
            orbital_gap_boundary_recovery_candidates.reserve(static_cast<std::size_t>(result.layout.sampleCount / 64));
        }
        for (std::size_t index = 0; index < supported.size(); ++index)
        {
            bool single_view_supported = false;
            bool multi_view_supported = false;
            bool geometry_verified_single_view_supported = false;
            const bool sample_supported =
                DepthTsdfSurfaceBuilder::isSampleSupported(evidenceSupportWeight[index],
                                                           support[index],
                                                           maximumEvidenceSupportObservationWeight[index],
                                                           options,
                                                           &single_view_supported,
                                                           &multi_view_supported,
                                                           maximumGeometrySupportCount[index],
                                                           &geometry_verified_single_view_supported,
                                                           strongAdaptiveSurfaceObservation[index] != 0);
            const bool field_weight_supported =
                DepthTsdfSurfaceBuilder::isSampleSupported(weight[index],
                                                           support[index],
                                                           maximumObservationWeight[index],
                                                           options,
                                                           nullptr,
                                                           nullptr,
                                                           maximumGeometrySupportCount[index],
                                                           nullptr);
            if (sample_supported && !field_weight_supported)
            {
                ++result.statistics.evidenceSupportRecoveredSampleCount;
            }
            if (multi_view_supported)
            {
                supported[index] = 1;
                ++result.statistics.multiViewSupportedSampleCount;
            }
            else if (single_view_supported)
            {
                if (geometry_verified_single_view_supported && options.enableGeometrySingleViewNeighborhoodGuard)
                {
                    guarded_geometry_single_view_candidates.push_back(index);
                }
                else
                {
                    supported[index] = 1;
                    ++result.statistics.singleViewSupportedSampleCount;
                    if (geometry_verified_single_view_supported)
                    {
                        ++result.statistics.geometryVerifiedSingleViewSupportedSampleCount;
                    }
                }
            }
            else if (!sample_supported && support[index] == 1)
            {
                ++result.statistics.rejectedSingleObservationWeightCount;
            }
            else if (!sample_supported && support[index] >= options.minimumDistinctCameraSupport)
            {
                ++result.statistics.rejectedAccumulatedWeightCount;
            }
            if (!sample_supported && result.statistics.effectiveOrbitalGapBoundaryRecovery && support[index] == 1 &&
                !orbitalGapBoundaryObservationWeight.empty() && orbitalGapBoundaryObservationWeight[index] > 1.0e-6f)
            {
                ++result.statistics.orbitalGapBoundarySingleObservationCount;
                const float maximum_spread = std::min(options.maximumBoundaryRecoveryInverseDepthSpread,
                                                      options.maximumCrossViewConsensusInverseDepthSpread);
                if (orbitalGapBoundaryObservationWeight[index] < options.orbitalGapBoundaryMinimumObservationWeight)
                {
                    ++result.statistics.orbitalGapBoundaryRejectedWeightCount;
                }
                else if (maximumGeometrySupportCount[index] < options.minimumBoundaryRecoveryGeometrySupport)
                {
                    ++result.statistics.orbitalGapBoundaryRejectedGeometrySupportCount;
                }
                else if (bitCount(geometrySourceMask[index]) < std::max(2, options.minimumSurfacePatchSourceCount))
                {
                    ++result.statistics.orbitalGapBoundaryRejectedSourceCount;
                }
                else if (minimumInverseDepthSpread[index] == std::numeric_limits<std::uint16_t>::max() ||
                         maximum_spread <= 0.0f ||
                         static_cast<float>(minimumInverseDepthSpread[index]) / 100000.0f > maximum_spread)
                {
                    ++result.statistics.orbitalGapBoundaryRejectedSpreadCount;
                }
                else if (surfaceObservationWeight[index] <= 1.0e-6f || weight[index] <= 1.0e-6f ||
                         surfaceObservationWeight[index] / weight[index] < options.minimumSurfacePatchWeightRatio ||
                         std::fabs(tsdf[index]) > options.maximumSurfacePatchAbsoluteTsdf)
                {
                    ++result.statistics.orbitalGapBoundaryRejectedFieldCount;
                }
                else
                {
                    orbital_gap_boundary_recovery_candidates.push_back(index);
                }
            }
            result.statistics.supportedSampleCount += supported[index] != 0;
        }
        if (result.statistics.effectiveOrbitalGapBoundaryRecovery && !orbital_gap_boundary_recovery_candidates.empty())
        {
            const int accepted_count = DepthTsdfSurfaceBuilder::growGeometryVerifiedSingleViewSamples(
                result.layout,
                tsdf,
                orbital_gap_boundary_recovery_candidates,
                options.minimumGeometrySingleViewNeighborCount,
                options.geometrySingleViewGrowthPasses,
                options.maximumGeometrySingleViewNeighborTsdfDelta,
                &supported);
            result.statistics.orbitalGapBoundaryRecoveryCandidateCount =
                orbital_gap_boundary_recovery_candidates.size();
            result.statistics.orbitalGapBoundaryRecoveryAcceptedCount = accepted_count;
            result.statistics.orbitalGapBoundaryRecoveryRejectedCount =
                orbital_gap_boundary_recovery_candidates.size() - static_cast<std::size_t>(accepted_count);
            result.statistics.singleViewSupportedSampleCount += accepted_count;
            result.statistics.geometryVerifiedSingleViewSupportedSampleCount += accepted_count;
            result.statistics.supportedSampleCount += accepted_count;
        }
        if (options.enableGeometrySingleViewNeighborhoodGuard && !guarded_geometry_single_view_candidates.empty())
        {
            const int accepted_count = DepthTsdfSurfaceBuilder::growGeometryVerifiedSingleViewSamples(
                result.layout,
                tsdf,
                guarded_geometry_single_view_candidates,
                options.minimumGeometrySingleViewNeighborCount,
                options.geometrySingleViewGrowthPasses,
                options.maximumGeometrySingleViewNeighborTsdfDelta,
                &supported);
            result.statistics.geometrySingleViewNeighborhoodCandidateCount =
                guarded_geometry_single_view_candidates.size();
            result.statistics.geometrySingleViewNeighborhoodAcceptedCount = accepted_count;
            result.statistics.geometrySingleViewNeighborhoodRejectedCount =
                guarded_geometry_single_view_candidates.size() - static_cast<std::size_t>(accepted_count);
            result.statistics.singleViewSupportedSampleCount += accepted_count;
            result.statistics.geometryVerifiedSingleViewSupportedSampleCount += accepted_count;
            result.statistics.supportedSampleCount += accepted_count;
        }
        result.statistics.effectiveMeasuredSupportConnectivity = options.enableMeasuredSupportConnectivity;
        result.statistics.effectiveMeasuredSupportMinimumObservationWeight =
            options.measuredSupportMinimumObservationWeight;
        result.statistics.effectiveMeasuredSupportMinimumSourceCount = options.measuredSupportMinimumSourceCount;
        result.statistics.effectiveMeasuredSupportMinimumGeometrySupport =
            options.measuredSupportMinimumGeometrySupport;
        result.statistics.effectiveMeasuredSupportMaximumInverseDepthSpread =
            options.measuredSupportMaximumInverseDepthSpread;
        result.statistics.effectiveMeasuredSupportMinimumSurfaceWeightRatio =
            options.measuredSupportMinimumSurfaceWeightRatio;
        result.statistics.effectiveMeasuredSupportMaximumAbsoluteTsdf = options.measuredSupportMaximumAbsoluteTsdf;
        result.statistics.effectiveMeasuredSupportMinimumSupportedCellCorners =
            options.measuredSupportMinimumSupportedCellCorners;
        result.statistics.effectiveMeasuredSupportMinimumComponentCells = options.measuredSupportMinimumComponentCells;
        result.statistics.effectiveMeasuredSupportMinimumAnchorCells = options.measuredSupportMinimumAnchorCells;
        result.statistics.effectiveMeasuredSupportMaximumSingleVoteAbsoluteTsdf =
            options.measuredSupportMaximumSingleVoteAbsoluteTsdf;
        if (options.enableMeasuredSupportConnectivity && !geometrySourceMask.empty())
        {
            const std::vector<std::uint8_t> measured_support_baseline = supported;
            DepthMeasuredSupportConnectivityInput connectivity_input;
            connectivity_input.sampleDimensions = {
                result.layout.cells[0] + 1, result.layout.cells[1] + 1, result.layout.cells[2] + 1};
            connectivity_input.tsdf = &tsdf;
            connectivity_input.weight = &weight;
            connectivity_input.surfaceObservationWeight = &surfaceObservationWeight;
            connectivity_input.maximumEvidenceObservationWeight = &maximumEvidenceSupportObservationWeight;
            connectivity_input.geometrySourceMask = &geometrySourceMask;
            connectivity_input.minimumInverseDepthSpread = &minimumInverseDepthSpread;
            connectivity_input.maximumGeometrySupportCount = &maximumGeometrySupportCount;
            DepthMeasuredSupportConnectivityOptions connectivity_options;
            connectivity_options.minimumObservationWeight = options.measuredSupportMinimumObservationWeight;
            connectivity_options.minimumSourceCount = options.measuredSupportMinimumSourceCount;
            connectivity_options.minimumGeometrySupport = options.measuredSupportMinimumGeometrySupport;
            connectivity_options.maximumInverseDepthSpread = options.measuredSupportMaximumInverseDepthSpread;
            connectivity_options.minimumSurfaceWeightRatio = options.measuredSupportMinimumSurfaceWeightRatio;
            connectivity_options.maximumAbsoluteTsdf = options.measuredSupportMaximumAbsoluteTsdf;
            connectivity_options.minimumSupportedCellCorners = options.measuredSupportMinimumSupportedCellCorners;
            connectivity_options.minimumComponentCells = options.measuredSupportMinimumComponentCells;
            connectivity_options.minimumAnchorCells = options.measuredSupportMinimumAnchorCells;
            connectivity_options.maximumSingleVoteAbsoluteTsdf = options.measuredSupportMaximumSingleVoteAbsoluteTsdf;
            const DepthMeasuredSupportConnectivityStatistics connectivity =
                DepthMeasuredSupportConnectivity::recover(connectivity_input, connectivity_options, &supported);
            result.statistics.measuredSupportConsideredSampleCount = connectivity.consideredSampleCount;
            result.statistics.measuredSupportRejectedObservationWeightCount =
                connectivity.rejectedObservationWeightCount;
            result.statistics.measuredSupportRejectedSourceCount = connectivity.rejectedSourceCount;
            result.statistics.measuredSupportRejectedGeometrySupportCount = connectivity.rejectedGeometrySupportCount;
            result.statistics.measuredSupportRejectedDepthSpreadCount = connectivity.rejectedDepthSpreadCount;
            result.statistics.measuredSupportRejectedSurfaceWeightCount = connectivity.rejectedSurfaceWeightCount;
            result.statistics.measuredSupportRejectedAbsoluteTsdfCount = connectivity.rejectedAbsoluteTsdfCount;
            result.statistics.measuredSupportEligibleSampleCount = connectivity.eligibleSampleCount;
            result.statistics.measuredSupportRejectedZeroCrossingCount = connectivity.rejectedZeroCrossingCount;
            result.statistics.measuredSupportRecoveredSampleCount = connectivity.recoveredSampleCount;
            result.statistics.measuredSupportUnlockedCellCount = connectivity.unlockedCellCount;
            result.statistics.measuredSupportCandidateCellCount = connectivity.candidateCellCount;
            result.statistics.measuredSupportAcceptedCellCount = connectivity.acceptedCellCount;
            result.statistics.measuredSupportComponentCount = connectivity.componentCount;
            result.statistics.measuredSupportAcceptedComponentCount = connectivity.acceptedComponentCount;
            result.statistics.measuredSupportRejectedSmallComponentCount = connectivity.rejectedSmallComponentCount;
            result.statistics.measuredSupportRejectedAnchorComponentCount = connectivity.rejectedAnchorComponentCount;
            result.statistics.measuredSupportRejectedBoundaryComponentCount =
                connectivity.rejectedBoundaryComponentCount;
            if (connectivity.recoveredSampleCount > 0)
            {
                if (options.isCancelled && options.isCancelled())
                {
                    supported = measured_support_baseline;
                    result.errorMessage = QStringLiteral("Measured-support topology transaction cancelled before "
                                                         "baseline extraction");
                    return false;
                }
                ComparableIsoSurfaceExtraction baseline_extraction =
                    extractComparableIsoSurface(result.layout.boundsMin,
                                                result.layout.boundsMax,
                                                result.layout.cells,
                                                tsdf,
                                                measured_support_baseline,
                                                options.enableMc33IsoSurfaceExtraction,
                                                options.mc33RequireSupportedSignChange,
                                                options.enableConsistentIsoSurfaceExtraction,
                                                options.isCancelled);
                if (!baseline_extraction.ok)
                {
                    supported = measured_support_baseline;
                    result.errorMessage = QStringLiteral("Measured-support topology transaction baseline "
                                                         "extraction failed: %1")
                                              .arg(baseline_extraction.errorMessage);
                    return false;
                }
                if (options.isCancelled && options.isCancelled())
                {
                    supported = measured_support_baseline;
                    result.errorMessage = QStringLiteral("Measured-support topology transaction cancelled after "
                                                         "baseline extraction");
                    return false;
                }
                ComparableIsoSurfaceExtraction candidate_extraction =
                    extractComparableIsoSurface(result.layout.boundsMin,
                                                result.layout.boundsMax,
                                                result.layout.cells,
                                                tsdf,
                                                supported,
                                                options.enableMc33IsoSurfaceExtraction,
                                                options.mc33RequireSupportedSignChange,
                                                options.enableConsistentIsoSurfaceExtraction,
                                                options.isCancelled);
                if (!candidate_extraction.ok)
                {
                    supported = measured_support_baseline;
                    result.errorMessage = QStringLiteral("Measured-support topology transaction candidate "
                                                         "extraction failed: %1")
                                              .arg(candidate_extraction.errorMessage);
                    return false;
                }
                if (options.isCancelled && options.isCancelled())
                {
                    supported = measured_support_baseline;
                    result.errorMessage = QStringLiteral("Measured-support topology transaction cancelled after "
                                                         "candidate extraction");
                    return false;
                }
                const MeshTopologyQualityStatistics baseline_quality =
                    evaluateMeshTopologyQuality(baseline_extraction.mesh);
                const MeshTopologyQualityStatistics candidate_quality =
                    evaluateMeshTopologyQuality(candidate_extraction.mesh);
                DepthTsdfSurfaceBuilder::finalizeMeasuredSupportTopologyTransaction(baseline_quality,
                                                                                    candidate_quality,
                                                                                    measured_support_baseline,
                                                                                    connectivity.recoveredSampleCount,
                                                                                    &supported,
                                                                                    &result.statistics);
            }
            result.statistics.supportedSampleCount += result.statistics.measuredSupportAppliedRecoveredSampleCount;
        }
        if ((options.enableSurfacePatchSupport || options.enableContourBandZeroCrossingSupport ||
             options.enableGeometryZeroCrossingRecovery || options.enableCrossViewAnchoredSurfaceRecovery ||
             options.enableGlobalImplicitRegularization || options.enableAdaptiveTgvRegularization) &&
            !geometrySourceMask.empty())
        {
            const int minimum_source_count = std::clamp(options.minimumSurfacePatchSourceCount, 2, 8);
            const int minimum_core_neighbor_count = std::clamp(options.minimumSurfacePatchCoreNeighborCount, 1, 26);
            const int growth_passes = std::clamp(options.surfacePatchGrowthPasses, 1, 6);
            const float maximum_spread = std::clamp(options.maximumSurfacePatchInverseDepthSpread, 0.001f, 0.05f);
            const float maximum_normal_angle = std::clamp(options.maximumSurfacePatchNormalAngleDegrees, 5.0f, 45.0f);
            const float maximum_absolute_tsdf = std::clamp(options.maximumSurfacePatchAbsoluteTsdf, 0.05f, 0.95f);
            const float maximum_contour_band_absolute_tsdf =
                std::clamp(options.maximumContourBandAbsoluteTsdf, maximum_absolute_tsdf, 0.95f);
            result.statistics.effectiveMaximumContourBandAbsoluteTsdf = maximum_contour_band_absolute_tsdf;
            const float minimum_surface_weight_ratio = std::clamp(options.minimumSurfacePatchWeightRatio, 0.01f, 1.0f);
            std::vector<float> surface_candidate_tsdf = tsdf;
            for (std::size_t index = 0; index < surface_candidate_tsdf.size(); ++index)
            {
                if (surfaceObservationWeight[index] > 1.0e-6f)
                {
                    surface_candidate_tsdf[index] = surfaceTsdfWeightedSum[index] / surfaceObservationWeight[index];
                }
            }
            std::vector<std::array<int, 3>> neighbor_offsets;
            neighbor_offsets.reserve(26);
            for (int delta_z = -1; delta_z <= 1; ++delta_z)
            {
                for (int delta_y = -1; delta_y <= 1; ++delta_y)
                {
                    for (int delta_x = -1; delta_x <= 1; ++delta_x)
                    {
                        if (delta_x != 0 || delta_y != 0 || delta_z != 0)
                        {
                            neighbor_offsets.push_back({delta_x, delta_y, delta_z});
                        }
                    }
                }
            }
            for (int growth_pass = 0; growth_pass < growth_passes; ++growth_pass)
            {
                const std::vector<std::uint8_t> core_supported = supported;
                int recovered_this_pass = 0;
                for (int z = 1; z < result.layout.cells[2]; ++z)
                {
                    for (int y = 1; y < result.layout.cells[1]; ++y)
                    {
                        for (int x = 1; x < result.layout.cells[0]; ++x)
                        {
                            const std::size_t index = sampleIndex(result.layout, x, y, z);
                            if (supported[index] != 0 || support[index] == 0)
                            {
                                continue;
                            }
                            ++result.statistics.surfacePatchConsideredSampleCount;
                            const bool has_contour_band_evidence = options.enableContourBandZeroCrossingSupport &&
                                                                   contourBandObservationWeight[index] > 1.0e-6f;
                            if (options.enableContourBandZeroCrossingSupport)
                            {
                                ++result.statistics.contourBandZeroCrossingConsideredSampleCount;
                                if (!has_contour_band_evidence)
                                {
                                    ++result.statistics.contourBandZeroCrossingRejectedNoContourCount;
                                }
                            }
                            if (maximumObservationWeight[index] < options.minimumSurfacePatchObservationWeight)
                            {
                                ++result.statistics.surfacePatchRejectedWeightCount;
                                if (has_contour_band_evidence)
                                {
                                    ++result.statistics.contourBandZeroCrossingRejectedWeightCount;
                                }
                                continue;
                            }
                            if (bitCount(geometrySourceMask[index]) < minimum_source_count)
                            {
                                ++result.statistics.surfacePatchRejectedSourceOverlapCount;
                                if (has_contour_band_evidence)
                                {
                                    ++result.statistics.contourBandZeroCrossingRejectedSourceOverlapCount;
                                }
                                continue;
                            }
                            const std::uint16_t spread_value = minimumInverseDepthSpread[index];
                            if (spread_value == std::numeric_limits<std::uint16_t>::max() ||
                                static_cast<float>(spread_value) / 100000.0f > maximum_spread)
                            {
                                ++result.statistics.surfacePatchRejectedDepthSpreadCount;
                                if (has_contour_band_evidence)
                                {
                                    ++result.statistics.contourBandZeroCrossingRejectedDepthSpreadCount;
                                }
                                continue;
                            }
                            const bool insufficient_surface_weight_ratio =
                                surfaceObservationWeight[index] <= 1.0e-6f || weight[index] <= 1.0e-6f ||
                                surfaceObservationWeight[index] / weight[index] < minimum_surface_weight_ratio;
                            const bool excessive_absolute_tsdf =
                                std::fabs(surface_candidate_tsdf[index]) > (has_contour_band_evidence
                                                                                ? maximum_contour_band_absolute_tsdf
                                                                                : maximum_absolute_tsdf);
                            if (insufficient_surface_weight_ratio || excessive_absolute_tsdf)
                            {
                                ++result.statistics.surfacePatchRejectedFreeSpaceCount;
                                if (insufficient_surface_weight_ratio)
                                {
                                    ++result.statistics.surfacePatchRejectedSurfaceWeightRatioCount;
                                }
                                if (excessive_absolute_tsdf)
                                {
                                    ++result.statistics.surfacePatchRejectedAbsoluteTsdfCount;
                                }
                                if (has_contour_band_evidence)
                                {
                                    ++result.statistics.contourBandZeroCrossingRejectedFreeSpaceCount;
                                    if (insufficient_surface_weight_ratio)
                                    {
                                        ++result.statistics.contourBandZeroCrossingRejectedSurfaceWeightRatioCount;
                                    }
                                    if (excessive_absolute_tsdf)
                                    {
                                        ++result.statistics.contourBandZeroCrossingRejectedAbsoluteTsdfCount;
                                    }
                                }
                                continue;
                            }

                            bool has_core_neighbor = false;
                            bool has_source_overlap = false;
                            bool has_normal_agreement = false;
                            int agreeing_core_neighbor_count = 0;
                            int same_sign_core_neighbor_count = 0;
                            int opposite_sign_core_neighbor_count = 0;
                            cv::Vec3f candidate_normal;
                            const bool candidate_normal_valid =
                                volumeNormalAt(result.layout, surface_candidate_tsdf, x, y, z, &candidate_normal);
                            for (const auto& offset : neighbor_offsets)
                            {
                                const int neighbor_x = x + offset[0];
                                const int neighbor_y = y + offset[1];
                                const int neighbor_z = z + offset[2];
                                const std::size_t neighbor_index =
                                    sampleIndex(result.layout, neighbor_x, neighbor_y, neighbor_z);
                                if (core_supported[neighbor_index] == 0)
                                {
                                    continue;
                                }
                                has_core_neighbor = true;
                                const bool candidate_negative = surface_candidate_tsdf[index] < 0.0f;
                                const bool neighbor_negative = tsdf[neighbor_index] < 0.0f;
                                if (candidate_negative == neighbor_negative)
                                {
                                    ++same_sign_core_neighbor_count;
                                }
                                else
                                {
                                    ++opposite_sign_core_neighbor_count;
                                }
                                if ((geometrySourceMask[index] & geometrySourceMask[neighbor_index]) == 0)
                                {
                                    continue;
                                }
                                has_source_overlap = true;
                                cv::Vec3f neighbor_normal;
                                if (!candidate_normal_valid || !volumeNormalAt(result.layout,
                                                                               surface_candidate_tsdf,
                                                                               neighbor_x,
                                                                               neighbor_y,
                                                                               neighbor_z,
                                                                               &neighbor_normal))
                                {
                                    continue;
                                }
                                const float cosine =
                                    std::clamp(std::fabs(candidate_normal.dot(neighbor_normal)), 0.0f, 1.0f);
                                const float angle = std::acos(cosine) * 180.0f / static_cast<float>(CV_PI);
                                if (angle <= maximum_normal_angle)
                                {
                                    has_normal_agreement = true;
                                    ++agreeing_core_neighbor_count;
                                }
                            }
                            if (!has_core_neighbor || !has_source_overlap)
                            {
                                ++result.statistics.surfacePatchRejectedSourceOverlapCount;
                                if (has_contour_band_evidence)
                                {
                                    ++result.statistics.contourBandZeroCrossingRejectedNeighborhoodCount;
                                }
                                continue;
                            }
                            const bool normal_patch_supported =
                                options.enableSurfacePatchSupport && has_normal_agreement &&
                                agreeing_core_neighbor_count >= minimum_core_neighbor_count;
                            const bool has_geometry_support =
                                maximumGeometrySupportCount[index] >= options.minimumBoundaryRecoveryGeometrySupport;
                            const bool has_sign_pair =
                                same_sign_core_neighbor_count >= 1 && opposite_sign_core_neighbor_count >= 1;
                            const bool zero_crossing_supported =
                                has_contour_band_evidence && has_geometry_support && has_sign_pair;
                            if (!normal_patch_supported && !zero_crossing_supported)
                            {
                                ++result.statistics.surfacePatchRejectedNormalCount;
                                if (has_contour_band_evidence)
                                {
                                    if (!has_geometry_support)
                                    {
                                        ++result.statistics.contourBandZeroCrossingRejectedGeometrySupportCount;
                                    }
                                    else if (!has_sign_pair)
                                    {
                                        ++result.statistics.contourBandZeroCrossingRejectedNoSignPairCount;
                                    }
                                }
                                continue;
                            }
                            supported[index] = 1;
                            tsdf[index] = surface_candidate_tsdf[index];
                            ++recovered_this_pass;
                            ++result.statistics.surfacePatchRecoveredSampleCount;
                            ++result.statistics.supportedSampleCount;
                            if (zero_crossing_supported && !normal_patch_supported)
                            {
                                ++result.statistics.contourBandZeroCrossingRecoveredSampleCount;
                            }
                        }
                    }
                }
                ++result.statistics.surfacePatchExecutedGrowthPassCount;
                if (recovered_this_pass == 0)
                {
                    break;
                }
            }
            if (options.enableCrossViewAnchoredSurfaceRecovery && !crossViewRepairedSurfaceWeight.empty())
            {
                std::vector<std::uint8_t> repaired_eligible(supported.size(), 0);
                const int minimum_repaired_weight =
                    std::clamp(static_cast<int>(std::ceil(
                                   std::clamp(options.crossViewAnchoredMinimumObservationWeight, 0.0f, 1.0f) * 255.0f)),
                               1,
                               255);
                const int minimum_geometry_support = std::max(2, options.minimumBoundaryRecoveryGeometrySupport);
                const float maximum_repaired_spread = std::min(options.maximumBoundaryRecoveryInverseDepthSpread,
                                                               options.maximumSurfacePatchInverseDepthSpread);
                for (std::size_t index = 0; index < supported.size(); ++index)
                {
                    if (crossViewRepairedSurfaceWeight[index] == 0)
                    {
                        continue;
                    }
                    ++result.statistics.crossViewAnchoredObservedSampleCount;
                    const std::uint16_t spread_value = minimumInverseDepthSpread[index];
                    const bool eligible =
                        supported[index] == 0 && support[index] > 0 &&
                        crossViewRepairedSurfaceWeight[index] >= minimum_repaired_weight &&
                        maximumGeometrySupportCount[index] >= minimum_geometry_support &&
                        bitCount(geometrySourceMask[index]) >= minimum_source_count && maximum_repaired_spread > 0.0f &&
                        spread_value != std::numeric_limits<std::uint16_t>::max() &&
                        static_cast<float>(spread_value) / 100000.0f <= maximum_repaired_spread &&
                        surfaceObservationWeight[index] > 1.0e-6f && weight[index] > 1.0e-6f &&
                        surfaceObservationWeight[index] / weight[index] >= minimum_surface_weight_ratio &&
                        std::fabs(surface_candidate_tsdf[index]) <= maximum_absolute_tsdf;
                    repaired_eligible[index] = eligible;
                    result.statistics.crossViewAnchoredEligibleSampleCount += eligible;
                }

                const int growth_passes = std::clamp(options.crossViewAnchoredGrowthPasses, 1, 4);
                for (int pass = 0; pass < growth_passes; ++pass)
                {
                    const std::vector<std::uint8_t> supported_before_recovery = supported;
                    const DepthTsdfZeroCrossingRecoveryStatistics recovery =
                        DepthTsdfSurfaceBuilder::recoverGeometryVerifiedZeroCrossingSamples(
                            result.layout,
                            surface_candidate_tsdf,
                            weight,
                            geometrySourceMask,
                            repaired_eligible,
                            options.crossViewAnchoredMinimumSupportedCorners,
                            options.crossViewAnchoredMinimumCellVotes,
                            &supported);
                    result.statistics.crossViewAnchoredCandidateSampleCount += recovery.candidateSampleCount;
                    result.statistics.crossViewAnchoredRecoveredSampleCount += recovery.recoveredSampleCount;
                    ++result.statistics.crossViewAnchoredExecutedGrowthPassCount;
                    if (recovery.recoveredSampleCount == 0)
                    {
                        break;
                    }
                    result.statistics.supportedSampleCount += recovery.recoveredSampleCount;
                    for (std::size_t index = 0; index < supported.size(); ++index)
                    {
                        if (supported_before_recovery[index] == 0 && supported[index] != 0)
                        {
                            tsdf[index] = surface_candidate_tsdf[index];
                        }
                    }
                }
            }
            std::vector<std::uint8_t> eligible;
            if (options.enableGeometryZeroCrossingRecovery || options.enableGlobalImplicitRegularization ||
                options.enableAdaptiveTgvRegularization)
            {
                eligible.assign(supported.size(), 0);
                for (std::size_t index = 0; index < supported.size(); ++index)
                {
                    const std::uint16_t spread_value = minimumInverseDepthSpread[index];
                    eligible[index] = supported[index] == 0 &&
                                      maximumEvidenceSupportObservationWeight[index] >=
                                          options.minimumSurfacePatchObservationWeight &&
                                      bitCount(geometrySourceMask[index]) >= minimum_source_count &&
                                      spread_value != std::numeric_limits<std::uint16_t>::max() &&
                                      static_cast<float>(spread_value) / 100000.0f <= maximum_spread &&
                                      surfaceObservationWeight[index] > 1.0e-6f && weight[index] > 1.0e-6f &&
                                      surfaceObservationWeight[index] / weight[index] >= minimum_surface_weight_ratio &&
                                      std::fabs(surface_candidate_tsdf[index]) <= maximum_absolute_tsdf;
                }
            }
            if (options.enableGeometryZeroCrossingRecovery)
            {
                const std::vector<std::uint8_t> supported_before_recovery = supported;
                const DepthTsdfZeroCrossingRecoveryStatistics recovery =
                    options.enableGeometryZeroCrossingCellSheets
                        ? recoverGeometryVerifiedZeroCrossingCellSheets(
                              result.layout,
                              surface_candidate_tsdf,
                              weight,
                              geometrySourceMask,
                              eligible,
                              options.geometryZeroCrossingMinimumSupportedCorners,
                              options.minimumGeometryZeroCrossingSheetCells,
                              options.minimumGeometryZeroCrossingSheetAnchorCells,
                              options.maximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf,
                              &supported)
                        : DepthTsdfSurfaceBuilder::recoverGeometryVerifiedZeroCrossingSamples(
                              result.layout,
                              surface_candidate_tsdf,
                              weight,
                              geometrySourceMask,
                              eligible,
                              options.geometryZeroCrossingMinimumSupportedCorners,
                              options.geometryZeroCrossingMinimumCellVotes,
                              &supported);
                result.statistics.geometryZeroCrossingCandidateSampleCount = recovery.candidateSampleCount;
                result.statistics.geometryZeroCrossingRecoveredSampleCount = recovery.recoveredSampleCount;
                result.statistics.geometryZeroCrossingSheetCandidateCellCount = recovery.candidateCellCount;
                result.statistics.geometryZeroCrossingSheetAcceptedCellCount = recovery.acceptedCellCount;
                result.statistics.geometryZeroCrossingSheetComponentCount = recovery.componentCount;
                result.statistics.geometryZeroCrossingSheetAcceptedComponentCount = recovery.acceptedComponentCount;
                result.statistics.geometryZeroCrossingSheetRejectedSmallComponentCount =
                    recovery.rejectedSmallComponentCount;
                result.statistics.geometryZeroCrossingSheetRejectedAnchorComponentCount =
                    recovery.rejectedAnchorComponentCount;
                result.statistics.supportedSampleCount += recovery.recoveredSampleCount;
                for (std::size_t index = 0; index < supported.size(); ++index)
                {
                    if (supported_before_recovery[index] == 0 && supported[index] != 0)
                    {
                        tsdf[index] = surface_candidate_tsdf[index];
                    }
                }
            }
            if (options.enableGlobalImplicitRegularization)
            {
                if (options.progress)
                {
                    options.progress(QStringLiteral("正在进行多尺度隐式场正则化..."), 73);
                }
                DepthImplicitFieldRegularizationOptions regularization_options;
                regularization_options.coarseToFineLevels = options.implicitRegularizationLevels;
                regularization_options.passesPerLevel = options.implicitRegularizationPassesPerLevel;
                regularization_options.smoothness = options.implicitRegularizationSmoothness;
                regularization_options.dataFidelity = options.implicitRegularizationDataFidelity;
                regularization_options.maximumUpdate = options.implicitRegularizationMaximumUpdate;
                regularization_options.edgeThreshold = options.implicitRegularizationEdgeThreshold;
                regularization_options.recoverAxialGaps = options.implicitRegularizationRecoverAxialGaps;
                regularization_options.minimumBridgeAxes = options.implicitRegularizationMinimumBridgeAxes;
                regularization_options.maximumBridgePredictionDelta =
                    options.implicitRegularizationMaximumBridgePredictionDelta;
                const DepthImplicitFieldRegularizationStatistics regularization =
                    DepthImplicitFieldRegularizer::regularize(
                        {result.layout.cells[0] + 1, result.layout.cells[1] + 1, result.layout.cells[2] + 1},
                        surface_candidate_tsdf,
                        surfaceObservationWeight,
                        geometrySourceMask,
                        eligible,
                        regularization_options,
                        &tsdf,
                        &supported,
                        options.isCancelled);
                result.statistics.implicitRegularizationBridgeCandidateCount = regularization.bridgeCandidateCount;
                result.statistics.implicitRegularizationRecoveredSampleCount = regularization.recoveredSampleCount;
                result.statistics.implicitRegularizationUpdateOperationCount = regularization.updateOperationCount;
                result.statistics.implicitRegularizationMeanAbsoluteUpdate = regularization.meanAbsoluteUpdate;
                result.statistics.implicitRegularizationMaximumAbsoluteUpdate = regularization.maximumAbsoluteUpdate;
                result.statistics.implicitRegularizationElapsedMs = regularization.elapsedMs;
                result.statistics.supportedSampleCount += regularization.recoveredSampleCount;
                if (regularization.cancelled)
                {
                    result.errorMessage = QStringLiteral("TSDF 隐式场正则化已取消");
                    return false;
                }
            }
            if (options.enableAdaptiveTgvRegularization)
            {
                if (options.progress)
                {
                    options.progress(QStringLiteral("正在构建 2:1 平衡可见性八叉树..."), 72);
                }

                std::vector<std::uint8_t> active(supported.size(), 0);
                std::vector<float> adaptive_field = tsdf;
                constexpr float kHalfHistogramBinWidth = 1.0f / static_cast<float>(kDepthVisibilityHistogramBinCount);
                result.statistics.effectiveAdaptiveTgvMaximumActiveAbsoluteField =
                    options.adaptiveTgvMaximumActiveAbsoluteField;
                for (std::size_t index = 0; index < active.size(); ++index)
                {
                    if (visibilityHistograms[index].empty())
                    {
                        continue;
                    }
                    const bool unsupported_sample = weight[index] <= 1.0e-6f;
                    if (unsupported_sample &&
                        (!options.adaptiveTgvUseGlobalVisibilityField || !options.adaptiveTgvRecoverUnsupportedSamples))
                    {
                        continue;
                    }
                    const DepthVisibilityHistogramSummary histogram = visibilityHistograms[index].summary();
                    const float median = histogram.weightedMedian();
                    float candidate_value = median;
                    if (weight[index] <= 1.0e-6f)
                    {
                        candidate_value = median;
                    }
                    else
                    {
                        candidate_value =
                            std::clamp(tsdf[index], median - kHalfHistogramBinWidth, median + kHalfHistogramBinWidth);
                    }
                    if (std::fabs(candidate_value) > options.adaptiveTgvMaximumActiveAbsoluteField)
                    {
                        continue;
                    }
                    adaptive_field[index] = candidate_value;
                    active[index] = 1;
                    ++result.statistics.adaptiveTgvHistogramSampleCount;
                    if (weight[index] <= 1.0e-6f)
                    {
                        ++result.statistics.adaptiveTgvGlobalVisibilitySampleCount;
                    }
                }

                AdaptiveTsdfOctreeOptions octree_options;
                octree_options.maximumMergeLevel = options.adaptiveTgvMaximumMergeLevel;
                octree_options.minimumMergeAbsoluteField = options.adaptiveTgvMinimumMergeAbsoluteField;
                octree_options.maximumMergeFieldRange = options.adaptiveTgvMaximumMergeFieldRange;
                octree_options.workerCount = workerCount;

                AdaptiveTsdfOctreeResult octree;
                const auto octree_start = std::chrono::steady_clock::now();
                const double octree_cpu_start = detail::processCpuTimeMilliseconds();
                try
                {
                    octree = AdaptiveTsdfOctree::build(
                        {result.layout.cells[0] + 1, result.layout.cells[1] + 1, result.layout.cells[2] + 1},
                        adaptive_field,
                        weight,
                        geometrySourceMask,
                        active,
                        supported,
                        visibilityHistograms,
                        octree_options);
                }
                catch (const std::bad_alloc&)
                {
                    result.errorMessage = QStringLiteral("自适应 TGV 八叉树内存分配失败");
                    return false;
                }
                catch (const std::exception& error)
                {
                    result.errorMessage =
                        QStringLiteral("自适应 TGV 八叉树构建失败: %1").arg(QString::fromUtf8(error.what()));
                    return false;
                }
                result.statistics.adaptiveTgvOctreeElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                                   std::chrono::steady_clock::now() - octree_start)
                                                                   .count();
                result.statistics.adaptiveTgvOctreeCpuTimeMs = detail::processCpuTimeMilliseconds() - octree_cpu_start;
                result.statistics.adaptiveTgvOctreeCpuDuty =
                    result.statistics.adaptiveTgvOctreeElapsedMs > 0
                        ? result.statistics.adaptiveTgvOctreeCpuTimeMs /
                              (static_cast<double>(result.statistics.adaptiveTgvOctreeElapsedMs) * workerCount)
                        : 0.0;
                result.statistics.adaptiveTgvInputActiveSampleCount = octree.statistics.inputActiveSampleCount;
                result.statistics.adaptiveTgvLeafCount = octree.leaves.size();
                result.statistics.adaptiveTgvMergedNodeCount = octree.statistics.mergedNodeCount;
                result.statistics.adaptiveTgvBalanceSplitCount = octree.statistics.balanceSplitCount;
                result.statistics.adaptiveTgvTwoToOneBalanced = octree.statistics.twoToOneBalanced;
                if (octree.leaves.empty())
                {
                    result.errorMessage = QStringLiteral("自适应 TGV 八叉树没有可求解节点");
                    return false;
                }
                if (options.isCancelled && options.isCancelled())
                {
                    result.errorMessage = QStringLiteral("自适应 TGV 八叉树构建已取消");
                    return false;
                }

                SparseTgvOptions tgv_options;
                tgv_options.maximumIterations = options.adaptiveTgvMaximumIterations;
                tgv_options.minimumIterations = options.adaptiveTgvMinimumIterations;
                tgv_options.firstOrderWeight = options.adaptiveTgvFirstOrderWeight;
                tgv_options.secondOrderWeight = options.adaptiveTgvSecondOrderWeight;
                tgv_options.dataFidelity = options.adaptiveTgvDataFidelity;
                tgv_options.primalStep = options.adaptiveTgvPrimalStep;
                tgv_options.dualStep = options.adaptiveTgvDualStep;
                tgv_options.convergenceTolerance = options.adaptiveTgvConvergenceTolerance;
                tgv_options.workerCount = workerCount;
                const SparseTgvStatistics tgv = SparseTgvSolver::solve(
                    tgv_options,
                    &octree,
                    options.isCancelled,
                    [&](int iteration, int maximum_iterations)
                    {
                        if (options.progress &&
                            (iteration == 1 || iteration % 5 == 0 || iteration == maximum_iterations))
                        {
                            options.progress(QStringLiteral("正在进行稀疏 TGV 全局优化（%1/%2）...")
                                                 .arg(iteration)
                                                 .arg(maximum_iterations),
                                             73 + iteration * 3 / std::max(1, maximum_iterations));
                        }
                    });
                result.statistics.adaptiveTgvIterationCount = tgv.iterationCount;
                result.statistics.adaptiveTgvInitialMeanAbsoluteCurvature = tgv.initialMeanAbsoluteCurvature;
                result.statistics.adaptiveTgvFinalMeanAbsoluteCurvature = tgv.finalMeanAbsoluteCurvature;
                result.statistics.adaptiveTgvFinalMeanAbsoluteUpdate = tgv.finalMeanAbsoluteUpdate;
                result.statistics.adaptiveTgvSolverElapsedMs = tgv.elapsedMs;
                result.statistics.adaptiveTgvSolverCpuTimeMs = tgv.cpuTimeMs;
                result.statistics.adaptiveTgvSolverCpuDuty = tgv.cpuDuty;
                result.statistics.adaptiveTgvSolverWorkerCount = tgv.effectiveWorkerCount;
                if (tgv.cancelled)
                {
                    result.errorMessage = QStringLiteral("稀疏 TGV 全局优化已取消");
                    return false;
                }

                for (const AdaptiveTsdfOctreeNode& leaf : octree.leaves)
                {
                    const int end_x = std::min(result.layout.cells[0] + 1, leaf.origin[0] + leaf.size);
                    const int end_y = std::min(result.layout.cells[1] + 1, leaf.origin[1] + leaf.size);
                    const int end_z = std::min(result.layout.cells[2] + 1, leaf.origin[2] + leaf.size);
                    for (int z = leaf.origin[2]; z < end_z; ++z)
                    {
                        for (int y = leaf.origin[1]; y < end_y; ++y)
                        {
                            for (int x = leaf.origin[0]; x < end_x; ++x)
                            {
                                const std::size_t index = sampleIndex(result.layout, x, y, z);
                                if (active[index] != 0)
                                {
                                    tsdf[index] = leaf.value;
                                }
                            }
                        }
                    }
                }
                if (options.adaptiveTgvUseGlobalVisibilityField)
                {
                    adaptiveTgvExtractionSupport = active;
                    if (!options.adaptiveTgvRecoverUnsupportedSamples)
                    {
                        for (std::size_t index = 0; index < adaptiveTgvExtractionSupport.size(); ++index)
                        {
                            adaptiveTgvExtractionSupport[index] =
                                adaptiveTgvExtractionSupport[index] != 0 && supported[index] != 0 ? 1 : 0;
                        }
                    }
                }
                else if (options.adaptiveTgvRecoverUnsupportedSamples)
                {
                    std::vector<std::uint8_t> robust_eligible = eligible;
                    const float maximum_conflict_ratio =
                        std::clamp(options.adaptiveTgvMaximumRecoveryConflictRatio, 0.0f, 0.5f);
                    for (std::size_t index = 0; index < robust_eligible.size(); ++index)
                    {
                        result.statistics.adaptiveTgvRecoveryEligibleSampleCount += robust_eligible[index] != 0;
                        if (robust_eligible[index] != 0 &&
                            visibilityHistograms[index].summary().conflictingSignRatio() > maximum_conflict_ratio)
                        {
                            robust_eligible[index] = 0;
                            ++result.statistics.adaptiveTgvRecoveryConflictRejectedSampleCount;
                        }
                    }
                    const int recovery_passes = std::clamp(options.adaptiveTgvRecoveryPasses, 1, 6);
                    for (int pass = 0; pass < recovery_passes; ++pass)
                    {
                        const DepthTsdfZeroCrossingRecoveryStatistics recovery =
                            options.enableGeometryZeroCrossingCellSheets
                                ? recoverGeometryVerifiedZeroCrossingCellSheets(
                                      result.layout,
                                      tsdf,
                                      weight,
                                      geometrySourceMask,
                                      robust_eligible,
                                      options.adaptiveTgvMinimumRecoveryNeighbors,
                                      options.minimumGeometryZeroCrossingSheetCells,
                                      options.minimumGeometryZeroCrossingSheetAnchorCells,
                                      options.maximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf,
                                      &supported)
                                : DepthTsdfSurfaceBuilder::recoverGeometryVerifiedZeroCrossingSamples(
                                      result.layout,
                                      tsdf,
                                      weight,
                                      geometrySourceMask,
                                      robust_eligible,
                                      options.adaptiveTgvMinimumRecoveryNeighbors,
                                      options.geometryZeroCrossingMinimumCellVotes,
                                      &supported);
                        result.statistics.adaptiveTgvRecoveredSampleCount += recovery.recoveredSampleCount;
                        result.statistics.geometryZeroCrossingSheetCandidateCellCount += recovery.candidateCellCount;
                        result.statistics.geometryZeroCrossingSheetAcceptedCellCount += recovery.acceptedCellCount;
                        result.statistics.geometryZeroCrossingSheetComponentCount += recovery.componentCount;
                        result.statistics.geometryZeroCrossingSheetAcceptedComponentCount +=
                            recovery.acceptedComponentCount;
                        result.statistics.geometryZeroCrossingSheetRejectedSmallComponentCount +=
                            recovery.rejectedSmallComponentCount;
                        result.statistics.geometryZeroCrossingSheetRejectedAnchorComponentCount +=
                            recovery.rejectedAnchorComponentCount;
                        result.statistics.supportedSampleCount += recovery.recoveredSampleCount;
                        if (recovery.recoveredSampleCount == 0)
                        {
                            break;
                        }
                    }
                }
            }
            result.statistics.surfacePatchCreatedComponentCount = 0;
        }
        const bool use_visual_hull_completion =
            options.enableVisualHullSignedDistanceCompletion && !options.enableVisibilityOccupancyCompletion;
        result.statistics.effectiveVisualHullSignedDistanceCompletion = use_visual_hull_completion;
        result.statistics.effectiveVisualHullCompletionTopologyGuard =
            use_visual_hull_completion && options.enableVisualHullCompletionTopologyGuard;
        result.statistics.effectiveVisualHullCompletionBandVoxels =
            use_visual_hull_completion ? std::clamp(options.visualHullCompletionBandVoxels, 1.0f, 24.0f) : 0.0f;

        if (use_visual_hull_completion)
        {
            if (options.progress)
            {
                options.progress(QStringLiteral("正在构建轮廓约束的局部有符号距离先验..."), 74);
            }
            const std::vector<std::uint8_t> occupied =
                buildVisualHullOccupancy(result.layout,
                                         retained_frames,
                                         options.visualHullCompletionMinimumVisibleViews,
                                         options.visualHullCompletionAllowedSilhouetteViolations,
                                         options.isCancelled);
            if (options.isCancelled && options.isCancelled())
            {
                result.errorMessage = QStringLiteral("TSDF 轮廓距离先验构建已取消");
                return false;
            }
            visual_hull_completion_tsdf = tsdf;
            visual_hull_completion_support = supported;
            std::vector<std::uint8_t> observed_conflict_veto;
            if (options.visualHullCompletionPreserveObservedTsdf &&
                occupied.size() == visual_hull_completion_support.size())
            {
                observed_conflict_veto.assign(visual_hull_completion_support.size(), 0);
                const float maximum_absolute_tsdf =
                    std::clamp(options.visualHullCompletionMaximumObservedAbsoluteTsdf, 0.05f, 1.0f);
                const int minimum_geometry_support = std::max(1, options.visualHullCompletionMinimumGeometrySupport);
                for (std::size_t index = 0; index < visual_hull_completion_support.size(); ++index)
                {
                    if (weight[index] <= 1.0e-6f || maximumGeometrySupportCount[index] < minimum_geometry_support ||
                        bitCount(geometrySourceMask[index]) < 2 ||
                        std::fabs(visual_hull_completion_tsdf[index]) > maximum_absolute_tsdf)
                    {
                        continue;
                    }
                    const bool hull_inside = occupied[index] != 0;
                    const bool tsdf_inside = visual_hull_completion_tsdf[index] < 0.0f;
                    if (hull_inside != tsdf_inside)
                    {
                        observed_conflict_veto[index] = 1;
                        ++result.statistics.visualHullCompletionObservedConflictVetoSampleCount;
                    }
                    if (visual_hull_completion_support[index] == 0)
                    {
                        visual_hull_completion_support[index] = 1;
                        ++result.statistics.visualHullCompletionPreservedObservedSampleCount;
                    }
                }
            }
            const std::vector<std::uint8_t> supported_before_hull_completion = visual_hull_completion_support;
            const DepthTsdfVisualHullCompletionStatistics completion =
                DepthTsdfSurfaceBuilder::completeUnsupportedSamplesWithVisualHullSignedDistance(
                    result.layout,
                    occupied,
                    result.statistics.effectiveVisualHullCompletionBandVoxels,
                    &visual_hull_completion_tsdf,
                    &visual_hull_completion_support,
                    observed_conflict_veto.empty() ? nullptr : &observed_conflict_veto);
            result.statistics.visualHullCompletionOccupiedSampleCount = completion.occupiedSampleCount;
            result.statistics.visualHullCompletionBoundarySampleCount = completion.boundarySampleCount;
            result.statistics.visualHullCompletionAnchorCellCount = completion.anchorCellCount;
            result.statistics.visualHullCompletionFrontierCellCount = completion.frontierCellCount;
            result.statistics.visualHullCompletionRecoveredSampleCount = completion.recoveredSampleCount;
            if (completion.recoveredSampleCount > 0 && options.visualHullCompletionRelaxationIterations > 0)
            {
                std::vector<std::uint8_t> completion_mask(visual_hull_completion_support.size(), 0);
                for (std::size_t index = 0; index < completion_mask.size(); ++index)
                {
                    completion_mask[index] =
                        supported_before_hull_completion[index] == 0 && visual_hull_completion_support[index] != 0 ? 1
                                                                                                                   : 0;
                }
                result.statistics.visualHullCompletionRelaxedSampleCount =
                    relaxVisualHullCompletionField(result.layout,
                                                   occupied,
                                                   completion_mask,
                                                   visual_hull_completion_support,
                                                   options.visualHullCompletionRelaxationIterations,
                                                   options.visualHullCompletionRelaxationLambda,
                                                   options.visualHullCompletionMaximumUpdate,
                                                   &visual_hull_completion_tsdf);
            }
        }
        result.statistics.effectiveVisibilityOccupancyCompletion = options.enableVisibilityOccupancyCompletion;
        result.statistics.effectiveVisibilityOccupancyUseSupportMaskSilhouette =
            options.visibilityOccupancyUseSupportMaskSilhouette;
        result.statistics.effectiveVisibilityOccupancyResolution =
            options.enableVisibilityOccupancyCompletion ? std::clamp(options.visibilityOccupancyResolution, 24, 128)
                                                        : 0;

        if (options.enableVisibilityOccupancyCompletion)
        {
            if (options.progress)
            {
                options.progress(QStringLiteral("正在求解可见性约束的全局空实占据场..."), 74);
            }
            std::vector<VisibilityOccupancyFrameView> occupancy_frames;
            occupancy_frames.reserve(static_cast<std::size_t>(frames.size()));
            for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
            {
                const DepthTsdfFrame& frame = frames[frame_index];
                if (frame.auxiliarySurfaceOnly)
                {
                    continue;
                }
                if (effective_frame_quality_weights[frame_index] <= 0.0f)
                {
                    continue;
                }
                VisibilityOccupancyFrameView view;
                view.camera = &frame.camera;
                view.depth = &frame.depth;
                view.confidence = &frame.confidence;
                view.depthValidMask =
                    erosion_pixels > 0 ? &effective_depth_valid_masks[frame_index] : &frame.depthValidMask;
                view.supportMask = &frame.supportMask;
                view.frameWeight = effective_frame_quality_weights[frame_index];
                occupancy_frames.push_back(view);
            }
            result.statistics.visibilityOccupancyInputFrameCount = static_cast<int>(occupancy_frames.size());

            VisibilityOccupancyOptions occupancy_options;
            occupancy_options.useSupportMaskSilhouette = options.visibilityOccupancyUseSupportMaskSilhouette;
            occupancy_options.resolution = result.statistics.effectiveVisibilityOccupancyResolution;
            if (options.visibilityOccupancyAlignCarrierGrid && options.resolution >= occupancy_options.resolution &&
                options.resolution % occupancy_options.resolution == 0)
            {
                const int grid_scale = options.resolution / occupancy_options.resolution;
                for (int axis = 0; axis < 3; ++axis)
                {
                    if (grid_scale > 1 && result.layout.cells[axis] % grid_scale == 0)
                    {
                        occupancy_options.sampleDimensions[axis] = result.layout.cells[axis] / grid_scale + 1;
                    }
                }
            }
            occupancy_options.minimumVisibleViews = std::clamp(options.visibilityOccupancyMinimumVisibleViews, 1, 16);
            occupancy_options.minimumSilhouetteViews =
                std::clamp(options.visibilityOccupancyMinimumSilhouetteViews, 1, 16);
            occupancy_options.minimumDepthFullViewsForSilhouettePrior =
                std::clamp(options.visibilityOccupancyMinimumDepthFullViewsForSilhouettePrior, 0, 16);
            occupancy_options.allowedSilhouetteViolations =
                std::clamp(options.visibilityOccupancyAllowedSilhouetteViolations, 0, 8);
            occupancy_options.frontTolerancePixelFootprints =
                std::clamp(options.visibilityOccupancyFrontTolerancePixelFootprints, 0.5f, 12.0f);
            occupancy_options.behindSurfaceBandPixelFootprints =
                std::clamp(options.visibilityOccupancyBehindSurfaceBandPixelFootprints, 1.0f, 64.0f);
            occupancy_options.depthEmptyCapacity = std::max(0, options.visibilityOccupancyDepthEmptyCapacity);
            occupancy_options.depthFullCapacity = std::max(0, options.visibilityOccupancyDepthFullCapacity);
            occupancy_options.silhouetteEmptyCapacity = std::max(0, options.visibilityOccupancySilhouetteEmptyCapacity);
            occupancy_options.silhouetteFullPriorCapacity =
                std::max(0, options.visibilityOccupancySilhouetteFullPriorCapacity);
            occupancy_options.pairwiseCapacity = std::max(0, options.visibilityOccupancyPairwiseCapacity);
            result.statistics.effectiveVisibilityOccupancyPairwiseCapacity = occupancy_options.pairwiseCapacity;
            occupancy_options.closingIterations = std::clamp(options.visibilityOccupancyClosingIterations, 0, 8);
            result.statistics.effectiveVisibilityOccupancyClosingIterations = occupancy_options.closingIterations;
            occupancy_options.maximumHandleRepairPasses =
                std::clamp(options.visibilityOccupancyMaximumHandleRepairPasses, 1, 16);
            occupancy_options.maximumHandleRepairAcceptedCandidateCount =
                std::clamp(options.visibilityOccupancyMaximumHandleRepairAcceptedCandidateCount, 0, 512);
            occupancy_options.maximumHandleRepairCandidateSampleCount =
                std::max<std::size_t>(1, options.visibilityOccupancyMaximumHandleRepairCandidateSampleCount);
            occupancy_options.maximumHandleRepairSubsetSampleCount =
                std::clamp<std::size_t>(options.visibilityOccupancyMaximumHandleRepairSubsetSampleCount, 1, 1024);
            occupancy_options.maximumHandleRepairSubsetSeedCount =
                std::clamp(options.visibilityOccupancyMaximumHandleRepairSubsetSeedCount, 0, 4096);
            occupancy_options.repairNonManifoldConfigurations = options.visibilityOccupancyCellBoundaryExtraction;
            occupancy_options.closingMinimumDepthEmptyViewsToProtect =
                std::clamp(options.visibilityOccupancyClosingMinimumDepthEmptyViewsToProtect, 1, 16);
            occupancy_options.closingMinimumSilhouetteOutsideViewsToProtect =
                std::clamp(options.visibilityOccupancyClosingMinimumSilhouetteOutsideViewsToProtect, 1, 16);
            occupancy_options.buildSignedDistanceSamples = false;
            occupancy_options.workerCount = workerCount;
            occupancy_options.isCancelled = options.isCancelled;
            const float minimum_full_fraction =
                std::clamp(options.visibilityOccupancyAdaptiveDepthSupportMinimumFullFraction, 0.0f, 0.25f);
            VisibilityOccupancyResult occupancy;
            int depth_support_fallback_count = 0;
            while (true)
            {
                occupancy = VisibilityOccupancySurfaceBuilder::build(
                    result.layout.boundsMin, result.layout.boundsMax, occupancy_frames, occupancy_options);
                const bool empty_cut = occupancy.error == "visibility occupancy cut contains no full samples";
                const bool accepted_cut = DepthTsdfSurfaceBuilder::shouldApplyVisibilityOccupancyCut(
                    occupancy.ok,
                    empty_cut,
                    occupancy.statistics.sampleCount,
                    occupancy.statistics.fullSampleCountAfterCleanup,
                    minimum_full_fraction);
                if (occupancy.cancelled || accepted_cut ||
                    occupancy_options.minimumDepthFullViewsForSilhouettePrior <= 0)
                {
                    break;
                }
                const int current_threshold = occupancy_options.minimumDepthFullViewsForSilhouettePrior;
                occupancy_options.minimumDepthFullViewsForSilhouettePrior = current_threshold > 1 ? 1 : 0;
                ++depth_support_fallback_count;
            }
            const bool rejected_empty_cut = occupancy.error == "visibility occupancy cut contains no full samples";
            const bool accepted_cut = DepthTsdfSurfaceBuilder::shouldApplyVisibilityOccupancyCut(
                occupancy.ok,
                rejected_empty_cut,
                occupancy.statistics.sampleCount,
                occupancy.statistics.fullSampleCountAfterCleanup,
                minimum_full_fraction);
            const bool rejected_collapsed_cut = occupancy.ok && !rejected_empty_cut && !accepted_cut;
            result.statistics.visibilityOccupancyRejectedEmptyCut = rejected_empty_cut;
            result.statistics.visibilityOccupancyRejectedCollapsedCut = rejected_collapsed_cut;
            result.statistics.effectiveVisibilityOccupancyCompletion = !rejected_empty_cut && !rejected_collapsed_cut;
            if (occupancy.cancelled)
            {
                result.errorMessage = QStringLiteral("Visibility occupancy solve was cancelled");
                return false;
            }
            if (!occupancy.ok && !rejected_empty_cut)
            {
                result.errorMessage =
                    occupancy.cancelled
                        ? QStringLiteral("可见性占据场求解已取消")
                        : QStringLiteral("可见性占据场求解失败: %1").arg(QString::fromStdString(occupancy.error));
                return false;
            }
            VisibilityOccupancyDistanceFieldResult distance_field;
            if (result.statistics.effectiveVisibilityOccupancyCompletion)
            {
                distance_field = VisibilityOccupancyDistanceField::build(
                    occupancy.sampleDimensions, occupancy.boundsMin, occupancy.boundsMax, occupancy.occupied);
            }
            if (result.statistics.effectiveVisibilityOccupancyCompletion && !distance_field.ok)
            {
                result.errorMessage =
                    QStringLiteral("可见性占据场距离场构建失败: %1").arg(QString::fromStdString(distance_field.error));
                return false;
            }
            if (result.statistics.effectiveVisibilityOccupancyCompletion)
            {
                occupancy.signedDistanceSamples = std::move(distance_field.signedWorldDistance);
                occupancy.signedDistanceSamplesAreWorldUnits = true;
            }
            if (result.statistics.effectiveVisibilityOccupancyCompletion &&
                (options.visibilityOccupancyNativeCarrierExtraction ||
                 options.visibilityOccupancyCellBoundaryExtraction))
            {
                native_carrier_bounds_min = occupancy.boundsMin;
                native_carrier_bounds_max = occupancy.boundsMax;
                native_carrier_dimensions = occupancy.sampleDimensions;
                for (int axis = 0; axis < 3; ++axis)
                {
                    native_carrier_cells[axis] = occupancy.sampleDimensions[axis] - 1;
                }
                native_carrier_occupied = occupancy.occupied;
                native_carrier_field = std::move(occupancy.signedDistanceSamples);
            }
            result.statistics.visibilityOccupancySampleCount = occupancy.statistics.sampleCount;
            result.statistics.effectiveVisibilityOccupancyMinimumDepthFullViewsForSilhouettePrior =
                occupancy_options.minimumDepthFullViewsForSilhouettePrior;
            result.statistics.visibilityOccupancyDepthSupportFallbackCount = depth_support_fallback_count;
            result.statistics.visibilityOccupancyDepthEmptyVoteCount = occupancy.statistics.depthEmptyVoteCount;
            result.statistics.visibilityOccupancyDepthFullVoteCount = occupancy.statistics.depthFullVoteCount;
            result.statistics.visibilityOccupancySilhouetteFullPriorCandidateSampleCount =
                occupancy.statistics.silhouetteFullPriorCandidateSampleCount;
            result.statistics.visibilityOccupancySilhouetteFullPriorSampleCount =
                occupancy.statistics.silhouetteFullPriorSampleCount;
            result.statistics.visibilityOccupancySilhouetteFullPriorRejectedWithoutDepthSupportSampleCount =
                occupancy.statistics.silhouetteFullPriorRejectedWithoutDepthSupportSampleCount;
            result.statistics.visibilityOccupancySilhouetteFullPriorCapacityTotal =
                occupancy.statistics.silhouetteFullPriorCapacityTotal;
            result.statistics.visibilityOccupancyFullSampleCount = occupancy.statistics.fullSampleCountAfterCleanup;
            result.statistics.visibilityOccupancyFilledBubbleSampleCount =
                occupancy.statistics.filledInteriorEmptySampleCount;
            result.statistics.visibilityOccupancyRemovedDustSampleCount =
                occupancy.statistics.removedFullDustSampleCount;
            result.statistics.visibilityOccupancyClosingChangedSampleCount =
                occupancy.statistics.closingChangedSampleCount;
            result.statistics.visibilityOccupancyClosingProposalAddedSampleCount =
                occupancy.statistics.closingProposalAddedSampleCount;
            result.statistics.visibilityOccupancyClosingProposalDepthEmptyAtLeastTwoSampleCount =
                occupancy.statistics.closingProposalDepthEmptyAtLeastTwoSampleCount;
            result.statistics.visibilityOccupancyClosingProposalDepthEmptyAtLeastThreeSampleCount =
                occupancy.statistics.closingProposalDepthEmptyAtLeastThreeSampleCount;
            result.statistics.visibilityOccupancyClosingProposalDepthEmptyAtLeastFourSampleCount =
                occupancy.statistics.closingProposalDepthEmptyAtLeastFourSampleCount;
            result.statistics.visibilityOccupancyClosingProposalDepthFullSampleCount =
                occupancy.statistics.closingProposalDepthFullSampleCount;
            result.statistics.visibilityOccupancyClosingProposalSilhouetteOutsideAtLeastTwoSampleCount =
                occupancy.statistics.closingProposalSilhouetteOutsideAtLeastTwoSampleCount;
            result.statistics.visibilityOccupancyClosingProtectedEmptySampleCount =
                occupancy.statistics.closingProtectedEmptySampleCount;
            result.statistics.visibilityOccupancyClosingDepthEmptyProtectedSampleCount =
                occupancy.statistics.closingDepthEmptyProtectedSampleCount;
            result.statistics.visibilityOccupancyClosingSilhouetteEmptyProtectedSampleCount =
                occupancy.statistics.closingSilhouetteEmptyProtectedSampleCount;
            result.statistics.visibilityOccupancyHandleRepairCandidateComponentCount =
                occupancy.statistics.handleRepairCandidateComponentCount;
            result.statistics.visibilityOccupancyHandleRepairAcceptedCandidateCount =
                occupancy.statistics.handleRepairAcceptedCandidateCount;
            result.statistics.visibilityOccupancyHandleRepairAcceptedSubsetCandidateCount =
                occupancy.statistics.handleRepairAcceptedSubsetCandidateCount;
            result.statistics.visibilityOccupancyHandleRepairAcceptedPlateauSubsetCandidateCount =
                occupancy.statistics.handleRepairAcceptedPlateauSubsetCandidateCount;
            result.statistics.visibilityOccupancyHandleRepairAttemptedSubsetSeedCount =
                occupancy.statistics.handleRepairAttemptedSubsetSeedCount;
            result.statistics.visibilityOccupancyHandleRepairRejectedProtectedCandidateCount =
                occupancy.statistics.handleRepairRejectedProtectedCandidateCount;
            result.statistics.visibilityOccupancyHandleRepairRejectedOversizedCandidateCount =
                occupancy.statistics.handleRepairRejectedOversizedCandidateCount;
            result.statistics.visibilityOccupancyHandleRepairRejectedTopologyCandidateCount =
                occupancy.statistics.handleRepairRejectedTopologyCandidateCount;
            result.statistics.visibilityOccupancyHandleRepairRejectedProtectedReachabilityCandidateCount =
                occupancy.statistics.handleRepairRejectedProtectedReachabilityCandidateCount;
            result.statistics.visibilityOccupancyHandleRepairBodyEulerBefore =
                occupancy.statistics.handleRepairBodyEulerBefore;
            result.statistics.visibilityOccupancyHandleRepairBodyEulerAfter =
                occupancy.statistics.handleRepairBodyEulerAfter;
            result.statistics.visibilityOccupancyWellComposedRepairFilledSampleCount =
                occupancy.statistics.wellComposedRepairFilledSampleCount;
            result.statistics.visibilityOccupancyWellComposedRepairAcceptedPassCount =
                occupancy.statistics.wellComposedRepairAcceptedPassCount;
            result.statistics.visibilityOccupancyWellComposedRepairBodyEulerBefore =
                occupancy.statistics.wellComposedRepairBodyEulerBefore;
            result.statistics.visibilityOccupancyWellComposedRepairBodyEulerAfter =
                occupancy.statistics.wellComposedRepairBodyEulerAfter;
            result.statistics.visibilityOccupancyWellComposedRepairRemainingEdgeCheckerboardCount =
                occupancy.statistics.wellComposedRepairRemainingEdgeCheckerboardCount;
            result.statistics.visibilityOccupancyWellComposedRepairRemainingVertexOccupiedDefectCount =
                occupancy.statistics.wellComposedRepairRemainingVertexOccupiedComponentDefectCount;
            result.statistics.visibilityOccupancyWellComposedRepairRemainingVertexEmptyDefectCount =
                occupancy.statistics.wellComposedRepairRemainingVertexEmptyComponentDefectCount;
            result.statistics.visibilityOccupancyCutEnergy = occupancy.statistics.minCut.cutEnergy;
            result.statistics.visibilityOccupancyWorkerCount = occupancy.statistics.effectiveWorkerCount;
            result.statistics.visibilityOccupancyProjectionElapsedMs = occupancy.statistics.projectionElapsedMs;
            result.statistics.visibilityOccupancyProjectionCpuTimeMs = occupancy.statistics.projectionCpuTimeMs;
            result.statistics.visibilityOccupancyProjectionCpuDuty = occupancy.statistics.projectionCpuDuty;
            result.statistics.visibilityOccupancyMinCutElapsedMs = occupancy.statistics.minCutElapsedMs;
            result.statistics.visibilityOccupancyMinCutCpuTimeMs = occupancy.statistics.minCutCpuTimeMs;
            result.statistics.visibilityOccupancyMinCutCpuDuty = occupancy.statistics.minCutCpuDuty;
            result.statistics.visibilityOccupancyCleanupElapsedMs = occupancy.statistics.cleanupElapsedMs;
            result.statistics.visibilityOccupancyCleanupCpuTimeMs = occupancy.statistics.cleanupCpuTimeMs;
            result.statistics.visibilityOccupancyCleanupCpuDuty = occupancy.statistics.cleanupCpuDuty;

            std::vector<float>* completion_tsdf =
                visual_hull_completion_tsdf.empty() ? &tsdf : &visual_hull_completion_tsdf;
            std::vector<std::uint8_t>* completion_support =
                visual_hull_completion_support.empty() ? &supported : &visual_hull_completion_support;
            VisibilityOccupancyTsdfCompletionOptions completion_options;
            completion_options.enableTopologyLockedResidualBlend =
                options.visibilityOccupancyTopologyLockedResidualBlend;
            completion_options.truncationDistanceWorld = truncation;
            completion_options.observedBand = std::clamp(options.visibilityOccupancyObservedBand, 0.01f, 1.0f);
            completion_options.carrierBand = std::clamp(options.visibilityOccupancyCarrierBand, 0.01f, 1.0f);
            completion_options.maximumResidual = std::clamp(options.visibilityOccupancyMaximumResidual, 0.0f, 1.0f);
            completion_options.detailBlend = std::clamp(options.visibilityOccupancyDetailBlend, 0.0f, 1.0f);
            completion_options.preserveAllObservedSamples = options.visibilityOccupancyPreserveAllObservedSamples;
            completion_options.preserveObservedNearSurface = options.visibilityOccupancyPreserveObservedNearSurface;
            completion_options.requireOccupancySignAgreement = options.visibilityOccupancyRequireSignAgreement;
            completion_options.maximumPreservedAbsoluteTsdf =
                std::clamp(options.visibilityOccupancyMaximumPreservedAbsoluteTsdf, 0.0f, 1.0f);
            completion_options.signedDistanceNormalizationSamples =
                std::clamp(options.visibilityOccupancySignedDistanceNormalizationSamples, 0.5f, 16.0f);
            VisibilityOccupancyTsdfCompletionStatistics completion;
            if (result.statistics.effectiveVisibilityOccupancyCompletion)
            {
                completion = VisibilityOccupancyTsdfCompletion::apply(
                    result.layout, occupancy, completion_options, completion_tsdf, completion_support);
            }
            result.statistics.visibilityOccupancyRecoveredUnsupportedSampleCount =
                completion.recoveredUnsupportedSampleCount;
            result.statistics.visibilityOccupancyPreservedObservedSampleCount = completion.preservedObservedSampleCount;
            result.statistics.visibilityOccupancyOverriddenObservedSampleCount =
                completion.overriddenObservedSampleCount;
            result.statistics.visibilityOccupancyForcedBoundarySampleCount =
                completion.forcedExteriorBoundarySampleCount;
            result.statistics.visibilityOccupancyAdjustedExactIsoValueSampleCount =
                completion.adjustedExactIsoValueSampleCount;
            result.statistics.visibilityOccupancyTrustedObservationSampleCount =
                completion.trustedObservationSampleCount;
            result.statistics.visibilityOccupancyIgnoredSignConflictObservationCount =
                completion.ignoredSignConflictObservationCount;
            result.statistics.visibilityOccupancyBlendedSampleCount = completion.blendedSampleCount;
            result.statistics.visibilityOccupancyClippedResidualSampleCount = completion.clippedResidualSampleCount;
            result.statistics.visibilityOccupancyCarrierSignMismatchSampleCount =
                completion.carrierSignMismatchSampleCount;
            result.statistics.visibilityOccupancyMaximumAppliedResidual = completion.maximumAppliedResidual;
        }
        if (result.statistics.supportedSampleCount == 0)
        {
            result.errorMessage = QStringLiteral("TSDF integration produced no multi-camera supported samples");
            return false;
        }

        result.statistics.effectiveZeroCrossingDiagnostics = options.collectZeroCrossingDiagnostics;
        result.statistics.effectiveVisibilityOccupancyCellBoundaryExtraction =
            result.statistics.effectiveVisibilityOccupancyCompletion &&
            options.visibilityOccupancyCellBoundaryExtraction;
        result.statistics.effectiveConsistentIsoSurfaceExtraction =
            options.enableConsistentIsoSurfaceExtraction &&
            !result.statistics.effectiveVisibilityOccupancyCellBoundaryExtraction;
        result.statistics.effectiveMc33IsoSurfaceExtraction =
            options.enableMc33IsoSurfaceExtraction &&
            !result.statistics.effectiveVisibilityOccupancyCellBoundaryExtraction;
        result.statistics.effectiveMc33RequireSupportedSignChange =
            result.statistics.effectiveMc33IsoSurfaceExtraction && options.mc33RequireSupportedSignChange;
        if (options.enableConsistentIsoSurfaceExtraction && options.enableMc33IsoSurfaceExtraction &&
            !result.statistics.effectiveVisibilityOccupancyCellBoundaryExtraction)
        {
            result.errorMessage =
                QStringLiteral("TSDF consistent and MC33 iso-surface extractors cannot both be enabled");
            return false;
        }
        if (options.collectZeroCrossingDiagnostics || options.collectAcquisitionGapReport)
        {
            const DepthTsdfZeroCrossingStatistics zero_crossings =
                DepthTsdfSurfaceBuilder::analyzeZeroCrossings(result.layout, tsdf, weight, supported);
            result.statistics.zeroCrossingObservedCellCount = zero_crossings.observedCellCount;
            result.statistics.zeroCrossingRawCandidateCellCount = zero_crossings.rawCandidateCellCount;
            result.statistics.zeroCrossingExtractableCellCount = zero_crossings.extractableCellCount;
            result.statistics.zeroCrossingSuppressedBySupportCellCount = zero_crossings.suppressedBySupportCellCount;
            result.statistics.zeroCrossingPositiveOnlySupportedCellCount =
                zero_crossings.positiveOnlySupportedCellCount;
            result.statistics.zeroCrossingNegativeOnlySupportedCellCount =
                zero_crossings.negativeOnlySupportedCellCount;
            result.statistics.zeroCrossingPartiallySupportedCellCount = zero_crossings.partiallySupportedCellCount;
            result.statistics.zeroCrossingFullyUnsupportedObservedCellCount =
                zero_crossings.fullyUnsupportedObservedCellCount;
        }

        if (options.progress)
        {
            options.progress(QStringLiteral("正在提取 TSDF 零等值面..."), 75);
        }

        return true;
    }

} // namespace xjw::mesh
