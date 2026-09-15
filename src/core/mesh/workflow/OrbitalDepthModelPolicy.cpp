#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow
{
    using namespace workflow_detail;

    xjw::mesh::DepthTsdfOptions depthTsdfOptionsFromSettings(const QJsonObject& settings, int requestedResolution)
    {
        return makeDepthTsdfOptions(settings, requestedResolution);
    }

    void applyOrbitalDepthTsdfDefaults(const QJsonObject& settings,
                                       xjw::mesh::DepthTsdfOptions* options,
                                       int maximumReliableResolution)
    {
        if (!options)
        {
            return;
        }

        if (!settings.contains(QStringLiteral("tsdfSupportMaskFreeSpaceCarving")))
        {
            options->enableSupportMaskFreeSpaceCarving = false;
        }
        if (!settings.contains(QStringLiteral("tsdfNarrowBandActivation")))
        {
            options->enableNarrowBandActivation = true;
        }
        if (!settings.contains(QStringLiteral("tsdfPixelEvidenceWeighting")))
        {
            options->enablePixelEvidenceWeighting = true;
        }
        if (!settings.contains(QStringLiteral("tsdfEvidenceSupportWeightDecoupling")))
        {
            options->enableEvidenceSupportWeightDecoupling = true;
        }
        if (!settings.contains(QStringLiteral("tsdfAdaptiveConflictRobustWeighting")))
        {
            options->enableAdaptiveConflictRobustWeighting = true;
        }
        if (!settings.contains(QStringLiteral("tsdfAdaptiveConflictWeightKnee")))
        {
            options->adaptiveConflictWeightKnee = 0.20f;
        }
        if (!settings.contains(QStringLiteral("tsdfAdaptiveConflictWeightZero")))
        {
            options->adaptiveConflictWeightZero = 0.75f;
        }
        if (!settings.contains(QStringLiteral("tsdfMinimumAdaptiveConflictWeightMultiplier")))
        {
            options->minimumAdaptiveConflictWeightMultiplier = 0.02f;
        }
        if (!settings.contains(QStringLiteral("tsdfWeakEvidenceSurfaceOnlyIntegration")))
        {
            options->enableWeakEvidenceSurfaceOnlyIntegration = true;
        }
        if (!settings.contains(QStringLiteral("tsdfGeometrySingleViewNeighborhoodGuard")))
        {
            options->enableGeometrySingleViewNeighborhoodGuard = true;
        }
        if (!settings.contains(QStringLiteral("tsdfMinimumSupportMaskFreeSpaceViews")))
        {
            options->minimumSupportMaskFreeSpaceViews = 5;
        }
        if (!settings.contains(QStringLiteral("tsdfRobustFrameQualityWeighting")))
        {
            options->enableRobustFrameQualityWeighting = true;
        }
        if (!settings.contains(QStringLiteral("tsdfRobustFrameQualityRejection")))
        {
            options->enableRobustFrameQualityRejection = false;
        }
        if (!settings.contains(QStringLiteral("tsdfOrbitalFrameCoverageProtection")))
        {
            options->enableOrbitalFrameCoverageProtection = true;
        }
        if (!settings.contains(QStringLiteral("tsdfPerFrameCoverageBounds")))
        {
            // A point on an orbital object's silhouette can be visible in only a
            // few adjacent camera sectors. Global point quantiles clip those valid
            // extrema, so preserve robust bounds from individual frames first.
            options->preservePerFrameCoverageBounds = true;
        }
        if (!settings.contains(QStringLiteral("tsdfMinimumComponentFaceRatio")))
        {
            // An orbital-object run reconstructs one physical body.  Small nested
            // shells above the generic 2.5% dust threshold are depth ghosts, not a
            // legitimate second object.  Keep genuinely large competing surfaces
            // for the final single-component quality gate to reject explicitly.
            options->minimumComponentFaceRatio = std::max(options->minimumComponentFaceRatio, 0.05f);
        }
        if (!settings.contains(QStringLiteral("tsdfTopologyQualityMaximumClosedGenus")))
        {
            // A single small-body target is expected to be sphere-like.  Handles
            // in an automatically completed carrier are unsupported topology,
            // not recoverable surface detail.
            options->topologyQualityMaximumClosedGenus = 0.0f;
        }
        if (!settings.contains(QStringLiteral("tsdfTopologyQualityMaximumTopologicalComplexity")))
        {
            options->topologyQualityMaximumTopologicalComplexity = 0;
        }
        if (!settings.contains(QStringLiteral("tsdfOrbitalGapBoundaryRecovery")))
        {
            options->enableOrbitalGapBoundaryRecovery = true;
        }
        if (!settings.contains(QStringLiteral("tsdfOrbitalGapAdaptiveTruncation")))
        {
            options->enableOrbitalGapAdaptiveTruncation = true;
        }
        if (!settings.contains(QStringLiteral("tsdfOrbitalGapAdaptiveTruncationScale")))
        {
            options->orbitalGapAdaptiveTruncationScale = 1.50f;
        }
        if (!settings.contains(QStringLiteral("tsdfOrbitalGapAdaptiveMaximumTruncationVoxels")))
        {
            options->orbitalGapAdaptiveMaximumTruncationVoxels = 16.0f;
        }
        if (!settings.contains(QStringLiteral("tsdfSurfaceEvidenceFreeSpaceVeto")))
        {
            options->enableSurfaceEvidenceFreeSpaceVeto = true;
        }
        if (!settings.contains(QStringLiteral("tsdfDepthCompletenessDiagnostics")))
        {
            options->enableDepthCompletenessDiagnostics = true;
        }
        if (!settings.contains(QStringLiteral("tsdfEnforceDepthCompletenessGate")))
        {
            options->enforceDepthCompletenessGate = true;
        }
        if (!settings.contains(QStringLiteral("tsdfDepthCompletenessToleranceVoxels")))
        {
            // Orbital MVS depth is fused across a complete viewing ring and then
            // simplified/denoised.  Judge observation support within one TSDF
            // truncation band instead of the generic four-voxel tolerance.
            options->depthCompletenessToleranceVoxels = std::max(6.0f, options->truncationVoxels);
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyCompletion")))
        {
            options->enableVisibilityOccupancyCompletion = true;
        }
        const bool high_detail_model =
            options->resolution >= 384 && options->simplifyTargetFaces > 0 &&
            (options->simplifyTargetFaces <= 120000 ||
             (options->enableTopologySafeSimplification && options->simplifyTargetFaces <= 240000));
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyResolution")))
        {
            int effective_tsdf_resolution = options->resolution;
            if (high_detail_model && settings.value(QStringLiteral("tsdfOrbitalAdaptiveResolution")).toBool(true) &&
                maximumReliableResolution > 0)
            {
                effective_tsdf_resolution = std::min(effective_tsdf_resolution, maximumReliableResolution);
            }
            // With interpolation disabled the occupancy field is the only
            // topology carrier, and its sign is authoritative even at directly
            // observed high-resolution TSDF samples. A 72-cell carrier visibly
            // quantizes thin measured silhouettes at a 320+ TSDF resolution.
            // Keep the cheaper legacy default elsewhere and preserve an explicit
            // user value in every mode.
            options->visibilityOccupancyResolution =
                interpolationIsDisabled(settings) && effective_tsdf_resolution >= 320 ? 96 : 72;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyCellBoundaryExtraction")))
        {
            // Keep the low-resolution visibility graph cut as a topology/sign
            // prior, but extract the surface from the completed high-resolution
            // TSDF.  Emitting the occupancy-cell boundary here quantizes the
            // geometry to the carrier grid and discards the sub-voxel zero
            // crossing retained by MC33 (the same distinction made by Open3D's
            // TSDF integration/extraction path).
            options->visibilityOccupancyCellBoundaryExtraction = false;
        }
        if (!settings.contains(QStringLiteral("tsdfMc33RequireSupportedSignChange")))
        {
            // Visibility completion deliberately supplies a topology-consistent
            // sign for samples without direct depth support. Requiring both edge
            // endpoints to be observed would cut those recovered regions away.
            options->mc33RequireSupportedSignChange = false;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyClosingIterations")))
        {
            options->visibilityOccupancyClosingIterations = 1;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyMaximumHandleRepairPasses")))
        {
            options->visibilityOccupancyMaximumHandleRepairPasses = 8;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyMaximumHandleRepairAcceptedCandidateCount")))
        {
            options->visibilityOccupancyMaximumHandleRepairAcceptedCandidateCount = 128;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyMaximumHandleRepairCandidateSampleCount")))
        {
            options->visibilityOccupancyMaximumHandleRepairCandidateSampleCount = 4096;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyMaximumHandleRepairSubsetSampleCount")))
        {
            options->visibilityOccupancyMaximumHandleRepairSubsetSampleCount = 256;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyMaximumHandleRepairSubsetSeedCount")))
        {
            options->visibilityOccupancyMaximumHandleRepairSubsetSeedCount = 4096;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyTopologyLockedResidualBlend")))
        {
            options->visibilityOccupancyTopologyLockedResidualBlend = true;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyObservedBand")))
        {
            options->visibilityOccupancyObservedBand = 1.0f;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyCarrierBand")))
        {
            options->visibilityOccupancyCarrierBand = 1.0f;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyMaximumResidual")))
        {
            options->visibilityOccupancyMaximumResidual = 1.0f;
        }
        if (!settings.contains(QStringLiteral("tsdfVisibilityOccupancyDetailBlend")))
        {
            options->visibilityOccupancyDetailBlend = 1.0f;
        }
        if (!settings.contains(QStringLiteral("tsdfTriangleQualityOptimizationMaximumPasses")))
        {
            options->triangleQualityOptimizationMaximumPasses = 12;
        }
        if (!settings.contains(QStringLiteral("tsdfTriangleQualityTangentialRelaxationPasses")))
        {
            options->triangleQualityTangentialRelaxationPasses = 8;
        }
        if (!settings.contains(QStringLiteral("tsdfTriangleQualityIsotropicRemeshingPasses")))
        {
            options->triangleQualityIsotropicRemeshingPasses = 4;
        }
        const bool no_depth_interpolation_topology_carrier =
            interpolationIsDisabled(settings) &&
            settings.value(QStringLiteral("tsdfVisibilityOccupancyCompletion")).toBool(true);
        if (no_depth_interpolation_topology_carrier)
        {
            options->enableVisibilityOccupancyCompletion = true;
            options->visibilityOccupancySilhouetteFullPriorCapacity = qBound(
                0, settings.value(QStringLiteral("tsdfVisibilityOccupancySilhouetteFullPriorCapacity")).toInt(2), 1000);
            if (options->simplifyTargetFaces > 0 &&
                !settings.contains(QStringLiteral("tsdfTriangleQualityOptimization")))
            {
                options->enableTriangleQualityOptimization = true;
            }
            if (options->enableTriangleQualityOptimization &&
                !settings.contains(QStringLiteral("tsdfTriangleQualityIsotropicRemeshing")))
            {
                options->enableTriangleQualityIsotropicRemeshing = true;
            }
        }
        if (!high_detail_model)
        {
            // Low-resolution and large-face orbital presets return before the
            // detail-only defaults below. Re-apply the no-depth-interpolation
            // contract without disabling the visibility-constrained carrier.
            enforceNoDepthInterpolationPolicy(settings, options, no_depth_interpolation_topology_carrier);
            return;
        }
        if (!settings.contains(QStringLiteral("tsdfInverseDepthSpreadWeighting")))
        {
            options->enableInverseDepthSpreadWeighting = true;
        }
        if (!settings.contains(QStringLiteral("tsdfInverseDepthSpreadWeightKnee")))
        {
            options->inverseDepthSpreadWeightKnee = 0.005f;
        }
        if (!settings.contains(QStringLiteral("tsdfInverseDepthSpreadWeightZero")))
        {
            options->inverseDepthSpreadWeightZero = 0.015f;
        }
        if (!settings.contains(QStringLiteral("tsdfMinimumInverseDepthSpreadWeightMultiplier")))
        {
            options->minimumInverseDepthSpreadWeightMultiplier = 0.05f;
        }
        if (!settings.contains(QStringLiteral("tsdfInverseDepthSpreadSupportWeightDecoupling")))
        {
            options->enableInverseDepthSpreadSupportWeightDecoupling = true;
        }
        if (!settings.contains(QStringLiteral("tsdfInverseDepthSpreadSupportWeightExponent")))
        {
            options->inverseDepthSpreadSupportWeightExponent = 0.25f;
        }
        if (settings.value(QStringLiteral("tsdfOrbitalAdaptiveResolution")).toBool(true) &&
            maximumReliableResolution > 0)
        {
            options->resolution = std::min(options->resolution, maximumReliableResolution);
            // The requested face count is an upper bound, while the extracted
            // surface already becomes naturally smaller at a lower voxel
            // resolution. Scaling the face budget by the resolution ratio applies
            // the same detail reduction twice and can turn a 200k-face orbital
            // surface into an unexpected 60k-face result.
        }
        if (!settings.contains(QStringLiteral("tsdfTruncationVoxels")))
        {
            options->truncationVoxels = 7.5f;
        }
        if (!settings.contains(QStringLiteral("tsdfSurfaceSupportBandVoxels")))
        {
            options->surfaceSupportBandVoxels = 7.5f;
        }
        if (!settings.contains(QStringLiteral("tsdfUncertaintyAdaptiveTruncation")))
        {
            options->enableUncertaintyAdaptiveTruncation = true;
        }
        if (!settings.contains(QStringLiteral("tsdfUncertaintyAdaptiveScale")))
        {
            options->uncertaintyAdaptiveScale = 0.40f;
        }
        if (!settings.contains(QStringLiteral("tsdfUncertaintyAdaptiveActivationRatio")))
        {
            options->uncertaintyAdaptiveActivationRatio = 1.20f;
        }
        if (!settings.contains(QStringLiteral("tsdfUncertaintyAdaptiveMaximumTruncationVoxels")))
        {
            options->uncertaintyAdaptiveMaximumTruncationVoxels = 12.0f;
        }
        if (!settings.contains(QStringLiteral("tsdfAllowInvalidNearestPixelRecovery")))
        {
            options->allowInvalidNearestPixelRecovery = false;
        }
        if (!settings.contains(QStringLiteral("tsdfMaximumInvalidNearestPixelRecoveryInverseDepthSpread")))
        {
            options->maximumInvalidNearestPixelRecoveryInverseDepthSpread = 0.01f;
        }
        if (!settings.contains(QStringLiteral("tsdfGeometryZeroCrossingRecovery")))
        {
            options->enableGeometryZeroCrossingRecovery = false;
        }
        if (!settings.contains(QStringLiteral("tsdfCrossViewAnchoredSurfaceRecovery")))
        {
            options->enableCrossViewAnchoredSurfaceRecovery = false;
        }
        if (!settings.contains(QStringLiteral("tsdfCrossViewAnchoredMinimumObservationWeight")))
        {
            options->crossViewAnchoredMinimumObservationWeight = 0.25f;
        }
        if (!settings.contains(QStringLiteral("tsdfCrossViewAnchoredMinimumSupportedCorners")))
        {
            options->crossViewAnchoredMinimumSupportedCorners = 2;
        }
        if (!settings.contains(QStringLiteral("tsdfCrossViewAnchoredMinimumCellVotes")))
        {
            options->crossViewAnchoredMinimumCellVotes = 1;
        }
        if (!settings.contains(QStringLiteral("tsdfCrossViewAnchoredGrowthPasses")))
        {
            options->crossViewAnchoredGrowthPasses = 2;
        }
        if (!settings.contains(QStringLiteral("tsdfGeometryZeroCrossingCellSheets")))
        {
            options->enableGeometryZeroCrossingCellSheets = false;
        }
        if (!settings.contains(QStringLiteral("tsdfContourBandZeroCrossingSupport")))
        {
            options->enableContourBandZeroCrossingSupport = false;
        }
        if (!settings.contains(QStringLiteral("tsdfAdaptiveTgvRegularization")))
        {
            options->enableAdaptiveTgvRegularization = true;
        }
        if (!settings.contains(QStringLiteral("tsdfAdaptiveTgvRecoverUnsupportedSamples")))
        {
            options->adaptiveTgvRecoverUnsupportedSamples = false;
        }
        if (!settings.contains(QStringLiteral("tsdfSurfacePatchSupport")))
        {
            options->enableSurfacePatchSupport = false;
        }
        if (!settings.contains(QStringLiteral("tsdfGeometryVerifiedBoundaryRecovery")))
        {
            options->enableGeometryVerifiedBoundaryRecovery = false;
        }
        if (!settings.contains(QStringLiteral("tsdfAllowGeometryVerifiedSingleObservation")))
        {
            options->allowGeometryVerifiedSingleObservation = false;
        }
        if (!settings.contains(QStringLiteral("tsdfMinimumGeometrySupportCount")))
        {
            options->minimumGeometrySupportCount = 4;
        }
        if (!settings.contains(QStringLiteral("tsdfDiscontinuityAwareSampling")))
        {
            options->enableDiscontinuityAwareSampling = false;
        }
        if (!settings.contains(QStringLiteral("tsdfCrossViewConsensusDepth")))
        {
            options->enableCrossViewConsensusDepth = true;
        }
        if (!settings.contains(QStringLiteral("tsdfMaximumCrossViewConsensusInverseDepthSpread")))
        {
            options->maximumCrossViewConsensusInverseDepthSpread = 0.008f;
        }
        if (!settings.contains(QStringLiteral("tsdfMaximumObservationInverseDepthSpread")))
        {
            options->maximumObservationInverseDepthSpread = 0.015f;
        }
        if (!settings.contains(QStringLiteral("tsdfDepthValidBoundaryErosionPixels")))
        {
            options->depthValidBoundaryErosionPixels = 1;
        }
        if (!settings.contains(QStringLiteral("tsdfBoundarySmoothingIterations")))
        {
            options->boundarySmoothingIterations = 1;
        }
        if (!settings.contains(QStringLiteral("tsdfTrimWeakBoundaryTips")))
        {
            options->trimWeakBoundaryTips = false;
        }
        if (options->fillSmallBoundaryHoles)
        {
            if (!settings.contains(QStringLiteral("tsdfSilhouetteAwareFinalHoleFill")))
            {
                options->enableSilhouetteAwareFinalHoleFill = true;
            }
            if (!settings.contains(QStringLiteral("tsdfVisibilityConstrainedFinalHoleFill")))
            {
                options->enableVisibilityConstrainedFinalHoleFill = true;
            }
            if (!settings.contains(QStringLiteral("tsdfFinalHoleFillMaximumBoundaryEdges")))
            {
                options->finalHoleFillMaximumBoundaryEdges = 24;
            }
            if (!settings.contains(QStringLiteral("tsdfFinalHoleFillMaximumDiameterVoxels")))
            {
                options->finalHoleFillMaximumDiameterVoxels = 4.0f;
            }
            if (!settings.contains(QStringLiteral("tsdfFinalHoleFillMaximumFaceGrowthRatio")))
            {
                options->finalHoleFillMaximumFaceGrowthRatio = 0.03f;
            }
            if (!settings.contains(QStringLiteral("tsdfVisibilityHoleFillMinimumSupportingViews")))
            {
                options->visibilityHoleFillMinimumSupportingViews = 2;
            }
            if (!settings.contains(QStringLiteral("tsdfVisibilityHoleFillMaximumConflictViews")))
            {
                options->visibilityHoleFillMaximumConflictViews = 0;
            }
        }
        enforceNoDepthInterpolationPolicy(settings, options, no_depth_interpolation_topology_carrier);
    }

    bool visibilityOccupancyDepthRefinementEnabled(const QJsonObject& settings, bool orbitalWorkspace)
    {
        const bool configured =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthRefinement")).toBool(orbitalWorkspace);
        if (!configured)
        {
            return false;
        }
        return !interpolationIsDisabled(settings) || orbitalWorkspace;
    }

    bool shouldUseOrbitalVisualHullCompletion(bool orbitalWorkspace,
                                              bool enabled,
                                              bool observationOnlySurface,
                                              double aggregateProjectionRecall,
                                              int boundaryEdgeCount,
                                              int faceCount)
    {
        if (!orbitalWorkspace || !enabled || observationOnlySurface || faceCount <= 0)
        {
            return false;
        }

        const double boundary_ratio =
            static_cast<double>(std::max(0, boundaryEdgeCount)) / static_cast<double>(std::max(1, faceCount));
        const bool projection_recall_failed = aggregateProjectionRecall > 0.0 && aggregateProjectionRecall < 0.82;
        const bool unresolved_boundary_loops = boundaryEdgeCount > 128;
        return projection_recall_failed || unresolved_boundary_loops || boundary_ratio > 0.025;
    }

    int selectOrbitalTsdfRetryResolution(int currentResolution,
                                         bool completenessAvailable,
                                         double medianRecall,
                                         double p10Recall,
                                         double minimumMedianRecall,
                                         double minimumP10Recall)
    {
        if (!completenessAvailable || currentResolution <= 176 || !std::isfinite(medianRecall) ||
            !std::isfinite(p10Recall) || !std::isfinite(minimumMedianRecall) || !std::isfinite(minimumP10Recall) ||
            medianRecall >= minimumMedianRecall && p10Recall >= minimumP10Recall)
        {
            return 0;
        }

        return currentResolution > 256 ? 256 : 176;
    }
} // namespace xjw::mesh::workflow
