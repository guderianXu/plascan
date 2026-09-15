#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow::workflow_detail
{
    using namespace workflow_detail;

    bool interpolationIsDisabled(const QJsonObject& settings)
    {
        return settings.value(QStringLiteral("interpolation")).toString(QStringLiteral("enabled")) ==
               QStringLiteral("disabled");
    }

    void enforceNoDepthInterpolationPolicy(const QJsonObject& settings,
                                           xjw::mesh::DepthTsdfOptions* options,
                                           bool preserveVisibilityTopologyCarrier)
    {
        if (!options || !interpolationIsDisabled(settings))
        {
            return;
        }

        // “插值禁用” applies to depth samples, not necessarily to final mesh
        // topology. Orbital models may retain a visibility/silhouette constrained
        // occupancy carrier while still rejecting every synthesized depth
        // observation and all depth-driven unsupported-surface recovery below.
        // Generic scenes keep the stricter measured-sign-change behavior.
        // Capping a bounded mesh boundary adds faces only; it does not create,
        // overwrite, or extrapolate a depth-map sample. Keep this topology-only
        // repair for orbital visibility carriers unless the user explicitly
        // disables hole filling. Generic measured-only surfaces remain open.
        options->fillSmallBoundaryHoles =
            preserveVisibilityTopologyCarrier && settings.value(QStringLiteral("holeFill")).toBool(true);
        options->enableSilhouetteAwareFinalHoleFill = false;
        options->enableVisibilityConstrainedFinalHoleFill = false;
        options->enableTinyBoundaryLoopCollapse = false;
        options->enableVisualHullSignedDistanceCompletion = false;
        if (!preserveVisibilityTopologyCarrier)
        {
            options->enableVisibilityOccupancyCompletion = false;
            options->visibilityOccupancySilhouetteFullPriorCapacity = 0;
        }
        options->visibilityOccupancyAlignCarrierGrid = false;
        options->visibilityOccupancyNativeCarrierExtraction = false;
        options->visibilityOccupancyCellBoundaryExtraction = false;
        options->allowInvalidNearestPixelRecovery = false;
        options->excludeAnchoredInterpolationObservations = true;
        options->enableAuxiliarySurfaceOnlyIntegration = false;
        // Revision-37 validation frames may still contribute their measured,
        // multi-view core when they are the minimum bridge needed to reconnect the
        // exact geometry-source graph. The builder fails closed for older or
        // incomplete manifests and applies an additional per-pixel evidence gate.
        options->enableAuxiliaryBridgeOnlyIntegration = true;
        options->enableGeometryZeroCrossingRecovery = false;
        options->enableCrossViewAnchoredSurfaceRecovery = false;
        options->enableGeometryZeroCrossingCellSheets = false;
        options->enableContourBandZeroCrossingSupport = false;
        options->enableSurfacePatchSupport = false;
        options->adaptiveTgvRecoverUnsupportedSamples = false;
        options->implicitRegularizationRecoverAxialGaps = false;
        options->mc33RequireSupportedSignChange = !preserveVisibilityTopologyCarrier;
    }

    xjw::mesh::DepthTsdfOptions makeDepthTsdfOptions(const QJsonObject& settings, int requested_resolution)
    {
        xjw::mesh::DepthTsdfOptions options;
        options.resolution = settings.contains(QStringLiteral("meshResolution")) ? meshResolutionFromSettings(settings)
                                                                                 : requested_resolution;
        options.calculateVertexColors = settings.value(QStringLiteral("calculateVertexColors")).toBool(true);
        options.useEvidenceAwareBounds =
            settings.value(QStringLiteral("tsdfEvidenceAwareBounds")).toBool(options.useEvidenceAwareBounds);
        options.preservePerFrameCoverageBounds =
            settings.value(QStringLiteral("tsdfPerFrameCoverageBounds")).toBool(options.preservePerFrameCoverageBounds);
        options.compensateColorExposure = settings.value(QStringLiteral("tsdfCompensateColorExposure")).toBool(false);
        options.coherentFacePrimaryViewColors =
            settings.value(QStringLiteral("tsdfCoherentFacePrimaryViewColors")).toBool(false);
        options.simplifyTargetFaces = qBound(0,
                                             settings.value(QStringLiteral("simplifyTargetFaces"))
                                                 .toInt(settings.value(QStringLiteral("targetFaces")).toInt(0)),
                                             2000000);
        options.enableQuadricSimplification = settings.contains(QStringLiteral("tsdfQuadricSimplification"))
                                                  ? settings.value(QStringLiteral("tsdfQuadricSimplification")).toBool()
                                                  : options.simplifyTargetFaces > 0;
        const bool arbitrary_surface =
            settings.value(QStringLiteral("surface_type")).toString(QStringLiteral("arbitrary_3d")) ==
            QStringLiteral("arbitrary_3d");
        const bool automatic_topology_safe_simplification = options.resolution >= 320 &&
                                                            options.simplifyTargetFaces > 0 &&
                                                            options.simplifyTargetFaces <= 240000 && arbitrary_surface;
        options.enableTopologySafeSimplification = settings.value(QStringLiteral("tsdfTopologySafeSimplification"))
                                                       .toBool(automatic_topology_safe_simplification);
        options.topologySafeMaximumNormalDeviationDegrees = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfTopologySafeMaximumNormalDeviationDegrees")).toDouble(180.0)),
            1.0f,
            180.0f);
        options.topologySafeMaximumNormalFlippingDegrees = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfTopologySafeMaximumNormalFlippingDegrees")).toDouble(75.0)),
            1.0f,
            180.0f);
        options.topologySafeSmoothingIterations =
            qBound(0,
                   settings.value(QStringLiteral("tsdfTopologySafeSmoothingIterations"))
                       .toInt(options.enableTopologySafeSimplification ? 12 : 2),
                   20);
        options.topologySafeSmoothingMaximumDisplacementVoxels = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfTopologySafeSmoothingMaximumDisplacementVoxels"))
                                   .toDouble(options.enableTopologySafeSimplification ? 1.60 : 0.40)),
            0.0f,
            2.0f);
        options.topologySafeSmoothingFeatureAngleDegrees =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfTopologySafeSmoothingFeatureAngleDegrees"))
                                              .toDouble(options.enableTopologySafeSimplification ? 175.0 : 120.0)),
                       1.0f,
                       180.0f);
        options.enableVoxelFallbackSimplification =
            settings.value(QStringLiteral("tsdfVoxelFallbackSimplification")).toBool(true);
        const bool automatic_voxel_fallback_qem_polish =
            options.resolution >= 384 && options.simplifyTargetFaces > 0 && options.simplifyTargetFaces <= 240000 &&
            settings.value(QStringLiteral("surface_type")).toString(QStringLiteral("arbitrary_3d")) ==
                QStringLiteral("arbitrary_3d");
        options.enableVoxelFallbackQemPolish =
            settings.value(QStringLiteral("tsdfVoxelFallbackQemPolish")).toBool(automatic_voxel_fallback_qem_polish);
        options.voxelFallbackMinimumProtectedBoundaryVertices = qBound(
            1, settings.value(QStringLiteral("tsdfVoxelFallbackMinimumProtectedBoundaryVertices")).toInt(1), 256);
        options.voxelFallbackMaximumCollapsibleBoundaryDiameterVoxels =
            std::clamp(static_cast<float>(
                           settings.value(QStringLiteral("tsdfVoxelFallbackMaximumCollapsibleBoundaryDiameterVoxels"))
                               .toDouble(0.0)),
                       0.0f,
                       16.0f);
        options.voxelFallbackMaximumNormalClusterAngleDegrees = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfVoxelFallbackMaximumNormalClusterAngleDegrees")).toDouble(180.0)),
            5.0f,
            180.0f);
        options.voxelFallbackInitialClusterFactor = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfVoxelFallbackInitialClusterFactor")).toDouble(1.0)),
            1.0f,
            4.0f);
        options.enableVoxelFallbackMultiViewSilhouetteProtection =
            settings.value(QStringLiteral("tsdfVoxelFallbackMultiViewSilhouetteProtection")).toBool(true);
        options.voxelFallbackMinimumSilhouetteViews =
            qBound(1, settings.value(QStringLiteral("tsdfVoxelFallbackMinimumSilhouetteViews")).toInt(2), 8);
        options.voxelFallbackSilhouetteBandPixels =
            qBound(1, settings.value(QStringLiteral("tsdfVoxelFallbackSilhouetteBandPixels")).toInt(2), 8);
        options.voxelFallbackSilhouetteDepthToleranceVoxels = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfVoxelFallbackSilhouetteDepthToleranceVoxels")).toDouble(8.0)),
            1.0f,
            32.0f);
        options.voxelFallbackMaximumSliverRatio = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfVoxelFallbackMaximumSliverRatio")).toDouble(0.08)),
            0.0f,
            0.50f);
        const bool automatic_triangle_quality_optimization =
            options.resolution >= 384 && options.simplifyTargetFaces > 0;
        options.enableTriangleQualityOptimization = settings.value(QStringLiteral("tsdfTriangleQualityOptimization"))
                                                        .toBool(automatic_triangle_quality_optimization);
        options.triangleQualityOptimizationMaximumPasses =
            qBound(1,
                   settings.value(QStringLiteral("tsdfTriangleQualityOptimizationMaximumPasses"))
                       .toInt(options.triangleQualityOptimizationMaximumPasses),
                   12);
        options.triangleQualityMinimumAspectImprovementRatio = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfTriangleQualityMinimumAspectImprovementRatio"))
                                   .toDouble(options.triangleQualityMinimumAspectImprovementRatio)),
            0.0f,
            0.25f);
        options.triangleQualityMaximumFeatureAngleDegrees = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfTriangleQualityMaximumFeatureAngleDegrees"))
                                   .toDouble(options.triangleQualityMaximumFeatureAngleDegrees)),
            5.0f,
            90.0f);
        options.triangleQualityMaximumNormalDeviationDegrees = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfTriangleQualityMaximumNormalDeviationDegrees"))
                                   .toDouble(options.triangleQualityMaximumNormalDeviationDegrees)),
            5.0f,
            90.0f);
        options.enableTriangleQualityTangentialRelaxation =
            settings.value(QStringLiteral("tsdfTriangleQualityTangentialRelaxation"))
                .toBool(options.enableTriangleQualityTangentialRelaxation);
        options.triangleQualityTangentialRelaxationPasses =
            qBound(0,
                   settings.value(QStringLiteral("tsdfTriangleQualityTangentialRelaxationPasses"))
                       .toInt(options.triangleQualityTangentialRelaxationPasses),
                   8);
        options.triangleQualityTangentialRelaxationLambda = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfTriangleQualityTangentialRelaxationLambda"))
                                   .toDouble(options.triangleQualityTangentialRelaxationLambda)),
            0.0f,
            1.0f);
        options.triangleQualityTangentialMaximumDisplacementEdgeRatio =
            std::clamp(static_cast<float>(
                           settings.value(QStringLiteral("tsdfTriangleQualityTangentialMaximumDisplacementEdgeRatio"))
                               .toDouble(options.triangleQualityTangentialMaximumDisplacementEdgeRatio)),
                       0.0f,
                       0.50f);
        const bool automatic_isotropic_remeshing =
            automatic_triangle_quality_optimization && options.simplifyTargetFaces <= 120000;
        options.enableTriangleQualityIsotropicRemeshing =
            settings.value(QStringLiteral("tsdfTriangleQualityIsotropicRemeshing"))
                .toBool(automatic_isotropic_remeshing);
        options.triangleQualityIsotropicRemeshingPasses =
            qBound(0,
                   settings.value(QStringLiteral("tsdfTriangleQualityIsotropicRemeshingPasses"))
                       .toInt(options.triangleQualityIsotropicRemeshingPasses),
                   4);
        options.triangleQualityIsotropicShortEdgeRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfTriangleQualityIsotropicShortEdgeRatio"))
                                              .toDouble(options.triangleQualityIsotropicShortEdgeRatio)),
                       0.05f,
                       0.45f);
        options.triangleQualityIsotropicLongEdgeRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfTriangleQualityIsotropicLongEdgeRatio"))
                                              .toDouble(options.triangleQualityIsotropicLongEdgeRatio)),
                       1.25f,
                       4.0f);
        options.triangleQualityIsotropicMaximumFaceGrowthRatio = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfTriangleQualityIsotropicMaximumFaceGrowthRatio"))
                                   .toDouble(options.triangleQualityIsotropicMaximumFaceGrowthRatio)),
            0.0f,
            0.20f);
        options.topologyQualityMaximumBoundaryEdgeRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfTopologyQualityMaximumBoundaryEdgeRatio"))
                                              .toDouble(options.topologyQualityMaximumBoundaryEdgeRatio)),
                       0.0f,
                       1.0f);
        options.topologyQualityMaximumHighAspectFaceRatio = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfTopologyQualityMaximumHighAspectFaceRatio"))
                                   .toDouble(options.topologyQualityMaximumHighAspectFaceRatio)),
            0.0f,
            1.0f);
        options.topologyQualityMaximumExtremeAspectFaceRatio = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfTopologyQualityMaximumExtremeAspectFaceRatio"))
                                   .toDouble(options.topologyQualityMaximumExtremeAspectFaceRatio)),
            0.0f,
            1.0f);
        options.topologyQualityMaximumClosedGenus =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfTopologyQualityMaximumClosedGenus"))
                                              .toDouble(options.topologyQualityMaximumClosedGenus)),
                       0.0f,
                       100000.0f);
        options.topologyQualityMaximumTopologicalComplexity =
            qBound(0,
                   settings.value(QStringLiteral("tsdfTopologyQualityMaximumTopologicalComplexity"))
                       .toInt(options.topologyQualityMaximumTopologicalComplexity),
                   200000);
        options.enableDepthCompletenessDiagnostics =
            settings.value(QStringLiteral("tsdfDepthCompletenessDiagnostics")).toBool(false);
        options.enforceDepthCompletenessGate =
            settings.value(QStringLiteral("tsdfEnforceDepthCompletenessGate")).toBool(false);
        options.depthCompletenessMaximumSamplesPerFrame =
            qBound(500,
                   settings.value(QStringLiteral("tsdfDepthCompletenessMaximumSamplesPerFrame"))
                       .toInt(options.depthCompletenessMaximumSamplesPerFrame),
                   50000);
        options.depthCompletenessToleranceVoxels =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfDepthCompletenessToleranceVoxels"))
                                              .toDouble(options.depthCompletenessToleranceVoxels)),
                       1.0f,
                       16.0f);
        options.minimumDepthCompletenessP10Recall =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMinimumDepthCompletenessP10Recall"))
                                              .toDouble(options.minimumDepthCompletenessP10Recall)),
                       0.0f,
                       1.0f);
        options.minimumDepthCompletenessMedianRecall =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMinimumDepthCompletenessMedianRecall"))
                                              .toDouble(options.minimumDepthCompletenessMedianRecall)),
                       0.0f,
                       1.0f);
        options.maximumSimplificationBoundaryEdgeGrowthRatio = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfMaximumSimplificationBoundaryEdgeGrowthRatio"))
                                   .toDouble(options.maximumSimplificationBoundaryEdgeGrowthRatio)),
            0.0f,
            1.0f);
        options.simplificationMaximumPasses = qBound(1,
                                                     settings.value(QStringLiteral("tsdfSimplificationMaximumPasses"))
                                                         .toInt(options.simplificationMaximumPasses),
                                                     96);
        options.simplificationFeatureAngleDegrees =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfSimplificationFeatureAngleDegrees"))
                                              .toDouble(options.simplificationFeatureAngleDegrees)),
                       5.0f,
                       85.0f);
        options.simplificationMaximumNormalDeviationDegrees = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfSimplificationMaximumNormalDeviationDegrees"))
                                   .toDouble(options.simplificationMaximumNormalDeviationDegrees)),
            5.0f,
            85.0f);
        options.simplificationMinimumSharpEdgeEndpointDegree =
            qBound(1,
                   settings.value(QStringLiteral("tsdfSimplificationMinimumSharpEdgeEndpointDegree"))
                       .toInt(options.simplificationMinimumSharpEdgeEndpointDegree),
                   4);
        options.simplifySimpleOpenBoundaries = settings.value(QStringLiteral("tsdfSimplifySimpleOpenBoundaries"))
                                                   .toBool(options.simplifySimpleOpenBoundaries);
        options.minimumInputFrames = 3;
        const int automatic_camera_support = 2;
        options.minimumDistinctCameraSupport = qBound(
            1, settings.value(QStringLiteral("tsdfMinimumDistinctCameraSupport")).toInt(automatic_camera_support), 16);
        options.minimumComponentFaces =
            qBound(0, settings.value(QStringLiteral("minFaces")).toInt(options.minimumComponentFaces), 100000);
        options.minimumComponentFaceRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMinimumComponentFaceRatio"))
                                              .toDouble(options.minimumComponentFaceRatio)),
                       0.0f,
                       1.0f);
        options.workerCount = qBound(0, settings.value(QStringLiteral("threads")).toInt(options.workerCount), 128);

        const QString filtering = settings.value(QStringLiteral("depthFiltering")).toString(QStringLiteral("moderate"));
        if (filtering == QStringLiteral("disabled"))
        {
            options.minimumConfidence = 0.10f;
            options.minimumSingleObservationWeight = 0.55f;
        }
        else if (filtering == QStringLiteral("mild"))
        {
            options.minimumConfidence = 0.20f;
            options.minimumSingleObservationWeight = 0.60f;
        }
        else if (filtering == QStringLiteral("aggressive"))
        {
            options.minimumConfidence = 0.40f;
            options.minimumSingleObservationWeight = 0.80f;
        }
        else
        {
            options.minimumConfidence = 0.25f;
            options.minimumSingleObservationWeight = 0.70f;
        }

        options.truncationVoxels =
            std::clamp(static_cast<float>(
                           settings.value(QStringLiteral("tsdfTruncationVoxels")).toDouble(options.truncationVoxels)),
                       1.0f,
                       12.0f);
        options.surfaceSupportBandVoxels =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfSurfaceSupportBandVoxels"))
                                              .toDouble(options.surfaceSupportBandVoxels)),
                       0.0f,
                       12.0f);
        options.enableUncertaintyAdaptiveTruncation =
            settings.value(QStringLiteral("tsdfUncertaintyAdaptiveTruncation")).toBool(false);
        options.uncertaintyAdaptiveScale =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfUncertaintyAdaptiveScale"))
                                              .toDouble(options.uncertaintyAdaptiveScale)),
                       0.0f,
                       2.0f);
        options.uncertaintyAdaptiveActivationRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfUncertaintyAdaptiveActivationRatio"))
                                              .toDouble(options.uncertaintyAdaptiveActivationRatio)),
                       1.0f,
                       4.0f);
        options.uncertaintyAdaptiveMaximumTruncationVoxels = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfUncertaintyAdaptiveMaximumTruncationVoxels"))
                                   .toDouble(options.uncertaintyAdaptiveMaximumTruncationVoxels)),
            1.0f,
            16.0f);
        options.uncertaintyAdaptiveMaximumSamplesPerFrame =
            qBound(256,
                   settings.value(QStringLiteral("tsdfUncertaintyAdaptiveMaximumSamplesPerFrame"))
                       .toInt(options.uncertaintyAdaptiveMaximumSamplesPerFrame),
                   100000);
        options.uncertaintyAdaptiveMinimumSampleCount =
            qBound(64,
                   settings.value(QStringLiteral("tsdfUncertaintyAdaptiveMinimumSampleCount"))
                       .toInt(options.uncertaintyAdaptiveMinimumSampleCount),
                   100000);
        options.minimumVoxelWeight = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfMinimumVoxelWeight")).toDouble(options.minimumVoxelWeight)),
            0.05f,
            20.0f);
        options.enablePixelEvidenceWeighting =
            settings.value(QStringLiteral("tsdfPixelEvidenceWeighting")).toBool(false);
        options.unconfirmedNativeObservationMultiplier =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfUnconfirmedNativeObservationMultiplier"))
                                              .toDouble(options.unconfirmedNativeObservationMultiplier)),
                       0.05f,
                       1.0f);
        options.weakNativeObservationMultiplier =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfWeakNativeObservationMultiplier"))
                                              .toDouble(options.weakNativeObservationMultiplier)),
                       0.05f,
                       1.0f);
        options.repairedObservationMultiplier =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfRepairedObservationMultiplier"))
                                              .toDouble(options.repairedObservationMultiplier)),
                       0.05f,
                       1.0f);
        options.adaptiveGeometryMinimumObservationMultiplier = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveGeometryMinimumObservationMultiplier"))
                                   .toDouble(options.adaptiveGeometryMinimumObservationMultiplier)),
            0.05f,
            1.0f);
        options.enableAdaptiveConflictRobustWeighting =
            settings.value(QStringLiteral("tsdfAdaptiveConflictRobustWeighting"))
                .toBool(options.enableAdaptiveConflictRobustWeighting);
        options.adaptiveConflictWeightKnee =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveConflictWeightKnee"))
                                              .toDouble(options.adaptiveConflictWeightKnee)),
                       0.0f,
                       0.99f);
        options.adaptiveConflictWeightZero =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveConflictWeightZero"))
                                              .toDouble(options.adaptiveConflictWeightZero)),
                       options.adaptiveConflictWeightKnee + 1.0e-6f,
                       1.0f);
        options.minimumAdaptiveConflictWeightMultiplier =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMinimumAdaptiveConflictWeightMultiplier"))
                                              .toDouble(options.minimumAdaptiveConflictWeightMultiplier)),
                       0.0f,
                       1.0f);
        options.adaptiveGeometryFullIntegrationMinimumSupportWeight = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveGeometryFullIntegrationMinimumSupportWeight"))
                                   .toDouble(options.adaptiveGeometryFullIntegrationMinimumSupportWeight)),
            0.0f,
            1.0f);
        options.adaptiveGeometryFullIntegrationMinimumEffectiveViewCount =
            std::max(1.0f,
                     static_cast<float>(
                         settings.value(QStringLiteral("tsdfAdaptiveGeometryFullIntegrationMinimumEffectiveViewCount"))
                             .toDouble(options.adaptiveGeometryFullIntegrationMinimumEffectiveViewCount)));
        options.adaptiveGeometryFullIntegrationMaximumConflictRatio = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveGeometryFullIntegrationMaximumConflictRatio"))
                                   .toDouble(options.adaptiveGeometryFullIntegrationMaximumConflictRatio)),
            0.0f,
            1.0f);
        options.enableInverseDepthSpreadWeighting =
            settings.value(QStringLiteral("tsdfInverseDepthSpreadWeighting")).toBool(false);
        options.inverseDepthSpreadWeightKnee =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfInverseDepthSpreadWeightKnee"))
                                              .toDouble(options.inverseDepthSpreadWeightKnee)),
                       0.0f,
                       0.099f);
        options.inverseDepthSpreadWeightZero =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfInverseDepthSpreadWeightZero"))
                                              .toDouble(options.inverseDepthSpreadWeightZero)),
                       options.inverseDepthSpreadWeightKnee + 1.0e-6f,
                       0.10f);
        options.minimumInverseDepthSpreadWeightMultiplier = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfMinimumInverseDepthSpreadWeightMultiplier"))
                                   .toDouble(options.minimumInverseDepthSpreadWeightMultiplier)),
            0.0f,
            1.0f);
        options.enableInverseDepthSpreadSupportWeightDecoupling =
            settings.value(QStringLiteral("tsdfInverseDepthSpreadSupportWeightDecoupling")).toBool(false);
        options.inverseDepthSpreadSupportWeightExponent =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfInverseDepthSpreadSupportWeightExponent"))
                                              .toDouble(options.inverseDepthSpreadSupportWeightExponent)),
                       0.05f,
                       1.0f);
        options.enableEvidenceSupportWeightDecoupling =
            settings.value(QStringLiteral("tsdfEvidenceSupportWeightDecoupling")).toBool(false);
        options.evidenceSupportWeightExponent =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfEvidenceSupportWeightExponent"))
                                              .toDouble(options.evidenceSupportWeightExponent)),
                       0.0f,
                       1.0f);
        options.enableWeakEvidenceSurfaceOnlyIntegration =
            settings.value(QStringLiteral("tsdfWeakEvidenceSurfaceOnlyIntegration")).toBool(false);
        options.weakEvidenceSurfaceBandVoxels =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfWeakEvidenceSurfaceBandVoxels"))
                                              .toDouble(options.weakEvidenceSurfaceBandVoxels)),
                       0.0f,
                       16.0f);
        options.minimumSingleObservationWeight =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMinimumSingleObservationWeight"))
                                              .toDouble(options.minimumSingleObservationWeight)),
                       0.05f,
                       1.0f);
        options.allowGeometryVerifiedSingleObservation =
            settings.value(QStringLiteral("tsdfAllowGeometryVerifiedSingleObservation"))
                .toBool(options.resolution >= 384);
        options.minimumGeometryVerifiedObservationWeight =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMinimumGeometryVerifiedObservationWeight"))
                                              .toDouble(options.minimumGeometryVerifiedObservationWeight)),
                       0.05f,
                       1.0f);
        const int automatic_minimum_geometry_support = options.resolution >= 384 ? 3 : 4;
        options.minimumGeometrySupportCount = qBound(
            2,
            settings.value(QStringLiteral("tsdfMinimumGeometrySupportCount")).toInt(automatic_minimum_geometry_support),
            16);
        options.enableGeometrySingleViewNeighborhoodGuard =
            settings.value(QStringLiteral("tsdfGeometrySingleViewNeighborhoodGuard")).toBool(false);
        options.minimumGeometrySingleViewNeighborCount =
            qBound(1,
                   settings.value(QStringLiteral("tsdfMinimumGeometrySingleViewNeighborCount"))
                       .toInt(options.minimumGeometrySingleViewNeighborCount),
                   26);
        options.geometrySingleViewGrowthPasses =
            qBound(1,
                   settings.value(QStringLiteral("tsdfGeometrySingleViewGrowthPasses"))
                       .toInt(options.geometrySingleViewGrowthPasses),
                   6);
        options.maximumGeometrySingleViewNeighborTsdfDelta = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfMaximumGeometrySingleViewNeighborTsdfDelta"))
                                   .toDouble(options.maximumGeometrySingleViewNeighborTsdfDelta)),
            0.01f,
            2.0f);
        options.enableDiscontinuityAwareSampling =
            settings.value(QStringLiteral("tsdfDiscontinuityAwareSampling")).toBool(options.resolution >= 384);
        options.maximumInterpolationRelativeDepthSpread =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMaximumInterpolationRelativeDepthSpread"))
                                              .toDouble(options.maximumInterpolationRelativeDepthSpread)),
                       0.001f,
                       0.10f);
        options.maximumObservationInverseDepthSpread =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMaximumObservationInverseDepthSpread"))
                                              .toDouble(options.maximumObservationInverseDepthSpread)),
                       0.0f,
                       0.10f);
        options.allowInvalidNearestPixelRecovery =
            settings.value(QStringLiteral("tsdfAllowInvalidNearestPixelRecovery")).toBool(options.resolution < 384);
        options.maximumInvalidNearestPixelRecoveryInverseDepthSpread =
            std::clamp(static_cast<float>(
                           settings.value(QStringLiteral("tsdfMaximumInvalidNearestPixelRecoveryInverseDepthSpread"))
                               .toDouble(options.maximumInvalidNearestPixelRecoveryInverseDepthSpread)),
                       0.0f,
                       0.10f);
        options.enableCrossViewConsensusDepth =
            settings.value(QStringLiteral("tsdfCrossViewConsensusDepth")).toBool(false);
        options.maximumCrossViewConsensusInverseDepthSpread = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfMaximumCrossViewConsensusInverseDepthSpread"))
                                   .toDouble(options.maximumCrossViewConsensusInverseDepthSpread)),
            0.001f,
            0.10f);
        options.crossViewConsensusContourBandOnly =
            settings.value(QStringLiteral("tsdfCrossViewConsensusContourBandOnly")).toBool(false);
        options.enableRobustFrameQualityWeighting =
            settings.value(QStringLiteral("tsdfRobustFrameQualityWeighting")).toBool(false);
        options.robustFrameQualityMinimumMultiplier =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfRobustFrameQualityMinimumMultiplier"))
                                              .toDouble(options.robustFrameQualityMinimumMultiplier)),
                       0.05f,
                       1.0f);
        options.robustFrameQualityMadFloor =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfRobustFrameQualityMadFloor"))
                                              .toDouble(options.robustFrameQualityMadFloor)),
                       0.001f,
                       0.10f);
        options.robustFrameQualityPenaltyOnset =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfRobustFrameQualityPenaltyOnset"))
                                              .toDouble(options.robustFrameQualityPenaltyOnset)),
                       0.0f,
                       4.0f);
        options.robustFrameQualityPenaltyStrength =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfRobustFrameQualityPenaltyStrength"))
                                              .toDouble(options.robustFrameQualityPenaltyStrength)),
                       0.0f,
                       4.0f);
        options.enableRobustFrameQualityRejection =
            settings.value(QStringLiteral("tsdfRobustFrameQualityRejection")).toBool(false);
        options.robustFrameQualityRejectionSigma =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfRobustFrameQualityRejectionSigma"))
                                              .toDouble(options.robustFrameQualityRejectionSigma)),
                       0.5f,
                       6.0f);
        options.robustFrameQualityMaximumRejectedRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfRobustFrameQualityMaximumRejectedRatio"))
                                              .toDouble(options.robustFrameQualityMaximumRejectedRatio)),
                       0.0f,
                       0.50f);
        options.robustFrameQualityMinimumRetainedFrames =
            qBound(2,
                   settings.value(QStringLiteral("tsdfRobustFrameQualityMinimumRetainedFrames"))
                       .toInt(options.robustFrameQualityMinimumRetainedFrames),
                   64);
        options.enableOrbitalFrameCoverageProtection =
            settings.value(QStringLiteral("tsdfOrbitalFrameCoverageProtection")).toBool(false);
        options.maximumOrbitalAngularGapRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMaximumOrbitalAngularGapRatio"))
                                              .toDouble(options.maximumOrbitalAngularGapRatio)),
                       1.0f,
                       6.0f);
        options.validationOnlyFrameWeightMultiplier =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfValidationOnlyFrameWeightMultiplier"))
                                              .toDouble(options.validationOnlyFrameWeightMultiplier)),
                       0.05f,
                       1.0f);
        options.enableAuxiliarySurfaceOnlyIntegration =
            settings.value(QStringLiteral("tsdfAuxiliarySurfaceOnlyIntegration"))
                .toBool(options.enableAuxiliarySurfaceOnlyIntegration);
        options.enableAuxiliaryBridgeOnlyIntegration =
            settings.value(QStringLiteral("tsdfAuxiliaryBridgeOnlyIntegration"))
                .toBool(options.enableAuxiliaryBridgeOnlyIntegration);
        options.auxiliaryBridgeMinimumGeometrySupport =
            qBound(2,
                   settings.value(QStringLiteral("tsdfAuxiliaryBridgeMinimumGeometrySupport"))
                       .toInt(options.auxiliaryBridgeMinimumGeometrySupport),
                   16);
        options.auxiliaryBridgeMinimumSourceCount =
            qBound(2,
                   settings.value(QStringLiteral("tsdfAuxiliaryBridgeMinimumSourceCount"))
                       .toInt(options.auxiliaryBridgeMinimumSourceCount),
                   16);
        options.auxiliaryBridgeMaximumInverseDepthSpread =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfAuxiliaryBridgeMaximumInverseDepthSpread"))
                                              .toDouble(options.auxiliaryBridgeMaximumInverseDepthSpread)),
                       0.001f,
                       0.02f);
        options.auxiliaryBridgeMaximumExtensionVoxels =
            qBound(1,
                   settings.value(QStringLiteral("tsdfAuxiliaryBridgeMaximumExtensionVoxels"))
                       .toInt(options.auxiliaryBridgeMaximumExtensionVoxels),
                   4);
        options.coverageProtectedFrameMinimumMultiplier =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfCoverageProtectedFrameMinimumMultiplier"))
                                              .toDouble(options.coverageProtectedFrameMinimumMultiplier)),
                       0.05f,
                       1.0f);
        options.enableOrbitalGapBoundaryRecovery =
            settings.value(QStringLiteral("tsdfOrbitalGapBoundaryRecovery")).toBool(false);
        options.orbitalGapBoundaryMinimumQualityMultiplier = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfOrbitalGapBoundaryMinimumQualityMultiplier"))
                                   .toDouble(options.orbitalGapBoundaryMinimumQualityMultiplier)),
            0.05f,
            1.0f);
        options.orbitalGapOppositeMinimumQualityMultiplier = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfOrbitalGapOppositeMinimumQualityMultiplier"))
                                   .toDouble(options.orbitalGapOppositeMinimumQualityMultiplier)),
            0.05f,
            1.0f);
        options.orbitalGapBoundaryMinimumObservationWeight = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfOrbitalGapBoundaryMinimumObservationWeight"))
                                   .toDouble(options.orbitalGapBoundaryMinimumObservationWeight)),
            0.05f,
            1.0f);
        options.enableOrbitalGapAdaptiveTruncation =
            settings.value(QStringLiteral("tsdfOrbitalGapAdaptiveTruncation")).toBool(false);
        options.orbitalGapAdaptiveTruncationScale =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfOrbitalGapAdaptiveTruncationScale"))
                                              .toDouble(options.orbitalGapAdaptiveTruncationScale)),
                       0.0f,
                       2.0f);
        options.orbitalGapAdaptiveMaximumTruncationVoxels = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfOrbitalGapAdaptiveMaximumTruncationVoxels"))
                                   .toDouble(options.orbitalGapAdaptiveMaximumTruncationVoxels)),
            1.0f,
            16.0f);
        const bool automatic_surface_patch_support =
            options.resolution >= 384 && options.simplifyTargetFaces > 0 &&
            (options.simplifyTargetFaces <= 120000 ||
             (options.enableTopologySafeSimplification && options.simplifyTargetFaces <= 240000));
        options.enableSurfacePatchSupport =
            settings.value(QStringLiteral("tsdfSurfacePatchSupport")).toBool(automatic_surface_patch_support);
        options.enableContourBandZeroCrossingSupport =
            settings.value(QStringLiteral("tsdfContourBandZeroCrossingSupport")).toBool(false);
        options.collectZeroCrossingDiagnostics =
            settings.value(QStringLiteral("tsdfCollectZeroCrossingDiagnostics")).toBool(false);
        options.collectAcquisitionGapReport = settings.value(QStringLiteral("tsdfAcquisitionGapReport")).toBool(false);
        options.enableMeasuredSupportConnectivity =
            settings.value(QStringLiteral("tsdfMeasuredSupportConnectivity")).toBool(false);
        options.measuredSupportMinimumObservationWeight =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMeasuredSupportMinimumObservationWeight"))
                                              .toDouble(options.measuredSupportMinimumObservationWeight)),
                       0.01f,
                       1.0f);
        options.measuredSupportMinimumSourceCount =
            qBound(2,
                   settings.value(QStringLiteral("tsdfMeasuredSupportMinimumSourceCount"))
                       .toInt(options.measuredSupportMinimumSourceCount),
                   16);
        options.measuredSupportMinimumGeometrySupport =
            qBound(2,
                   settings.value(QStringLiteral("tsdfMeasuredSupportMinimumGeometrySupport"))
                       .toInt(options.measuredSupportMinimumGeometrySupport),
                   16);
        options.measuredSupportMaximumInverseDepthSpread =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMeasuredSupportMaximumInverseDepthSpread"))
                                              .toDouble(options.measuredSupportMaximumInverseDepthSpread)),
                       0.001f,
                       0.05f);
        options.measuredSupportMinimumSurfaceWeightRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMeasuredSupportMinimumSurfaceWeightRatio"))
                                              .toDouble(options.measuredSupportMinimumSurfaceWeightRatio)),
                       0.01f,
                       1.0f);
        options.measuredSupportMaximumAbsoluteTsdf =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMeasuredSupportMaximumAbsoluteTsdf"))
                                              .toDouble(options.measuredSupportMaximumAbsoluteTsdf)),
                       0.05f,
                       0.95f);
        options.measuredSupportMinimumSupportedCellCorners =
            qBound(1,
                   settings.value(QStringLiteral("tsdfMeasuredSupportMinimumSupportedCellCorners"))
                       .toInt(options.measuredSupportMinimumSupportedCellCorners),
                   7);
        options.measuredSupportMinimumComponentCells =
            qBound(1,
                   settings.value(QStringLiteral("tsdfMeasuredSupportMinimumComponentCells"))
                       .toInt(options.measuredSupportMinimumComponentCells),
                   4096);
        options.measuredSupportMinimumAnchorCells =
            qBound(1,
                   settings.value(QStringLiteral("tsdfMeasuredSupportMinimumAnchorCells"))
                       .toInt(options.measuredSupportMinimumAnchorCells),
                   4096);
        options.measuredSupportMaximumSingleVoteAbsoluteTsdf = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfMeasuredSupportMaximumSingleVoteAbsoluteTsdf"))
                                   .toDouble(options.measuredSupportMaximumSingleVoteAbsoluteTsdf)),
            0.0f,
            1.0f);
        const bool consistent_extraction_explicitly_enabled =
            settings.contains(QStringLiteral("tsdfConsistentIsoSurfaceExtraction")) &&
            settings.value(QStringLiteral("tsdfConsistentIsoSurfaceExtraction")).toBool(false);
        options.enableMc33IsoSurfaceExtraction =
            settings.value(QStringLiteral("tsdfMc33IsoSurfaceExtraction"))
                .toBool(options.enableTopologySafeSimplification && !consistent_extraction_explicitly_enabled &&
                        Mc33IsoSurfaceExtractor::isAvailable());
        options.enableConsistentIsoSurfaceExtraction =
            settings.value(QStringLiteral("tsdfConsistentIsoSurfaceExtraction"))
                .toBool(!options.enableMc33IsoSurfaceExtraction);
        options.mc33RequireSupportedSignChange =
            settings.value(QStringLiteral("tsdfMc33RequireSupportedSignChange")).toBool(true);
        options.enableGeometryZeroCrossingRecovery =
            settings.value(QStringLiteral("tsdfGeometryZeroCrossingRecovery")).toBool(false);
        options.geometryZeroCrossingMinimumSupportedCorners =
            qBound(1,
                   settings.value(QStringLiteral("tsdfGeometryZeroCrossingMinimumSupportedCorners"))
                       .toInt(options.geometryZeroCrossingMinimumSupportedCorners),
                   7);
        options.geometryZeroCrossingMinimumCellVotes =
            qBound(1,
                   settings.value(QStringLiteral("tsdfGeometryZeroCrossingMinimumCellVotes"))
                       .toInt(options.geometryZeroCrossingMinimumCellVotes),
                   8);
        options.enableCrossViewAnchoredSurfaceRecovery =
            settings.value(QStringLiteral("tsdfCrossViewAnchoredSurfaceRecovery")).toBool(false);
        options.crossViewAnchoredMinimumObservationWeight = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfCrossViewAnchoredMinimumObservationWeight"))
                                   .toDouble(options.crossViewAnchoredMinimumObservationWeight)),
            0.05f,
            1.0f);
        options.crossViewAnchoredMinimumSupportedCorners =
            qBound(1,
                   settings.value(QStringLiteral("tsdfCrossViewAnchoredMinimumSupportedCorners"))
                       .toInt(options.crossViewAnchoredMinimumSupportedCorners),
                   7);
        options.crossViewAnchoredMinimumCellVotes =
            qBound(1,
                   settings.value(QStringLiteral("tsdfCrossViewAnchoredMinimumCellVotes"))
                       .toInt(options.crossViewAnchoredMinimumCellVotes),
                   8);
        options.crossViewAnchoredGrowthPasses =
            qBound(1,
                   settings.value(QStringLiteral("tsdfCrossViewAnchoredGrowthPasses"))
                       .toInt(options.crossViewAnchoredGrowthPasses),
                   4);
        options.enableGeometryZeroCrossingCellSheets =
            settings.value(QStringLiteral("tsdfGeometryZeroCrossingCellSheets")).toBool(false);
        options.minimumGeometryZeroCrossingSheetCells =
            qBound(1,
                   settings.value(QStringLiteral("tsdfMinimumGeometryZeroCrossingSheetCells"))
                       .toInt(options.minimumGeometryZeroCrossingSheetCells),
                   4096);
        options.minimumGeometryZeroCrossingSheetAnchorCells =
            qBound(1,
                   settings.value(QStringLiteral("tsdfMinimumGeometryZeroCrossingSheetAnchorCells"))
                       .toInt(options.minimumGeometryZeroCrossingSheetAnchorCells),
                   4096);
        options.maximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf =
            std::clamp(static_cast<float>(
                           settings.value(QStringLiteral("tsdfMaximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf"))
                               .toDouble(options.maximumGeometryZeroCrossingSheetSingleVoteAbsoluteTsdf)),
                       0.0f,
                       1.0f);
        options.enableGlobalImplicitRegularization =
            settings.value(QStringLiteral("tsdfGlobalImplicitRegularization")).toBool(false);
        options.implicitRegularizationLevels = qBound(1,
                                                      settings.value(QStringLiteral("tsdfImplicitRegularizationLevels"))
                                                          .toInt(options.implicitRegularizationLevels),
                                                      3);
        options.implicitRegularizationPassesPerLevel =
            qBound(1,
                   settings.value(QStringLiteral("tsdfImplicitRegularizationPassesPerLevel"))
                       .toInt(options.implicitRegularizationPassesPerLevel),
                   4);
        options.implicitRegularizationSmoothness =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfImplicitRegularizationSmoothness"))
                                              .toDouble(options.implicitRegularizationSmoothness)),
                       0.0f,
                       2.0f);
        options.implicitRegularizationDataFidelity =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfImplicitRegularizationDataFidelity"))
                                              .toDouble(options.implicitRegularizationDataFidelity)),
                       0.01f,
                       8.0f);
        options.implicitRegularizationMaximumUpdate =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfImplicitRegularizationMaximumUpdate"))
                                              .toDouble(options.implicitRegularizationMaximumUpdate)),
                       0.0f,
                       0.5f);
        options.implicitRegularizationEdgeThreshold =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfImplicitRegularizationEdgeThreshold"))
                                              .toDouble(options.implicitRegularizationEdgeThreshold)),
                       0.02f,
                       1.0f);
        options.implicitRegularizationRecoverAxialGaps =
            settings.value(QStringLiteral("tsdfImplicitRegularizationRecoverAxialGaps"))
                .toBool(options.implicitRegularizationRecoverAxialGaps);
        options.implicitRegularizationMinimumBridgeAxes =
            qBound(1,
                   settings.value(QStringLiteral("tsdfImplicitRegularizationMinimumBridgeAxes"))
                       .toInt(options.implicitRegularizationMinimumBridgeAxes),
                   3);
        options.implicitRegularizationMaximumBridgePredictionDelta = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfImplicitRegularizationMaximumBridgePredictionDelta"))
                                   .toDouble(options.implicitRegularizationMaximumBridgePredictionDelta)),
            0.01f,
            0.5f);
        options.enableAdaptiveTgvRegularization =
            settings.value(QStringLiteral("tsdfAdaptiveTgvRegularization")).toBool(false);
        options.adaptiveTgvMaximumMergeLevel = qBound(0,
                                                      settings.value(QStringLiteral("tsdfAdaptiveTgvMaximumMergeLevel"))
                                                          .toInt(options.adaptiveTgvMaximumMergeLevel),
                                                      10);
        options.adaptiveTgvMinimumMergeAbsoluteField =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveTgvMinimumMergeAbsoluteField"))
                                              .toDouble(options.adaptiveTgvMinimumMergeAbsoluteField)),
                       0.1f,
                       1.0f);
        options.adaptiveTgvMaximumMergeFieldRange =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveTgvMaximumMergeFieldRange"))
                                              .toDouble(options.adaptiveTgvMaximumMergeFieldRange)),
                       0.01f,
                       1.0f);
        options.adaptiveTgvMaximumActiveAbsoluteField =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveTgvMaximumActiveAbsoluteField"))
                                              .toDouble(options.adaptiveTgvMaximumActiveAbsoluteField)),
                       0.10f,
                       1.0f);
        options.adaptiveTgvMaximumIterations = qBound(10,
                                                      settings.value(QStringLiteral("tsdfAdaptiveTgvMaximumIterations"))
                                                          .toInt(options.adaptiveTgvMaximumIterations),
                                                      400);
        options.adaptiveTgvMinimumIterations = qBound(5,
                                                      settings.value(QStringLiteral("tsdfAdaptiveTgvMinimumIterations"))
                                                          .toInt(options.adaptiveTgvMinimumIterations),
                                                      options.adaptiveTgvMaximumIterations);
        options.adaptiveTgvFirstOrderWeight =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveTgvFirstOrderWeight"))
                                              .toDouble(options.adaptiveTgvFirstOrderWeight)),
                       0.001f,
                       2.0f);
        options.adaptiveTgvSecondOrderWeight =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveTgvSecondOrderWeight"))
                                              .toDouble(options.adaptiveTgvSecondOrderWeight)),
                       0.001f,
                       2.0f);
        options.adaptiveTgvDataFidelity =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveTgvDataFidelity"))
                                              .toDouble(options.adaptiveTgvDataFidelity)),
                       0.001f,
                       2.0f);
        options.adaptiveTgvPrimalStep = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfAdaptiveTgvPrimalStep")).toDouble(options.adaptiveTgvPrimalStep)),
            0.001f,
            0.24f);
        options.adaptiveTgvDualStep = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfAdaptiveTgvDualStep")).toDouble(options.adaptiveTgvDualStep)),
            0.001f,
            0.24f);
        options.adaptiveTgvConvergenceTolerance =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveTgvConvergenceTolerance"))
                                              .toDouble(options.adaptiveTgvConvergenceTolerance)),
                       0.0f,
                       0.01f);
        options.adaptiveTgvUseGlobalVisibilityField =
            settings.value(QStringLiteral("tsdfAdaptiveTgvUseGlobalVisibilityField"))
                .toBool(options.adaptiveTgvUseGlobalVisibilityField);
        options.adaptiveTgvRecoverUnsupportedSamples =
            settings.value(QStringLiteral("tsdfAdaptiveTgvRecoverUnsupportedSamples"))
                .toBool(options.adaptiveTgvRecoverUnsupportedSamples);
        options.adaptiveTgvRecoveryPasses = qBound(
            1,
            settings.value(QStringLiteral("tsdfAdaptiveTgvRecoveryPasses")).toInt(options.adaptiveTgvRecoveryPasses),
            6);
        options.adaptiveTgvMinimumRecoveryNeighbors =
            qBound(1,
                   settings.value(QStringLiteral("tsdfAdaptiveTgvMinimumRecoveryNeighbors"))
                       .toInt(options.adaptiveTgvMinimumRecoveryNeighbors),
                   6);
        options.adaptiveTgvMaximumRecoveryConflictRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfAdaptiveTgvMaximumRecoveryConflictRatio"))
                                              .toDouble(options.adaptiveTgvMaximumRecoveryConflictRatio)),
                       0.0f,
                       0.5f);
        options.enableVisualHullSignedDistanceCompletion =
            settings.value(QStringLiteral("tsdfVisualHullSignedDistanceCompletion"))
                .toBool(options.enableVisualHullSignedDistanceCompletion);
        options.visualHullCompletionMinimumVisibleViews =
            qBound(2,
                   settings.value(QStringLiteral("tsdfVisualHullCompletionMinimumVisibleViews"))
                       .toInt(options.visualHullCompletionMinimumVisibleViews),
                   64);
        options.visualHullCompletionAllowedSilhouetteViolations =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisualHullCompletionAllowedSilhouetteViolations"))
                       .toInt(options.visualHullCompletionAllowedSilhouetteViolations),
                   16);
        options.visualHullCompletionBandVoxels =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfVisualHullCompletionBandVoxels"))
                                              .toDouble(options.visualHullCompletionBandVoxels)),
                       1.0f,
                       24.0f);
        options.visualHullCompletionPreserveObservedTsdf =
            settings.value(QStringLiteral("tsdfVisualHullCompletionPreserveObservedTsdf"))
                .toBool(options.visualHullCompletionPreserveObservedTsdf);
        options.visualHullCompletionMaximumObservedAbsoluteTsdf = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfVisualHullCompletionMaximumObservedAbsoluteTsdf"))
                                   .toDouble(options.visualHullCompletionMaximumObservedAbsoluteTsdf)),
            0.05f,
            1.0f);
        options.visualHullCompletionMinimumGeometrySupport =
            qBound(1,
                   settings.value(QStringLiteral("tsdfVisualHullCompletionMinimumGeometrySupport"))
                       .toInt(options.visualHullCompletionMinimumGeometrySupport),
                   16);
        options.visualHullCompletionRelaxationIterations =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisualHullCompletionRelaxationIterations"))
                       .toInt(options.visualHullCompletionRelaxationIterations),
                   24);
        options.visualHullCompletionRelaxationLambda =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfVisualHullCompletionRelaxationLambda"))
                                              .toDouble(options.visualHullCompletionRelaxationLambda)),
                       0.0f,
                       0.49f);
        options.visualHullCompletionMaximumUpdate =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfVisualHullCompletionMaximumUpdate"))
                                              .toDouble(options.visualHullCompletionMaximumUpdate)),
                       0.01f,
                       1.0f);
        options.enableVisibilityOccupancyCompletion =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyCompletion")).toBool(false);
        options.visibilityOccupancyUseSupportMaskSilhouette =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyUseSupportMaskSilhouette"))
                .toBool(options.visibilityOccupancyUseSupportMaskSilhouette);
        options.visibilityOccupancyResolution =
            qBound(24,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyResolution"))
                       .toInt(options.visibilityOccupancyResolution),
                   128);
        options.visibilityOccupancyAlignCarrierGrid =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyAlignCarrierGrid"))
                .toBool(options.visibilityOccupancyAlignCarrierGrid);
        options.visibilityOccupancyNativeCarrierExtraction =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyNativeCarrierExtraction"))
                .toBool(options.visibilityOccupancyNativeCarrierExtraction);
        options.visibilityOccupancyCellBoundaryExtraction =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyCellBoundaryExtraction"))
                .toBool(options.visibilityOccupancyCellBoundaryExtraction);
        options.visibilityOccupancyMinimumVisibleViews =
            qBound(1,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyMinimumVisibleViews"))
                       .toInt(options.visibilityOccupancyMinimumVisibleViews),
                   16);
        options.visibilityOccupancyMinimumSilhouetteViews =
            qBound(1,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyMinimumSilhouetteViews"))
                       .toInt(options.visibilityOccupancyMinimumSilhouetteViews),
                   16);
        options.visibilityOccupancyMinimumDepthFullViewsForSilhouettePrior =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyMinimumDepthFullViewsForSilhouettePrior"))
                       .toInt(options.visibilityOccupancyMinimumDepthFullViewsForSilhouettePrior),
                   16);
        options.visibilityOccupancyAdaptiveDepthSupportMinimumFullFraction = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfVisibilityOccupancyAdaptiveDepthSupportMinimumFullFraction"))
                    .toDouble(options.visibilityOccupancyAdaptiveDepthSupportMinimumFullFraction)),
            0.0f,
            0.25f);
        options.visibilityOccupancyAllowedSilhouetteViolations =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyAllowedSilhouetteViolations"))
                       .toInt(options.visibilityOccupancyAllowedSilhouetteViolations),
                   8);
        options.visibilityOccupancyFrontTolerancePixelFootprints = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfVisibilityOccupancyFrontTolerancePixelFootprints"))
                                   .toDouble(options.visibilityOccupancyFrontTolerancePixelFootprints)),
            0.5f,
            12.0f);
        options.visibilityOccupancyBehindSurfaceBandPixelFootprints = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfVisibilityOccupancyBehindSurfaceBandPixelFootprints"))
                                   .toDouble(options.visibilityOccupancyBehindSurfaceBandPixelFootprints)),
            1.0f,
            16.0f);
        options.visibilityOccupancyDepthEmptyCapacity =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthEmptyCapacity"))
                       .toInt(options.visibilityOccupancyDepthEmptyCapacity),
                   1000);
        options.visibilityOccupancyDepthFullCapacity =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthFullCapacity"))
                       .toInt(options.visibilityOccupancyDepthFullCapacity),
                   1000);
        options.visibilityOccupancySilhouetteEmptyCapacity =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancySilhouetteEmptyCapacity"))
                       .toInt(options.visibilityOccupancySilhouetteEmptyCapacity),
                   1000);
        options.visibilityOccupancySilhouetteFullPriorCapacity =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancySilhouetteFullPriorCapacity"))
                       .toInt(options.visibilityOccupancySilhouetteFullPriorCapacity),
                   1000);
        options.visibilityOccupancyPairwiseCapacity =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyPairwiseCapacity"))
                       .toInt(options.visibilityOccupancyPairwiseCapacity),
                   1000);
        options.visibilityOccupancyClosingIterations =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyClosingIterations"))
                       .toInt(options.visibilityOccupancyClosingIterations),
                   8);
        options.visibilityOccupancyMaximumHandleRepairPasses =
            qBound(1,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyMaximumHandleRepairPasses"))
                       .toInt(options.visibilityOccupancyMaximumHandleRepairPasses),
                   16);
        options.visibilityOccupancyMaximumHandleRepairAcceptedCandidateCount =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyMaximumHandleRepairAcceptedCandidateCount"))
                       .toInt(options.visibilityOccupancyMaximumHandleRepairAcceptedCandidateCount),
                   512);
        options.visibilityOccupancyMaximumHandleRepairCandidateSampleCount = static_cast<std::size_t>(
            qBound(1,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyMaximumHandleRepairCandidateSampleCount"))
                       .toInt(static_cast<int>(
                           std::min<std::size_t>(options.visibilityOccupancyMaximumHandleRepairCandidateSampleCount,
                                                 static_cast<std::size_t>(65536)))),
                   65536));
        options.visibilityOccupancyMaximumHandleRepairSubsetSampleCount = static_cast<std::size_t>(qBound(
            1,
            settings.value(QStringLiteral("tsdfVisibilityOccupancyMaximumHandleRepairSubsetSampleCount"))
                .toInt(static_cast<int>(std::min<std::size_t>(
                    options.visibilityOccupancyMaximumHandleRepairSubsetSampleCount, static_cast<std::size_t>(1024)))),
            1024));
        options.visibilityOccupancyMaximumHandleRepairSubsetSeedCount =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyMaximumHandleRepairSubsetSeedCount"))
                       .toInt(options.visibilityOccupancyMaximumHandleRepairSubsetSeedCount),
                   4096);
        options.visibilityOccupancyClosingMinimumDepthEmptyViewsToProtect =
            qBound(1,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyClosingMinimumDepthEmptyViewsToProtect"))
                       .toInt(options.visibilityOccupancyClosingMinimumDepthEmptyViewsToProtect),
                   16);
        options.visibilityOccupancyClosingMinimumSilhouetteOutsideViewsToProtect = qBound(
            1,
            settings.value(QStringLiteral("tsdfVisibilityOccupancyClosingMinimumSilhouetteOutsideViewsToProtect"))
                .toInt(options.visibilityOccupancyClosingMinimumSilhouetteOutsideViewsToProtect),
            16);
        options.visibilityOccupancyTopologyLockedResidualBlend =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyTopologyLockedResidualBlend"))
                .toBool(options.visibilityOccupancyTopologyLockedResidualBlend);
        options.visibilityOccupancyObservedBand =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfVisibilityOccupancyObservedBand"))
                                              .toDouble(options.visibilityOccupancyObservedBand)),
                       0.01f,
                       1.0f);
        options.visibilityOccupancyCarrierBand =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfVisibilityOccupancyCarrierBand"))
                                              .toDouble(options.visibilityOccupancyCarrierBand)),
                       0.01f,
                       1.0f);
        options.visibilityOccupancyMaximumResidual =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfVisibilityOccupancyMaximumResidual"))
                                              .toDouble(options.visibilityOccupancyMaximumResidual)),
                       0.0f,
                       1.0f);
        options.visibilityOccupancyDetailBlend =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfVisibilityOccupancyDetailBlend"))
                                              .toDouble(options.visibilityOccupancyDetailBlend)),
                       0.0f,
                       1.0f);
        options.visibilityOccupancyPreserveAllObservedSamples =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyPreserveAllObservedSamples"))
                .toBool(options.visibilityOccupancyPreserveAllObservedSamples);
        options.visibilityOccupancyPreserveObservedNearSurface =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyPreserveObservedNearSurface"))
                .toBool(options.visibilityOccupancyPreserveObservedNearSurface);
        options.visibilityOccupancyRequireSignAgreement =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyRequireSignAgreement"))
                .toBool(options.visibilityOccupancyRequireSignAgreement);
        options.visibilityOccupancyMaximumPreservedAbsoluteTsdf = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfVisibilityOccupancyMaximumPreservedAbsoluteTsdf"))
                                   .toDouble(options.visibilityOccupancyMaximumPreservedAbsoluteTsdf)),
            0.0f,
            1.0f);
        options.visibilityOccupancySignedDistanceNormalizationSamples =
            std::clamp(static_cast<float>(
                           settings.value(QStringLiteral("tsdfVisibilityOccupancySignedDistanceNormalizationSamples"))
                               .toDouble(options.visibilityOccupancySignedDistanceNormalizationSamples)),
                       0.5f,
                       16.0f);
        const float automatic_minimum_surface_patch_observation_weight = 0.60f;
        options.minimumSurfacePatchObservationWeight =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMinimumSurfacePatchObservationWeight"))
                                              .toDouble(automatic_minimum_surface_patch_observation_weight)),
                       0.05f,
                       1.0f);
        options.minimumSurfacePatchSourceCount =
            qBound(2,
                   settings.value(QStringLiteral("tsdfMinimumSurfacePatchSourceCount"))
                       .toInt(options.minimumSurfacePatchSourceCount),
                   8);
        options.minimumSurfacePatchCoreNeighborCount =
            qBound(1,
                   settings.value(QStringLiteral("tsdfMinimumSurfacePatchCoreNeighborCount"))
                       .toInt(options.minimumSurfacePatchCoreNeighborCount),
                   26);
        options.surfacePatchGrowthPasses =
            qBound(1, settings.value(QStringLiteral("tsdfSurfacePatchGrowthPasses")).toInt(1), 6);
        options.maximumSurfacePatchInverseDepthSpread =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMaximumSurfacePatchInverseDepthSpread"))
                                              .toDouble(options.maximumSurfacePatchInverseDepthSpread)),
                       0.001f,
                       0.05f);
        options.maximumSurfacePatchNormalAngleDegrees =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMaximumSurfacePatchNormalAngleDegrees"))
                                              .toDouble(options.maximumSurfacePatchNormalAngleDegrees)),
                       5.0f,
                       45.0f);
        options.maximumSurfacePatchAbsoluteTsdf =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMaximumSurfacePatchAbsoluteTsdf"))
                                              .toDouble(options.maximumSurfacePatchAbsoluteTsdf)),
                       0.05f,
                       0.95f);
        options.maximumContourBandAbsoluteTsdf =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMaximumContourBandAbsoluteTsdf"))
                                              .toDouble(options.maximumSurfacePatchAbsoluteTsdf)),
                       options.maximumSurfacePatchAbsoluteTsdf,
                       0.95f);
        options.minimumSurfacePatchWeightRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMinimumSurfacePatchWeightRatio"))
                                              .toDouble(options.minimumSurfacePatchWeightRatio)),
                       0.01f,
                       1.0f);
        options.enableSupportMaskFreeSpaceCarving =
            settings.value(QStringLiteral("tsdfSupportMaskFreeSpaceCarving")).toBool(false);
        options.enableSurfaceEvidenceFreeSpaceVeto =
            settings.value(QStringLiteral("tsdfSurfaceEvidenceFreeSpaceVeto")).toBool(true);
        options.maximumFreeSpaceVoxels = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfMaximumFreeSpaceVoxels")).toDouble(options.maximumFreeSpaceVoxels)),
            0.0f,
            64.0f);
        const int automatic_boundary_erosion = options.resolution >= 384 ? 2 : 1;
        options.depthValidBoundaryErosionPixels = qBound(
            0,
            settings.value(QStringLiteral("tsdfDepthValidBoundaryErosionPixels")).toInt(automatic_boundary_erosion),
            4);
        const bool automatic_boundary_recovery =
            options.resolution >= 384 && options.simplifyTargetFaces > 0 &&
            (options.simplifyTargetFaces <= 120000 ||
             (options.enableTopologySafeSimplification && options.simplifyTargetFaces <= 240000));
        options.enableGeometryVerifiedBoundaryRecovery =
            settings.value(QStringLiteral("tsdfGeometryVerifiedBoundaryRecovery")).toBool(automatic_boundary_recovery);
        options.minimumBoundaryRecoveryGeometrySupport =
            qBound(2,
                   settings.value(QStringLiteral("tsdfMinimumBoundaryRecoveryGeometrySupport"))
                       .toInt(options.minimumBoundaryRecoveryGeometrySupport),
                   16);
        options.maximumBoundaryRecoveryInverseDepthSpread = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfMaximumBoundaryRecoveryInverseDepthSpread"))
                                   .toDouble(options.maximumBoundaryRecoveryInverseDepthSpread)),
            0.001f,
            0.05f);
        options.supportMaskFreeSpaceWeight =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfSupportMaskFreeSpaceWeight"))
                                              .toDouble(options.supportMaskFreeSpaceWeight)),
                       0.0f,
                       1.0f);
        options.minimumSupportMaskFreeSpaceViews =
            qBound(1,
                   settings.value(QStringLiteral("tsdfMinimumSupportMaskFreeSpaceViews"))
                       .toInt(options.minimumSupportMaskFreeSpaceViews),
                   16);
        options.enableNarrowBandActivation = settings.value(QStringLiteral("tsdfNarrowBandActivation")).toBool(false);
        options.narrowBandActivationBlockSizeSamples =
            qBound(2,
                   settings.value(QStringLiteral("tsdfNarrowBandActivationBlockSizeSamples"))
                       .toInt(options.narrowBandActivationBlockSizeSamples),
                   32);
        options.narrowBandActivationDepthStride =
            qBound(1,
                   settings.value(QStringLiteral("tsdfNarrowBandActivationDepthStride"))
                       .toInt(options.narrowBandActivationDepthStride),
                   16);
        options.narrowBandActivationRayStepVoxels =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfNarrowBandActivationRayStepVoxels"))
                                              .toDouble(options.narrowBandActivationRayStepVoxels)),
                       0.25f,
                       4.0f);
        options.narrowBandActivationHaloBlocks =
            qBound(0,
                   settings.value(QStringLiteral("tsdfNarrowBandActivationHaloBlocks"))
                       .toInt(options.narrowBandActivationHaloBlocks),
                   3);

        const QString interpolation =
            settings.value(QStringLiteral("interpolation")).toString(QStringLiteral("enabled"));
        options.fillSmallBoundaryHoles =
            interpolation != QStringLiteral("disabled") && settings.value(QStringLiteral("holeFill")).toBool(true);
        options.splitPinchedBoundaryVertices = settings.value(QStringLiteral("tsdfSplitPinchedBoundaryVertices"))
                                                   .toBool(options.splitPinchedBoundaryVertices);
        const double maximum_hole_area = std::max(1.0, settings.value(QStringLiteral("maxHoleSize")).toDouble(100.0));
        const int conservative_edges =
            std::clamp(static_cast<int>(std::lround(std::sqrt(maximum_hole_area) * 1.6)), 8, 32);
        options.maximumHoleBoundaryEdges = interpolation == QStringLiteral("extrapolated")
                                               ? std::clamp(std::max(64, conservative_edges * 4), 64, 128)
                                               : std::clamp(std::max(48, conservative_edges * 3), 32, 64);
        options.maximumHoleDiameterVoxels = interpolation == QStringLiteral("extrapolated") ? 16.0f : 10.0f;
        options.maximumHoleBoundaryEdges = qBound(
            3,
            settings.value(QStringLiteral("tsdfMaximumHoleBoundaryEdges")).toInt(options.maximumHoleBoundaryEdges),
            256);
        options.maximumHoleDiameterVoxels =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfMaximumHoleDiameterVoxels"))
                                              .toDouble(options.maximumHoleDiameterVoxels)),
                       0.0f,
                       64.0f);
        const bool automatic_final_hole_fill = options.fillSmallBoundaryHoles && options.resolution >= 384 &&
                                               options.simplifyTargetFaces > 0 && options.simplifyTargetFaces <= 120000;
        options.enableSilhouetteAwareFinalHoleFill =
            settings.value(QStringLiteral("tsdfSilhouetteAwareFinalHoleFill")).toBool(automatic_final_hole_fill);
        if (automatic_final_hole_fill)
        {
            // This stage is a residual micro-hole repair.  Large boundary loops
            // represent missing implicit geometry and must remain visible instead
            // of being hidden behind a triangle fan.
            options.finalHoleFillMaximumBoundaryEdges = 24;
            options.finalHoleFillMaximumDiameterVoxels = 4.0f;
            options.finalHoleFillMaximumFaceGrowthRatio = 0.03f;
        }
        options.finalHoleFillMaximumBoundaryEdges =
            qBound(3,
                   settings.value(QStringLiteral("tsdfFinalHoleFillMaximumBoundaryEdges"))
                       .toInt(options.finalHoleFillMaximumBoundaryEdges),
                   512);
        options.finalHoleFillMaximumDiameterVoxels =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfFinalHoleFillMaximumDiameterVoxels"))
                                              .toDouble(options.finalHoleFillMaximumDiameterVoxels)),
                       0.0f,
                       64.0f);
        options.finalHoleFillMaximumFaceGrowthRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfFinalHoleFillMaximumFaceGrowthRatio"))
                                              .toDouble(options.finalHoleFillMaximumFaceGrowthRatio)),
                       0.0f,
                       0.50f);
        options.finalHoleFillMaximumSliverRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfFinalHoleFillMaximumSliverRatio"))
                                              .toDouble(options.finalHoleFillMaximumSliverRatio)),
                       0.0f,
                       0.50f);
        options.enableVisibilityConstrainedFinalHoleFill =
            settings.value(QStringLiteral("tsdfVisibilityConstrainedFinalHoleFill"))
                .toBool(automatic_final_hole_fill && options.enableSilhouetteAwareFinalHoleFill);
        options.visibilityHoleFillMinimumSupportingViews =
            qBound(1,
                   settings.value(QStringLiteral("tsdfVisibilityHoleFillMinimumSupportingViews"))
                       .toInt(options.visibilityHoleFillMinimumSupportingViews),
                   8);
        options.visibilityHoleFillMaximumConflictViews = qBound(
            0,
            settings.value(QStringLiteral("tsdfVisibilityHoleFillMaximumConflictViews"))
                .toInt(
                    automatic_final_hole_fill &&
                            settings.value(QStringLiteral("surface_type")).toString(QStringLiteral("arbitrary_3d")) ==
                                QStringLiteral("arbitrary_3d")
                        ? 2
                        : options.visibilityHoleFillMaximumConflictViews),
            16);
        options.visibilityHoleFillDepthToleranceVoxels =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfVisibilityHoleFillDepthToleranceVoxels"))
                                              .toDouble(options.visibilityHoleFillDepthToleranceVoxels)),
                       1.0f,
                       32.0f);
        options.visibilityHoleFillStrongSilhouetteRatio =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfVisibilityHoleFillStrongSilhouetteRatio"))
                                              .toDouble(options.visibilityHoleFillStrongSilhouetteRatio)),
                       0.0f,
                       1.0f);
        options.enableTinyBoundaryLoopCollapse =
            settings.value(QStringLiteral("tsdfTinyBoundaryLoopCollapse"))
                .toBool(automatic_final_hole_fill && options.enableVisibilityConstrainedFinalHoleFill);
        options.tinyBoundaryLoopCollapseMaximumEdges =
            qBound(3,
                   settings.value(QStringLiteral("tsdfTinyBoundaryLoopCollapseMaximumEdges"))
                       .toInt(options.tinyBoundaryLoopCollapseMaximumEdges),
                   16);
        options.tinyBoundaryLoopCollapseMaximumDiameterVoxels = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfTinyBoundaryLoopCollapseMaximumDiameterVoxels"))
                                   .toDouble(options.tinyBoundaryLoopCollapseMaximumDiameterVoxels)),
            0.0f,
            8.0f);
        options.tinyBoundaryLoopCollapseMaximumEdgeVoxels = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfTinyBoundaryLoopCollapseMaximumEdgeVoxels"))
                                   .toDouble(options.tinyBoundaryLoopCollapseMaximumEdgeVoxels)),
            0.0f,
            1.0f);
        options.tinyBoundaryLoopCollapseMaximumPasses =
            qBound(1,
                   settings.value(QStringLiteral("tsdfTinyBoundaryLoopCollapseMaximumPasses"))
                       .toInt(options.tinyBoundaryLoopCollapseMaximumPasses),
                   8);
        const int automatic_boundary_smoothing_iterations = options.resolution >= 384 ? 2 : 1;
        options.boundarySmoothingIterations = qBound(0,
                                                     settings.value(QStringLiteral("tsdfBoundarySmoothingIterations"))
                                                         .toInt(automatic_boundary_smoothing_iterations),
                                                     4);
        options.boundarySmoothingLambda =
            std::clamp(static_cast<float>(settings.value(QStringLiteral("tsdfBoundarySmoothingLambda"))
                                              .toDouble(options.boundarySmoothingLambda)),
                       0.0f,
                       0.5f);
        options.maximumBoundarySmoothingDisplacementVoxels = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfMaximumBoundarySmoothingDisplacementVoxels"))
                                   .toDouble(options.maximumBoundarySmoothingDisplacementVoxels)),
            0.0f,
            1.0f);
        const bool automatic_surface_denoising =
            options.resolution >= 256 && options.simplifyTargetFaces > 0 && options.simplifyTargetFaces <= 240000 &&
            settings.value(QStringLiteral("surface_type")).toString(QStringLiteral("arbitrary_3d")) ==
                QStringLiteral("arbitrary_3d");
        const bool high_face_budget_surface_denoising =
            automatic_surface_denoising && options.simplifyTargetFaces > 120000;
        const int automatic_surface_denoising_iterations = automatic_surface_denoising
                                                               ? (high_face_budget_surface_denoising ? 8 : 1)
                                                               : options.surfaceDenoisingIterations;
        if (automatic_surface_denoising)
        {
            options.surfaceDenoisingLambda = high_face_budget_surface_denoising ? 0.50f : 0.30f;
            options.surfaceDenoisingMu = high_face_budget_surface_denoising ? -0.53f : 0.0f;
            options.maximumSurfaceDenoisingDisplacementVoxels = high_face_budget_surface_denoising ? 0.75f : 0.12f;
            options.maximumSurfaceDenoisingNormalAngleDegrees = high_face_budget_surface_denoising ? 120.0f : 25.0f;
            options.surfaceDenoisingBoundaryProtectionRings = high_face_budget_surface_denoising ? 0 : 1;
            options.enableProtectedTaubinSurfaceDenoising = high_face_budget_surface_denoising;
        }
        options.surfaceDenoisingIterations = qBound(0,
                                                    settings.value(QStringLiteral("tsdfSurfaceDenoisingIterations"))
                                                        .toInt(automatic_surface_denoising_iterations),
                                                    8);
        options.surfaceDenoisingLambda = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfSurfaceDenoisingLambda")).toDouble(options.surfaceDenoisingLambda)),
            0.0f,
            0.75f);
        options.surfaceDenoisingMu = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfSurfaceDenoisingMu")).toDouble(options.surfaceDenoisingMu)),
            -1.0f,
            0.0f);
        options.maximumSurfaceDenoisingDisplacementVoxels = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfMaximumSurfaceDenoisingDisplacementVoxels"))
                                   .toDouble(options.maximumSurfaceDenoisingDisplacementVoxels)),
            0.0f,
            1.0f);
        options.maximumSurfaceDenoisingNormalAngleDegrees = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfMaximumSurfaceDenoisingNormalAngleDegrees"))
                                   .toDouble(options.maximumSurfaceDenoisingNormalAngleDegrees)),
            5.0f,
            170.0f);
        options.surfaceDenoisingBoundaryProtectionRings =
            qBound(0,
                   settings.value(QStringLiteral("tsdfSurfaceDenoisingBoundaryProtectionRings"))
                       .toInt(options.surfaceDenoisingBoundaryProtectionRings),
                   2);
        options.enableProtectedTaubinSurfaceDenoising =
            settings.value(QStringLiteral("tsdfProtectedTaubinSurfaceDenoising"))
                .toBool(options.enableProtectedTaubinSurfaceDenoising);
        options.enablePostSimplificationSurfaceDenoising =
            settings.value(QStringLiteral("tsdfPostSimplificationSurfaceDenoising"))
                .toBool(options.enablePostSimplificationSurfaceDenoising);
        const bool automatic_trim_weak_boundary_tips = options.resolution >= 384;
        options.trimWeakBoundaryTips =
            settings.value(QStringLiteral("tsdfTrimWeakBoundaryTips")).toBool(automatic_trim_weak_boundary_tips);
        const int automatic_weak_boundary_trim_passes = options.resolution >= 384 ? 2 : 1;
        options.weakBoundaryTipTrimPasses = qBound(
            1,
            settings.value(QStringLiteral("tsdfWeakBoundaryTipTrimPasses")).toInt(automatic_weak_boundary_trim_passes),
            4);
        enforceNoDepthInterpolationPolicy(settings, &options);
        return options;
    }
} // namespace xjw::mesh::workflow::workflow_detail
